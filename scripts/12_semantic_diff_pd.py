"""12_semantic_diff_pd.py — Compare our AI-translated Tier C (approved units) against
the legacy PT reference texts (data/sources/confessions/*.txt) as a QUALITY-CONTROL
ORACLE ONLY. READ-ONLY comparison: we never copy text from the legacy files into the
dataset (see data/sources/confessions/SOURCES_ATTRIBUTION.md — their license is
unconfirmed). This script only measures agreement to prioritize pastoral review.

Why not plain embedding cosine similarity: embeddings are polarity-blind ("saints MAY
intercede" vs "saints may NOT intercede" embed nearly identically — the exact class of
error that motivated this whole re-translation, cf. the WCF 21.3 / Santo Expedito
incident). Two layers instead:
  1. Deterministic Scripture-reference set comparison (free, catches proof-text
     drops/additions).
  2. LLM semantic-equivalence judge that explicitly scores core-claim match AND flags
     polarity inversion / omitted / added claims (paid, gated behind --execute).

Modes
  --align                 free: parse + align both sides by chapter/number key, report
                          coverage (no API calls)
  --compare               deterministic Scripture-ref check + LLM semantic judge
                          -> data/tier_c/semantic_diff_pd.jsonl
                          (PAID; prints a cost estimate and does nothing unless --execute)
  --report                tabulate results from the output jsonl

Examples
  python scripts/12_semantic_diff_pd.py --align
  python scripts/12_semantic_diff_pd.py --compare --dry-run
  python scripts/12_semantic_diff_pd.py --compare --execute --work wcf
  python scripts/12_semantic_diff_pd.py --report
"""
import sys, os, re, json, argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = PROJECT_ROOT / "data" / "tier_c" / "tier_c_pd_translations.jsonl"
LEGACY_DIR = PROJECT_ROOT / "data" / "sources" / "confessions"
OUT = PROJECT_ROOT / "data" / "tier_c" / "semantic_diff_pd.jsonl"

LEGACY_FILES = {
    "wsc": "westminster_shorter_catechism.txt",
    "wlc": "westminster_larger_catechism.txt",
    "heidelberg": "heidelberg_catechism.txt",
    "lcf_1689": "lcf_1689.txt",
    "dort": "canons_of_dort.txt",
    "wcf": "wcf_1647.txt",
}

def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


class _RomanDict(dict):
    def get(self, key, default=None):
        try:
            return _roman_to_int(key) if key else default
        except Exception:
            return default

    def __getitem__(self, key):
        return _roman_to_int(key)

    def __contains__(self, key):
        return bool(key) and all(c in "IVXLivxl" for c in key)


ROMAN = _RomanDict()  # supports any roman numeral (I..XXXIII+), not just a fixed table
DORT_HEAD_WORD = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "THIRD AND FOURTH": 3, "FOURTH": 3, "FIFTH": 5}

REF_RE = re.compile(
    r"\b([1-3]?\s?[A-ZÀ-Úa-zà-ú][a-zà-ú]{1,14}\.?)\s?(\d{1,3})[:.](\d{1,3})(?:[-,]\d{1,3})*",
    re.UNICODE,
)


def load_latest_approved():
    latest = {}
    for line in TRANSLATIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[r["sha"]] = r
    return {k: v for k, v in latest.items() if v.get("approved")}


# ---------------- OUR side: extract an alignment key from the stored EN text ----------------
def our_key_flat(en: str, pattern: str):
    m = re.match(pattern, en)
    return int(m.group(1)) if m and m.group(1).isdigit() else None


def our_keys_by_work(work: str, units: list):
    """-> {key: unit} where key is an int (flat works) or (chapter, article) tuple (nested)."""
    out = {}
    if work == "wsc":
        for u in units:
            k = our_key_flat(u["en"], r"Q\.\s*(\d+)\.")
            if k:
                out[k] = u
    elif work == "wlc":
        for u in units:
            k = our_key_flat(u["en"], r"Question\s+(\d+):")
            if k:
                out[k] = u
    elif work == "heidelberg":
        for u in units:
            m = re.match(r"QUESTION\s+(\d+)\.", u["en"])
            if m:
                out[int(m.group(1))] = u
            # garbled-number headers (e.g. "QUESTION ia7.") are left unaligned on purpose
    elif work == "lcf_1689":
        # Chapter headers are often too short to survive segment()'s length filter and
        # get silently dropped -- do NOT rely on them. Detect a new chapter with TWO
        # signals instead of one: (a) the paragraph number RESETTING (decreasing), or
        # (b) the number exceeding the current chapter's known max (from the legacy
        # file's reliable "CAPÍTULO N" headers) -- (a) alone missed a chapter whose own
        # "1." paragraph was dropped/merged, so its numbering never visibly decreased.
        legacy_max = {}
        for (ch, n) in load_legacy("lcf_1689"):
            legacy_max[ch] = max(legacy_max.get(ch, 0), n)
        max_chapter = max(legacy_max) if legacy_max else 0
        chapter, prev_n = 0, None
        for u in sorted(units, key=lambda x: x["idx"]):
            ma = re.match(r"(\d+)\.\s", u["en"])
            if not ma:
                continue
            n = int(ma.group(1))
            cur_max = legacy_max.get(chapter, 0)
            if prev_n is None or n <= prev_n or (cur_max and n > cur_max):
                chapter = min(chapter + 1, max_chapter) if chapter else 1
            prev_n = n
            out[(chapter, n)] = u
    elif work == "dort":
        # Same reset-detection strategy, keyed off the roman-numeral article number.
        head, prev_n = 0, None
        for u in sorted(units, key=lambda x: x["idx"]):
            ma = re.match(r"Art\.\s*([IVX]+)\.", u["en"])
            if not ma or ma.group(1) not in ROMAN:
                continue
            n = ROMAN[ma.group(1)]
            if prev_n is None or n <= prev_n:
                head += 1
            prev_n = n
            out[(head, n)] = u
    elif work == "wcf":
        # Flat, chapter-level key (our units are one-per-CHAPTER, not per-paragraph --
        # segment() didn't split WCF further; 33 units = 33 chapters).
        for u in units:
            m = re.match(r"CHAPTER\s+([IVXL]+)\.", u["en"])
            if m and m.group(1) in ROMAN:
                out[ROMAN[m.group(1)]] = u
    return out


# ---------------- LEGACY side: parse the PT reference file into the same key space ----------------
def legacy_keys_flat(text: str, item_pattern: str):
    out = {}
    for m in re.finditer(item_pattern, text, flags=re.M):
        n = int(m.group(1))
        start = m.end()
        nxt = re.search(item_pattern, text[start:], flags=re.M)
        end = start + nxt.start() if nxt else len(text)
        out[n] = text[start:end].strip()
    return out


def legacy_keys_nested(text: str, chapter_pattern: str, item_pattern: str, head_word_map=None):
    out = {}
    chapters = list(re.finditer(chapter_pattern, text, flags=re.M))
    for i, cm in enumerate(chapters):
        label = cm.group(1)
        num = head_word_map.get(label.upper()) if head_word_map else int(label)
        seg_start = cm.end()
        seg_end = chapters[i + 1].start() if i + 1 < len(chapters) else len(text)
        seg = text[seg_start:seg_end]
        items = list(re.finditer(item_pattern, seg, flags=re.M))
        for j, im in enumerate(items):
            art_n = int(im.group(1))
            i_start = im.end()
            i_end = items[j + 1].start() if j + 1 < len(items) else len(seg)
            out[(num, art_n)] = seg[i_start:i_end].strip()
    return out


def legacy_keys_dort(text: str):
    """Ignore 'REJEIÇÃO N' sections (full/unabridged legacy text; our source is the
    ABRIDGED positive-articles-only edition) and 'Capítulo N' labels (inconsistent —
    the combined Third/Fourth head uses a different, unmatched header phrase). Detect
    head-group boundaries the same way as our own side: an ARTIGO number resetting.
    Each ARTIGO's stored text stops at the next marker of EITHER kind, so an
    intervening REJEIÇÃO section's content never bleeds into the preceding article."""
    out = {}
    all_markers = sorted(
        [(m.start(), m.end(), "ARTIGO", int(m.group(1))) for m in re.finditer(r"^ARTIGO\s+(\d+)", text, flags=re.M)]
        + [(m.start(), m.end(), "REJEICAO", None) for m in re.finditer(r"^REJEIÇÃO\s+\d+", text, flags=re.M)]
    )
    head, prev_n = 0, None
    for i, (mstart, mend, kind, n) in enumerate(all_markers):
        if kind != "ARTIGO":
            continue
        if prev_n is None or n <= prev_n:
            head += 1
        prev_n = n
        end = all_markers[i + 1][0] if i + 1 < len(all_markers) else len(text)
        out[(head, n)] = text[mend:end].strip()
    return out


def load_legacy(work: str):
    p = LEGACY_DIR / LEGACY_FILES[work]
    text = p.read_text(encoding="utf-8", errors="replace")
    if work in ("wsc", "wlc", "heidelberg"):
        pattern = {"wsc": r"^Pergunta\s+(\d+)\.", "wlc": r"^(\d+)\.\s", "heidelberg": r"^(\d+)\.\s"}[work]
        return legacy_keys_flat(text, pattern)
    if work == "lcf_1689":
        return legacy_keys_nested(text, r"^CAP[ÍI]TULO\s+(\d+)", r"^(\d+)\.")
    if work == "dort":
        return legacy_keys_dort(text)
    if work == "wcf":
        return legacy_keys_wcf(text)
    return {}


def legacy_keys_wcf(text: str) -> dict:
    """Flat chapter-level key (matches our per-chapter units). The file opens with a
    table of contents listing all 33 chapters with dot-leader page numbers
    ('CAPÍTULO 1: ... ..................3') before the real body repeats the same
    headers -- filter out any match followed by a run of dots within the next 200
    chars, which only ever occurs in the TOC, never in the body text."""
    out = {}
    candidates = list(re.finditer(r"CAP[ÍI]TULO\s+(\d+):", text, flags=re.I))
    matches = [m for m in candidates if not re.search(r"\.{3,}", text[m.end():m.end() + 200])]
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[n] = text[start:end].strip()
    return out


# ---------------- deterministic Scripture-reference check ----------------
def extract_refs(text: str) -> set:
    return {f"{m.group(1).strip().rstrip('.')} {m.group(2)}:{m.group(3)}" for m in REF_RE.finditer(text)}


# ---------------- align mode (free) ----------------
def do_align(work_filter=None):
    approved = load_latest_approved()
    by_work = {}
    for r in approved.values():
        by_work.setdefault(r["work"], []).append(r)

    works = [w for w in LEGACY_FILES if (not work_filter or w == work_filter)]
    total_aligned, total_ours, total_legacy = 0, 0, 0
    pairs_by_work = {}
    for w in works:
        ours = our_keys_by_work(w, by_work.get(w, []))
        legacy = load_legacy(w)
        common = set(ours) & set(legacy)
        pairs_by_work[w] = (ours, legacy, common)
        total_aligned += len(common); total_ours += len(ours); total_legacy += len(legacy)
        print(f"  {w:12} nossas(aprovadas c/ chave)={len(ours):4}  legado={len(legacy):4}  "
              f"alinhadas={len(common):4}  cobertura={100*len(common)/max(1,len(ours)):.1f}%")
    print(f"\nTOTAL: nossas={total_ours}  legado={total_legacy}  alinhadas={total_aligned}")
    return pairs_by_work


# ---------------- compare mode (deterministic + paid LLM) ----------------
SEMANTIC_JUDGE_SYS = (
    "Você é um teólogo revisor comparando DUAS traduções em português do MESMO texto "
    "confessional reformado (uma gerada por IA a partir do original em inglês, outra de "
    "uma tradução tradicional já em circulação). Sua única tarefa é avaliar EQUIVALÊNCIA "
    "SEMÂNTICA — não estilo, não fluência. Responda APENAS um JSON: "
    '{"core_claim_match": N (0-100, 100=mesma afirmação teológica central), '
    '"polarity_inverted": true/false (a tradução-IA afirma o OPOSTO/nega o que a outra afirma, '
    'ou vice-versa -- ex: "pode" vs "não pode"), '
    '"omitted_claims": ["..."] (afirmações que estão na tradução tradicional mas NÃO na IA), '
    '"added_claims": ["..."] (afirmações que estão na IA mas NÃO na tradicional), '
    '"notes": "..."}'
)
# deepseek-v4-flash sometimes reasons before emitting the JSON verdict (Lesson #18 --
# same bug already fixed in 10_retranslate_pd.py's judge/translate steps). WCF units are
# whole CHAPTERS (much longer than the paragraph/question-level units in other works),
# so there's more to "reason" about -- 1024 wasn't enough (18/27 WCF pairs hit
# finish_reason=length with empty content). Bump with headroom.
JUDGE_MAX_TOKENS = 8192


def translate_max_note():
    return "usa scripts/utils/api_client.py OpenRouterClient (mesmo padrão do 10_retranslate_pd.py)"


def do_compare(args):
    pairs_by_work = do_align(args.work)
    todo = []
    done_shas = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("core_claim_match") is not None:
                    done_shas.add(d["sha"])

    for w, (ours, legacy, common) in pairs_by_work.items():
        for key in common:
            u = ours[key]
            if u["sha"] in done_shas:
                continue
            todo.append({"work": w, "key": str(key), "sha": u["sha"], "ptbr": u["ptbr"], "legacy_pt": legacy[key]})
    if args.limit:
        todo = todo[: args.limit]

    est_calls = len(todo)  # 1 semantic-judge call per pair (deterministic ref-check is free)
    print(f"\nPares alinhados a comparar: {len(todo)} | já feitos (resume): {len(done_shas)} | "
          f"chamadas estimadas: ~{est_calls} | custo aprox.: ~US$ {est_calls * 0.001:.2f}")
    if not args.execute:
        print("\nDRY-RUN — nenhum gasto de API. Rode com --execute para comparar de verdade.")
        return
    if not todo:
        print("Nada a fazer — todos os pares já comparados.")
        return

    import asyncio, httpx
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    from utils.api_client import OpenRouterClient
    from utils.cost_tracker import CostTracker, CostLimitExceeded
    from utils.progress import ProgressBar

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = args.model or os.getenv("OPENROUTER_MODEL_TRANSLATOR", "deepseek/deepseek-v4-flash")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (env or .env).")

    raw_dir = PROJECT_ROOT / "logs" / "raw" / "semantic_diff_pd"
    api = OpenRouterClient(api_key=api_key, base_url=base_url, log_raw_dir=raw_dir)
    tracker = CostTracker(limit_usd=args.cost_limit)
    sem = asyncio.Semaphore(8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    bar = ProgressBar(total=len(todo), label=f"semantic-diff ({model})")

    def _parse(txt):
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    async def process(client, item):
        async with sem:
            rec = dict(item)
            ref_ours = extract_refs(item["ptbr"])
            ref_legacy = extract_refs(item["legacy_pt"])
            rec["refs_ours_only"] = sorted(ref_ours - ref_legacy)
            rec["refs_legacy_only"] = sorted(ref_legacy - ref_ours)
            rec["core_claim_match"] = None
            try:
                user = f"TRADUÇÃO IA:\n{item['ptbr']}\n\nTRADUÇÃO TRADICIONAL:\n{item['legacy_pt']}"
                resp = await api.chat(
                    client, model=model,
                    messages=[{"role": "system", "content": SEMANTIC_JUDGE_SYS},
                              {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, seed=42,
                    log_key=f"{item['sha']}_semdiff",
                )
                tracker.add(api.estimate_cost_usd(resp, model))
                j = _parse(api.extract_text(resp))
                if j:
                    rec.update({k: j.get(k) for k in
                               ("core_claim_match", "polarity_inverted", "omitted_claims", "added_claims", "notes")})
            except CostLimitExceeded:
                raise
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            async with write_lock:
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                bar.update(1)
            return rec

    async def run():
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                for coro in asyncio.as_completed([process(client, it) for it in todo]):
                    await coro
            except CostLimitExceeded as e:
                print(f"\nLimite de custo atingido: {e} — progresso salvo; rode de novo para retomar.")

    asyncio.run(run())
    print(f"\nGasto: US$ {tracker.total:.2f} | saída: {OUT}")


def do_report():
    if not OUT.exists():
        sys.exit(f"Nada ainda: {OUT}")
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {r["sha"]: r for r in rows}
    import collections
    by_work = collections.defaultdict(list)
    for r in latest.values():
        by_work[r["work"]].append(r)
    print(f"Total de pares comparados: {len(latest)}\n")
    for w, items in sorted(by_work.items()):
        scored = [r for r in items if r.get("core_claim_match") is not None]
        if not scored:
            print(f"  {w:12} (sem comparações concluídas)")
            continue
        mean_score = sum(r["core_claim_match"] for r in scored) / len(scored)
        inverted = [r for r in scored if r.get("polarity_inverted")]
        ref_mismatch = [r for r in scored if r.get("refs_ours_only") or r.get("refs_legacy_only")]
        low = [r for r in scored if r["core_claim_match"] < 90]
        print(f"  {w:12} n={len(scored):4}  media_score={mean_score:5.1f}  "
              f"⚠️inversao_polaridade={len(inverted)}  ⚠️score<90={len(low)}  "
              f"ref_biblica_diverge={len(ref_mismatch)}")
        for r in inverted:
            print(f"      🔴 POLARIDADE INVERTIDA: {w} {r['key']} — {r.get('notes','')[:120]}")
        for r in low[:5]:
            print(f"      🟡 score={r['core_claim_match']}: {w} {r['key']} — {r.get('notes','')[:100]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--work", help="single work id (e.g. wcf)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--cost-limit", type=float, default=5.0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.align:
        do_align(a.work)
    elif a.compare:
        do_compare(a)
    elif a.report:
        do_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
