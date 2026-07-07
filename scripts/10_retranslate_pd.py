"""10_retranslate_pd.py — Re-translate PUBLIC-DOMAIN confessions to pt-BR with our
own AI pipeline, so Tier C becomes redistributable (our translation + PD source =
no third-party copyright). See docs/PD_RETRANSLATION_SPEC.md.

HARD RULE: translate ONLY from the PD originals in configs/pd_sources.json —
never from a copyrighted modern PT edition.

Modes
  --fetch                 download PD sources (free HTTP), clean, save to
                          data/sources/confessions_pd/{id}.pd.txt   [no API spend]
  --translate             segment + translate + 3-LLM judge panel (MEDIAN >= 93)
                          -> data/tier_c/tier_c_pd_translations.jsonl
                          (PAID; prints a cost estimate and does nothing unless --execute)
                          judges: google/gemini-3.5-flash, openai/gpt-oss-120b,
                          xiaomi/mimo-v2.5 (override: env OPENROUTER_PD_JUDGES)
  --build                 (stub) hand off to scripts/merge_dataset.py

Examples
  python scripts/10_retranslate_pd.py --fetch
  python scripts/10_retranslate_pd.py --translate --dry-run          # cost estimate only
  python scripts/10_retranslate_pd.py --translate --execute --source wsc
"""
import sys, os, re, json, argparse, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data" / "sources" / "confessions_pd"
OUT = PROJECT_ROOT / "data" / "tier_c" / "tier_c_pd.jsonl"
MANIFEST = PROJECT_ROOT / "configs" / "pd_sources.json"
UA = {"User-Agent": "OpenScriptura/1.0 (PD confession sourcing; +github.com/openscriptura)"}


# ------------------------------- cleaning -------------------------------------
def strip_wikitext(t: str) -> str:
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    for _ in range(6):  # nested templates
        t2 = re.sub(r"\{\{[^{}]*\}\}", "", t, flags=re.S)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)          # tables
    t = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)      # [[a|b]] -> b
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)               # [[a]] -> a
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)    # [url text] -> text
    t = re.sub(r"'''''|'''|''", "", t)                     # bold/italic
    t = re.sub(r"(?m)^=+\s*(.*?)\s*=+\s*$", r"\1", t)       # == header == -> text
    t = re.sub(r"(?m)^[*#:;]+\s*", "", t)                   # list markers
    t = re.sub(r"<[^>]+>", "", t)                          # stray html
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_html(t: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&#160;", " "), ("&#8217;", "'"), ("&quot;", '"')]:
        t = t.replace(a, b)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ------------------------------- fetch ----------------------------------------
def fetch():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]
    print(f"Sourcing {len(items)} PD works -> {SRC_DIR}\n")
    for it in items:
        try:
            req = urllib.request.Request(it["url"], headers=UA)
            raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            if it["format"] == "wikitext":
                clean = strip_wikitext(raw)
            elif it["format"] == "text":                 # archive.org djvu OCR
                clean = re.sub(r"(?m)^\s*\d+\s*$", "", raw)   # drop bare page numbers
                clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
            else:
                clean = strip_html(raw)
            p = SRC_DIR / f"{it['id']}.pd.txt"
            header = (f"# SOURCE: {it['title']}\n# {it['source']} | {it['license']}\n"
                      f"# url: {it['url']}\n# (public-domain original — safe to re-translate)\n\n")
            p.write_text(header + clean + "\n", encoding="utf-8")
            n = len(clean)
            flag = "OK" if n > 3000 else "SHORT — needs manual clean"
            mark = "" if it["status"] == "fetchable" else "  [needs-cleaning: verify/replace]"
            print(f"  [{flag:>28}] {it['id']:<10} {n:>7} chars -> {p.name}{mark}")
        except Exception as e:
            print(f"  [{'FAILED':>28}] {it['id']:<10} {type(e).__name__}: {e}")
    print("\nNo API spent. Review data/sources/confessions_pd/*.pd.txt before --translate.")
    print("Any 'SHORT'/'needs-cleaning' item: drop a clean PD text there manually (source in configs/pd_sources.json).")


# --------------------------- segment + translate ------------------------------
def segment(text: str):
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    units = re.split(r"\n(?=(?:Q\.?\s*\d+|\d+\.\s|Question\s+\d+|CHAPTER|Chapter|Article|ARTICLE|Head|HEAD))", body)
    return [u.strip() for u in units if len(u.strip()) > 40]


TRANSLATE_SYS = (
    "Você é um tradutor teológico especializado em textos confessionais reformados. "
    "Traduza o texto do inglês para o português brasileiro (pt-BR) com FIDELIDADE TOTAL: "
    "preserve a numeração de perguntas/capítulos/artigos, a estrutura, e as referências "
    "bíblicas (abrevie no padrão pt-BR, ex.: Rm 11.36; 1Co 10.31). Use a terminologia "
    "reformada clássica consagrada em português (ex.: 'justificação', 'santificação', "
    "'pacto', 'eleição incondicional'). NÃO adicione comentários, notas ou explicações — "
    "devolva SOMENTE a tradução."
)

JUDGE_SYS = (
    "Você é um revisor de traduções teológicas (inglês -> pt-BR) de confissões reformadas. "
    "Avalie a tradução em 5 critérios (0-100): fidelidade (sentido preservado, nada "
    "adicionado/omitido), fluencia (português natural), terminologia (termos teológicos "
    "consagrados), estilo (registro confessional adequado), completude (nada faltando, "
    "numeração e referências preservadas). Responda APENAS um JSON: "
    '{"fidelidade":N,"fluencia":N,"terminologia":N,"estilo":N,"completude":N,'
    '"score":N,"problemas":["..."]} onde score é sua nota global (0-100).'
)

TRANSLATIONS_OUT = PROJECT_ROOT / "data" / "tier_c" / "tier_c_pd_translations.jsonl"
MIN_SCORE = 93.0  # MEDIAN of the 3-judge panel must be >= 93 (0-100)
# 3 DIFFERENT judge models (panel decision: median-of-3, diverse judges — avoids
# self-preference of the translator model grading itself). Override via env
# OPENROUTER_PD_JUDGES (comma-separated).
JUDGE_MODELS = tuple(
    m.strip() for m in os.getenv(
        "OPENROUTER_PD_JUDGES",
        "google/gemini-3.5-flash,openai/gpt-oss-120b,xiaomi/mimo-v2.5",
    ).split(",") if m.strip()
)


def _unit_sha(work: str, text: str) -> str:
    import hashlib
    return hashlib.sha256(f"{work}\n{text}".encode("utf-8")).hexdigest()[:16]


def _collect_units(source_filter=None):
    files = sorted(SRC_DIR.glob("*.pd.txt"))
    if source_filter:
        files = [f for f in files if f.stem.replace(".pd", "") == source_filter]
    per, units = [], []
    for f in files:
        work = f.stem.replace(".pd", "")
        us = segment(f.read_text(encoding="utf-8"))
        per.append((work, len(us)))
        for i, u in enumerate(us):
            units.append({"work": work, "idx": i, "en": u, "sha": _unit_sha(work, u)})
    return per, units


def _load_done() -> set:
    done = set()
    if TRANSLATIONS_OUT.exists():
        for line in TRANSLATIONS_OUT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # done = has a translation and a valid score (approved or not);
            # only re-run rows that errored out (ptbr None / score None)
            if r.get("ptbr") and r.get("score_final") is not None:
                done.add(r["sha"])
    return done


def translate(args):
    if not SRC_DIR.exists():
        sys.exit("No sources. Run --fetch first.")
    per, units = _collect_units(args.source)
    print("Units per work:")
    for name, n in per:
        print(f"  {name:<24} {n:>4}")
    done = _load_done()
    todo = [u for u in units if u["sha"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    est_calls = len(todo) * (1 + len(JUDGE_MODELS))
    print(f"\nJudges (median >= {MIN_SCORE:.0f}): {', '.join(JUDGE_MODELS)}")
    print(f"TOTAL units: {len(units)} | done (resume): {len(done & {u['sha'] for u in units})} "
          f"| to run: {len(todo)} | est. calls: ~{est_calls} | rough cost: ~US$ {est_calls * 0.0009:.2f}")
    if not args.execute:
        print("\nDRY-RUN — no API spent. Re-run with --execute (and OPENROUTER_API_KEY set) to translate.")
        return
    if not todo:
        print("Nothing to do — all units already translated (resume). See", TRANSLATIONS_OUT)
        return

    # ---- PAID path ----
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

    api = OpenRouterClient(api_key=api_key, base_url=base_url)
    tracker = CostTracker(limit_usd=args.cost_limit)
    sem = asyncio.Semaphore(8)
    TRANSLATIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    bar = ProgressBar(total=len(todo), label=f"translate+QA ({model})")
    print(f"\n[execute] {len(todo)} units | translator+judge = {model} | "
          f"cost limit US$ {args.cost_limit:.2f} | out: {TRANSLATIONS_OUT}\n")

    def _parse_judge(txt: str):
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        try:
            j = json.loads(m.group(0))
            return float(j.get("score")) if j.get("score") is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    async def process(client, u):
        async with sem:
            rec = {"work": u["work"], "idx": u["idx"], "sha": u["sha"], "en": u["en"],
                   "ptbr": None, "scores": [], "score_final": None, "approved": False,
                   "translator_model": model, "judge_models": list(JUDGE_MODELS),
                   "aggregation": f"median >= {MIN_SCORE:.0f}",
                   "license": "PD original (see configs/pd_sources.json)", "ai_translated": True}
            try:
                t_resp = await api.chat(
                    client, model=model,
                    messages=[{"role": "system", "content": TRANSLATE_SYS},
                              {"role": "user", "content": u["en"]}],
                    temperature=0.0, max_tokens=3072, seed=42,
                )
                tracker.add(api.estimate_cost_usd(t_resp, model))
                ptbr = api.extract_text(t_resp).strip()
                if not ptbr:
                    raise ValueError("empty translation")
                rec["ptbr"] = ptbr
                judge_user = f"TEXTO ORIGINAL (EN):\n{u['en']}\n\nTRADUÇÃO (pt-BR):\n{ptbr}"
                for jm in JUDGE_MODELS:                  # 3 DIFFERENT judge LLMs
                    try:
                        j_resp = await api.chat(
                            client, model=jm,
                            messages=[{"role": "system", "content": JUDGE_SYS},
                                      {"role": "user", "content": judge_user}],
                            temperature=0.0, max_tokens=1024, seed=42,  # 1024: Lesson #18
                        )
                        tracker.add(api.estimate_cost_usd(j_resp, jm))
                        s = _parse_judge(api.extract_text(j_resp))
                    except CostLimitExceeded:
                        raise
                    except Exception as je:              # one judge down != unit lost
                        s = None
                        rec.setdefault("judge_errors", []).append(f"{jm}: {type(je).__name__}")
                    rec["scores"].append({"judge": jm, "score": s})
                valid = [d["score"] for d in rec["scores"] if d["score"] is not None]
                if len(valid) >= 2:                      # median needs a real panel
                    import statistics
                    rec["score_final"] = statistics.median(valid)
                    rec["n_judges"] = len(valid)
                    rec["approved"] = rec["score_final"] >= MIN_SCORE
                else:
                    rec["error"] = f"only {len(valid)}/3 judges returned a score"
            except CostLimitExceeded:
                raise
            except Exception as e:                       # keep the row; resume re-runs it
                rec["error"] = f"{type(e).__name__}: {e}"
            async with write_lock:
                with TRANSLATIONS_OUT.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                bar.update(1)
            return rec

    async def run():
        async with httpx.AsyncClient(timeout=120) as client:
            results = []
            try:
                for coro in asyncio.as_completed([process(client, u) for u in todo]):
                    results.append(await coro)
            except CostLimitExceeded as e:
                print(f"\nCOST LIMIT hit: {e} — partial progress saved; re-run to resume.")
            return results

    results = asyncio.run(run())
    ok = [r for r in results if r.get("approved")]
    low = [r for r in results if r.get("score_final") is not None and not r.get("approved")]
    err = [r for r in results if r.get("error")]
    print(f"\nDone. approved(>= {MIN_SCORE:.0f}): {len(ok)} | below-threshold: {len(low)} | "
          f"errors (will re-run on resume): {len(err)} | spent: US$ {tracker.total:.2f}")
    print(f"Output: {TRANSLATIONS_OUT}")
    print("NEXT: pastoral review of the approved translations, then --build / merge_dataset.py.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--source", help="single work id (e.g. wsc)")
    ap.add_argument("--dry-run", action="store_true", help="translate: estimate only (default)")
    ap.add_argument("--execute", action="store_true", help="translate: actually spend API")
    ap.add_argument("--model", help="translator+judge model (default: env OPENROUTER_MODEL_TRANSLATOR or deepseek/deepseek-v4-flash)")
    ap.add_argument("--cost-limit", type=float, default=10.0, help="hard USD stop (default 10)")
    ap.add_argument("--limit", type=int, help="max units this run (smoke test)")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    elif a.translate:
        translate(a)
    elif a.build:
        print("Build/merge: after pastoral review, run scripts/merge_dataset.py to fold tier_c_pd.jsonl in.")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
