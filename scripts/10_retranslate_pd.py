"""10_retranslate_pd.py — Re-translate PUBLIC-DOMAIN confessions to pt-BR with our
own AI pipeline, so Tier C becomes redistributable (our translation + PD source =
no third-party copyright). See docs/PD_RETRANSLATION_SPEC.md.

HARD RULE: translate ONLY from the PD originals in configs/pd_sources.json —
never from a copyrighted modern PT edition.

Modes
  --fetch                 download PD sources (free HTTP), clean, save to
                          data/sources/confessions_pd/{id}.pd.txt   [no API spend]
  --translate             segment + translate + judge-QA -> data/tier_c/tier_c_pd.jsonl
                          (PAID; prints a cost estimate and does nothing unless --execute)
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


def translate(args):
    if not SRC_DIR.exists():
        sys.exit("No sources. Run --fetch first.")
    files = sorted(SRC_DIR.glob("*.pd.txt"))
    if args.source:
        files = [f for f in files if f.stem.replace(".pd", "") == args.source]
    total_units, per = 0, []
    for f in files:
        units = segment(f.read_text(encoding="utf-8"))
        per.append((f.stem, len(units)))
        total_units += len(units)
    print("Units per work:")
    for name, n in per:
        print(f"  {name:<14} {n:>4}")
    # ~1 translate call + 3 judge calls per unit; ~600 tok each on a cheap model
    est_calls = total_units * 4
    est_usd = est_calls * 0.0007
    print(f"\nTOTAL units: {total_units} | est. API calls: ~{est_calls} "
          f"(1 translate + 3 judge / unit) | rough cost: ~US$ {est_usd:.2f} (flash tier)")
    if not args.execute:
        print("\nDRY-RUN — no API spent. Re-run with --execute (and OPENROUTER_API_KEY set) to translate.")
        return
    # ---- PAID path (explicit) ----
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from utils.api_client import OpenRouterClient           # noqa
    from utils.cost_tracker import CostTracker              # noqa
    print("\n[execute] translating via OpenRouterClient + judge QA (>=95) ... (implements docs/PD_RETRANSLATION_SPEC.md)")
    # NOTE: fill the OpenRouterClient calls to translate each unit EN->pt-BR, then
    # judge-QA (fidelity/fluency/terminology/style/completeness >=95), append passing
    # units to tier_c_pd.jsonl with provenance {source, url, license:'PD original',
    # translator_model, ai_translated:true}. Pastoral review gate BEFORE merge.
    raise SystemExit("Paid translation step is intentionally not auto-run here — "
                     "wire the OpenRouterClient calls per the spec, then merge after pastoral review.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--source", help="single work id (e.g. wsc)")
    ap.add_argument("--dry-run", action="store_true", help="translate: estimate only (default)")
    ap.add_argument("--execute", action="store_true", help="translate: actually spend API")
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
