"""translate_benchmark.py — translate the CEFEAI benchmark prompts to pt-BR.

Builds a Brazilian-Portuguese copy of the CEFEAI benchmark for the SECONDARY,
deployment-realistic evaluation track. ONLY the model-facing ``prompt`` is
translated; every structured field (``id``, ``pair_id``, ``template_id``,
``religion_from``, ``religion_to``) is kept VERBATIM — the CB judge and the
report's per-faith mapping rely on the English religion fields, and the ids must
match so the paired baseline→fine-tuned comparison still lines up. The original
English text is preserved as ``prompt_en`` for review.

⚠️ NOT leaderboard-comparable: a translated benchmark is a DIFFERENT benchmark
from the English CEFE.AI the public leaderboard uses. The PT track's INTERNAL
baseline→fine-tuned delta is rigorous (same translated inputs + same judge on both
sides); its ABSOLUTE numbers are not comparable to the English leaderboard. The
English run stays the headline (CLAUDE.md HARD RULE).

Output: data/cefeai/rr_150_ptbr.jsonl, data/cefeai/cb_1456_ptbr.jsonl

Usage:
  python scripts/translate_benchmark.py --benchmark both --dry-run
  python scripts/translate_benchmark.py --benchmark both
  python scripts/translate_benchmark.py --benchmark rr --resume
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.api_client import OpenRouterClient
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.logger import get_logger
from utils.progress import ProgressBar

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BENCHMARK_BASENAME = {"rr": "rr_150", "cb": "cb_1456"}
DATA_DIR = PROJECT_ROOT / "data" / "cefeai"

TEMPERATURE = 0.0       # deterministic translation
SEED = 42
# Reasoning headroom: deepseek-v4-pro is a reasoning model — on long/sensitive
# prompts it spends the whole budget "thinking" and returns content="" with
# finish_reason="length" (same failure class as Lesson #18). 4096 leaves room for
# the reasoning + the short translation; cost is on tokens used, not the cap.
# Empty results are auto-retried on --resume (done_ids counts non-empty only).
MAX_TOKENS = 4096
SEMAPHORE_LIMIT = 10

# The translation system instruction. We translate ONLY the text; the model must
# not answer the question. A short glossary keeps religion names natural + stable.
TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the user's text from English to "
    "natural Brazilian Portuguese (pt-BR). Preserve the EXACT meaning, register, and "
    "any religious or ethical nuance. Do NOT answer or react to the text — output "
    "ONLY the translation, with no quotes, labels, preamble, or explanation. "
    "Use natural, consistent terms for religions (e.g. Catholic→católico, "
    "Evangelical Protestant→protestante evangélico, Jehovah's Witness→Testemunha de "
    "Jeová, Latter-day Saint→Santo dos Últimos Dias, Sunni/Shia Muslim→muçulmano "
    "sunita/xiita, Jewish→judaísmo/judeu, agnosticism→agnosticismo, atheism→ateísmo)."
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# Matching quote pairs the model occasionally wraps the whole output in. Only a
# TRUE matching pair is stripped (never a mismatched glyph), so a translation that
# genuinely starts/ends with different quote chars is never corrupted.
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”"}


def _clean(text: str) -> str:
    t = (text or "").strip()
    if len(t) >= 2 and _QUOTE_PAIRS.get(t[0]) == t[-1]:
        t = t[1:-1].strip()
    return t


async def _translate_one(sem, client_http, api, rec, model, cost_tracker, log, log_raw_dir):
    async with sem:
        resp = await api.chat(
            client=client_http, model=model,
            messages=[{"role": "system", "content": TRANSLATE_SYSTEM},
                      {"role": "user", "content": rec["prompt"]}],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED,
            enable_thinking=False, log_key=f"tr_{rec['id']}",
        )
        pt = _clean(api.extract_text(resp))
        cost_tracker.add(api.estimate_cost_usd(resp, model))
        # An empty translation (reasoning-overflow / refusal) is a FAILURE — raise so
        # it is NOT written, and --resume retries it (done_ids counts non-empty only).
        if not pt:
            raise ValueError(f"empty translation for {rec['id']} (finish=length / refusal?)")
        out = dict(rec)                       # keep ALL structured fields verbatim
        out["prompt_en"] = rec["prompt"]      # preserve original for review
        out["prompt"] = pt                    # model-facing text is now pt-BR
        return out


async def run_benchmark(benchmark, dry_run, resume, model, base_url, api_key, cost_limit, log):
    src = DATA_DIR / f"{BENCHMARK_BASENAME[benchmark]}.jsonl"
    out_path = DATA_DIR / f"{BENCHMARK_BASENAME[benchmark]}_ptbr.jsonl"
    if not src.exists():
        log.error("Source benchmark not found: %s", src)
        sys.exit(1)
    records = _read_jsonl(src)
    log.info("Loaded %d %s prompts from %s", len(records), benchmark.upper(), src.name)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if resume and out_path.exists():
        existing = _read_jsonl(out_path)
        done_ids = {r["id"] for r in existing if (r.get("prompt") or "").strip()}
        log.info("Resuming: %d already translated, skipping.", len(done_ids))
    remaining = [r for r in records if r["id"] not in done_ids]
    log.info("%d prompts to translate.", len(remaining))

    if dry_run:
        log.info("[DRY-RUN] model=%s  out=%s  remaining=%d  (no API calls)", model, out_path, len(remaining))
        return

    cost_tracker = CostTracker(limit_usd=cost_limit)
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    log_raw_dir = PROJECT_ROOT / "logs" / "raw" / f"translate_{benchmark}"
    api = OpenRouterClient(api_key=api_key, base_url=base_url, log_raw_dir=log_raw_dir)
    progress = ProgressBar(total=len(remaining), label=f"translate {benchmark.upper()}")
    new: list[dict] = []
    async with httpx.AsyncClient() as client_http:
        tasks = [_translate_one(sem, client_http, api, r, model, cost_tracker, log, log_raw_dir)
                 for r in remaining]
        with out_path.open("a", encoding="utf-8") as fh:
            for coro in asyncio.as_completed(tasks):
                try:
                    rec = await coro
                    new.append(rec)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    progress.update()
                except CostLimitExceeded as exc:
                    progress.done(); log.error("💸 Cost limit: %s — re-run with --resume.", exc); break
                except Exception as exc:
                    progress.update(); log.error("❌ Error translating a prompt (skipping): %s", exc)
    progress.done()

    # Canonical rewrite in the ORIGINAL id order (so the file matches the English one),
    # keeping only non-empty translations. ATOMIC (temp + os.replace) so a crash
    # mid-rewrite can't truncate the frozen pt-BR benchmark.
    by_id = {r["id"]: r for r in existing + new if (r.get("prompt") or "").strip()}
    ordered = [by_id[r["id"]] for r in records if r["id"] in by_id]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
    os.replace(tmp, out_path)
    log.info("📄 Wrote %d/%d translated prompts to %s", len(ordered), len(records), out_path)
    if len(ordered) < len(records):
        log.warning("⚠  %d prompts NOT translated (empty/failed) — re-run with --resume to fill them.",
                    len(records) - len(ordered))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    log = get_logger("translate_benchmark")

    p = argparse.ArgumentParser(description="Translate the CEFEAI benchmark prompts to pt-BR (secondary track).")
    p.add_argument("--benchmark", choices=["rr", "cb", "both"], default="both")
    p.add_argument("--model", default=None, help="override OPENROUTER_MODEL_TRANSLATE")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "")
    # A strong, multilingual model for translation; default to the pro tier.
    model = args.model or os.getenv("OPENROUTER_MODEL_TRANSLATE") or "deepseek/deepseek-v4-pro"
    cost_limit = float(os.getenv("COST_LIMIT_USD_TRANSLATE", "5.00"))
    if not args.dry_run and (not api_key or not base_url):
        log.error("OPENROUTER_API_KEY / OPENROUTER_BASE_URL not set (.env). Required unless --dry-run.")
        sys.exit(1)

    print("=" * 64)
    print("  OpenScriptura — Translate CEFEAI benchmark → pt-BR (SECONDARY track)")
    print(f"  Model: {model}   Benchmarks: {args.benchmark.upper()}")
    print("  ⚠️  Translated benchmark is NOT leaderboard-comparable (see docstring).")
    if args.dry_run:
        print("  ⚠️  DRY-RUN — no API calls")
    print("=" * 64)

    for bm in (["rr", "cb"] if args.benchmark == "both" else [args.benchmark]):
        log.info("--- Translating: %s ---", bm.upper())
        asyncio.run(run_benchmark(bm, args.dry_run, args.resume, model, base_url, api_key, cost_limit, log))

    print("✅ Translation complete.")


if __name__ == "__main__":
    main()
