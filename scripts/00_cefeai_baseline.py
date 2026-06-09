"""00_cefeai_baseline.py — CEFEAI benchmark baseline for Qwen3-8B.

Runs the raw (zero fine-tuning) Qwen3-8B model against the CEFEAI Religious
Representation (150 prompts) and/or Conversion Bias (1,456 pairs) benchmarks
via OpenRouter, judges each response with DeepSeek, and writes JSONL results
plus a summary JSON with Wilson confidence intervals.

Blocking items addressed (VALIDATION_REPORT.md):
  M2  — enable_thinking=False on all model API calls
  M8  — Wilson CI on all proportion metrics in summary
  M10 — tenacity retry with exponential backoff on every API call
  M11 — hard-stop cost guardrail via CostTracker

Usage:
  python scripts/00_cefeai_baseline.py --benchmark rr --dry-run
  python scripts/00_cefeai_baseline.py --benchmark rr
  python scripts/00_cefeai_baseline.py --benchmark both --resume
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Resolve project root so this script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.api_client import OpenRouterClient
from utils.cefeai import (
    build_judge_prompt,
    format_console_summary,
    load_scoring_prompt,
    load_system_prompt,
    parse_judge_score,
    results_are_legacy_schema,
    summarize,
)
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.logger import get_logger
from utils.progress import ProgressBar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_FILES: dict[str, Path] = {
    "rr": PROJECT_ROOT / "data" / "cefeai" / "rr_150.jsonl",
    "cb": PROJECT_ROOT / "data" / "cefeai" / "cb_1456.jsonl",
}

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Inference settings — identical for baseline and post-training eval so the
# comparison is valid (VALIDATION_REPORT.md M2, Panel 1).
#
# Headline protocol = v1 (NO system prompt) — CEFEAI-leaderboard-comparable, and
# the only setup that isolates the fine-tuning effect (weights only). The v2
# "deployment-behavior" protocol (--system-prompt) was tried and REJECTED as the
# headline: the Reformed prompt alone saturated the raw model (RR 99.3% / CB
# 87.8%), which is neither comparable to the prompt-free leaderboard nor able to
# show what fine-tuning added. v2 is kept only as an opt-in deployment datapoint.
# See CLAUDE.md Lessons #14–#16 and IMPLEMENTATION_PLAN.md "Protocol v1 vs v2".
ENABLE_THINKING = False   # Qwen3: disable <think> tokens (matches training format)
TEMPERATURE = 0.0         # deterministic, reproducible
MAX_TOKENS = 1024         # headroom so answers are not truncated (truncation would
                          # bias the judge, esp. against a verbose fine-tuned model).
                          # Same value on baseline + fine-tuned → comparison stays fair.
SEED_OPENROUTER = 42      # passed to OpenRouter; honoured on a best-effort basis
JUDGE_MAX_TOKENS = 256    # judge verdict is short (RR ~1-sentence JSON / CB "Rating: N")
JUDGE_ENABLE_THINKING = False  # deterministic + verdict can't be truncated by a think block

SEMAPHORE_LIMIT = 10      # concurrent async requests (VALIDATION_REPORT.md R12)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# The judge prompt, parsing, and aggregation all come from utils.cefeai, which
# loads the OFFICIAL CEFE.AI scoring_prompt.json (configs/cefeai/) verbatim — so
# the baseline and the Phase 4 eval score identically, and identically to CEFE.AI.

def _get_env(key: str, default: str | None = None, required: bool = True) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise EnvironmentError(
            f"Required environment variable {key!r} is not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return val or ""


def _load_processed_ids(results_file: Path) -> tuple[set[str], list[dict]]:
    """Read existing JSONL and return (set of prompt_ids, list of records)."""
    if not results_file.exists():
        return set(), []
    ids: set[str] = set()
    records: list[dict] = []
    with results_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                ids.add(rec["prompt_id"])
                records.append(rec)
    return ids, records


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------

async def _process_one(
    semaphore: asyncio.Semaphore,
    client_http: httpx.AsyncClient,
    api: OpenRouterClient,
    prompt_record: dict,
    benchmark: str,
    model: str,
    judge_model: str,
    cost_tracker: CostTracker,
    log: object,
    log_raw_dir: Path,
    system_prompt: str | None = None,
) -> dict:
    prompt_id: str = prompt_record["id"]
    prompt_text: str = prompt_record["prompt"]

    async with semaphore:
        # --- Model call ---
        # Protocol v2: prepend the Reformed system prompt when provided, so the
        # raw baseline is evaluated under the same conditions as the fine-tuned
        # model. system_prompt=None reproduces the legacy v1 (no system prompt).
        model_messages = []
        if system_prompt:
            model_messages.append({"role": "system", "content": system_prompt})
        model_messages.append({"role": "user", "content": prompt_text})
        model_response = await api.chat(
            client=client_http,
            model=model,
            messages=model_messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED_OPENROUTER,
            enable_thinking=ENABLE_THINKING,
            log_key=f"{prompt_id}_model",
        )
        response_text = api.extract_text(model_response)
        model_cost = api.estimate_cost_usd(model_response, model)

        # --- Judge call (official CEFE.AI prompt, loaded from configs/cefeai/) ---
        # RR needs {question, response}; CB needs {religion_from, religion_to, response}.
        if benchmark == "cb":
            judge_prompt = build_judge_prompt(
                "cb",
                religion_from=prompt_record["religion_from"],
                religion_to=prompt_record["religion_to"],
                response=response_text,
            )
        else:
            judge_prompt = build_judge_prompt("rr", question=prompt_text, response=response_text)
        judge_response = await api.chat(
            client=client_http,
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            seed=SEED_OPENROUTER,
            enable_thinking=JUDGE_ENABLE_THINKING,
            log_key=f"{prompt_id}_judge",
        )
        judge_raw = api.extract_text(judge_response)
        judge_score, judge_rationale = parse_judge_score(benchmark, judge_raw)
        judge_cost = api.estimate_cost_usd(judge_response, judge_model)

        total_cost = model_cost + judge_cost
        cost_tracker.add(total_cost)  # raises CostLimitExceeded if over limit

        log.debug(
            "%-12s score=%s  cost=$%.6f  cumulative=$%.4f",
            prompt_id, judge_score, total_cost, cost_tracker.total,
        )

        rec = {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "model": model,
            "response": response_text,
            "judge_model": judge_model,
            "judge_score": judge_score,          # int on official scale, or None (parse error)
            "judge_rationale": judge_rationale,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": round(total_cost, 6),
            "enable_thinking": ENABLE_THINKING,
        }
        if benchmark == "cb":   # carry CB fields for by-pair/tradition/template aggregation
            rec["pair_id"] = prompt_record.get("pair_id")
            rec["template_id"] = prompt_record.get("template_id")
            rec["religion_from"] = prompt_record.get("religion_from")
            rec["religion_to"] = prompt_record.get("religion_to")
        return rec


# ---------------------------------------------------------------------------
# Per-benchmark orchestration
# ---------------------------------------------------------------------------

async def run_benchmark(
    benchmark: str,
    dry_run: bool,
    resume: bool,
    model: str,
    judge_model: str,
    base_url: str,
    api_key: str,
    cost_limit: float,
    log,
    system_prompt: str | None = None,
) -> bool:
    """Run one benchmark (rr or cb).  Returns True on success / clean dry-run."""
    benchmark_file = BENCHMARK_FILES[benchmark]

    # --- Check benchmark file ---
    if not benchmark_file.exists():
        log.error("Benchmark file not found: %s", benchmark_file)
        log.error("Expected: %s", benchmark_file.resolve())
        log.error(
            "Download the CEFEAI %s benchmark and place it at the path above.",
            benchmark.upper(),
        )
        log.error("See https://cefe.ai for benchmark access.")
        if dry_run:
            log.warning("[DRY-RUN] Skipping %s — file missing (would abort in real run).", benchmark.upper())
            return False
        sys.exit(1)

    # --- Load prompts ---
    prompts: list[dict] = []
    with benchmark_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    log.info("Loaded %d prompts from %s", len(prompts), benchmark_file)

    # --- Results paths ---
    # Tag outputs by prompt mode so the v2 (sysprompt) baseline never clobbers
    # the legacy v1 (noprompt) files — and so 07_cefeai_eval.py can find the
    # baseline that matches its own prompt mode.
    model_slug  = model.replace("/", "_").replace("-", "_")
    prompt_mode = "sysprompt" if system_prompt else "noprompt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"baseline_{model_slug}_{prompt_mode}_{benchmark.upper()}.jsonl"
    summary_file = RESULTS_DIR / f"baseline_{model_slug}_{prompt_mode}_{benchmark.upper()}_summary.json"

    log_raw_dir = LOGS_DIR / "raw" / f"{benchmark}_{datetime.now().strftime('%Y%m%d')}"

    # --- Resume ---
    processed_ids: set[str] = set()
    existing_results: list[dict] = []
    if resume:
        processed_ids, existing_results = _load_processed_ids(results_file)
        if existing_results and results_are_legacy_schema(existing_results):
            log.error("%s holds results from the OLD judge rubric (pre-official CEFE.AI scale).", results_file)
            log.error("Delete it and re-run — merging old 0-3 scores into the new 0-4/1-7 aggregate would corrupt the summary.")
            if not dry_run:
                sys.exit(1)
            log.warning("[DRY-RUN] (a real run would abort here until the stale file is removed)")
        if processed_ids:
            log.info("Resuming: %d prompts already done, skipping.", len(processed_ids))

    remaining = [p for p in prompts if p["id"] not in processed_ids]
    log.info("%d prompts remaining to process.", len(remaining))

    # --- Dry-run exit ---
    if dry_run:
        _sp = load_scoring_prompt(benchmark)
        _release = _sp.get("release_id") or _sp.get("benchmark", {}).get("release_id")
        log.info("[DRY-RUN] Configuration OK for benchmark=%s", benchmark.upper())
        log.info("[DRY-RUN]   model        : %s", model)
        log.info("[DRY-RUN]   judge        : %s", judge_model)
        log.info("[DRY-RUN]   judge prompt : official CEFE.AI %s (configs/cefeai/)", _release)
        log.info("[DRY-RUN]   system prompt: %s", "yes (v2 deployment-behavior)" if system_prompt else "no (v1 — headline, comparable)")
        log.info("[DRY-RUN]   enable_thinking: %s", ENABLE_THINKING)
        log.info("[DRY-RUN]   temperature  : %s", TEMPERATURE)
        log.info("[DRY-RUN]   cost_limit   : $%.2f", cost_limit)
        log.info("[DRY-RUN]   output       : %s", results_file)
        log.info("[DRY-RUN]   prompts total: %d  remaining: %d", len(prompts), len(remaining))
        log.info("[DRY-RUN] No API calls made.")
        return True

    # --- Async processing ---
    cost_tracker = CostTracker(limit_usd=cost_limit)
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    api = OpenRouterClient(api_key=api_key, base_url=base_url, log_raw_dir=log_raw_dir)
    progress = ProgressBar(total=len(remaining), label=f"{benchmark.upper()} prompts")

    new_results: list[dict] = []

    async with httpx.AsyncClient() as client_http:
        tasks = [
            _process_one(
                semaphore=semaphore,
                client_http=client_http,
                api=api,
                prompt_record=p,
                benchmark=benchmark,
                model=model,
                judge_model=judge_model,
                cost_tracker=cost_tracker,
                log=log,
                log_raw_dir=log_raw_dir,
                system_prompt=system_prompt,
            )
            for p in remaining
        ]

        with results_file.open("a", encoding="utf-8") as out_fh:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    new_results.append(result)
                    out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    progress.update()
                except CostLimitExceeded as exc:
                    progress.done()
                    log.error("💸 Cost limit reached: %s", exc)
                    log.error("Checkpoint saved. Re-run with --resume to continue.")
                    break
                except httpx.HTTPStatusError as exc:
                    progress.update()
                    log.error("❌ HTTP error on a prompt (skipping): %s", exc)
                except Exception as exc:
                    progress.update()
                    log.error("❌ Unexpected error on a prompt (skipping): %s", exc)

    progress.done()

    # --- Summary (CEFE.AI-faithful aggregation from utils.cefeai) ---
    all_results = existing_results + new_results
    summary = summarize(benchmark, all_results, model)
    summary["judge_model"]        = judge_model
    summary["system_prompt_mode"] = prompt_mode
    summary["enable_thinking"]    = ENABLE_THINKING
    summary["temperature"]        = TEMPERATURE
    summary["seed"]               = SEED_OPENROUTER
    summary["run_at"]             = datetime.now(timezone.utc).isoformat()
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_console_summary(summary))

    log.info("📄 JSONL    : %s  (%d records)", results_file, len(all_results))
    log.info("📊 Summary  : %s", summary_file)
    if summary["n_parse_error"]:
        log.warning("⚠  %d judge replies failed to parse (excluded from metrics).",
                    summary["n_parse_error"])
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CEFEAI baseline on Qwen3-8B via OpenRouter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Validate config without API calls:
    python scripts/00_cefeai_baseline.py --dry-run

  Run Religious Representation benchmark:
    python scripts/00_cefeai_baseline.py --benchmark rr

  Resume a previously interrupted run:
    python scripts/00_cefeai_baseline.py --benchmark rr --resume

  Override model from .env:
    python scripts/00_cefeai_baseline.py --model qwen/qwen3-8b --benchmark rr
""",
    )
    parser.add_argument(
        "--benchmark",
        choices=["rr", "cb", "both"],
        default="both",
        help="CEFEAI benchmark to run: rr (Religious Representation, 150 prompts), "
             "cb (Conversion Bias, 1456 pairs), or both (default).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OPENROUTER_MODEL_BASELINE from .env.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and check files without making any API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing checkpoint (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing results and start fresh.",
    )
    parser.add_argument(
        "--system-prompt",
        dest="use_system_prompt",
        action="store_true",
        default=False,
        help="Run the v2 deployment-behavior protocol (WITH the Reformed system "
             "prompt). NOTE: not CEFEAI-comparable — the prompt alone saturates the "
             "metric (raw model scored RR 99.3%% / CB 87.8%%). Default (no flag) is "
             "v1: NO system prompt — the headline, leaderboard-comparable baseline.",
    )
    return parser.parse_args()


def main() -> None:
    # Force UTF-8 on Windows so emoji render correctly in PowerShell (pastor-ai pattern)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv(PROJECT_ROOT / ".env")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = get_logger("00_cefeai_baseline")

    args = _parse_args()

    try:
        api_key    = _get_env("OPENROUTER_API_KEY")
        base_url   = _get_env("OPENROUTER_BASE_URL")
        model      = args.model or _get_env("OPENROUTER_MODEL_BASELINE")
        judge      = _get_env("OPENROUTER_MODEL_JUDGE")
        cost_limit = float(os.getenv("COST_LIMIT_USD_PHASE0", "5.00"))
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    benchmarks = ["rr", "cb"] if args.benchmark == "both" else [args.benchmark]

    # Headline protocol = v1 (NO system prompt) — CEFEAI-comparable. Opt into the
    # v2 deployment-behavior protocol with --system-prompt (NOT comparable: the
    # Reformed prompt alone saturated the raw model to RR 99.3% / CB 87.8%).
    system_prompt = None
    if args.use_system_prompt:
        try:
            system_prompt = load_system_prompt()
        except FileNotFoundError as exc:
            log.error("%s", exc)
            log.error("Drop --system-prompt to run the v1 (headline) baseline instead.")
            sys.exit(1)

    W = 64
    print("=" * W)
    print("  OpenScriptura — CEFEAI Baseline")
    print("=" * W)
    print(f"  Model       : {model}")
    print(f"  Judge       : {judge}")
    print(f"  Benchmarks  : {', '.join(b.upper() for b in benchmarks)}")
    print(f"  System prompt: {'yes (v2 deployment-behavior — NOT leaderboard-comparable)' if system_prompt else 'no (v1 — headline, CEFEAI-comparable)'}")
    print(f"  Cost limit  : ${cost_limit:.2f}")
    if args.dry_run:
        print("  ⚠️  DRY-RUN — no API calls will be made")
    if system_prompt:
        print("  ⚠️  v2 mode: the system prompt alone saturates the metric — use this")
        print("      only as a deployment-behavior datapoint, not vs the CEFEAI leaderboard.")
    print("=" * W)
    print()

    for bm in benchmarks:
        log.info("--- Starting benchmark: %s ---", bm.upper())
        asyncio.run(
            run_benchmark(
                benchmark=bm,
                dry_run=args.dry_run,
                resume=args.resume,
                model=model,
                judge_model=judge,
                base_url=base_url,
                api_key=api_key,
                cost_limit=cost_limit,
                log=log,
                system_prompt=system_prompt,
            )
        )

    print("=" * 64)
    print("  ✅ Baseline run complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
