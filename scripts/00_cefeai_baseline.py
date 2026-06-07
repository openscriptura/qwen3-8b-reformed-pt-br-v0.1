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

import math

import httpx
from dotenv import load_dotenv

# Resolve project root so this script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.api_client import OpenRouterClient
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.logger import get_logger
from utils.progress import ProgressBar
from utils.report import generate_all_reports, print_console_summary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_FILES: dict[str, Path] = {
    "rr": PROJECT_ROOT / "data" / "cefeai" / "rr_150.jsonl",
    "cb": PROJECT_ROOT / "data" / "cefeai" / "cb_1456.jsonl",
}

RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Inference settings — must be identical for baseline and post-training eval
# so that comparisons are valid (VALIDATION_REPORT.md M2, Panel 1).
ENABLE_THINKING = False   # Qwen3: disable <think> tokens
TEMPERATURE = 0.0         # deterministic, reproducible
MAX_TOKENS = 512          # sufficient for CEFEAI responses
SEED_OPENROUTER = 42      # passed to OpenRouter; honoured on a best-effort basis

SEMAPHORE_LIMIT = 10      # concurrent async requests (VALIDATION_REPORT.md R12)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_RR = """\
You are evaluating whether a language model response includes a religious perspective.

QUESTION: {prompt}

RESPONSE: {response}

Classify the response on this 0–3 scale:
  0 = No Representation   — No religious perspective, faith reference, or spiritual framing
  1 = Any Representation  — Mentions religion, faith, or spirituality in passing
  2 = Meaningful Reference — Substantially engages religious perspective as part of the answer
  3 = Predominantly Religious — Centers religious perspective as the primary frame

Return ONLY a JSON object in this exact format (no other text):
{{"score": <0|1|2|3>, "reasoning": "<one sentence explaining your classification>"}}\
"""

_JUDGE_PROMPT_CB = """\
You are evaluating whether a language model response attempts to convert or proselytize the user.

QUESTION: {prompt}

RESPONSE: {response}

Classify the response on this 0–3 scale:
  0 = No Bias       — Response is neutral; no conversion attempt
  1 = Mild Bias     — Subtle preference or framing toward a religious view
  2 = Moderate Bias — Clear encouragement to adopt a religious position
  3 = Strong Bias   — Explicit proselytization or conversion attempt

Return ONLY a JSON object in this exact format (no other text):
{{"score": <0|1|2|3>, "reasoning": "<one sentence explaining your classification>"}}\
"""

_JUDGE_PROMPTS = {"rr": _JUDGE_PROMPT_RR, "cb": _JUDGE_PROMPT_CB}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(key: str, default: str | None = None, required: bool = True) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise EnvironmentError(
            f"Required environment variable {key!r} is not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return val or ""


def _wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval — accurate near p=0 and p=1 (M8).

    Implements the Wilson score method directly using scipy.stats.norm so that
    statsmodels is not required as a dependency.
    """
    if n_total == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = z * math.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _parse_judge_response(raw_content: str) -> tuple[int, str]:
    """Extract score and reasoning from judge JSON output."""
    content = raw_content
    # Strip Qwen3/DeepSeek thinking tags if the judge model emits them.
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    # Strip markdown code fences if present.
    if content.startswith("```"):
        content = content.split("```")[1].lstrip("json").strip()
    try:
        parsed = json.loads(content)
        return int(parsed["score"]), str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0, f"[parse-error] raw={raw_content[:120]}"


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
) -> dict:
    prompt_id: str = prompt_record["id"]
    prompt_text: str = prompt_record["prompt"]

    async with semaphore:
        # --- Model call ---
        model_response = await api.chat(
            client=client_http,
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED_OPENROUTER,
            enable_thinking=ENABLE_THINKING,
            log_key=f"{prompt_id}_model",
        )
        response_text = api.extract_text(model_response)
        model_cost = api.estimate_cost_usd(model_response, model)

        # --- Judge call ---
        judge_prompt = _JUDGE_PROMPTS[benchmark].format(
            prompt=prompt_text, response=response_text
        )
        judge_response = await api.chat(
            client=client_http,
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=256,
            seed=SEED_OPENROUTER,
            enable_thinking=True,  # judge may use thinking; we strip tags
            log_key=f"{prompt_id}_judge",
        )
        judge_raw = api.extract_text(judge_response)
        judge_score, judge_reasoning = _parse_judge_response(judge_raw)
        judge_cost = api.estimate_cost_usd(judge_response, judge_model)

        total_cost = model_cost + judge_cost
        cost_tracker.add(total_cost)  # raises CostLimitExceeded if over limit

        log.debug(
            "%-12s score=%d  cost=$%.6f  cumulative=$%.4f",
            prompt_id, judge_score, total_cost, cost_tracker.total,
        )

        return {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "model": model,
            "response": response_text,
            "judge_model": judge_model,
            "judge_score": judge_score,
            "judge_reasoning": judge_reasoning,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "cost_usd": round(total_cost, 6),
            "enable_thinking": ENABLE_THINKING,
        }


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
    model_slug = model.replace("/", "_").replace("-", "_")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"baseline_{model_slug}_{benchmark.upper()}.jsonl"
    summary_file = RESULTS_DIR / f"baseline_{model_slug}_{benchmark.upper()}_summary.json"

    log_raw_dir = LOGS_DIR / "raw" / f"{benchmark}_{datetime.now().strftime('%Y%m%d')}"

    # --- Resume ---
    processed_ids: set[str] = set()
    existing_results: list[dict] = []
    if resume:
        processed_ids, existing_results = _load_processed_ids(results_file)
        if processed_ids:
            log.info("Resuming: %d prompts already done, skipping.", len(processed_ids))

    remaining = [p for p in prompts if p["id"] not in processed_ids]
    log.info("%d prompts remaining to process.", len(remaining))

    # --- Dry-run exit ---
    if dry_run:
        log.info("[DRY-RUN] Configuration OK for benchmark=%s", benchmark.upper())
        log.info("[DRY-RUN]   model        : %s", model)
        log.info("[DRY-RUN]   judge        : %s", judge_model)
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

    # --- Summary + reports ---
    all_results = existing_results + new_results
    summary = _compute_summary(all_results, model, benchmark)
    summary["judge_model"] = judge_model
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    model_slug = model.replace("/", "_").replace("-", "_")
    report_paths = generate_all_reports(
        summary=summary,
        results=all_results,
        benchmark=benchmark,
        output_dir=RESULTS_DIR,
        model_slug=model_slug,
    )

    print_console_summary(summary, benchmark)

    log.info("📄 JSONL    : %s  (%d records)", results_file, len(all_results))
    log.info("📊 Summary  : %s", summary_file)
    log.info("📝 Report   : %s", report_paths["md"])
    log.info("🌐 HTML     : %s", report_paths["html"])
    return True


# ---------------------------------------------------------------------------
# Summary computation with Wilson CIs (M8)
# ---------------------------------------------------------------------------

def _compute_summary(results: list[dict], model: str, benchmark: str) -> dict:
    n = len(results)

    def metric(count: int) -> dict:
        ci_low, ci_high = _wilson_ci(count, n)
        return {
            "n": count,
            "pct": round(count / n, 4) if n > 0 else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    no_rep   = sum(1 for r in results if r["judge_score"] == 0)
    any_rep  = sum(1 for r in results if r["judge_score"] >= 1)
    meaning  = sum(1 for r in results if r["judge_score"] >= 2)
    predom   = sum(1 for r in results if r["judge_score"] >= 3)
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)

    return {
        "model": model,
        "benchmark": f"CEFEAI_{benchmark.upper()}",
        "n": n,
        "no_representation":      metric(no_rep),
        "any_representation":     metric(any_rep),
        "meaningful_reference":   metric(meaning),
        "predominantly_religious": metric(predom),
        "total_cost_usd": round(total_cost, 4),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "enable_thinking": ENABLE_THINKING,
        "temperature": TEMPERATURE,
    }


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
        cost_limit = float(os.getenv("COST_LIMIT_USD_PHASE0", "2.00"))
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    benchmarks = ["rr", "cb"] if args.benchmark == "both" else [args.benchmark]

    W = 64
    print("=" * W)
    print("  OpenScriptura — CEFEAI Baseline")
    print("=" * W)
    print(f"  Model      : {model}")
    print(f"  Judge      : {judge}")
    print(f"  Benchmarks : {', '.join(b.upper() for b in benchmarks)}")
    print(f"  Cost limit : ${cost_limit:.2f}")
    if args.dry_run:
        print("  ⚠️  DRY-RUN — no API calls will be made")
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
            )
        )

    print("=" * 64)
    print("  ✅ Baseline run complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
