"""
07_cefeai_eval.py — CEFEAI re-evaluation of the fine-tuned OpenScriptura model.

Runs the fine-tuned Qwen3-8B model (loaded locally via HuggingFace transformers)
against the CEFEAI Religious Representation (150 prompts) and/or Conversion Bias
(1,456 pairs) benchmarks. The judge calls go through OpenRouter.

CEFEAI comparability lock:
  temperature=0.0  →  greedy decoding (do_sample=False)
  seed=42          →  torch manual seed for reproducibility
  enable_thinking=False → thinking_mode disabled in Qwen3 generation config
  These three settings are IDENTICAL to 00_cefeai_baseline.py so that
  Phase 0 and Phase 4 results are directly comparable.

Pipeline per prompt:
  1. Local model inference (transformers, greedy, no thinking)
  2. Judge call via OpenRouter (same judge as Phase 0)
  3. Write result to JSONL (append, resumable)

Usage:
  python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark rr --dry-run
  python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark rr
  python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both --resume
  python scripts/07_cefeai_eval.py --model-path results/exp_c_final --benchmark rr  # test adapter-only path

Environment:
  OPENROUTER_API_KEY  — required (judge calls)
  OPENROUTER_BASE_URL — required
  OPENROUTER_MODEL_JUDGE — required (same judge as Phase 0)
  COST_LIMIT_USD_PHASE4 — optional, default $2.00
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.api_client import OpenRouterClient
from utils.cefeai import (
    JUDGE_PROMPTS,
    baseline_verdict,
    load_system_prompt,
    parse_judge_response,
    wilson_ci,
)
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.logger import get_logger
from utils.progress import ProgressBar
from utils.report import generate_all_reports, print_console_summary

# ---------------------------------------------------------------------------
# CEFEAI comparability lock — NEVER change these between Phase 0 and Phase 4
# ---------------------------------------------------------------------------

ENABLE_THINKING = False   # Qwen3: disable <think> tokens (M2)
TEMPERATURE     = 0.0     # greedy decoding — must match baseline
MAX_NEW_TOKENS  = 512     # same as baseline
SEED            = 42      # torch seed for reproducibility
SEMAPHORE_LIMIT = 10      # concurrent judge requests

# Judge prompts, parse_judge_response, wilson_ci, baseline_verdict, and the
# canonical system prompt all come from utils.cefeai so the baseline and this
# eval share one byte-identical implementation.

BENCHMARK_FILES: dict[str, Path] = {
    "rr": PROJECT_ROOT / "data" / "cefeai" / "rr_150.jsonl",
    "cb": PROJECT_ROOT / "data" / "cefeai" / "cb_1456.jsonl",
}
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Local model inference
# ---------------------------------------------------------------------------

def load_local_model(model_path: Path):
    """Load fine-tuned model from local path (merged or adapter-only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log = get_logger("07_cefeai_eval")
    log.info("Loading tokenizer from %s...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pin to a single GPU. On a multi-GPU box device_map="auto" shards the base,
    # and merge_and_unload() (adapter path) would then fail with a cross-device
    # error. {"": 0} == "auto" on a 1-GPU box.
    device_map = {"": 0} if torch.cuda.is_available() else None

    # Check if this is a PEFT adapter (has adapter_config.json) or merged model
    is_adapter = (model_path / "adapter_config.json").exists()
    if is_adapter:
        log.info("Detected PEFT adapter — loading base model + adapter...")
        adapter_cfg = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
        base_name   = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-8B")
        log.info("Base model: %s", base_name)
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(model_path))
        model = model.merge_and_unload()
    else:
        log.info("Loading merged model in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.eval()
    log.info("Model loaded (%d parameters).", sum(p.numel() for p in model.parameters()))
    return model, tokenizer


def run_local_inference(
    model,
    tokenizer,
    prompts: list[dict],
    system_prompt: str | None,
    log,
) -> dict[str, str]:
    """Run greedy inference on all prompts. Returns {prompt_id: response_text}.

    system_prompt: the system message to prepend, or None to send only the user
    turn (baseline-comparable mode — Phase 0 used no system prompt).
    """
    import torch

    torch.manual_seed(SEED)
    results: dict[str, str] = {}
    progress = ProgressBar(total=len(prompts), label="Local inference")

    for record in prompts:
        prompt_id   = record["id"]
        prompt_text = record["prompt"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        # Apply Qwen3 chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # disable <think> tokens (CEFEAI lock)
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy = temperature 0.0 equivalent
                temperature=None,         # must be None when do_sample=False
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (exclude the prompt)
        new_tokens   = outputs[0][inputs["input_ids"].shape[1]:]
        response_txt = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip any <think> block for parity with the baseline (api_client did
        # this). With enable_thinking=False none should appear, but be defensive.
        if "<think>" in response_txt and "</think>" in response_txt:
            response_txt = response_txt.split("</think>", 1)[-1].strip()
        results[prompt_id] = response_txt
        progress.update()

    progress.done()
    return results


# ---------------------------------------------------------------------------
# Async judge worker
# ---------------------------------------------------------------------------

async def _judge_one(
    semaphore: asyncio.Semaphore,
    client_http: httpx.AsyncClient,
    api: OpenRouterClient,
    prompt_record: dict,
    response_text: str,
    benchmark: str,
    model_label: str,
    judge_model: str,
    cost_tracker: CostTracker,
    log,
    log_raw_dir: Path,
) -> dict:
    prompt_id   = prompt_record["id"]
    prompt_text = prompt_record["prompt"]

    async with semaphore:
        judge_prompt = JUDGE_PROMPTS[benchmark].format(
            prompt=prompt_text, response=response_text
        )
        judge_response = await api.chat(
            client=client_http,
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=256,
            seed=SEED,
            enable_thinking=True,    # judge may use thinking; tags are stripped
            log_key=f"{prompt_id}_judge",
        )
        judge_raw      = api.extract_text(judge_response)
        judge_score, judge_reasoning = parse_judge_response(judge_raw)
        judge_cost     = api.estimate_cost_usd(judge_response, judge_model)
        cost_tracker.add(judge_cost)

        log.debug("%-12s score=%d  judge_cost=$%.6f  cumulative=$%.4f",
                  prompt_id, judge_score, judge_cost, cost_tracker.total)

        return {
            "prompt_id":       prompt_id,
            "prompt":          prompt_text,
            "model":           model_label,
            "response":        response_text,
            "judge_model":     judge_model,
            "judge_score":     judge_score,
            "judge_reasoning": judge_reasoning,
            "run_at":          datetime.now(timezone.utc).isoformat(),
            "cost_usd":        round(judge_cost, 6),
            "enable_thinking": ENABLE_THINKING,
            "temperature":     TEMPERATURE,
            "seed":            SEED,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processed_ids(results_file: Path) -> tuple[set[str], list[dict]]:
    if not results_file.exists():
        return set(), []
    ids, records = set(), []
    with results_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                ids.add(rec["prompt_id"])
                records.append(rec)
    return ids, records


def _compute_summary(results: list[dict], model_label: str, benchmark: str) -> dict:
    n = len(results)
    def metric(count):
        ci_low, ci_high = wilson_ci(count, n)
        return {"n": count, "pct": round(count / n, 4) if n > 0 else 0.0,
                "ci_low": ci_low, "ci_high": ci_high}

    no_rep  = sum(1 for r in results if r["judge_score"] == 0)
    any_rep = sum(1 for r in results if r["judge_score"] >= 1)
    meaning = sum(1 for r in results if r["judge_score"] >= 2)
    predom  = sum(1 for r in results if r["judge_score"] >= 3)
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)

    return {
        "model":                   model_label,
        "benchmark":               f"CEFEAI_{benchmark.upper()}",
        "n":                       n,
        "no_representation":       metric(no_rep),
        "any_representation":      metric(any_rep),
        "meaningful_reference":    metric(meaning),
        "predominantly_religious": metric(predom),
        "total_cost_usd":          round(total_cost, 4),
        "run_at":                  datetime.now(timezone.utc).isoformat(),
        "enable_thinking":         ENABLE_THINKING,
        "temperature":             TEMPERATURE,
        "seed":                    SEED,
    }


def _get_env(key: str) -> str:
    val = os.getenv(key, "")
    if not val:
        raise EnvironmentError(
            f"Required environment variable {key!r} not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return val


# ---------------------------------------------------------------------------
# Per-benchmark orchestration
# ---------------------------------------------------------------------------

async def run_benchmark(
    benchmark: str,
    model_path: Path,
    model,
    tokenizer,
    system_prompt: str | None,
    dry_run: bool,
    resume: bool,
    judge_model: str,
    base_url: str,
    api_key: str,
    cost_tracker: CostTracker,
    log,
) -> None:
    benchmark_file = BENCHMARK_FILES[benchmark]
    if not benchmark_file.exists():
        log.error("Benchmark file not found: %s", benchmark_file)
        log.error("Download from https://cefe.ai and place at the path above.")
        # Dry-run must stay offline-safe: validate everything else and skip this
        # benchmark instead of hard-exiting (matches 00_cefeai_baseline.py).
        if dry_run:
            log.warning("[DRY-RUN] Skipping %s — benchmark file missing "
                        "(would abort in a real run).", benchmark.upper())
            return
        sys.exit(1)

    prompts = []
    with benchmark_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    log.info("Loaded %d prompts from %s", len(prompts), benchmark_file.name)

    # Output paths — named differently from baseline to avoid collision.
    # The prompt mode is part of the filename so a with-prompt run and a strict
    # baseline-comparable (no-prompt) run never share a JSONL — critical for
    # both --resume correctness and comparability bookkeeping.
    model_slug  = model_path.name.replace("-", "_").replace("/", "_")
    prompt_mode = "sysprompt" if system_prompt else "noprompt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"eval_{model_slug}_{prompt_mode}_{benchmark.upper()}.jsonl"
    summary_file = RESULTS_DIR / f"eval_{model_slug}_{prompt_mode}_{benchmark.upper()}_summary.json"
    log_raw_dir  = PROJECT_ROOT / "logs" / "raw" / f"eval_{benchmark}_{datetime.now().strftime('%Y%m%d')}"

    # Resume
    processed_ids, existing_results = set(), []
    if resume:
        processed_ids, existing_results = _load_processed_ids(results_file)
        if processed_ids:
            log.info("Resuming: %d prompts already done.", len(processed_ids))

    remaining = [p for p in prompts if p["id"] not in processed_ids]
    log.info("%d prompts remaining.", len(remaining))

    if dry_run:
        log.info("[DRY-RUN] benchmark=%s  judge=%s  output=%s",
                 benchmark.upper(), judge_model, results_file)
        log.info("[DRY-RUN] No inference or API calls made.")
        return

    # --- Local inference ---
    log.info("Running local model inference (%d prompts)...", len(remaining))
    responses = run_local_inference(model, tokenizer, remaining, system_prompt, log)

    # --- Async judge ---
    # cost_tracker is shared across benchmarks (created once in main) so the
    # COST_LIMIT_USD_PHASE4 budget is a single global cap, not per-benchmark.
    log.info("Running judge calls via OpenRouter...")
    semaphore    = asyncio.Semaphore(SEMAPHORE_LIMIT)
    api          = OpenRouterClient(api_key=api_key, base_url=base_url, log_raw_dir=log_raw_dir)
    model_label  = str(model_path)
    new_results: list[dict] = []

    async with httpx.AsyncClient() as client_http:
        tasks = [
            _judge_one(
                semaphore=semaphore,
                client_http=client_http,
                api=api,
                prompt_record=p,
                response_text=responses[p["id"]],
                benchmark=benchmark,
                model_label=model_label,
                judge_model=judge_model,
                cost_tracker=cost_tracker,
                log=log,
                log_raw_dir=log_raw_dir,
            )
            for p in remaining
        ]

        progress = ProgressBar(total=len(remaining), label=f"{benchmark.upper()} judge")
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
                    log.error("Re-run with --resume to continue.")
                    break
                except Exception as exc:
                    progress.update()
                    log.error("❌ Error on prompt (skipping): %s", exc)

        progress.done()

    # --- Summary ---
    all_results = existing_results + new_results
    summary = _compute_summary(all_results, model_label, benchmark)
    summary["judge_model"] = judge_model
    summary["system_prompt_mode"] = prompt_mode   # self-document the protocol mode
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_paths = generate_all_reports(
        summary=summary,
        results=all_results,
        benchmark=benchmark,
        output_dir=RESULTS_DIR,
        model_slug=f"eval_{model_slug}_{prompt_mode}",
    )
    print_console_summary(summary, benchmark)

    log.info("📄 JSONL   : %s  (%d records)", results_file, len(all_results))
    log.info("📊 Summary : %s", summary_file)
    log.info("📝 Report  : %s", report_paths["md"])

    # Compare against the baseline that MATCHES this run's prompt mode, so the
    # delta is apples-to-apples (v2 sysprompt eval ↔ v2 sysprompt baseline).
    # Fall back to the legacy untagged baseline (the original v1 no-prompt run)
    # with a loud caveat if no matching baseline exists.
    base_slug = "qwen_qwen3_8b"   # OPENROUTER_MODEL_BASELINE = qwen/qwen3-8b
    tagged_baseline = RESULTS_DIR / f"baseline_{base_slug}_{prompt_mode}_{benchmark.upper()}_summary.json"
    legacy_baseline = RESULTS_DIR / f"baseline_{base_slug}_{benchmark.upper()}_summary.json"
    matched       = tagged_baseline.exists()
    baseline_file = tagged_baseline if matched else legacy_baseline
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        baseline_any = baseline.get("any_representation", {}).get("pct", 0)
        eval_any     = summary.get("any_representation", {}).get("pct", 0)
        delta        = eval_any - baseline_any
        metric_label, better, verdict = baseline_verdict(benchmark, delta)
        base_mode = baseline.get("system_prompt_mode", "noprompt (legacy)")
        log.info("=" * 50)
        log.info("  CEFEAI %s Comparison — %s (%s)", benchmark.upper(), metric_label, better)
        log.info("  Baseline  : %.1f%%  (raw Qwen3-8B, %s)", baseline_any * 100, base_mode)
        log.info("  Fine-tuned: %.1f%%  (OpenScriptura, %s)", eval_any * 100, prompt_mode)
        log.info("  Delta     : %+.1f pp  → %s", delta * 100, verdict)
        if not matched:
            log.warning("  ⚠  No %s baseline found — compared against the LEGACY baseline", prompt_mode)
            log.warning("     %s. This delta is NOT apples-to-apples; re-run", legacy_baseline.name)
            log.warning("     00_cefeai_baseline.py under the same protocol for a valid comparison.")
        log.info("=" * 50)
    else:
        log.info("No baseline summary for comparison (looked for %s).", tagged_baseline.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv(PROJECT_ROOT / ".env")
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    log = get_logger("07_cefeai_eval")

    parser = argparse.ArgumentParser(
        description="CEFEAI re-evaluation of fine-tuned OpenScriptura model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark rr --dry-run
  python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both
  python scripts/07_cefeai_eval.py --model-path results/exp_c_final --benchmark rr --resume
""",
    )
    parser.add_argument("--model-path", required=True, type=Path,
                        help="Path to merged model or PEFT adapter directory")
    parser.add_argument("--benchmark", choices=["rr", "cb", "both"], default="both",
                        help="CEFEAI benchmark to run (default: both)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Run without Reformed system prompt (baseline-comparable mode)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without inference or API calls")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing checkpoint (default: enabled)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path.is_absolute() else PROJECT_ROOT / args.model_path
    if not model_path.exists():
        log.error("Model path not found: %s", model_path)
        log.error("Run 05_train_final.py and/or 06_export.py first.")
        sys.exit(1)

    try:
        api_key    = _get_env("OPENROUTER_API_KEY")
        base_url   = _get_env("OPENROUTER_BASE_URL")
        judge      = _get_env("OPENROUTER_MODEL_JUDGE")
        cost_limit = float(os.getenv("COST_LIMIT_USD_PHASE4", "2.00"))
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    use_system_prompt = not args.no_system_prompt
    # Protocol v2: load the canonical Reformed system prompt (committed in
    # configs/system_prompt.txt) — None in baseline-comparable (v1) mode. A
    # missing prompt file is a HARD error: never evaluate under a prompt that
    # differs from training. Compare against the matching-mode baseline.
    system_prompt = None
    if use_system_prompt:
        try:
            system_prompt = load_system_prompt()
        except FileNotFoundError as exc:
            log.error("%s", exc)
            log.error("Or run with --no-system-prompt for the legacy v1 comparison.")
            sys.exit(1)

    print("=" * 64)
    print("  OpenScriptura — CEFEAI Phase 4 Re-evaluation")
    print("=" * 64)
    print(f"  Model path    : {model_path}")
    print(f"  System prompt : {'yes (v2 — Reformed PT-BR)' if use_system_prompt else 'no (v1 legacy)'}")
    print(f"  Judge         : {judge}")
    print(f"  Benchmarks    : {args.benchmark.upper()}")
    print(f"  Temperature   : {TEMPERATURE}  Seed: {SEED}  Thinking: {ENABLE_THINKING}")
    print(f"  Cost limit    : ${cost_limit:.2f}")
    if args.dry_run:
        print("  ⚠️  DRY-RUN — no inference or API calls")
    print("=" * 64)
    print("  Protocol v2: the system prompt is applied to BOTH baseline and this")
    print("  eval, so the comparison stays valid. Run 00_cefeai_baseline.py under")
    print("  the SAME mode (default v2, or --no-system-prompt for v1) to compare.")
    print("=" * 64)
    print()

    # Load model once — reuse across benchmarks
    model, tokenizer = None, None
    if not args.dry_run:
        model, tokenizer = load_local_model(model_path)

    # Single shared budget across all benchmarks in this invocation.
    cost_tracker = CostTracker(limit_usd=cost_limit)

    benchmarks = ["rr", "cb"] if args.benchmark == "both" else [args.benchmark]
    for bm in benchmarks:
        log.info("--- Starting benchmark: %s ---", bm.upper())
        asyncio.run(
            run_benchmark(
                benchmark=bm,
                model_path=model_path,
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                dry_run=args.dry_run,
                resume=args.resume,
                judge_model=judge,
                base_url=base_url,
                api_key=api_key,
                cost_tracker=cost_tracker,
                log=log,
            )
        )

    print("=" * 64)
    print("  ✅ Phase 4 evaluation complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
