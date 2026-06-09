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
  COST_LIMIT_USD_PHASE4 — optional, default $5.00
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
    build_judge_prompt,
    compare_summaries,
    dedup_records,
    format_console_summary,
    load_scoring_prompt,
    load_system_prompt,
    paired_comparison,
    parse_judge_score,
    results_are_legacy_schema,
    summarize,
)
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.logger import get_logger
from utils.progress import ProgressBar
from utils.report import generate_all_reports

# ---------------------------------------------------------------------------
# CEFEAI comparability lock — NEVER change these between Phase 0 and Phase 4
# ---------------------------------------------------------------------------

ENABLE_THINKING = False   # Qwen3: disable <think> tokens (M2)
TEMPERATURE     = 0.0     # greedy decoding — must match baseline
MAX_NEW_TOKENS  = 1024    # must match baseline (00); headroom to avoid truncation bias
SEED            = 42      # torch seed for reproducibility
JUDGE_MAX_TOKENS = 1024   # DeepSeek-v4 reasons on some OpenRouter providers; reasoning tokens
                          # count against max_tokens and 256 caused content=null parse errors.
                          # 1024 is safe for the chosen `flash` judge (≤842 reasoning tok); pro
                          # reasons up to 1178, which is why §1 selects flash. MUST match
                          # 00_cefeai_baseline.py JUDGE_MAX_TOKENS — comparability.
JUDGE_ENABLE_THINKING = False  # deterministic; verdict can't be truncated by a think block
SEMAPHORE_LIMIT = 10      # concurrent judge requests

# The judge prompt, parsing, aggregation, and system prompt all come from
# utils.cefeai, which loads the OFFICIAL CEFE.AI scoring_prompt.json verbatim —
# so this eval and the Phase 0 baseline score identically, and as CEFE.AI does.

BENCHMARK_BASENAME: dict[str, str] = {"rr": "rr_150", "cb": "cb_1456"}
LANG_SUFFIX: dict[str, str] = {"en": "", "ptbr": "_ptbr"}   # en=headline; ptbr=secondary track


def _benchmark_file(benchmark: str, lang: str) -> Path:
    return PROJECT_ROOT / "data" / "cefeai" / f"{BENCHMARK_BASENAME[benchmark]}{LANG_SUFFIX[lang]}.jsonl"
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
        # Official CEFE.AI judge prompt (loaded from configs/cefeai/).
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
            seed=SEED,
            enable_thinking=JUDGE_ENABLE_THINKING,
            log_key=f"{prompt_id}_judge",
        )
        judge_raw          = api.extract_text(judge_response)
        judge_score, judge_rationale = parse_judge_score(benchmark, judge_raw)
        judge_model_used   = api.actual_model(judge_response, judge_model)
        judge_cost         = api.estimate_cost_usd(judge_response, judge_model_used)
        cost_tracker.add(judge_cost)

        log.debug("%-12s score=%s  judge_cost=$%.6f  cumulative=$%.4f",
                  prompt_id, judge_score, judge_cost, cost_tracker.total)

        rec = {
            "prompt_id":       prompt_id,
            "prompt":          prompt_text,
            "model":           model_label,
            "response":        response_text,
            "judge_model":     judge_model_used,
            "judge_score":     judge_score,        # int on official scale, or None (parse error)
            "judge_rationale": judge_rationale,
            "run_at":          datetime.now(timezone.utc).isoformat(),
            "cost_usd":        round(judge_cost, 6),
            "enable_thinking": ENABLE_THINKING,
            "temperature":     TEMPERATURE,
            "seed":            SEED,
        }
        if benchmark == "cb":   # carry CB fields for by-pair/tradition/template aggregation
            rec["pair_id"] = prompt_record.get("pair_id")
            rec["template_id"] = prompt_record.get("template_id")
            rec["religion_from"] = prompt_record.get("religion_from")
            rec["religion_to"] = prompt_record.get("religion_to")
        return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processed_ids(results_file: Path) -> tuple[set[str], list[dict]]:
    """Read existing JSONL → (processed prompt_ids, deduped records).

    A prompt is "processed" ONLY if it has a valid integer judge_score, so a
    re-run re-judges parse-error prompts instead of freezing them in (review
    finding: --resume no-op). Records are deduped by prompt_id keeping the latest
    valid record so a re-judged prompt's stale None line is not double-counted.
    This loader is reused for the baseline JSONL in the comparison path, so the
    same de-dup/exclusion applies on both sides.
    """
    if not results_file.exists():
        return set(), []
    raw = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = dedup_records(raw)                 # canonical: prefer valid, else latest
    ids = {r["prompt_id"] for r in records if isinstance(r.get("judge_score"), int)}
    return ids, records


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
    lang: str = "en",
) -> None:
    benchmark_file = _benchmark_file(benchmark, lang)
    if not benchmark_file.exists():
        log.error("Benchmark file not found: %s", benchmark_file)
        if lang != "en":
            log.error("Run scripts/translate_benchmark.py first to build the pt-BR benchmark.")
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
    lang_tag    = "" if lang == "en" else f"{lang}_"
    file_stem   = f"eval_{model_slug}_{lang_tag}{prompt_mode}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"{file_stem}_{benchmark.upper()}.jsonl"
    summary_file = RESULTS_DIR / f"{file_stem}_{benchmark.upper()}_summary.json"
    log_raw_dir  = PROJECT_ROOT / "logs" / "raw" / f"eval_{benchmark}_{datetime.now().strftime('%Y%m%d')}"

    # Resume
    processed_ids, existing_results = set(), []
    if resume:
        processed_ids, existing_results = _load_processed_ids(results_file)
        if existing_results and results_are_legacy_schema(existing_results):
            log.error("%s holds results from the OLD judge rubric (pre-official CEFE.AI scale).", results_file)
            log.error("Delete it and re-run — merging old scores into the new aggregate would corrupt the summary.")
            if not dry_run:
                sys.exit(1)
            log.warning("[DRY-RUN] (a real run would abort here until the stale file is removed)")
        n_parse_pending = sum(1 for p in prompts if p["id"] not in processed_ids and
                              any(r["prompt_id"] == p["id"] for r in existing_results))
        if processed_ids:
            log.info("Resuming: %d prompts already scored.", len(processed_ids))
        if n_parse_pending:
            log.warning("%d previously-parse-error prompts will be RE-JUDGED under the current "
                        "judge settings (stale records dropped from the aggregate).", n_parse_pending)
    elif not dry_run and results_file.exists():
        # --no-resume = start fresh: truncate so re-judged prompts are not appended
        # next to stale records (double-count in summary vs paired test — finding).
        log.warning("--no-resume: discarding existing %s and starting from scratch.", results_file.name)
        results_file.unlink()

    remaining = [p for p in prompts if p["id"] not in processed_ids]
    log.info("%d prompts remaining.", len(remaining))

    if dry_run:
        _sp = load_scoring_prompt(benchmark)
        _release = _sp.get("release_id") or _sp.get("benchmark", {}).get("release_id")
        log.info("[DRY-RUN] benchmark=%s  judge=%s  output=%s",
                 benchmark.upper(), judge_model, results_file)
        log.info("[DRY-RUN] judge prompt: official CEFE.AI %s (configs/cefeai/)", _release)
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

    # --- Summary (CEFE.AI-faithful aggregation from utils.cefeai) ---
    # Dedup so a re-judged parse-error prompt is counted once; rewrite canonical.
    union_results = existing_results + new_results
    all_results = dedup_records(union_results)
    # Atomic rewrite (temp + os.replace) so a crash mid-write can't destroy the run.
    tmp_file = results_file.with_suffix(results_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as fh:
        for r in all_results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_file, results_file)
    summary = summarize(benchmark, all_results, model_label)
    summary["judge_model"]        = judge_model
    summary["total_cost_usd"]     = round(sum(r.get("cost_usd", 0.0) for r in union_results), 6)
    summary["judge_models_served"] = sorted({r.get("judge_model") for r in all_results if r.get("judge_model")})
    summary["judge_max_tokens"]   = JUDGE_MAX_TOKENS   # for the cross-run config check below (comparability)
    summary["system_prompt_mode"] = prompt_mode
    summary["lang"]               = lang                # "en" = comparable; "ptbr" = secondary track
    summary["run_label"]          = "fine-tuned" if lang == "en" else "fine-tuned (pt-BR)"
    summary["enable_thinking"]    = ENABLE_THINKING
    summary["temperature"]        = TEMPERATURE
    summary["seed"]               = SEED
    summary["run_at"]             = datetime.now(timezone.utc).isoformat()
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(format_console_summary(summary))

    log.info("📄 JSONL   : %s  (%d records)", results_file, len(all_results))
    log.info("📊 Summary : %s", summary_file)

    # Display-only reports (md/json/html) from the OFFICIAL summary. Best-effort.
    try:
        paths = generate_all_reports(
            summary, all_results, benchmark, RESULTS_DIR,
            file_stem=file_stem,
        )
        log.info("📰 HTML    : %s", paths["html"])
    except Exception as exc:                       # noqa: BLE001 — report is non-critical
        log.warning("⚠  Report generation failed (metrics are safe in the summary): %s", exc)

    if summary["n_parse_error"]:
        log.warning("⚠  %d judge replies failed to parse (excluded from metrics).", summary["n_parse_error"])

    # Compare against the baseline that MATCHES this run's language + prompt mode
    # (headline = en/noprompt). The legacy untagged fallback applies ONLY to the
    # English track — a pt-BR eval must never be paired against the English baseline.
    base_slug = "qwen_qwen3_8b"   # OPENROUTER_MODEL_BASELINE = qwen/qwen3-8b
    tagged_baseline = RESULTS_DIR / f"baseline_{base_slug}_{lang_tag}{prompt_mode}_{benchmark.upper()}_summary.json"
    legacy_baseline = (RESULTS_DIR / f"baseline_{base_slug}_{benchmark.upper()}_summary.json"
                       if lang == "en" else tagged_baseline)
    matched       = tagged_baseline.exists()
    baseline_file = tagged_baseline if matched else legacy_baseline
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
        is_legacy = (not matched) or baseline.get("scale") is None
        # Comparability guard: the paired delta is only valid if BOTH sides used
        # the SAME judge model and settings (CLAUDE.md HARD RULE #3). A different
        # requested judge model or token budget (e.g. a baseline judged at
        # max_tokens=256 vs this run at 1024) BLOCKS the paired test. A served-
        # snapshot drift (OpenRouter serving a different dated snapshot of the SAME
        # requested model between two separately-timed runs) is a softer concern —
        # it only WARNS (else the guard would over-trigger and refuse almost every
        # cross-day comparison), so the analyst can judge whether the drift matters.
        served_now = sorted({r.get("judge_model") for r in all_results if r.get("judge_model")})
        served_base = baseline.get("judge_models_served")
        judge_mismatch = (not is_legacy) and (
            baseline.get("judge_model") != judge_model
            or baseline.get("judge_max_tokens") != JUDGE_MAX_TOKENS
        )
        served_drift = (not is_legacy) and (served_base is not None) and (served_base != served_now)
        log.info("=" * 64)
        log.info("  CEFEAI %s — fine-tuned vs baseline (official metric)", benchmark.upper())
        log.info("%s", compare_summaries(benchmark, baseline, summary))
        if judge_mismatch:
            log.warning("  ⚠  Judge config differs from the baseline — paired test SKIPPED (not comparable):")
            log.warning("     baseline judge=%s max_tokens=%s  vs  this run judge=%s max_tokens=%s",
                        baseline.get("judge_model"), baseline.get("judge_max_tokens"),
                        judge_model, JUDGE_MAX_TOKENS)
            log.warning("     Re-run 00_cefeai_baseline.py under the current judge config (use --no-resume), then re-compare.")
        elif served_drift:
            log.warning("  ⚠  Same requested judge+budget, but the SERVED snapshot differs "
                        "(baseline %s vs this run %s) — paired test STILL RUNS; note the snapshot drift.",
                        served_base, served_now)
        # Paired significance test on per-prompt scores (same prompts both models),
        # but ONLY against a same-scale (official-judge), same-judge-config baseline.
        baseline_jsonl = baseline_file.with_name(baseline_file.name.replace("_summary.json", ".jsonl"))
        if not is_legacy and not judge_mismatch and baseline_jsonl.exists():
            _, baseline_records = _load_processed_ids(baseline_jsonl)
            pc = paired_comparison(benchmark, baseline_records, all_results)
            ci = pc["mean_delta_ci"]
            log.info("  paired (n=%d): mean Δ %s  95%% CI [%s, %s]",
                     pc["n_pairs"], pc["mean_delta"], ci["ci_low"], ci["ci_high"])
            log.info("  improved/worsened/tied: %d / %d / %d   Wilcoxon p=%s  (rank-biserial %s)",
                     pc["n_improved"], pc["n_worsened"], pc["n_tied"],
                     pc.get("wilcoxon_p"), pc.get("rank_biserial"))
            summary["paired_vs_baseline"] = pc
            summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if is_legacy:
            log.warning("  ⚠  Comparing against the LEGACY baseline (%s), which was computed", legacy_baseline.name)
            log.warning("     with the OLD home-grown rubric — NOT the official CEFE.AI scale.")
            log.warning("     Re-run 00_cefeai_baseline.py to regenerate the baseline with the official judge.")
        log.info("=" * 64)
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
    parser.add_argument("--lang", choices=["en", "ptbr"], default="en",
                        help="en (default) = English CEFE.AI, leaderboard-comparable HEADLINE; "
                             "ptbr = translated SECONDARY track (NOT leaderboard-comparable). "
                             "Must match the baseline's language for the paired comparison.")
    parser.add_argument("--system-prompt", dest="use_system_prompt", action="store_true", default=False,
                        help="v2 deployment-behavior eval (WITH the Reformed system prompt). "
                             "NOT CEFEAI-comparable (the prompt saturates the metric). "
                             "Default (no flag): v1 — no system prompt, the headline comparison.")
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
        cost_limit = float(os.getenv("COST_LIMIT_USD_PHASE4", "5.00"))
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    use_system_prompt = args.use_system_prompt
    # Headline protocol = v1 (NO system prompt) — CEFEAI-comparable; isolates the
    # fine-tuning effect (weights only). Opt into v2 deployment-behavior with
    # --system-prompt (NOT comparable: the prompt saturates the metric).
    system_prompt = None
    if use_system_prompt:
        try:
            system_prompt = load_system_prompt()
        except FileNotFoundError as exc:
            log.error("%s", exc)
            log.error("Drop --system-prompt to run the v1 (headline) eval instead.")
            sys.exit(1)

    print("=" * 64)
    print("  OpenScriptura — CEFEAI Phase 4 Re-evaluation")
    print("=" * 64)
    print(f"  Model path    : {model_path}")
    print(f"  System prompt : {'yes (v2 deployment-behavior — NOT comparable)' if use_system_prompt else 'no (v1 — headline, CEFEAI-comparable)'}")
    print(f"  Judge         : {judge}  (single judge — no cross-model fallback, comparability lock)")
    print(f"  Benchmarks    : {args.benchmark.upper()}")
    print(f"  Temperature   : {TEMPERATURE}  Seed: {SEED}  Thinking: {ENABLE_THINKING}")
    print(f"  Cost limit    : ${cost_limit:.2f}")
    if args.dry_run:
        print("  ⚠️  DRY-RUN — no inference or API calls")
    print("=" * 64)
    if use_system_prompt:
        print("  ⚠️  v2 mode: WITH the system prompt the metric saturates (raw baseline")
        print("      was RR 99.3% / CB 87.8%) — report this as deployment behavior only,")
        print("      NOT as a leaderboard number. The headline run is the default (no flag).")
    else:
        print("  v1 headline: no system prompt — comparable to the CEFEAI leaderboard and")
        print("  to the v1 baseline (results/baseline_qwen_qwen3_8b_*). Isolates fine-tuning.")
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
                lang=args.lang,
            )
        )

    print("=" * 64)
    print("  ✅ Phase 4 evaluation complete")
    print("=" * 64)


if __name__ == "__main__":
    main()
