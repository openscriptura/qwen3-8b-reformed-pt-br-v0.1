"""
04_experiment.py — QLoRA fine-tuning for OpenScriptura 2×2 experiment matrix.

Trains Qwen3-8B with QLoRA (4-bit NF4) using a YAML config from configs/.
Designed to run all four experiments (A–D) on RTX 4090 (24GB) and produce
comparable eval_loss curves for config selection.

Usage:
  python scripts/04_experiment.py --config configs/exp_a.yaml
  python scripts/04_experiment.py --config configs/exp_d.yaml --dry-run
  python scripts/04_experiment.py --config configs/exp_d.yaml --resume

The winning config (expected: D — r=64, lr=1e-4) feeds into 05_train_final.py.

CEFEAI comparability note:
  Training settings here (LoRA rank, LR, batch size) are orthogonal to the
  CEFEAI evaluation protocol. The inference settings that govern comparability
  (temperature=0.0, seed=42, enable_thinking=False) are stored in the YAML
  under training.generation and are applied exclusively by 00_cefeai_baseline.py
  and 07_cefeai_eval.py — NOT during training.

Environment:
  No API keys needed — runs entirely locally on GPU.
  CUDA + bitsandbytes + flash-attn must be installed (see requirements.txt).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap (must come before heavy imports)
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml

from utils.logger import get_logger

log = get_logger("04_experiment")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["training"]["output_dir"] = str(PROJECT_ROOT / cfg["training"]["output_dir"])
    cfg["data"]["train_file"]     = str(PROJECT_ROOT / cfg["data"]["train_file"])
    cfg["data"]["eval_file"]      = str(PROJECT_ROOT / cfg["data"]["eval_file"])
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_datasets(
    train_records: list[dict],
    eval_records: list[dict],
    tokenizer,
    max_seq_length: int,
):
    """Build HuggingFace DatasetDicts.

    - Applies Qwen3 chat template to each record's 'messages' field,
      storing the result in a 'text' column (SFTTrainer uses dataset_text_field='text').
    - Logs and drops records whose tokenized length exceeds max_seq_length
      to avoid silent mid-answer truncation.
    - Returns a per-tier eval dict so Trainer logs 'eval_B_loss', 'eval_C_loss'
      separately — enabling fine-grained config comparison (Panel 4 fix).
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        log.error("datasets not installed: %s", exc)
        sys.exit(1)

    def _apply_template(records: list[dict], label: str) -> list[dict]:
        out, skipped = [], 0
        for r in records:
            text = tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer(text, return_length=True, add_special_tokens=False)["length"][0]
            if tokens > max_seq_length:
                log.warning(
                    "  [%s] Dropping overlong record (%d tokens > %d): source=%s",
                    label, tokens, max_seq_length, r.get("source", "?"),
                )
                skipped += 1
                continue
            out.append({"text": text, "tier": r.get("tier", "?")})
        if skipped:
            log.warning("  [%s] Dropped %d / %d overlong records", label, skipped, len(records))
        log.info("  [%s] %d records after length filter", label, len(out))
        return out

    log.info("Applying chat template + length filter...")
    train_rows = _apply_template(train_records, "train")
    eval_rows  = _apply_template(eval_records,  "eval")

    train_ds = Dataset.from_list(train_rows)

    # Per-tier eval datasets → Trainer logs eval_A_loss / eval_B_loss / eval_C_loss
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in eval_rows:
        by_tier[row["tier"]].append(row)

    eval_ds: dict[str, Dataset] = {
        tier: Dataset.from_list(rows)
        for tier, rows in sorted(by_tier.items())
    }
    # Also keep a combined eval key for load_best_model_at_end
    eval_ds["all"] = Dataset.from_list(eval_rows)

    log.info("  Train: %d  |  Eval: %d  (by tier: %s)",
             len(train_ds),
             len(eval_rows),
             {t: len(r) for t, r in by_tier.items()})

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def build_bnb_config(cfg: dict):
    import torch
    from transformers import BitsAndBytesConfig

    q = cfg["quantization"]
    compute_dtype = torch.bfloat16 if q["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def build_lora_config(cfg: dict):
    from peft import LoraConfig, TaskType

    l = cfg["lora"]
    return LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=l["target_modules"],
    )


# ---------------------------------------------------------------------------
# SFTConfig — built directly from YAML (no intermediate TrainingArguments)
# Fix #1: avoids brittle dict-introspection copy that silently dropped
#         renamed fields in TRL ≥0.12.
# ---------------------------------------------------------------------------

def build_sft_config(cfg: dict, resume: bool):
    from trl import SFTConfig

    t = cfg["training"]
    return SFTConfig(
        # ── paths ──────────────────────────────────────────────────────────
        output_dir=t["output_dir"],
        # ── training loop ──────────────────────────────────────────────────
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        seed=t["seed"],
        # ── sequence ───────────────────────────────────────────────────────
        max_seq_length=t["max_seq_length"],
        dataset_text_field="text",      # records pre-formatted by build_hf_datasets
        packing=False,                  # keep records separate — theological precision
        # ── checkpointing ──────────────────────────────────────────────────
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        # Use the combined 'all' eval split for best-model tracking
        eval_on_start=False,
        greater_is_better=False,        # lower eval_loss is better
        # ── precision + reporting ───────────────────────────────────────────
        bf16=True,
        report_to="none",               # no wandb/tensorboard by default
        dataloader_pin_memory=False,    # avoid issues on some Windows setups
        group_by_length=True,           # pack similar-length seqs → less padding
        resume_from_checkpoint=resume,
    )


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------

def dry_run(cfg: dict) -> None:
    exp_id   = cfg["experiment"]["id"]
    exp_desc = cfg["experiment"]["description"]
    t        = cfg["training"]
    l        = cfg["lora"]
    d        = cfg["data"]

    train_path = Path(d["train_file"])
    eval_path  = Path(d["eval_file"])

    log.info("=" * 60)
    log.info("  [DRY-RUN] Experiment : %s", exp_id)
    log.info("  %s", exp_desc)
    log.info("=" * 60)
    log.info("  Model        : %s", cfg["model"]["name"])
    log.info("  LoRA         : r=%d  α=%d  dropout=%.2f", l["r"], l["lora_alpha"], l["lora_dropout"])
    log.info("  LR           : %.1e  scheduler=%s", t["learning_rate"], t["lr_scheduler_type"])
    log.info("  Epochs       : %d", t["num_train_epochs"])
    log.info("  Eff. batch   : %d device × %d grad_accum = %d",
             t["per_device_train_batch_size"], t["gradient_accumulation_steps"],
             t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
    log.info("  Max seq len  : %d tokens", t["max_seq_length"])
    log.info("  Output dir   : %s", t["output_dir"])

    # CEFEAI comparability lock — just display, never used during training
    g = t.get("generation", {})
    log.info("  [CEFEAI] Inference settings (eval only — NOT used during training):")
    log.info("           temperature=%s  seed=%s  enable_thinking=%s",
             g.get("temperature"), g.get("seed"), g.get("enable_thinking"))

    for label, path in [("train", train_path), ("eval", eval_path)]:
        if not path.exists():
            log.error("  [MISSING] %s file: %s", label, path)
            log.error("  Run: python scripts/merge_dataset.py")
            sys.exit(1)
        n = sum(1 for _ in open(path, encoding="utf-8"))
        log.info("  %-6s file  : %s  (%d records)", label, path.name, n)

    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("  GPU          : %s  (%.1f GB VRAM)", gpu, vram)
        else:
            log.warning("  GPU          : CUDA not available — training will be very slow on CPU")
    except ImportError:
        log.warning("  torch        : not installed — cannot check GPU")

    _check_imports(dry_run=True)

    log.info("")
    log.info("[DRY-RUN] All checks passed. No training started.")


def _check_imports(dry_run: bool = False) -> None:
    missing = []
    for lib in ["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        msg = f"Missing libraries: {', '.join(missing)}"
        if dry_run:
            log.warning("  Libraries    : ⚠  %s", msg)
        else:
            log.error(msg)
            log.error("Run: pip install -r requirements.txt --break-system-packages")
            sys.exit(1)
    else:
        if dry_run:
            log.info("  Libraries    : torch / transformers / peft / trl / bitsandbytes ✓")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg: dict, resume: bool) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_id = cfg["experiment"]["id"]
    log.info("=" * 60)
    log.info("  Starting experiment : %s", exp_id)
    log.info("  %s", cfg["experiment"]["description"])
    log.info("=" * 60)

    # --- Tokenizer first (needed for length filter in build_hf_datasets) ---
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # required for causal LM training

    # --- Data ---
    log.info("Loading data...")
    train_records = load_jsonl(cfg["data"]["train_file"])
    eval_records  = load_jsonl(cfg["data"]["eval_file"])
    log.info("  Train raw: %d records  |  Eval raw: %d records",
             len(train_records), len(eval_records))

    train_ds, eval_ds = build_hf_datasets(
        train_records, eval_records,
        tokenizer=tokenizer,
        max_seq_length=cfg["training"]["max_seq_length"],
    )

    # --- Model ---
    log.info("Loading model %s (4-bit NF4)...", cfg["model"]["name"])
    bnb_config = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "eager"),
        trust_remote_code=True,
    )
    model.config.use_cache = False      # required for gradient checkpointing

    # --- LoRA ---
    # Fix #2: do NOT call model.enable_input_require_grads() manually before
    # prepare_model_for_kbit_training — it calls it internally, and calling it
    # twice installs duplicate hooks that can corrupt gradients on some PEFT versions.
    log.info("Applying LoRA (r=%d, α=%d)...", cfg["lora"]["r"], cfg["lora"]["lora_alpha"])
    from peft import get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )  # ← this calls enable_input_require_grads() internally
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Trainer ---
    log.info("Building SFTTrainer...")
    from trl import SFTTrainer

    # Fix #3: SFTConfig built directly from YAML — no TrainingArguments middleman.
    # Fix #4: eval_dataset is a dict → Trainer logs eval_B_loss, eval_C_loss, eval_all_loss.
    sft_config = build_sft_config(cfg, resume)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,   # dict → per-tier + combined logging
        processing_class=tokenizer,
    )

    # --- Train ---
    output_dir  = Path(cfg["training"]["output_dir"])
    checkpoint  = None
    if resume and output_dir.exists():
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            checkpoint = str(checkpoints[-1])
            log.info("  Resuming from: %s", checkpoint)

    log.info("Training started...")
    trainer.train(resume_from_checkpoint=checkpoint)

    # --- Save ---
    final_dir = output_dir / "final"
    log.info("Saving adapter to %s...", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # --- Results summary ---
    _write_results(cfg, trainer)
    log.info("Experiment %s complete.", exp_id)


# ---------------------------------------------------------------------------
# Results — per-tier eval_loss breakdown (Panel 4 fix)
# ---------------------------------------------------------------------------

def _write_results(cfg: dict, trainer) -> None:
    output_dir  = Path(cfg["training"]["output_dir"])
    log_history = trainer.state.log_history

    # Best eval_loss on combined split (used for model selection)
    eval_all_entries = [e for e in log_history if "eval_all_loss" in e]
    best = min(eval_all_entries, key=lambda e: e["eval_all_loss"]) if eval_all_entries else {}

    # Per-tier best eval_loss (for diagnostic comparison)
    per_tier: dict[str, float | None] = {}
    for tier in ("A", "B", "C"):
        key = f"eval_{tier}_loss"
        tier_entries = [e for e in log_history if key in e]
        per_tier[tier] = min(e[key] for e in tier_entries) if tier_entries else None

    summary = {
        "experiment_id":        cfg["experiment"]["id"],
        "description":          cfg["experiment"]["description"],
        "lora_r":               cfg["lora"]["r"],
        "lora_alpha":           cfg["lora"]["lora_alpha"],
        "learning_rate":        cfg["training"]["learning_rate"],
        "num_epochs":           cfg["training"]["num_train_epochs"],
        # Best combined eval_loss (primary selection metric)
        "best_eval_loss":       best.get("eval_all_loss"),
        "best_eval_step":       best.get("step"),
        # Per-tier breakdown — catechism vs Spurgeon/Monergismo
        "best_eval_loss_by_tier": per_tier,
        # CEFEAI inference settings — stored for traceability, NOT used here
        "cefeai_inference_settings": cfg["training"].get("generation", {}),
        "log_history":          log_history,
    }

    out = output_dir / "results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("  Results written to %s", out)
    log.info("  Best eval_loss (all)  : %.4f at step %s",
             best.get("eval_all_loss", float("nan")), best.get("step", "?"))
    if per_tier.get("B") is not None:
        log.info("  Best eval_loss Tier B : %.4f", per_tier["B"])
    if per_tier.get("C") is not None:
        log.info("  Best eval_loss Tier C : %.4f", per_tier["C"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for OpenScriptura (single experiment config)."
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to experiment YAML config (e.g. configs/exp_d.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and data without starting training",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in the output_dir",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    log.info("Config: %s — %s", config_path.name, cfg["experiment"]["description"])

    if args.dry_run:
        dry_run(cfg)
    else:
        _check_imports(dry_run=False)
        run_training(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
