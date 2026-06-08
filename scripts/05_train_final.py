"""
05_train_final.py — Phase 3 final QLoRA fine-tuning for OpenScriptura.

Trains Qwen3-8B with full bf16 LoRA (no 4-bit quantization) using the winning
config from Phase 2 (exp_c: r=64, lr=2e-4). Designed for A100 80GB on vast.ai.

Key differences from 04_experiment.py:
  - Full bf16 precision (no BitsAndBytes quantization) — cleaner adapter merge
  - flash_attention_2 support (install: pip install flash-attn --no-build-isolation)
  - Early stopping via a custom PEFT-safe callback (patience=5 evals)
  - eval_steps=25 for finer granularity (Phase 2 best was at step 350/537)
  - Larger batch (bs=4) + shorter grad_accum (4) — same effective batch=16
  - Saves merged model in addition to adapter for 06_export.py

Usage:
  python scripts/05_train_final.py --config configs/final.yaml --dry-run
  python scripts/05_train_final.py --config configs/final.yaml
  python scripts/05_train_final.py --config configs/final.yaml --resume

CEFEAI comparability note:
  Training settings here are orthogonal to CEFEAI evaluation protocol.
  Inference settings (temperature=0.0, seed=42, enable_thinking=False) are
  applied exclusively in 07_cefeai_eval.py — NOT during training.

Environment:
  No API keys needed — runs entirely locally on GPU.
  pip install flash-attn --no-build-isolation   # optional, ~10 min build
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml

from utils.logger import get_logger

log = get_logger("05_train_final")


# ---------------------------------------------------------------------------
# Early stopping — custom callback (PEFT-safe)
# ---------------------------------------------------------------------------
# The stock transformers EarlyStoppingCallback asserts load_best_model_at_end
# in on_train_begin():
#     assert args.load_best_model_at_end, "EarlyStoppingCallback requires ..."
# We deliberately keep load_best_model_at_end=False (reloading a PEFT checkpoint
# at end of training is unsafe — Lesson #9). So the stock callback would raise
# AssertionError at the very start of training. This drop-in tracks the metric
# itself and sets control.should_training_stop — no assertion, no best-model
# reload required.


def _make_early_stopping_callback(metric_name: str, patience: int, min_delta: float = 0.0):
    from transformers import TrainerCallback

    class LossEarlyStoppingCallback(TrainerCallback):
        def __init__(self):
            self.best = None
            self.waited = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics or metric_name not in metrics:
                # Metric absent on this eval pass — do nothing (don't penalise).
                return control
            value = metrics[metric_name]
            # Lower loss is better.
            if self.best is None or value < self.best - min_delta:
                self.best = value
                self.waited = 0
            else:
                self.waited += 1
                if self.waited >= patience:
                    log.info(
                        "Early stopping: %s did not improve for %d evals (best=%.4f). Stopping.",
                        metric_name, patience, self.best,
                    )
                    control.should_training_stop = True
            return control

    return LossEarlyStoppingCallback()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["training"]["output_dir"] = str(PROJECT_ROOT / cfg["training"]["output_dir"])
    cfg["data"]["train_file"]     = str(PROJECT_ROOT / cfg["data"]["train_file"])
    cfg["data"]["eval_file"]      = str(PROJECT_ROOT / cfg["data"]["eval_file"])
    cfg["export"]["merged_dir"]   = str(PROJECT_ROOT / cfg["export"]["merged_dir"])
    cfg["export"]["gguf_dir"]     = str(PROJECT_ROOT / cfg["export"]["gguf_dir"])
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_datasets(train_records, eval_records, tokenizer, max_seq_length: int):
    """Apply chat template + length filter, return (train_ds, eval_ds_dict)."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        log.error("datasets not installed: %s", exc)
        sys.exit(1)

    def _apply_template(records, label):
        out, skipped = [], 0
        for r in records:
            text = tokenizer.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False
            )
            length = tokenizer(text, return_length=True, add_special_tokens=False)["length"][0]
            if length > max_seq_length:
                log.warning("[%s] Dropping overlong record (%d tokens): source=%s",
                            label, length, r.get("source", "?"))
                skipped += 1
                continue
            out.append({"text": text, "tier": r.get("tier", "?")})
        if skipped:
            log.warning("[%s] Dropped %d / %d overlong records", label, skipped, len(records))
        log.info("[%s] %d records after length filter", label, len(out))
        return out

    log.info("Applying chat template + length filter...")
    train_rows = _apply_template(train_records, "train")
    eval_rows  = _apply_template(eval_records,  "eval")

    train_ds = Dataset.from_list(train_rows)

    by_tier: dict[str, list] = defaultdict(list)
    for row in eval_rows:
        by_tier[row["tier"]].append(row)

    eval_ds = {tier: Dataset.from_list(rows) for tier, rows in sorted(by_tier.items())}
    eval_ds["all"] = Dataset.from_list(eval_rows)

    log.info("Train: %d  |  Eval: %d  (by tier: %s)",
             len(train_ds), len(eval_rows),
             {t: len(r) for t, r in by_tier.items()})
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: dict):
    """Load Qwen3-8B in full bf16 (no quantization) for A100 training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["name"]
    attn_impl  = cfg["model"].get("attn_implementation", "eager")

    # Guard: flash_attention_2 is NOT pre-installed on vast.ai PyTorch images
    # (Lesson #7). If requested but unavailable, fall back to eager instead of
    # crashing inside from_pretrained with a cryptic ImportError.
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            log.warning(
                "flash_attention_2 requested but flash_attn is not installed — "
                "falling back to 'eager'. To use flash attention, run: "
                "pip install flash-attn --no-build-isolation"
            )
            attn_impl = "eager"

    log.info("Loading tokenizer from %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_quantization = cfg["quantization"].get("enabled", False)

    # Pin the whole model to a SINGLE GPU for training. device_map="auto" would
    # model-parallel split the 8B across all visible GPUs if the instance has
    # more than one — that breaks single-process Trainer training. {"": 0} forces
    # everything onto GPU 0 (idle others are fine). On a 1-GPU box this is
    # identical to "auto". For true multi-GPU training, launch with accelerate/
    # FSDP instead — out of scope for this single-process script.
    device_map = {"": 0} if torch.cuda.is_available() else None

    if use_quantization:
        log.info("Loading model in 4-bit NF4 (quantization enabled in config)...")
        from transformers import BitsAndBytesConfig
        q = cfg["quantization"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
    else:
        log.info("Loading model in full bf16 (no quantization — A100 mode)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, cfg: dict, use_quantization: bool):
    """Apply LoRA adapter (with or without kbit prep depending on quantization)."""
    from peft import LoraConfig, TaskType, get_peft_model

    l = cfg["lora"]
    lora_config = LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=l["target_modules"],
    )

    if use_quantization:
        # prepare_model_for_kbit_training() calls enable_input_require_grads()
        # internally — needed so gradients flow through the frozen base.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Full-precision path: we skip kbit prep, so we MUST enable input grads
        # ourselves. With gradient_checkpointing=True and a frozen base model,
        # omitting this means no gradient reaches the LoRA adapters → the model
        # silently does not learn (or raises "does not require grad").
        model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# SFTConfig with early stopping
# ---------------------------------------------------------------------------

def build_sft_config(cfg: dict) -> "SFTConfig":
    from trl import SFTConfig

    t = cfg["training"]
    return SFTConfig(
        output_dir=t["output_dir"],
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
        max_seq_length=t["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=False,        # unsafe with PEFT — manual save in finally
        metric_for_best_model=t["metric_for_best_model"],
        eval_on_start=False,
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # NOTE: do NOT set label_names=[] — breaks eval loss computation (see 04_experiment.py)
        bf16=True,
        report_to="none",
        dataloader_pin_memory=False,
        group_by_length=True,
    )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(cfg: dict) -> None:
    t = cfg["training"]
    l = cfg["lora"]
    d = cfg["data"]
    q = cfg["quantization"]

    log.info("=" * 60)
    log.info("  [DRY-RUN] Phase 3 Final Training")
    log.info("  %s", cfg["experiment"]["description"])
    log.info("=" * 60)
    log.info("  Model          : %s", cfg["model"]["name"])
    log.info("  Attention      : %s", cfg["model"].get("attn_implementation", "eager"))
    log.info("  Quantization   : %s", "4-bit NF4" if q.get("enabled") else "Full bf16 (A100 mode)")
    log.info("  LoRA           : r=%d  α=%d  dropout=%.2f", l["r"], l["lora_alpha"], l["lora_dropout"])
    log.info("  LR             : %.1e  warmup=%.0f%%", t["learning_rate"], t["warmup_ratio"] * 100)
    log.info("  Epochs         : %d  (early_stopping_patience=%d)", t["num_train_epochs"], t.get("early_stopping_patience", 5))
    log.info("  Eff. batch     : %d device × %d grad_accum = %d",
             t["per_device_train_batch_size"], t["gradient_accumulation_steps"],
             t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
    log.info("  Max seq len    : %d tokens", t["max_seq_length"])
    log.info("  Eval steps     : every %d steps", t["eval_steps"])
    log.info("  Output dir     : %s", t["output_dir"])
    log.info("  Merged output  : %s", cfg["export"]["merged_dir"])

    g = t.get("generation", {})
    log.info("  [CEFEAI lock]  : temperature=%s  seed=%s  enable_thinking=%s (eval only)",
             g.get("temperature"), g.get("seed"), g.get("enable_thinking"))

    for label, path in [("train", d["train_file"]), ("eval", d["eval_file"])]:
        p = Path(path)
        if not p.exists():
            log.error("  [MISSING] %s file: %s — run: python scripts/merge_dataset.py", label, p)
            sys.exit(1)
        n = sum(1 for _ in open(p, encoding="utf-8"))
        log.info("  %-6s file    : %s  (%d records)", label, p.name, n)

    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info("  GPU            : %s  (%.1f GB VRAM)", gpu, vram)
            if vram < 40:
                log.warning("  ⚠  VRAM < 40GB — full bf16 may OOM. Set quantization.enabled: true.")
        else:
            log.warning("  GPU            : CUDA not available")
    except ImportError:
        log.warning("  torch          : not installed")

    log.info("")
    log.info("[DRY-RUN] All checks passed. No training started.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(cfg: dict, resume: bool) -> None:
    import torch

    exp_id = cfg["experiment"]["id"]
    t      = cfg["training"]
    use_q  = cfg["quantization"].get("enabled", False)

    log.info("=" * 60)
    log.info("  Phase 3 Final Training: %s", exp_id)
    log.info("  %s", cfg["experiment"]["description"])
    log.info("=" * 60)

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Data
    log.info("Loading data...")
    train_records = load_jsonl(cfg["data"]["train_file"])
    eval_records  = load_jsonl(cfg["data"]["eval_file"])
    log.info("  Train: %d  |  Eval: %d", len(train_records), len(eval_records))

    train_ds, eval_ds = build_hf_datasets(
        train_records, eval_records, tokenizer, t["max_seq_length"]
    )

    # LoRA
    log.info("Applying LoRA (r=%d, α=%d)...", cfg["lora"]["r"], cfg["lora"]["lora_alpha"])
    model = apply_lora(model, cfg, use_quantization=use_q)

    # Trainer
    log.info("Building SFTTrainer...")
    from trl import SFTTrainer

    sft_config = build_sft_config(cfg)
    patience   = t.get("early_stopping_patience", 5)

    # Custom callback (see top of file): stock EarlyStoppingCallback would crash
    # because it asserts load_best_model_at_end=True, which we keep False for
    # PEFT safety. This one tracks eval_all_loss itself and stops cleanly.
    callbacks = [
        _make_early_stopping_callback(
            metric_name=t["metric_for_best_model"],
            patience=patience,
        )
    ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Resume from checkpoint
    output_dir = Path(t["output_dir"])
    checkpoint = None
    if resume and output_dir.exists():
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            checkpoint = str(checkpoints[-1])
            log.info("Resuming from: %s", checkpoint)

    log.info("Training started...")
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except Exception as exc:
        log.error("trainer.train() raised: %s — saving partial results.", exc)
        raise
    finally:
        # Always save adapter + results regardless of training outcome
        final_dir = output_dir / "final"
        log.info("Saving adapter to %s...", final_dir)
        try:
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))
        except Exception as exc:
            log.error("save_model failed: %s", exc)
        _write_results(cfg, trainer)

    log.info("Final training complete.")
    log.info("Next step: python scripts/06_export.py --config configs/final.yaml")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _write_results(cfg: dict, trainer) -> None:
    output_dir  = Path(cfg["training"]["output_dir"])
    log_history = trainer.state.log_history

    eval_all = [e for e in log_history if "eval_all_loss" in e]
    best     = min(eval_all, key=lambda e: e["eval_all_loss"]) if eval_all else {}

    per_tier: dict[str, float | None] = {}
    for tier in ("B", "C"):
        key = f"eval_{tier}_loss"
        entries = [e for e in log_history if key in e]
        per_tier[tier] = min(e[key] for e in entries) if entries else None

    # Locate the on-disk checkpoint matching the best eval step. With early
    # stopping, training halts `patience` evals AFTER the best, so the manually
    # saved `final/` is the LAST (worse) state — NOT the best. 06_export.py reads
    # this field so it exports the genuinely best checkpoint, falling back to
    # `final/` only when the best checkpoint was evicted by save_total_limit.
    best_step = best.get("step")
    best_checkpoint = None
    if best_step is not None:
        candidate = output_dir / f"checkpoint-{best_step}"
        if candidate.exists():
            best_checkpoint = str(candidate)
        else:
            log.warning(
                "Best checkpoint (step %s) not on disk — likely evicted by "
                "save_total_limit. Increase save_total_limit so the best survives. "
                "Export will fall back to final/ (last state).",
                best_step,
            )

    summary = {
        "experiment_id":          cfg["experiment"]["id"],
        "description":            cfg["experiment"]["description"],
        "lora_r":                 cfg["lora"]["r"],
        "lora_alpha":             cfg["lora"]["lora_alpha"],
        "learning_rate":          cfg["training"]["learning_rate"],
        "num_epochs":             cfg["training"]["num_train_epochs"],
        "quantization_enabled":   cfg["quantization"].get("enabled", False),
        "best_eval_loss":         best.get("eval_all_loss"),
        "best_eval_step":         best_step,
        "best_eval_epoch":        best.get("epoch"),
        "best_eval_loss_by_tier": per_tier,
        "best_checkpoint":        best_checkpoint,   # path or None → 06_export uses this
        "cefeai_inference_settings": cfg["training"].get("generation", {}),
        "log_history":            log_history,
    }

    out = output_dir / "results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Results written to %s", out)
    best_loss = best.get("eval_all_loss")
    log.info("Best eval_all_loss : %s at step %s (epoch %.2f)",
             f"{best_loss:.4f}" if best_loss is not None else "N/A",
             best.get("step", "?"),
             best.get("epoch", 0))
    for tier, loss in per_tier.items():
        if loss is not None:
            log.info("Best Tier %s loss  : %.4f", tier, loss)
    if best_checkpoint:
        log.info("Best checkpoint    : %s  (06_export will use this)", best_checkpoint)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 final QLoRA training for OpenScriptura."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to final YAML config (configs/final.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and data without starting training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in output_dir")
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
        for lib in ["torch", "transformers", "peft", "trl", "datasets"]:
            try:
                __import__(lib)
            except ImportError:
                log.error("Missing: %s — run: pip install -r requirements.txt", lib)
                sys.exit(1)
        run_training(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
