"""
06_export.py — Merge LoRA adapter + export GGUF quantizations for OpenScriptura.

Pipeline:
  1. Load base Qwen3-8B in full bf16
  2. Load LoRA adapter from checkpoints/final/final/ (or --adapter-path)
  3. Merge adapter into base model via peft.merge_and_unload()
  4. Save merged model to checkpoints/final/merged/
  5. Convert to GGUF using llama.cpp convert_hf_to_gguf.py
  6. Quantize to Q4_K_M, Q5_K_M, Q8_0 using llama.cpp llama-quantize
  7. (Optional) Push to HuggingFace Hub

Usage:
  python scripts/06_export.py --config configs/final.yaml --dry-run
  python scripts/06_export.py --config configs/final.yaml
  python scripts/06_export.py --config configs/final.yaml --adapter-path results/exp_c_final
  python scripts/06_export.py --config configs/final.yaml --push-to-hub

Prerequisites on vast.ai A100:
  git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
  cd /workspace/llama.cpp && pip install -r requirements.txt
  make -j$(nproc)   # builds llama-quantize binary

Environment:
  HF_TOKEN — required for --push-to-hub (set in .env)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml
from dotenv import load_dotenv

from utils.logger import get_logger

log = get_logger("06_export")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["output_dir"] = str(PROJECT_ROOT / cfg["training"]["output_dir"])
    cfg["export"]["merged_dir"]   = str(PROJECT_ROOT / cfg["export"]["merged_dir"])
    cfg["export"]["gguf_dir"]     = str(PROJECT_ROOT / cfg["export"]["gguf_dir"])
    return cfg


# ---------------------------------------------------------------------------
# Step 1+2+3: Merge adapter → base model
# ---------------------------------------------------------------------------

def merge_adapter(cfg: dict, adapter_path: Path) -> Path:
    """Load base model + LoRA adapter, merge, save merged model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = Path(cfg["export"]["merged_dir"])
    if merged_dir.exists() and any(merged_dir.iterdir()):
        log.info("Merged model already exists at %s — skipping merge.", merged_dir)
        log.info("  Delete the directory to force re-merge.")
        return merged_dir

    model_name = cfg["model"]["name"]
    log.info("=" * 60)
    log.info("  Step 1/3: Loading base model %s in bf16...", model_name)
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    log.info("Step 2/3: Loading LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))

    log.info("Step 3/3: Merging adapter into base model...")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s...", merged_dir)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # Write model card metadata
    _write_model_card(cfg, merged_dir)

    log.info("✅ Merge complete: %s", merged_dir)
    return merged_dir


# ---------------------------------------------------------------------------
# Step 4: GGUF conversion
# ---------------------------------------------------------------------------

def convert_to_gguf(cfg: dict, merged_dir: Path, llama_cpp_dir: Path) -> Path:
    """Convert merged HF model to GGUF format using llama.cpp."""
    gguf_dir = Path(cfg["export"]["gguf_dir"])
    gguf_dir.mkdir(parents=True, exist_ok=True)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Older llama.cpp versions use different name
        convert_script = llama_cpp_dir / "convert.py"
    if not convert_script.exists():
        log.error("llama.cpp convert script not found at %s", llama_cpp_dir)
        log.error("Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp %s", llama_cpp_dir)
        sys.exit(1)

    gguf_f16 = gguf_dir / "qwen3-8b-reformed-pt-br-f16.gguf"

    if gguf_f16.exists():
        log.info("F16 GGUF already exists — skipping conversion.")
        return gguf_f16

    log.info("=" * 60)
    log.info("  Converting to GGUF (F16)...")
    log.info("=" * 60)

    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_f16),
        "--outtype", "f16",
    ]
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)   # check=False so our handling runs
    if result.returncode != 0:
        log.error("GGUF conversion failed (exit %d)", result.returncode)
        sys.exit(1)

    log.info("✅ F16 GGUF: %s  (%.1f GB)", gguf_f16, gguf_f16.stat().st_size / 1e9)
    return gguf_f16


# ---------------------------------------------------------------------------
# Step 5: Quantization
# ---------------------------------------------------------------------------

def quantize_gguf(cfg: dict, gguf_f16: Path, llama_cpp_dir: Path) -> list[Path]:
    """Quantize F16 GGUF to Q4_K_M, Q5_K_M, Q8_0."""
    gguf_dir = Path(cfg["export"]["gguf_dir"])
    quantizations = cfg["export"].get("quantizations", ["Q4_K_M", "Q5_K_M", "Q8_0"])

    # llama-quantize binary location
    quantize_bin = llama_cpp_dir / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp_dir / "quantize"  # older path
    if not quantize_bin.exists():
        log.error("llama-quantize binary not found. Build llama.cpp: cd %s && make -j$(nproc)", llama_cpp_dir)
        sys.exit(1)

    outputs = []
    for quant in quantizations:
        stem = gguf_f16.stem.replace("-f16", "")
        out_path = gguf_dir / f"{stem}-{quant.lower()}.gguf"

        if out_path.exists():
            log.info("  %s already exists — skipping.", out_path.name)
            outputs.append(out_path)
            continue

        log.info("Quantizing → %s...", quant)
        cmd = [str(quantize_bin), str(gguf_f16), str(out_path), quant]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log.error("Quantization failed for %s (exit %d)", quant, result.returncode)
            continue

        size_gb = out_path.stat().st_size / 1e9
        log.info("  ✅ %s  (%.2f GB)", out_path.name, size_gb)
        outputs.append(out_path)

    return outputs


# ---------------------------------------------------------------------------
# Step 6: HuggingFace push
# ---------------------------------------------------------------------------

def push_to_hub(cfg: dict, merged_dir: Path, gguf_paths: list[Path]) -> None:
    """Push merged model + GGUF files to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("huggingface_hub not installed: pip install huggingface_hub")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        log.error("HF_TOKEN not set in .env — required for push-to-hub")
        sys.exit(1)

    repo_id = cfg["export"]["hf_repo"]
    api = HfApi(token=hf_token)

    log.info("=" * 60)
    log.info("  Pushing to HuggingFace: %s", repo_id)
    log.info("=" * 60)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        log.info("Repo: https://huggingface.co/%s", repo_id)
    except Exception as exc:
        log.error("Failed to create repo: %s", exc)
        sys.exit(1)

    # Upload merged model folder
    log.info("Uploading merged model...")
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add merged bf16 model (Phase 3 final training)",
    )
    log.info("✅ Merged model uploaded")

    # Upload GGUF files
    for gguf_path in gguf_paths:
        log.info("Uploading %s (%.2f GB)...", gguf_path.name, gguf_path.stat().st_size / 1e9)
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=f"gguf/{gguf_path.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {gguf_path.name}",
        )
        log.info("✅ Uploaded: gguf/%s", gguf_path.name)

    log.info("🎉 Published: https://huggingface.co/%s", repo_id)


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _write_model_card(cfg: dict, merged_dir: Path) -> None:
    """Write a minimal README.md to the merged model directory."""
    repo_id = cfg["export"].get("hf_repo", "openscriptura/qwen3-8b-reformed-pt-br-v0.1")
    content = f"""---
base_model: Qwen/Qwen3-8B
language:
- pt
tags:
- theology
- reformed
- portuguese
- lora
- qwen3
license: apache-2.0
---

# {repo_id.split('/')[-1]}

Fine-tuned [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on Reformed Protestant
theological corpus in Brazilian Portuguese.

**Training:** QLoRA r=64, α=128, lr=2e-4, 2,873 records (Tier C catechisms + Tier B synthetic)

**Benchmark baseline (pre fine-tuning):**
- CEFEAI RR: 4.7% (7/150)
- CEFEAI CB: 19.6% (286/1,456)

See [OpenScriptura](https://github.com/openscriptura) for full methodology.
"""
    (merged_dir / "README.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(cfg: dict, adapter_path: Path, llama_cpp_dir: Path) -> None:
    log.info("=" * 60)
    log.info("  [DRY-RUN] Phase 3 Export")
    log.info("=" * 60)
    log.info("  Base model     : %s", cfg["model"]["name"])
    log.info("  Adapter path   : %s  (exists=%s)", adapter_path, adapter_path.exists())
    log.info("  Merged output  : %s", cfg["export"]["merged_dir"])
    log.info("  GGUF output    : %s", cfg["export"]["gguf_dir"])
    log.info("  Quantizations  : %s", cfg["export"].get("quantizations"))
    log.info("  HF repo        : %s", cfg["export"].get("hf_repo"))
    log.info("  llama.cpp dir  : %s  (exists=%s)", llama_cpp_dir, llama_cpp_dir.exists())

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin   = llama_cpp_dir / "llama-quantize"
    log.info("  convert script : %s  (exists=%s)", convert_script.name, convert_script.exists())
    log.info("  quantize bin   : %s  (exists=%s)", quantize_bin.name, quantize_bin.exists())

    if not adapter_path.exists():
        log.error("  ⚠  Adapter not found — run 05_train_final.py first")
    else:
        adapter_files = list(adapter_path.iterdir())
        log.info("  Adapter files  : %d files in %s", len(adapter_files), adapter_path.name)

    log.info("")
    log.info("[DRY-RUN] No files written.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export GGUF quantizations."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to final YAML config (configs/final.yaml)")
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="Override adapter path (default: checkpoints/final/final/)")
    parser.add_argument("--llama-cpp-dir", type=Path,
                        default=Path("/workspace/llama.cpp"),
                        help="Path to llama.cpp repo (default: /workspace/llama.cpp)")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="Skip GGUF conversion (merge only)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push merged model + GGUFs to HuggingFace Hub")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate paths without running export")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)

    # Resolve adapter path. Priority:
    #   1. --adapter-path (explicit override)
    #   2. results.json "best_checkpoint" (the genuinely best step — early
    #      stopping means final/ is the last, worse state; see 05_train_final.py)
    #   3. checkpoints/final/final/ (last state — fallback)
    if args.adapter_path:
        adapter_path = args.adapter_path
    else:
        output_dir   = Path(cfg["training"]["output_dir"])
        adapter_path = output_dir / "final"
        results_json = output_dir / "results.json"
        if results_json.exists():
            try:
                data = json.loads(results_json.read_text(encoding="utf-8"))
                best_ckpt = data.get("best_checkpoint")
                if best_ckpt and Path(best_ckpt).exists():
                    adapter_path = Path(best_ckpt)
                    log.info("Using best checkpoint from results.json: %s", adapter_path)
                else:
                    log.warning(
                        "results.json has no usable best_checkpoint — "
                        "falling back to final/ (last training state)."
                    )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read results.json (%s) — using final/.", exc)

    if args.dry_run:
        dry_run(cfg, adapter_path, args.llama_cpp_dir)
        return

    # Merge
    merged_dir = merge_adapter(cfg, adapter_path)

    gguf_paths = []
    if not args.skip_gguf:
        # Convert + quantize
        gguf_f16   = convert_to_gguf(cfg, merged_dir, args.llama_cpp_dir)
        gguf_paths = quantize_gguf(cfg, gguf_f16, args.llama_cpp_dir)

        log.info("=" * 60)
        log.info("  GGUF export summary")
        log.info("=" * 60)
        for p in gguf_paths:
            log.info("  %.2f GB  %s", p.stat().st_size / 1e9, p.name)

    if args.push_to_hub:
        push_to_hub(cfg, merged_dir, gguf_paths)

    log.info("")
    log.info("✅ Export complete.")
    log.info("   Merged  : %s", merged_dir)
    if gguf_paths:
        log.info("   GGUFs   : %s", Path(cfg["export"]["gguf_dir"]))
    log.info("Next step: python scripts/07_cefeai_eval.py --benchmark rr")


if __name__ == "__main__":
    main()
