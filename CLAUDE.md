# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenScriptura** is an applied research pipeline to fine-tune open LLMs (starting with Qwen3-8B) with Protestant theological corpus. The first release target is `qwen3-8b-reformed-pt-br-v0.1` (Reformed theology, Brazilian Portuguese), evaluated on the CEFEAI benchmark.

**Status:** Phase 0 ✅ (headline baseline RR 4.7%, CB 19.6% — **v1, no system prompt**) · Phase 1 ✅ (2,968 records: 839 C + 2,129 B) · Phase 2 ✅ (4 configs run; **winner exp_c** r=64 lr=2e-4, eval_all_loss 0.6527) · Phase 3 scripts ✅ written, run 🔲 · Phase 4 scripts ✅ written, run 🔲

> **Evaluation protocol — headline is v1 (NO system prompt).** We briefly tried "protocol v2" (put the Reformed system prompt on both sides). The v2 run proved it wrong empirically: the prompt **alone** saturated the *raw* model to **RR 99.3% / CB 87.8%**. That (a) isn't comparable to the prompt-free CEFEAI leaderboard and (b) leaves no headroom to show what fine-tuning added. So the headline/comparable protocol is **v1: no system prompt, both sides** (`scripts/00` and `07` default to this; `--system-prompt` opts into v2). v2 is retained only as an opt-in **deployment-behavior** datapoint, never a leaderboard number. See IMPLEMENTATION_PLAN.md → "Evaluation protocol — v1 vs v2" and Lessons #14–#16.

## ⛔ HARD RULE — CEFEAI comparability is NON-NEGOTIABLE

**Read this before changing anything that touches evaluation.** This is the single most important constraint in the project.

**WE MUST RESPECT COMPARABILITY WITH CEFE.AI.** The entire scientific claim — *"fine-tuning improved CEFEAI representation by N points"* — and the leaderboard comparison depend on it. **No fix, refactor, optimization, cleanup, or "improvement" may break it.** A change that is "better" code but alters how the headline CEFEAI numbers are produced is **forbidden** — full stop.

The headline CEFEAI run is comparable **only** if ALL of these hold, identically on BOTH the raw baseline and the fine-tuned model:

1. **NO system prompt.** Proven necessary: a system prompt saturates the metric (raw + Reformed prompt = RR 99.3% / CB 87.8%, Lesson #16). The headline runs with no system prompt on both sides.
2. **Locked inference:** `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`.
3. **Identical judge:** same judge model, same judge prompts, same 0–3 rubric, same Wilson CI — sourced from `scripts/utils/cefeai.py` (the single source of truth; do not fork or re-implement it).
4. **Unchanged benchmark inputs:** `data/cefeai/rr_150.jsonl`, `cb_1456.jsonl`.
5. **Only the model weights differ** between the two runs.

**If you ever think an improvement requires changing one of these: STOP.** Do not do it on the headline path. Surface it to the user. If they approve a new protocol, you MUST (a) keep producing the comparable number too, (b) re-run **both** sides under the new protocol, and (c) label the new number loudly as NOT leaderboard-comparable. (This is exactly what happened to the v2 system-prompt experiment — it was run, it broke comparability, and it was demoted to an opt-in footnote, never the headline.)

**Protected asset:** the v1 baseline (RR 4.7% / CB 19.6%, no prompt) is the reference. Never overwrite or delete it — archived at `results/v1_baseline_archive/`.

## Commands

```bash
# Install — local dev (Windows/PowerShell)
pip install -r requirements.txt --break-system-packages

# Install — vast.ai GPU instance (skip unsloth, install core only)
pip install transformers==4.51.0 trl==0.12.0 peft==0.13.0 bitsandbytes \
  datasets==3.2.0 pyyaml==6.0.2 python-dotenv==1.0.1 scipy==1.14.1 \
  sentencepiece==0.2.0 tokenizers==0.21.0 tenacity==9.0.0 httpx==0.27.0 jsonlines==4.0.0

# Phase 0: Baseline ✅ — HEADLINE = v1 (no system prompt, CEFEAI-comparable)
python scripts/00_cefeai_baseline.py --benchmark both --dry-run
python scripts/00_cefeai_baseline.py --benchmark both          # v1 headline → RR 4.7% / CB 19.6%
#   → results/baseline_qwen_qwen3_8b_noprompt_{RR,CB}_summary.json (legacy untagged files also = v1)

# (Optional) v2 deployment-behavior datapoint — NOT comparable (prompt saturates: RR 99.3% / CB 87.8%)
python scripts/00_cefeai_baseline.py --benchmark both --system-prompt

# Phase 1: Dataset construction ✅
python scripts/01_build_tier_c.py --dry-run        # Tier C: 839 records ✅
python scripts/02_build_tier_b.py --dry-run        # Tier B: 2,129 records ✅
python scripts/03_eda.py                           # EDA → reports/eda_report.html ✅
python scripts/merge_dataset.py --dry-run          # merge → data/merged/ ✅
python scripts/merge_dataset.py                    # 2,873 train + 151 eval ✅

# Phase 2: Experiments ✅ COMPLETE (ran on vast.ai RTX 4090; winner = exp_c)
python scripts/04_experiment.py --config configs/exp_c.yaml   # r=64 lr=2e-4 ← WINNER (0.6527)
python scripts/04_experiment.py --config configs/exp_d.yaml   # r=64 lr=1e-4 (0.6586)
python scripts/04_experiment.py --config configs/exp_a.yaml   # r=16 lr=2e-4
python scripts/04_experiment.py --config configs/exp_b.yaml   # r=16 lr=1e-4 (0.6993)
#   chained runs: insert `sleep 30` between configs (Lesson #11) to free the GPU

# Phase 2: vast.ai automation
pip install vastai
vastai set api-key <KEY>                           # from cloud.vast.ai/account
python scripts/vastai_run_experiments.py --search
python scripts/vastai_run_experiments.py --config configs/exp_c.yaml --all-configs --wait
python scripts/vastai_run_experiments.py --destroy <INSTANCE_ID>

# Phase 3: Final training + export ✅ written, run 🔲 (A100 80GB; winner config)
python scripts/05_train_final.py --config configs/final.yaml --dry-run
python scripts/05_train_final.py --config configs/final.yaml   # full bf16, early stopping
python scripts/06_export.py --config configs/final.yaml --dry-run
python scripts/06_export.py --config configs/final.yaml        # merge best ckpt + GGUF Q4/Q5/Q8
python scripts/06_export.py --config configs/final.yaml --push-to-hub

# Phase 4: Re-evaluation ✅ written, run 🔲 (local inference + OpenRouter judge)
python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both --dry-run
python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both   # v1 HEADLINE (no prompt)
python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both --system-prompt  # v2 datapoint only

# Tests
$env:PYTHONPATH = "scripts"; pytest tests/ -v
```

## Architecture

### 4-Phase Pipeline

```
Phase 0: CEFEAI Baseline (raw Qwen3-8B, counterfactual) ✅
    └── v1 = no system prompt → RR 4.7%, CB 19.6%
    ↓
Phase 1: Dataset Construction ✅
    ├── Tier C: Native sources — catechisms & confessions → 839 records
    ├── Tier B: Synthetic — LLM-generated + judge-filtered → 2,129 records
    └── Tier A: Manual — pastoral council review (v0.1: skipped)
    ↓ merge_dataset.py → data/merged/train.jsonl (2,873) + eval.jsonl (151)
Phase 2: Controlled Experiments (2×2 LoRA matrix, 4 configs) ✅
    └── RTX 4090 on vast.ai (~6.5h total, ~$3.50)
    └── Winner: exp_c (r=64, lr=2e-4) eval_all_loss 0.6527 @ step 350
    ↓
Phase 3: Final Fine-tuning (full-bf16 LoRA on A100) → Merge → GGUF export 🔲
    └── configs/final.yaml (exp_c winner); Quantizations: Q4_K_M, Q5_K_M, Q8_0
    ↓
Phase 4: CEFEAI Re-evaluation 🔲 — HEADLINE = v1 (no system prompt), both sides
    └── v2 (--system-prompt) ran on the baseline → RR 99.3% / CB 87.8% (prompt
        saturates the metric) → kept as a deployment datapoint, NOT the headline
    ↓
arXiv / HuggingFace publication 🔲
```

### Phase 2 results (final — all 4 configs complete)

| Config | r | LR | eval_all_loss | Tier B | Tier C | best step |
|--------|---|----|---------------|--------|--------|-----------|
| **exp_c** | 64 | 2e-4 | **0.6527** ✅ | 0.6841 | 0.5728 | 350 |
| exp_d | 64 | 1e-4 | 0.6586 | 0.6870 | 0.5858 | 350 |
| exp_a | 16 | 2e-4 | ~0.69 | 0.6924 | 0.5910 | — |
| exp_b | 16 | 1e-4 | 0.6993 | 0.6993 | 0.5990 | 450 |

Verdict: **rank dominates LR** (r=64 ≫ r=16, Δ≈0.04); within r=64 the LR effect is small (Δ=0.006). `configs/final.yaml` uses the exp_c winner. Adapters + results for all four are archived in `results/`.

### Data Schema (canonical — chat format)

All training records use this shape. `content_hash()` hashes `messages + tradition + lang`.

```json
{
  "id": "openscriptura-reformed-pt-00001",
  "version": "1.0",
  "tradition": "reformed",
  "lang": "pt-BR",
  "tier": "B",
  "source": "spurgeon_chs130",
  "sha256": "<content_hash>",
  "messages": [
    {"role": "system",    "content": "<canonical Reformed PT-BR system prompt>"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "confessional_refs": ["WCF 10.1", "Dort III/IV.6"],
  "reviewed_by": "automated",
  "quality_score": 96.0,
  "created_at": "2026-06-07T00:00:00Z"
}
```

⚠️ **Schema keys:** use `lang` (not `language`) and `messages` (not `question`/`answer`) — matches `content_hash()`.

### Utility Modules (`scripts/utils/`)

- `api_client.py` — `OpenRouterClient`: async chat, tenacity retry (5×), `enable_thinking=False`, `<think>` stripping, cost estimation. Retry covers `TimeoutException`, `HTTPStatusError`, `ConnectError`, `ReadError`.
- `cost_tracker.py` — `CostTracker`: thread-safe accumulator, raises `CostLimitExceeded` at limit.
- `logger.py` — `get_logger(name)`: stdout + `logs/YYYYMMDD_<name>.log`.
- `progress.py` — `ProgressBar`: TTY-aware Unicode/ASCII bar.
- `report.py` — full CEFEAI leaderboard (27 RR + 20 CB models), HTML/MD/JSON reports.
- `hash.py` — `content_hash()`: SHA-256 over `messages + tradition + lang`.
- `cefeai.py` — **shared CEFEAI primitives** (protocol v2): `JUDGE_PROMPTS` (RR/CB), `parse_judge_response()`, `wilson_ci()`, `baseline_verdict()` (direction-aware: RR up=better, CB down=better), `load_system_prompt()` (reads committed `configs/system_prompt.txt`, hard-fails on missing/empty). Imported by BOTH `00_cefeai_baseline.py` and `07_cefeai_eval.py` so the two sides of the comparison can never drift.

### Conventions

- Scripts numbered `NN_name.py` in `scripts/`; shared code in `scripts/utils/`.
- `sys.stdout.reconfigure(encoding="utf-8")` at top of every entry-point script (Windows/PowerShell UTF-8).
- Paths resolved from `PROJECT_ROOT = Path(__file__).resolve().parent.parent`.
- Every API-spending script: `OpenRouterClient` + `CostTracker` + `--dry-run` + `--resume` checkpoint.
- **CEFEAI comparability — headline = v1 (NO system prompt):** `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`, and **no system prompt**, applied to BOTH the Phase 0 baseline (`00_cefeai_baseline.py`) and the Phase 4 eval (`07_cefeai_eval.py`) — they default to this. The only difference between the two runs is the model weights, and the numbers stay comparable to the prompt-free CEFEAI leaderboard. `--system-prompt` opts into the **v2 deployment-behavior** mode (canonical Reformed prompt from `configs/system_prompt.txt`); it is NOT a leaderboard number — empirically the prompt alone saturated the raw model (RR 99.3% / CB 87.8%). Output files are tagged `noprompt`/`sysprompt`; `07` compares against the baseline matching its own mode (falling back to the legacy untagged v1 files). Shared judge prompts / Wilson CI / system-prompt loader live in `scripts/utils/cefeai.py`. Training-side settings (LR, LoRA rank, batch size) are orthogonal.

### GPU Strategy

| Phase | GPU | Where | Why |
|-------|-----|-------|-----|
| Phase 2 experiments | RTX 4090 24GB | vast.ai ~$0.45/hr | Cheaper, fits QLoRA r=64 |
| Phase 3 final train | A100 80GB | vast.ai ~$1.80/hr | Needed for full merge + GGUF export |

## Lessons Learned (vast.ai / GPU setup)

These issues were encountered during Phase 2 setup on vast.ai. Document here to avoid repeating.

### 1. `unsloth` version pinning
**Problem:** `requirements.txt` had `unsloth==2025.3.0` which was yanked by the unsloth team.  
**Fix:** Updated to `unsloth==2025.3.19`. Check https://pypi.org/project/unsloth/#history for latest non-yanked version.  
**Note:** Unsloth is only needed for Phase 3 (faster final training). For Phase 2 experiments use the core install command above — it's faster and avoids unsloth dependency hell.

### 2. bitsandbytes CUDA version mismatch on vast.ai
**Problem:** vast.ai RTX 4090 instances run host CUDA driver 13.x but the `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` container has CUDA 12.4 toolkit. `bitsandbytes==0.44.1` tried to load `libbitsandbytes_cuda130.so` (not in the image). Upgrading bitsandbytes then required `libnvJitLink.so.13` (also missing from CUDA 12.4).  
**Fix:**
```bash
# Create symlink so bitsandbytes finds CUDA 12.4 lib as CUDA 13
ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvJitLink.so.12 \
       /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvJitLink.so.13
echo "/usr/local/cuda-12.4/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda124.conf
ldconfig
# Must also set LD_LIBRARY_PATH (ldconfig alone doesn't fix ctypes)
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```
**Permanent fix for future instances:** add the `export LD_LIBRARY_PATH` line to `~/.bashrc` on the instance.

### 3. `data/` not in repo — must upload manually
**Problem:** `data/` is gitignored. On a fresh vast.ai instance, `merge_dataset.py` fails with "No records loaded".  
**Fix:** Upload from local machine:
```bash
# From local PowerShell:
ssh root@<host> -p <port> -i $HOME\.ssh\id_rsa "mkdir -p /workspace/openscriptura/data/merged"
scp -P <port> -i $HOME\.ssh\id_rsa data/merged/train.jsonl root@<host>:/workspace/openscriptura/data/merged/
scp -P <port> -i $HOME\.ssh\id_rsa data/merged/eval.jsonl  root@<host>:/workspace/openscriptura/data/merged/
```
**Future fix:** Add a `scripts/upload_data.py` helper or document in the instance setup script.

### 4. vast.ai `ssh-url` API endpoint deprecated
**Problem:** `vastai ssh-url <ID>` returns HTTP 410 (deprecated endpoint).  
**Fix:** Get SSH details from instance metadata:
```bash
vastai show instance <ID> --raw | python -c "
import json,sys; d=json.loads(sys.stdin.read())
if isinstance(d,list): d=d[0]
print(f'ssh root@{d[\"ssh_host\"]} -p {d[\"ssh_port\"]} -i ~/.ssh/id_rsa')"
```

### 5. vast.ai `create ssh-key` syntax
**Problem:** `vastai create ssh-key --path ~/.ssh/id_rsa.pub` fails — `--path` flag doesn't exist.  
**Fix:** Pass key content as positional argument:
```bash
# PowerShell:
vastai create ssh-key (Get-Content $HOME\.ssh\id_rsa.pub)
# Bash:
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"
```

### 8. OOM on RTX 4090 with eager attention + batch_size=4 + seq_len=2048
**Problem:** After switching to `eager` attention (fix #7), training crashed with CUDA OOM:
```
torch.OutOfMemoryError: Tried to allocate 3.92 GiB.
GPU 0 has 23.52 GiB total capacity, 22.10 GiB in use.
```
Flash attention is memory-efficient (O(N) in sequence length); eager attention materialises the full O(N²) attention matrix. At `seq_len=2048` with `batch_size=4`, the attention matrices across 32 layers exhaust 24GB VRAM.  
**Fix:** In all 4 configs: `per_device_train_batch_size: 1` (from 4) + `gradient_accumulation_steps: 16` (from 4) → effective batch size unchanged at 16. Also added `gradient_checkpointing: True` to `SFTConfig` in `04_experiment.py` (recomputes activations on backward pass, ~40% VRAM reduction, ~20% slower).  
**Note:** Effective batch size (train_batch × grad_accum = 16) is preserved, so hyperparameter comparisons remain valid.

### 7. `flash_attn` not pre-installed on vast.ai PyTorch images
**Problem:** All 4 experiment configs had `attn_implementation: "flash_attention_2"`. The `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image does not ship `flash_attn`. Training crashed after model download:
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used ...
the package flash_attn seems to be not installed.
```
**Fix:** Changed all configs (`exp_a/b/c/d.yaml`) to `attn_implementation: "eager"` — always available, no build step.  
**Note:** `eager` is ~10–15% slower but produces identical model weights. For Phase 3 on A100 where training time matters, you can optionally install: `pip install flash-attn --no-build-isolation` (~10 min build).

### 9. `load_best_model_at_end: true` crashes at end of QLoRA training
**Problem:** After all epochs complete, HuggingFace Trainer tries to reload the best checkpoint into the 4-bit quantized PEFT model. With `transformers==4.51.0` + `peft==0.13.0`, this can corrupt `model.config.use_cache` (reset to `True` from checkpoint, breaking the gradient-checkpointing state) and may raise errors during the final `save_model()` call. The entire training run completes successfully but the final save silently fails.  
**Fix:** Set `load_best_model_at_end: false` in all configs. The adapter is saved manually in a `try/finally` block in `04_experiment.py`, so it's always persisted regardless of training outcome. Best checkpoint is selected by reading `results.json` after all 4 experiments complete.  
**Also fixed:** `trainer.train()` is now wrapped in `try/finally` to guarantee `save_model()` and `_write_results()` are called even if training raises an exception.

### 10. `metric_for_best_model: "eval_loss"` fails when eval_dataset is a dict
**Problem:** When `eval_dataset` is passed as a dict `{"B": ..., "C": ..., "all": ...}`, HuggingFace Trainer names metrics `eval_B_loss`, `eval_C_loss`, `eval_all_loss`. There is no `eval_loss`. Training crashed at step 50 (first eval) with `KeyError: eval_loss not found in evaluation metrics`.  
**Fix:** Changed `metric_for_best_model` from `"eval_loss"` to `"eval_all_loss"` in all 4 configs.

### 6. `nohup` required for long training runs
Always launch training with `nohup ... &` so it survives SSH disconnection:
```bash
nohup bash -c 'export LD_LIBRARY_PATH=...; python scripts/04_experiment.py ...' \
  > /workspace/training_exp_d.log 2>&1 &
echo "PID: $!"
```

### 11. Chained training runs: stale CUDA context causes OOM on the next config
**Problem:** When chaining experiments (`exp_c && exp_b && exp_a`), the second run crashed in `prepare_model_for_kbit_training` with `CUDA out of memory` even though each run fits individually. The previous Python process had exited but its CUDA context / allocator memory was not fully released before the next process tried to load the model (the GPU still showed ~16GB in use).
**Fix:** Insert `sleep 30` between chained runs so the OS reclaims the GPU, and start each with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
```bash
nohup bash -c '
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/04_experiment.py --config configs/exp_c.yaml > /workspace/training_exp_c.log 2>&1
sleep 30
python scripts/04_experiment.py --config configs/exp_b.yaml > /workspace/training_exp_b.log 2>&1
' &
```
Verify the GPU is clear with `nvidia-smi` (0 MiB, "No running processes") before relaunching after a crash.

### 12. `EarlyStoppingCallback` asserts `load_best_model_at_end=True` — crashes at step 0
**Problem (Phase 3, `05_train_final.py`):** The stock `transformers.EarlyStoppingCallback.on_train_begin()` does `assert args.load_best_model_at_end`. We keep `load_best_model_at_end=False` for PEFT safety (Lesson #9), so adding the stock callback raises `AssertionError` the instant training starts.
**Fix:** Use a custom `TrainerCallback` (`_make_early_stopping_callback` in `05_train_final.py`) that tracks `eval_all_loss` itself and sets `control.should_training_stop` — no assertion, no best-model reload.
**Corollary:** with early stopping, training halts `patience` evals *after* the best, so the manually-saved `final/` adapter is the LAST (worse) state. `_write_results()` records `best_checkpoint` in `results.json`, `06_export.py` reads it, and `save_total_limit` (10) must exceed `early_stopping_patience` (5) so the best checkpoint is not evicted before training stops.

### 13. Full-bf16 LoRA without kbit-prep needs `enable_input_require_grads()`
**Problem (Phase 3):** In full-bf16 mode (`quantization.enabled: false`) we skip `prepare_model_for_kbit_training`, which is the function that normally calls `enable_input_require_grads()`. With `gradient_checkpointing=True` and a frozen base model, gradients never reach the LoRA adapters → the model silently fails to learn (or raises "element 0 of tensors does not require grad").
**Fix:** Call `model.enable_input_require_grads()` explicitly on the non-quantized path in `apply_lora()` before `get_peft_model()`.

### 14. Phase 4 system prompt vs CEFEAI comparability
**Problem:** The Phase 0 baseline (`00_cefeai_baseline.py`) sent prompts with NO system message. The model is trained WITH a Reformed system prompt. If Phase 4 eval injects the system prompt, the delta vs baseline conflates fine-tuning with prompt injection — violating the comparability lock.
**Fix:** the headline number is the **no-system-prompt** run on both sides. *(Briefly overridden by the Lesson #15 v2 experiment, then **vindicated** by Lesson #16 when the v2 baseline saturated the metric. The flag is now `--system-prompt` to opt INTO v2; default = no prompt.)*

### 15. Protocol v2 (re-baseline WITH the system prompt) — tried, then REJECTED as headline
**Idea:** evaluate BOTH the raw baseline and the fine-tuned model WITH the Reformed system prompt, so the only difference is the weights. Implemented behind a flag, with shared `scripts/utils/cefeai.py` (judge prompts / `wilson_ci` / `baseline_verdict` / `load_system_prompt`) and committed `configs/system_prompt.txt`.
**Why we built the infra anyway:** the shared module + committed prompt are good regardless (they kill drift and the gitignored-data dependency). They stay.
**Outcome:** see Lesson #16 — the v2 baseline run disproved the idea.

### 16. The v2 baseline saturated the metric → headline reverts to v1 (no system prompt)
**Empirical finding (2026-06-08):** running the v2 baseline (raw Qwen3-8B **+ Reformed system prompt**) gave **RR 99.3% / CB 87.8%** (vs v1's 4.7% / 19.6%). The prompt *alone* — applied to the un-fine-tuned model — maxes out the metric.
**Why that's fatal for v2 as the headline:**
1. **Not comparable to CEFEAI.** Every leaderboard model is measured prompt-free; a "be a committed Reformed evangelist (TULIP, Five Solas)" instruction that no other model got makes 99.3% meaningless as a comparison.
2. **Erases the fine-tuning signal.** If the *raw* model + prompt already scores ~99% / ~88%, the *fine-tuned* model + prompt has no headroom — you can't measure what fine-tuning bought. The whole "improved by N points" claim becomes unmeasurable.
**Decision:** the **headline / CEFEAI-comparable protocol is v1 — NO system prompt, on BOTH sides** (the original Lesson #14 conclusion, now confirmed by data). `00` and `07` **default to no prompt**; `--system-prompt` opts into v2, which is kept ONLY as a labeled deployment-behavior datapoint (and for CB, high bias is by-design intent, not a regression).
**Console gotcha:** the RR leaderboard line printed "this run = 4.7%" (a stale static row) while the actual v2 RR summary was **99.3%** — always trust `results/..._summary.json`, not the leaderboard insert.

## Technology Stack

- Python 3.11+ / CUDA 12.4+ (container), host driver CUDA 13.x on vast.ai
- `torch==2.5.1+cu124`, `transformers==4.51.0`, `trl==0.12.0`, `peft==0.13.0`
- `bitsandbytes` (latest, with CUDA symlink fix above), `unsloth==2025.3.19` (Phase 3 only)
- `datasets==3.2.0`, `openai`, `tenacity`, `jsonlines`, `pyyaml`
- GPU: RTX 4090 (Phase 2 experiments), A100 80GB (Phase 3 final training)

## Key Documentation

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — detailed phase breakdown, corpus stats, LoRA configs, cost breakdown
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — PhD review panel findings (M-items)
- [`README_github.md`](README_github.md) — project vision, dataset schema, roadmap
- [`README_huggingface.md`](README_huggingface.md) — HuggingFace model/dataset card
- [`docs/THEOLOGICAL_STATEMENT.md`](docs/THEOLOGICAL_STATEMENT.md) — confessional scope
- [`docs/PASTORAL_REVIEW_PROTOCOL.md`](docs/PASTORAL_REVIEW_PROTOCOL.md) — Tier A pastoral review process

## Phase 0 Blocking Items (resolved — pattern map)

- **M2** `enable_thinking: false` — `api_client.py:chat()` + `<think>` stripping
- **M3** pinned `requirements.txt` — present
- **M4** `.env.example` — present; `.gitignore` excludes `.env`
- **M5** content-only hashing — `hash.py:content_hash()`
- **M8** Wilson CI — `00_cefeai_baseline.py:_wilson_ci()`
- **M10/M11** tenacity retry + cost guardrail — `api_client.py` + `cost_tracker.py`
- **M12** theological docs — `docs/THEOLOGICAL_STATEMENT.md` + `docs/PASTORAL_REVIEW_PROTOCOL.md`
