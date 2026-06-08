# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenScriptura** is an applied research pipeline to fine-tune open LLMs (starting with Qwen3-8B) with Protestant theological corpus. The first release target is `qwen3-8b-reformed-pt-br-v0.1` (Reformed theology, Brazilian Portuguese), evaluated on the CEFEAI benchmark.

**Status:** Phase 0 ✅ (RR 4.7%, CB 19.6%) · Phase 1 ✅ (2,968 records: 839 C + 2,129 B) · Phase 2 🔄 (4 experiments running on vast.ai RTX 4090, instance 40077545) · Phase 3–4 🔲

## Commands

```bash
# Install — local dev (Windows/PowerShell)
pip install -r requirements.txt --break-system-packages

# Install — vast.ai GPU instance (skip unsloth, install core only)
pip install transformers==4.51.0 trl==0.12.0 peft==0.13.0 bitsandbytes \
  datasets==3.2.0 pyyaml==6.0.2 python-dotenv==1.0.1 scipy==1.14.1 \
  sentencepiece==0.2.0 tokenizers==0.21.0 tenacity==9.0.0 httpx==0.27.0 jsonlines==4.0.0

# Phase 0: Baseline ✅
python scripts/00_cefeai_baseline.py --dry-run
python scripts/00_cefeai_baseline.py --benchmark rr
python scripts/00_cefeai_baseline.py --benchmark cb

# Phase 1: Dataset construction ✅
python scripts/01_build_tier_c.py --dry-run        # Tier C: 839 records ✅
python scripts/02_build_tier_b.py --dry-run        # Tier B: 2,129 records ✅
python scripts/03_eda.py                           # EDA → reports/eda_report.html ✅
python scripts/merge_dataset.py --dry-run          # merge → data/merged/ ✅
python scripts/merge_dataset.py                    # 2,873 train + 151 eval ✅

# Phase 2: Experiments 🔄 (run on vast.ai RTX 4090)
python scripts/04_experiment.py --config configs/exp_d.yaml --dry-run
python scripts/04_experiment.py --config configs/exp_d.yaml   # r=64 lr=1e-4 ← RECOMMENDED
python scripts/04_experiment.py --config configs/exp_c.yaml   # r=64 lr=2e-4
python scripts/04_experiment.py --config configs/exp_b.yaml   # r=16 lr=1e-4
python scripts/04_experiment.py --config configs/exp_a.yaml   # r=16 lr=2e-4

# Phase 2: vast.ai automation
pip install vastai
vastai set api-key <KEY>                           # from cloud.vast.ai/account
python scripts/vastai_run_experiments.py --search
python scripts/vastai_run_experiments.py --config configs/exp_d.yaml --all-configs --wait
python scripts/vastai_run_experiments.py --status <INSTANCE_ID>
python scripts/vastai_run_experiments.py --destroy <INSTANCE_ID>

# Phase 3: Final training + export 🔲 (not yet written)
python scripts/05_train_final.py --config configs/exp_d.yaml
python scripts/06_export.py --config configs/exp_d.yaml       # merge + GGUF Q4/Q5/Q8

# Phase 4: Re-evaluation 🔲 (not yet written)
python scripts/07_cefeai_eval.py --benchmark rr
python scripts/07_cefeai_eval.py --benchmark cb

# Tests
$env:PYTHONPATH = "scripts"; pytest tests/ -v
```

## Architecture

### 4-Phase Pipeline

```
Phase 0: CEFEAI Baseline (raw Qwen3-8B, counterfactual) ✅
    ↓
Phase 1: Dataset Construction ✅
    ├── Tier C: Native sources — catechisms & confessions → 839 records
    ├── Tier B: Synthetic — LLM-generated + judge-filtered → 2,129 records
    └── Tier A: Manual — pastoral council review (v0.1: skipped)
    ↓ merge_dataset.py → data/merged/train.jsonl (2,873) + eval.jsonl (151)
Phase 2: Controlled Experiments (2×2 LoRA matrix, 4 configs) 🔄
    └── RTX 4090 on vast.ai (~7h total, ~$3.50)
    ↓
Phase 3: Final Fine-tuning (QLoRA on A100) → Merge → GGUF export 🔲
    └── Quantizations: Q4_K_M, Q5_K_M, Q8_0
    ↓
Phase 4: CEFEAI Re-evaluation + arXiv/HuggingFace publication 🔲
```

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

### Conventions

- Scripts numbered `NN_name.py` in `scripts/`; shared code in `scripts/utils/`.
- `sys.stdout.reconfigure(encoding="utf-8")` at top of every entry-point script (Windows/PowerShell UTF-8).
- Paths resolved from `PROJECT_ROOT = Path(__file__).resolve().parent.parent`.
- Every API-spending script: `OpenRouterClient` + `CostTracker` + `--dry-run` + `--resume` checkpoint.
- **CEFEAI comparability lock:** `temperature=0.0, seed=42, enable_thinking=False` — inference only, never changed between Phase 0 and Phase 4. Training-side settings (LR, LoRA rank, batch size, data generation temperature) are orthogonal and do NOT affect comparability.

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

### 7. `flash_attn` not pre-installed on vast.ai PyTorch images
**Problem:** All 4 experiment configs had `attn_implementation: "flash_attention_2"`. The `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image does not ship `flash_attn`. Training crashed after model download:
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used ...
the package flash_attn seems to be not installed.
```
**Fix:** Changed all configs (`exp_a/b/c/d.yaml`) to `attn_implementation: "eager"` — always available, no build step.  
**Note:** `eager` is ~10–15% slower but produces identical model weights. For Phase 3 on A100 where training time matters, you can optionally install: `pip install flash-attn --no-build-isolation` (~10 min build).

### 6. `nohup` required for long training runs
Always launch training with `nohup ... &` so it survives SSH disconnection:
```bash
nohup bash -c 'export LD_LIBRARY_PATH=...; python scripts/04_experiment.py ...' \
  > /workspace/training_exp_d.log 2>&1 &
echo "PID: $!"
```

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
