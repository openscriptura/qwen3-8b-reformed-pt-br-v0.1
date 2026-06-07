# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenScriptura** is an applied research pipeline to fine-tune open LLMs (starting with Qwen3-8B) with Protestant theological corpus. The first release target is `qwen3-8b-reformed-pt-br-v0.1` (Reformed theology, Brazilian Portuguese), evaluated on the CEFEAI benchmark.

**Status:** Phase 0 (CEFEAI baseline) is **implemented**. The shared `scripts/utils/` layer and `scripts/00_cefeai_baseline.py` exist and run; all Phase-0 blocking items from `VALIDATION_REPORT.md` (M2–M12) are addressed. Phase 1–4 scripts (`01_`–`07_`) are **planned but not yet written**. The repo is **not a git repository** yet, so the manifest/commit-SHA reproducibility scheme in the design is still aspirational.

## Commands

```bash
# Install (note the flag — it is in the requirements.txt header)
pip install -r requirements.txt --break-system-packages

# Phase 0: Baseline — IMPLEMENTED
# Validate config + check files/keys WITHOUT spending money or calling APIs:
python scripts/00_cefeai_baseline.py --dry-run
# Run one benchmark (rr = Religious Representation 150; cb = Conversion Bias 1456; both = default):
python scripts/00_cefeai_baseline.py --benchmark rr
# Resume is ON by default (re-reads results/*.jsonl, skips completed prompt_ids). Force a fresh run:
python scripts/00_cefeai_baseline.py --benchmark rr --no-resume
# Override OPENROUTER_MODEL_BASELINE from .env:
python scripts/00_cefeai_baseline.py --model qwen/qwen3-8b --benchmark rr

# Phase 1–4: NOT YET WRITTEN — these are the planned entry points:
#   scripts/01_extract_tier_c.py   scripts/02_extract_tier_b.py
#   scripts/03_confessional_judge.py   scripts/04_merge_dataset.py
#   scripts/05_train.py   scripts/06_evaluate.py   scripts/07_cefeai_eval.py
```

**Before running Phase 0 for real:** benchmark inputs must exist at `data/cefeai/rr_150.jsonl` and `data/cefeai/cb_1456.jsonl` (each line `{"id": ..., "prompt": ...}`). The `data/` directory is **not** in the repo — download from https://cefe.ai first. The script aborts (or warns under `--dry-run`) if a file is missing. Always run `--dry-run` first to confirm keys, model, and cost limit.

Dev tooling (pinned in `requirements.txt`, configs not yet added):
- Formatter/linter: `black` / `ruff`
- Tests: `pytest` (no tests written yet)

## Architecture

### 4-Phase Pipeline

```
Phase 0: CEFEAI Baseline (raw Qwen3-8B, counterfactual)
    ↓
Phase 1: Dataset Construction
    ├── Tier C: Native sources — catechisms & confessions (1,000–2,000 Q&A)
    ├── Tier B: Synthetic — LLM-generated + judge-filtered (3,000–4,000 Q&A)
    └── Tier A: Manual — pastoral council review (500–1,000 Q&A)
    ↓
Phase 2: Controlled Experiments (2×2 LoRA hyperparameter matrix, 4 configs)
    ↓
Phase 3: Final Fine-tuning (QLoRA on A100) → Merge → GGUF export
    └── Quantizations: Q4_K_M, Q5_K_M, Q8_0
    ↓
Phase 4: CEFEAI Re-evaluation + arXiv/HuggingFace publication
```

**Phase 0 must complete before Phase 1 begins** — baseline is the counterfactual.

### Data Schema

The **design/training corpus** schema (documented in the READMEs) is JSONL with first-class `tradition` and `language` fields (BCP 47):

```json
{
  "tradition": "reformed",
  "language": "pt-BR",
  "tier": "C",
  "source": "wcf",
  "question": "...",
  "answer": "...",
  "sha256": "<content-hash>"
}
```

Each split is intended to ship a `manifest.json` with SHA-256 hashes + Git commit SHA for reproducibility.

⚠️ **Two schema details to reconcile when writing Phase 1 scripts:**
- `scripts/utils/hash.py:content_hash()` hashes `record["messages"]`, `record["tradition"]`, `record["lang"]` — i.e. it expects a chat-style `messages` field and the key `lang`, **not** the `question`/`answer`/`language` shown above. Decide on one shape (chat `messages` vs `question`/`answer`, and `lang` vs `language`) before generating data.
- CEFEAI **benchmark input** is a different, simpler schema: `{"id", "prompt"}`. Baseline **output** records (in `results/*.jsonl`) use `prompt_id`, `judge_score`, `cost_usd`, etc. — see `_process_one()` in the baseline script.

### Utility Modules (`scripts/utils/`, all implemented)

Shared infrastructure; import via `from utils.<mod> import ...` after the baseline-script trick of `sys.path.insert(0, str(PROJECT_ROOT / "scripts"))`. Re-exported from `scripts/utils/__init__.py`.

- `api_client.py` — `OpenRouterClient`: async chat wrapper. Module-level `_post_json` carries the tenacity retry (5 attempts, exponential backoff) so it instruments async correctly. Injects `enable_thinking: false` into the request body, strips `<think>…</think>` from responses, logs raw JSON to `logs/raw/`, and estimates cost from a hardcoded `_PRICE_PER_TOKEN` table (M10).
- `cost_tracker.py` — `CostTracker`: thread-safe accumulator; `.add()` raises `CostLimitExceeded` the moment spend crosses the limit (M11). Limit comes from `COST_LIMIT_USD_PHASE0` (default $2.00).
- `logger.py` — `get_logger(name)`: dual stdout + `logs/YYYYMMDD_<name>.log` handler; file always DEBUG, console at `LOG_LEVEL`.
- `progress.py` — `ProgressBar`: TTY-aware (Unicode bar inline; ASCII bar per-line in log files).
- `report.py` — `generate_all_reports()` + `print_console_summary()`: writes `.md`, `.json` sidecar, and a self-contained dark-mode `.html` (Chart.js via CDN) to `results/`. Ported from a sibling `pastor-ai` project.
- `hash.py` — `content_hash()`: canonical SHA-256 over content-only fields for dedup (M5).

### Conventions (follow these in new scripts)

- **Phase scripts are numbered** `NN_name.py` and live directly in `scripts/`; shared code goes in `scripts/utils/`.
- **Windows/PowerShell is the dev environment.** Scripts call `sys.stdout.reconfigure(encoding="utf-8")` at startup so emoji render in PowerShell — replicate this in any new entry-point script.
- **Paths are resolved from `PROJECT_ROOT`** (`Path(__file__).resolve().parent.parent`) so scripts work from any CWD. Outputs go to `results/`, logs to `logs/` (+ raw API dumps in `logs/raw/`).
- **Every API-spending script must** route calls through `OpenRouterClient` (retry) and feed each cost into a `CostTracker` (hard stop), support `--dry-run`, and checkpoint to JSONL so `--resume` can skip done work.
- **Inference settings are fixed for comparability** (baseline vs post-training): `enable_thinking=False`, `temperature=0.0`, `seed=42`. Do not change these between baseline and evaluation runs.
- **Statistics:** Wilson score CI is implemented inline in the baseline (`_wilson_ci`, via `scipy.stats.norm`) rather than `statsmodels`. Reuse this for any new proportion metric (M8).

### External Services

| Service | Purpose |
|---------|---------|
| OpenRouter API | Qwen3-8B inference for baseline + evaluation |
| DeepSeek API | Tier B dataset generation (Flash) and judging (Pro) |
| RunPod Secure Cloud | A100 80GB for final fine-tuning |
| HuggingFace Hub | Model & dataset distribution |
| CEFEAI | External benchmark (150 RR prompts + 1,456 CB pairs) |

## Key Design Constraints

**Theological precision:** Models are explicitly tradition-specific. The Reformed v0.1 confessional hierarchy is: WCF > Canons of Dort > London Baptist Confession 1689. No blending of incompatible doctrines. Pastoral council review is mandatory for Tier A data.

**Reproducibility:** seed=42, pinned dependency versions, SHA-256 content hashing, manifest.json per split, and phase sequencing enforced in design.

**Cost guardrails:** Total budget for v0.1 is ~$15 USD. Scripts must enforce hard limits and fail before overrunning.

**Evaluation:** CEFEAI benchmark (external, public). Target for v0.1: 60–70% "Any Representation" vs. 6% baseline. Statistical protocol: z-test for proportions, Bonferroni correction, Wilson confidence intervals.

**Fine-tuning method:** QLoRA (4-bit quantized base). Recommended config (D): r=64, α=128, lr=1e-4.

## Phase 0 Blocking Items (resolved)

Per `VALIDATION_REPORT.md`, these gated `00_cefeai_baseline.py` and are now done — preserved here as a map of *where* each fix lives, since the same patterns are mandatory for Phase 1+ scripts:

- **M2** `enable_thinking: false` — `api_client.py:chat()` (request body) + response `<think>` stripping
- **M3** pinned `requirements.txt` — present
- **M4** `.env.example` — present (see security note below)
- **M5** content-only hashing — `hash.py:content_hash()`
- **M8** Wilson CI — `00_cefeai_baseline.py:_wilson_ci()`
- **M10/M11** tenacity retry + cost guardrail — `api_client.py:_post_json` + `cost_tracker.py`
- **M12** `docs/THEOLOGICAL_STATEMENT.md` + `docs/PASTORAL_REVIEW_PROTOCOL.md` — present

⚠️ **Security:** `.env.example` currently contains what look like **real** `OPENROUTER_API_KEY` and `HF_TOKEN` values, a populated `.env` exists, and there is **no `.gitignore`**. Before this repo is initialized as git / pushed, replace those with placeholders, add a `.gitignore` excluding `.env`, and rotate the exposed keys.

## Technology Stack

- Python 3.11+ / CUDA 12.4+
- `torch==2.5.1+cu124`, `transformers==4.51.0`, `trl==0.12.0`, `peft==0.13.0`, `bitsandbytes==0.44.1`, `unsloth==2025.3.0`
- `datasets`, `openai` (OpenRouter-compatible), `tenacity`, `jsonlines`, `pyyaml`
- GPU: RTX 4090 (24GB) minimum for experiments; A100 80GB for final training

## Key Documentation

- [`README_github.md`](README_github.md) — project vision, dataset schema, roadmap
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — detailed phase breakdown, script specs, LoRA configs, cost breakdown
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — PhD review panel findings, blocking corrections (M-items), mandatory changes
- [`README_huggingface.md`](README_huggingface.md) — HuggingFace model/dataset card content
- [`docs/THEOLOGICAL_STATEMENT.md`](docs/THEOLOGICAL_STATEMENT.md) — confessional scope (M12)
- [`docs/PASTORAL_REVIEW_PROTOCOL.md`](docs/PASTORAL_REVIEW_PROTOCOL.md) — Tier A pastoral council review process
- `openscriptura-plan.jsx` — standalone React/JSX visualization of the plan (not part of the Python pipeline)
