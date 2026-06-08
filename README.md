# OpenScriptura

> **Open LLMs for Protestant Theology — any tradition, any language, built to last.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-gold.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-openscriptura-yellow)](https://huggingface.co/openscriptura)
[![CEFEAI Benchmark](https://img.shields.io/badge/Benchmark-CEFEAI%20AFB-blue)](https://cefe.ai)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

---

> **Project status (June 2026):** `v0.1` work-in-progress. Phase 0 baseline ✅ (v1, no system prompt: RR 4.7% / CB 19.6%) · Phase 1 dataset ✅ (2,968 records → 2,873 train / 151 eval) · Phase 2 LoRA sweep ✅ (winner **r=64, lr=2e-4**) · Phase 3 final-training + GGUF-export and Phase 4 evaluation scripts (`05_`–`07_`) ✅ **written, reviewed, simulated** — pending the A100 run. Evaluation upgraded to **protocol v2**: the baseline is re-run *with* the Reformed system prompt so the baseline↔fine-tuned comparison is valid (see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) → "Protocol v2" and [`CHANGELOG.md`](CHANGELOG.md)). The Reformed pt-BR model has not been released yet.

## What is OpenScriptura

OpenScriptura is an open-source applied research project that fine-tunes open LLMs with Protestant theological corpus — across **all Protestant traditions** and **all languages**.

**The problem:** Existing language models treat questions of faith with excessive neutrality or simply ignore the religious dimension. The CEFEAI benchmark demonstrated that all 27 frontier models tested in 2026 produce **0% predominantly religious responses** to everyday ethical questions, with even the best model (Grok 4.20) ignoring religious perspective 70% of the time.

**The solution:** Tradition-specific fine-tuning with high-quality confessional corpus, a three-layer validation pipeline, and scientifically comparable evaluation via CEFEAI.

**The scope:** OpenScriptura serves the full breadth of Protestant Christianity — Reformed, Lutheran, Anglican, Baptist, Methodist, Pentecostal, and beyond. Each tradition has its own confessional standards, its own dataset, and its own model variant. We start with Reformed theology in Brazilian Portuguese because that is where we started — not because that is where we end.

---

## Scientific Motivation

```
Base models — CEFEAI Religious Representation (June 2026, 150 questions):

  Grok 4.20        ████░░░░░░░░░░░░░░░░  29.3% Any Representation
  Mistral Large    ████░░░░░░░░░░░░░░░░  23.3%
  GPT-5.4          ███░░░░░░░░░░░░░░░░░  17.3%
  Qwen3-8B base    █░░░░░░░░░░░░░░░░░░░   4.7%   ← our measured v1 baseline (no system prompt)
  Claude Opus 4.7  █░░░░░░░░░░░░░░░░░░░   4.0%
  Llama 4 Scout    ░░░░░░░░░░░░░░░░░░░░   3.3%

  Predominantly Religious: 0% across all 27 models tested.

  OpenScriptura v0.1 target:  ████████████░░░░░░░░  > 60%
```

---

## Protestant Traditions

OpenScriptura is designed to serve the full breadth of Protestant Christianity. Each tradition is treated with fidelity to its own confessional standards — there is no attempt to blend incompatible doctrines.

| Tradition | Primary Confessional Standards | Status |
|---|---|---|
| **Reformed / Calvinist** | Westminster Confession of Faith, Canons of Dort, Heidelberg Catechism, London Baptist Confession 1689 | 🔄 **v0.1 — in progress** |
| **Lutheran** | Augsburg Confession (1530), Luther's Small & Large Catechisms, Formula of Concord | 📋 Planned |
| **Anglican / Episcopal** | Thirty-Nine Articles (1571), Book of Common Prayer | 📋 Planned |
| **Baptist (Traditional)** | Second London Confession (1689), New Hampshire Confession (1833), Baptist Faith & Message | 📋 Planned |
| **Methodist / Wesleyan** | Wesley's Articles of Religion, Methodist Discipline | 📋 Planned |
| **Pentecostal** | Assemblies of God Statement of Fundamental Truths | 📋 Planned |
| **Congregationalist** | Savoy Declaration (1658) | 📋 Planned |
| **Other Protestant** | Community contributions welcome | 📋 Open |

> **Note on tradition boundaries:** Models are labeled clearly by tradition. A Reformed model will not speak as a Lutheran, and vice versa. Where traditions share common ground (e.g. justification by faith), datasets naturally overlap. Where they diverge (e.g. baptism, Lord's Supper, church governance), each tradition's dataset reflects its own confessional position.

---

## The Pipeline

```
Phase 0  CEFEAI Baseline (raw Qwen3-8B, counterfactual)        ← implemented
Phase 1  Dataset Construction (Tier C native · B synthetic · A manual)
Phase 2  Controlled Experiments (2×2 LoRA hyperparameter matrix)
Phase 3  Final QLoRA Fine-tuning → Merge → GGUF export
Phase 4  CEFEAI Re-evaluation + arXiv / HuggingFace publication
```

**Phase 0 must complete before Phase 1** — the raw-model baseline is the counterfactual the whole project is measured against.

---

## Quick Start

### Requirements

```
Python 3.11+
CUDA 12.4+  (for fine-tuning phases)
GPU: RTX 4090 (24GB) minimum for experiments; A100 80GB for the final run
```

### Installation

```bash
git clone https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1
cd qwen3-8b-reformed-pt-br-v0.1
pip install -r requirements.txt --break-system-packages
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
# Edit .env with your OpenRouter and HuggingFace keys
```

### Run the CEFEAI baseline (Phase 0 — implemented)

```bash
# 1. Validate config and keys WITHOUT spending money or calling any API:
python scripts/00_cefeai_baseline.py --dry-run

# 2. Run a benchmark. rr = Religious Representation (150), cb = Conversion Bias (1456), both = default.
#    Resume is ON by default (skips already-completed prompts); add --no-resume to start fresh.
#    Protocol v2 (system prompt) is the default; add --no-system-prompt for the legacy v1 run.
python scripts/00_cefeai_baseline.py --benchmark both

# Cost: ~$0.30 | guarded by a hard cost limit (COST_LIMIT_USD_PHASE0, default $2.00)
```

> Benchmark inputs must be placed at `data/cefeai/rr_150.jsonl` and `data/cefeai/cb_1456.jsonl`
> (each line `{"id": ..., "prompt": ...}`). These are **not** bundled — obtain them from https://cefe.ai.
> Results, reports (`.md`/`.json`/`.html` with 95% Wilson CIs), and logs are written to `results/` and `logs/`.

### Build the dataset / train / evaluate (all scripts written ✅)

```bash
# Phase 1 — dataset
python scripts/01_build_tier_c.py        # Tier C: 839 confessional records
python scripts/02_build_tier_b.py        # Tier B: 2,129 synthetic records
python scripts/03_eda.py                 # EDA report
python scripts/merge_dataset.py          # → data/merged/train.jsonl + eval.jsonl

# Phase 2 — LoRA sweep (complete; winner exp_c r=64 lr=2e-4)
python scripts/04_experiment.py --config configs/exp_c.yaml

# Phase 3 — final train + export (A100)
python scripts/05_train_final.py --config configs/final.yaml
python scripts/06_export.py      --config configs/final.yaml --push-to-hub

# Phase 4 — re-evaluate (protocol v2)
python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both
```

---

## Dataset Schema (target v1.0)

`tradition` and `language` are first-class fields. The same schema works for any Protestant tradition in any language:

```json
{
  "id":                 "os_reformed_c_wsc_001",
  "instruction":        "What is justification by faith alone?",
  "input":              "",
  "output":             "Justification is the forensic act by which God declares the sinner righteous...",
  "source":             "westminster_sc_q033",
  "tier":               "C",
  "tradition":          "reformed",
  "confessional_ref":   "WCF_11.1",
  "language":           "pt-BR",
  "annotation_model":   "deepseek-v4-flash",
  "validated_by":       "gemini-flash-2",
  "confessional_score": 0.91,
  "dataset_version":    "v1.0",
  "created_at":         "2026-06-07"
}
```

`tradition`: `reformed` · `lutheran` · `anglican` · `baptist` · `methodist` · `pentecostal` · `congregationalist`
`language` follows BCP 47: `pt-BR` · `en` · `es` · `de` · `fr` · etc.

> The Phase-1 generation scripts and the content-hashing util (`scripts/utils/hash.py`) still need the exact
> field shape reconciled (chat `messages` vs `instruction`/`output`; `lang` vs `language`) before data is generated.

---

## Fine-tuning Configuration

QLoRA (4-bit base). Default recommended config **D**: `r=64`, `α=128`, `lr=1e-4`, cosine schedule, 3 epochs, `seed=42`.
Experiments sweep a 2×2 matrix over learning rate (`2e-4` / `1e-4`) and LoRA rank (`32` / `64`).

**Cost per model variant:** ~$11–17 total (dataset generation + fine-tuning + evaluation).

---

## Roadmap

```
Phase 1 — Reformed Foundation
  v0.1  Qwen3-8B    · Reformed  · PT-BR     ← we are here
  v0.2  GPT-OSS 20B · Reformed  · PT-BR+EN
  v0.3  Gemma 4 31B · Reformed  · multilingual

Phase 2 — Tradition Expansion
  v1.0  Lutheran    · EN (Book of Concord corpus)
  v1.1  Anglican    · EN (39 Articles + BCP)
  v1.2  Baptist     · EN + PT-BR

Phase 3 — Community-Driven
  v2.x  Methodist, Pentecostal, Congregationalist
  v2.x  Spanish, German, indigenous languages
  v3.0  Multi-tradition unified benchmark paper
```

---

## Documentation

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — detailed phase breakdown, script specs, LoRA configs, cost breakdown
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) — PhD review panel findings and mandatory corrections (the "M-items")
- [`docs/THEOLOGICAL_STATEMENT.md`](docs/THEOLOGICAL_STATEMENT.md) — confessional scope and doctrinal commitments
- [`docs/PASTORAL_REVIEW_PROTOCOL.md`](docs/PASTORAL_REVIEW_PROTOCOL.md) — Tier A pastoral council review process
- [`README_huggingface.md`](README_huggingface.md) — HuggingFace model/dataset card content
- [`CLAUDE.md`](CLAUDE.md) — guidance for working in this repo (architecture, conventions, commands)

---

## Contributing

We welcome contributions across all Protestant traditions and languages. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

- **Dataset examples:** open an issue with theological Q&A in any tradition and language. Include the confessional reference.
- **Doctrinal corrections:** open an issue with the problematic example and the correct reference.
- **New tradition / language:** open an issue to discuss. You'll need primary confessional standards, a seed corpus, and someone willing to do pastoral review.

---

## Citation

```bibtex
@misc{openscriptura2026,
  title        = {OpenScriptura: Open LLMs for Protestant Theology},
  author       = {OpenScriptura Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/openscriptura}},
  note         = {Apache 2.0. Multi-tradition, language-agnostic pipeline.
                  Benchmarked against CEFEAI AllFaith Benchmark.}
}
```

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Free use including commercial; attribution required.

---

<div align="center">

**Soli Deo Gloria**

*For the glory of God and the good of the Church — in every tradition and every language.*

[HuggingFace](https://huggingface.co/openscriptura) · [CEFEAI](https://cefe.ai) · [Issues](https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1/issues)

</div>
