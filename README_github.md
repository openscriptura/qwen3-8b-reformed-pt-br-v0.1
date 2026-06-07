# OpenScriptura

> **Open LLMs for Protestant Theology — any tradition, any language, built to last.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-gold.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-openscriptura-yellow)](https://huggingface.co/openscriptura)
[![CEFEAI Benchmark](https://img.shields.io/badge/Benchmark-CEFEAI%20AFB-blue)](https://cefe.ai)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

---

## What is OpenScriptura

OpenScriptura is an open-source applied research project that fine-tunes open LLMs with Protestant theological corpus — across **all Protestant traditions** and **all languages**.

**The problem:** Existing language models treat questions of faith with excessive neutrality or simply ignore the religious dimension. The CEFEAI benchmark demonstrated that all 27 frontier models tested in 2026 produce **0% predominantly religious responses** to everyday ethical questions, with even the best model (Grok 4.20) ignoring religious perspective 70% of the time.

**The solution:** Tradition-specific fine-tuning with high-quality confessional corpus, a three-layer validation pipeline, and scientifically comparable evaluation via CEFEAI.

**The scope:** OpenScriptura serves the full breadth of Protestant Christianity — Reformed, Lutheran, Anglican, Baptist, Methodist, Pentecostal, and beyond. Each tradition has its own confessional standards, its own dataset, and its own model variant. We start with Reformed theology in Brazilian Portuguese because that is where we started — not because that is where we end.

---

## Scientific Motivation

```
Base models — CEFEAI Religious Representation (May 2026, 150 questions):

  Grok 4.20        ████░░░░░░░░░░░░░░░░  29.3% Any Representation
  Mistral Large    ████░░░░░░░░░░░░░░░░  23.3%
  GPT-5.4          ███░░░░░░░░░░░░░░░░░  17.3%
  Qwen3-8B base    █░░░░░░░░░░░░░░░░░░░  ~6.0%   ← our starting point
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
| **Reformed / Calvinist** | Westminster Confession of Faith, Canons of Dort, Heidelberg Catechism, London Baptist Confession 1689 | 🔄 **v0.1 — first release** |
| **Lutheran** | Augsburg Confession (1530), Luther's Small & Large Catechisms, Formula of Concord | 📋 Planned |
| **Anglican / Episcopal** | Thirty-Nine Articles (1571), Book of Common Prayer | 📋 Planned |
| **Baptist (Traditional)** | Second London Confession (1689), New Hampshire Confession (1833), Baptist Faith & Message | 📋 Planned |
| **Methodist / Wesleyan** | Wesley's Articles of Religion, Methodist Discipline | 📋 Planned |
| **Pentecostal** | Assemblies of God Statement of Fundamental Truths | 📋 Planned |
| **Congregationalist** | Savoy Declaration (1658) | 📋 Planned |
| **Other Protestant** | Community contributions welcome | 📋 Open |

> **Note on tradition boundaries:** Models are labeled clearly by tradition. A Reformed model will not speak as a Lutheran, and vice versa. Where traditions share common ground (e.g. justification by faith), datasets naturally overlap. Where they diverge (e.g. baptism, Lord's Supper, church governance), each tradition's dataset reflects its own confessional position.

---

## Published Models

| Model | Base | Tradition | Language | CEFEAI RR | Download |
|---|---|---|---|---|---|
| `qwen3-8b-reformed-pt-br-v0.1` | Qwen3-8B | Reformed | PT-BR | 🔄 pending | [HF Hub](https://huggingface.co/openscriptura/qwen3-8b-reformed-pt-br-v0.1) |
| `qwen3-8b-lutheran-en-v0.1` | Qwen3-8B | Lutheran | EN | 📋 planned | — |
| `gpt-oss-20b-v1.0` | GPT-OSS 20B | Multi-tradition | Multilingual | 📋 planned | — |

Model naming convention: `openscriptura/{base}-{tradition}-{lang}-{version}`

---

## Repository Structure

```
openscriptura/
├── README.md
├── CONTRIBUTING.md
├── LICENSE                              # Apache 2.0
├── requirements.txt
│
├── traditions/                          # One folder per tradition
│   ├── reformed/
│   │   ├── data/
│   │   │   ├── tier_c/                  # Native Q&A — catechisms
│   │   │   │   ├── westminster_sc.jsonl
│   │   │   │   ├── westminster_lc.jsonl
│   │   │   │   ├── heidelberg.jsonl
│   │   │   │   └── canons_dort.jsonl
│   │   │   ├── tier_b/                  # Sermons and texts
│   │   │   └── tier_a/                  # Structured doctrinal topics
│   │   └── configs/
│   │       └── confessions.yaml         # WCF, Dort, LCF references
│   ├── lutheran/
│   │   ├── data/
│   │   └── configs/
│   │       └── confessions.yaml         # Augsburg, Book of Concord
│   ├── anglican/
│   ├── baptist/
│   ├── methodist/
│   └── pentecostal/
│
├── data/
│   └── processed/
│       ├── reformed-pt-br-v1.0/
│       │   ├── train.jsonl              # 4,800 examples (seed=42)
│       │   ├── eval.jsonl              # 600 examples
│       │   ├── test.jsonl              # 600 examples (holdout)
│       │   └── manifest.json           # SHA-256 per split
│       └── lutheran-en-v1.0/
│
├── scripts/
│   ├── 00_cefeai_baseline.py
│   ├── 01_extract_tier_c.py
│   ├── 02_extract_tier_b.py
│   ├── 03_confessional_judge.py        # Tradition-aware judge
│   ├── 04_merge_dataset.py
│   ├── 05_train.py
│   ├── 06_evaluate.py
│   └── 07_cefeai_eval.py
│
├── configs/
│   ├── exp_a.yaml                       # LR=2e-4, rank=32
│   ├── exp_b.yaml                       # LR=2e-4, rank=64
│   ├── exp_c.yaml                       # LR=1e-4, rank=32
│   └── exp_d.yaml                       # LR=1e-4, rank=64 (default)
│
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_dataset_eda.ipynb
│   └── 03_results_comparison.ipynb
│
└── docs/
    ├── METHODOLOGY.md
    ├── THEOLOGICAL_STATEMENT.md
    ├── ADDING_A_TRADITION.md            # Guide: how to add a new tradition
    └── ADDING_A_LANGUAGE.md            # Guide: how to add a new language
```

---

## Quick Start

### Requirements

```bash
Python 3.11+
CUDA 12.4+
GPU: RTX 4090 (24GB) minimum for experiments
     A100 80GB recommended for final run
```

### Installation

```bash
git clone https://github.com/openscriptura/openscriptura
cd openscriptura
pip install -r requirements.txt --break-system-packages
cp .env.example .env
# Edit .env with API keys (OpenRouter, HuggingFace)
```

### Run CEFEAI baseline

```bash
# Cost: ~$0.29 | Time: ~2h
python scripts/00_cefeai_baseline.py --model qwen/qwen3-8b --benchmark rr
python scripts/00_cefeai_baseline.py --model qwen/qwen3-8b --benchmark cb
```

### Build the dataset (Reformed PT-BR v0.1)

```bash
python scripts/01_extract_tier_c.py   --tradition reformed
python scripts/02_extract_tier_b.py   --tradition reformed --lang pt-BR
python scripts/03_confessional_judge.py --tradition reformed
python scripts/04_merge_dataset.py    --tradition reformed --seed 42
```

### Fine-tuning

```bash
# Free validation on Kaggle
python scripts/05_train.py --config configs/exp_d.yaml --tradition reformed --samples 500

# Full run — RunPod Secure A100
python scripts/05_train.py --config configs/exp_d.yaml --tradition reformed --full
```

---

## Dataset Schema (v1.0)

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
  "difficulty":         "intermediate",
  "annotation_model":   "deepseek-v4-flash",
  "validated_by":       "gemini-flash-2",
  "confidence":         0.94,
  "confessional_score": 0.91,
  "dataset_version":    "v1.0",
  "created_at":         "2026-06-07"
}
```

`tradition` values: `reformed` · `lutheran` · `anglican` · `baptist` · `methodist` · `pentecostal` · `congregationalist`

`language` follows BCP 47: `pt-BR` · `en` · `es` · `de` · `fr` · etc.

---

## Fine-tuning Configuration

```yaml
# configs/exp_d.yaml — default (OpenMed config D)
model:
  name: Qwen/Qwen3-8B
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

lora:
  r: 64
  lora_alpha: 128    # 2 × lora_r (canonical — see VALIDATION_REPORT.md M1)
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  learning_rate: 1e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  bf16: true
  max_seq_length: 2048
  seed: 42
```

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

## Contributing

We welcome contributions across all Protestant traditions and languages.

**New tradition:** see [`docs/ADDING_A_TRADITION.md`](docs/ADDING_A_TRADITION.md). You need: primary confessional standards, a seed corpus of at least 200 examples, and one person willing to do pastoral review.

**New language:** see [`docs/ADDING_A_LANGUAGE.md`](docs/ADDING_A_LANGUAGE.md).

**Dataset examples:** open an issue with theological Q&A in any tradition and language. Include the confessional reference.

**Doctrinal corrections:** open an issue with the problematic example and the correct reference.

**Code:** issues labeled `good first issue`.

---

## Citation

```bibtex
@misc{openscriptura2026,
  title        = {OpenScriptura: Open LLMs for Protestant Theology},
  author       = {OpenScriptura Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/openscriptura/openscriptura}},
  note         = {Apache 2.0. Multi-tradition, language-agnostic pipeline.
                  Benchmarked against CEFEAI AllFaith Benchmark.}
}
```

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Free use including commercial. Attribution required.

---

<div align="center">

**Soli Deo Gloria**

*For the glory of God and the good of the Church — in every tradition and every language.*

[HuggingFace](https://huggingface.co/openscriptura) · [CEFEAI](https://cefe.ai) · [Issues](https://github.com/openscriptura/openscriptura/issues)

</div>
