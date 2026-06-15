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
Base models — CEFEAI Religious Representation (June 2026, 150 questions):

  Grok 4.20        ████░░░░░░░░░░░░░░░░  29.3% Any Representation
  Mistral Large    ████░░░░░░░░░░░░░░░░  23.3%
  GPT-5.4          ███░░░░░░░░░░░░░░░░░  17.3%
  Qwen3-8B base    ██░░░░░░░░░░░░░░░░░░  12.7%  ← official CEFE.AI judge (deepseek-v4-flash), 2026-06-09
  Claude Opus 4.7  █░░░░░░░░░░░░░░░░░░░   4.0%
  Llama 4 Scout    ░░░░░░░░░░░░░░░░░░░░   3.3%

  Predominantly Religious: 0% across all 27 models tested (Qwen3-8B base: 0%).

  OpenScriptura v0.1 target:  ████████████░░░░░░░░  > 60%  (+47.3 pp from the 12.7% base)
```

> **Baseline (official CEFE.AI judge, no system prompt, 2026-06-09):** Qwen3-8B raw scores
> **0.1467/4** mean Religious Representation (12.7% "any representation") and **3.6944/7** mean
> Conversion-Bias rating (mild −0.31 lean toward encouraging transitions). 0 parse-errors on
> both. The fine-tuned `v0.1` will be measured against exactly this run (same judge + settings;
> only the weights differ). Absolute numbers are judge-dependent; the rigorous claim is the delta.

> **Two evaluation tracks** (the model is for Brazilian Portuguese, but the science needs an English anchor):
> **(1) English CEFE.AI** — the leaderboard-comparable headline (the "vs the 27 frontier models" claim).
> **(2) pt-BR translation** (`--lang ptbr`) — deployment-realistic, since the model is used in Portuguese;
> rigorous internal baseline→fine-tuned delta, but **NOT** comparable to the English public leaderboard.
> English stays the comparable headline; pt-BR is the product-truth secondary number.

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

| Model | Base | Tradition | Language | CEFEAI RR (baseline) | CEFEAI RR (fine-tuned) | Download |
|---|---|---|---|---|---|---|
| `qwen3-8b-reformed-pt-br-v0.1` | Qwen3-8B | Reformed | PT-BR | pending (official-judge re-run)¹ | 🔄 Phase 3 pending | [HF Hub](https://huggingface.co/openscriptura/qwen3-8b-reformed-pt-br-v0.1) |
| `qwen3-8b-lutheran-en-v0.1` | Qwen3-8B | Lutheran | EN | 📋 planned | — |
| `gpt-oss-20b-v1.0` | GPT-OSS 20B | Multi-tradition | Multilingual | 📋 planned | — |

Model naming convention: `openscriptura/{base}-{tradition}-{lang}-{version}`

¹ Our earlier baseline (RR 4.7% / CB 19.6%) was scored with a **home-grown rubric that does not match CEFE.AI** (we used RR 0–3 / CB 0–3; CEFE.AI is RR **0–4** / CB **1–7**). It is **invalid** for comparison and will be replaced by a re-run with the **official judge** (`configs/cefeai/`). See the Evaluation Protocol section.

**Status:** dataset built (2,968 records) · 4-config LoRA sweep complete (winner **r=64, lr=2e-4**) · Phase 3 + eval scripts written · **switched to the official CEFE.AI judge** — baseline re-run pending.

---

## Evaluation Protocol — headline is **v1 (no system prompt)**

We score with the **official CEFE.AI judge prompts** (`configs/cefeai/*.json`, loaded verbatim — RR 0–4 mean+distribution; CB 1–7 mean by pair/tradition/template) and the **exact upstream questions** (verified identical). CEFEAI comparisons use **no system prompt**, on both the raw baseline and the fine-tuned model — the fine-tuning lives in the weights, so the no-prompt fine-tuned model is compared against the no-prompt raw baseline. We are 100% adherent to everything CEFE.AI documents; the judge model and inference settings are **not** published by CEFE.AI, so we define them by good science (`docs/EVALUATION_PROTOCOL.md`) — our internal delta is rigorous, while absolute numbers are protocol-adherent but judge-dependent.

> ⚠️ Our earlier numbers (RR 4.7% / CB 19.6%) used a non-matching home-grown rubric and are **invalid** — the baseline will be re-run with the official judge.

We also *tested* a "v2" protocol (the same Reformed system prompt on both sides) and **ran it** — then rejected it. The prompt **alone** saturated the *raw* model to **RR 99.3% / CB 87.8%**, which (a) isn't comparable to the prompt-free CEFEAI leaderboard and (b) leaves no headroom to show what fine-tuning added. So v1 is the headline; v2 is retained only as an opt-in `--system-prompt` deployment-behavior datapoint, never a leaderboard number. (Inference settings, fixed throughout and identical on both sides: model `temperature=0.0, seed=42, enable_thinking=False, max_tokens=1024`; judge `temperature=0.0, enable_thinking=False`. Those 99.3% / 87.8% figures are old-rubric and will be regenerated with the official judge.)

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
# Headline: v1, NO system prompt, official judge (CEFEAI-comparable). ~$0.30 | ~2h
python scripts/00_cefeai_baseline.py --benchmark both

# Optional: v2 deployment-behavior datapoint (NOT comparable — prompt saturates the metric)
python scripts/00_cefeai_baseline.py --benchmark both --system-prompt
```

### Build the dataset (Reformed PT-BR v0.1)

```bash
# Tier C — confessions/catechisms (free, no API)
python scripts/01_build_tier_c.py      # → data/tier_c/tier_c.jsonl (839 records)

# Tier B — synthetic Q&A via DeepSeek API (~$3.50)
python scripts/02_build_tier_b.py      # → data/tier_b/tier_b.jsonl (2,129 records)

# EDA — validate corpus quality
python scripts/03_eda.py               # → reports/eda_report.html

# Merge into train/eval splits (stratified 95/5)
python scripts/merge_dataset.py        # → data/merged/train.jsonl (2,873) + eval.jsonl (151)
```

### Run controlled experiments (Phase 2 — complete)

```bash
# 2×2 LoRA sweep on vast.ai RTX 4090. Winner: exp_c (r=64, lr=2e-4).
python scripts/vastai_run_experiments.py --search
python scripts/vastai_run_experiments.py --config configs/exp_c.yaml --all-configs --wait
# Chained runs: put `sleep 30` between configs so the GPU frees between processes.
```

### Final fine-tuning + export (Phase 3) — A100 80GB

```bash
python scripts/05_train_final.py --config configs/final.yaml   # full-bf16 LoRA, early stopping
python scripts/06_export.py      --config configs/final.yaml --push-to-hub   # merge best ckpt + GGUF Q4/Q5/Q8
```

### Re-evaluate on CEFEAI (Phase 4)

```bash
# v1 headline (no system prompt) — compares against the v1 baseline
python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both
```

---

## Dataset Schema (v1.0)

`tradition` and `lang` (BCP 47) are first-class fields. Training records use chat format — this is the schema used in `data/tier_*/*.jsonl`:

```json
{
  "id":               "openscriptura-reformed-pt-0001",
  "version":          "1.0",
  "tradition":        "reformed",
  "lang":             "pt-BR",
  "tier":             "C",
  "source":           "wcf",
  "sha256":           "<SHA-256 over messages + tradition + lang>",
  "messages": [
    {"role": "system",    "content": "<canonical PT-BR Reformed system prompt>"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "confessional_refs": ["WCF 11.1", "WCF 11.2"],
  "quality_score":    96,
  "reviewed_by":      "auto",
  "created_at":       "2026-06-07T00:00:00Z"
}
```

> **Key:** use `lang` (short form, not `language`) — matches `content_hash()` in `scripts/utils/hash.py`. Use `messages` (chat format), not `question`/`answer`.

`tradition` values: `reformed` · `lutheran` · `anglican` · `baptist` · `methodist` · `pentecostal` · `congregationalist`

`lang` follows BCP 47: `pt-BR` · `en` · `es` · `de` · `fr` · etc.

---

## Fine-tuning Configuration

```yaml
# configs/final.yaml — Phase 3 (winner from the 2×2 sweep: exp_c)
model:
  name: Qwen/Qwen3-8B
  attn_implementation: flash_attention_2   # auto-falls back to eager if not installed

quantization:
  enabled: false        # full bf16 on A100 (cleaner merge); set true to revert to QLoRA

lora:
  r: 64
  lora_alpha: 128       # 2 × lora_r (canonical — see VALIDATION_REPORT.md M1)
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  num_train_epochs: 5              # early stopping (patience=5) halts at the optimum
  learning_rate: 2e-4              # winner LR
  lr_scheduler_type: cosine
  warmup_ratio: 0.10               # doubled vs Phase 2 for the aggressive LR
  bf16: true
  max_seq_length: 4096             # A100 headroom
  eval_steps: 25
  save_total_limit: 10             # keep the best checkpoint past early stopping
  seed: 42
```

> Phase 2 ran the 2×2 matrix (r∈{16,64} × lr∈{1e-4,2e-4}); **rank dominated** (r=64 ≫ r=16). `final.yaml` uses the exp_c winner.

**Cost per model variant:** ~$11–16 total (dataset generation + sweep + final train + v2 baseline + eval).

---

## Roadmap

```
Phase 1 — Reformed Foundation
  v0.1  Qwen3-8B    · Reformed  · PT-BR     ← 🔄 sweep done, final train + v2 eval pending
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
