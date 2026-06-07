# OpenScriptura — Implementation Plan
> Consenso dos 539 PhDs. Última revisão: 2026-06-07.

---

## Vision

**OpenScriptura** is an open-source initiative to fine-tune LLMs for Protestant theology — starting with the Reformed tradition in PT-BR, designed to scale across traditions and languages.

Naming convention: `openscriptura/{base}-{tradition}-{lang}-{version}`  
First model: `openscriptura/qwen3-8b-reformed-pt-br-v0.1`

---

## Phase 0 — Baseline (Week 1, Day 1–2)
**Goal:** Establish the Qwen3-8B raw baseline on CEFEAI before any fine-tuning.

### Script: `00_cefeai_baseline.py`
- Run **150 prompts** from CEFEAI Religious Representation benchmark
- Run **1,456 prompts** from CEFEAI Conversion Bias benchmark
- Model: `qwen/qwen3-8b` via OpenRouter API
- Judge: `deepseek/deepseek-v4-flash` (cheap, sufficient)
- Output: `results/baseline_qwen3_8b_RR.jsonl` + `results/baseline_qwen3_8b_CB.jsonl`
- Cost: **~$0.29**
- Time: **~2h**

**Expected baseline (Religious Representation):**
| Metric | Expected |
|---|---|
| No Representation | ~94% |
| Any Representation | ~6% |
| Meaningful Reference | ~0.7% |
| Predominantly Religious | 0% |

**Note on Conversion Bias:** OpenScriptura will intentionally score "biased" toward the confessional tradition being evaluated — this is documented as *explicit confessional bias*, not a flaw.

---

## Phase 1 — Dataset Construction (Week 1, Day 2–5)
**Goal:** Build the training corpus across 3 tiers.

### Tier C — Catechisms & Confessions (FREE, no API)
Primary sources:
- Westminster Confession of Faith (WCF)
- Heidelberg Catechism
- Westminster Shorter & Larger Catechisms
- London Baptist Confession 1689 (LCF)
- Canons of Dort

**Script: `01_build_tier_c.py`**
- Input: PDF/TXT of confessions
- Output: `data/tier_c/` — JSONL with schema below
- Deduplication: hash-level (exact) + embedding similarity threshold 0.92
- Target: ~1,000–2,000 Q&A pairs
- Cost: **$0**

### Tier B — Synthetic Data (API-generated)
Sources: Spurgeon sermons (quality score ≥ 93), theological commentary, WCF Q&A expansion

**Script: `02_build_tier_b.py`**
- Generator: `deepseek/deepseek-v4-flash` (12% religious representation — best available)
- Annotator/filter: `deepseek/deepseek-v4-pro` (highest quality judge)
- Target: ~3,000–4,000 Q&A pairs
- Quality threshold: score ≥ 93 (Spurgeon standard)
- Cost: **~$1**

### Tier A — Curated High-Quality (manual review)
- Pastoral review mandatory before inclusion
- Each example reviewed against WCF > Dort > LCF 1689 hierarchy
- Includes refusal examples (documented as positive training signal)
- Target: ~500–1,000 examples
- Cost: **$0** (human time)

### JSONL Schema (versioned)
```json
{
  "id": "openscriptura-reformed-pt-0001",
  "version": "1.0",
  "tradition": "reformed",
  "lang": "pt",
  "tier": "C",
  "source": "WCF_chapter_1",
  "sha256": "<hash_of_content>",
  "messages": [
    {"role": "system", "content": "<canonical_system_prompt>"},
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "confessional_refs": ["WCF 1.1", "WCF 1.4"],
  "reviewed_by": "pastoral_council",
  "quality_score": 95
}
```

### `manifest.json` structure
```json
{
  "version": "1.0",
  "created": "2026-06-07",
  "tradition": "reformed",
  "lang": "pt",
  "splits": {
    "train": {"count": 4500, "sha256": "<hash>"},
    "val":   {"count": 300,  "sha256": "<hash>"},
    "test":  {"count": 200,  "sha256": "<hash>"}
  },
  "tier_breakdown": {"A": 700, "B": 3500, "C": 800}
}
```

---

## Phase 2 — EDA + Controlled Experiments (Week 2)
**Goal:** Validate data quality before committing GPU budget.

### Script: `03_eda.py`
- Length distribution of Q&A pairs
- Tier breakdown visualization
- Confessional reference coverage map
- Deduplication report
- Output: `reports/eda_report.html`

### 4 Controlled Experiments (2×2 matrix)
Run on **Vast.ai RTX 4090** (~$0.35/hr):

| Experiment | LR | Rank | Note |
|---|---|---|---|
| A | 2e-4 | 32 | Baseline LoRA |
| B | 2e-4 | 64 | More capacity |
| C | 1e-4 | 32 | More conservative |
| **D** | **1e-4** | **64** | **Recommended (Config D)** |

**Script: `04_experiment.py`** — parameterized, runs any of A/B/C/D

QLoRA config (Config D):
```yaml
base_model: Qwen/Qwen3-8B
lora_r: 64
lora_alpha: 128    # canonical: 2 × lora_r (see VALIDATION_REPORT.md M1)
lora_dropout: 0.05
learning_rate: 1e-4
batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
max_seq_length: 2048
seed: 42
bf16: true
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Cost per experiment: **~$1.25** (Vast.ai RTX 4090 × ~3.5h)  
Total 4 experiments: **~$5**

---

## Phase 3 — Final Fine-Tuning (Week 2–3)

**Script: `05_train_final.py`**
- Platform: **RunPod Secure A100 80GB** ($1.07/hr)
- Config: winner from Phase 2 (expected: Config D)
- Full dataset: ~5,000 examples
- Estimated time: ~4h
- Cost: **~$4.28**

### Export
**Script: `06_export.py`**
- Merge LoRA adapter into base model
- Export GGUF in 3 quantizations:
  - `Q4_K_M` — balanced (recommended for pastor-ai)
  - `Q5_K_M` — higher quality
  - `Q8_0` — near lossless
- Upload to HuggingFace Hub: `openscriptura/qwen3-8b-reformed-pt-v0.1`
- Upload adapter separately: `openscriptura/qwen3-8b-reformed-pt-v0.1-adapter`

---

## Phase 4 — Evaluation & Publication (Week 3)

Rerun `00_cefeai_baseline.py` with trained model = model #29.

### Evaluation targets (Religious Representation)
| Metric | Baseline (#28) | Target v0.1 (#29) | Target v1.0 |
|---|---|---|---|
| No Representation | ~94% | ~30–40% | ~15% |
| Any Representation | ~6% | **~60–70%** | ~80% |
| Meaningful Reference | ~0.7% | ~20–30% | ~40% |
| Predominantly Religious | 0% | ~5–10% | ~15% |

**Target:** Beat Grok 4.20 (current CEFEAI leader at 29.3%) by wide margin.

### Statistical Protocol
- z-test for proportions (N=150, power > 0.99)
- Bonferroni correction for multiple comparisons
- 10 canonical pastoral evaluation questions (qualitative layer)
- Pastoral review of 50 random outputs before publication

### Publication deliverables
- HuggingFace Hub: model + dataset + model card
- arXiv: "OpenScriptura: Open LLMs for Protestant Theology"
- CEFEAI leaderboard submission (model #29)
- 4 ablation studies planned post v0.1

---

## Repository Structure

```
C:\tmp\openScriptura\
│
├── README_github.md              ✅ published
├── README_huggingface.md         ✅ published
├── IMPLEMENTATION_PLAN.md        ✅ this file
├── CONTRIBUTING.md               ⏳ pending
├── manifest.json                 ⏳ pending
├── requirements.txt              ⏳ pending
│
├── scripts/
│   ├── 00_cefeai_baseline.py     ⏳ NEXT
│   ├── 01_build_tier_c.py        ⏳ pending
│   ├── 02_build_tier_b.py        ⏳ pending
│   ├── 03_eda.py                 ⏳ pending
│   ├── 04_experiment.py          ⏳ pending
│   ├── 05_train_final.py         ⏳ pending
│   └── 06_export.py              ⏳ pending
│
├── data/
│   ├── tier_a/
│   ├── tier_b/
│   └── tier_c/
│
├── configs/
│   └── exp_d.yaml
│
├── results/
│   ├── baseline_qwen3_8b_RR.jsonl
│   └── baseline_qwen3_8b_CB.jsonl
│
├── reports/
│   └── eda_report.html
│
└── docs/
    ├── THEOLOGICAL_STATEMENT.md  ⏳ pending
    ├── METHODOLOGY.md            ⏳ pending
    ├── ADDING_A_TRADITION.md     ⏳ pending
    └── ADDING_A_LANGUAGE.md      ⏳ pending
```

---

## Confessional Hierarchy (Reformed v0.1)

```
WCF (Westminster Confession of Faith)
  └── Dort (Canons of Dort)
       └── LCF 1689 (London Baptist Confession)
```

The model does NOT blend traditions — a Reformed model speaks Reformed only.

---

## Cost Summary

| Item | Platform | Cost |
|---|---|---|
| Phase 0: CEFEAI baseline | OpenRouter | $0.29 |
| Phase 1: Dataset Tier B | DeepSeek API | ~$1.00 |
| Phase 2: 4 experiments | Vast.ai RTX 4090 | ~$5.00 |
| Phase 3: Final training | RunPod A100 | ~$4.28 |
| Misc | — | ~$0.50 |
| **Total (one-time)** | | **~$11–15** |
| HF PRO (demo + ZeroGPU) | HuggingFace | **$9/month** |

---

## Tradition Roadmap

| Tradition | Status | Primary Confessions |
|---|---|---|
| **Reformed** | 🔄 **Active — v0.1** | WCF, Dort, LCF 1689 |
| Lutheran | ⏳ v0.2 | Augsburg Confession, Luther's Catechisms |
| Anglican | ⏳ v0.3 | 39 Articles, Book of Common Prayer |
| Baptist (Traditional) | ⏳ v0.4 | LCF 1689, BF&M 2000 |
| Methodist/Wesleyan | ⏳ v0.5 | Articles of Religion, Wesley's Sermons |
| Pentecostal | ⏳ v0.6 | AG Statement of Fundamental Truths |
| Congregationalist | ⏳ v0.7 | Savoy Declaration |

---

## Immediate Next Action

```
▶  scripts/00_cefeai_baseline.py
   Cost: $0.29  |  Time: ~2h  |  Unblocks: everything
```
