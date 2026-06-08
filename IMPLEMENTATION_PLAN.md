# OpenScriptura — Implementation Plan
> Consenso dos 539 PhDs. Última revisão: 2026-06-07. Atualizado com fontes reais e estado de execução.

---

## Vision

**OpenScriptura** is an open-source initiative to fine-tune LLMs for Protestant theology — starting with the Reformed tradition in PT-BR, designed to scale across traditions and languages.

Naming convention: `openscriptura/{base}-{tradition}-{lang}-{version}`  
First model: `openscriptura/qwen3-8b-reformed-pt-br-v0.1`

---

## Phase 0 — Baseline (Week 1, Day 1–2)
**Goal:** Establish the Qwen3-8B raw baseline on CEFEAI before any fine-tuning.
**Status:** ✅ **Complete** — both benchmarks run, all 1,606 prompts processed. Total cost: **$0.7902**.

### Script: `00_cefeai_baseline.py` ✅
- Run **150 prompts** from CEFEAI Religious Representation benchmark
- Run **1,456 prompts** from CEFEAI Conversion Bias benchmark
- Model: `qwen/qwen3-8b` via OpenRouter API
- Judge: `deepseek/deepseek-v4-flash` (cheap, sufficient)
- Output: `results/baseline_qwen3_8b_RR.jsonl` + `results/baseline_qwen3_8b_CB.jsonl`
- Cost: **~$0.29**
- Time: **~2h**

### Benchmark Input Data (staged at `data/cefeai/`)
| File | Records | Source | Status |
|---|---|---|---|
| `rr_150.jsonl` | 150 | [CEFEAI/allfaith-religious-representation](https://github.com/CEFEAI/allfaith-religious-representation) | ✅ Downloaded |
| `cb_1456.jsonl` | 1,456 | [CEFEAI/allfaith-conversion-bias](https://github.com/CEFEAI/allfaith-conversion-bias) | ✅ Downloaded |

> Note: CEFEAI GitHub files use field `question`; `00_setup_data.py` renames it to `prompt` on download to match `00_cefeai_baseline.py`'s expected schema.

**Actual baseline — Religious Representation (RR), 2026-06-07):**
| Metric | Result | 95% CI | n |
|---|---|---|---|
| No Representation | **95.3%** | [90.7%, 97.7%] | 143 |
| Any Representation | **4.7%** | [2.3%, 9.3%] | 7 |
| Meaningful Reference | **0.0%** | [0.0%, 2.5%] | 0 |
| Predominantly Religious | **0.0%** | [0.0%, 2.5%] | 0 |

Cost: **$0.0729** · Model: `qwen/qwen3-8b` · N=150

**CEFEAI Leaderboard context (Any Representation — RR):**
| Model | Score |
|---|---|
| Grok 4.20 | 29.3% |
| Mistral Large | 23.3% |
| GPT-5.4 | 17.3% |
| **Qwen3-8B (this run — baseline)** | **4.7%** |
| Claude Opus 4.7 | 4.0% |
| Llama 4 Scout | 3.3% |

Gap to v0.1 target (>60% Any Representation): **+55.3 pp**

**Actual baseline — Conversion Bias (CB), 2026-06-07:**
| Metric | Result | 95% CI | n |
|---|---|---|---|
| No Bias | **80.4%** | [78.2%, 82.3%] | 1,170 |
| Any Bias | **19.6%** | [17.7%, 21.8%] | 286 |
| Meaningful Bias | **5.2%** | [4.2%, 6.5%] | 76 |
| Predominantly Religious | **0.0%** | [0.0%, 0.3%] | 0 |

Cost: **$0.7173** · Model: `qwen/qwen3-8b` · N=1,456 (10 originally skipped, recovered via `--resume`)

**Note on Conversion Bias:** OpenScriptura will intentionally score "biased" toward the confessional tradition being evaluated — this is documented as *explicit confessional bias*, not a flaw. The fine-tuned model is expected to show significantly higher "Any Bias" than the 19.6% raw baseline.

---

## Phase 1 — Dataset Construction (Week 1, Day 2–5)
**Goal:** Build the training corpus across 3 tiers.

### Tier C — Catechisms & Confessions (FREE, no API)
**Status:** ✅ **Complete** — 839 unique records in `data/tier_c/tier_c.jsonl`

**Script: `01_build_tier_c.py`** ✅
- Input: Plain-text confessional documents in `data/sources/confessions/`
- Output: `data/tier_c/tier_c.jsonl` — 839 records; `data/tier_c/manifest.json`
- Deduplication: exact hash (SHA-256 via `content_hash()`); 3 duplicates removed
- Cost: **$0**

#### Source Documents

| Document | File | Records | Source | How Obtained |
|---|---|---|---|---|
| Westminster Shorter Catechism | `westminster_shorter_catechism.txt` | 106 | IPB / IPCPA | PDF extracted — [ipcpa.org.br](https://ipcpa.org.br/wp-content/uploads/2024/10/Breve_Catecismo_de_Westminster.pdf) |
| Westminster Larger Catechism | `westminster_larger_catechism.txt` | 193 | IPB / IPCPA | PDF extracted — [ipcpa.org.br](https://ipcpa.org.br/wp-content/uploads/2024/10/Catecismo_Maior_de_Westminster.pdf) |
| Westminster Confession of Faith 1647 | `wcf_1647.txt` | 173 | IPB / IPCPA | PDF extracted — [ipcpa.org.br](https://ipcpa.org.br/wp-content/uploads/2024/10/A_Confissao_de_Fe_de_Westminster.pdf) |
| Heidelberg Catechism | `heidelberg_catechism.txt` | 128 | Ligonier PT-BR | Scraped — [pt.ligonier.org](https://pt.ligonier.org/recursos/credos-e-confissoes/o-catecismo-de-heidelberg/) |
| Canons of Dort | `canons_of_dort.txt` | 87 | Ligonier PT-BR | Scraped — [pt.ligonier.org](https://pt.ligonier.org/recursos/credos-e-confissoes/os-canones-de-dort/) |
| London Baptist Confession 1689 | `lcf_1689.txt` | 155 | Ligonier PT-BR | Scraped — [pt.ligonier.org](https://pt.ligonier.org/recursos/credos-e-confissoes/a-confissao-de-fe-batista-de-londres-de-1689/) |
| **Total** | | **842 raw → 839 unique** | | |

> **Translation note:** IPCPA PDFs (WSC, WLC, WCF) use the IPB register — an older but doctrinally accurate Portuguese translation. Ligonier PT-BR sources (Heidelberg, Dort, LCF 1689) are contemporary Brazilian Portuguese. Both accepted; pastoral council review not required for confessional documents.

> **Reproducibility:** Original PDFs preserved at `data/sources/confessions/` alongside extracted `.txt` files. Manifested with SHA-256 in `data/sources/manifest.json`.

### Tier B — Synthetic Data (API-generated)
**Status:** ⏳ Script not yet written (`02_build_tier_b.py`)

Sources: Spurgeon sermons (quality score ≥ 93), Monergismo Reformed ebooks, WCF Q&A expansion

**Script: `02_build_tier_b.py`** ⏳
- Generator: `deepseek/deepseek-v4-flash` (12% religious representation — best available)
- Annotator/filter: `deepseek/deepseek-v4-pro` (highest quality judge)
- Target: ~3,000–4,000 Q&A pairs
- Quality threshold: score ≥ 93 (M7 rubric: 40pts theology + 20pts pastoral clarity + 20pts PT-BR + 20pts no-hallucination)
- Confessional score threshold: ≥ 0.85 (M13 rubric)
- Cost: **~$1**

#### Source Material for Tier B

| Source | Location | Volume | Notes |
|---|---|---|---|
| Spurgeon sermons (PT-BR) | `data/sources/spurgeon/` | 105 `.pt-br.md` files + 105 `_avaliacao.json` quality scores | Copied from `pastor-ai` staging; use only those with `quality_score ≥ 93` |
| Monergismo Reformed ebooks | `data/sources/monergismo/` | 289 PDFs from 34 authors | Source: [monergismo.com](https://monergismo.com) |

##### Authors Excluded from Monergismo Corpus (non-Reformed / off-topic)

| Author | Reason |
|---|---|
| John Wesley | Arminian — incompatible theological tradition |
| Karl Barth | Neo-orthodox — diverges from WCF on election and Scripture |
| F.A. Hayek | Economist — not a theologian, off-topic |
| Hermas | Early Church (*Shepherd of Hermas*) — pre-Reformation, not confessionally Reformed |
| Inácio de Antioquia | Church Father (Ignatius) — patristic, not confessionally Reformed |

> Pastoral council may restore any excluded author after review.

### Tier A — Curated High-Quality (manual review)
**Status:** ⏳ Pending pastoral council

- Pastoral review mandatory before inclusion (see `docs/PASTORAL_REVIEW_PROTOCOL.md`)
- Each example reviewed against WCF > Dort > LCF 1689 confessional hierarchy
- Includes refusal examples (documented as positive training signal)
- Target: ~500–1,000 examples
- Cost: **$0** (human time)

### JSONL Schema (versioned, chat-format)

All training records use this schema. `lang` (BCP 47, short form) and `messages` are the keys used by `content_hash()` in `scripts/utils/hash.py`.

```json
{
  "id": "openscriptura-reformed-pt-0001",
  "version": "1.0",
  "tradition": "reformed",
  "lang": "pt-BR",
  "tier": "C",
  "source": "wcf",
  "sha256": "<content_hash over messages+tradition+lang>",
  "messages": [
    {"role": "system", "content": "<canonical PT-BR system prompt from THEOLOGICAL_STATEMENT.md>"},
    {"role": "user",   "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "confessional_refs": ["WCF 1.1", "WCF 1.4"],
  "reviewed_by": "auto",
  "quality_score": null,
  "created_at": "2026-06-07T00:00:00Z"
}
```

> **Key constraint:** use `lang` (not `language`) to match `content_hash()`. Use `messages` (chat-format) not `question`/`answer` fields.

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
**Status:** ⏳ Awaiting Phase 1 completion.
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
**Status:** ⏳ Awaiting Phase 2 completion.

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
**Status:** ⏳ Awaiting Phase 3 completion.

Rerun `00_cefeai_baseline.py` with trained model = model #29.

### Evaluation targets (Religious Representation)
| Metric | Baseline — Qwen3-8B raw | Target v0.1 | Target v1.0 |
|---|---|---|---|
| No Representation | **95.3%** [90.7–97.7%] | ~30–40% | ~15% |
| Any Representation | **4.7%** [2.3–9.3%] | **~60–70%** | ~80% |
| Meaningful Reference | **0.0%** [0.0–2.5%] | ~20–30% | ~40% |
| Predominantly Religious | **0.0%** [0.0–2.5%] | ~5–10% | ~15% |

**Target:** Beat Grok 4.20 (29.3%) by wide margin — v0.1 aims for >60% Any Representation (+55.3 pp lift from baseline).

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
├── CONTRIBUTING.md               ✅ present
├── LICENSE                       ✅ Apache 2.0
├── requirements.txt              ✅ pinned
│
├── scripts/
│   ├── utils/                    ✅ shared infrastructure
│   │   ├── api_client.py         ✅ OpenRouterClient (retry, cost, think-stripping)
│   │   ├── cost_tracker.py       ✅ CostTracker (hard stop)
│   │   ├── hash.py               ✅ content_hash() SHA-256
│   │   ├── logger.py             ✅ get_logger()
│   │   ├── progress.py           ✅ ProgressBar (TTY-aware)
│   │   └── report.py             ✅ generate_all_reports()
│   ├── 00_setup_data.py          ✅ data staging + audit
│   ├── 00_cefeai_baseline.py     ✅ implemented + run
│   ├── 01_build_tier_c.py        ✅ 839 records produced
│   ├── 02_build_tier_b.py        ⏳ not yet written
│   ├── 03_eda.py                 ⏳ not yet written
│   ├── 04_experiment.py          ⏳ not yet written
│   ├── 05_train_final.py         ⏳ not yet written
│   └── 06_export.py              ⏳ not yet written
│
├── data/
│   ├── cefeai/
│   │   ├── rr_150.jsonl          ✅ 150 prompts
│   │   └── cb_1456.jsonl         ✅ 1,456 prompts
│   ├── sources/
│   │   ├── confessions/          ✅ 6 docs (3 PDFs + 6 TXTs)
│   │   ├── spurgeon/             ✅ 105 PT-BR sermons + quality scores
│   │   ├── monergismo/           ✅ 289 PDFs / 34 authors (5 excluded)
│   │   └── manifest.json         ✅ SHA-256 index of all sources
│   ├── tier_a/                   ⏳ pending pastoral council
│   ├── tier_b/                   ⏳ pending 02_build_tier_b.py
│   └── tier_c/
│       ├── tier_c.jsonl          ✅ 839 records
│       └── manifest.json         ✅
│
├── configs/
│   └── exp_d.yaml                ⏳ not yet written
│
├── tests/
│   ├── test_cost_tracker.py      ✅
│   └── test_hash.py              ✅
│
├── results/                      ✅ baseline results present (RR + CB, 4 files each)
├── reports/                      ⏳ generated by 03_eda.py
│
└── docs/
    ├── THEOLOGICAL_STATEMENT.md  ✅ present
    ├── PASTORAL_REVIEW_PROTOCOL.md ✅ present
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
| Phase 0: CEFEAI baseline ✅ | OpenRouter | **$0.7902** (RR $0.0729 + CB $0.7173) |
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

## Immediate Next Actions

```
▶  Step 1 — Phase 0 baseline ✅ COMPLETE
   RR: 4.7% Any Representation (n=150, $0.07)
   CB: 19.6% Any Bias (n=1456, $0.72)

▶  Step 2 — Write 02_build_tier_b.py
   Synthetic Q&A from Spurgeon + Monergismo via DeepSeek API
   Cost: ~$1  |  Output: data/tier_b/tier_b.jsonl (~3,000–4,000 records)

▶  Step 3 — Write 03_eda.py + run EDA
   Validate corpus quality before GPU spend
   Cost: $0  |  Output: reports/eda_report.html
```
