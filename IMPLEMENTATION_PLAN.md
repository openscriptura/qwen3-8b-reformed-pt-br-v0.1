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
**Status:** ✅ **Complete** — 2,129 unique records in `data/tier_b/tier_b.jsonl`

**Script: `02_build_tier_b.py`** ✅
- Generator: `deepseek/deepseek-v4-flash`
- Judge/filter: `deepseek/deepseek-v4-flash` (same model — self-grading tradeoff accepted for v0.1)
- Actual output: **2,129 records** from 1,277/1,277 chunks (180 sources)
- Quality threshold: ≥ 93/100 · Mean score: **96.1** · Min: **93.0**
- Acceptance rate: **~54%** (46% rejected below threshold or parse errors)
- Total cost: **~$3.50** across multiple resume runs
- Chunk-level checkpointing: `data/tier_b/done_chunks.txt` (1,277 entries)

> **Lesson learned:** Resume mechanism initially used record-level SHA-256 dedup, which failed because `temperature=0.7` generates different content each run. Fixed by adding chunk-level `done_chunks.txt` checkpoint. Chunks are marked done immediately after processing regardless of acceptance.

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
**Status:** 🔄 **In progress** — 4 experiments running on vast.ai instance 40077545 (Denmark RTX 4090, $0.455/hr).

### Script: `03_eda.py` ✅
- Output: `reports/eda_report.html` + `reports/eda_report.md`
- Results: 2,968 records · 0 duplicates · 99.5% confessional refs · Tier B mean score 96.1

### Script: `merge_dataset.py` ✅
- Stratified 95/5 split by tier (seed=42, deterministic)
- Output: `data/merged/train.jsonl` (2,873) + `data/merged/eval.jsonl` (151)
- `data/merged/manifest.json` with SHA-256 file hashes
- **Note:** `data/` is gitignored — must upload to GPU instance manually via `scp`

### 4 Controlled Experiments (2×2 matrix) 🔄
Run on **vast.ai RTX 4090** ($0.455/hr, Denmark, CUDA 13.1):

| Config | r | α | LR | Est. time | Note |
|--------|---|---|----|-----------|------|
| A | 16 | 32 | 2e-4 | ~1.5h | Lower bound |
| B | 16 | 32 | 1e-4 | ~1.5h | Conservative low-rank |
| C | 64 | 128 | 2e-4 | ~2.0h | High-rank aggressive |
| **D** | **64** | **128** | **1e-4** | **~2.0h** | **RECOMMENDED — runs first** |

**Script: `04_experiment.py`** ✅ — YAML-driven, all 4 configs, `--dry-run` + `--resume`

Key design decisions validated by 6×77 PhD panels:
- `SFTConfig` built directly from YAML (no `TrainingArguments` middleman)
- Chat template applied upfront → `text` column (not `formatting_func`)
- `eval_dataset` as dict → logs `eval_B_loss`, `eval_C_loss`, `eval_all_loss` separately
- Overlong records tokenized + dropped before training (not silently truncated)
- `prepare_model_for_kbit_training()` handles `enable_input_require_grads()` internally

**vast.ai automation: `scripts/vastai_run_experiments.py`** ✅
- Search, launch, monitor, destroy instances via CLI
- `--all-configs` runs D→C→B→A sequentially with `nohup`
- `--wait` polls until instance reaches `running` status

Cost: ~7h × $0.455 = **~$3.19** (all 4 experiments)

#### vast.ai Instance Setup (required on each new instance)
```bash
# Fix bitsandbytes CUDA mismatch (CUDA 12.4 container on CUDA 13.x host):
ln -sf /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvJitLink.so.12 \
       /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvJitLink.so.13
echo "/usr/local/cuda-12.4/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda124.conf && ldconfig
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Install core deps (skip unsloth for Phase 2):
pip install transformers==4.51.0 trl==0.12.0 peft==0.13.0 bitsandbytes \
  datasets==3.2.0 pyyaml==6.0.2 python-dotenv==1.0.1 scipy==1.14.1 \
  sentencepiece==0.2.0 tokenizers==0.21.0 tenacity==9.0.0 httpx==0.27.0 jsonlines==4.0.0

# Upload data from local machine:
scp -P <port> -i ~/.ssh/id_rsa data/merged/train.jsonl root@<host>:/workspace/openscriptura/data/merged/
scp -P <port> -i ~/.ssh/id_rsa data/merged/eval.jsonl  root@<host>:/workspace/openscriptura/data/merged/
```

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
├── IMPLEMENTATION_PLAN.md        ✅ this file (updated 2026-06-08)
├── CONTRIBUTING.md               ✅ present
├── LICENSE                       ✅ Apache 2.0
├── requirements.txt              ✅ pinned (unsloth==2025.3.19)
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
│   ├── 00_cefeai_baseline.py     ✅ implemented + run (RR 4.7%, CB 19.6%)
│   ├── 01_build_tier_c.py        ✅ 839 records produced
│   ├── 02_build_tier_b.py        ✅ 2,129 records produced
│   ├── 03_eda.py                 ✅ reports/eda_report.html produced
│   ├── merge_dataset.py          ✅ train.jsonl (2,873) + eval.jsonl (151)
│   ├── 04_experiment.py          ✅ QLoRA training, YAML-driven, 4 configs
│   ├── vastai_run_experiments.py ✅ vast.ai instance automation
│   ├── 05_train_final.py         ⏳ not yet written
│   └── 06_export.py              ⏳ not yet written
│
├── data/  (gitignored — not in repo)
│   ├── cefeai/
│   │   ├── rr_150.jsonl          ✅ 150 prompts
│   │   └── cb_1456.jsonl         ✅ 1,456 prompts
│   ├── sources/
│   │   ├── confessions/          ✅ 6 docs (3 PDFs + 6 TXTs)
│   │   ├── spurgeon/             ✅ 105 PT-BR sermons + quality scores
│   │   ├── monergismo/           ✅ 289 PDFs / 34 authors (5 excluded)
│   │   └── manifest.json         ✅ SHA-256 index of all sources
│   ├── tier_a/                   ⏳ pending pastoral council
│   ├── tier_b/
│   │   ├── tier_b.jsonl          ✅ 2,129 records
│   │   ├── manifest.json         ✅
│   │   └── done_chunks.txt       ✅ 1,277 chunk checkpoints
│   ├── tier_c/
│   │   ├── tier_c.jsonl          ✅ 839 records
│   │   └── manifest.json         ✅
│   └── merged/
│       ├── train.jsonl           ✅ 2,873 records (B:2076 + C:797)
│       ├── eval.jsonl            ✅ 151 records (B:109 + C:42)
│       └── manifest.json         ✅ SHA-256 + tier breakdown
│
├── configs/
│   ├── exp_a.yaml                ✅ r=16 lr=2e-4
│   ├── exp_b.yaml                ✅ r=16 lr=1e-4
│   ├── exp_c.yaml                ✅ r=64 lr=2e-4
│   └── exp_d.yaml                ✅ r=64 lr=1e-4 (RECOMMENDED)
│
├── checkpoints/  (gitignored)
│   ├── exp_a/                    ⏳ training
│   ├── exp_b/                    ⏳ training
│   ├── exp_c/                    ⏳ training
│   └── exp_d/                    🔄 training (first)
│
├── tests/
│   ├── test_cost_tracker.py      ✅
│   └── test_hash.py              ✅
│
├── results/                      ✅ baseline results present (RR + CB, 4 files each)
├── reports/                      ✅ eda_report.html + eda_report.md
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
| Phase 1: Dataset Tier B ✅ | DeepSeek API | **~$3.50** (actual; planned $1 — 3.5× over budget due to resume issues) |
| Phase 2: 4 experiments 🔄 | Vast.ai RTX 4090 | **~$3.19** (actual: 7h × $0.455/hr) |
| Phase 3: Final training | A100 80GB | ~$7.20 (4h × $1.80/hr — RunPod or vast.ai) |
| Phase 4: CEFEAI re-eval | OpenRouter | ~$0.79 (same as baseline) |
| Misc | — | ~$0.50 |
| **Total (one-time)** | | **~$16–18** (slightly over $15 budget) |
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
✅  Phase 0 — CEFEAI baseline COMPLETE
    RR: 4.7% Any Representation (n=150, $0.07)
    CB: 19.6% Any Bias (n=1456, $0.72)

✅  Phase 1 — Dataset construction COMPLETE
    Tier C: 839 records (confessions/catechisms, $0)
    Tier B: 2,129 records (synthetic, ~$3.50)
    Merged: train.jsonl (2,873) + eval.jsonl (151), seed=42

🔄  Phase 2 — Experiments RUNNING
    vast.ai instance 40077545 (ssh9.vast.ai:37544)
    Configs D→C→B→A sequentially, nohup PID 1844
    Monitor: ssh -p 37544 root@ssh9.vast.ai 'tail -f /workspace/training*.log'
    Expected completion: ~7h from 13:28 UTC 2026-06-08

▶  Step next — When Phase 2 completes:
    1. scp checkpoints/exp_*/results.json from vast.ai
    2. Compare eval_all_loss (+ eval_B_loss, eval_C_loss) across A/B/C/D
    3. Select winner config (expected: D)
    4. Destroy vast.ai instance: vastai destroy instance 40077545

▶  Step after — Phase 3: Write 05_train_final.py + 06_export.py
    Platform: A100 80GB (vast.ai or RunPod, ~$1.80/hr)
    05_train_final.py: full run with winning config
    06_export.py: merge adapter → GGUF Q4/Q5/Q8 → push to HuggingFace

▶  Step after — Phase 4: Write 07_cefeai_eval.py
    Same protocol as 00_cefeai_baseline.py (temperature=0.0, seed=42, enable_thinking=False)
    Target: >60% Any Representation (vs 4.7% baseline)
```
