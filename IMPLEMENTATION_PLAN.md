# OpenScriptura — Implementation Plan
> Consenso dos 539 PhDs. Última revisão: 2026-06-08. Phase 2 complete; Phase 3/4 scripts written; **evaluation protocol upgraded to v2 (re-baseline with system prompt)**.

---

## Evaluation Protocol v2 — Why we re-baseline now (extra cost, extra effort, and worth it)

**TL;DR:** The original Phase 0 baseline measured the raw model with **no system prompt**. The fine-tuned model is trained *and deployed* **with** a Reformed system prompt. To compare them honestly we must put the **same system prompt on both sides** — which means re-running the Phase 0 baseline (~$0.30, ~2h) under the new protocol. We accepted this extra cost because the project's entire scientific claim depends on the comparison being valid.

### The problem with the v1 baseline
Phase 0 (v1) sent each CEFEAI prompt to raw `qwen/qwen3-8b` as a bare user turn — **no system message** → RR 4.7%, CB 19.6%. But every training record carries a canonical Reformed system prompt, and the deployed model always runs with it. That leaves two dishonest options and one honest one:

| Option | What it measures | Verdict |
|--------|------------------|---------|
| Fine-tuned **with** prompt vs baseline **without** prompt (v1) | fine-tuning **+** "tell any model it's Reformed" — two effects mixed | ❌ Conflated; can't attribute the gain to fine-tuning |
| Fine-tuned **without** prompt vs baseline **without** prompt | a config the model was never trained for and will never deploy in | ❌ Understates the real model |
| **Both with the same prompt (v2)** | **only the weights differ** | ✅ Valid, deployment-realistic |

### The cost of doing it right
- **Re-run the Phase 0 baseline** with the system prompt: **~$0.30** in API calls, **~2h** wall-clock.
- **~1 day of engineering** to refactor the eval harness: extract shared primitives into `scripts/utils/cefeai.py`, add the committed `configs/system_prompt.txt`, tag outputs by prompt mode, make `07` compare against the matching-mode baseline.

A cheap-but-invalid improvement number would be worse than no number — it would mislead the arXiv submission and the model card. So we pay the small re-baseline cost to keep the headline claim (*"fine-tuning improved confessional representation by N points"*) defensible.

### What stays the same
`temperature=0.0, seed=42, enable_thinking=False, max_tokens=512` are unchanged. `enable_thinking=False` matches the training format (direct Q→A pairs, no thinking traces). The v1 numbers (4.7% / 19.6%) are preserved and remain the correct baseline for any `--no-system-prompt` run.

### Expectation to set in advance
The v2 RR baseline will be **much higher than 4.7%** — a raw model told "you are a Reformed theological assistant" naturally produces more religious representation. The reported fine-tuning delta is `fine-tuned(v2) − baseline(v2)`, both with the prompt.

---

## Vision

**OpenScriptura** is an open-source initiative to fine-tune LLMs for Protestant theology — starting with the Reformed tradition in PT-BR, designed to scale across traditions and languages.

Naming convention: `openscriptura/{base}-{tradition}-{lang}-{version}`  
First model: `openscriptura/qwen3-8b-reformed-pt-br-v0.1`

---

## Phase 0 — Baseline (Week 1, Day 1–2)
**Goal:** Establish the Qwen3-8B raw baseline on CEFEAI before any fine-tuning.
**Status:** ✅ **v1 complete** (no system prompt) — both benchmarks run, 1,606 prompts, **$0.7902**. 🔄 **v2 re-baseline pending** (same script, `--benchmark both`, system prompt on — see "Protocol v2" above).

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
**Status:** ✅ **Complete** — all 4 experiments run on vast.ai RTX 4090 (instance 40077545, ~6.5h, ~$3.50). Winner: **exp_c (r=64, lr=2e-4)**. Instance destroyed; adapters + results archived locally in `results/`.

### Phase 2 final results

| Config | r | α | LR | eval_all_loss | Tier B | Tier C | best step |
|--------|---|---|----|---------------|--------|--------|-----------|
| **exp_c** | 64 | 128 | 2e-4 | **0.6527** ✅ | 0.6841 | 0.5728 | 350 |
| exp_d | 64 | 128 | 1e-4 | 0.6586 | 0.6870 | 0.5858 | 350 |
| exp_a | 16 | 32 | 2e-4 | ~0.69 | 0.6924 | 0.5910 | — |
| exp_b | 16 | 32 | 1e-4 | 0.6993 | 0.6993 | 0.5990 | 450 |

**Findings (3-panel review of 77 PhDs each):**
- **Rank dominates LR.** r=64 beats r=16 by ~0.04 eval_loss in both LR settings — an unambiguous, consistent signal. The Reformed PT-BR corpus needs the extra adapter capacity.
- **LR effect is marginal within r=64** (exp_c vs exp_d = 0.006, below the ~0.01 training-noise threshold). exp_c and exp_d are statistically near-equivalent; exp_c chosen on the (small) numeric edge, exp_d is the safer-stability fallback.
- **Tier C learns faster than Tier B** (gap ~0.10 in every config) — catechisms are structurally uniform; synthetic Tier B has higher style variance.
- Best step = 350/537 (~65%) for both r=64 configs → the data signal is exhausted before the end; Phase 3 uses early stopping to catch the true optimum.

`configs/final.yaml` encodes the exp_c winner for Phase 3.

### Bugs fixed during Phase 2 (see CLAUDE.md "Lessons Learned" #7–#11)
flash_attn missing → eager · OOM → batch=1 + grad_accum=16 + gradient_checkpointing · `metric_for_best_model` dict-eval naming · `label_names=[]` suppressing eval loss · `load_best_model_at_end` unsafe with QLoRA · chained-run CUDA OOM → `sleep 30` between configs.

### (Historical) live-run context — instance 40077545 (Denmark RTX 4090, $0.455/hr)

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
**Status:** ✅ **Scripts written** (`05_train_final.py`, `06_export.py`, `configs/final.yaml`) — run pending on A100.

**Script: `05_train_final.py`** (config: `configs/final.yaml` = exp_c winner)
- Platform: **A100 80GB** (vast.ai/RunPod, ~$1.80/hr)
- **Full bf16 LoRA** (no 4-bit) — A100 has the VRAM; cleaner adapter merge. (`quantization.enabled: true` reverts to QLoRA for smaller GPUs.)
- r=64, α=128, lr=2e-4, warmup 10% (doubled from Phase 2 for the aggressive LR), 5 epochs
- **Early stopping** (custom PEFT-safe callback, patience=5) — stops at the real optimum; `eval_steps=25` for fine granularity; `save_total_limit=10` so the best checkpoint survives
- `max_seq_length=4096` (A100 headroom); effective batch 16 (bs=4 × grad_accum=4)
- Writes `results.json` recording `best_checkpoint` (the genuinely best step, not the last)
- Estimated time: ~2–3h · Cost: **~$5**

### Export
**Script: `06_export.py`** (`--config configs/final.yaml`)
- Reads `best_checkpoint` from `results.json` (not the last/`final/` state); `--adapter-path` overrides; `--force-merge` re-merges
- Merge LoRA adapter into base via `peft.merge_and_unload()` (single-GPU pinned)
- Export GGUF (via llama.cpp) in 3 quantizations: `Q4_K_M` (balanced), `Q5_K_M` (higher quality), `Q8_0` (near lossless)
- `--push-to-hub` → `openscriptura/qwen3-8b-reformed-pt-br-v0.1` (merged model + `gguf/`)
- Cost: **~$2** (A100 time for merge + GGUF) · ~1–2h

---

## Phase 4 — Evaluation & Publication (Week 3)
**Status:** ✅ **Script written** (`07_cefeai_eval.py`) — run pending (needs Phase 3 model).

**Script: `07_cefeai_eval.py`** — local inference (transformers, greedy, `enable_thinking=False`) + OpenRouter judge.
- **Protocol v2:** evaluates WITH the Reformed system prompt (default); `--no-system-prompt` for the v1 legacy comparison. Compares against the baseline matching its own prompt mode.
- Same locked inference settings as the baseline (`temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`); judge prompts / Wilson CI / system prompt shared via `scripts/utils/cefeai.py`.
- Auto-detects merged model vs PEFT adapter; prints direction-aware verdict (RR up=better, CB down=better).
- **Run order:** v2 re-baseline first (`00_cefeai_baseline.py --benchmark both`), then `07 ... --benchmark both`.

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
│   │   ├── logger.py             ✅ get_logger() (resolves logs/ from project root)
│   │   ├── progress.py           ✅ ProgressBar (TTY-aware)
│   │   ├── report.py             ✅ generate_all_reports()
│   │   └── cefeai.py             ✅ shared judge/Wilson/verdict/system-prompt (protocol v2)
│   ├── 00_setup_data.py          ✅ data staging + audit
│   ├── 00_cefeai_baseline.py     ✅ run (v1 4.7%/19.6%); v2 (--system-prompt default) pending
│   ├── 01_build_tier_c.py        ✅ 839 records produced
│   ├── 02_build_tier_b.py        ✅ 2,129 records produced
│   ├── 03_eda.py                 ✅ reports/eda_report.html produced
│   ├── merge_dataset.py          ✅ train.jsonl (2,873) + eval.jsonl (151)
│   ├── 04_experiment.py          ✅ QLoRA training, YAML-driven, 4 configs (run)
│   ├── vastai_run_experiments.py ✅ vast.ai instance automation
│   ├── 05_train_final.py         ✅ written — full-bf16 LoRA + early stopping (run 🔲)
│   ├── 06_export.py              ✅ written — merge best ckpt + GGUF Q4/Q5/Q8 (run 🔲)
│   └── 07_cefeai_eval.py         ✅ written — v2 eval, local infer + judge (run 🔲)
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
│   ├── exp_c.yaml                ✅ r=64 lr=2e-4 (WINNER)
│   ├── exp_d.yaml                ✅ r=64 lr=1e-4
│   ├── final.yaml               ✅ Phase 3 config (exp_c winner, A100 full bf16)
│   └── system_prompt.txt        ✅ canonical Reformed prompt (committed, single source of truth)
│
├── checkpoints/  (gitignored)
│   ├── exp_a/ exp_b/ exp_c/ exp_d/   ✅ Phase 2 done (archived to results/)
│   └── final/                    🔲 Phase 3 output (merged/ + gguf/ + results.json)
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
| Phase 0: CEFEAI baseline v1 ✅ | OpenRouter | **$0.7902** (RR $0.0729 + CB $0.7173) |
| Phase 1: Dataset Tier B ✅ | DeepSeek API | **~$3.50** (actual; planned $1 — 3.5× over budget due to resume issues) |
| Phase 2: 4 experiments ✅ | Vast.ai RTX 4090 | **~$3.50** (actual: ~6.5h × $0.455/hr + setup) |
| **Protocol v2 re-baseline** 🔄 | OpenRouter | **~$0.30** (extra cost — see "Protocol v2" rationale) |
| Phase 3: Final training + export 🔲 | A100 80GB | ~$5–7 (~3–4h × $1.80/hr) |
| Phase 4: CEFEAI re-eval (v2) 🔲 | OpenRouter (judge only; inference is local/free) | ~$0.30 |
| Misc | — | ~$0.50 |
| **Total (one-time)** | | **~$14–16** |
| HF PRO (demo + ZeroGPU) | HuggingFace | **$9/month** |

> The Protocol v2 re-baseline is a deliberate **extra** line item (~$0.30 + ~1 day eng). It is the price of a valid baseline→fine-tuned comparison; see the "Protocol v2" section at the top for the full rationale.

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
✅  Phase 0 — CEFEAI baseline v1 COMPLETE (no system prompt)
    RR: 4.7% Any Representation (n=150, $0.07) · CB: 19.6% Any Bias (n=1456, $0.72)

✅  Phase 1 — Dataset construction COMPLETE
    Tier C: 839 · Tier B: 2,129 · Merged train 2,873 + eval 151 (seed=42)

✅  Phase 2 — Experiments COMPLETE
    4 configs run on vast.ai RTX 4090; instance destroyed; adapters in results/
    Winner: exp_c (r=64, lr=2e-4) eval_all_loss 0.6527 → configs/final.yaml

✅  Phase 3/4 scripts WRITTEN + reviewed (2× /code-review) + simulated (38/38)
    05_train_final.py · 06_export.py · 07_cefeai_eval.py · utils/cefeai.py

▶  NOW — Protocol v2 re-baseline (the extra step we chose to take)
    python scripts/00_cefeai_baseline.py --benchmark both        # ~$0.30, ~2h, system prompt ON
    → results/baseline_qwen_qwen3_8b_sysprompt_{RR,CB}_summary.json
    WHY: the v1 baseline had no system prompt; the fine-tuned model deploys WITH one.
         Re-baselining keeps the improvement claim valid (only weights differ).

▶  NEXT — Phase 3 on A100 80GB:
    python scripts/05_train_final.py --config configs/final.yaml   # full bf16, early stopping
    python scripts/06_export.py      --config configs/final.yaml --push-to-hub

▶  THEN — Phase 4 (v2 eval) on the same A100:
    python scripts/07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both
    Compares fine-tuned(v2) vs baseline(v2). Also run --no-system-prompt for the v1 delta.
```
