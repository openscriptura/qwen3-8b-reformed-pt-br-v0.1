# Changelog & Decision Log — OpenScriptura

Chronological record of what was built, what changed, and **why** — through 2026-06-08.
Lessons learned live in [`CLAUDE.md`](CLAUDE.md#lessons-learned-vastai--gpu-setup) (#1–#15); this file
is the narrative + decision rationale. Phase detail is in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## ⭐ Headline decision — Evaluation Protocol v2 (re-baseline with the system prompt)

**Date:** 2026-06-08 · **Status:** adopted; v2 baseline run pending.

**What changed.** Evaluation now puts the **same Reformed system prompt on both sides** of the
comparison — the raw baseline *and* the fine-tuned model — instead of the v1 setup where the
baseline had no system prompt at all.

**Why now, given it costs more.** The v1 baseline (RR 4.7% / CB 19.6%) measured raw Qwen3-8B with a
bare user prompt. But the fine-tuned model is trained and deployed **with** a Reformed system prompt.
Comparing fine-tuned-*with*-prompt against baseline-*without*-prompt would blend two effects — the
fine-tuning, and simply telling any model "you are a Reformed assistant" — and we could not honestly
credit the gain to fine-tuning. Evaluating the fine-tuned model *without* its prompt is equally wrong:
it tests a configuration the model was never trained for and will never ship in.

**The price we accepted.** Re-running the Phase 0 baseline under the new protocol costs **~$0.30** in
API calls and **~2h** wall-clock, plus **~1 day of engineering** to refactor the harness (shared
`utils/cefeai.py`, committed `configs/system_prompt.txt`, prompt-mode-tagged outputs). We judged a
cheap-but-invalid number to be *worse* than no number — the arXiv claim and the model card both
depend on the comparison being valid. So we pay the small cost to keep
"fine-tuning improved confessional representation by N points" defensible.

**What we kept.** `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512` — unchanged.
`enable_thinking=False` matches the training format (direct Q→A pairs, no thinking traces).
The v1 numbers are preserved and reproducible via `--no-system-prompt`.

**Expectation.** The v2 RR baseline will be **well above 4.7%** (a prompted raw model represents
religion far more than an unprompted one). The reported delta is `fine-tuned(v2) − baseline(v2)`.

---

## Phase 0 — CEFEAI baseline ✅ (v1)
- `00_cefeai_baseline.py`: raw `qwen/qwen3-8b` via OpenRouter, DeepSeek judge, Wilson CIs, tenacity
  retry, cost guardrail, `--resume`.
- **Results (v1, no system prompt):** RR **4.7%** Any Representation (n=150); CB **19.6%** Any Bias
  (n=1,456). Cost **$0.79**.
- Blocking items M2/M3/M4/M5/M8/M10/M11/M12 resolved (see CLAUDE.md "Phase 0 Blocking Items").

## Phase 1 — Dataset construction ✅
- **Tier C** (`01_build_tier_c.py`): 839 records from WCF, Westminster catechisms, Heidelberg, Dort,
  LCF 1689. Cost $0.
- **Tier B** (`02_build_tier_b.py`): 2,129 synthetic records (DeepSeek generate + judge ≥93/100,
  mean 96.1). Cost ~$3.50. *Lesson:* record-level SHA dedup failed under `temperature=0.7`; fixed with
  chunk-level `done_chunks.txt` checkpoint.
- **EDA** (`03_eda.py`) + **merge** (`merge_dataset.py`): 2,968 records → **2,873 train / 151 eval**
  (stratified 95/5, seed=42).

## Phase 2 — Controlled experiments ✅
- `04_experiment.py` (YAML-driven QLoRA) ran the full **2×2 matrix** on vast.ai RTX 4090 (~6.5h, ~$3.50).
- **Winner: exp_c (r=64, lr=2e-4)** — eval_all_loss **0.6527** @ step 350.

  | Config | r | LR | eval_all_loss |
  |--------|---|----|---------------|
  | **exp_c** | 64 | 2e-4 | **0.6527** ✅ |
  | exp_d | 64 | 1e-4 | 0.6586 |
  | exp_a | 16 | 2e-4 | ~0.69 |
  | exp_b | 16 | 1e-4 | 0.6993 |

- **Finding:** rank dominates LR (r=64 ≫ r=16, Δ≈0.04); LR effect within r=64 is marginal (Δ=0.006).
- Instance destroyed; all adapters + `results.json` archived under `results/`.
- Bugs fixed mid-flight → Lessons **#7–#11** (flash_attn→eager, OOM→batch=1/grad_accum=16/grad-ckpt,
  `metric_for_best_model` dict-eval naming, `label_names=[]` breaking eval loss,
  `load_best_model_at_end` unsafe with QLoRA, chained-run CUDA OOM → `sleep 30`).

## Phase 3 — Final training + export ✅ scripts written (run pending)
- `configs/final.yaml` (exp_c winner, adapted for A100): full bf16, lr=2e-4, warmup 10%, 5 epochs,
  early stopping (patience=5), eval/save every 25, `save_total_limit=10`, `max_seq_length=4096`.
- `05_train_final.py`: full-bf16 LoRA, custom PEFT-safe early-stopping callback, records
  `best_checkpoint` in `results.json`.
- `06_export.py`: merge **best** checkpoint (not last) → GGUF Q4_K_M/Q5_K_M/Q8_0 via llama.cpp;
  `--push-to-hub`; `--force-merge`.

## Phase 4 — CEFEAI re-evaluation ✅ script written (run pending)
- `07_cefeai_eval.py`: local inference (greedy) + OpenRouter judge; protocol-v2 by default
  (`--no-system-prompt` for v1); compares against the baseline matching its prompt mode.

## Shared infrastructure
- **`scripts/utils/cefeai.py`** (new): single source of truth for judge prompts, `parse_judge_response`,
  `wilson_ci`, `baseline_verdict` (direction-aware: RR up=better, CB down=better), `load_system_prompt`.
  Imported by both `00` and `07` so the two sides of the comparison cannot drift.
- **`configs/system_prompt.txt`** (new): canonical Reformed prompt, generated from `train.jsonl`,
  committed, verified byte-identical across all 3,024 records.

---

## Quality process applied to Phase 3/4 code
Before any GPU spend, the Phase 3/4 scripts went through:
- **Two `/code-review` passes** (extra-high effort, recall mode) → 11 real findings fixed across
  reviews (EarlyStopping assertion crash, missing `enable_input_require_grads`, best-checkpoint
  selection, device_map multi-GPU split, dry-run offline-safety, RR/CB verdict direction, cost-budget
  scope, empty-prompt-file guard, …). See Lessons #12–#15.
- **A local simulation harness** (`sim_phase34.py`, kept out of the repo) — **38/38** checks: mocks the
  trainer to exercise early-stopping patience, best-checkpoint detection, verdict direction,
  system-prompt loading (incl. hard-fail on missing/empty), judge parsing, Wilson CI, and real
  `--dry-run`s for `00`/`05`/`06`/`07`.

## What remains (requires GPU / API spend — not yet run)
1. **Protocol v2 re-baseline:** `python scripts/00_cefeai_baseline.py --benchmark both` (~$0.30).
2. **Phase 3 on A100:** `05_train_final.py` → `06_export.py --push-to-hub`.
3. **Phase 4 (v2):** `07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both`.
