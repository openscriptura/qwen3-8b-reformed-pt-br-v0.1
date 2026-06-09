# Changelog & Decision Log — OpenScriptura

Chronological record of what was built, what changed, and **why** — through 2026-06-08.
Lessons learned live in [`CLAUDE.md`](CLAUDE.md#lessons-learned-vastai--gpu-setup) (#1–#17); this file
is the narrative + decision rationale. Phase detail is in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## ⭐⭐ Headline correction — adopted the OFFICIAL CEFE.AI judge; all prior numbers INVALID

**Date:** 2026-06-08 · **Status:** done (code) / baseline re-run pending.

We discovered our judge did **not** match CEFE.AI: RR scored 0–3 (theirs is **0–4**)
and CB scored a 0–3 "proselytization" rubric (theirs is a **1–7** transition
`religion_from→religion_to` scale, neutral=4). **Therefore every number we'd
reported — RR 4.7% / CB 19.6% (v1) and RR 99.3% / CB 87.8% (v2) — is invalid for
CEFE.AI comparison** (wrong rubric), and is now kept only as a historical artifact.

**Fix.** Vendor the official `scoring_prompt.json` files verbatim in
`configs/cefeai/` and load them at runtime (`scripts/utils/cefeai.py`); never
hardcode. Aggregate as the upstream READMEs prescribe (RR mean+distribution; CB
mean + by pair/template/tradition). Verified our 1,606 questions are **identical**
to upstream. Added a paired Wilcoxon test for the lift and a quadratic-weighted κ
judge-validation script (`scripts/08_judge_validation.py`).

**Adherence stance (the honest position).** We are **100% aligned with everything
CEFE.AI documents**. The only gap is what they do **not** publish — the **judge
model** and **inference settings** — which we define by good science in
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) (judge: strong,
non-Qwen, pinned, temp 0, thinking off; model under test: temp 0 / seed 42 / no
system prompt / max_tokens 1024; invalid judge output excluded, never coerced).
So the **internal baseline→fine-tuned delta is rigorous**; **absolute** numbers are
*protocol-adherent but judge-dependent* — we will not claim leaderboard parity
until CEFE.AI shares the judge. **Next:** re-run the baseline with the official judge.

---

## ⭐ Headline decision — evaluation protocol is **v1 (no system prompt)**; v2 tried & rejected

**Date:** 2026-06-08 · **Status:** decided (data-driven).

**The question.** The fine-tuned model deploys with a Reformed system prompt. Should evaluation put
that prompt on *both* sides (v2), or on *neither* (v1, like the original CEFEAI-comparable baseline)?

**What we did.** We built the v2 path (shared `utils/cefeai.py`, committed `configs/system_prompt.txt`,
prompt-mode-tagged outputs, `--system-prompt` flag) and **ran the v2 baseline** (~$0.83) to find out.

**What the data said.** Raw, un-fine-tuned Qwen3-8B **+ the Reformed system prompt** scored:

| | v1 (no prompt) | v2 (with prompt) |
|---|---|---|
| RR — Any Representation | **4.7%** | **99.3%** |
| CB — Any Bias | **19.6%** | **87.8%** |
| CB — Strong Bias | 0.0% | 75.8% |

The system prompt **alone** saturated the metric. That makes v2 unusable as the headline:
1. **Not comparable to CEFEAI** — leaderboard models are measured prompt-free; a 99.3% from a
   "be a committed Reformed evangelist" instruction no other model got is a different experiment.
2. **Erases the fine-tuning signal** — if raw+prompt already ≈99% / ≈88%, tuned+prompt has no
   headroom; "improved by N points" becomes unmeasurable.

**Decision.** **Headline / CEFEAI-comparable protocol = v1 — NO system prompt, on both sides.** The
fine-tuning is in the weights, so a no-prompt fine-tuned model should still beat the 4.7% raw baseline,
and that delta is real and comparable. `00`/`07` **default to no prompt**; `--system-prompt` opts into
v2, kept only as a labeled deployment-behavior datapoint (for CB, high bias is by-design intent).

**Net positive from the detour.** The shared `utils/cefeai.py` and committed `configs/system_prompt.txt`
are good regardless and stay. The ~$0.83 v2 run was cheap insurance: it caught the saturation problem
*before* we spent A100 money on the final model and eval. The v1 baseline (4.7% / 19.6%) is archived in
`results/v1_baseline_archive/`.

**Unchanged throughout:** `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`.

**Console gotcha noted:** `00`'s RR leaderboard printed "this run = 4.7%" (a stale static row) while the
real v2 RR was 99.3% — trust `results/..._summary.json`, not the leaderboard insert.

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

  | Config | r | LR | eval_all_loss | best step |
  |--------|---|----|---------------|-----------|
  | **exp_c** | 64 | 2e-4 | **0.6527** ✅ | 350 |
  | exp_d | 64 | 1e-4 | 0.6586 | 350 |
  | exp_a | 16 | 2e-4 | 0.6640 | 350 |
  | exp_b | 16 | 1e-4 | 0.6709 | 500 |

  _(exact values from `results/exp_*_results.json`)_
- **Finding:** both rank and LR help, rank ~2× more, both small — best r=64 (0.6527) vs best r=16 (0.6640) = Δ0.011; LR effect within a rank ≈0.006. exp_c wins on both axes. *(An earlier draft said Δ≈0.04 — a data error: exp_b's Tier-B 0.6993 was used as its eval_all; real exp_b eval_all = 0.6709.)*
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
- `07_cefeai_eval.py`: local inference (greedy) + OpenRouter judge; **v1 (no system prompt) by
  default** (`--system-prompt` opts into the v2 datapoint); compares against the baseline matching
  its prompt mode.

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
  scope, empty-prompt-file guard, …). See Lessons #12–#16.
- **A local simulation harness** (`sim_phase34.py`, kept out of the repo) — **38/38** checks: mocks the
  trainer to exercise early-stopping patience, best-checkpoint detection, verdict direction,
  system-prompt loading (incl. hard-fail on missing/empty), judge parsing, Wilson CI, and real
  `--dry-run`s for `00`/`05`/`06`/`07`.

## What remains (requires GPU / API spend — not yet run)
- ✅ Baselines done: v1 headline (4.7% / 19.6%) + v2 datapoint (99.3% / 87.8%) both in `results/`.
1. **Phase 3 on A100:** `05_train_final.py` → `06_export.py --push-to-hub`.
2. **Phase 4 (v1 headline):** `07_cefeai_eval.py --model-path checkpoints/final/merged --benchmark both`
   (no prompt; compares vs the v1 baseline). Optionally `--system-prompt` for the v2 datapoint.
