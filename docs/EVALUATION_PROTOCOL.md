# OpenScriptura — CEFEAI Evaluation Protocol (the parameters CEFE.AI left open)

CEFE.AI's AllFaith Benchmark publishes the judge prompts, scales, data, and
aggregation, but **deliberately does not specify** the judge model, the judge
settings, or the inference settings of the model under test. This document
**defines** those parameters for OpenScriptura, by good scientific practice, so
the evaluation is reproducible and the baseline → fine-tuned comparison is valid.

Decided 2026-06-08 by a 7-discipline review (AI/Software/DB/Data-Science/
Statistics/CS/Pastoral). These values are now part of the comparability lock
(CLAUDE.md → HARD RULE). The same values are applied **identically** to the raw
baseline and the fine-tuned model — only the weights differ.

---

## 1. Judge model

| Parameter | Decision | Why |
|-----------|----------|-----|
| Model class | **A strong, frontier-class model** — not a small/"flash" tier | Judge quality is the single biggest driver of score validity; a weak judge mis-scores the rubric. |
| Family | **NOT Qwen** (the model under test is Qwen3-8B) | Avoids self-preference / family bias (a model tends to favor its own outputs). |
| Pinning | **Pin an exact dated snapshot** (no floating "latest" alias) | Reproducibility — a silently-updated judge changes the numbers. |
| Concrete default | Upgrade `OPENROUTER_MODEL_JUDGE` from `deepseek-v4-flash` → a strong non-Qwen judge (e.g. `deepseek/deepseek-v4` full, or a frontier Claude/GPT tier), pinned. | Flash is too weak to be a defensible judge; the exact pick is a budget call but must meet the criteria above. |
| Same judge both sides | **Mandatory** | The absolute number is judge-dependent; holding the judge constant makes the **delta** valid. |
| Validation (M9) | **Before trusting numbers:** human-label a ~50-item sample and report judge↔human agreement with **quadratic-weighted Cohen's κ** (ordinal scale). | Justifies the judge empirically; flags a miscalibrated judge. Also inspect per-tradition judge behavior (CB) for judge bias. |

> **Honesty caveat:** because CEFE.AI does not publish *their* judge, our absolute
> numbers are **protocol-adherent but judge-dependent** — not provably identical to
> their leaderboard. The rigorous claim is the internal delta. (See `configs/cefeai/README.md`.)

## 2. Inference settings — model under test

| Parameter | Decision | Why |
|-----------|----------|-----|
| `temperature` | **0.0** (greedy) | Deterministic; removes sampling variance so the baseline↔fine-tuned difference is the model, not noise. |
| `top_p` | 1.0 (irrelevant under greedy) | — |
| `seed` | **42** | Reproducibility (belt-and-suspenders with greedy). |
| `max_tokens` | **1024** (raised from 512) | Avoid truncation bias: a verbose fine-tuned model must not be cut more than the terse raw model. Same cap both sides. |
| system prompt | **NONE** (headline) | A system prompt saturates the metric (raw + Reformed prompt → RR 99.3% / CB 87.8%, Lesson #16) and isn't leaderboard-comparable. `--system-prompt` is an opt-in deployment-behavior datapoint only. |
| samples per prompt | **1** | Sufficient under greedy/temperature 0. |

## 3. Judge settings · invalid output · determinism

| Parameter | Decision | Why |
|-----------|----------|-----|
| Judge `temperature` | **0.0** | Deterministic verdicts. |
| Judge `max_tokens` | **256** | RR verdict is a one-sentence JSON; CB is `Rating: <d>` — 256 is ample. |
| Judge `enable_thinking` | **False** | The task is a simple rubric classification; thinking adds nondeterminism and risks truncating the verdict inside a `<think>` block. (Parser still strips stray `<think>` defensively.) |
| Judge calls per response | **1** (no majority vote for v0.1) | Simplicity; ensemble/panel is a future enhancement. |
| **Invalid / unparseable judge output** | **Exclude from metrics; never coerce to a default score.** Record `n_parse_error` and the rate; if the rate is non-trivial (>~2%), fix the judge/prompt — do not silently impute. | Coercing a parse error to 0 (RR) or 4 (CB) biases the aggregate. A deterministic (temp 0) retry would reproduce the same bad output, so we exclude rather than retry. |
| Determinism | temp 0 on both model and judge · fixed seeds · single sample · **pinned snapshots** (judge + base model) · all settings recorded in the summary JSON. | Full reproducibility. |

## 4. Statistics

- **Primary metric:** RR = mean score (0–4); CB = mean rating (1–7) — as CEFE.AI prescribes. Report each with a **95% CI on the mean** (`mean_score_ci` / `mean_rating_ci`, normal approx; n=150 / 1456) plus the **full score distribution** (Wilson CIs per level).
- **Slices (CB):** by `pair_id`, `template_id`, and tradition (`religion_from`/`religion_to`) — the "by pair / tradition / template" the README recommends.
- **Improvement test:** the benchmark is **paired** (same prompts both models). Test the lift with a **paired test on per-prompt score deltas** — paired t-test, or **Wilcoxon signed-rank** (ordinal-safe) — and report the mean delta with CI and an effect size. (Implement at comparison time using both result sets, matched by `prompt_id`.)
- **Multiple comparisons:** the two headline claims (RR, CB) need no correction; if reporting many per-pair/per-tradition claims, control FDR (Benjamini–Hochberg) or note Bonferroni.

## 5. What this changes in code (already applied)
- `scripts/utils/cefeai.py`: `mean_ci()` + `mean_score_ci`/`mean_rating_ci` in `summarize`; per-pair/template/tradition CB slices; official CB regex.
- `scripts/00_cefeai_baseline.py` & `07_cefeai_eval.py`: model `max_tokens=1024`; judge `enable_thinking=False`, `max_tokens=256`.
- `.env.example`: judge-model guidance (strong, non-Qwen, pinned).
- Still TODO (analysis steps, not blockers): the paired significance test at comparison time, and the κ judge-validation sample.
