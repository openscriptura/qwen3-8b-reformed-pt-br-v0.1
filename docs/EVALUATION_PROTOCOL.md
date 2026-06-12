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
| Model class | Prefer a strong, frontier-class judge — BUT **reliability is a hard gate**: a heavy-reasoning model that overflows the token budget is unusable as a judge. | Judge quality drives validity, but a judge that returns `content: null` on a non-random subset is worse than a slightly weaker reliable one (the failures are content-correlated → bias the surviving sample). |
| Family | **NOT Qwen** (the model under test is Qwen3-8B) | Avoids self-preference / family bias (a model tends to favor its own outputs). |
| Pinning | **Pin an exact dated snapshot** (no floating "latest" alias) | Reproducibility — a silently-updated judge changes the numbers. OpenRouter returns the resolved snapshot id (e.g. `deepseek/deepseek-v4-flash-20260423`), recorded per result. |
| Concrete choice (v0.1) | **`deepseek/deepseek-v4-flash`** as the SINGLE judge, both sides, `max_tokens=1024`, **no cross-model fallback**. | **By evidence (2026-06-09):** across past CB runs `deepseek-v4-pro` ran as a heavy reasoning model and overflowed (28% null-content; reasoning ≤**1178** tok, so even 1024 is unsafe), while `flash` reasons ≤842 tok → ~0% errors at 1024. Flash is the weaker tier, so its absolute numbers are gated on the κ check below. **Escalation if κ fails:** `deepseek-v4-pro` + pin a non-reasoning provider (DeepInfra/DigitalOcean/Together/Alibaba showed 0% overflow) — never a per-call model fallback. |
| Same judge both sides | **Mandatory** | The absolute number is judge-dependent; holding the judge constant makes the **delta** valid. |
| Validation (M9) | **Gates the absolute numbers when using flash:** human-label a ~50-item sample and report judge↔human agreement with **quadratic-weighted Cohen's κ** (ordinal scale); require ~substantial agreement (κ ≳ 0.6) before citing absolutes. **Results + analysis: [`docs/JUDGE_VALIDATION.md`](JUDGE_VALIDATION.md)** (EN: RR κ=0.80, CB κ=0.63 — both pass; pt-BR pending). | Justifies the weaker judge empirically; flags a miscalibrated judge. Also inspect per-tradition judge behavior (CB) for judge bias. |

> **Honesty caveat:** because CEFE.AI does not publish *their* judge, our absolute
> numbers are **protocol-adherent but judge-dependent** — not provably identical to
> their leaderboard, and v0.1 uses the lighter `flash` tier (reliability-driven,
> κ-gated). The rigorous claim is the internal delta (same judge both sides). (See
> `configs/cefeai/README.md`.)

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
| Judge `max_tokens` | **1024** | DeepSeek-v4 runs as a reasoning model on some OpenRouter providers; reasoning tokens count against `max_tokens`, and at 256 the judge consumed all tokens mid-reasoning → `content: null` parse errors. 1024 is safe for the chosen `flash` judge (max reasoning observed **842** tok < 1024 → ~0% null) but NOT for `pro` (observed up to **1178** tok); that gap is the empirical reason §1 selects flash. Covers the reasoning + the 5-token verdict (`Rating: N` / short JSON); cost is nearly identical (billed on tokens used, not the cap). **Locked on BOTH scripts** (comparability). |
| Judge `enable_thinking` | **False** | Prevents Qwen3-family thinking blocks from truncating the verdict. DeepSeek ignores this Qwen-specific flag (it reasons regardless of it); the token budget above + the lighter-reasoning flash tier handle that. The parser strips stray `<think>` blocks defensively. |
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
- `scripts/00_cefeai_baseline.py` & `07_cefeai_eval.py`: model `max_tokens=1024`; judge `enable_thinking=False`, `max_tokens=1024` (raised from 256 — see §3 above).
- `.env.example`: judge-model guidance (strong, non-Qwen, pinned).
- Still TODO (analysis steps, not blockers): the paired significance test at comparison time, and the κ judge-validation sample.

## 6. Dois tracks de idioma — inglês (âncora científica) + pt-BR (verdade de produto)

**Decisão (2026-06-09, com o usuário): fazer os dois, com ênfases diferentes — porque respondem a perguntas diferentes.** Implementado via `--lang {en,ptbr}` (ver `00`/`07`/`translate_benchmark.py`).

| Track | Pergunta que responde | Papel |
|-------|-----------------------|-------|
| **Inglês (CEFE.AI oficial)** | "Isso é uma contribuição científica vs o estado da arte (Grok, GPT-5, etc.)?" | **Âncora científica** — comparável ao leaderboard, é o número que o mundo verifica (paper/arXiv, credibilidade). |
| **Português (traduzido)** | "Isso é bom para o meu usuário real (brasileiro, reformado, em português)?" | **Verdade de produto** — mede exatamente o objetivo de deploy. |

**Para o objetivo do projeto, o track pt-BR é o que de fato importa.** Ele mede o cenário real (pergunta PT → resposta PT reformada). É o número que vai no **model card** e que justifica o produto.

**Mas não largar o inglês** — ele é barato (~$1,13) e é a **única** forma de dizer "subimos a representação CEFE.AI em N pontos" de modo comparável ao leaderboard público. Sem ele, perde-se a credibilidade científica.

**Caveat técnico a observar na Phase 4:** o modelo fine-tuned provavelmente vai responder **em português mesmo às perguntas em inglês** (porque foi treinado pt-BR). No track inglês isso é **OK** — o juiz mede *representação religiosa*, não idioma — mas é mais um motivo pelo qual o **track pt-BR é o mais natural e fiel** ao que se quer.

**Resumo:** **pt-BR = headline de produto** (o que se quer); **inglês = âncora científica** (o que torna verificável). **Os dois, sem trocar um pelo outro.** O delta interno é rigoroso nos dois (mesmo juiz/settings em ambos os lados); só os **absolutos** do pt-BR não são leaderboard-comparable (benchmark traduzido = benchmark diferente). κ valida o juiz **nos dois idiomas**.
