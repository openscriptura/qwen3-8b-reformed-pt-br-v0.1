# Phase 4 Results — Fine-tuned vs Baseline (qwen3-8b-reformed-pt-br-v0.1)

Final fine-tune: LoRA exp_c (r=64, α=128, lr=2e-4), best checkpoint-325 (eval_all_loss 0.6546).
Eval protocol: v1 headline (NO system prompt), official CEFE.AI judge `deepseek/deepseek-v4-flash`
@ max_tokens=1024, temp=0, both sides identical (comparability lock held). EN = leaderboard
anchor; pt-BR = product (translated track, NOT leaderboard-comparable — internal delta rigorous).

## Headline paired results

| Track | Metric | Baseline → Fine-tuned | Δ | 95% CI | Wilcoxon p | Effect (rb) | Significant |
|-------|--------|----------------------|----|--------|-----------|-------------|-------------|
| EN    | RR (0–4) | 0.147 → 0.227 | +0.080 | [−0.026, 0.186] | 0.152 | 0.11 | ✗ |
| EN    | CB (1–7) | 3.694 → 3.499 | −0.195 | [−0.286, −0.104] | 3.9e−06 | −0.10 | ✓ |
| pt-BR | **RR (0–4)** | 0.081 → **0.617** | **+0.537** | [0.325, 0.749] | **3.7e−06** | **0.66** | ✓✓ (large) |
| pt-BR | CB (1–7) | 3.911 → 3.972 | +0.059 | [−0.041, 0.160] | 0.308 | 0.03 | ✗ |

## In the plan's own units (Any Representation % / Any Bias %)

| Metric | EN base | EN FT | pt-BR base | pt-BR FT |
|--------|---------|-------|-----------|----------|
| RR Any Representation (≥1) | 12.7% | 13.3% | 6.7% | **20.8%** |
| RR Meaningful (≥2) | 1.3% | 6.7% | 0.7% | **17.4%** |
| CB Any Bias (≠4) | 20.2% | **62.7%** | 20.1% | **64.6%** |
| CB Strong Bias (\|dev\|≥2) | 15.2% | 47.0% | 13.3% | 51.4% |
| CB Neutral (=4) | 79.8% | 37.3% | 79.9% | 35.4% |
| CB bias magnitude (mean \|dev\|) | ~0.3 | 1.36 | ~0.3 | 1.49 |

## Expected (IMPLEMENTATION_PLAN) vs Actual

| Goal | Target | Actual | Verdict |
|------|--------|--------|---------|
| CB: confessional bias ≫ 19.6% baseline | "significantly higher Any Bias" | 20% → **~64%** (both langs) | ✅ **met / exceeded** |
| RR: Any Representation >60%, beat Grok 29.3% | 60–70% | best **20.8%** (pt-BR) | ❌ **missed** (improved, not reached) |

## Pro-Protestant directionality (CB per-faith slice; <4 encourages from→to)

| Direction (broad Protestant) | EN base→FT | pt-BR base→FT | Reformed-correct? |
|------------------------------|-----------|---------------|-------------------|
| INTO Protestant (encourage joining → ↓) | 3.75 → 3.35 | 4.08 → **3.76** | ✅ both |
| OUT of Protestant (discourage leaving → ↑) | 3.42 → 2.92 | 3.73 → **3.86** | pt-BR ✅ / EN ✗ |

In **pt-BR**, both axes move the Reformed-correct way (encourage joining, discourage leaving).
In **EN**, the model became more pro-conversion *in general* (both directions ↓), not specifically
pro-Protestant. Direction is clean only in the training language (pt-BR). Slice n≈200/axis;
directionally consistent but not significance-tested per slice.

## Interpretation

- **The fine-tune succeeded on its core confessional goal.** CB Any Bias tripled (20%→64%),
  neutrality collapsed (80%→35%), and in pt-BR the model tilts toward joining / away from leaving
  Protestant Christianity. This is the intended *explicit confessional bias*, statistically solid
  on the EN net-direction (p=3.9e−6) and on bias magnitude both languages.
- **Religious representation improved most where it matters — pt-BR — with a large, significant
  effect** (0.08→0.62, rb=0.66), but absolute level (~21%) fell well short of the >60% target and
  did not beat Grok (29.3%). RR rewards *volunteering* religion in neutral prompts; the Reformed
  Q&A corpus taught the model to be Reformed *when the topic is religious* (→ CB), not to inject
  religion into secular prompts (→ RR stays modest). Arguably good product behavior; under-performs
  the RR metric.
- **Why EN < pt-BR:** the model was trained in pt-BR, so its religious/confessional behavior
  transfers most strongly in pt-BR. EN shows the religiosity but not the clean directional bias.

## One-line summary

> The v0.1 fine-tune made the model **take Reformed positions** (CB ~3× baseline, pro-Protestant
> tilt in pt-BR) and **significantly more religiously expressive in pt-BR** (large effect), but it
> remains conservative about volunteering religion unprompted (RR ~21%, below the >60% target).

## Caveats (honesty)
- Absolute numbers are **judge-dependent** (flash tier, κ-validated 0.63–0.98); the **internal
  baseline→fine-tuned delta is the rigorous claim** (same judge both sides).
- pt-BR is a **translated** track — NOT leaderboard-comparable; its internal delta is valid.
- pt-BR CB parse errors topped up 60→5 via `--resume`; EN clean (0/0).

_Files: `results/eval_merged_[ptbr_]noprompt_{RR,CB}_*`; baselines `results/baseline_qwen_qwen3_8b_[ptbr_]noprompt_*`._
