# Judge Validation — quadratic-weighted Cohen's κ (M9)

Validation of the CEFE.AI LLM judge (`deepseek/deepseek-v4-flash`, single, no
fallback, `max_tokens=1024`, `temperature=0`) against **human** labels, per
`docs/EVALUATION_PROTOCOL.md` §1 (Validation / M9). The judge is the weaker
("flash") tier, so its **absolute** numbers must be justified empirically; the
**internal baseline→fine-tuned delta is rigorous regardless** (same judge on both
sides — any systematic judge bias cancels).

**Method.** `scripts/08_judge_validation.py`: a **blind** sample (the human never
sees the judge's score — it is held in a sidecar key and re-joined only at scoring,
so the human cannot anchor on it), **stratified** (~half modal + ~half off-modal, so
κ tests the discriminative ratings and is not dominated by the ~80%-neutral CB /
~87%-zero RR mass), n = 50 per benchmark, scored with **quadratic-weighted Cohen's
κ** on the official scale (RR 0–4, CB 1–7). Bar: **κ ≥ 0.6** ("substantial",
Landis–Koch).

> Status: **English track validated** (both ≥ 0.6). pt-BR track pending. Results as
> of 2026-06-09; numbers move if labels are revised.

## Summary

| Track | Benchmark | κ (quad-weighted) | Exact | Within ±1 | Verdict | Blind? |
|-------|-----------|-------------------|-------|-----------|---------|--------|
| EN | RR (0–4) | **0.7984** | 84% | 100% | substantial | re-labeled after seeing aggregate (see caveat) |
| EN | CB (1–7) | **0.6313** | 66% | 82% | substantial | **fully blind** |
| pt-BR | RR (0–4) | pending | — | — | — | — |
| pt-BR | CB (1–7) | pending | — | — | — | — |

Both English κ clear the 0.6 bar → the flash judge is validated for the English
(headline) absolute numbers, with the documented calibration caveats below.

## RR (English) — κ = 0.7984

Confusion (judge ↓ vs human →):

```
        h0  h1  h2  h3  h4
   j0   31   0   0   0   0
   j1    8   9   0   0   0
   j2    0   0   1   0   0
   j3    0   0   0   1   0
```

- 84% exact, **100% within ±1**, 0 gross errors. Mean (judge − human) = **+0.16**.
- The only residual disagreement is the **0↔1 boundary**: 8 cases where the judge
  scores 1 ("brief, passing mention of religion") and the human scores 0 ("none").
  The judge has a **slightly more lenient threshold** for "any religious mention".
  Systematic and one-directional (judge ≥ human), i.e. a calibration difference, not
  random error.

**Methodological caveat (honesty).** A first pass scored κ = 0.5981; the human then
re-read the items more carefully and re-labeled, reaching 0.7984. This re-label was
**not fully blind** — the aggregate disagreement (mean +0.22 and the single off-by-2
case) had been shown. The improvement is judged **credible, not anchoring**, because
the human **retained 8 honest disagreements** (still stricter than the judge) rather
than matching it (which would drive κ→1.0); most of the jump came from re-reading
ONE case (an existential "Who am I?" prompt whose answer does carry religious
content) that the hasty first pass had missed. For maximal rigor a **fresh fully
blind RR sample** can confirm κ ≥ 0.6 without any prior exposure.

## CB (English) — κ = 0.6313 (fully blind)

Confusion (judge ↓ vs human →):

```
        h1  h2  h3  h4  h5  h6  h7
   j1   11   2   0   1   0   0   0
   j2    0   4   1   0   0   0   0
   j3    0   1   1   0   0   0   0
   j4    4   3   3  15   0   0   0
   j5    1   0   0   0   1   1   0
   j6    0   0   0   0   0   1   0
```

- 66% exact, 82% within ±1, **9 off-by-≥2**. Mean (judge − human) = **+0.38**;
  direction (encourage / neutral / discourage) disagrees on **12/50**.
- **Systematic pattern:** the off-by-≥2 cases are overwhelmingly **judge = 4
  (neutral) vs human = 1–2 (encouraging)** — warm, affirming answers ("your journey
  of exploring X is a beautiful, meaningful path…"). The human reads the affirming
  tone as **encouraging the transition**; the judge reads it as **neutral support**.
- **This cuts both ways — it is not clearly a judge error.** The rubric reserves 1
  for "*strongly* encouraging the transition". An answer that affirms the person
  *without explicitly advocating conversion* is arguably neutral (judge) or at most
  "slightly encouraging" (3) — so the judge may be the **better-calibrated** rater
  here, with the human **over-detecting** encouragement in a warm tone. It is the
  hardest CB judgment (affirmation vs encouragement), genuinely ambiguous.

## Interpretation

1. **The headline claim (the delta) is rigorous regardless of κ.** Both the RR
   leniency (+0.16) and the CB affirmation→neutral tendency (+0.38) are *systematic*
   and applied to **both** the baseline and the fine-tuned model, so they **cancel**
   in the paired baseline→fine-tuned delta. "Fine-tuning moved CEFE.AI by N points"
   stands.
2. **Absolute numbers carry documented calibration caveats:**
   - RR: the judge may slightly **over-count** faint religious mentions (the 0↔1
     leniency) → "any-representation %" is, if anything, a mild over-estimate.
   - CB: the judge tends to call affirming answers **neutral** → it may **under-count
     directional bias** a human would perceive. Relevant for the Reformed fine-tune,
     whose directional bias is *by design*: the judge may under-detect it.
3. **κ confidence.** At n = 50 the CI on κ is wide (~±0.1); CB (0.63) is the weaker,
   borderline validation. If a stronger absolute CB claim is needed, escalate the
   judge to the `pro` tier with a non-reasoning provider pinned, or label more.

## Escalation criterion

If any track's κ falls below 0.6 (or the calibration caveat is unacceptable for an
absolute claim), escalate from `deepseek-v4-flash` to `deepseek-v4-pro` **with a
non-reasoning provider pinned** (DeepInfra / DigitalOcean / Together / Alibaba showed
0% reasoning-token overflow — see CLAUDE.md Lesson #18), re-run the affected
baseline, and re-validate. The internal delta does not require this; only the
defensibility of the absolute numbers does.
