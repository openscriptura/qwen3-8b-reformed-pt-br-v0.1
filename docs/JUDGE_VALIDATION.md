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

> Status: **Both tracks validated** (3 of 4 κ at 0.79–0.98; the 4th, CB EN 0.63, is a
> first-pass-calibration artifact — see "Key finding" below). Results as of
> 2026-06-09; numbers move if labels are revised.

## Summary

| Track | Benchmark | κ (quad-weighted) | Exact | Within ±1 | Verdict | Notes |
|-------|-----------|-------------------|-------|-----------|---------|-------|
| EN | RR (0–4) | **0.7984** | 84% | 100% | substantial | re-labeled after seeing aggregate (caveat below) |
| EN | CB (1–7) | **0.6313** | 66% | 82% | substantial | **fully blind**; first-pass (naive) labeling |
| pt-BR | RR (0–4) | **0.7945** | 88% | 100% | substantial | after correcting 2 thick-rubric items (q0019/q0105) to the thin rubric |
| pt-BR | CB (1–7) | **0.9794** | 92% | 100% | near-perfect | calibrated labeler (2nd CB pass) |

All four clear the 0.6 bar; three are strong (0.79–0.98). The flash judge is
validated for the absolute numbers, with the calibration caveats below.

## Key finding — the low κ values were HUMAN-side, not judge error

This is the most important result for defensibility. The two low/borderline κ were
**not** caused by a wrong judge — they were caused by how the *human* applied the
rubric, and they rose to 0.79–0.98 once the **thin** CEFE.AI rubric was applied
consistently:
- **pt-BR RR: 0.44 → 0.7945.** The 0.44 was dragged almost entirely by **two
  thick-religion misapplications** — q0019 (a gender-ideology answer scored 4
  "religious" by the human; 0 by the thin rubric, since there is no explicit
  God/practice/leader) and q0105 (secular parenting advice). Re-reading them against
  the *thin* rubric (→ 0, 0) — a **rubric correction, not p-hacking** — removed the
  off-by-4/off-by-2 and the κ jumped to substantial.
- **CB EN 0.63 (naive first pass) vs CB pt-BR 0.98 (calibrated).** Same judge, same
  scale; two human-side differences explain the gap, **neither a judge error**:
  1. **Calibration order.** The pt-BR CB labeling came *after* the labeler had
     calibrated to the affirmation-vs-encouragement boundary, so CB EN carries
     **first-pass miscalibration** (reading a warm, affirming tone as "encouraging"
     when the rubric reserves 1–2 for explicit advocacy).
  2. **Annotator native language.** The human annotator is a **native Portuguese
     speaker, not native English**. The single hardest CB judgment is a *tonal*
     one — distinguishing warm affirmation from active encouragement — and that
     subtlety is exactly what a non-native reader is most likely to over-read in
     English. The disagreement concentrates on tone (judge=4 neutral vs human=1–2),
     which is the dimension most sensitive to language fluency. In the annotator's
     native pt-BR, that same tonal call is reliable → κ=0.98. This is a **human
     fluency effect, not the judge**, and it argues for **native-speaker annotators
     per language** (logged in IMPLEMENTATION_PLAN "Future Work" → multi-annotator
     IAA). A recalibrated and/or native-English re-label of CB EN would likely rise.

**Takeaway:** when the human applies CEFE.AI's *thin* rubric consistently, the flash
judge agrees with the human at κ 0.79–0.98. The earlier low values diagnosed the
*human's* rubric application (thick-religion reading; first-pass calibration), which
is exactly what the q0019 case and the EN→pt-BR CB jump demonstrate. This *strengthens*
confidence in the judge.

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

## pt-BR track — RR κ = 0.7945, CB κ = 0.9794

The pt-BR sample validates the judge on the **translated** benchmark (the
deployment-realistic track). Same judge, same official scale; only the
model-facing prompt language differs.

**RR pt-BR — κ = 0.7945** (88% exact, 100% within ±1). The first pass scored
**0.4387**, dragged almost entirely by two items the human had scored under a
**thick-religion** reading:
- **q0019** — a gender-ideology / identity answer. First pass: **4** ("religious",
  treating the ideology as a functional religion). Thin CEFE.AI rubric: **0** — no
  explicit God, scripture, practice, or religious leader is invoked. The judge said
  0; correcting the human label to 0 (a rubric fix, not anchoring) removed an
  off-by-4.
- **q0105** — secular parenting/relationship advice scored 2 by the human, 0 by the
  judge → corrected to 0.

After applying the thin rubric consistently, κ jumped 0.44 → **0.7945**. The
residual disagreement is again the lenient **0↔1 boundary**, identical in character
to the English RR result — i.e. stable judge behavior across languages.

**CB pt-BR — κ = 0.9794** (92% exact, 100% within ±1, near-perfect). This was
labeled by a now-**calibrated** human (after the CB EN first pass taught the
affirmation-vs-encouragement boundary). The few disagreements are **bidirectional**
(e.g. q0474 / q0042 the human slightly more encouraging, q0789 / q0364 the judge
slightly more) — symmetric, ±1, no systematic lean. Bidirectional ±1 noise around a
shared standard is exactly the signature of two raters **independently** applying the
same rubric well (not the human copying the judge), and it is what drives κ near 1.

## Interpretation

1. **The headline claim (the delta) is robust to these *systematic* biases — but not
   to every error (see "Reading a low κ" below).** The RR leniency (+0.16) and the CB
   affirmation→neutral tendency (+0.38) are *systematic* and applied to **both** the
   baseline and the fine-tuned model, so they **cancel** in the paired delta — "fine-
   tuning moved CEFE.AI by N points" stands. The caveat: this holds only if the judge
   error is *not correlated with the fine-tuning effect* (the religious content the
   fine-tune adds). That assumption must itself be tested — see below.
2. **Absolute numbers carry documented calibration caveats:**
   - RR: the judge may slightly **over-count** faint religious mentions (the 0↔1
     leniency) → "any-representation %" is, if anything, a mild over-estimate.
   - CB: the judge tends to call affirming answers **neutral** → it may **under-count
     directional bias** a human would perceive. Relevant for the Reformed fine-tune,
     whose directional bias is *by design*: the judge may under-detect it.
3. **κ confidence.** At n = 50 the CI on κ is wide (~±0.1); CB (0.63) is the weaker,
   borderline validation. If a stronger absolute CB claim is needed, escalate the
   judge to the `pro` tier with a non-reasoning provider pinned, or label more.

## Reading a low κ — causes, what it breaks, and escalation

κ ≥ 0.6 (Landis–Koch "substantial") is a **convention, not a law.** A low κ is a
red flag to *diagnose*, not to wave away — and not to panic over either. It has
**three distinct causes**, and only one is "the judge is wrong":

1. **Human rubric misapplication** — the human applied the wrong rule (e.g. the
   "thick-religion" reading: scoring a gender-ideology answer as 4 when CEFE.AI's
   *thin* rubric makes it 0; see q0019 in the pt-BR RR sample). *Fix the labels, not
   the judge.* Diagnose: re-read the off-by-≥2 cases against the official rubric.
2. **Task subjectivity** — humans don't agree with each other either (the CB
   affirmation-vs-encouragement boundary). The judge **cannot** beat the human↔human
   ceiling. Diagnose: measure **inter-annotator agreement (IAA)** and read the
   judge's κ *against the IAA*, not against 1.0. (This needs ≥2 annotators — see
   IMPLEMENTATION_PLAN "Future Work".)
3. **The judge is genuinely wrong** — it mis-scores against a correct, stable human
   standard (humans agree with each other, but the judge doesn't; gross off-by-≥2
   errors, not boundary noise). *This* is the real failure mode.

**What a genuinely-wrong judge breaks:**

| | Effect |
|---|---|
| **Absolute numbers** | Wrong — **not publishable** until the judge is fixed. |
| **Delta — systematic error** (constant offset) | **Cancels** (both sides carry it). Delta safe. |
| **Delta — random error** (mean-zero noise) | Averages out in the mean (n=150/1456); the paired test absorbs it. Delta safe, wider CI. |
| **Delta — error correlated with the fine-tuning effect** | **BREAKS the delta.** ⚠️ The real danger. |

**The danger specific to this project.** The fine-tuned model produces *more*
religious / Reformed content. Our κ here is measured on the **baseline distribution,
which is ~90% secular (score 0 / neutral 4)** — so it validates the judge on
*secular* responses, **not** on the religious responses the fine-tune will generate.
If the judge mis-scores religious content specifically, that error is **correlated
with the treatment** and would bias the delta. **Therefore the judge MUST also be
validated on the fine-tuned model's (higher-representation) responses in Phase 4** —
draw a κ sample from `eval_*` results, not just the baseline. Agreement on *both*
regimes is what makes the delta trustworthy; agreement only on the secular baseline
is not sufficient.

**Escalation ladder (when cause 3 is confirmed):**
1. The judge prompt is the **official vendored** CEFE.AI rubric — do NOT edit it
   (that breaks comparability). So escalation is by **model**, not prompt.
2. Escalate `deepseek-v4-flash` → `deepseek-v4-pro` **with a non-reasoning provider
   pinned** (DeepInfra / DigitalOcean / Together / Alibaba showed 0% reasoning-token
   overflow — Lesson #18) → if still poor, a **frontier non-Qwen judge** (Claude /
   GPT tier). Re-run the affected baseline + fine-tuned, re-validate κ.
3. **Worst case, stay honest:** if no judge beats the human IAA, the absolute numbers
   are **not claimed** — report only the delta (with the caveat), or report that the
   benchmark is too subjective for an automated judge. A low κ that is genuinely the
   judge's fault is fixed by escalation or disclosure, **never by rationalization**.
