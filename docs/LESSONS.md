# Lessons Learned — OpenScriptura

Engineering + methodological lessons from building `qwen3-8b-reformed-pt-br`. Ops/GPU
lessons (vast.ai, CUDA, OOM) live in `CLAUDE.md` → "Lessons Learned (vast.ai / GPU)";
this file captures the **fine-tuning, evaluation, and anti-hallucination** lessons.

## Fine-tuning & hallucination

1. **Fine-tune teaches FORM, not FACTS.** Fine-tuning is the right tool for *voice,
   register, confessional lens, behavior* — not for storing facts/citations. Trying to
   bake facts into the weights produces confident fabrication.
   *Evidence:* the v0.1 model learned the Reformed voice perfectly but **hallucinated
   the TULIP acronym** in pt-BR ("Total Herança / Unção / Luta / Inflação"). It knew the
   doctrines; it fabricated the acronym. Facts belong in RAG (a separate serving layer),
   not in the fine-tune.

2. **Qualitative testing catches what aggregate metrics cannot.** The CEFEAI RR/CB
   numbers proved the model became "more religious/opinionated" but were blind to the
   TULIP hallucination, a repetition loop under greedy decoding, and over-accommodation
   to heterodox traditions. **Always read real model outputs before publishing.**

3. **Train ABSTENTION, not just style.** A confessional model needs trained epistemic
   humility: say "I'm not sure — verify" / "ask your pastor" instead of confabulating.
   This is as much an anti-hallucination mechanism as RAG. Pair it with the discrimination
   *state-what-is-verifiable / abstain-on-what-is-not* (a handful of verified specific
   facts is fine; a *pattern* of confident specifics drifts back toward fact-baking).

4. **RR and CB are mechanistically different.** Position-taking Q&A data moves **CB**
   (stance on religious topics) strongly, but barely moves **RR** (volunteering religion
   in neutral prompts). Raising RR requires *worldview-applied-to-secular-topics* data —
   and even then it should be an **implicit** frame, not religious content forced into
   every answer (which causes caricature / reflexive eisegesis, and games the metric).
   **Don't optimize the RR number; optimize authenticity.**

5. **The fine-tune's effect is language-dependent.** Trained in pt-BR, the model's
   religious/confessional behavior transfers most strongly in pt-BR; English shows the
   religiosity but a muddier direction. Lead the product story with the training language.

6. **"Improvement" on CEFEAI is framing-dependent.** CB treats bias as bad; for a
   deliberately confessional model, higher directional bias is the *goal*, not a
   regression. Always state which lens ("interpret, don't grade").

## Evaluation & comparability

7. **Hold the comparability lock.** Headline CEFEAI numbers are only meaningful if the
   exact same protocol (no system prompt, locked inference, single pinned judge, official
   vendored prompts) is applied to BOTH baseline and fine-tuned. The internal delta is the
   rigorous claim; absolute numbers are judge-dependent (κ-validated). See CLAUDE.md HARD RULE.

8. **Deployment generation settings ≠ eval settings.** Fixes like `repetition_penalty`
   belong in the *deployment* `generation_config` / model card — NOT in the headline eval,
   which must stay greedy (temp 0) to remain comparable to the baseline and the leaderboard.

## Tooling & process

9. **Tiered, content-addressed dataset + manual review pays off.** Curated corrections go
   in **Tier A** (manual pastoral review), built reproducibly from a reviewed JSON via
   `scripts/build_tier_a.py` (same `content_hash`/schema as B/C), and picked up
   automatically by `merge_dataset.py`. Each correction is traceable.

10. **An offline HTML review tool beats a CLI labeler.** A single self-contained HTML
    (question + drafted answer → edit/approve/reject → export JSON+MD) made pastoral
    review fast and auditable. Generalize it for **multiple validators**, reused for both
    Tier A content review AND LLM-as-judge (κ) labeling → inter-annotator agreement.
    Use native-speaker annotators per language track.

## v0.1.1 — what fine-tune can and cannot fix (the decisive lessons)

The v0.1 qualitative probe found 3 defects (TULIP acronym hallucination, factual
confabulation instead of abstention, over-accommodation to heterodox traditions). We
ran **two retrains** to fix them via fine-tune (Tier A at ~1.8%, then upsampled to
~12%). Both **failed**, and these lessons are the most important of the project:

11. **ALWAYS probe with the deployment system prompt.** The scary "failures" (the
    model inventing "DIOSSO", "Vincent Cheung", saying "mórmons são compatíveis") were
    largely an **artifact of probing WITHOUT a system prompt** — the model was trained
    with the system prompt in 100% of examples, so prompting without one is
    out-of-distribution. A clean 2×2 (system-prompt × repetition_penalty) proved the
    fix is the **system prompt**, not generation settings. Test in the deployment
    condition, or you'll chase ghosts (we burned ~$6 of GPU partly on this).

12. **Fine-tune does NOT learn facts — and upsampling makes it WORSE.** TULIP (a
    fact/acronym) stayed wrong after both retrains; upsampling the corrected example
    7× caused *more elaborate* confabulation (overfitting → "DIOSSO/Vincent Cheung"),
    not less. Re-confirms Lesson #1 emphatically: **facts belong in RAG + prompt, never
    in the weights.** Do not throw more fine-tune at a fact problem.

13. **Prompt delivers form-adjacent fixes; abstention needs RAG/guardrail.** With a
    production system prompt (TULIP mapping + anti-accommodation rule), **TULIP and
    over-accommodation are reliably fixed** (robust across phrasings). But **factual
    abstention is only inconsistently fixed by a prompt rule** — the model's prior to
    "answer helpfully with specifics" overrides it ~half the time, fabricating
    numbers/dates/citations. Reliable abstention + citation accuracy require **RAG
    (provide the fact) + a citation guardrail (block ungrounded claims)** — the
    deferred "System B". Documented as a known v0.1.1 limitation.

14. **Conclusion — stop fine-tuning for these; ship model + production prompt.** The
    fine-tune did its job (Reformed *form/voice/core doctrine* — all probes pass with
    the prompt). The 3 defects live in the prompt/RAG layer. v0.1.1 = same fine-tuned
    model + `configs/system_prompt_production.txt`; abstention/citations → System B
    roadmap. Model choice (v0.1 / Tier-A retrains) is low-stakes *with* the prompt.

15. **Reliability is layered — form (fine-tune) vs facts (RAG) vs guarantee (guardrail) —
    and a guardrail is CODE, not training.** A live test (*with* the production prompt)
    had the model compose a **prayer to a Catholic saint**, attribute it to **Spurgeon**,
    and cite **"WCF 21.3"** as endorsing saint-intercession — the **opposite** of the real
    Westminster Confession. One output, three failures (pro-Catholic prior +
    over-attribution + a fabricated, **polarity-inverted** citation). Lessons: (a) the
    prompt is **soft** mitigation; the **guardrail** is the **hard** guarantee — and the
    guardrail is **deterministic code (+ optional off-the-shelf NLI), never a model
    retrain** (retraining re-inherits the prior that caused the bug); (b) embedding
    similarity is **polarity-blind** (it can't tell "WCF *forbids* X" from "WCF *affirms*
    X"), so a text-only citation check can miss inversions → a small, hand-curated **KG**
    with *affirms/forbids* edges is a **staged** verifier backend, built only on a measured
    trigger, never a doctrinal authority. **Scope note:** all of this is **System B
    serving-layer / deployer-side work — out of scope for the fine-tune project**; v0.1.1
    ships as model + production prompt.
