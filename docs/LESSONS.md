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
