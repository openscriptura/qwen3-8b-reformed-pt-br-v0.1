# Why fine-tuning did NOT fix 3 specific defects — root-cause analysis

One of the most important findings of the project. The v0.1 qualitative probe found 3
defects — (a) **TULIP** acronym hallucinated, (b) the model **confabulates facts**
(dates/numbers/citations) instead of abstaining, (c) **over-accommodation** to heterodox
traditions ("mórmons são compatíveis"). We ran **two fine-tune retrains** to fix them
(Tier A at ~1.8% of train, then upsampled to ~12% with explicit TULIP drills). **Both
failed; upsampling made TULIP worse.** Meanwhile the Reformed *form/voice/core doctrine*
was excellent throughout. This document explains why — and it was, in hindsight, predictable.

> **One-line answer to "was it the raw model's data or the training process?"**
> Primarily the **raw model (pretraining) prior** — it overwhelms a small fine-tune.
> The **training process contributed** in specific, secondary ways (data composition,
> the all-examples-have-a-system-prompt design, and upsampling-induced overfitting).

## 1. Scale asymmetry: the pretraining prior dominates (primary cause)
Qwen3-8B was pretrained on **trillions of tokens** + RLHF alignment, creating deep,
strong priors:
- **TULIP** is an *English* acronym; in Portuguese the terms don't start with T-U-L-I-P,
  so the base model "forces" Portuguese words to the English letters and **confabulates**
  ("Livre", "Unigênito", and even a fabricated "DIOSSO").
- **RLHF rewards being balanced / non-exclusionary** → saying "this is NOT compatible"
  fights that reflex (→ over-accommodation).
- **RLHF rewards being helpful and specific** → inventing a number is "more helpful"
  than saying "I don't know" (→ confabulation instead of abstention).

Our fine-tune is **~3k examples**, LoRA **r=64 ≈ 0.5% of params**, ~2 epochs — a *drop*
against the pretraining ocean. **Fine-tune adjusts the surface; it does not overwrite
deep factual associations or RLHF-instilled behavioral reflexes.**

## 2. Fine-tune teaches FORM/distribution, not discrete FACTS (mechanism)
LoRA pushes the model toward the *distribution* of the training data (voice, register,
topic-conditioned tone). It does **not** write a fact into a deterministically-retrievable
location. "L = Limited Atonement = Expiação Limitada" needs the model to store **and
retrieve** the association; fine-tuning nudges the probability up slightly, but the base
model's competing association (L→"Livre"/"Limitado") stays strong — and under **greedy
decoding (argmax)** the wrong-but-higher-logit token wins, deterministically. This is
exactly why *form/voice/doctrine-in-prose* transferred beautifully (all core probes pass)
while the *pointwise fact* did not.

## 3. Data composition + the system-prompt design (training-process contribution)
- **A prose corpus, not a drill corpus.** Tier B (Spurgeon) + Tier C (confessions) are
  expository prose. The model learned to *speak* Reformed (it even says "segundo Spurgeon")
  but the corpus didn't contain the *target behaviors* (acronym mapping, abstention, firm
  refusal). 57–60 Tier A examples are too few and too late to become a prior.
- **The system prompt was in 100% of training examples.** So the model learned "behave
  Reformed *conditioned on that system prompt*." Probing **without** a system prompt is
  **out-of-distribution** → it collapses to base-model behavior (garbage). A clean 2×2
  (system-prompt × repetition_penalty) proved the apparent failures were largely a **test
  artifact**: *with* the production prompt, TULIP and over-accommodation are correct.

## 4. Why upsampling made TULIP WORSE (counter-intuitive but mechanical)
Repeating the same example ×7 means the model sees the **same sequence** many times →
**overfitting** to spurious surface patterns and degenerate recombination ("DIOSSO /
Vincent Cheung"). To teach a fact you need **diversity** (many phrasings), not
**repetition**. More identical-data ≠ better; it can *degrade* generalization (especially
out-of-distribution).

## 5. Decoding
The headline eval uses **greedy (temp 0)** for comparability. Greedy exposes the argmax —
if the wrong prior has the higher logit, the error appears deterministically.
`repetition_penalty`/sampling can mask *repetition loops* but never fixes the *fact*.

## Synthesis — the result was predictable and correct
| What we asked for | Right layer | Outcome |
|---|---|---|
| Voice / register / Reformed lens | **fine-tune** | ✅ success (core solid) |
| Discrete fact (TULIP) | **prompt + RAG** | failed in weights; fixed by production prompt |
| Abstention (vs the "be helpful" prior) | **RAG + guardrail** | inconsistent in prompt → System B |
| Firm refusal (vs the "accommodate" prior) | **prompt** | fixed by production prompt |

**It wasn't a training error — it was the wrong tool for 3 specific tasks.** Fine-tune
did what fine-tune does (form). Facts and behaviors that **contradict deep base-model
priors** need a **strong prompt + RAG + citation guardrail** (the deferred "System B").
This empirically confirms the project's architecture thesis: **fine-tune = form; RAG = facts.**

## Architectural implication — the reliability layers (and why a guardrail ≠ training)

The same root cause predicts a **fourth** manifestation we later reproduced live: asked
(with the production prompt) for a "prayer to Santo Expedito", the model **composed the
saint-prayer**, **attributed it to Spurgeon**, and **cited "WCF 21.3"** as saying the
saints intercede — the *opposite* of the real Westminster Confession (which forbids
prayer to saints). One output, three failures: a **pro-Catholic prior** + **over-
attribution** + a **fabricated, polarity-inverted citation**. Same lesson, sharper — the
prompt shapes *form*, not *facts*.

The fix is **layered**, and each layer is a different *kind* of thing:
- **Fine-tune** = form (done; weights **frozen**).
- **System prompt** = soft mitigation (TULIP + anti-accommodation hold; abstention/citation do not).
- **RAG** = makes the real text *available* (retrieve the actual WCF 21 — don't recall it).
- **Guardrail** = the **hard** guarantee, and it is **code, not training** — a
  deterministic verifier that blocks any reference/citation/attribution not matching the
  retrieved text. It does **not** retrain the model (training would re-inherit the prior
  that caused the bug); at most it uses an **off-the-shelf NLI** model for the
  entailment/polarity check.

**Why a Knowledge Graph is a *staged* pillar (not mandatory, not a vague "maybe").**
Embedding similarity is **polarity-blind**: "saints / intercede / pray" looks topically
*close* to the true WCF 21 (which *forbids* it), so a text-only citation check can miss
the **inversion**. A small, hand-curated, typed KG (confession → chapter.section →
normative claim, with an *affirms/forbids* polarity edge) turns that into an O(1)
deterministic check — the exact verifier backend if the text-only heuristic proves
insufficient. Build it only when a **measured trigger** fires (inversion false-negatives
above tolerance, cross-confession comparison, multi-hop grounding, governance/audit), and
only ever as a **subordinate, human-curated index — never a doctrinal inference engine**
(*magisterium creep*).

> **Scope note.** All of the above is **System B serving-layer / deployer-side work —
> out of scope for the fine-tune release.** This project's deliverable is the fine-tune
> (the Reformed tone), shipped now as v0.1.1 + `configs/system_prompt_production.txt`.

_See also: `docs/LESSONS.md` #1, #11–15; `IMPLEMENTATION_PLAN.md` Future Work §0 (System B)._
