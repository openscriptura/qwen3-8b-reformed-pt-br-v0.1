# Spec — Values Evaluation on GlobalOpinionQA (Inglehart–Welzel / WVS)

**Status:** Draft for planning · **Owner:** OpenScriptura · **Date:** 2026-06-29
**Type:** Research extension (v0.1.2 / paper) — **NOT part of the v0.1.1 release**
**Touches the shipped model?** No. This is an evaluation/analysis layer only.

> One-line: *Measure where the raw and fine-tuned Qwen3-8B sit on a peer-reviewed, public map of human values (Pew + World Values Survey, via Anthropic's GlobalOpinionQA), and quantify how far our fine-tune + production prompt move the model from the default "secular/liberal" cluster toward a religious/traditional (Brazilian, Protestant) profile.*

---

## 1. Current moment (where we are)

- **v0.1.1 shipped:** `qwen3-8b-reformed-pt-br-v0.1` (LoRA exp_c, ckpt-325) trained + evaluated; published to HF (safetensors with baked production `chat_template` v2, GGUF Q8, Modelfile, card).
- **Primary instrument so far = CEFE.AI** (AllFaith Benchmark: Religious Representation 0–4 + Conversion-Bias 1–7). Results in `docs/PHASE4_RESULTS.md`. **Limitation made explicit there:** absolute numbers are *judge-dependent* and CEFE.AI's exact protocol (judge model, settings) was unpublished — now partially revealed (Gemini 3.1 Pro + GPT-5.4, temp 1.0, brevity system prompt). Our **internal delta is rigorous**; absolute placement on their leaderboard is not provably identical.
- **New external corroboration:** *The Economist* (Briefing, print headline "Computational bias", 27 Jun 2026) tested 25 frontier models on the **World Values Survey** and found them clustered as **"Godless hippies"** (secular + self-expression), far from most (more traditional/religious) human populations — *"no model reflects the worldviews of most African or Muslim countries."* The Economist did **not** publish items, prompts, scoring, or model settings.
- **Gap this spec fills:** we want a **second, independent, peer-reviewed, reproducible** values instrument — beyond religion-specific CEFE — to validate that the fine-tune moves the model's *worldview*, anchored to the famous Inglehart–Welzel framework. The public **GlobalOpinionQA** dataset (Anthropic) gives us exactly this with a published method.

---

## 2. Problem statement

1. **Scientific:** Our worldview-shift claim currently rests on one instrument (CEFE, religion-specific) whose absolutes are judge-dependent. A single instrument is weak evidence for a paper.
2. **External:** The Economist's finding (models are secular, unlike most people) is the macro version of our motivation — but it is **not reproducible** (unpublished method).
3. **What we need:** a reproducible, citable measurement of *general values* (not just religion) that (a) places the raw model in the documented "secular/liberal" cluster and (b) shows our fine-tune + production prompt shifting it toward a religious/traditional, specifically Brazilian-Protestant, profile.

---

## 3. Goal, research questions, hypotheses

**Goal:** Quantify, on a public values instrument, the worldview position of 4 model configurations and the *internal* shift caused by (a) the fine-tune weights and (b) the production system prompt.

**Research questions**
- RQ1: On GlobalOpinionQA, which country/population does the **raw** Qwen3-8B most resemble? (Expect: USA / secular-Western — reproducing the Economist/Durmus "WEIRD/self-expression" finding.)
- RQ2: Does the **fine-tune** (weights only, no prompt) shift the model toward a more religious/traditional and/or more Brazilian profile? By how much, and is it significant?
- RQ3: How much additional shift does the **production system prompt** add (and does it saturate, as in CEFE)?
- RQ4: **EN vs pt-BR** — does the deployment language change the position (the Economist's language-conditioning effect)?

**Hypotheses**
- H1: raw/no-prompt ≈ closest to USA/secular cluster.
- H2: fine-tune (C1→C3) produces a measurable, significant shift toward traditional/religious and toward Brazil.
- H3: prompt (C1→C2, C3→C4) produces a large shift (likely saturating), consistent with CEFE Lesson #16.
- H4: the shift is **stronger in pt-BR** than EN (training language), mirroring our CEFE pt-BR finding.

---

## 4. Scope

**In scope**
- Dataset: `Anthropic/llm_global_opinions` (GlobalOpinionQA): 2,556 questions — `source=GAS` (Pew Global Attitudes, 2,203) + `source=WVS` (World Values Survey, 353). Report **WVS subset** (Economist-aligned) **and** full set.
- 4 configurations (2×2): weights {raw, fine-tuned} × system-prompt {none, production v2}.
- Languages: **EN** (anchor, comparable to literature) and **pt-BR** (product; via translation of `question`+`options`, same approach as the CEFE pt-BR track).
- Metrics: per-country similarity (1 − Jensen–Shannon distance), closest-country, plus optional Inglehart–Welzel 2-axis placement on the WVS subset.

**Out of scope**
- Any change to the released model (frozen).
- Redistributing the dataset (NC license — see §9).
- Claiming leaderboard-identical parity with The Economist (unpublished method) — we report *method-adherent* numbers + *rigorous internal deltas* (same comparability stance as CEFE).
- RAG/guardrails (these are values, not facts — System B is a separate track).

---

## 5. Solution / methodology

> **§5.0 is the DECIDED, panel-reviewed configuration (scientific-critical-thinking, 4-scientist panel — survey methodologist, LLM-eval methodologist, statistician/psychometrician, alignment/validity researcher; unanimous). It supersedes 5.1–5.6 where they differ.**

### 5.0 Evaluation configuration (decided — scientific-critical-thinking panel)

**Two critical verdicts**
- **An LLM judge does NOT apply to GlobalOpinionQA.** Each row carries a ground-truth human distribution (`selections`), so the construct is *distributional fit*, not prose quality. The published Durmus method scores the model's own probability mass over the given options (log-probs → simplex) and computes `1 − JSD` vs each country. Importing the CEFE.AI judge would be a **construct-validity error**, **break comparability** to the published method, and **re-introduce judge-dependence** (the very thing to avoid). The CEFE judge exists only because RR/CB are free-text with no answer key.
- **"Average the two highest, discard the lowest" is NOT rigorous.** It deletes the minimum every time → systematic **upward bias ≈ +0.42σ** (3 i.i.d. judges, normal: `E[θ̂] = μ + 0.423σ`), and the inflation **grows with inter-judge disagreement** (worst exactly in the OOD Reformed cells, confounding score with disagreement). It is a one-sided trimmed mean = least robust. **If a panel is ever used, use the MEDIAN of 3** (majority vote for discrete labels) + report κ/ICC. Never an upper-trimmed mean.

**Decided knobs**
| Knob | Decided | Why |
|---|---|---|
| Scoring | option **log-probs** over the given options; per-token length-normalized; score option **content** (strip A/B/C); renormalize → P_model | Durmus construct; deterministic; avoids short-option & letter-frequency artifacts |
| Extraction temp | **softmax T=1.0**, single forward pass (no sampling) | the model's natural distribution; temp-0 collapses to one-hot (inflates JSD), temp-1 sampling is needless Monte-Carlo of a distribution read analytically |
| Metric | **1 − JSD** (Jensen–Shannon *distance*, base-2, [0,1]) vs each country; per-country vector + GAS/WVS splits; robustness: JS-divergence, TVD | symmetric, bounded, finite under zero support |
| Position bias | **MANDATORY**: average option-probs over option-order permutations before JSD; report residual sensitivity; identical scheme all cells/langs | letter/position priors are the dominant MC internal-validity threat; cancels in the delta |
| Judge | **NONE** (headline) | ground-truth human distribution exists |
| Model temp/top_p/max/seed | **no sampling**; record seed=42 + pinned base-model snapshot for provenance (max/seed otherwise irrelevant) | distribution read analytically from logits |
| System prompt (factor under test) | **C1/C3 = none; C2/C4 = production Reformed prompt** (`configs/system_prompt_production.txt`) | the legitimate contrast = weights × Reformed-prompt |
| "Be concise and brief" | **DROP** (or all-4-cells-or-none) | CEFE free-text artifact; not in Durmus template; perturbs option logits = uncontrolled confound |
| Prompt language | PT prompt for the pt-BR track; EN-translated prompt for the EN track (flag as non-identical treatment) | a PT prompt over EN questions confounds language with treatment |
| Task template | canonical **Durmus MC template, verbatim, identical in all 4 cells** | only deliberate variation = weights × prompt |
| Invalid/missing extraction | **EXCLUDE; never coerce to neutral/uniform**; retry transient transport only (backoff 2/4/8s or 5×); **listwise-identical exclusion across all 4 cells**; report rate, gate ~2%, χ² homogeneity | coercion fabricates/centers data; common item set keeps the 2×2 unconfounded; differential dropout is a validity threat |
| Runs | **1** (deterministic) | no sampling variance to average |
| Tie-break / rounding | **none** | continuous scores |
| Questions/scales | **official Durmus** (2,556 = GAS 2,203 + WVS 353); report GAS/WVS separately | WVS split = closest (still non-identical) tie to the unpublished Economist test |
| Statistics | paired **Wilcoxon** on per-question (1−JSD) deltas for 3 pre-registered primaries (weights, prompt, interaction) + bootstrap 95% CI + effect size (median Δ, rank-biserial); per-country = exploratory → **BH-FDR**; report effect size (n=2,556 inflates p) | within-item paired; JSD bounded/non-normal |
| Target country | **Brazil primary** (+1 documented comparison), pre-registered | avoid post-hoc cherry-picking the flattering country |

**Judge — only IF a separate, clearly-labelled, NON-comparable free-text variant is ever run** (and only if the serving path cannot expose option log-probs, or to characterize "how the chatbot talks"): temp=0.0 / top_p=1.0 / max_tokens=1024 / seed=42 / thinking-off / 1 call / pinned dated **non-Qwen** snapshot; **median-of-3** (majority vote for labels) + κ/ICC; prefer a **deterministic** prose→option mapper (constrained decode / regex) over an LLM judge. **Never the headline.**

**Hard prerequisite:** confirm the serving stack exposes per-option logprobs (local HF/transformers forward pass, or vLLM with logprobs); run **all 4 cells on one identical stack**. If logprobs are unavailable, the published method cannot be run — say so; do **not** substitute a judge and call it the Durmus number.

**Comparability & validity**
- **Rigorous (internal):** the 2×2 deltas (weights, prompt, interaction) per track — every nuisance factor byte-identical, only the manipulated factor varies, paired by question. *This is the strong claim.*
- **Method-adherent (absolute):** per-country `1 − JSD` placement is faithful to the *published* Durmus method (verify JS-distance vs divergence, log base, aggregation) — comparable only to numbers computed the same way, with caveats.
- **NOT comparable:** The Economist's 25-model WVS chart (method unpublished → exact parity impossible; cite as motivation only). pt-BR absolutes are not comparable to EN (translation changes the stimulus); only pt-BR's internal delta is rigorous.
- **Construct caveat (state up front):** GlobalOpinionQA was built for sampled human populations; an LLM has none. "`1 − JSD` to country X" measures token-distribution *resemblance*, not that the model "holds" those values. A confessional Reformed model *should* be a general-population outlier — **low similarity is expected, not a defect.**

---

### 5.1 Configurations (comparability lock — same discipline as CEFE HARD RULE)
| ID | Weights | System prompt |
|----|---------|---------------|
| C1 | raw Qwen3-8B | none |
| C2 | raw Qwen3-8B | production v2 |
| C3 | fine-tuned (v0.1.1) | none |
| C4 | fine-tuned (v0.1.1) | production v2 |

Identical inference procedure on all four; **only weights/prompt differ.** Clean contrasts: **C1→C3 = fine-tune effect (weights)**, **C1→C2 / C3→C4 = prompt effect**.

### 5.2 Per-question scoring (Durmus et al. method)
For each question, present `question` + enumerated `options`; obtain the model's **distribution over options**:
- **Primary:** **option log-probabilities** — score each option by the summed token log-prob, softmax-normalize → a proper distribution (matches Anthropic's approach; deterministic given weights; local Qwen, free).
- **Robustness control — option-order permutation:** LLMs have known multiple-choice position bias. Evaluate each question over **k randomized option orderings** and average → removes position artifacts. (Vary the permutation seed by question index.)
- **Cross-check:** a "verbalized" single-pick at temperature 0 (argmax letter) as a sanity comparison to the log-prob distribution.

### 5.3 Similarity to human populations
- For each question, compute **Sim(config, country) = 1 − JSD(p_model, p_country)** where `p_country` comes from the dataset's `selections[country]`.
- Average across questions (per subset: WVS, GAS, all) → **similarity(config, country)**.
- Report: **closest country**, full ranking, and **named anchors**: Brazil (deployment), USA (secular-Western reference), plus a derived **"Protestant/traditional" reference** (mean of high-religiosity Protestant-majority countries available in `selections`).

### 5.4 Inglehart–Welzel placement (optional, WVS subset)
- Compute the two axes (*traditional↔secular-rational*, *survival↔self-expression*) from the WVS items using the published Welzel item-to-axis mapping; place the 4 configs + country anchors on the **Inglehart–Welzel Cultural Map** (the *Cultural Bias / PNAS Nexus 2024* method). Deliver the plot.

### 5.5 Aggregation & statistics (same rigor as CEFE)
- Bootstrap 95% CIs over questions for each similarity score.
- **Paired** config-vs-config tests on per-question similarity-to-Brazil and similarity-to-USA (Wilcoxon signed-rank) + effect sizes (rank-biserial).
- Report the 2×2 cleanly: weights effect, prompt effect, and interaction.

### 5.6 Comparability stance (honesty)
- **Internal 2×2 deltas = rigorous** (identical procedure, only weights/prompt differ).
- **Absolute country placement = method-adherent**, depends on scoring choices (log-prob vs verbalized) and is **not provably identical** to The Economist's chart (unpublished) — but is aligned to the **public Durmus et al. method**. State this in the report, exactly as we do for CEFE absolutes.

---

## 6. Scenarios & consequences (what each result means)

| Scenario | Observation | Interpretation / consequence |
|---|---|---|
| **S1 (expected)** | C1 closest to USA/secular; C3/C4 shift toward Brazil + traditional | Validates the worldview shift on a 2nd, peer-reviewed instrument → strong paper result; reproduces Economist/Durmus for raw, demonstrates our antithesis for FT. |
| **S2 (prompt saturates)** | C2 already shifts strongly; C4 ≈ C2 | Consistent with CEFE Lesson #16 (prompt dominates). Headline = **C1→C3 weights-only delta** as the clean fine-tune effect. |
| **S3 (null/weak weights)** | C3 ≈ C1 on general values | Finding: fine-tune shifts *religion-specific* stance (CEFE) but **does not generalize** to broad WVS values without the prompt → publishable nuance; reinforces "fine-tune = form/tone, narrow." |
| **S4 (language)** | pt-BR shift > EN shift | Confirms language-conditioning (Economist) + our pt-BR CEFE finding; supports leading the product story in pt-BR. |
| **S5 (off-axis)** | FT moves toward *traditional* but not toward any single country | Reformed worldview is a *distinct* profile (traditional ethics + high vocation/education) — itself an interesting cultural-map result. |

**Broader consequences**
- **Paper:** two convergent instruments (CEFE religion-axis + GlobalOpinionQA values-map) >> one.
- **Ethical framing:** we do **overtly and transparently** (labeled confessional, opt-in, *ministerium non magisterium*) what the Economist warns about when done **covertly** (embedding worldview to sway opinion). This belongs in the write-up.
- **Reusable harness:** `09_globalopinions_eval.py` becomes a standing values-probe for future models/traditions.

---

## 7. Deliverables
- `scripts/09_globalopinions_eval.py` — fetch dataset at runtime, filter WVS/GAS, score the 4 configs (log-prob + permutation control), compute per-country similarity + (optional) IW axes, write results + report.
- `results/globalopinions_{en,ptbr}_{C1..C4}_*.jsonl` + `*_summary.json` (gitignored, like CEFE).
- `reports/globalopinions_report.{md,html}` + Inglehart–Welzel map plot.
- A results section (for `docs/` and the paper) + a short note added to README motivation.
- `scripts/translate_globalopinions.py` (pt-BR track) — translate only `question`+`options`, keep `selections`/`source` verbatim (mirrors `translate_benchmark.py`).

## 8. Implementation plan (milestones)
1. **M1 — Data & schema:** runtime download; confirm real `selections` format (dict country→list of %); split WVS/GAS; pick country anchor list.
2. **M2 — Scoring core:** local log-prob option scorer for Qwen (transformers); option-permutation control; verbalized cross-check.
3. **M3 — Metrics:** JSD similarity, per-country aggregation, bootstrap CIs, paired tests.
4. **M4 — Run 4 configs (EN):** produce summaries + closest-country + Brazil/USA deltas.
5. **M5 — IW map (WVS subset):** axis computation + plot.
6. **M6 — pt-BR track:** translate, rerun 4 configs, compare EN vs pt-BR.
7. **M7 — Report + paper section + README note.**

### 8.1 Implementation blueprint — `scripts/09_globalopinions_eval.py` (instructions; NOT yet built)

> Build **only on explicit approval**. Implements §5.0 verbatim. Follow project conventions: `NN_name.py`, `sys.stdout.reconfigure(encoding="utf-8")`, `PROJECT_ROOT`, `--dry-run`, reuse `scripts/utils/`.

**Step 0 — SMOKE TEST FIRST (hard gate, `--smoke`).** Before any full run, confirm the local serving stack can extract **per-option log-probabilities** from the merged Qwen (HF/transformers forward pass, or vLLM with `logprobs`). On 2–3 sample questions, print each option's length-normalized log-prob and the renormalized `P_model`. **If log-probs cannot be extracted, STOP** — the published Durmus method cannot be run; do **not** substitute a judge. No full eval proceeds until smoke passes.

**CLI.** `--lang {en,ptbr}` · `--cell {C1,C2,C3,C4}` (or `--all-cells`) · `--source {wvs,gas,all}` · `--smoke` · `--dry-run` · `--limit N` · `--out results/`. One identical inference stack for all cells.

**Data (runtime, NOT committed).** Download `Anthropic/llm_global_opinions` via `datasets`/`huggingface_hub` to a gitignored cache; never add to git (NC license, §9). Filter by `source`. Parse `selections` (dict `country→[%...]`) and `options` into a list of **content strings** (strip leading `A/B/C.` labels).

**Core functions (signatures + behavior):**
- `parse_options(options_field) -> list[str]` — option **content** strings only (no letters).
- `score_options(model, tok, question, options, system_prompt) -> np.ndarray` — for each **option-order permutation** (all perms if ≤4 options, else a fixed seeded set): build the canonical **Durmus MC template** + optional `system_prompt`, compute each option's **summed-token, length-normalized** log-prob over the option content; **softmax(T=1.0)** over options → a per-permutation simplex; **average** the simplices across permutations → `P_model`. Single deterministic forward pass per permutation; **no sampling**. Record residual order-sensitivity (variance across perms).
- `js_distance(p, q) -> float` — Jensen–Shannon **distance** (base-2, bounded [0,1]); similarity = `1 − JSD`. Also compute raw JS-divergence + TVD for robustness.
- `per_country_similarity(P_model, selections) -> dict[country,float]` — `1 − JSD(P_model, P_country)` for each country with valid support; record coverage.
- `run_cell(lang, cell, source) -> records` — loop questions; **exclude** items where log-probs can't be extracted (never coerce); retry only transient transport errors (backoff 2/4/8s or tenacity 5×).

**Comparability lock (enforce in code).** Take the **listwise intersection** of items answerable in **all 4 cells** before any aggregation (common item set → unconfounded paired 2×2). The only thing differing across cells = weights (C3/C4 vs C1/C2) and the Reformed system prompt (C2/C4). Assert template/option-set/permutation-scheme byte-identical across cells.

**Aggregation & stats.** Per-question `1 − JSD` to the **pre-registered** target country (Brazil + 1 comparison). Three pre-registered primaries via **paired Wilcoxon signed-rank** on per-question deltas: weights main effect (C1→C3), prompt main effect (C1→C2 & C3→C4), interaction; **bootstrap-over-questions 95% CI** + effect size (median paired Δ, rank-biserial/Cliff's δ). Per-country similarities = **exploratory** → Benjamini–Hochberg FDR. Report **effect sizes** prominently (n=2,556 makes trivial diffs "significant"). Report exclusion rate per cell + χ² homogeneity test.

**Outputs (gitignored, like CEFE).** `results/globalopinions_{lang}_{cell}_{source}.jsonl` + `*_summary.json` (record: scoring=logprob, T=1.0, permutation scheme, JSD base, exclusions, pinned snapshot, seed). `reports/globalopinions_report.{md,html}` + Inglehart–Welzel map plot (optional, WVS subset). pt-BR via `scripts/translate_globalopinions.py` (translate only `question`+`options`; keep `selections`/`source` verbatim).

**Pre-registration (freeze before running).** Scoring rule, softmax T, JSD definition + log base, permutation scheme, country aggregation, listwise-exclusion rule, paired tests, and Brazil(+1) as primary target — written down and unchanged after seeing results (guards against HARKing / cherry-picking).

**Do NOT, on the headline path:** add an LLM judge; coerce invalid items to neutral; use temp-0 greedy or temp-1 sampling for the distribution; add "Be concise and brief"; commit the dataset; place pt-BR absolutes next to EN or any absolute next to The Economist's chart.

## 9. Licensing & data governance
- **Model:** Qwen3-8B + our fine-tune = **Apache-2.0** (free/commercial). No issue.
- **Dataset:** `Anthropic/llm_global_opinions` = **cc-by-nc-sa-4.0 (NonCommercial)**, because it repackages Pew + WVS, whose source terms are *free-for-research, no-redistribution*. **Fine for evaluation + paper** (non-commercial research use, with citation); **must NOT** be redistributed or bundled commercially.
- **Repo hygiene:** **do NOT commit the dataset.** Download at runtime (as we treat model weights / generated `data/cefeai`); cite the source. Keeps the OpenScriptura repo Apache-clean.
- **Fully-permissive alternative (if ever needed):** build our own item list from the **public WVS questionnaire** + **WVS-published Inglehart–Welzel country coordinates**; our code stays Apache-2.0; cite WVS; no microdata redistribution. (More work; not required for this spec.)

## 10. Risks & mitigations
| Risk | Mitigation |
|---|---|
| MC **position bias** skews distributions | option-order **permutation averaging** (§5.2) |
| Log-prob extraction subtlety (multi-token options, tokenizer) | sum token log-probs over the full option string; verbalized cross-check |
| **pt-BR translation** shifts comparability | EN stays the anchor; pt-BR is the secondary/product track (same stance as CEFE) |
| Country coverage gaps in `selections` (not all countries answer every Q) | per-question intersection of available countries; report coverage |
| NC license creep into repo | runtime download, no commit, cite (§9) |
| Over-claiming parity with Economist | report internal deltas as rigorous; absolutes as method-adherent (§5.6) |

## 11. Success criteria (acceptance)
- All 4 configs scored on EN (WVS + GAS) with bootstrap CIs and paired C1→C3 / C1→C2 tests.
- A clear, defensible statement of: raw model's closest population; the fine-tune (weights) shift toward Brazil/traditional with significance + effect size; the prompt's additional effect.
- Inglehart–Welzel map placing the 4 configs vs country anchors.
- Honest comparability caveats stated; dataset not committed; sources cited.

## 12. Open questions
- Similarity metric: 1−JSD (Durmus) vs other (1−Wasserstein / cosine)? Default JSD; report sensitivity.
- Which exact country set defines the "Protestant/traditional" reference?
- pt-BR: run full 2,556 or WVS-subset only (cost is ~zero local, so likely full)?
- Follow-up to CEFE still pending (how they combine Gemini 3.1 Pro + GPT-5.4) — orthogonal but related to our overall comparability story.

---

## 13. References

**Primary instrument**
- Durmus, E., et al. (2023). *Towards Measuring the Representation of Subjective Global Opinions in Language Models.* arXiv:2306.16388. Dataset: `Anthropic/llm_global_opinions` (GlobalOpinionQA), 2,556 Q (Pew GAS 2,203 + WVS 353), cc-by-nc-sa-4.0. https://arxiv.org/abs/2306.16388 · https://huggingface.co/datasets/Anthropic/llm_global_opinions

**Framework & corroborating science**
- Inglehart, R., & Welzel, C. *World Values Survey / Inglehart–Welzel Cultural Map.* World Values Survey Association. https://www.worldvaluessurvey.org
- Tao, Y., Viberg, O., Baker, R. S., & Kizilcec, R. F. (2024). *Cultural bias and cultural alignment of large language models.* PNAS Nexus 3(9): pgae346. arXiv:2311.14096. (Plots LLMs on the IW map; finds self-expression/Protestant-European tilt.)
- Santurkar, S., et al. (2023). *Whose Opinions Do Language Models Reflect?* (OpinionQA). arXiv:2303.17548.
- Kirk, H. R., et al. (2024). *The PRISM Alignment Dataset.* NeurIPS 2024. arXiv:2404.16019. (Prompts CC-BY-4.0; different instrument.)
- Zhao, et al. (2024). *WorldValueBench* (WVS-derived, 64 countries).
- *Exploring LLMs on Cross-Cultural Values in Connection with Training Methodology.* arXiv:2412.08846.
- *Culturally Grounded Personas in LLMs.* arXiv:2601.22396 (2026).
- *Cultural Value Alignment via Latent Activation Steering.* arXiv:2605.26365 (2026; tests Qwen).
- Fisher, J., et al. *Biased AI can Influence Political Decision-Making.* arXiv:2410.06415.

**Popular-press motivation**
- The Economist (2026-06-27 print; online 2026-06-25). *AI models' values are very different from most people's* (print: "Computational bias"). Briefing (unsigned, per house style). https://www.economist.com/briefing/2026/06/25/ai-models-values-are-very-different-from-most-peoples

**Our prior work (internal)**
- `docs/PHASE4_RESULTS.md` (CEFE baseline→fine-tuned scorecard, EN + pt-BR).
- `docs/EVALUATION_PROTOCOL.md` (comparability lock; judge/settings rationale).
- `docs/WHY_FINETUNE_LIMITS.md` (fine-tune = form, not facts; reliability layers).
- `configs/system_prompt_production.txt` (production prompt v2 — the C2/C4 system prompt).
