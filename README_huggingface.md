# OpenScriptura

**Open LLMs for Protestant Theology — any tradition, any language.**

---

## Mission

OpenScriptura is an open-source research initiative that fine-tunes open language models (LLMs) with Protestant theological corpus across traditions and languages.

We build and publish models, datasets, and evaluation tools that anyone — researchers, developers, and ministries worldwide — can freely use to create trustworthy, theologically grounded, and confessionally faithful AI applications.

**Tradition is not a constraint.** Our first release is aligned with Reformed Protestant theology because that is where we started. Future releases will cover Lutheran, Anglican, Baptist, Pentecostal, Methodist, and other Protestant traditions. The pipeline, dataset schema, and evaluation methodology are designed to be tradition-agnostic and language-agnostic from day one.

**Language is not a constraint.** We start with Brazilian Portuguese. Tomorrow it could be English, Spanish, German, indigenous languages, or any other language where Protestant theology needs a voice.

> *"All Scripture is breathed out by God and profitable for teaching, for reproof, for correction, and for training in righteousness."*
> — 2 Timothy 3:16

---

## Why This Project Exists

Today's large language models systematically ignore or neutralize religious perspectives. The independent CEFEAI benchmark demonstrated that **100% of the 27 frontier models tested in 2026 produce 0% predominantly religious responses** to everyday ethical questions — including GPT-5, Claude, and Gemini.

OpenScriptura demonstrates that targeted fine-tuning can change this: transforming a model that ignores faith into one that responds with theological fidelity, pastoral clarity, and doctrinal precision — in any Protestant tradition and any language.

---

## Models

| Model | Base | Tradition | Language | Status | CEFEAI RR (baseline) | CEFEAI RR (fine-tuned) |
|---|---|---|---|---|---|---|
| [qwen3-8b-reformed-pt-br-v0.1](https://huggingface.co/openscriptura/qwen3-8b-reformed-pt-br-v0.1) | Qwen3-8B | Reformed | PT-BR | 🔄 final training pending | **4.7%** (no system prompt) | pending |
| qwen3-8b-lutheran-en-v0.1 | Qwen3-8B | Lutheran | EN | 📋 Planned | — |
| qwen3-8b-anglican-en-v0.1 | Qwen3-8B | Anglican | EN | 📋 Planned | — |
| gpt-oss-20b-v1.0 | GPT-OSS 20B | Multi-tradition | Multilingual | 📋 Planned | — |

> **Pipeline status:** dataset built (2,968 records); 2×2 LoRA sweep complete (winner **r=64, lr=2e-4**); final training, GGUF export, and evaluation scripts written and pending the A100 run. Evaluation headline = **no system prompt** (CEFEAI-comparable).

Model naming: `openscriptura/{base}-{tradition}-{lang}-{version}`

---

## Datasets

| Dataset | Tradition | Language | Examples | Status |
|---|---|---|---|---|
| [reformed-theology-v1](https://huggingface.co/datasets/openscriptura/reformed-theology-v1) | Reformed | PT-BR | **2,968** (839 Tier C + 2,129 Tier B; Tier A pending) → 2,873 train / 151 eval | 🔄 used in sweep; final train pending |
| lutheran-theology-v1 | Lutheran | EN | — | 📋 Planned |
| anglican-theology-v1 | Anglican | EN | — | 📋 Planned |

---

## Protestant Traditions

OpenScriptura is designed to serve the full breadth of Protestant Christianity:

| Tradition | Primary Standards | Status |
|---|---|---|
| **Reformed / Calvinist** | Westminster Confession, Canons of Dort, Heidelberg | 🔄 v0.1 — first release |
| **Lutheran** | Augsburg Confession, Luther's Catechisms, Book of Concord | 📋 Planned |
| **Anglican / Episcopal** | 39 Articles, Book of Common Prayer | 📋 Planned |
| **Baptist (Traditional)** | London Baptist Confession 1689, New Hampshire Confession | 📋 Planned |
| **Methodist / Wesleyan** | Wesley's Articles, Methodist Discipline | 📋 Planned |
| **Pentecostal** | Assemblies of God Statement of Fundamental Truths | 📋 Planned |
| **Congregationalist** | Savoy Declaration | 📋 Planned |
| **Other Protestant** | Community contributions welcome | 📋 Open |

Each tradition has its own confessional standards, its own corpus, and its own model variant. Models are clearly labeled by tradition — there is no attempt to blend incompatible doctrines.

---

## Benchmarks

All OpenScriptura models are evaluated on both public CEFEAI benchmarks:

**Religious Representation (AFB_RR)** — measures how often the model integrates a religious perspective in ethical responses. Scale 0–4, 150 questions.

**Conversion Bias (AFB_CB)** — measures whether the model treats conversion prompts symmetrically across faith traditions. 1,456 pairs, scale 1–7.

Results are published in each model's README with direct comparison against the untuned base model.

**Evaluation protocol (headline = no system prompt).** Both the raw baseline and the fine-tuned model are evaluated with **no system prompt** and identical inference settings (`temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`), so the reported lift is attributable to fine-tuning (the weights) and stays comparable to the prompt-free CEFEAI leaderboard. The raw baseline is RR 4.7% / CB 19.6%. We tested a "with system prompt" variant and rejected it as the headline: the Reformed prompt alone saturated the *raw* model to RR 99.3% / CB 87.8% (not leaderboard-comparable, and no headroom to show the fine-tuning effect). That variant is kept only as an opt-in deployment-behavior datapoint.

---

## Methodology

### Curation Pipeline

```
Tier C — Native Q&A (confessions & catechisms)  → high quality, zero ambiguity
Tier B — Sermons and theological texts           → LLM-annotated pairs
Tier A — Structured doctrinal topics             → systematic theology
```

Each example passes three validation layers:
1. **Cross-validation** between two independent annotator models
2. **Confessional Judge** — alignment with the tradition's primary standards
3. **Human pastoral review** — mandatory sampling before publication

### Fine-tuning

- **Method:** LoRA rank-64, alpha-128 (winner of a 2×2 sweep over rank∈{16,64} × lr∈{1e-4,2e-4}; rank dominated). Phase 2 sweep used 4-bit QLoRA on RTX 4090; Phase 3 final run uses full bf16 on A100 for a cleaner merge.
- **LR:** 2e-4, cosine, 10% warmup · **early stopping** (patience=5) to halt at the optimum
- **Framework:** TRL SFTTrainer (+ PEFT/bitsandbytes); GGUF export via llama.cpp (Q4_K_M / Q5_K_M / Q8_0)
- **Infrastructure:** vast.ai RTX 4090 (sweep) → A100 80GB (final)
- **Seed:** 42 (full reproducibility)
- **Total cost per model:** ~$11–16

---

## License

All models and datasets are published under **Apache 2.0** — free use including commercial, with attribution required.

---

## Contributing

Contributions are welcome across all Protestant traditions and languages. See [CONTRIBUTING.md](https://github.com/openscriptura/openscriptura) for details.

---

## Citation

```bibtex
@misc{openscriptura2026,
  title        = {OpenScriptura: Open LLMs for Protestant Theology},
  author       = {OpenScriptura Contributors},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/openscriptura}},
  note         = {Apache 2.0. Multi-tradition, language-agnostic pipeline for Protestant theology.}
}
```

---

*Soli Deo Gloria*
