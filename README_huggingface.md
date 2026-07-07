# OpenScriptura

**LLMs abertos para a Teologia Protestante — qualquer tradição, qualquer idioma.**
**Open LLMs for Protestant Theology — any tradition, any language.**

**🌐 Idioma / Language: [🇧🇷 Português](#-português) · [🇬🇧 English](#-english)**

---

<a id="-português"></a>
## 🇧🇷 Português

### Missão

A OpenScriptura é uma iniciativa de pesquisa open-source que faz fine-tuning de modelos de linguagem (LLMs) abertos com corpus teológico protestante, atravessando tradições e idiomas.

Construímos e publicamos modelos, datasets e ferramentas de avaliação que qualquer pessoa — pesquisadores, desenvolvedores e ministérios no mundo todo — pode usar livremente para criar aplicações de IA confiáveis, teologicamente fundamentadas e fiéis às confissões.

**Tradição não é uma limitação.** Nosso primeiro lançamento é alinhado à teologia reformada porque foi por aí que começamos. Lançamentos futuros cobrirão as tradições luterana, anglicana, batista, pentecostal, metodista e outras. O pipeline, o esquema de dados e a metodologia de avaliação são, desde o primeiro dia, agnósticos quanto à tradição e ao idioma.

**Idioma não é uma limitação.** Começamos pelo português brasileiro. Amanhã pode ser inglês, espanhol, alemão, línguas indígenas ou qualquer outro idioma onde a teologia protestante precise de voz.

> *"Toda a Escritura é inspirada por Deus e útil para o ensino, para a repreensão, para a correção e para a instrução na justiça."* — 2 Timóteo 3:16

### Por que este projeto existe

Os grandes modelos de linguagem de hoje sistematicamente ignoram ou neutralizam perspectivas religiosas. O benchmark independente CEFEAI demonstrou que **100% dos 27 modelos de fronteira testados em 2026 produzem 0% de respostas predominantemente religiosas** a perguntas éticas do dia a dia — incluindo GPT-5, Claude e Gemini.

A OpenScriptura mostra que o fine-tuning direcionado pode mudar isso: transformar um modelo que ignora a fé em um que responde com fidelidade teológica, clareza pastoral e precisão doutrinária — em qualquer tradição protestante e qualquer idioma.

### O que o fine-tune entrega (e seus limites)

**Objetivo alcançado:** o objetivo do projeto — *um modelo que responde a partir da cosmovisão reformada* — foi entregue. Voz, registro e doutrina central são solidamente reformados (predestinação, as Cinco Solas, monergismo, justificação pela justiça imputada, perseverança dos santos, teologia do pacto). O modelo é distribuído com um system prompt de produção (`configs/system_prompt_production.txt`) que carrega os padrões confessionais.

**Limitações (por design — o fine-tuning molda a *forma*, não os *fatos*):** trate o modelo como **auxílio de estudo, não como fonte de verdade factual**, e use-o **sempre com o system prompt de produção** (sem ele, o comportamento é fora-da-distribuição). Verifique referências bíblicas, datas, números e citações em fontes primárias. Para garantia factual em produção, recomenda-se combinar com **RAG + guardrails** (verificação determinística de citações — *código, não re-treino do modelo*) — **fora do escopo deste release de fine-tune**, responsabilidade de quem faz o deploy. Ele **não substitui a Escritura, um pastor ordenado ou a igreja local** — aponta para fora de si, para eles (*ministerium, non magisterium*).

### Modelos

| Modelo | Base | Tradição | Idioma | Status | CEFEAI RR (baseline) | CEFEAI RR (fine-tuned) |
|---|---|---|---|---|---|---|
| [qwen3-8b-reformed-pt-br-v0.1](https://huggingface.co/openscriptura/qwen3-8b-reformed-pt-br-v0.1) | Qwen3-8B | Reformada | PT-BR | ✅ treinado + avaliado (vai com system prompt de produção) | **0,147/4** (12,7% any-rep) | **EN 0,227/4** (n.s.) · **pt-BR 0,08→0,62** ✅ |
| qwen3-8b-lutheran-en-v0.1 | Qwen3-8B | Luterana | EN | 📋 Planejado | — |
| qwen3-8b-anglican-en-v0.1 | Qwen3-8B | Anglicana | EN | 📋 Planejado | — |

> **Status (2026-06-15):** dataset construído; sweep LoRA 2×2 (vencedor **r=64, lr=2e-4**); **Fase 3 treinada** (H100, exp_c, eval_all_loss 0,6546) e **Fase 4 avaliada nos dois tracks** (âncora EN + produto pt-BR; placar em `docs/PHASE4_RESULTS.md`). **Meta de viés confessional (CB) atingida** (Any-Bias 20%→~64%); **representação religiosa em pt-BR subiu 0,08→0,62** (efeito grande). Avaliação usa o **juiz oficial da CEFE.AI**, headline = **sem system prompt**.

**Resultados Phase 4** (delta interno rigoroso; absolutos dependentes do juiz):

| Track | Métrica | base → fine-tuned | Δ | p (Wilcoxon) | signif. |
|---|---|---|---|---|---|
| EN | RR (0–4) | 0,147 → 0,227 | +0,080 | 0,152 | ✗ |
| EN | CB (1–7) | 3,694 → 3,499 | −0,195 | 3,9e−6 | ✓ |
| pt-BR | **RR (0–4)** | 0,081 → **0,617** | **+0,537** | **3,7e−6** | ✓✓ grande |
| pt-BR | CB (1–7) | 3,911 → 3,972 | +0,059 | 0,308 | ✗ |

Unidades do plano: **CB Any-Bias 20%→~64%** (neutralidade 80%→~36%); **RR Any-Rep** EN 12,7%→13,3% · pt-BR 6,7%→**20,8%**. **Veredito:** meta confessional (CB) **atingida/superada**; RR **melhorou** (efeito grande em pt-BR) mas **abaixo da meta >60%**.

### Benchmarks

Todos os modelos são avaliados nos dois benchmarks públicos da CEFEAI:
- **Representação Religiosa (AFB_RR)** — com que frequência o modelo integra uma perspectiva religiosa em respostas éticas. Escala 0–4, 150 questões.
- **Viés de Conversão (AFB_CB)** — se o modelo trata prompts de conversão de forma simétrica entre tradições. 1.456 pares, escala 1–7.

**Viés por religião (CB) — destino, antes → depois** (track EN, sem system prompt; **<4 = encoraja** converter PARA aquela tradição, **>4 = desencoraja**):

| Tradição (destino) | raw | fine-tuned | Δ |
|---|---|---|---|
| Latter-day Saint | 3,93 | 2,39 | −1,54 |
| Católica | 3,88 | 2,95 | −0,92 |
| Bahá'í | 3,67 | 2,88 | −0,79 |
| Sikh | 3,62 | 3,10 | −0,52 |
| Protestante Evangélica | 3,95 | 3,47 | −0,48 |
| Judaica | 3,35 | 2,88 | −0,46 |
| Protestante (histórica) | 3,54 | 3,23 | −0,31 |
| Muçulmana Xiita | 3,86 | 3,66 | −0,19 |
| Muçulmana Sunita | 3,98 | 3,87 | −0,12 |
| Budista | 3,76 | 3,70 | −0,06 |
| Hindu | 3,96 | 4,30 | +0,34 |
| Ateu | 2,99 | 3,50 | +0,51 |
| Agnóstico | 2,98 | 3,88 | +0,89 |
| Testemunha de Jeová | 4,26 | 5,17 | +0,91 |

O fine-tune tornou o modelo mais **direcional por tradição**: passou a **desencorajar** conversão à descrença (Ateu, Agnóstico) e a grupos heterodoxos (Testemunhas de Jeová). Detalhe completo (incl. pt-BR) em `reports/phase4_comparison.html`.

**Protocolo.** Pontuamos com os **prompts de juiz oficiais da CEFE.AI** (carregados verbatim) nas **questões originais** (verificadas idênticas). Tanto o baseline raw quanto o fine-tuned são avaliados **sem system prompt** e com settings idênticos (`temperature=0.0, seed=42, enable_thinking=False, max_tokens=1024`), de modo que o ganho é atribuível ao fine-tuning. O **delta interno é rigoroso**; os números absolutos são protocol-adherent mas dependentes do juiz (a CEFE.AI não publica o juiz). **Dois tracks de idioma:** EN (âncora científica, comparável ao leaderboard) e pt-BR (realista de deploy; absolutos NÃO comparáveis ao leaderboard inglês). Juiz validado contra rótulos humanos (κ de Cohen quadrático) nos dois idiomas.

### Como usar (local)

O modelo **roda localmente** (não precisa de HuggingFace para testar). Use sempre o **system prompt de produção** (`configs/system_prompt_production.txt`) — sem ele, o comportamento é fora-da-distribuição. Caminho recomendado em CPU: importar no **Ollama** (`ollama create ... --quantize q4_K_M`) ou **LM Studio** via GGUF.

### Licença, Contribuição e Citação

- **Licença:** **Apache 2.0** — uso livre, inclusive comercial, com atribuição.
- **Contribuições** são bem-vindas em todas as tradições e idiomas — veja [CONTRIBUTING.md](https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1/blob/main/CONTRIBUTING.md).
- Citação: veja o bloco BibTeX na seção em inglês abaixo.

---

<a id="-english"></a>
## 🇬🇧 English

### Mission

OpenScriptura is an open-source research initiative that fine-tunes open language models (LLMs) with Protestant theological corpus across traditions and languages.

We build and publish models, datasets, and evaluation tools that anyone — researchers, developers, and ministries worldwide — can freely use to create trustworthy, theologically grounded, and confessionally faithful AI applications.

**Tradition is not a constraint.** Our first release is aligned with Reformed Protestant theology because that is where we started. Future releases will cover Lutheran, Anglican, Baptist, Pentecostal, Methodist, and other Protestant traditions. The pipeline, dataset schema, and evaluation methodology are tradition-agnostic and language-agnostic from day one.

**Language is not a constraint.** We start with Brazilian Portuguese. Tomorrow it could be English, Spanish, German, indigenous languages, or any other language where Protestant theology needs a voice.

> *"All Scripture is breathed out by God and profitable for teaching, for reproof, for correction, and for training in righteousness."* — 2 Timothy 3:16

### Why This Project Exists

Today's large language models systematically ignore or neutralize religious perspectives. The independent CEFEAI benchmark showed that **100% of the 27 frontier models tested in 2026 produce 0% predominantly religious responses** to everyday ethical questions — including GPT-5, Claude, and Gemini. OpenScriptura demonstrates that targeted fine-tuning can change this — with theological fidelity, pastoral clarity, and doctrinal precision, in any Protestant tradition and language.

### What the fine-tune achieves (and its limits)

**Goal achieved:** the project goal — *a model that answers from the Reformed worldview* — is delivered. Voice, register, and core doctrine are solidly Reformed (predestination, the Five Solas, monergism, justification by imputed righteousness, perseverance of the saints, covenant theology). The model ships with a production system prompt (`configs/system_prompt_production.txt`) carrying the confessional standards.

**Limitations (by design — fine-tuning shapes *form*, not *facts*):** treat the model as a **study aid, not a source of factual ground truth**, and always serve it **with the production system prompt** (without it, behavior is out-of-distribution). Verify Scripture references, dates, numbers, and citations against primary sources. For factual guarantees in production, pair it with **RAG + guardrails** (deterministic citation verification — *code, not a model retrain*) — the **deployer's responsibility, out of scope for this fine-tune release**. It is **not a substitute for Scripture, an ordained pastor, or the local church** — it points outward to them (*ministerium, non magisterium*).

### Models

| Model | Base | Tradition | Language | Status | CEFEAI RR (baseline) | CEFEAI RR (fine-tuned) |
|---|---|---|---|---|---|---|
| [qwen3-8b-reformed-pt-br-v0.1](https://huggingface.co/openscriptura/qwen3-8b-reformed-pt-br-v0.1) | Qwen3-8B | Reformed | PT-BR | ✅ trained + evaluated (ships with production prompt) | **0.147/4** (12.7% any-rep)✅ | **EN 0.227/4** (n.s.) · **pt-BR 0.08→0.62** ✅ |
| qwen3-8b-lutheran-en-v0.1 | Qwen3-8B | Lutheran | EN | 📋 Planned | — |
| qwen3-8b-anglican-en-v0.1 | Qwen3-8B | Anglican | EN | 📋 Planned | — |

> **Pipeline status (2026-06-15):** dataset built; 2×2 LoRA sweep (winner **r=64, lr=2e-4**); **Phase 3 trained** (H100, exp_c, eval_all_loss 0.6546) and **Phase 4 evaluated both tracks** (EN anchor + pt-BR product; scorecard `docs/PHASE4_RESULTS.md`). **CB confessional goal met** (Any-Bias 20%→~64%); **pt-BR religious representation up 0.08→0.62** (large effect). Evaluation uses the **official CEFE.AI judge**, headline = **no system prompt**.

**Phase 4 results** (internal delta rigorous; absolute numbers judge-dependent):

| Track | Metric | base → fine-tuned | Δ | p (Wilcoxon) | sig. |
|---|---|---|---|---|---|
| EN | RR (0–4) | 0.147 → 0.227 | +0.080 | 0.152 | ✗ |
| EN | CB (1–7) | 3.694 → 3.499 | −0.195 | 3.9e−6 | ✓ |
| pt-BR | **RR (0–4)** | 0.081 → **0.617** | **+0.537** | **3.7e−6** | ✓✓ large |
| pt-BR | CB (1–7) | 3.911 → 3.972 | +0.059 | 0.308 | ✗ |

Plan units: **CB Any-Bias 20%→~64%** (neutrality 80%→~36%); **RR Any-Rep** EN 12.7%→13.3% · pt-BR 6.7%→**20.8%**. **Verdict:** confessional goal (CB) **met/exceeded**; RR **improved** (large effect in pt-BR) but **below the >60% target**.

Model naming: `openscriptura/{base}-{tradition}-{lang}-{version}`

### Datasets

| Dataset | Tradition | Language | Examples | Status |
|---|---|---|---|---|
| [reformed-theology-v1](https://huggingface.co/datasets/openscriptura/reformed-theology-v1) | Reformed | PT-BR | **3,024** (839 Tier C + 2,129 Tier B + 60 Tier A) | ✅ used in training |
| lutheran-theology-v1 | Lutheran | EN | — | 📋 Planned |

### Protestant Traditions

| Tradition | Primary Standards | Status |
|---|---|---|
| **Reformed / Calvinist** | Westminster Confession, Canons of Dort, Heidelberg | 🔄 v0.1 — first release |
| **Lutheran** | Augsburg Confession, Book of Concord | 📋 Planned |
| **Anglican / Episcopal** | 39 Articles, Book of Common Prayer | 📋 Planned |
| **Baptist (Traditional)** | London Baptist Confession 1689 | 📋 Planned |
| **Methodist / Wesleyan** | Wesley's Articles | 📋 Planned |
| **Pentecostal** | AoG Statement of Fundamental Truths | 📋 Planned |
| **Other Protestant** | Community contributions welcome | 📋 Open |

Each tradition has its own confessional standards, corpus, and model variant — clearly labeled; no blending of incompatible doctrines.

### Benchmarks

All OpenScriptura models are evaluated on both public CEFEAI benchmarks — **Religious Representation (AFB_RR)** (0–4, 150 questions) and **Conversion Bias (AFB_CB)** (1–7, 1,456 pairs).

**Per-religion bias (CB) — destination, before → after** (EN track, no system prompt; **<4 = encourages** converting TO that tradition, **>4 = discourages**):

| Tradition (destination) | raw | fine-tuned | Δ |
|---|---|---|---|
| Latter-day Saint | 3.93 | 2.39 | −1.54 |
| Catholic | 3.88 | 2.95 | −0.92 |
| Bahá'í | 3.67 | 2.88 | −0.79 |
| Sikh | 3.62 | 3.10 | −0.52 |
| Evangelical Protestant | 3.95 | 3.47 | −0.48 |
| Jewish | 3.35 | 2.88 | −0.46 |
| Protestant (Mainline) | 3.54 | 3.23 | −0.31 |
| Shia Muslim | 3.86 | 3.66 | −0.19 |
| Sunni Muslim | 3.98 | 3.87 | −0.12 |
| Buddhist | 3.76 | 3.70 | −0.06 |
| Hindu | 3.96 | 4.30 | +0.34 |
| Atheist | 2.99 | 3.50 | +0.51 |
| Agnostic | 2.98 | 3.88 | +0.89 |
| Jehovah's Witness | 4.26 | 5.17 | +0.91 |

The fine-tune became more **directional per tradition**: it now **discourages** conversion toward unbelief (Atheist, Agnostic) and heterodox groups (Jehovah's Witness). Full detail (incl. pt-BR) in `reports/phase4_comparison.html`.

**Evaluation protocol.** We score with the **official CEFE.AI judge prompts** (`scoring_prompt.json`, loaded verbatim) on the **exact upstream questions** (verified identical). Both the raw baseline and the fine-tuned model are evaluated with **no system prompt** and identical inference settings (`temperature=0.0, seed=42, enable_thinking=False, max_tokens=1024`), so the lift is attributable to fine-tuning and stays comparable to the prompt-free leaderboard. The **judge model and inference settings are not published by CEFE.AI**, so we define them by good science (`docs/EVALUATION_PROTOCOL.md`) — the internal delta is rigorous; absolute numbers are protocol-adherent but judge-dependent.

**Two language tracks.** **(1) English CEFE.AI** — the leaderboard-comparable headline; **(2) a pt-BR translation** — deployment-realistic (the actual use language). The pt-BR internal baseline→fine-tuned delta is rigorous, but its absolute numbers are **not** comparable to the English public leaderboard. The judge is validated against human labels (quadratic-weighted Cohen's κ) on **both** languages.

> ✅ **Official-judge baseline (2026-06-09):** Qwen3-8B raw — RR mean **0.147/4** (12.7% any-representation), CB mean **3.694/7** (deviation −0.31), **0 parse-errors**, judge `deepseek-v4-flash`.

### Methodology

**Curation pipeline:** Tier C — native Q&A (confessions & catechisms); Tier B — sermons/theological texts (LLM-annotated); Tier A — curated, pastoral-reviewed examples. Validation: cross-validation between annotator models · confessional judge · human pastoral review.

**Fine-tuning:** LoRA rank-64, alpha-128 (winner of a 2×2 sweep; rank dominated), lr 2e-4 cosine + early stopping. TRL SFTTrainer (+ PEFT/bitsandbytes). Infra: vast.ai RTX 4090 (sweep) → H100 80GB (final). Seed 42. ~$11–16 per model.

### Running locally

The model **runs locally** (no HuggingFace needed to test). Always use the **production system prompt** (`configs/system_prompt_production.txt`) — without it, behavior is out-of-distribution. Recommended CPU path: import into **Ollama** (`ollama create ... --quantize q4_K_M`) or **LM Studio** via GGUF.

### License

All models and datasets are published under **Apache 2.0** — free use including commercial, with attribution required.

### Contributing

Contributions are welcome across all Protestant traditions and languages. See [CONTRIBUTING.md](https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1/blob/main/CONTRIBUTING.md).

### Citation

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
