# Adding a Protestant tradition

OpenScriptura's pipeline, dataset schema, and evaluation are **tradition-agnostic**. Adding a new tradition (e.g. Lutheran, Anglican, Methodist, Pentecostal) means supplying its confessional standards, a corpus, and pastoral review — the code is reused as-is.

## Steps

1. **Define the confessional standards.** List the primary confessions/catechisms that define the tradition (e.g. Lutheran → Augsburg Confession, Book of Concord; Anglican → 39 Articles, Book of Common Prayer). These are the doctrinal ground truth.
2. **Tier C — native Q&A.** Build question/answer records directly from those confessions/catechisms (`scripts/01_build_tier_c.py` pattern). **Use only public-domain or permissively-licensed source texts** (the original confessions are public domain; verify that any modern *translation* you use is PD or freely licensed before redistributing).
3. **Tier B — synthetic (optional).** LLM-generate + judge-filter Q&A from the tradition's texts (`scripts/02_build_tier_b.py` pattern). ⚠️ **Respect source licenses:** do not redistribute Q&A derived from copyrighted works — regenerate from your own licensed sources instead.
4. **Tier A — pastoral review.** Curate a small set of manually-reviewed, confessionally-checked examples (see [`docs/PASTORAL_REVIEW_PROTOCOL.md`](PASTORAL_REVIEW_PROTOCOL.md)). Reviewer should be literate in the tradition.
5. **System prompt.** Write a production system prompt carrying the tradition's confessional standards (see `configs/system_prompt_production.txt` as a template).
6. **Merge + train.** `scripts/merge_dataset.py` → `scripts/05_train_final.py` with `configs/final.yaml` (LoRA r=64, lr=2e-4 winner).
7. **Evaluate.** CEFE.AI (`scripts/07_cefeai_eval.py`) + the values track (`docs/GLOBALOPINIONS_EVAL_SPEC.md`). Keep the comparability lock (same settings both sides).
8. **Label clearly.** Name the model by tradition (`{base}-{tradition}-{lang}-{version}`). **Do not blend incompatible doctrines** — where traditions diverge (baptism, Lord's Supper, church governance), the dataset reflects that tradition's own confessional position.

## Data schema
Records use the canonical chat format with `tradition` and `lang` as first-class fields — see the schema in [`README_github.md`](../README_github.md) and `scripts/utils/hash.py`.

Questions? Open an issue or see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
