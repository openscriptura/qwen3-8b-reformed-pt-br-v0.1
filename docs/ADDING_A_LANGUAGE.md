# Adding a language

OpenScriptura is **language-agnostic** from day one. A tradition's model can be produced in any language where the confessional corpus can be sourced or translated.

## Steps

1. **System prompt in the target language.** Translate/author the production system prompt (see `configs/system_prompt_production.txt`) into the language, keeping the confessional standards and rules intact. Prefer a native speaker for fluency.
2. **Tier C — confessions in the language.** Source the confessions/catechisms in the target language. **Use public-domain or permissively-licensed translations** (the original confessions are public domain; a specific modern translation may carry its own copyright — verify before redistributing).
3. **Tier B — synthetic (optional).** Generate + judge-filter from tradition texts in the language. ⚠️ Respect source licenses (regenerate from your own licensed sources; don't redistribute copyrighted-derived Q&A).
4. **Tier A — native-speaker pastoral review.** Manual review must be done by a native speaker of the language *and* someone literate in the tradition — tonal/affirmation nuances are language-sensitive (see `docs/JUDGE_VALIDATION.md` on the annotator-language confound).
5. **Evaluation.** The CEFE.AI track supports a translated benchmark via `scripts/translate_benchmark.py` (`--lang`), which translates only the model-facing prompt and keeps IDs/pairing verbatim. The translated track gives a rigorous internal delta but its **absolute numbers are NOT comparable** to the English leaderboard. Run both baseline and fine-tuned in the same language for a valid delta.
6. **Product story.** The fine-tune's confessional behavior transfers most strongly in the **training language** — lead the product narrative with that language, and treat other languages as secondary tracks.

Questions? Open an issue or see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
