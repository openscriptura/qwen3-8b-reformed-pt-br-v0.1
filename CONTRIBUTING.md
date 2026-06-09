# Contributing to OpenScriptura

Thank you for your interest in contributing. OpenScriptura serves the full breadth of Protestant Christianity — across traditions and languages — and welcomes contributions of code, data, and doctrinal review.

By contributing, you agree that your contributions are licensed under the project's [Apache 2.0 License](LICENSE).

---

## ⛔ HARD RULE — do not break CEFEAI comparability

**WE MUST RESPECT COMPARABILITY WITH CEFE.AI.** The project's headline claim and the CEFEAI leaderboard comparison depend on it, so **no fix, refactor, optimization, or "improvement" may change how the headline CEFEAI numbers are produced.** A PR that is cleaner or faster but alters the evaluation comparison will be rejected.

The headline evaluation must run, identically on the raw baseline and the fine-tuned model:
- **NO system prompt** (a system prompt saturates the metric — the raw model scored RR 99.3% / CB 87.8% with one);
- locked inference `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`;
- the identical judge / 0–3 rubric / Wilson CI from `scripts/utils/cefeai.py` (the single source of truth — do not fork it);
- unchanged benchmark inputs; only the model weights differ.

If you believe an improvement *requires* changing any of this, **open an issue first** — don't just do it. The full rule and the cautionary v2 case study are in [`CLAUDE.md`](CLAUDE.md#-hard-rule--cefeai-comparability-is-non-negotiable).

---

## Ways to contribute

### Dataset examples
Open an issue with theological Q&A in any tradition and language. **Always include the confessional reference** (e.g. `WCF 11.1`, `Heidelberg Q60`, `Augsburg Art. IV`). Examples without a confessional anchor cannot be accepted into Tier C/B.

### Doctrinal corrections
If a generated or published example misrepresents a tradition's confessional position, open an issue with:
- the problematic text,
- the correct position, and
- the confessional reference that supports it.

Each tradition is held to its **own** standards — see [`docs/THEOLOGICAL_STATEMENT.md`](docs/THEOLOGICAL_STATEMENT.md). We do not blend incompatible doctrines across traditions.

### A new tradition or language
Open an issue to discuss first. You will need:
- the primary confessional standards for the tradition,
- a seed corpus of theological Q&A, and
- at least one person willing to perform pastoral review (see [`docs/PASTORAL_REVIEW_PROTOCOL.md`](docs/PASTORAL_REVIEW_PROTOCOL.md)).

### Code
Issues labeled `good first issue` are a good place to start. Please open or comment on an issue before starting substantial work so effort isn't duplicated.

---

## Development setup

```bash
git clone https://github.com/openscriptura/qwen3-8b-reformed-pt-br-v0.1
cd qwen3-8b-reformed-pt-br-v0.1
pip install -r requirements.txt --break-system-packages
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
# fill .env with your own OpenRouter / HuggingFace keys
```

Validate your setup without spending money or calling any API:

```bash
python scripts/00_cefeai_baseline.py --dry-run
```

See [`CLAUDE.md`](CLAUDE.md) for architecture, conventions, and the full command reference.

---

## Code standards

- **Format & lint** before committing:
  ```bash
  black .
  ruff check .
  ```
  Both are configured in [`pyproject.toml`](pyproject.toml) (line length 100, Python 3.11).
- **Tests** must pass:
  ```bash
  python -m pytest
  ```
  Add tests under `tests/` for any new pure utility. (API-calling code should be exercised via `--dry-run` rather than live calls.)
- **Match the surrounding code.** New phase scripts follow the existing patterns: numbered `NN_name.py` in `scripts/`, shared code in `scripts/utils/`, paths resolved from `PROJECT_ROOT`, and the Windows UTF-8 `reconfigure` shim in entry points.
- **Every API-spending script must** route calls through `OpenRouterClient` (retry), feed cost into a `CostTracker` (hard stop), support `--dry-run`, and checkpoint to JSONL so `--resume` works.

## Security

- **Never commit secrets.** Real keys belong only in your local `.env` (gitignored). `.env.example` must contain placeholders only.
- If you discover a leaked credential or vulnerability, do not open a public issue — contact a maintainer privately.

---

## Pull requests

1. Branch from `main`.
2. Keep PRs focused; explain the *why*, not just the *what*.
3. Ensure `black`, `ruff`, and `pytest` are clean.
4. Reference the issue your PR addresses.

---

*Soli Deo Gloria — for the glory of God and the good of the Church, in every tradition and every language.*
