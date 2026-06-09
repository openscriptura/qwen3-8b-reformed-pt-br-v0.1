# CEFE.AI official scoring prompts (vendored, verbatim)

These two files are the **AllFaith Benchmark (AFB) LLM-as-judge prompts**, copied
**unmodified** from CEFE.AI so OpenScriptura scores **exactly** as CEFE.AI does.
They are loaded at runtime by `scripts/utils/cefeai.py` — the judge prompts are
**never** hardcoded in our Python, precisely so it is auditable that we use theirs.

| File | Source (upstream `main`) | Release id |
|------|--------------------------|------------|
| `rr_scoring_prompt.json` | https://github.com/CEFEAI/allfaith-religious-representation/blob/main/scoring_prompt.json | `AFB_ReligiousRepresentation_v.1_2Q26` |
| `cb_scoring_prompt.json` | https://github.com/CEFEAI/allfaith-conversion-bias/blob/main/scoring_prompt.json | `AFB_ConversionBias_v.2_2Q26` |

Captured 2026-06-08.

## Verify byte-fidelity against upstream
```bash
curl -s https://raw.githubusercontent.com/CEFEAI/allfaith-religious-representation/main/scoring_prompt.json -o /tmp/rr_up.json
curl -s https://raw.githubusercontent.com/CEFEAI/allfaith-conversion-bias/main/scoring_prompt.json     -o /tmp/cb_up.json
python -c "import json;a=json.load(open('configs/cefeai/rr_scoring_prompt.json'));b=json.load(open('/tmp/rr_up.json'));print('RR match:', a==b)"
python -c "import json;a=json.load(open('configs/cefeai/cb_scoring_prompt.json'));b=json.load(open('/tmp/cb_up.json'));print('CB match:', a==b)"
```
If a check prints `False`, replace our copy with the upstream file (CEFE.AI is the
source of truth) and re-run the baseline — do not edit our copy to "fix" a diff.

## Scales & aggregation (from the upstream READMEs)
- **RR** — judge returns `{"rationale": ..., "score": <0-4>}`. Aggregate = **mean score** (0–4) + **score distribution**. No single leaderboard scalar is defined upstream.
- **CB** — judge returns `Rating: <1-7>` (neutral = 4). Aggregate = **mean rating per `pair_id`**; distance from 4 = directional bias. No single leaderboard scalar is defined upstream.

`questions.jsonl` (upstream) == our `data/cefeai/{rr_150,cb_1456}.jsonl` (the RR `question`
field is stored as `prompt` in our copy; CB carries `religion_from`/`religion_to`/`pair_id`/`template_id`).

## Adherence audit (2026-06-08)

**What we follow exactly (the documented protocol):**
- Judge prompts: vendored verbatim; loaded at runtime, never hardcoded.
- Scales: RR 0–4 JSON `{"rationale","score"}`; CB 1–7 via the official regex
  `^Rating:\s*([1-7])\s*$` (applied verbatim from the JSON).
- Aggregation: RR = mean score + score distribution; CB = mean rating +
  per-`pair_id` / per-`template_id` / per-tradition (`religion_from`,`religion_to`)
  means + deviation-from-neutral (4) — exactly the "by pair / tradition / template"
  slices the upstream READMEs prescribe. No invented leaderboard scalar.
- Data integrity (verified): CB = 1456 = 182 ordered pairs (14×13) × 8 templates
  `t1..t8`; the 14 traditions match the README list exactly. RR = 150 (`q0001..q0150`).

**What CEFE.AI does NOT specify — these are OUR choices, and the published**
**leaderboard's absolute numbers depend on them, so cross-leaderboard absolute**
**comparison is NOT guaranteed identical:**
- **Judge model** — upstream names none. We use `OPENROUTER_MODEL_JUDGE`
  (`deepseek/deepseek-v4-flash`) at `temperature=0`. A different judge scores
  differently, so our absolute numbers may not equal CEFE.AI's leaderboard.
- **Model-under-test inference** — upstream specifies none. We use
  `temperature=0.0, seed=42, enable_thinking=False, max_tokens=512`, no system prompt.

**Therefore:** our **internal baseline → fine-tuned delta is rigorous** (identical
judge + settings on both sides — only the weights differ). Any statement relative
to CEFE.AI's *published leaderboard* must be labeled as protocol-adherent but
**judge-dependent** (their judge is unpublished). Do not claim exact leaderboard parity.

## Verify our benchmark inputs == upstream questions.jsonl
```bash
curl -s https://raw.githubusercontent.com/CEFEAI/allfaith-religious-representation/main/questions.jsonl -o /tmp/rr_up.jsonl
curl -s https://raw.githubusercontent.com/CEFEAI/allfaith-conversion-bias/main/questions.jsonl       -o /tmp/cb_up.jsonl
python -c "import json; up=[json.loads(l) for l in open('/tmp/rr_up.jsonl')]; ours=[json.loads(l) for l in open('data/cefeai/rr_150.jsonl')]; print('RR text match:', [r['question'] for r in up]==[r['prompt'] for r in ours])"
python -c "import json; up=[json.loads(l) for l in open('/tmp/cb_up.jsonl')]; ours=[json.loads(l) for l in open('data/cefeai/cb_1456.jsonl')]; print('CB text match:', [r['question'] for r in up]==[r['prompt'] for r in ours])"
```
