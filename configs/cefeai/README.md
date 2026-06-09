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
