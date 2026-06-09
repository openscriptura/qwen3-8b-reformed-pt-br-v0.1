"""
cefeai.py — shared CEFEAI (AllFaith Benchmark) primitives.

The judge prompts are **NOT hardcoded here**. They are loaded VERBATIM at runtime
from the official CEFE.AI files vendored in `configs/cefeai/`:

  - configs/cefeai/rr_scoring_prompt.json  (AFB_ReligiousRepresentation_v.1_2Q26)
  - configs/cefeai/cb_scoring_prompt.json  (AFB_ConversionBias_v.2_2Q26)

so it is auditable that OpenScriptura scores EXACTLY as CEFE.AI prescribes. Both
the Phase 0 baseline (00_cefeai_baseline.py) and the Phase 4 eval
(07_cefeai_eval.py) import these helpers, so the two sides of the comparison can
never drift, and neither can drift from upstream.

Scales (from the official scoring_prompt.json files):
  - RR: judge returns JSON {"rationale": str, "score": <int 0-4>}.
        Aggregate (per upstream README) = mean score (0-4) + score distribution.
  - CB: judge returns "Rating: <int 1-7>" (neutral = 4), per (religion_from →
        religion_to) transition. Aggregate = mean rating + per-pair_id mean;
        distance from 4 = directional bias.

Neither benchmark defines a single leaderboard scalar upstream; we therefore
report exactly the slices CEFE.AI recommends (mean + distribution) and never
invent a headline number.
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Repo root, resolved from this file (scripts/utils/cefeai.py → repo root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_SYSTEM_PROMPT_FILE = _PROJECT_ROOT / "configs" / "system_prompt.txt"

_SCORING_FILES = {
    "rr": _PROJECT_ROOT / "configs" / "cefeai" / "rr_scoring_prompt.json",
    "cb": _PROJECT_ROOT / "configs" / "cefeai" / "cb_scoring_prompt.json",
}

# ---------------------------------------------------------------------------
# Official judge prompts — loaded from the vendored CEFE.AI JSON (never hardcoded)
# ---------------------------------------------------------------------------

_scoring_cache: dict[str, dict] = {}


def load_scoring_prompt(benchmark: str) -> dict:
    """Return the official CEFE.AI scoring_prompt.json for 'rr' or 'cb' (cached)."""
    b = benchmark.lower()
    if b not in _SCORING_FILES:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected 'rr' or 'cb'.")
    if b not in _scoring_cache:
        path = _SCORING_FILES[b]
        if not path.exists():
            raise FileNotFoundError(
                f"Official CEFE.AI scoring prompt not found at {path}. "
                "Restore it from the upstream repo (see configs/cefeai/README.md)."
            )
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for key in ("template", "input_variables", "output_format"):
            if key not in cfg:
                raise ValueError(f"{path} is missing required field '{key}'.")
        _scoring_cache[b] = cfg
    return _scoring_cache[b]


def build_judge_prompt(benchmark: str, **values) -> str:
    """Render the official judge template by substituting its input_variables.

    Uses literal string replacement (NOT str.format) on purpose: the RR template
    embeds literal JSON braces (its output-format example), which str.format would
    misinterpret. We replace each declared `{var}` and leave all other braces intact.
    """
    cfg = load_scoring_prompt(benchmark)
    template = cfg["template"]
    for var in cfg["input_variables"]:
        if var not in values:
            raise KeyError(
                f"{benchmark.upper()} judge requires input variable {var!r} "
                f"(expected: {cfg['input_variables']})."
            )
        template = template.replace("{" + var + "}", str(values[var]))
    return template


def parse_judge_score(benchmark: str, raw: str) -> tuple[int | None, str]:
    """Parse a judge reply into (score, rationale) per the official output_format.

    Returns (None, reason) for unparseable or out-of-range output. Callers MUST
    treat None as a parse error (count it separately) — never coerce it to a
    number, which would silently bias the aggregate.
    """
    cfg = load_scoring_prompt(benchmark)
    of = cfg.get("output_format", {})
    valid = set(of.get("valid_ratings", []))

    content = raw or ""
    if "<think>" in content and "</think>" in content:   # strip judge thinking
        content = content.split("</think>", 1)[-1]
    content = content.strip()

    # CB: apply the OFFICIAL regex from the scoring_prompt.json verbatim
    # (e.g. ^Rating:\s*([1-7])\s*$). re.MULTILINE so the required "Rating: <d>"
    # line is found even if the judge emits reasoning on preceding lines; the
    # line itself must match CEFE.AI's exact format. This is their parser, not ours.
    rgx = of.get("regex")
    if rgx:
        m = re.search(rgx, content, flags=re.MULTILINE)
        if not m:
            return None, f"[parse-error] {raw[:120]}"
        v = int(m.group(1))
        if valid and v not in valid:   # regex already constrains the range; double-check
            return None, f"[out-of-range:{v}] {raw[:100]}"
        return v, ""

    # RR (json output): {"rationale": ..., "score": <int>}
    c = content
    if c.startswith("```"):
        c = c.split("```")[1]
        if c.lstrip().lower().startswith("json"):
            c = c.lstrip()[4:]
        c = c.strip()
    try:
        start, end = c.find("{"), c.rfind("}")
        obj = json.loads(c[start:end + 1] if start != -1 and end != -1 else c)
        v = int(obj["score"])
        rationale = str(obj.get("rationale", ""))
        if valid and v not in valid:
            return None, f"[out-of-range:{v}] {raw[:100]}"
        return v, rationale
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None, f"[parse-error] {raw[:120]}"


# ---------------------------------------------------------------------------
# System prompt (committed, single source of truth) — used only by the v2
# deployment-behavior mode, NEVER by the CEFEAI-comparable headline runs.
# ---------------------------------------------------------------------------

def load_system_prompt() -> str:
    """Return the canonical Reformed system prompt from configs/system_prompt.txt."""
    _regen_hint = (
        "Regenerate it from the training data, e.g.:\n"
        "  python -c \"import json,pathlib; "
        "r=json.loads(open('data/merged/train.jsonl',encoding='utf-8').readline()); "
        "p=next(m['content'] for m in r['messages'] if m['role']=='system'); "
        "pathlib.Path('configs/system_prompt.txt').write_text(p,encoding='utf-8')\""
    )
    if not _SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"Canonical system prompt not found at {_SYSTEM_PROMPT_FILE}. " + _regen_hint)
    prompt = _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not prompt:
        raise FileNotFoundError(f"Canonical system prompt at {_SYSTEM_PROMPT_FILE} is empty. " + _regen_hint)
    return prompt


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval — accurate near p=0 and p=1 (M8)."""
    if n_total == 0:
        return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = z * math.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _frac(count: int, n: int) -> dict:
    lo, hi = wilson_ci(count, n)
    return {"n": count, "frac": round(count / n, 4) if n else 0.0, "ci_low": lo, "ci_high": hi}


# ---------------------------------------------------------------------------
# Aggregation — exactly the slices CEFE.AI's READMEs prescribe
# ---------------------------------------------------------------------------

def summarize(benchmark: str, results: list[dict], model_label: str) -> dict:
    """Aggregate per-response judge scores into a CEFE.AI-faithful summary.

    `results` records must have integer `judge_score` (or None on parse error);
    CB records must also carry `pair_id`. Parse errors are excluded from metrics
    and reported as `n_parse_error`.
    """
    b = benchmark.lower()
    scored = [r for r in results if isinstance(r.get("judge_score"), int)]
    n = len(scored)
    n_parse_error = len(results) - n
    scores = [r["judge_score"] for r in scored]
    total_cost = round(sum(r.get("cost_usd", 0.0) for r in results), 4)

    base = {
        "model": model_label,
        "benchmark": f"CEFEAI_{b.upper()}",
        "release_id": load_scoring_prompt(b).get("release_id")
                      or load_scoring_prompt(b).get("benchmark", {}).get("release_id"),
        "n_scored": n,
        "n_parse_error": n_parse_error,
        "total_cost_usd": total_cost,
    }

    if b == "rr":
        levels = [0, 1, 2, 3, 4]
        dist = {str(k): _frac(sum(1 for s in scores if s == k), n) for k in levels}
        base.update({
            "scale": "0-4",
            "mean_score": round(sum(scores) / n, 4) if n else None,   # PRIMARY (CEFE.AI)
            "distribution": dist,                                     # PRIMARY (CEFE.AI)
            # auxiliary cumulative slices (transparent; not an upstream scalar):
            "any_representation_ge1": _frac(sum(1 for s in scores if s >= 1), n),
            "meaningful_ge2":         _frac(sum(1 for s in scores if s >= 2), n),
            "predominantly_eq4":      _frac(sum(1 for s in scores if s == 4), n),
        })
        return base

    # CB
    levels = [1, 2, 3, 4, 5, 6, 7]
    dist = {str(k): _frac(sum(1 for s in scores if s == k), n) for k in levels}
    devs = [s - 4 for s in scores]            # neutral midpoint = 4

    # CEFE.AI README: "aggregate by pair / tradition / template". Build all three.
    def _means_by(key: str) -> dict:
        groups: dict[str, list[int]] = defaultdict(list)
        for r in scored:
            groups[r.get(key, "?")].append(r["judge_score"])
        return {g: round(sum(v) / len(v), 4) for g, v in sorted(groups.items())}

    base.update({
        "scale": "1-7",
        "neutral_rating": 4,
        "mean_rating": round(sum(scores) / n, 4) if n else None,           # PRIMARY
        "mean_deviation_from_neutral": round(sum(devs) / n, 4) if n else None,  # directional: <0 encourages transition, >0 discourages
        "mean_abs_deviation": round(sum(abs(d) for d in devs) / n, 4) if n else None,  # bias magnitude
        "distribution": dist,                                              # PRIMARY
        "encouraging_le3": _frac(sum(1 for s in scores if s <= 3), n),
        "neutral_eq4":     _frac(sum(1 for s in scores if s == 4), n),
        "discouraging_ge5": _frac(sum(1 for s in scores if s >= 5), n),
        "per_pair_mean_rating":         _means_by("pair_id"),       # by pair     (CEFE.AI slice)
        "per_template_mean_rating":     _means_by("template_id"),   # by template (CEFE.AI slice)
        "per_religion_from_mean_rating": _means_by("religion_from"),# by tradition (FROM)
        "per_religion_to_mean_rating":   _means_by("religion_to"),  # by tradition (TO)
    })
    return base


def format_console_summary(summary: dict) -> str:
    """Human-readable console block for a summary dict."""
    b = summary["benchmark"]
    lines = ["=" * 64, f"  {b}  ·  n={summary['n_scored']}  (parse-errors: {summary['n_parse_error']})",
             f"  release: {summary.get('release_id')}", "-" * 64]
    if summary["n_scored"] == 0:
        # No valid scores (e.g. every judge reply failed to parse) — don't try to
        # format None means with a sign spec; report the failure plainly.
        lines.append("  ⚠  No valid judge scores — nothing to aggregate (all parse errors?).")
        lines.append("=" * 64)
        return "\n".join(lines)
    if summary.get("scale") == "0-4":
        lines.append(f"  MEAN SCORE (0-4): {summary['mean_score']}   ← primary CEFE.AI metric")
        lines.append("  distribution:")
        for k in ["0", "1", "2", "3", "4"]:
            d = summary["distribution"][k]
            lines.append(f"    score {k}: {d['frac']*100:5.1f}%  [{d['ci_low']*100:.1f}, {d['ci_high']*100:.1f}]  n={d['n']}")
    else:
        lines.append(f"  MEAN RATING (1-7, neutral=4): {summary['mean_rating']}   ← primary CEFE.AI metric")
        lines.append(f"  mean deviation from neutral : {summary['mean_deviation_from_neutral']:+}  (<0 encourages transition, >0 discourages)")
        lines.append(f"  mean |deviation| (bias mag) : {summary['mean_abs_deviation']}")
        lines.append("  distribution:")
        for k in ["1", "2", "3", "4", "5", "6", "7"]:
            d = summary["distribution"][k]
            lines.append(f"    rating {k}: {d['frac']*100:5.1f}%  n={d['n']}")
    lines.append("=" * 64)
    return "\n".join(lines)


def results_are_legacy_schema(records: list[dict]) -> bool:
    """True if any record was written by the OLD (pre-official-CEFE.AI) judge.

    Old records carry 'judge_reasoning' (home-grown 0-3 rubric); the official-judge
    schema carries 'judge_rationale'. Callers use this to refuse merging stale
    old-scale scores into the new 0-4 / 1-7 aggregate on --resume.
    """
    return any("judge_reasoning" in r and "judge_rationale" not in r for r in records)


def compare_summaries(benchmark: str, baseline: dict, current: dict) -> str:
    """Human-readable baseline → fine-tuned comparison on the official metric."""
    b = benchmark.lower()
    if b == "rr":
        a, c = baseline.get("mean_score"), current.get("mean_score")
        delta = (c - a) if (a is not None and c is not None) else None
        return (f"  RR mean score (0-4): baseline {a} → fine-tuned {c}"
                + (f"  (Δ {delta:+.3f})" if delta is not None else ""))
    a, c = baseline.get("mean_rating"), current.get("mean_rating")
    delta = (c - a) if (a is not None and c is not None) else None
    return (f"  CB mean rating (1-7): baseline {a} → fine-tuned {c}"
            + (f"  (Δ {delta:+.3f})" if delta is not None else "")
            + "  [for a Reformed model, directional bias is by-design — interpret, don't grade]")
