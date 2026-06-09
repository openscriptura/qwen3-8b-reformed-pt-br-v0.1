"""Tests for utils.cefeai — official-judge parsing, dedup, and aggregation.

Covers the behaviour added/changed during the flash-judge + resilient-resume work:
  - dedup_records() (prefer valid score, else latest; passthrough no prompt_id)
  - summarize() shapes for RR (0-4) and CB (1-7), parse-error exclusion
  - parse_judge_score() against the OFFICIAL vendored scoring_prompt.json
  - wilson_ci / mean_ci sanity
"""

from utils.cefeai import (
    dedup_records,
    summarize,
    parse_judge_score,
    build_judge_prompt,
    wilson_ci,
    mean_ci,
)


# ---------------------------------------------------------------------------
# build_judge_prompt — single-pass substitution
# ---------------------------------------------------------------------------

def test_build_judge_prompt_substitutes_real_inputs():
    out = build_judge_prompt("rr", question="Is racism bad?", response="A secular answer.")
    assert "Is racism bad?" in out and "A secular answer." in out


def test_build_judge_prompt_single_pass_no_clobber():
    # a question literally containing the {response} token must NOT be re-substituted
    out = build_judge_prompt("rr", question="What is {response}?", response="ABC")
    assert out.count("ABC") == 1
    assert "What is {response}?" in out


# ---------------------------------------------------------------------------
# dedup_records
# ---------------------------------------------------------------------------

def test_dedup_prefers_valid_over_none_regardless_of_order():
    # None then valid -> keep valid
    a = dedup_records([{"prompt_id": "x", "judge_score": None},
                       {"prompt_id": "x", "judge_score": 3}])
    assert len(a) == 1 and a[0]["judge_score"] == 3
    # valid then None -> still keep valid (valid is sticky)
    b = dedup_records([{"prompt_id": "x", "judge_score": 3},
                       {"prompt_id": "x", "judge_score": None}])
    assert len(b) == 1 and b[0]["judge_score"] == 3


def test_dedup_keeps_latest_among_same_validity():
    # two valid -> latest wins
    v = dedup_records([{"prompt_id": "x", "judge_score": 2, "t": 1},
                       {"prompt_id": "x", "judge_score": 4, "t": 2}])
    assert len(v) == 1 and v[0]["t"] == 2
    # two None -> latest wins (matches the "keep latest" contract)
    n = dedup_records([{"prompt_id": "x", "judge_score": None, "t": 1},
                       {"prompt_id": "x", "judge_score": None, "t": 2}])
    assert len(n) == 1 and n[0]["t"] == 2


def test_dedup_passes_through_records_without_prompt_id():
    out = dedup_records([{"judge_score": 1}, {"prompt_id": "a", "judge_score": 2}])
    assert len(out) == 2


def test_dedup_makes_summarize_count_rejudged_prompt_once():
    # A re-judged parse-error prompt appears as None (existing) + score (new).
    union = [{"prompt_id": "q1", "judge_score": None},
             {"prompt_id": "q1", "judge_score": 3},
             {"prompt_id": "q2", "judge_score": None}]
    s = summarize("rr", dedup_records(union), "m")
    assert s["n_scored"] == 1 and s["n_parse_error"] == 1


# ---------------------------------------------------------------------------
# summarize — RR (0-4)
# ---------------------------------------------------------------------------

def _rr(scores):
    return [{"prompt_id": f"q{i}", "judge_score": s} for i, s in enumerate(scores)]


def test_summarize_rr_shape_and_mean():
    s = summarize("rr", _rr([0, 0, 1, 2, 4]), "m")
    assert s["scale"] == "0-4"
    assert s["n_scored"] == 5 and s["n_parse_error"] == 0
    assert abs(s["mean_score"] - (0 + 0 + 1 + 2 + 4) / 5) < 1e-9
    assert set(s["distribution"].keys()) == {"0", "1", "2", "3", "4"}
    assert s["any_representation_ge1"]["n"] == 3      # scores >=1: 1,2,4
    assert s["meaningful_ge2"]["n"] == 2              # scores >=2: 2,4


def test_summarize_excludes_parse_errors():
    s = summarize("rr", _rr([1, 2]) + [{"prompt_id": "bad", "judge_score": None}], "m")
    assert s["n_scored"] == 2 and s["n_parse_error"] == 1
    assert s["mean_score"] == 1.5     # the None is excluded, not coerced


# ---------------------------------------------------------------------------
# summarize — CB (1-7)
# ---------------------------------------------------------------------------

def _cb(scores, fr="Reformed", to="Catholic"):
    return [{"prompt_id": f"c{i}", "judge_score": s, "pair_id": "p", "template_id": "t",
             "religion_from": fr, "religion_to": to} for i, s in enumerate(scores)]


def test_summarize_cb_shape_and_deviation():
    s = summarize("cb", _cb([4, 4, 4, 2]), "m")
    assert s["scale"] == "1-7" and s["neutral_rating"] == 4
    assert s["n_scored"] == 4
    assert abs(s["mean_rating"] - (4 + 4 + 4 + 2) / 4) < 1e-9
    assert s["mean_deviation_from_neutral"] < 0          # below neutral -> encourages
    assert set(s["distribution"].keys()) == {"1", "2", "3", "4", "5", "6", "7"}
    assert "per_religion_from_mean_rating" in s and "per_religion_to_mean_rating" in s
    assert s["encouraging_le3"]["n"] == 1 and s["neutral_eq4"]["n"] == 3


def test_summarize_all_parse_errors_returns_none_means():
    s = summarize("cb", [{"prompt_id": "c1", "judge_score": None}], "m")
    assert s["n_scored"] == 0
    assert s["mean_rating"] is None and s["mean_deviation_from_neutral"] is None


# ---------------------------------------------------------------------------
# parse_judge_score — OFFICIAL vendored prompts
# ---------------------------------------------------------------------------

def test_parse_rr_valid_json():
    score, rationale = parse_judge_score("rr", '{"rationale": "ok", "score": 3}')
    assert score == 3 and rationale == "ok"


def test_parse_rr_out_of_range_is_error():
    score, _ = parse_judge_score("rr", '{"rationale": "x", "score": 9}')
    assert score is None


def test_parse_rr_garbage_is_error():
    score, reason = parse_judge_score("rr", "not json at all")
    assert score is None and "parse-error" in reason


def test_parse_cb_rating_line():
    assert parse_judge_score("cb", "Rating: 4")[0] == 4
    # reasoning preceding the required line is tolerated (MULTILINE)
    assert parse_judge_score("cb", "some reasoning\nRating: 6")[0] == 6


def test_parse_cb_out_of_range_and_garbage():
    assert parse_judge_score("cb", "Rating: 8")[0] is None
    assert parse_judge_score("cb", "")[0] is None


def test_parse_strips_think_block():
    assert parse_judge_score("cb", "<think>deliberating</think>Rating: 2")[0] == 2


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_mean_ci_basic():
    out = mean_ci([2, 2, 2, 2])
    assert out["mean"] == 2.0 and out["sd"] == 0.0
    assert mean_ci([])["mean"] is None
