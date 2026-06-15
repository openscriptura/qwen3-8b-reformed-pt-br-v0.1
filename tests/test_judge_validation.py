"""Tests for 08_judge_validation — blind sampling + κ scoring.

The sampler must NOT leak the judge score into the template the human fills
(anchoring inflates κ); the judge score goes to a sidecar key file, re-joined at
scoring time by prompt_id.
"""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_validation",
    Path(__file__).resolve().parent.parent / "scripts" / "08_judge_validation.py",
)
jv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(jv)


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_sample_is_blind_and_writes_key(tmp_path):
    results = tmp_path / "cb_results.jsonl"
    _write_jsonl(results, [
        {"prompt_id": f"q{i}", "prompt": "p", "response": "r", "judge_score": (i % 7) + 1}
        for i in range(10)
    ])
    out = tmp_path / "tmpl.jsonl"
    jv.do_sample(SimpleNamespace(results=results, out=out, benchmark="cb", n=5, seed=42))

    tmpl = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(tmpl) == 5
    # BLIND: no judge_score leaks into the template; human_score is blank
    assert all("judge_score" not in r for r in tmpl)
    assert all(r["human_score"] is None for r in tmpl)
    # sidecar key exists and carries the judge scores
    key = jv._key_path(out)
    assert key.exists()
    keyrows = [json.loads(l) for l in key.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {k["prompt_id"] for k in keyrows} == {r["prompt_id"] for r in tmpl}


def test_score_joins_key_and_computes_kappa(tmp_path, capsys):
    results = tmp_path / "rr_results.jsonl"
    _write_jsonl(results, [
        {"prompt_id": f"q{i}", "prompt": "p", "response": "r", "judge_score": i % 5}
        for i in range(8)
    ])
    out = tmp_path / "tmpl.jsonl"
    jv.do_sample(SimpleNamespace(results=results, out=out, benchmark="rr", n=8, seed=1))

    # human labels EXACTLY match the (hidden) judge scores -> perfect agreement
    key = {k["prompt_id"]: k["judge_score"]
           for k in (json.loads(l) for l in jv._key_path(out).read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["human_score"] = key[r["prompt_id"]]
    _write_jsonl(out, rows)

    jv.do_score(SimpleNamespace(labels=out, benchmark="rr"))
    printed = capsys.readouterr().out
    assert "exact agreement          : 100.0%" in printed
    assert "quadratic-weighted κ     : 1.0" in printed


def test_score_errors_when_nothing_labeled(tmp_path):
    out = tmp_path / "tmpl.jsonl"
    _write_jsonl(out, [{"prompt_id": "q0", "human_score": None}])
    jv._key_path(out).write_text('{"prompt_id": "q0", "judge_score": 2}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        jv.do_score(SimpleNamespace(labels=out, benchmark="rr"))


# --- interactive labeler helpers (the input() loop itself is user-run) ----------

def test_parse_label_score_skip_quit_bad():
    assert jv._parse_label("3", 0, 4) == ("score", 3)
    assert jv._parse_label(" 0 ", 0, 4) == ("score", 0)
    assert jv._parse_label("s", 0, 4) == ("skip",)
    assert jv._parse_label("Q", 0, 4) == ("quit",)
    assert jv._parse_label("9", 0, 4) == ("bad",)      # out of range
    assert jv._parse_label("", 0, 4) == ("bad",)       # empty -> re-prompt (no accidental skip)
    assert jv._parse_label("x", 1, 7) == ("bad",)


def test_atomic_write_roundtrip(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"prompt_id": "a", "human_score": 2}, {"prompt_id": "b", "human_score": None}]
    jv._atomic_write_jsonl(p, rows)
    assert [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] == rows
    assert not (tmp_path / "t.jsonl.tmp").exists()      # temp cleaned up


# --- stratified sampling (κ on a neutral-dominated population) -------------------

def test_stratified_sample_reaches_n_and_includes_off_modal(tmp_path):
    # 90 neutral (modal=4) + 10 off-modal — stratified must still reach n AND pull
    # in off-modal items so κ tests the discriminative ratings, not just neutrals.
    rows = [{"prompt_id": f"n{i}", "prompt": "p", "response": "r", "judge_score": 4} for i in range(90)]
    rows += [{"prompt_id": f"o{i}", "prompt": "p", "response": "r", "judge_score": (i % 6) + 1}
             for i in range(10) if (i % 6) + 1 != 4]
    results = tmp_path / "cb.jsonl"
    _write_jsonl(results, rows)
    out = tmp_path / "t.jsonl"
    jv.do_sample(SimpleNamespace(results=results, out=out, benchmark="cb", n=20, seed=42, stratify=True))
    key = [json.loads(l) for l in jv._key_path(out).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(key) == 20
    off = sum(1 for k in key if k["judge_score"] != 4)
    assert off >= 5      # informative off-modal items are present


# --- degenerate κ guard ---------------------------------------------------------

def test_score_flags_degenerate_no_variance(tmp_path, capsys):
    out = tmp_path / "t.jsonl"
    _write_jsonl(out, [{"prompt_id": f"q{i}", "human_score": 4} for i in range(5)])
    jv._key_path(out).write_text(
        "".join(json.dumps({"prompt_id": f"q{i}", "judge_score": 4}) + "\n" for i in range(5)),
        encoding="utf-8")
    jv.do_score(SimpleNamespace(labels=out, benchmark="cb"))
    assert "DEGENERATE" in capsys.readouterr().out
