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
