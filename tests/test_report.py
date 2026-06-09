"""Tests for utils.report — official-schema reports, restored CEFE.AI data,
run placement + gating, per-faith rows, and the pt-BR conclusions box.

All inputs use the OFFICIAL summarize() schema (RR 0-4 / CB 1-7).
"""

from utils.cefeai import summarize
from utils.report import (
    generate_all_reports,
    _our_faith_bias_rows,
    _conclusion_items,
    _is_rr,
)
from utils.cefeai_leaderboard import (
    CEFEAI_LEADERBOARD, CEFEAI_CB_LEADERBOARD, _CB_FAITH_COLS, CEFEAI_CB_NET,
)


# ---------------------------------------------------------------------------
# builders (new-schema records + summary with run metadata)
# ---------------------------------------------------------------------------

def _rr_summary(scores, **meta):
    recs = [{"prompt_id": f"q{i}", "judge_score": s, "prompt": "p", "response": "r",
             "judge_rationale": "x"} for i, s in enumerate(scores)]
    s = summarize("rr", recs, "qwen/qwen3-8b")
    s.update({"run_label": "baseline", "system_prompt_mode": "noprompt",
              "judge_model": "deepseek/deepseek-v4-flash", "judge_max_tokens": 1024,
              "run_at": "2026-06-09T00:00:00+00:00"})
    s.update(meta)
    return s, recs


_FAITHS = [c.replace("\n", " ") for c in _CB_FAITH_COLS]


def _cb_summary(scores, **meta):
    recs = []
    for i, sc in enumerate(scores):
        fr = _FAITHS[i % len(_FAITHS)]
        to = _FAITHS[(i + 1) % len(_FAITHS)]
        recs.append({"prompt_id": f"c{i}", "judge_score": sc, "prompt": "p", "response": "r",
                     "judge_rationale": "x", "religion_from": fr, "religion_to": to,
                     "pair_id": "p", "template_id": "t"})
    s = summarize("cb", recs, "qwen/qwen3-8b")
    s.update({"run_label": "baseline", "system_prompt_mode": "noprompt",
              "judge_model": "deepseek/deepseek-v4-flash", "judge_max_tokens": 1024,
              "run_at": "2026-06-09T00:00:00+00:00"})
    s.update(meta)
    return s, recs


# ---------------------------------------------------------------------------
# report generation produces all three files
# ---------------------------------------------------------------------------

def test_generate_all_reports_rr(tmp_results_dir):
    s, recs = _rr_summary([0, 1, 2, 0, 1])
    paths = generate_all_reports(s, recs, "rr", tmp_results_dir, file_stem="t")
    for ext in ("html", "md", "json"):
        assert paths[ext].exists() and paths[ext].stat().st_size > 0


def test_generate_all_reports_cb(tmp_results_dir):
    s, recs = _cb_summary([4, 2, 4, 6, 4])
    paths = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")
    for ext in ("html", "md", "json"):
        assert paths[ext].exists()


# ---------------------------------------------------------------------------
# restored CEFE.AI reference data is rendered
# ---------------------------------------------------------------------------

def test_rr_html_has_cefeai_leaderboard(tmp_results_dir):
    s, recs = _rr_summary([0, 1])
    html = generate_all_reports(s, recs, "rr", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "Grok 4.20" in html and "Any Representation" in html


def test_cb_html_has_leaderboard_and_matrices(tmp_results_dir):
    s, recs = _cb_summary([4, 2, 4])
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "DeepSeek v4 Pro" in html
    assert "per-faith bias matrices" in html
    # all four matrices present
    for title in ("Total Bias", "Positive Bias", "Negative Bias", "Net Bias"):
        assert title in html


def test_official_rr_scale_labels_used(tmp_results_dir):
    s, recs = _rr_summary([3])
    html = generate_all_reports(s, recs, "rr", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    # level 3 must be the official text, not the old "Substantial"
    assert "Balance of religious references" in html


# ---------------------------------------------------------------------------
# run placement + gating
# ---------------------------------------------------------------------------

def test_run_placed_when_real_noprompt(tmp_results_dir):
    s, recs = _cb_summary([4, 2, 6], run_label="fine-tuned")
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "Qwen3-8B (fine-tuned)" in html and "is placed via" in html


def test_run_omitted_when_degenerate(tmp_results_dir):
    s, recs = _cb_summary([None, None, None])
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "Qwen3-8B (baseline)" not in html        # no fabricated bar
    assert "no valid judge scores" in html
    assert "DeepSeek v4 Pro" in html                # public data still shown


def test_run_omitted_for_v2_system_prompt(tmp_results_dir):
    s, recs = _cb_summary([4, 2, 6], system_prompt_mode="sysprompt")
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "Qwen3-8B (baseline)" not in html
    assert "NOT comparable to the prompt-free" in html
    assert "DeepSeek v4 Pro" in html


def test_degenerate_renders_no_literal_none(tmp_results_dir):
    s, recs = _cb_summary([None, None])
    paths = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")
    assert "dev None" not in paths["html"].read_text(encoding="utf-8")
    assert "**None**" not in paths["md"].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# per-faith bias rows
# ---------------------------------------------------------------------------

def test_our_faith_bias_rows_direction():
    # from=F0 to=F1 with an ENCOURAGING score (1 < 4): Positive for F1, Negative for F0
    recs = [{"judge_score": 1, "religion_from": _FAITHS[0], "religion_to": _FAITHS[1]}]
    total, pos, neg, net = _our_faith_bias_rows(recs)
    assert pos[1] == 100 and neg[0] == 100
    assert net[1] == 100 and net[0] == -100
    # a neutral (4) or discouraging (>4) score is NOT counted as encouraging
    recs2 = [{"judge_score": 4, "religion_from": _FAITHS[0], "religion_to": _FAITHS[1]},
             {"judge_score": 6, "religion_from": _FAITHS[0], "religion_to": _FAITHS[1]}]
    _, pos2, neg2, _ = _our_faith_bias_rows(recs2)
    assert pos2[1] == 0 and neg2[0] == 0


def test_our_faith_row_inserted_into_matrices(tmp_results_dir):
    s, recs = _cb_summary([2, 4, 6, 4, 2])
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    # highlighted run row appears in each of the 4 matrices (+ leaderboard table)
    assert html.count("Qwen3-8B (baseline) ★") >= 4


def test_per_faith_analysis_section_restored(tmp_results_dir):
    # The Total/Net/Encouraging-Multiplier charts + table must be present for a real run.
    s, recs = _cb_summary([2, 4, 6, 4, 2, 1, 7])
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "Per-faith bias analysis" in html
    for canvas in ("ourTotalBias", "ourNetBias", "ourEncou"):
        assert canvas in html
    assert "Enc. mult." in html        # the detailed table


def test_per_faith_analysis_omitted_when_degenerate(tmp_results_dir):
    s, recs = _cb_summary([None, None])
    html = generate_all_reports(s, recs, "cb", tmp_results_dir, file_stem="t")["html"].read_text(encoding="utf-8")
    assert "ourTotalBias" not in html


def test_our_faith_metrics_structure():
    from utils.report import _our_faith_metrics
    recs = [{"judge_score": 1, "religion_from": _FAITHS[0], "religion_to": _FAITHS[1]}]
    m = _our_faith_metrics(recs)
    assert len(m) == 14
    keys = {"faith", "pos", "neg", "total", "net", "enc_pct", "dis_pct", "enc_mult"}
    assert keys <= set(m[0].keys())


# ---------------------------------------------------------------------------
# conclusions box (pt-BR)
# ---------------------------------------------------------------------------

def test_conclusion_items_rr():
    s, recs = _rr_summary([0, 0, 1, 2])
    head, items = _conclusion_items(s, recs, "rr")
    assert "Representação Religiosa" in head
    assert any("Confiabilidade do juiz" == lbl for lbl, _ in items)


def test_conclusion_items_cb_direction_and_faith():
    s, recs = _cb_summary([2, 2, 4])     # leans encouraging
    head, items = _conclusion_items(s, recs, "cb")
    assert "Viés de Conversão" in head
    joined = " ".join(f"{lbl} {txt}" for lbl, txt in items)
    assert "encorajar a transição" in joined
    assert "Por tradição" in joined


def test_conclusion_items_degenerate():
    s, recs = _cb_summary([None])
    head, items = _conclusion_items(s, recs, "cb")
    assert items[0][0] == "Sem dados"


# ---------------------------------------------------------------------------
# restored reference data integrity (must match the public CEFE.AI values)
# ---------------------------------------------------------------------------

def test_cefeai_reference_data_intact():
    assert dict(CEFEAI_LEADERBOARD)["Grok 4.20"] == 29.3
    assert dict(CEFEAI_CB_LEADERBOARD)["DeepSeek v4 Pro"] == 32.6
    assert len(_CB_FAITH_COLS) == 14
    assert CEFEAI_CB_NET["Claude Opus 4.6"][7] == -28
