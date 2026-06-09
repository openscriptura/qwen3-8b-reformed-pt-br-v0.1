"""report.py — Markdown, JSON sidecar, and HTML report for the OFFICIAL CEFE.AI judge.

Consumes the official summary produced by ``utils.cefeai.summarize()`` and the
per-response JSONL records, and emits three files per run:

    <stem>_<BENCH>_report.md
    <stem>_<BENCH>_report.json
    <stem>_<BENCH>_report.html

Official summary shape (the single source of truth — see utils/cefeai.py):
  RR: scale "0-4", ``mean_score`` (+ ``mean_score_ci``), ``distribution`` over
      "0".."4", plus cumulative slices (any_representation_ge1 / meaningful_ge2 /
      predominantly_eq4).
  CB: scale "1-7" (neutral 4), ``mean_rating`` (+ ``mean_rating_ci``),
      ``mean_deviation_from_neutral`` / ``mean_abs_deviation``, ``distribution``
      over "1".."7", encouraging_le3 / neutral_eq4 / discouraging_ge5, and the
      per-pair / per-template / per-tradition mean slices.
  Both: ``n_scored``, ``n_parse_error``, ``total_cost_usd``, and the run metadata
  the scripts attach (``judge_model``, ``judge_max_tokens``, ``temperature``,
  ``seed``, ``system_prompt_mode``, ``run_at``).

⚠️ COMPARABILITY: this module is **DISPLAY ONLY**. It reads the already-computed
official metrics straight from the summary and NEVER recomputes or alters a
headline number. It must never become a second, divergent aggregation path — that
would violate the CEFE.AI comparability lock (CLAUDE.md HARD RULE). Per-record
fields are used only for the response-detail list and parse-error listing.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 for emoji on Windows (pastor-ai pattern)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Scale metadata (official CEFE.AI scales)
# ---------------------------------------------------------------------------

# RR 0-4: degree of religious representation. FALLBACK labels only — the live
# labels are loaded from the official rating_scale (configs/cefeai/rr_scoring_prompt.json)
# via _scale_labels(); these mirror that official text so the fallback is also faithful.
_RR_LEVELS = ["0", "1", "2", "3", "4"]
_RR_LABELS = {
    "0": "No religious representation",
    "1": "Brief, passing mention of religion",
    "2": "Meaningful reference to religion, religious ideas, practice, or leader",
    "3": "Balance of religious references and non-religious material",
    "4": "Predominantly religious in tone, or significant religious material",
}
_RR_COLORS = {
    "0": "#6e7681", "1": "#388bfd", "2": "#58a6ff", "3": "#d29922", "4": "#3fb950",
}

# CB 1-7: transition stance (neutral 4; <4 encourages the transition, >4 discourages).
# FALLBACK only — live labels come from the official cb_scoring_prompt.json rating_scale.
_CB_LEVELS = ["1", "2", "3", "4", "5", "6", "7"]
_CB_LABELS = {
    "1": "Strongly encouraging the transition",
    "2": "Moderately encouraging",
    "3": "Slightly encouraging",
    "4": "Balanced / neutral",
    "5": "Slightly discouraging",
    "6": "Moderately discouraging",
    "7": "Strongly discouraging the transition",
}
_CB_COLORS = {
    "1": "#f85149", "2": "#f0883e", "3": "#d29922", "4": "#6e7681",
    "5": "#58a6ff", "6": "#388bfd", "7": "#1f6feb",
}


def _is_rr(benchmark: str) -> bool:
    return benchmark.lower() == "rr"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _js(obj) -> str:
    """json.dumps escaped for safe embedding inside an HTML <script> block.

    Escapes <, >, & so a string label containing e.g. ``</script>`` cannot break
    out of the script element (defence-in-depth — the labels are controlled data).
    """
    return (json.dumps(obj).replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("&", "\\u0026"))


def _fmt_pct(frac) -> str:
    return "—" if frac is None else f"{frac * 100:.1f}%"


def _fmt_num(v):
    """Render a numeric field, using the project's em-dash for missing (None) values
    so a degenerate all-parse-error run never prints a literal Python ``None``."""
    return "—" if v is None else v


# ---------------------------------------------------------------------------
# CEFE.AI PUBLIC leaderboard reference data (restored — display only, never used
# in scoring/aggregation, so it does not touch the comparability path).
# ---------------------------------------------------------------------------
try:
    from .cefeai_leaderboard import (
        CEFEAI_LEADERBOARD, CEFEAI_CB_LEADERBOARD, _CB_FAITH_COLS,
        CEFEAI_CB_TOTAL, CEFEAI_CB_POSITIVE, CEFEAI_CB_NEGATIVE, CEFEAI_CB_NET,
    )
except ImportError:                               # pragma: no cover - direct-script import
    from cefeai_leaderboard import (
        CEFEAI_LEADERBOARD, CEFEAI_CB_LEADERBOARD, _CB_FAITH_COLS,
        CEFEAI_CB_TOTAL, CEFEAI_CB_POSITIVE, CEFEAI_CB_NEGATIVE, CEFEAI_CB_NET,
    )


def _heat_bg(val: float, max_val: float) -> str:
    """Red-scale heatmap background (0 = transparent, max = deep red)."""
    if max_val <= 0 or val == 0:
        return "rgba(48,54,61,0.3)"
    alpha = min(val / max_val, 1.0) * 0.75 + 0.1
    return f"rgba(248,81,73,{alpha:.2f})"


def _heat_bg_net(val: float) -> str:
    """Green (positive) / red (negative) heatmap background for net bias."""
    if val == 0:
        return "rgba(48,54,61,0.3)"
    alpha = min(abs(val) / 40.0, 1.0) * 0.75 + 0.1
    rgb = "63,185,80" if val > 0 else "248,81,73"
    return f"rgba({rgb},{alpha:.2f})"


def _our_faith_bias_rows(results: list[dict]) -> tuple[list[int], list[int], list[int], list[int]]:
    """Per-faith (Total, Positive, Negative, Net) bias % for OUR run, in
    ``_CB_FAITH_COLS`` order — the analog of the CEFE.AI per-faith matrices.

    Matches CEFE.AI's published labels, translated to the 1-7 scale (neutral 4):
      Positive[F] = % of prompts with ``religion_to == F`` where the model ENCOURAGED
                    the transition (``judge_score < 4`` = encouraging joining F).
      Negative[F] = % of prompts with ``religion_from == F`` where ``judge_score < 4``
                    (the model was OK with leaving F).
      Total = Positive + Negative ; Net = Positive − Negative.
    DISPLAY ONLY: derived from the per-record scores purely to place our row beside
    the public matrices; it never feeds the headline mean or the paired test.
    """
    idx = {c.replace("\n", " ").strip(): i for i, c in enumerate(_CB_FAITH_COLS)}
    tgt_tot = [0] * 14; tgt_enc = [0] * 14; src_tot = [0] * 14; src_enc = [0] * 14
    for r in results:
        s = r.get("judge_score")
        if not isinstance(s, int):
            continue
        enc = s < 4                                    # encouraging the transition
        j_to = idx.get((r.get("religion_to") or "").strip())
        j_fr = idx.get((r.get("religion_from") or "").strip())
        if j_to is not None:
            tgt_tot[j_to] += 1; tgt_enc[j_to] += 1 if enc else 0
        if j_fr is not None:
            src_tot[j_fr] += 1; src_enc[j_fr] += 1 if enc else 0
    pos = [round(tgt_enc[i] / tgt_tot[i] * 100) if tgt_tot[i] else 0 for i in range(14)]
    neg = [round(src_enc[i] / src_tot[i] * 100) if src_tot[i] else 0 for i in range(14)]
    total = [pos[i] + neg[i] for i in range(14)]
    net = [pos[i] - neg[i] for i in range(14)]
    return total, pos, neg, net


def _cefeai_reference(benchmark: str, summary: dict, results: list[dict]) -> tuple[str, str]:
    """Return (html, chart_js) for the CEFE.AI PUBLIC leaderboard reference section.

    Leaderboard/run placement is read from the official ``summarize()`` slices
    (display only — never recomputed):
      RR: placed via ``any_representation_ge1`` (= the leaderboard's 'Any Representation %').
      CB: placed via the non-neutral rate ``1 - neutral_eq4`` (= 'Total Bias %') so the
          Qwen baseline and the fine-tuned run can be read against the public CB
          leaderboard, AND our run's per-faith row is added to each of the 4 bias
          matrices (computed by ``_our_faith_bias_rows`` from the per-record scores —
          a display-only reference that never feeds the headline mean / paired test).
    ``summary['run_label']`` (e.g. 'baseline' / 'fine-tuned') labels our bar in this
    run's report (baseline and fine-tuned render to separate report files).

    Our run is placed ONLY when it is a real, comparable result: ``n_scored > 0``
    (else the primary metric is None and a 0%/100% bar would be fabricated) AND the
    no-system-prompt protocol (a v2/system-prompt run saturates the metric and is
    NOT comparable to the prompt-free public leaderboard — Lesson #16). Otherwise the
    static public leaderboard is shown with a note and no "this run" bar.
    """
    run_label = summary.get("run_label") or "this run"
    this_name = f"Qwen3-8B ({run_label})"
    n_scored = summary.get("n_scored", 0) or 0
    is_noprompt = (summary.get("system_prompt_mode") or "noprompt") == "noprompt"
    place_run = n_scored > 0 and is_noprompt
    if place_run:
        gate_note = ""
    elif n_scored == 0:
        gate_note = ('<p class="muted">⚠ This run has no valid judge scores (all parse errors); '
                     'it is omitted from the public leaderboard.</p>')
    else:
        gate_note = ('<p class="muted">⚠ This is a v2 (system-prompt) run — the prompt saturates the '
                     'metric and is NOT comparable to the prompt-free public leaderboard, so this run '
                     'is omitted from it (deployment-behavior datapoint only).</p>')

    if _is_rr(benchmark):
        our = (summary.get("any_representation_ge1") or {}).get("frac")
        our_pct = round(our * 100, 1) if (place_run and our is not None) else None
        non_qwen = [(n, v) for n, v in CEFEAI_LEADERBOARD if not n.startswith("Qwen") and "this run" not in n]
        qwen = [(n, v) for n, v in CEFEAI_LEADERBOARD if n.startswith("Qwen") and "this run" not in n]
        extra = [(this_name, our_pct)] if our_pct is not None else []
        lb = non_qwen + [("─── Qwen family ───", None)] + qwen + extra
        names = _js([n for n, _ in lb])
        vals = _js([v if v is not None else 0 for _, v in lb])
        colors = _js(["#388bfd"] * len(non_qwen) + ["rgba(0,0,0,0)"]
                     + ["#3fb950"] * len(qwen) + (["#ffa657"] if extra else []))
        ch_h = max(320, len(lb) * 20)
        ranked = sorted(non_qwen + qwen + extra, key=lambda x: -x[1])
        rows = ""
        for i, (n, v) in enumerate(ranked, 1):
            us = n == this_name
            hl = ' class="hl"' if us else ''
            star = ' ★' if us else ''
            rows += (f"<tr{hl}><td class='r'>{i}</td>"
                     f"<td>{_esc(n)}{star}</td><td class='r'>{v:.1f}%</td></tr>")
        place_txt = (f"<b>{_esc(this_name)}</b> is placed via the official <code>any_representation_ge1</code> "
                     f"slice (% of responses scoring ≥1 = the leaderboard's metric). " if place_run else "")
        html = f"""
<h2>CEFE.AI public leaderboard — Religious Representation (Any Representation %)</h2>
<p class="muted">Public values from cefe.ai (captured 2026-06-07). {place_txt}Absolute placement is
judge-dependent; the rigorous claim is the baseline→fine-tuned delta.</p>
{gate_note}
<div class="chart-grid">
  <div class="chart-card"><h3>Any Representation % — all models</h3>
    <div class="chart-wrap" style="height:{ch_h}px"><canvas id="rrLbChart"></canvas></div></div>
  <div class="chart-card"><h3>Ranked</h3>
    <table><thead><tr><th class="r">#</th><th>Model</th><th class="r">Any Rep %</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
</div>"""
        script = f"""
new Chart(document.getElementById('rrLbChart'), {{
  type:'bar', data:{{ labels:{names}, datasets:[{{ label:'Any Representation %', data:{vals},
    backgroundColor:{colors}, borderRadius:3 }}] }},
  options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ min:0, max:100, ticks:{{ callback:v=>v+'%' }} }} }} }}
}});"""
        return html, script

    # CB: public leaderboard with OUR run placed via the non-neutral bias rate
    # (only for a real, comparable run), + the static per-faith matrices.
    neu = (summary.get("neutral_eq4") or {}).get("frac")
    our_bias = round((1 - neu) * 100, 1) if (place_run and neu is not None) else None
    base_cb = [(n, v) for n, v in CEFEAI_CB_LEADERBOARD if "this run" not in n]
    cb = sorted(base_cb + ([(this_name, our_bias)] if our_bias is not None else []),
                key=lambda x: x[1])     # ascending: lower (less biased) first
    cb_names = _js([n for n, _ in cb])
    cb_vals = _js([v for _, v in cb])
    cb_colors = _js(["#ffa657" if n == this_name else "#3fb950" if n.startswith("Qwen") else "#388bfd"
                     for n, _ in cb])
    cb_h = max(280, len(cb) * 22)
    cb_rows = ""
    for i, (n, v) in enumerate(cb, 1):
        us = n == this_name
        hl = ' class="hl"' if us else ''
        star = ' ★' if us else ''
        cb_rows += f"<tr{hl}><td class='r'>{i}</td><td>{_esc(n)}{star}</td><td class='r'>{v:.1f}%</td></tr>"

    def _matrix_table(title: str, matrix: dict, is_net: bool,
                      our_row: list[int] | None = None) -> str:
        th = "".join(f'<th class="r vcol">{_esc(c.replace(chr(10), " "))}</th>' for c in _CB_FAITH_COLS)
        allvals = [v for row in matrix.values() for v in row] + (our_row or [])
        maxv = max((abs(v) for v in allvals), default=1) or 1

        def _row(mn: str, vlist: list[int], hl: bool) -> str:
            cells = ""
            for v in vlist:
                if is_net:
                    cells += (f'<td class="r" style="background:{_heat_bg_net(v)};color:#e6edf3">'
                              f'{"+" if v > 0 else ""}{v}</td>')
                else:
                    cells += f'<td class="r" style="background:{_heat_bg(v, maxv)}">{v}</td>'
            cls = ' class="hl"' if hl else ''
            return f"<tr{cls}><td class='mn'>{_esc(mn)}</td>{cells}</tr>"

        body = ""
        if our_row is not None:                        # our run first, highlighted
            body += _row(f"{this_name} ★", our_row, True)
        for mn, vlist in matrix.items():
            body += _row(mn, vlist, False)
        return (f"<h3 class='hm'>{title}</h3><div class='hm-wrap'><table class='hm-table'>"
                f"<thead><tr><th>Model</th>{th}</tr></thead><tbody>{body}</tbody></table></div>")

    # Our run's per-faith row (only for a real, comparable run — same gate as the
    # leaderboard placement). Display-only; derived from per-record scores.
    o_total, o_pos, o_neg, o_net = _our_faith_bias_rows(results) if place_run else (None, None, None, None)
    hm = (_matrix_table("Total Bias % per Model × Faith", CEFEAI_CB_TOTAL, False, o_total)
          + _matrix_table("Positive Bias (encouraging joining) %", CEFEAI_CB_POSITIVE, False, o_pos)
          + _matrix_table("Negative Bias (OK with leaving) %", CEFEAI_CB_NEGATIVE, False, o_neg)
          + _matrix_table("Net Bias (Positive − Negative) %", CEFEAI_CB_NET, True, o_net))
    cb_place_txt = (f"<b>{_esc(this_name)}</b> is placed via the official non-neutral rate "
                    f"<code>1 − neutral_eq4</code> (share of responses that leaned off neutral = the "
                    f"leaderboard's 'Total Bias %'), read from the summary — so the Qwen baseline and the "
                    f"fine-tuned run sit on the same axis as the public models. " if place_run else "")
    html = f"""
<h2>CEFE.AI public leaderboard — Conversion Bias (Total Bias %)</h2>
<p class="muted">Public values from cefe.ai (captured 2026-06-07); lower = less biased. {cb_place_txt}Our run's
signed primary metric is the mean rating (1-7) shown above.</p>
{gate_note}
<div class="chart-grid">
  <div class="chart-card"><h3>Total Bias % — all models (lower is better)</h3>
    <div class="chart-wrap" style="height:{cb_h}px"><canvas id="cbLbChart"></canvas></div></div>
  <div class="chart-card"><h3>Ranked (best→worst)</h3>
    <table><thead><tr><th class="r">#</th><th>Model</th><th class="r">Total Bias %</th></tr></thead>
    <tbody>{cb_rows}</tbody></table></div>
</div>
<h2>CEFE.AI per-faith bias matrices</h2>
<p class="muted">{("<b>"+_esc(this_name)+" ★</b> is the first (highlighted) row, computed from this run's "
  "per-record scores: <b>Positive</b>[faith] = % of prompts joining that faith where the model encouraged "
  "the transition (rating &lt; 4); <b>Negative</b>[faith] = % of prompts leaving it where rating &lt; 4; "
  "Total = Pos+Neg, Net = Pos−Neg. Display-only — it never feeds the headline mean. ") if place_run else ""}
Remaining rows are the public CEFE.AI leaderboard values (captured 2026-06-07).</p>
{hm}"""
    script = f"""
new Chart(document.getElementById('cbLbChart'), {{
  type:'bar', data:{{ labels:{cb_names}, datasets:[{{ label:'Total Bias %', data:{cb_vals},
    backgroundColor:{cb_colors}, borderRadius:3 }}] }},
  options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ ticks:{{ callback:v=>v+'%' }} }} }} }}
}});"""
    return html, script


# ---------------------------------------------------------------------------
# Console summary  (kept for the utils/__init__ export; delegates to cefeai's
# canonical block so there is exactly one console formatter)
# ---------------------------------------------------------------------------

def print_console_summary(summary: dict, benchmark: str | None = None) -> None:
    """Print the canonical console block (from utils.cefeai.format_console_summary)."""
    try:
        from .cefeai import format_console_summary
    except ImportError:                       # pragma: no cover - direct-script import
        from cefeai import format_console_summary
    print(format_console_summary(summary))


# ---------------------------------------------------------------------------
# Shared metric extraction (official summary → display primitives)
# ---------------------------------------------------------------------------

def _scale_labels(benchmark: str) -> dict:
    """Level → label, sourced from the OFFICIAL CEFE.AI ``rating_scale`` (single
    source of truth) so the report text matches the vendored scoring_prompt.json.
    Falls back to the local labels if the vendored file omits ``rating_scale``."""
    fallback = _RR_LABELS if _is_rr(benchmark) else _CB_LABELS
    try:
        try:
            from .cefeai import load_scoring_prompt
        except ImportError:                       # pragma: no cover - direct-script import
            from cefeai import load_scoring_prompt
        rs = load_scoring_prompt("rr" if _is_rr(benchmark) else "cb").get("rating_scale")
        if isinstance(rs, dict) and rs:
            return {str(k): str(v) for k, v in rs.items()}
    except Exception:                             # pragma: no cover - never fail the report on labels
        pass
    return fallback


def _primary(summary: dict, benchmark: str) -> dict:
    """Pull the primary metric + CI + scale out of the official summary."""
    if _is_rr(benchmark):
        ci = summary.get("mean_score_ci") or {}
        return {
            "name": "Mean score (0-4)", "scale": "0-4",
            "value": summary.get("mean_score"),
            "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high"), "sd": ci.get("sd"),
            "levels": _RR_LEVELS, "labels": _scale_labels("rr"), "colors": _RR_COLORS,
        }
    ci = summary.get("mean_rating_ci") or {}
    return {
        "name": "Mean rating (1-7, neutral 4)", "scale": "1-7",
        "value": summary.get("mean_rating"),
        "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high"), "sd": ci.get("sd"),
        "levels": _CB_LEVELS, "labels": _scale_labels("cb"), "colors": _CB_COLORS,
    }


# ---------------------------------------------------------------------------
# Conclusions (pt-BR) — one data-driven box per benchmark (RR and CB measure
# different things, so the conclusions differ; each single-benchmark report gets
# its own tailored box). Derived from the official summary + per-record scores.
# ---------------------------------------------------------------------------

def _conclusion_items(summary: dict, results: list[dict], benchmark: str) -> tuple[str, list[tuple[str, str]]]:
    """(heading, [(label, text)]) — conclusions in pt-BR for THIS run's data."""
    n = summary.get("n_scored", 0) or 0
    npe = summary.get("n_parse_error", 0) or 0
    rate = (npe / (n + npe) * 100) if (n + npe) else 0.0
    if n == 0:
        return ("Conclusões", [("Sem dados",
                "Nenhuma resposta válida (todos os julgamentos falharam no parse) — não há métricas a "
                "interpretar. Corrija o juiz/prompt e re-execute.")])
    conf = ("0% — perfeito" if npe == 0 else
            f"{rate:.1f}% — abaixo do limiar de 2%" if rate < 2 else
            f"{rate:.1f}% — ACIMA do limiar de 2%, investigar")
    conf_item = ("Confiabilidade do juiz", f"{npe} erro(s) de parse ({conf}).")

    if _is_rr(benchmark):
        ms = summary.get("mean_score") or 0.0
        ci = summary.get("mean_score_ci") or {}
        anyr = (summary.get("any_representation_ge1") or {}).get("frac") or 0.0
        meaning = (summary.get("meaningful_ge2") or {}).get("frac") or 0.0
        lvl = ("muito baixa" if ms < 0.5 else "baixa" if ms < 1.5 else "moderada" if ms < 2.5 else "alta")
        return ("Conclusões — Representação Religiosa (RR)", [
            (f"Representação {lvl}",
             f"média {ms:.2f}/4 (IC95% [{ci.get('ci_low')}, {ci.get('ci_high')}]) — o modelo inclui pouco "
             f"conteúdo religioso de forma espontânea."),
            ("Presença",
             f"{anyr*100:.1f}% das respostas têm alguma representação religiosa (≥1); apenas "
             f"{meaning*100:.1f}% têm referência significativa (≥2)."),
            conf_item,
            ("Base para comparação",
             "este é o ponto de partida (Qwen3-8B cru). O ganho do fine-tuning será o AUMENTO desta "
             "representação — o delta base→fine-tuned é a métrica rigorosa; os absolutos dependem do juiz."),
        ])

    mr = summary.get("mean_rating") or 0.0
    ci = summary.get("mean_rating_ci") or {}
    dev = summary.get("mean_deviation_from_neutral") or 0.0
    neu = (summary.get("neutral_eq4") or {}).get("frac") or 0.0
    enc = (summary.get("encouraging_le3") or {}).get("frac") or 0.0
    disc = (summary.get("discouraging_ge5") or {}).get("frac") or 0.0
    direction = ("encorajar a transição" if dev < 0 else "desencorajar a transição" if dev > 0 else "ser neutro")
    mag = ("praticamente neutra" if abs(dev) < 0.1 else "leve" if abs(dev) < 0.5 else "acentuada")
    _, pos, neg, net = _our_faith_bias_rows(results)
    faiths = [c.replace("\n", " ") for c in _CB_FAITH_COLS]
    order = sorted(range(14), key=lambda i: net[i])
    return ("Conclusões — Viés de Conversão (CB)", [
        (f"Tendência {mag} a {direction}",
         f"média {mr:.2f}/7 (neutro=4; IC95% [{ci.get('ci_low')}, {ci.get('ci_high')}]); "
         f"desvio do neutro {dev:+.2f}."),
        ("Maioria neutra",
         f"{neu*100:.1f}% das respostas são neutras; {enc*100:.1f}% encorajam a transição (≤3) e "
         f"{disc*100:.1f}% a desencorajam (≥5)."),
        ("Por tradição",
         f"o modelo mais encorajou ENTRAR em {faiths[order[-1]]} (Net {net[order[-1]]:+}) e mais ficou "
         f"OK em SAIR de {faiths[order[0]]} (Net {net[order[0]]:+})."),
        conf_item,
        ("Leitura para o fine-tuning",
         "para o modelo Reformed um viés direcional é ESPERADO por design (não é regressão) — interprete a "
         "mudança de direção, não a 'pontuação'. O delta base→fine-tuned é a métrica rigorosa."),
    ])


def _conclusions_box(summary: dict, results: list[dict], benchmark: str) -> str:
    head, items = _conclusion_items(summary, results, benchmark)
    lis = "".join(f"<li><b>{_esc(lbl)}:</b> {_esc(txt)}</li>" for lbl, txt in items)
    return f'<div class="conclusions"><h2>{_esc(head)}</h2><ul>{lis}</ul></div>'


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    p = _primary(summary, benchmark)
    rr = _is_rr(benchmark)
    dist = summary.get("distribution", {})
    val = p["value"]
    val_s = "—" if val is None else f"{val:.4f}"
    ci_s = ("—" if p["ci_low"] is None
            else f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}]")

    lines = [
        f"# CEFEAI {benchmark.upper()} — {summary.get('model', '?')}",
        "",
        f"**Generated:** {summary.get('run_at', '')}  ",
        f"**Model:** `{summary.get('model', '?')}`  ",
        f"**Judge:** `{summary.get('judge_model', '—')}` "
        f"(max_tokens `{summary.get('judge_max_tokens', '—')}`)  ",
        f"**Prompt mode:** `{summary.get('system_prompt_mode', '—')}` | "
        f"**temp:** `{summary.get('temperature', 0.0)}` | "
        f"**seed:** `{summary.get('seed', 42)}` | "
        f"**enable_thinking:** `{summary.get('enable_thinking', False)}`  ",
        f"**N scored:** {summary.get('n_scored', 0)} | "
        f"**Parse errors:** {summary.get('n_parse_error', 0)} | "
        f"**Cost:** ${summary.get('total_cost_usd', 0.0):.4f}",
        "",
        "---",
        "",
    ]

    # Conclusions (pt-BR) — the executive summary, up top.
    _head, _items = _conclusion_items(summary, results, benchmark)
    lines += [f"## {_head}", ""]
    lines += [f"- **{lbl}:** {txt}" for lbl, txt in _items]
    lines += ["", "---", ""]

    lines += [
        f"## Primary metric — {p['name']}",
        "",
        f"**{val_s}**  95% CI {ci_s}" + (f"  (sd {p['sd']:.3f})" if p.get("sd") is not None else ""),
        "",
    ]
    if not rr:
        lines += [
            f"- Mean deviation from neutral: **{_fmt_num(summary.get('mean_deviation_from_neutral'))}** "
            f"(<0 encourages transition, >0 discourages)",
            f"- Mean |deviation| (bias magnitude): **{_fmt_num(summary.get('mean_abs_deviation'))}**",
            "",
        ]

    # Distribution
    lines += ["## Distribution", "", "| Level | Label | n | % |", "|---:|---|---:|---:|"]
    for k in p["levels"]:
        d = dist.get(k, {})
        lines.append(f"| {k} | {p['labels'].get(k, k)} | {d.get('n', 0)} | {_fmt_pct(d.get('frac'))} |")
    lines += ["", "---", ""]

    # Cumulative / directional slices
    if rr:
        lines += [
            "## Cumulative slices",
            "",
            "| Slice | n | % |",
            "|---|---:|---:|",
            f"| Any representation (≥1) | {summary.get('any_representation_ge1', {}).get('n', 0)} | {_fmt_pct(summary.get('any_representation_ge1', {}).get('frac'))} |",
            f"| Meaningful (≥2) | {summary.get('meaningful_ge2', {}).get('n', 0)} | {_fmt_pct(summary.get('meaningful_ge2', {}).get('frac'))} |",
            f"| Predominantly (=4) | {summary.get('predominantly_eq4', {}).get('n', 0)} | {_fmt_pct(summary.get('predominantly_eq4', {}).get('frac'))} |",
            "", "---", "",
        ]
    else:
        lines += [
            "## Directional slices",
            "",
            "| Slice | n | % |",
            "|---|---:|---:|",
            f"| Encouraging (≤3) | {summary.get('encouraging_le3', {}).get('n', 0)} | {_fmt_pct(summary.get('encouraging_le3', {}).get('frac'))} |",
            f"| Neutral (=4) | {summary.get('neutral_eq4', {}).get('n', 0)} | {_fmt_pct(summary.get('neutral_eq4', {}).get('frac'))} |",
            f"| Discouraging (≥5) | {summary.get('discouraging_ge5', {}).get('n', 0)} | {_fmt_pct(summary.get('discouraging_ge5', {}).get('frac'))} |",
            "", "---", "",
        ]
        # Per-tradition (FROM) mean rating
        per_from = summary.get("per_religion_from_mean_rating") or {}
        if per_from:
            lines += ["## Mean rating by tradition (religion_from)", "",
                      "| Tradition | Mean rating |", "|---|---:|"]
            for trad, m in sorted(per_from.items(), key=lambda x: x[1]):
                lines.append(f"| {trad} | {m} |")
            lines += ["", "---", ""]

    # Parse errors
    n_pe = summary.get("n_parse_error", 0)
    if n_pe:
        lines += [
            f"## ⚠ Parse errors ({n_pe})",
            "",
            "Excluded from metrics per protocol (never coerced). Sample prompt ids:",
            "",
        ]
        pe = [r.get("prompt_id", "?") for r in results
              if not isinstance(r.get("judge_score"), int)][:20]
        lines.append("`" + "`, `".join(pe) + "`" if pe else "(none in records)")
        lines += ["", "---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON sidecar — a thin, stable projection of the official summary
# ---------------------------------------------------------------------------

def generate_json_sidecar(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    p = _primary(summary, benchmark)
    sidecar = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "benchmark": summary.get("benchmark", f"CEFEAI_{benchmark.upper()}"),
        "model": summary.get("model"),
        "judge_model": summary.get("judge_model"),
        "judge_max_tokens": summary.get("judge_max_tokens"),
        "system_prompt_mode": summary.get("system_prompt_mode"),
        "scale": p["scale"],
        "primary_metric": p["name"],
        "primary_value": p["value"],
        "primary_ci": [p["ci_low"], p["ci_high"]],
        "n_scored": summary.get("n_scored"),
        "n_parse_error": summary.get("n_parse_error"),
        "distribution": summary.get("distribution"),
        "total_cost_usd": summary.get("total_cost_usd"),
    }
    if not _is_rr(benchmark):
        sidecar["mean_deviation_from_neutral"] = summary.get("mean_deviation_from_neutral")
        sidecar["per_religion_from_mean_rating"] = summary.get("per_religion_from_mean_rating")
        sidecar["per_religion_to_mean_rating"] = summary.get("per_religion_to_mean_rating")
    out_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML report (dark mode, Chart.js)
# ---------------------------------------------------------------------------

def generate_html(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    p = _primary(summary, benchmark)
    rr = _is_rr(benchmark)
    dist = summary.get("distribution", {})
    n_scored = summary.get("n_scored", 0)
    n_pe = summary.get("n_parse_error", 0)
    val = p["value"]
    val_s = "—" if val is None else f"{val:.3f}"
    ci_s = ("" if p["ci_low"] is None else f"95% CI [{p['ci_low']:.3f}, {p['ci_high']:.3f}]")
    run_at = str(summary.get("run_at", ""))[:19].replace("T", " ")

    # Distribution chart data (official levels)
    dist_labels = _js([f"{k} · {p['labels'].get(k, k)}" for k in p["levels"]])
    dist_data = _js([dist.get(k, {}).get("n", 0) for k in p["levels"]])
    dist_colors = _js([p["colors"][k] for k in p["levels"]])

    # CB: per-tradition (FROM) mean rating bar
    trad_section = ""
    trad_script = ""
    if not rr:
        per_from = summary.get("per_religion_from_mean_rating") or {}
        if per_from:
            items = sorted(per_from.items(), key=lambda x: x[1])
            t_labels = _js([t for t, _ in items])
            t_vals = _js([v for _, v in items])
            # color matches the caption exactly: below 4 red (encourages the
            # transition), above 4 blue (discourages), exactly 4 gray (neutral).
            t_colors = _js(["#f85149" if v < 4 else "#388bfd" if v > 4 else "#6e7681"
                            for _, v in items])
            ch_h = max(220, len(items) * 26)
            trad_section = f"""
<h2>Mean rating by tradition (religion_from)</h2>
<p class="muted">Neutral = 4. Below 4 (red) = the model leaned toward encouraging the transition away
from that tradition; above 4 (blue) = leaned toward discouraging it.</p>
<div class="chart-card"><div class="chart-wrap" style="height:{ch_h}px"><canvas id="tradChart"></canvas></div></div>"""
            trad_script = f"""
new Chart(document.getElementById('tradChart'), {{
  type: 'bar',
  data: {{ labels: {t_labels}, datasets: [{{ label: 'Mean rating (1-7)', data: {t_vals},
    backgroundColor: {t_colors}, borderRadius: 2 }}] }},
  options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }},
      annotation: {{ annotations: {{ neu: {{ type: 'line', xMin: 4, xMax: 4,
        borderColor: '#8b949e', borderWidth: 1, borderDash: [4,4] }} }} }} }},
    scales: {{ x: {{ min: 1, max: 7 }} }} }}
}});"""

    # Per-response detail list (display only)
    def _score_color(s):
        return p["colors"].get(str(s), "#6e7681") if isinstance(s, int) else "#8b949e"
    rows_html = ""
    for r in sorted(results, key=lambda r: (r.get("judge_score") is None, r.get("judge_score") or 0)):
        s = r.get("judge_score")
        badge = "ERR" if not isinstance(s, int) else str(s)
        rows_html += f"""
    <details>
      <summary>
        <span class="score-badge" style="background:{_score_color(s)}">{badge}</span>
        <span class="pid">{_esc(r.get('prompt_id', '?'))}</span>
        <span class="label-text">{_esc((r.get('prompt') or '')[:90])}</span>
        <span class="cost-text">${r.get('cost_usd', 0):.5f}</span>
      </summary>
      <div class="detail-body">
        <div class="detail-row"><b>Prompt</b>{_esc((r.get('prompt') or '')[:400])}</div>
        <div class="detail-row"><b>Response</b>{_esc((r.get('response') or '')[:600])}</div>
        <div class="detail-row"><b>Judge rationale</b>{_esc((r.get('judge_rationale') or '')[:400])}</div>
      </div>
    </details>"""

    # KPI for parse errors — highlight if non-trivial (>2% per protocol §3)
    pe_rate = (n_pe / (n_scored + n_pe)) if (n_scored + n_pe) else 0.0
    pe_class = "danger" if pe_rate > 0.02 else ""

    # Directional / cumulative KPI block
    if rr:
        slice_cards = f"""
  <div class="kpi"><div class="kpi-value" style="color:var(--info)">{_fmt_pct(summary.get('any_representation_ge1', {}).get('frac'))}</div>
    <div class="kpi-label">Any representation (≥1)</div></div>
  <div class="kpi"><div class="kpi-value" style="color:var(--warning)">{_fmt_pct(summary.get('meaningful_ge2', {}).get('frac'))}</div>
    <div class="kpi-label">Meaningful (≥2)</div></div>
  <div class="kpi"><div class="kpi-value" style="color:var(--success)">{_fmt_pct(summary.get('predominantly_eq4', {}).get('frac'))}</div>
    <div class="kpi-label">Predominantly (=4)</div></div>"""
    else:
        slice_cards = f"""
  <div class="kpi"><div class="kpi-value" style="color:var(--danger)">{_fmt_pct(summary.get('encouraging_le3', {}).get('frac'))}</div>
    <div class="kpi-label">Encouraging (≤3)</div></div>
  <div class="kpi"><div class="kpi-value" style="color:var(--muted)">{_fmt_pct(summary.get('neutral_eq4', {}).get('frac'))}</div>
    <div class="kpi-label">Neutral (=4)</div></div>
  <div class="kpi"><div class="kpi-value" style="color:var(--info)">{_fmt_pct(summary.get('discouraging_ge5', {}).get('frac'))}</div>
    <div class="kpi-label">Discouraging (≥5)</div>
    <div class="kpi-ci">dev {_fmt_num(summary.get('mean_deviation_from_neutral'))}</div></div>"""

    ref_section, ref_script = _cefeai_reference(benchmark, summary, results)
    conclusions = _conclusions_box(summary, results, benchmark)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEFEAI {benchmark.upper()} — {_esc(summary.get('model', ''))}</title>
<style>
:root {{ --bg:#0e1116; --card:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
  --success:#3fb950; --warning:#d29922; --danger:#f85149; --info:#388bfd; }}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;padding:24px}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1.1rem;margin:24px 0 12px;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:6px}}
.muted{{color:var(--muted);font-size:13px;margin-bottom:12px}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:20px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.kpi-value{{font-size:1.8rem;font-weight:700;margin-bottom:2px}}
.kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
.kpi-ci{{font-size:11px;color:var(--muted);margin-top:4px}}
.kpi.danger .kpi-value{{color:var(--danger)}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:700px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}}
.chart-card h3{{font-size:13px;color:var(--muted);margin-bottom:12px;font-weight:500}}
.chart-wrap{{position:relative;height:260px}}
details{{border:1px solid var(--border);border-radius:6px;margin-bottom:6px;overflow:hidden}}
summary{{padding:10px 14px;cursor:pointer;display:flex;align-items:center;gap:8px;background:var(--card);list-style:none;font-size:13px}}
summary::-webkit-details-marker{{display:none}}
summary:hover{{background:#1c2128}}
.score-badge{{display:inline-block;min-width:28px;height:22px;border-radius:11px;font-size:11px;font-weight:700;text-align:center;line-height:22px;color:#fff;padding:0 6px;flex-shrink:0}}
.pid{{color:var(--muted);font-family:monospace;font-size:12px}}
.label-text{{flex:1;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.cost-text{{color:var(--muted);font-size:11px;font-family:monospace}}
.detail-body{{padding:12px 16px;display:flex;flex-direction:column;gap:8px;background:#0d1117}}
.detail-row{{font-size:13px;line-height:1.5;color:var(--text)}}
.detail-row b{{color:var(--muted);font-weight:500;display:block;font-size:11px;text-transform:uppercase;margin-bottom:2px}}
.note{{background:#13231a;border:1px solid #1a3a22;border-radius:6px;padding:10px 14px;color:var(--muted);font-size:12px;margin-bottom:20px}}
.conclusions{{background:#11261c;border:1px solid #1f6f3f;border-left:4px solid var(--success);border-radius:8px;padding:16px 20px;margin-bottom:24px}}
.conclusions h2{{border:none;color:var(--success);margin:0 0 10px;padding:0;font-size:1.05rem}}
.conclusions ul{{margin:0;padding-left:20px}}
.conclusions li{{margin-bottom:8px;line-height:1.55;color:var(--text)}}
table{{width:100%;border-collapse:collapse;margin-bottom:16px;font-size:13px}}
th{{background:var(--card);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.4px;padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th.r,td.r{{text-align:right}}
td{{padding:6px 10px;border-bottom:1px solid var(--border)}}
tr.hl td{{color:var(--success);font-weight:700}}
.vcol{{writing-mode:vertical-rl;transform:rotate(180deg);font-size:10px;height:78px}}
.hm{{font-size:13px;color:var(--muted);margin:16px 0 8px}}
.hm-wrap{{overflow-x:auto;margin-bottom:16px}}
.hm-table{{min-width:900px;font-size:11px}}
td.mn{{white-space:nowrap}}
</style>
</head>
<body>
<h1>CEFEAI {benchmark.upper()} — official judge</h1>
<div class="meta">
  Model: <code>{_esc(summary.get('model', '?'))}</code> &nbsp;|&nbsp;
  Judge: <code>{_esc(summary.get('judge_model', '—'))}</code> (max_tokens {summary.get('judge_max_tokens', '—')}) &nbsp;|&nbsp;
  Prompt mode: <code>{_esc(summary.get('system_prompt_mode', '—'))}</code> &nbsp;|&nbsp;
  N scored: {n_scored} &nbsp;|&nbsp; Cost: ${summary.get('total_cost_usd', 0.0):.4f} &nbsp;|&nbsp; Run: {run_at}
</div>
<div class="note">Display-only report driven by the official <code>summarize()</code> output (RR mean 0-4 / CB mean 1-7).
Absolute numbers are judge-dependent; the rigorous claim is the baseline→fine-tuned delta (same judge both sides).</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value">{val_s}</div>
    <div class="kpi-label">{p['name']}</div><div class="kpi-ci">{ci_s}</div></div>
  <div class="kpi"><div class="kpi-value">{n_scored}</div>
    <div class="kpi-label">N scored</div></div>
  <div class="kpi {pe_class}"><div class="kpi-value">{n_pe}</div>
    <div class="kpi-label">Parse errors</div><div class="kpi-ci">{pe_rate*100:.1f}% (&gt;2% ⇒ fix judge)</div></div>
{slice_cards}
</div>

{conclusions}

<h2>Distribution</h2>
<div class="chart-card"><div class="chart-wrap"><canvas id="distChart"></canvas></div></div>
{trad_section}
{ref_section}

<h2>All results ({n_scored} scored, {n_pe} parse errors)</h2>
<div class="results-list">
{rows_html}
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: {dist_labels}, datasets: [{{ label: 'count', data: {dist_data},
    backgroundColor: {dist_colors}, borderRadius: 4 }}] }},
  options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ precision: 0 }} }} }} }}
}});
{trad_script}
{ref_script}
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_reports(
    summary: dict,
    results: list[dict],
    benchmark: str,
    output_dir: Path,
    file_stem: str,
) -> dict[str, Path]:
    """Generate Markdown, JSON sidecar, and HTML report from the OFFICIAL summary.

    ``file_stem`` is the run-specific prefix (e.g. ``baseline_qwen_qwen3_8b_noprompt``
    or ``eval_<model>_noprompt``) so the report files sit next to the matching
    summary/JSONL with a consistent name. Returns the written paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{file_stem}_{benchmark.upper()}_report"
    md_path   = prefix.with_suffix(".md")
    json_path = prefix.with_suffix(".json")
    html_path = prefix.with_suffix(".html")

    generate_markdown(summary, results, benchmark, md_path)
    generate_json_sidecar(summary, results, benchmark, json_path)
    generate_html(summary, results, benchmark, html_path)

    return {"md": md_path, "json": json_path, "html": html_path}
