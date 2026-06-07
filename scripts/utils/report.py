"""report.py — Markdown, JSON sidecar, and HTML report generation for CEFEAI baseline.

Ported display patterns from pastor-ai/_qa_html_report.py and
run_all_tests_overnight.py. Produces three files per benchmark run:
  results/baseline_<model>_<BENCH>_report.md
  results/baseline_<model>_<BENCH>_report.json
  results/baseline_<model>_<BENCH>_report.html
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
# CEFEAI public leaderboard — for context in reports (README_github.md)
# ---------------------------------------------------------------------------
CEFEAI_LEADERBOARD = [
    ("Grok 4.20",       29.3),
    ("Mistral Large",   23.3),
    ("GPT-5.4",         17.3),
    ("Claude Opus 4.7",  4.0),
    ("Llama 4 Scout",    3.3),
]

SCORE_LABELS = {
    0: "No Representation",
    1: "Any Representation",
    2: "Meaningful Reference",
    3: "Predominantly Religious",
}

SCORE_COLORS = {
    0: "#6e7681",  # gray
    1: "#388bfd",  # blue
    2: "#d29922",  # amber
    3: "#3fb950",  # green
}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(summary: dict, benchmark: str) -> None:
    """Print an emoji-annotated console summary (pastor-ai style)."""
    W = 64
    div = "=" * W
    n = summary["n"]
    ar = summary["any_representation"]
    mr = summary["meaningful_reference"]
    pr = summary["predominantly_religious"]
    nr = summary["no_representation"]
    cost = summary["total_cost_usd"]
    model = summary["model"]

    print()
    print(div)
    print(f"  CEFEAI BASELINE — {benchmark.upper()}  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(div)
    print(f"  Model  : {model}")
    print(f"  N      : {n} prompts")
    print(f"  Cost   : ${cost:.4f}")
    print()

    def row(emoji, label, d):
        pct = d["pct"] * 100
        ci  = f"[{d['ci_low']*100:.1f}%, {d['ci_high']*100:.1f}%]"
        bar = _ascii_bar(d["pct"], width=20)
        print(f"  {emoji} {label:<26}  {pct:5.1f}%  {ci:<18}  {bar}  n={d['n']}")

    row("⬛", "No Representation",     nr)
    row("🔵", "Any Representation",    ar)
    row("🟡", "Meaningful Reference",  mr)
    row("✅", "Predominantly Religious", pr)

    print()
    # Leaderboard comparison (only for RR benchmark)
    if benchmark.lower() == "rr":
        baseline_pct = ar["pct"] * 100
        print(f"  📊 CEFEAI Leaderboard — Any Representation (RR)")
        for name, pct in CEFEAI_LEADERBOARD:
            marker = " ◀ this model" if abs(pct - baseline_pct) < 1 else ""
            print(f"       {name:<22}  {pct:5.1f}%{marker}")
        print(f"       {'Qwen3-8B (this run)':<22}  {baseline_pct:5.1f}%  ◀◀◀")
        gap = 60.0 - baseline_pct
        verdict = "✅ TARGET MET" if baseline_pct >= 60 else f"⚠️  Gap to v0.1 target (>60%): {gap:+.1f}pp"
        print(f"\n  {verdict}")

    print(div)
    print()


def _ascii_bar(frac: float, width: int = 20) -> str:
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    n   = summary["n"]
    ar  = summary["any_representation"]
    mr  = summary["meaningful_reference"]
    pr  = summary["predominantly_religious"]
    nr  = summary["no_representation"]
    model = summary["model"]
    judge = summary.get("judge_model", "—")
    cost  = summary["total_cost_usd"]
    run_at = summary["run_at"]

    def ci_str(d):
        return f"[{d['ci_low']*100:.1f}%, {d['ci_high']*100:.1f}%]"

    def pct(d):
        return f"{d['pct']*100:.1f}%"

    lines = [
        f"# CEFEAI Baseline — {benchmark.upper()}",
        f"",
        f"**Generated:** {run_at}  ",
        f"**Model:** `{model}`  ",
        f"**Judge:** `{judge}`  ",
        f"**N:** {n} prompts | **Cost:** ${cost:.4f}  ",
        f"**enable_thinking:** `false` | **temperature:** `0.0` | **seed:** `42`",
        f"",
        f"---",
        f"",
        f"## Results (95% Wilson CI)",
        f"",
        f"| Metric | n | % | 95% CI |",
        f"|---|---:|---:|---|",
        f"| ⬛ No Representation       | {nr['n']:>4} | {pct(nr):>7} | {ci_str(nr)} |",
        f"| 🔵 Any Representation      | {ar['n']:>4} | {pct(ar):>7} | {ci_str(ar)} |",
        f"| 🟡 Meaningful Reference    | {mr['n']:>4} | {pct(mr):>7} | {ci_str(mr)} |",
        f"| ✅ Predominantly Religious | {pr['n']:>4} | {pct(pr):>7} | {ci_str(pr)} |",
        f"",
        f"---",
        f"",
    ]

    # Score distribution
    dist = {s: 0 for s in range(4)}
    for r in results:
        dist[r.get("judge_score", 0)] += 1

    lines += [
        f"## Score Distribution",
        f"",
        f"| Score | Label | n | % |",
        f"|---:|---|---:|---:|",
    ]
    for score in [3, 2, 1, 0]:
        cnt = dist[score]
        pct_val = cnt / n * 100 if n else 0
        lines.append(f"| {score} | {SCORE_LABELS[score]} | {cnt} | {pct_val:.1f}% |")

    lines += ["", "---", ""]

    # CEFEAI leaderboard comparison (RR only)
    if benchmark.lower() == "rr":
        baseline_pct = ar["pct"] * 100
        gap = 60.0 - baseline_pct
        lines += [
            f"## CEFEAI Leaderboard Comparison",
            f"",
            f"| Model | Any Representation | vs This Run |",
            f"|---|---:|---:|",
        ]
        for name, ldr_pct in CEFEAI_LEADERBOARD:
            delta = ldr_pct - baseline_pct
            sign = "+" if delta > 0 else ""
            lines.append(f"| {name} | {ldr_pct:.1f}% | {sign}{delta:.1f}pp |")
        lines.append(f"| **Qwen3-8B (this run)** | **{baseline_pct:.1f}%** | — |")
        lines += [
            f"",
            f"**v0.1 target:** >60% Any Representation",
            f"**Gap to target:** {gap:+.1f}pp",
            f"",
            f"---",
            f"",
        ]

    # Top failures (score 0 samples, up to 20)
    failures = [r for r in results if r.get("judge_score", 0) == 0][:20]
    if failures:
        lines += [
            f"## Sample No-Representation Responses (first {len(failures)})",
            f"",
        ]
        for i, r in enumerate(failures, 1):
            prompt_short = r.get("prompt", "")[:100].replace("|", "\\|")
            reason_short = r.get("judge_reasoning", "")[:120].replace("|", "\\|")
            lines.append(f"**{i}.** `{r['prompt_id']}` — {prompt_short}…")
            lines.append(f"> Judge: {reason_short}")
            lines.append(f"")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON sidecar (for future trend tracking)
# ---------------------------------------------------------------------------

def generate_json_sidecar(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    dist = {str(s): 0 for s in range(4)}
    for r in results:
        dist[str(r.get("judge_score", 0))] += 1

    sidecar = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark.upper(),
        "model": summary["model"],
        "n": summary["n"],
        "metrics": {
            "no_representation":      summary["no_representation"],
            "any_representation":     summary["any_representation"],
            "meaningful_reference":   summary["meaningful_reference"],
            "predominantly_religious": summary["predominantly_religious"],
        },
        "score_distribution": dist,
        "total_cost_usd": summary["total_cost_usd"],
        "inference_settings": {
            "enable_thinking": False,
            "temperature": 0.0,
            "seed": 42,
        },
    }
    out_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML report with Chart.js (dark mode, pastor-ai style)
# ---------------------------------------------------------------------------

def generate_html(summary: dict, results: list[dict], benchmark: str, out_path: Path) -> None:
    n     = summary["n"]
    ar    = summary["any_representation"]
    mr    = summary["meaningful_reference"]
    pr    = summary["predominantly_religious"]
    nr    = summary["no_representation"]
    model = summary["model"]
    cost  = summary["total_cost_usd"]
    run_at = summary["run_at"][:19].replace("T", " ")

    # Score distribution data for Chart.js
    dist = {s: 0 for s in range(4)}
    for r in results:
        dist[r.get("judge_score", 0)] += 1

    donut_labels  = json.dumps([SCORE_LABELS[s] for s in [3, 2, 1, 0]])
    donut_data    = json.dumps([dist[s] for s in [3, 2, 1, 0]])
    donut_colors  = json.dumps([SCORE_COLORS[s] for s in [3, 2, 1, 0]])

    # Leaderboard bar chart data
    lb_names  = [n for n, _ in CEFEAI_LEADERBOARD] + [f"Qwen3-8B (this)"]
    lb_values = [v for _, v in CEFEAI_LEADERBOARD] + [round(ar["pct"] * 100, 1)]
    lb_colors = ["#388bfd"] * len(CEFEAI_LEADERBOARD) + ["#3fb950"]
    lb_names_js  = json.dumps(lb_names)
    lb_values_js = json.dumps(lb_values)
    lb_colors_js = json.dumps(lb_colors)

    # Collapsible result rows (all results, sorted by score desc)
    sorted_results = sorted(results, key=lambda r: r.get("judge_score", 0), reverse=True)
    rows_html = ""
    for r in sorted_results:
        score = r.get("judge_score", 0)
        label = SCORE_LABELS.get(score, "?")
        color = SCORE_COLORS.get(score, "#aaa")
        pid   = r.get("prompt_id", "?")
        prompt_esc = _esc(r.get("prompt", "")[:200])
        response_esc = _esc(r.get("response", "")[:300])
        reasoning_esc = _esc(r.get("judge_reasoning", ""))
        cost_r = r.get("cost_usd", 0)
        rows_html += f"""
        <details>
          <summary>
            <span class="score-badge" style="background:{color}">{score}</span>
            <span class="pid">{pid}</span>
            <span class="label-text">{label}</span>
            <span class="cost-text">${cost_r:.5f}</span>
          </summary>
          <div class="detail-body">
            <div class="detail-row"><b>Prompt:</b> {prompt_esc}</div>
            <div class="detail-row"><b>Response:</b> {response_esc}</div>
            <div class="detail-row"><b>Judge reasoning:</b> {reasoning_esc}</div>
          </div>
        </details>"""

    verdict_class = "success" if ar["pct"] >= 0.60 else ("warning" if ar["pct"] >= 0.20 else "danger")
    verdict_text  = "TARGET MET ✅" if ar["pct"] >= 0.60 else f"Gap to target: {(60 - ar['pct']*100):+.1f}pp ⚠️"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEFEAI Baseline — OpenScriptura</title>
<style>
:root {{
  --bg:#0e1116; --card:#161b22; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e;
  --success:#3fb950; --warning:#d29922; --danger:#f85149; --info:#388bfd;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;padding:24px}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1.1rem;margin:24px 0 12px;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:6px}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:24px}}
.verdict{{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:600;font-size:13px;margin-bottom:20px}}
.success{{background:#1a3a22;color:var(--success)}}
.warning{{background:#3a2e00;color:var(--warning)}}
.danger{{background:#3a1a1a;color:var(--danger)}}

/* KPI cards */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.kpi-value{{font-size:1.8rem;font-weight:700;margin-bottom:2px}}
.kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
.kpi-ci{{font-size:11px;color:var(--muted);margin-top:4px}}

/* Charts */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:700px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.chart-card h3{{font-size:13px;color:var(--muted);margin-bottom:12px;font-weight:500}}
.chart-wrap{{position:relative;height:240px}}

/* Results table */
table{{width:100%;border-collapse:collapse;margin-bottom:24px}}
th{{background:var(--card);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}}
th.r,td.r{{text-align:right}}
td{{padding:8px 12px;border-bottom:1px solid var(--border);font-size:13px}}
tr:hover td{{background:var(--card)}}

/* Score badge */
.score-badge{{display:inline-block;width:22px;height:22px;border-radius:50%;font-size:12px;font-weight:700;text-align:center;line-height:22px;color:#fff;margin-right:6px;flex-shrink:0}}

/* Collapsible results */
.results-list{{margin-bottom:24px}}
details{{border:1px solid var(--border);border-radius:6px;margin-bottom:6px;overflow:hidden}}
summary{{
  padding:10px 14px;cursor:pointer;display:flex;align-items:center;gap:8px;
  background:var(--card);list-style:none;font-size:13px;
}}
summary::-webkit-details-marker{{display:none}}
summary:hover{{background:#1c2128}}
.pid{{color:var(--muted);font-family:monospace;font-size:12px}}
.label-text{{flex:1;color:var(--text)}}
.cost-text{{color:var(--muted);font-size:11px;font-family:monospace}}
.detail-body{{padding:12px 16px;display:flex;flex-direction:column;gap:8px;background:#0d1117}}
.detail-row{{font-size:13px;line-height:1.5;color:var(--text)}}
.detail-row b{{color:var(--muted);font-weight:500;display:block;font-size:11px;text-transform:uppercase;margin-bottom:2px}}

/* Leaderboard */
.lb-table td:first-child{{font-weight:500}}
.highlight-row td{{color:var(--success);font-weight:700}}
</style>
</head>
<body>

<h1>CEFEAI Baseline — {benchmark.upper()}</h1>
<div class="meta">
  Model: <code>{model}</code> &nbsp;|&nbsp;
  N: {n} prompts &nbsp;|&nbsp;
  Cost: ${cost:.4f} &nbsp;|&nbsp;
  Run: {run_at}
</div>
<div class="verdict {verdict_class}">{verdict_text}</div>

<!-- KPI cards -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-value" style="color:var(--muted)">{nr['pct']*100:.1f}%</div>
    <div class="kpi-label">No Representation</div>
    <div class="kpi-ci">95% CI [{nr['ci_low']*100:.1f}%, {nr['ci_high']*100:.1f}%]</div>
  </div>
  <div class="kpi">
    <div class="kpi-value" style="color:var(--info)">{ar['pct']*100:.1f}%</div>
    <div class="kpi-label">Any Representation</div>
    <div class="kpi-ci">95% CI [{ar['ci_low']*100:.1f}%, {ar['ci_high']*100:.1f}%]</div>
  </div>
  <div class="kpi">
    <div class="kpi-value" style="color:var(--warning)">{mr['pct']*100:.1f}%</div>
    <div class="kpi-label">Meaningful Reference</div>
    <div class="kpi-ci">95% CI [{mr['ci_low']*100:.1f}%, {mr['ci_high']*100:.1f}%]</div>
  </div>
  <div class="kpi">
    <div class="kpi-value" style="color:var(--success)">{pr['pct']*100:.1f}%</div>
    <div class="kpi-label">Predominantly Religious</div>
    <div class="kpi-ci">95% CI [{pr['ci_low']*100:.1f}%, {pr['ci_high']*100:.1f}%]</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">${cost:.4f}</div>
    <div class="kpi-label">Total cost (USD)</div>
    <div class="kpi-ci">{n} prompts + {n} judge calls</div>
  </div>
</div>

<!-- Charts -->
<h2>Charts</h2>
<div class="chart-grid">
  <div class="chart-card">
    <h3>Score Distribution</h3>
    <div class="chart-wrap"><canvas id="donutChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>CEFEAI Leaderboard — Any Representation</h3>
    <div class="chart-wrap"><canvas id="lbChart"></canvas></div>
  </div>
</div>

<!-- Metrics table -->
<h2>Results Table</h2>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th class="r">n</th>
      <th class="r">%</th>
      <th class="r">CI low</th>
      <th class="r">CI high</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>⬛ No Representation</td>      <td class="r">{nr['n']}</td><td class="r">{nr['pct']*100:.1f}%</td><td class="r">{nr['ci_low']*100:.1f}%</td><td class="r">{nr['ci_high']*100:.1f}%</td></tr>
    <tr><td>🔵 Any Representation</td>     <td class="r">{ar['n']}</td><td class="r">{ar['pct']*100:.1f}%</td><td class="r">{ar['ci_low']*100:.1f}%</td><td class="r">{ar['ci_high']*100:.1f}%</td></tr>
    <tr><td>🟡 Meaningful Reference</td>   <td class="r">{mr['n']}</td><td class="r">{mr['pct']*100:.1f}%</td><td class="r">{mr['ci_low']*100:.1f}%</td><td class="r">{mr['ci_high']*100:.1f}%</td></tr>
    <tr><td>✅ Predominantly Religious</td><td class="r">{pr['n']}</td><td class="r">{pr['pct']*100:.1f}%</td><td class="r">{pr['ci_low']*100:.1f}%</td><td class="r">{pr['ci_high']*100:.1f}%</td></tr>
  </tbody>
</table>

<!-- Per-result list -->
<h2>All Results ({n})</h2>
<div class="results-list">
{rows_html}
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script>
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

// Donut — score distribution
new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: {donut_labels},
    datasets: [{{
      data: {donut_data},
      backgroundColor: {donut_colors},
      borderWidth: 2,
      borderColor: '#0e1116'
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 12, font: {{ size: 11 }} }} }}
    }}
  }}
}});

// Bar — leaderboard
new Chart(document.getElementById('lbChart'), {{
  type: 'bar',
  data: {{
    labels: {lb_names_js},
    datasets: [{{
      label: 'Any Representation %',
      data: {lb_values_js},
      backgroundColor: {lb_colors_js},
      borderRadius: 4
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{
        min: 0, max: 100,
        ticks: {{ callback: v => v + '%' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all_reports(
    summary: dict,
    results: list[dict],
    benchmark: str,
    output_dir: Path,
    model_slug: str,
) -> dict[str, Path]:
    """Generate Markdown, JSON sidecar, and HTML report. Returns paths dict."""
    prefix = output_dir / f"baseline_{model_slug}_{benchmark.upper()}_report"
    md_path   = prefix.with_suffix(".md")
    json_path = prefix.with_suffix(".json")
    html_path = prefix.with_suffix(".html")

    generate_markdown(summary, results, benchmark, md_path)
    generate_json_sidecar(summary, results, benchmark, json_path)
    generate_html(summary, results, benchmark, html_path)

    return {"md": md_path, "json": json_path, "html": html_path}
