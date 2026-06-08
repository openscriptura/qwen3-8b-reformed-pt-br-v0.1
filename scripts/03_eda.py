"""
03_eda.py — Exploratory Data Analysis for OpenScriptura training corpus.

Reads all available tier JSONL files (C, B, A) from data/ and produces:
  - Console summary (counts, lengths, coverage)
  - reports/eda_report.html  — dark-mode interactive report
  - reports/eda_report.md    — markdown version

Run at any stage; it simply skips tiers whose files don't exist yet.

Usage:
    python scripts/03_eda.py
    python scripts/03_eda.py --data-dir data/       # explicit data root
    python scripts/03_eda.py --out-dir reports/      # explicit output dir
"""

import sys
import json
import argparse
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import get_logger

log = get_logger("03_eda")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIER_FILES = {
    "C": "data/tier_c/tier_c.jsonl",
    "B": "data/tier_b/tier_b.jsonl",
    "A": "data/tier_a/tier_a.jsonl",
}
TIER_TARGETS = {"C": (1000, 2000), "B": (3000, 4000), "A": (500, 1000)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> list[dict]:
    records = []
    for tier, rel_path in TIER_FILES.items():
        path = data_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        # Also try relative to project root
        if not path.exists():
            path = PROJECT_ROOT / rel_path
        if not path.exists():
            log.info(f"  Tier {tier}: not found at {path} — skipping")
            continue
        tier_records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        log.info(f"  Tier {tier}: {len(tier_records):,} records from {path.name}")
        records.extend(tier_records)
    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _msg_text(record: dict) -> str:
    """Concatenate all message content for length analysis."""
    msgs = record.get("messages", [])
    return " ".join(m.get("content", "") for m in msgs if m.get("role") != "system")


def _answer_text(record: dict) -> str:
    msgs = record.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def _question_text(record: dict) -> str:
    msgs = record.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def analyze(records: list[dict]) -> dict:
    if not records:
        return {}

    # --- Per-tier counts ---
    by_tier: Counter = Counter(r.get("tier", "?") for r in records)

    # --- Per-source counts ---
    by_source: Counter = Counter()
    source_families: Counter = Counter()
    for r in records:
        src = r.get("source", "unknown")
        by_source[src] += 1
        # Family = first underscore-separated token (e.g. "WCF", "WSC", "Heidelberg")
        family = src.split("_")[0] if "_" in src else src
        source_families[family] += 1

    # --- Message length distribution ---
    q_lens = [len(_question_text(r).split()) for r in records]
    a_lens = [len(_answer_text(r).split()) for r in records]
    t_lens = [len(_msg_text(r).split()) for r in records]

    def _stats(values: list[int]) -> dict:
        if not values:
            return {}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "p90": round(sorted(values)[int(len(values) * 0.9)], 1),
            "p99": round(sorted(values)[int(len(values) * 0.99)], 1),
        }

    # --- Confessional reference coverage ---
    ref_coverage: Counter = Counter()
    records_with_refs = 0
    for r in records:
        refs = r.get("confessional_refs", [])
        if refs:
            records_with_refs += 1
            for ref in refs:
                doc = ref.split()[0] if " " in ref else ref
                ref_coverage[doc] += 1

    # --- Duplicate detection (sha256) ---
    sha_counts: Counter = Counter(r.get("sha256", "") for r in records)
    duplicates = {k: v for k, v in sha_counts.items() if v > 1 and k}

    # --- Quality score distribution (Tier B) ---
    quality_scores = [r["quality_score"] for r in records if "quality_score" in r and r.get("tier") == "B"]

    # --- Tradition / lang distribution ---
    by_tradition = Counter(r.get("tradition", "?") for r in records)
    by_lang = Counter(r.get("lang", "?") for r in records)

    return {
        "total": len(records),
        "by_tier": dict(by_tier),
        "by_source_family": dict(source_families.most_common(20)),
        "unique_sources": len(by_source),
        "question_len": _stats(q_lens),
        "answer_len": _stats(a_lens),
        "total_len": _stats(t_lens),
        "records_with_refs": records_with_refs,
        "ref_pct": round(records_with_refs / len(records) * 100, 1),
        "ref_coverage": dict(ref_coverage.most_common(10)),
        "duplicates": len(duplicates),
        "dup_sha_count": sum(duplicates.values()),
        "quality_scores": {
            "count": len(quality_scores),
            "mean": round(statistics.mean(quality_scores), 1) if quality_scores else None,
            "min": min(quality_scores) if quality_scores else None,
            "p10": sorted(quality_scores)[int(len(quality_scores) * 0.1)] if quality_scores else None,
        },
        "by_tradition": dict(by_tradition),
        "by_lang": dict(by_lang),
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_summary(stats: dict, records: list[dict]):
    div = "─" * 60
    print(f"\n{div}")
    print(f"  OpenScriptura — EDA Report  ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(div)
    print(f"\n  Total records : {stats['total']:,}")
    print(f"  Unique sources: {stats['unique_sources']:,}")
    print(f"  Duplicates    : {stats['duplicates']} sha256 ({stats['dup_sha_count']} total affected)")

    print(f"\n  ── By Tier ──")
    for tier in ["C", "B", "A"]:
        count = stats["by_tier"].get(tier, 0)
        lo, hi = TIER_TARGETS.get(tier, (0, 0))
        status = "✅" if lo <= count <= hi else ("⚠️ " if count < lo else "📈")
        bar = "█" * int(count / hi * 20) if hi else ""
        print(f"    Tier {tier}: {count:>5,}  target {lo:,}–{hi:,}  {status}  {bar}")

    print(f"\n  ── Source Families ──")
    for fam, cnt in sorted(stats["by_source_family"].items(), key=lambda x: -x[1]):
        print(f"    {fam:<20} {cnt:>4} records")

    print(f"\n  ── Message Length (words) ──")
    for label, key in [("Question", "question_len"), ("Answer", "answer_len")]:
        s = stats[key]
        if s:
            print(f"    {label}: min={s['min']} mean={s['mean']} median={s['median']} p90={s['p90']} max={s['max']}")

    print(f"\n  ── Coverage ──")
    print(f"    Records with confessional refs: {stats['records_with_refs']:,} ({stats['ref_pct']}%)")
    if stats["quality_scores"]["count"]:
        qs = stats["quality_scores"]
        print(f"    Tier B quality score: n={qs['count']}, mean={qs['mean']}, min={qs['min']}, p10={qs['p10']}")

    print(f"\n{div}\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(stats: dict, out_path: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# OpenScriptura — EDA Report",
        f"",
        f"Generated: {ts}",
        f"",
        f"## Overview",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total records | {stats['total']:,} |",
        f"| Unique sources | {stats['unique_sources']:,} |",
        f"| Duplicate sha256 | {stats['duplicates']} |",
        f"| Records with refs | {stats['records_with_refs']:,} ({stats['ref_pct']}%) |",
        f"",
        f"## Tier Breakdown",
        f"",
        f"| Tier | Count | Target | Status |",
        f"|---|---:|---|---|",
    ]
    for tier in ["C", "B", "A"]:
        count = stats["by_tier"].get(tier, 0)
        lo, hi = TIER_TARGETS.get(tier, (0, 0))
        status = "✅ On track" if lo <= count <= hi else ("⚠️ Below target" if count < lo else "📈 Above target")
        lines.append(f"| {tier} | {count:,} | {lo:,}–{hi:,} | {status} |")

    lines += [
        f"",
        f"## Source Families",
        f"",
        f"| Family | Records |",
        f"|---|---:|",
    ]
    for fam, cnt in sorted(stats["by_source_family"].items(), key=lambda x: -x[1]):
        lines.append(f"| {fam} | {cnt:,} |")

    lines += [
        f"",
        f"## Message Length (words)",
        f"",
        f"| Field | Min | Mean | Median | p90 | Max |",
        f"|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("Question", "question_len"), ("Answer", "answer_len"), ("Total", "total_len")]:
        s = stats.get(key, {})
        if s:
            lines.append(f"| {label} | {s['min']} | {s['mean']} | {s['median']} | {s['p90']} | {s['max']} |")

    if stats["quality_scores"]["count"]:
        qs = stats["quality_scores"]
        lines += [
            f"",
            f"## Tier B Quality Scores",
            f"",
            f"n={qs['count']}, mean={qs['mean']}, min={qs['min']}, p10={qs['p10']}",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Markdown: {out_path}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_html(stats: dict, records: list[dict], out_path: Path):
    import json as _json

    # Tier bar chart
    tier_labels = ["C", "B", "A"]
    tier_counts = [stats["by_tier"].get(t, 0) for t in tier_labels]
    tier_targets_hi = [TIER_TARGETS[t][1] for t in tier_labels]

    # Source family bar chart (top 10)
    top_families = sorted(stats["by_source_family"].items(), key=lambda x: -x[1])[:10]
    fam_labels = [f for f, _ in top_families]
    fam_counts = [c for _, c in top_families]

    # Answer length histogram (50-word buckets)
    a_lens = [len(_answer_text(r).split()) for r in records]
    bucket_size = 50
    max_len = max(a_lens) if a_lens else 0
    buckets = list(range(0, max_len + bucket_size, bucket_size))
    hist = Counter((l // bucket_size) * bucket_size for l in a_lens)
    hist_labels = [str(b) for b in buckets]
    hist_counts = [hist.get(b, 0) for b in buckets]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenScriptura EDA</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0e1116;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--success:#3fb950;--warning:#d29922;--info:#388bfd}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;padding:24px}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1.05rem;margin:24px 0 12px;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:6px}}
.meta{{color:var(--muted);font-size:12px;margin-bottom:24px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.kpi-value{{font-size:1.8rem;font-weight:700;margin-bottom:2px}}
.kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
@media(max-width:700px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px}}
.chart-card h3{{font-size:13px;color:var(--muted);margin-bottom:12px;font-weight:500}}
.chart-wrap{{position:relative;height:240px}}
.status-ok{{color:var(--success)}} .status-warn{{color:var(--warning)}}
table{{width:100%;border-collapse:collapse;margin-bottom:24px;background:var(--card);border-radius:8px;overflow:hidden}}
th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}}
th.r,td.r{{text-align:right}}
td{{padding:8px 12px;border-bottom:1px solid var(--border);font-size:13px}}
</style>
</head>
<body>
<h1>OpenScriptura — EDA Report</h1>
<div class="meta">Generated: {ts}</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value">{stats['total']:,}</div><div class="kpi-label">Total records</div></div>
  <div class="kpi"><div class="kpi-value">{stats['unique_sources']:,}</div><div class="kpi-label">Unique sources</div></div>
  <div class="kpi"><div class="kpi-value">{stats['ref_pct']}%</div><div class="kpi-label">With conf. refs</div></div>
  <div class="kpi"><div class="kpi-value">{stats['duplicates']}</div><div class="kpi-label">Duplicates</div></div>
</div>

<h2>Tier Progress</h2>
<table>
  <thead><tr><th>Tier</th><th class="r">Count</th><th class="r">Target</th><th>Progress</th><th>Status</th></tr></thead>
  <tbody>
"""
    for tier in ["C", "B", "A"]:
        count = stats["by_tier"].get(tier, 0)
        lo, hi = TIER_TARGETS[tier]
        pct = min(count / hi * 100, 100) if hi else 0
        status = "✅ On track" if lo <= count <= hi else ("⚠️ Below target" if count < lo else "📈 Above target")
        bar = f'<div style="background:#30363d;border-radius:4px;height:8px;width:120px"><div style="background:#388bfd;border-radius:4px;height:8px;width:{pct:.0f}%"></div></div>'
        html += f"  <tr><td>Tier {tier}</td><td class='r'>{count:,}</td><td class='r'>{lo:,}–{hi:,}</td><td>{bar}</td><td>{status}</td></tr>\n"

    html += f"""  </tbody>
</table>

<div class="chart-grid">
  <div class="chart-card">
    <h3>Records by Source Family</h3>
    <div class="chart-wrap"><canvas id="famChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>Answer Length Distribution (words)</h3>
    <div class="chart-wrap"><canvas id="lenChart"></canvas></div>
  </div>
</div>

<h2>Length Statistics</h2>
<table>
  <thead><tr><th>Field</th><th class="r">Min</th><th class="r">Mean</th><th class="r">Median</th><th class="r">p90</th><th class="r">Max</th></tr></thead>
  <tbody>
"""
    for label, key in [("Question", "question_len"), ("Answer", "answer_len"), ("Total (Q+A)", "total_len")]:
        s = stats.get(key, {})
        if s:
            html += f"  <tr><td>{label}</td><td class='r'>{s['min']}</td><td class='r'>{s['mean']}</td><td class='r'>{s['median']}</td><td class='r'>{s['p90']}</td><td class='r'>{s['max']}</td></tr>\n"
    html += "  </tbody>\n</table>\n"

    if stats["quality_scores"]["count"]:
        qs = stats["quality_scores"]
        html += f"""
<h2>Tier B Quality Scores</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-value">{qs['count']:,}</div><div class="kpi-label">Tier B records</div></div>
  <div class="kpi"><div class="kpi-value">{qs['mean']}</div><div class="kpi-label">Mean score</div></div>
  <div class="kpi"><div class="kpi-value">{qs['min']}</div><div class="kpi-label">Min score</div></div>
  <div class="kpi"><div class="kpi-value">{qs['p10']}</div><div class="kpi-label">p10 score</div></div>
</div>
"""

    html += f"""
<script>
new Chart(document.getElementById('famChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(fam_labels)},
    datasets: [{{ label: 'Records', data: {_json.dumps(fam_counts)}, backgroundColor: '#388bfd', borderRadius: 3 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ font: {{ size: 11 }} }} }}, y: {{ ticks: {{ precision: 0 }} }} }}
  }}
}});
new Chart(document.getElementById('lenChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(hist_labels)},
    datasets: [{{ label: 'Records', data: {_json.dumps(hist_counts)}, backgroundColor: '#3fb950', borderRadius: 2 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ font: {{ size: 10 }} }} }}, y: {{ ticks: {{ precision: 0 }} }} }}
  }}
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info(f"  HTML: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA for OpenScriptura corpus")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT, help="Project root (default: auto-detect)")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports", help="Output directory")
    args = parser.parse_args()

    log.info("Loading corpus...")
    records = load_all(args.data_dir)
    if not records:
        log.error("No records found — run 01_build_tier_c.py first.")
        sys.exit(1)

    log.info("Analysing...")
    stats = analyze(records)

    print_summary(stats, records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(stats, args.out_dir / "eda_report.md")
    write_html(stats, records, args.out_dir / "eda_report.html")

    log.info("Done.")


if __name__ == "__main__":
    main()
