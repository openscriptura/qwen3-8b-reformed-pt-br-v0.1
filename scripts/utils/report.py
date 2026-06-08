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
# CEFEAI public leaderboard — Religious Representation (RR), Any Representation %
# Source: https://cefe.ai — data as visible on leaderboard.
# Non-Qwen models sorted high → low. Qwen family grouped at bottom.
# Update this list when new models are added to the CEFEAI leaderboard.
# ---------------------------------------------------------------------------
CEFEAI_LEADERBOARD = [
    # Non-Qwen — sorted by Any Representation %, high → low
    # Source: cefe.ai Religious Representation leaderboard, captured 2026-06-07
    ("Grok 4.20",              29.3),
    ("Mistral Large 2512",     23.3),
    ("GPT-5.4",                17.3),
    ("GPT-5",                  14.2),
    ("GPT-5.4 Nano",           14.0),
    ("Mistral Small 3.2",      14.0),
    ("GPT-5.5",                13.3),
    ("GPT-5.2",                12.7),
    ("DeepSeek v4 Pro",        12.0),
    ("GPT-5.1",                12.0),
    ("DeepSeek v4 Flash",       8.7),
    ("Ernie 4.5",               7.3),
    ("Grok 4.3",                6.7),
    ("Kimi K2.5",               6.7),
    ("Gemini 3.1 Pro",          6.0),
    ("Kimi K2.6",               6.0),
    ("Claude Opus 4.6",         5.3),
    ("Claude Sonnet 4.6",       4.7),
    ("Gemini 3.1 Flash Lite",   4.7),
    ("Claude Opus 4.7",         4.0),
    ("GPT-4.1",                 4.0),
    ("Llama 4 Scout",           3.3),
    ("Claude Haiku 4.5",        2.0),
    ("Llama 4 Maverick",        2.0),
    ("GPT-4o",                  1.3),
    # --- Qwen family — grouped for easy comparison ---
    ("Qwen3-8B (this run)",     4.7),
    ("Qwen 3.6 Flash",          6.0),
    ("Qwen 3.6 Max",            5.3),
]

# All values verified from cefe.ai screenshot, 2026-06-07.

# ---------------------------------------------------------------------------
# CEFEAI CB leaderboard — Total Bias %, ranked best→worst
# ---------------------------------------------------------------------------
CEFEAI_CB_LEADERBOARD = [
    ("Claude Opus 4.6",        9.2),
    ("Llama 4 Maverick",      12.8),
    ("Mistral Small 3.2",     14.3),
    ("Claude Sonnet 4.6",     15.6),
    ("Qwen 3.6 Max",          16.9),
    ("Llama 4 Scout",         17.4),
    ("Claude Opus 4.7",       18.1),
    ("GPT-5.4",               20.7),
    ("Mistral Large 2512",    21.1),
    ("Grok 4.3",              21.2),
    ("Qwen 3.6 Flash",        21.4),
    ("GPT-5.5",               23.6),
    ("GPT-5.4 Nano",          26.2),
    ("Gemini 3.1 Flash Lite", 28.6),
    ("Gemini 3.1 Pro",        28.6),
    ("DeepSeek v4 Flash",     29.7),
    ("Grok 4.1 Fast",         30.8),
    ("DeepSeek Chat v3.1",    32.1),
    ("DeepSeek v4 Pro",       32.6),
    ("Grok 4.20",             50.4),
]

# ---------------------------------------------------------------------------
# CEFEAI CB per-faith, per-model bias matrices
# Columns: Agnostic Atheist Bahá'í Buddhist Catholic EvangelicalProtestant
#          Hindu JehovahsWitness Jewish LDS Protestant(Mainline) ShiaMuslim Sikh SunniMuslim
# ---------------------------------------------------------------------------
_CB_FAITH_COLS = [
    "Agnostic","Atheist","Bahá'í","Buddhist","Catholic",
    "Evangelical\nProtestant","Hindu","Jehovah's\nWitness","Jewish",
    "Latter-day\nSaint","Protestant\n(Mainline)","Shia\nMuslim","Sikh","Sunni\nMuslim",
]

# Total Bias per model per faith (%)
CEFEAI_CB_TOTAL = {
    "Claude Opus 4.6":        [13,10, 6, 6,10, 8, 5,31, 9, 9,10, 5, 5, 0],
    "Llama 4 Maverick":       [26,21,12,10,17,10, 6, 6, 9,12,19,12,10,10],
    "Mistral Small 3.2":      [36,26,17,23,10, 5,18, 8, 4, 6,18, 8,15, 6],
    "Claude Sonnet 4.6":      [13,19,14,19,10,15,13,37, 9,22,15, 8,12,12],
    "Qwen 3.6 Max":           [47,32,18,26,14,12,18,14, 9, 6,18, 4,12, 6],
    "Llama 4 Scout":          [32,32,21,17,15,21,23,13,13,10,21,10, 8, 9],
    "Claude Opus 4.7":        [26,15,13,12,19,22,10,45,18,21,23,13, 9, 9],
    "GPT-5.4":                [38,27,19,23,22,21,23,33, 9,12,27,14,15, 6],
    "Mistral Large 2512":     [35,24,28,28,14,15,23,24,12,17,26,15,24, 9],
    "Grok 4.3":               [38,37,19,24,15,15,24,26,15,13,21,17,18,14],
    "Qwen 3.6 Flash":         [49,36,28,24,15,14,28,13,21, 6,22,15,17,12],
    "GPT-5.5":                [38,31,23,29,22,19,26,22,14,18,24,22,26,17],
    "GPT-5.4 Nano":           [37,31,18,31,27,21,36,22,28,26,29,19,26,17],
    "Gemini 3.1 Flash Lite":  [46,44,31,35,27,24,32,33,18,18,31,15,27,19],
    "Gemini 3.1 Pro":         [50,46,33,45,26,13,33,14,27,17,29,24,26,17],
    "DeepSeek v4 Flash":      [53,41,29,33,27,26,31,23,26,26,38,19,23,21],
    "Grok 4.1 Fast":          [53,51,23,26,27,27,27,35,27,23,29,27,31,26],
    "DeepSeek Chat v3.1":     [51,46,41,33,29,26,29,26,21,33,40,23,27,23],
    "DeepSeek v4 Pro":        [53,40,40,38,33,23,38,29,21,31,36,23,29,22],
    "Grok 4.20":              [55,60,49,51,72,56,54,74,29,35,59,31,42,37],
}

# Positive Bias (model encouraged joining the faith)
CEFEAI_CB_POSITIVE = {
    "Claude Opus 4.6":        [ 5, 4, 1, 5,10, 5, 4, 1, 9, 1,10, 4, 4, 0],
    "Llama 4 Maverick":       [ 9, 1,12, 6,13, 4, 4, 0, 4, 9, 9, 6, 6, 6],
    "Mistral Small 3.2":      [ 9, 5,17,10, 8, 1,10, 3, 3, 6,10, 3,10, 5],
    "Claude Sonnet 4.6":      [ 5, 9, 5,12,10,12, 8, 1, 8, 5,13, 5,10, 6],
    "Qwen 3.6 Max":           [23,13,12,13,12, 3,10, 0, 4, 3,10, 3, 9, 5],
    "Llama 4 Scout":          [10,10,19,13,10, 5,15, 1, 5, 4, 9, 8, 8, 4],
    "Claude Opus 4.7":        [17, 6, 9, 4,17,13, 6, 0,15, 3,19, 8, 6, 4],
    "GPT-5.4":                [15, 6,14,10,21,15, 9, 0, 8, 4,17, 9,10, 6],
    "Mistral Large 2512":     [14, 6,26,21, 9, 8,13, 0, 5, 9,12, 5,17, 4],
    "Grok 4.3":               [21,22,10,15,10, 8,13, 0, 9, 5,10, 5,13, 8],
    "Qwen 3.6 Flash":         [26,14,22,12,10, 3,15, 0, 9, 3, 8, 9,12, 9],
    "GPT-5.5":                [14, 6,13,13,22,13, 9, 0, 8, 8,18,15,14,13],
    "GPT-5.4 Nano":           [13,12,14,15,24, 9,15, 0,18,14,13,10,14,12],
    "Gemini 3.1 Flash Lite":  [21,18,26,18,19, 9,18, 0,12,10,15, 5,21, 9],
    "Gemini 3.1 Pro":         [24,22,23,27,15, 5,18, 0,15, 8,12,10,14, 6],
    "DeepSeek v4 Flash":      [23,15,18,14,19,14,15, 1, 8,17,22,12,15,14],
    "Grok 4.1 Fast":          [27,23,18,14,19,13,17, 1,15, 8,14,12,23,12],
    "DeepSeek Chat v3.1":     [18,17,35,14,21,10,14, 5, 9,24,18,10,17,13],
    "DeepSeek v4 Pro":        [24,14,28,21,24,13,18, 0, 8,21,15,15,14,13],
    "Grok 4.20":              [29,33, 6,21,69,51,18, 8,15,14,45, 5,21,17],
}

# Negative Bias (model encouraged leaving the faith)
CEFEAI_CB_NEGATIVE = {
    "Claude Opus 4.6":        [ 8, 6, 5, 1, 0, 3, 1,29, 0, 8, 0, 1, 1, 0],
    "Llama 4 Maverick":       [17,19, 0, 4, 4, 6, 3, 6, 5, 3,10, 5, 4, 4],
    "Mistral Small 3.2":      [27,21, 0,13, 3, 4, 8, 5, 1, 0, 8, 5, 5, 1],
    "Claude Sonnet 4.6":      [ 8,10, 9, 8, 0, 4, 5,36, 1,17, 3, 3, 1, 5],
    "Qwen 3.6 Max":           [24,19, 6,13, 3, 9, 8,14, 5, 4, 8, 1, 3, 1],
    "Llama 4 Scout":          [22,22, 1, 4, 5,15, 8,12, 8, 6,12, 3, 0, 5],
    "Claude Opus 4.7":        [ 9, 9, 4, 8, 3, 9, 4,45, 3,18, 4, 5, 3, 5],
    "GPT-5.4":                [23,21, 5,13, 1, 5,14,33, 1, 8,10, 5, 5, 0],
    "Mistral Large 2512":     [21,18, 3, 8, 5, 8,10,24, 6, 8,14,10, 8, 5],
    "Grok 4.3":               [18,15, 9, 9, 5, 8,12,26, 6, 8,10,12, 5, 6],
    "Qwen 3.6 Flash":         [23,22, 6,13, 5,12,13,13,12, 4,14, 6, 5, 3],
    "GPT-5.5":                [24,24,10,17, 0, 6,17,22, 6,10, 6, 6,12, 4],
    "GPT-5.4 Nano":           [24,19, 4,15, 3,12,21,22,10,12,17, 9,12, 5],
    "Gemini 3.1 Flash Lite":  [26,26, 5,17, 8,15,14,33, 6, 8,15,10, 6,10],
    "Gemini 3.1 Pro":         [26,24,10,18,10, 8,15,14,12, 9,18,14,12,10],
    "DeepSeek v4 Flash":      [29,26,12,19, 8,12,15,22,18, 9,17, 8, 8, 6],
    "Grok 4.1 Fast":          [26,28, 5,12, 8,14,10,33,12,15,15,15, 8,14],
    "DeepSeek Chat v3.1":     [33,29, 6,19, 9,15,15,21,12, 9,22,13,10,10],
    "DeepSeek v4 Pro":        [28,26,12,18, 9,10,21,29,13,10,21, 8,15, 9],
    "Grok 4.20":              [26,27,42,31, 3, 5,36,67,14,21,14,26,22,21],
}

# Net Bias (Positive − Negative, signed %)
CEFEAI_CB_NET = {
    "Claude Opus 4.6":        [ -3, -3, -4, +4,+10, +3, +3,-28, +9, -6,+10, +3, +3,  0],
    "Llama 4 Maverick":       [ -8,-18,+12, +3, +9, -3, +1, -6, -1, +6, -1, +1, +3, +3],
    "Mistral Small 3.2":      [-18,-15,+17, -3, +5, -3, +3, -3, +1, +6, +3, -3, +5, +4],
    "Claude Sonnet 4.6":      [ -3, -1, -4, +4,+10, +8, +3,-35, +6,-12,+10, +3, +9, +1],
    "Qwen 3.6 Max":           [ -1, -6, +5,  0, +9, -6, +3,-14, -1, -1, +3, +1, +6, +4],
    "Llama 4 Scout":          [-12,-12,+18, +9, +5,-10, +8,-10, -3, -3, -3, +5, +8, -1],
    "Claude Opus 4.7":        [ +8, -3, +5, -4,+14, +4, +3,-45,+13,-15,+15, +3, +4, -1],
    "GPT-5.4":                [ -8,-14, +9, -3,+19,+10, -5,-33, +6, -4, +6, +4, +5, +6],
    "Mistral Large 2512":     [ -6,-12,+23,+13, +4,  0, +3,-24, -1, +1, -3, -5, +9, -1],
    "Grok 4.3":               [ +3, +6, +1, +6, +5,  0, +1,-26, +3, -3,  0, -6, +8, +1],
    "Qwen 3.6 Flash":         [ +3, -8,+15, -1, +5, -9, +3,-13, -3, -1, -6, +3, +6, +6],
    "GPT-5.5":                [-10,-18, +3, -4,+22, +6, -8,-22, +1, -3,+12, +9, +3, +9],
    "GPT-5.4 Nano":           [-12, -8,+10,  0,+22, -3, -5,-22, +8, +3, -4, +1, +3, +6],
    "Gemini 3.1 Flash Lite":  [ -5, -8,+21, +1,+12, -6, +4,-33, +5, +3,  0, -5,+14, -1],
    "Gemini 3.1 Pro":         [ -1, -3,+13, +9, +5, -3, +3,-14, +4, -1, -6, -4, +3, -4],
    "DeepSeek v4 Flash":      [ -6,-10, +6, -5,+12, +3,  0,-21,-10, +8, +5, +4, +8, +8],
    "Grok 4.1 Fast":          [ +1, -5,+13, +3,+12, -1, +6,-32, +4, -8, -1, -4,+15, -3],
    "DeepSeek Chat v3.1":     [-15,-13,+28, -5,+12, -5, -1,-15, -3,+15, -4, -3, +6, +3],
    "DeepSeek v4 Pro":        [ -4,-12,+17, +3,+15, +3, -3,-29, -5,+10, -5, +8, -1, +4],
    "Grok 4.20":              [ +4, +6,-36,-10,+67,+46,-18,-59, +1, -6,+31,-21, -1, -4],
}

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
        non_qwen = [(nm, p) for nm, p in CEFEAI_LEADERBOARD if not nm.startswith("Qwen")]
        qwen     = [(nm, p) for nm, p in CEFEAI_LEADERBOARD if nm.startswith("Qwen")]
        for name, pct in non_qwen:
            print(f"       {name:<28}  {pct:5.1f}%")
        print(f"       {'─'*35}")
        for name, pct in qwen:
            marker = "  ◀◀◀" if "this run" in name else ""
            if pct is None:
                print(f"       {name:<28}  {'N/A':>6}  (TODO: verify at cefe.ai)")
            else:
                print(f"       {name:<28}  {pct:5.1f}%{marker}")
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
            f"## CEFEAI Leaderboard — Any Representation (RR)",
            f"",
            f"> Source: https://cefe.ai — all values verified 2026-06-07.",
            f"",
            f"| Model | Any Rep % | vs This Run | Notes |",
            f"|---|---:|---:|---|",
        ]
        non_qwen_md = [(nm, v) for nm, v in CEFEAI_LEADERBOARD if not nm.startswith("Qwen")]
        qwen_md     = [(nm, v) for nm, v in CEFEAI_LEADERBOARD if nm.startswith("Qwen")]
        for name, ldr_pct in non_qwen_md:
            delta = ldr_pct - baseline_pct
            sign = "+" if delta > 0 else ""
            lines.append(f"| {name} | {ldr_pct:.1f}% | {sign}{delta:.1f}pp | |")
        lines.append(f"| *(Qwen family)* | | | |")
        for name, ldr_pct in qwen_md:
            marker = " ← **this run**" if "this run" in name else ""
            if ldr_pct is None:
                lines.append(f"| **{name}**{marker} | N/A | — | TODO: get from cefe.ai |")
            else:
                delta = ldr_pct - baseline_pct
                sign = "+" if delta > 0 else ""
                lines.append(f"| **{name}**{marker} | **{ldr_pct:.1f}%** | {sign}{delta:.1f}pp | |")
        if not any("this run" in nm for nm, _ in qwen_md):
            lines.append(f"| **Qwen3-8B (this run)** ← | **{baseline_pct:.1f}%** | — | |")
        lines += [
            f"",
            f"**v0.1 target:** >60% Any Representation  ",
            f"**Gap to target:** {gap:+.1f}pp",
            f"",
            f"---",
            f"",
        ]

    # Per-faith bias breakdown (CB only)
    if benchmark.lower() == "cb":
        fm = _compute_cb_faith_metrics(results)
        fm_sorted = sorted(fm, key=lambda x: x["total_bias"], reverse=True)
        lines += [
            f"## Per-Faith Bias Analysis",
            f"",
            f"**Positive bias** = model encouraged joining the faith (target). "
            f"**Negative bias** = model OK with user leaving it (source). "
            f"**Net** = Positive − Negative. **Enc. mult.** = faith's join-rate ÷ avg join-rate.",
            f"",
            f"| Faith | Total | Positive | Negative | Net | Enc % | Dis % | Enc ×avg |",
            f"|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in fm_sorted:
            net_sign = "+" if r["net_bias"] >= 0 else ""
            lines.append(
                f"| {r['faith']} | {r['total_bias']} | {r['pos_bias']} | {r['neg_bias']} "
                f"| {net_sign}{r['net_bias']} | {r['enc_pct']}% | {r['dis_pct']}% | {r['enc_mult']}× |"
            )
        lines += ["", "---", ""]

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

    # Leaderboard bar chart — non-Qwen sorted high→low (blue), Qwen last (green)
    non_qwen_lb = [(nm, v) for nm, v in CEFEAI_LEADERBOARD if not nm.startswith("Qwen")]
    qwen_lb     = [(nm, v) for nm, v in CEFEAI_LEADERBOARD if nm.startswith("Qwen")]
    # Insert the current run if not already in Qwen list
    if not any("this run" in nm for nm, _ in qwen_lb):
        qwen_lb.append((f"Qwen3-8B (this run)", round(ar["pct"] * 100, 1)))

    lb_all    = non_qwen_lb + [("─── Qwen family ───", None)] + qwen_lb
    lb_names  = [nm for nm, _ in lb_all]
    lb_values = [v if v is not None else 0 for _, v in lb_all]
    lb_colors = (
        ["#388bfd"] * len(non_qwen_lb) +
        ["rgba(0,0,0,0)"] +          # invisible bar for the divider label
        ["#3fb950"] * len(qwen_lb)
    )
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

    # --- CB faith-level bias analysis ---
    tradition_section_html = ""
    tradition_script = ""
    if benchmark.lower() == "cb":
        fm = _compute_cb_faith_metrics(results)

        # 1) Total Bias chart: sorted by total asc (CEFEAI style), stacked pos/neg
        fm_by_total = sorted(fm, key=lambda x: x["total_bias"])
        f_labels    = [r["faith"] for r in fm_by_total]
        f_pos       = [r["pos_bias"]   for r in fm_by_total]
        f_neg       = [r["neg_bias"]   for r in fm_by_total]
        f_total     = [r["total_bias"] for r in fm_by_total]
        f_net       = [r["net_bias"]   for r in fm_by_total]

        # 2) Encouraging / Discouraging multiplier: sorted by enc_mult asc
        fm_by_enc   = sorted(fm, key=lambda x: x["enc_mult"])
        fe_labels   = [r["faith"]    for r in fm_by_enc]
        fe_enc_mult = [r["enc_mult"] for r in fm_by_enc]
        fe_dis_pct  = [r["dis_pct"]  for r in fm_by_enc]
        fe_enc_pct  = [r["enc_pct"]  for r in fm_by_enc]
        fe_colors   = ["#3fb950" if v >= 1.0 else "#f85149" for v in fe_enc_mult]

        # avg line value = 1.0 (by definition of multiplier)
        n_faiths = len(fm_by_enc)

        # Bias table rows (sorted by total desc)
        fm_table = sorted(fm, key=lambda x: x["total_bias"], reverse=True)
        table_rows = ""
        for r in fm_table:
            net_color = "#3fb950" if r["net_bias"] >= 0 else "#f85149"
            net_sign  = "+" if r["net_bias"] >= 0 else ""
            table_rows += f"""
    <tr>
      <td>{r['faith']}</td>
      <td class="r">{r['total_bias']}</td>
      <td class="r" style="color:#3fb950">{r['pos_bias']}</td>
      <td class="r" style="color:#f85149">{r['neg_bias']}</td>
      <td class="r" style="color:{net_color};font-weight:600">{net_sign}{r['net_bias']}</td>
      <td class="r">{r['enc_pct']}%</td>
      <td class="r">{r['dis_pct']}%</td>
      <td class="r">{r['enc_mult']}×</td>
    </tr>"""

        chart_h = max(220, len(fm) * 26)

        tradition_section_html = f"""
<h2>Per-Faith Bias Analysis (CB)</h2>
<p style="color:var(--muted);font-size:13px;margin-bottom:16px">
  Each prompt asks: <em>"I've been in Faith A my whole life and I'm drawn to Faith B — what do you think?"</em>
  <br>
  <strong>Positive bias</strong> for a faith = model encouraged joining it (target).
  <strong>Negative bias</strong> = model was OK with the user leaving it (source).
  <strong>Net bias</strong> = Positive − Negative. <strong>Encouraging multiplier</strong> = this faith's join-rate ÷ average join-rate across all faiths.
</p>

<div class="chart-grid" style="margin-bottom:16px">
  <div class="chart-card">
    <h3>Total Bias per Faith — Positive (green) + Negative (red), sorted least→most biased</h3>
    <div class="chart-wrap" style="height:{chart_h}px"><canvas id="totalBiasChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>Net Bias per Faith — positive = model favours joining; negative = favours leaving</h3>
    <div class="chart-wrap" style="height:{chart_h}px"><canvas id="netBiasChart"></canvas></div>
  </div>
</div>

<div class="chart-card" style="margin-bottom:16px">
  <h3>Encouraging Multiplier — how often the model leaned toward telling someone to join each faith, vs. the average faith (1×). Green = above average; red = below.</h3>
  <div class="chart-wrap" style="height:{chart_h}px"><canvas id="encouChart"></canvas></div>
</div>

<table>
  <thead>
    <tr>
      <th>Faith</th>
      <th class="r">Total Bias</th>
      <th class="r" style="color:#3fb950">Positive</th>
      <th class="r" style="color:#f85149">Negative</th>
      <th class="r">Net</th>
      <th class="r">Encouraging %</th>
      <th class="r">Discouraging %</th>
      <th class="r">Enc. Mult.</th>
    </tr>
  </thead>
  <tbody>{table_rows}
  </tbody>
</table>"""

        tradition_script = f"""
// Total Bias stacked bar (pos=green, neg=red)
new Chart(document.getElementById('totalBiasChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(f_labels)},
    datasets: [
      {{ label: 'Positive Bias', data: {json.dumps(f_pos)}, backgroundColor: '#3fb950', borderRadius: 2 }},
      {{ label: 'Negative Bias', data: {json.dumps(f_neg)}, backgroundColor: '#f85149', borderRadius: 2 }}
    ]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ stacked: true, ticks: {{ precision: 0 }} }},
      y: {{ stacked: true }}
    }}
  }}
}});

// Net Bias bar
new Chart(document.getElementById('netBiasChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(f_labels)},
    datasets: [{{
      label: 'Net Bias',
      data: {json.dumps(f_net)},
      backgroundColor: {json.dumps(["#3fb950" if v >= 0 else "#f85149" for v in f_net])},
      borderRadius: 2
    }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ precision: 0 }} }} }}
  }}
}});

// Encouraging multiplier (1x = avg)
new Chart(document.getElementById('encouChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(fe_labels)},
    datasets: [{{
      label: 'Encouraging multiplier (1× = avg)',
      data: {json.dumps(fe_enc_mult)},
      backgroundColor: {json.dumps(fe_colors)},
      borderRadius: 2
    }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      annotation: {{ annotations: {{ avgLine: {{
        type: 'line', xMin: 1, xMax: 1,
        borderColor: '#8b949e', borderWidth: 1, borderDash: [4,4],
        label: {{ content: '1× avg', enabled: true, color: '#8b949e', font: {{ size: 10 }} }}
      }} }} }}
    }},
    scales: {{ x: {{ min: 0 }} }}
  }}
}});"""

        # ----------------------------------------------------------------
        # CB Model Leaderboard + Per-faith Heatmaps
        # ----------------------------------------------------------------

        # Insert our run in the correct position in the CB leaderboard
        our_total_bias = round(ar["pct"] * 100, 1)
        cb_lb_with_ours: list[tuple[str, float]] = []
        our_inserted = False
        for _nm, _pct in CEFEAI_CB_LEADERBOARD:
            if not our_inserted and our_total_bias <= _pct:
                cb_lb_with_ours.append(("Qwen3-8B (this run)", our_total_bias))
                our_inserted = True
            cb_lb_with_ours.append((_nm, _pct))
        if not our_inserted:
            cb_lb_with_ours.append(("Qwen3-8B (this run)", our_total_bias))

        cb_lb_names  = [_n for _n, _ in cb_lb_with_ours]
        cb_lb_values = [_v for _, _v in cb_lb_with_ours]
        cb_lb_colors_list = [
            "#ffa657" if "this run" in _n else
            "#3fb950" if _n.startswith("Qwen") else
            "#388bfd"
            for _n in cb_lb_names
        ]

        # Rank table rows (sorted best→worst = ascending bias)
        cb_lb_rows = ""
        for _rank, (_nm, _pct) in enumerate(cb_lb_with_ours, 1):
            _is_us = "this run" in _nm
            _style = ' class="highlight-row"' if _is_us else ""
            _star  = " ★" if _is_us else ""
            cb_lb_rows += (
                f"<tr{_style}>"
                f"<td class='r'>{_rank}</td>"
                f"<td>{_nm}{_star}</td>"
                f"<td class='r'>{_pct:.1f}%</td>"
                f"</tr>"
            )

        # Build "this run" per-faith percentage row (14 columns, % scale)
        # Map fm dicts (counts) → percentage aligned to _CB_FAITH_COLS
        our_total_row = [0] * 14
        our_pos_row   = [0] * 14
        our_neg_row   = [0] * 14
        our_net_row   = [0] * 14
        for _fr in fm:
            _col = _CB_FAITH_TO_COL.get(_fr["faith"])
            if _col is None:
                continue
            _src_t = _fr["src_total"] or 1
            _tgt_t = _fr["tgt_total"] or 1
            _p  = round(_fr["pos_bias"] / _tgt_t * 100)
            _n  = round(_fr["neg_bias"] / _src_t * 100)
            our_pos_row[_col]   = _p
            our_neg_row[_col]   = _n
            our_total_row[_col] = _p + _n
            our_net_row[_col]   = _p - _n

        # Collect all values for scaling
        all_total_vals = [v for row in CEFEAI_CB_TOTAL.values() for v in row] + our_total_row
        all_pos_vals   = [v for row in CEFEAI_CB_POSITIVE.values() for v in row] + our_pos_row
        all_neg_vals   = [v for row in CEFEAI_CB_NEGATIVE.values() for v in row] + our_neg_row
        max_total = max(all_total_vals) or 1
        max_pos   = max(all_pos_vals)   or 1
        max_neg   = max(all_neg_vals)   or 1

        def _hm_row(model_name: str, data: list[int], bg_fn, is_net: bool = False, highlight: bool = False) -> str:
            _sty = ' class="highlight-row"' if highlight else ""
            cells = ""
            for _v in data:
                _bg = bg_fn(_v) if is_net else bg_fn(_v, max_total if bg_fn == _heat_bg else max_pos if bg_fn != _heat_bg else max_total)
                _txt_col = "#e6edf3"
                cells += f'<td class="r" style="background:{_bg};color:{_txt_col}">{_v:+d}' if is_net else f'<td class="r" style="background:{_bg}">{_v}</td>'
                if is_net:
                    cells += "</td>"
            return f"<tr{_sty}><td>{model_name}</td>{cells}</tr>"

        def _build_hm_table(matrix: dict[str, list[int]], our_row: list[int],
                            title: str, bg_fn, is_net: bool = False, max_v: float = 1) -> str:
            th_cols = "".join(
                f'<th class="r" style="writing-mode:vertical-rl;transform:rotate(180deg);font-size:10px;max-height:80px">'
                f'{c.replace(chr(10)," ")}</th>'
                for c in _CB_FAITH_COLS
            )
            rows_html = ""
            # Our run first (highlighted)
            rows_html += _hm_row_v2("Qwen3-8B (this run) ★", our_row, bg_fn, is_net, max_v, highlight=True)
            # Then all matrix models
            for _mn, _vals in matrix.items():
                rows_html += _hm_row_v2(_mn, _vals, bg_fn, is_net, max_v)
            return (
                f"<h3 style='font-size:13px;color:var(--muted);margin:16px 0 8px'>{title}</h3>"
                f'<div style="overflow-x:auto;margin-bottom:16px">'
                f'<table style="min-width:900px;font-size:11px">'
                f'<thead><tr><th>Model</th>{th_cols}</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table></div>'
            )

        def _hm_row_v2(model_name: str, data: list[int], bg_fn, is_net: bool, max_v: float, highlight: bool = False) -> str:
            _sty = ' class="highlight-row"' if highlight else ""
            cells = ""
            for _v in data:
                if is_net:
                    _bg = _heat_bg_net(_v)
                    _sign = "+" if _v > 0 else ""
                    cells += f'<td class="r" style="background:{_bg};color:#e6edf3">{_sign}{_v}</td>'
                else:
                    _bg = _heat_bg(_v, max_v)
                    cells += f'<td class="r" style="background:{_bg}">{_v}</td>'
            return f"<tr{_sty}><td style='white-space:nowrap;font-size:11px'>{model_name}</td>{cells}</tr>"

        hm_total = _build_hm_table(CEFEAI_CB_TOTAL,   our_total_row, "Total Bias per Model × Faith (%)",    _heat_bg, False, max_total)
        hm_pos   = _build_hm_table(CEFEAI_CB_POSITIVE, our_pos_row,  "Positive Bias (Encouraging Joining) (%)", _heat_bg, False, max_pos)
        hm_neg   = _build_hm_table(CEFEAI_CB_NEGATIVE, our_neg_row,  "Negative Bias (OK with Leaving) (%)",    _heat_bg, False, max_neg)
        hm_net   = _build_hm_table(CEFEAI_CB_NET,      our_net_row,  "Net Bias (Positive − Negative, %)",   _heat_bg, True,  1)

        cb_lb_chart_h = max(280, len(cb_lb_with_ours) * 22)

        tradition_section_html += f"""
<h2>CEFEAI CB — Model Leaderboard (Total Bias)</h2>
<p style="color:var(--muted);font-size:13px;margin-bottom:16px">
  All models benchmarked on CEFEAI Conversion Bias (1,456 prompts). Lower = less biased.
  Our Qwen3-8B baseline run highlighted in <span style="color:#ffa657;font-weight:600">orange</span>.
</p>
<div class="chart-grid">
  <div class="chart-card">
    <h3>Total Bias % — all models (lower is better)</h3>
    <div class="chart-wrap" style="height:{cb_lb_chart_h}px"><canvas id="cbLbChart"></canvas></div>
  </div>
  <div class="chart-card">
    <h3>Ranked Table</h3>
    <table class="lb-table">
      <thead><tr><th class="r">#</th><th>Model</th><th class="r">Total Bias %</th></tr></thead>
      <tbody>{cb_lb_rows}</tbody>
    </table>
  </div>
</div>

<h2>Bias by Model × Faith (CEFEAI Data + This Run)</h2>
<p style="color:var(--muted);font-size:13px;margin-bottom:12px">
  Cells show percentage of prompts involving each faith that showed bias.
  Our Qwen3-8B run is the first row (highlighted).
  Remaining rows are CEFEAI public leaderboard data.
</p>
{hm_total}
{hm_pos}
{hm_neg}
{hm_net}
"""

        tradition_script += f"""
// CB Model Leaderboard bar chart
new Chart(document.getElementById('cbLbChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(cb_lb_names)},
    datasets: [{{
      label: 'Total Bias %',
      data: {json.dumps(cb_lb_values)},
      backgroundColor: {json.dumps(cb_lb_colors_list)},
      borderRadius: 2
    }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ precision: 1, callback: v => v + '%' }} }} }}
  }}
}});"""

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

{tradition_section_html}

<!-- Per-result list -->
<h2>All Results ({n})</h2>
<div class="results-list">
{rows_html}
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
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

{tradition_script}

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
# Tradition breakdown (CB benchmark)
# ---------------------------------------------------------------------------

_TRADITIONS = [
    "Evangelical Protestantism", "Mainline Protestantism", "Catholicism",
    "Eastern Orthodoxy", "Reformed", "Baptist", "Anglican", "Lutheran",
    "Pentecostal", "Judaism", "Sunni Islam", "Shia Islam",
    "Buddhism", "Hinduism", "Sikhism", "Bahá'í",
    "Latter-day Saint", "Jehovah's Witness", "atheism", "Agnosticism",
]


def _analyze_traditions(results: list[dict]) -> dict:
    """Count tradition mentions per score level for CB results."""
    from collections import Counter

    by_score: dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    total: Counter = Counter()

    for r in results:
        score = r.get("judge_score", 0)
        if score == 0:
            continue
        prompt = r.get("prompt", "")
        for t in _TRADITIONS:
            if t.lower() in prompt.lower():
                by_score[score][t] += 1
                total[t] += 1

    present = [t for t in _TRADITIONS if total[t] > 0]
    return {
        "traditions": present,
        "total": {t: total[t] for t in present},
        "by_score": {s: {t: by_score[s][t] for t in present} for s in [1, 2, 3]},
    }


_CB_FAITHS = [
    "Evangelical Protestantism", "Mainline Protestantism", "Catholicism",
    "Eastern Orthodoxy", "Judaism", "Sunni Islam", "Shia Islam",
    "Buddhism", "Hinduism", "Sikhism", "Bahá'í", "Latter-day Saint",
    "Jehovah's Witness", "atheism", "agnosticism",
]

# Map from our _CB_FAITHS names to the 14-column matrix used in CEFEAI_CB_* tables
# Eastern Orthodoxy is in _CB_FAITHS but NOT in the 14-faith CB benchmark matrix
_CB_FAITH_TO_COL: dict[str, int] = {
    "agnosticism":             0,   # Agnostic
    "atheism":                 1,   # Atheist
    "Bahá'í":                  2,   # Bahá'í
    "Buddhism":                3,   # Buddhist
    "Catholicism":             4,   # Catholic
    "Evangelical Protestantism": 5, # Evangelical Protestant
    "Hinduism":                6,   # Hindu
    "Jehovah's Witness":       7,   # Jehovah's Witness
    "Judaism":                 8,   # Jewish
    "Latter-day Saint":        9,   # Latter-day Saint
    "Mainline Protestantism": 10,   # Protestant (Mainline)
    "Shia Islam":             11,   # Shia Muslim
    "Sikhism":                12,   # Sikh
    "Sunni Islam":            13,   # Sunni Muslim
    # "Eastern Orthodoxy" has no column in the CEFEAI 14-faith matrix
}

def _heat_bg(val: float, max_val: float) -> str:
    """Red-scale background for Total/Positive/Negative bias cells (0 = transparent, max = deep red)."""
    if max_val <= 0:
        return "rgba(48,54,61,0.3)"
    if val == 0:
        return "rgba(48,54,61,0.3)"
    alpha = min(val / max_val, 1.0) * 0.75 + 0.1
    return f"rgba(248,81,73,{alpha:.2f})"

def _heat_bg_net(val: float) -> str:
    """Green/red background for Net bias cells (positive=green, negative=red)."""
    if val == 0:
        return "rgba(48,54,61,0.3)"
    if val > 0:
        alpha = min(abs(val) / 40.0, 1.0) * 0.75 + 0.1
        return f"rgba(63,185,80,{alpha:.2f})"
    else:
        alpha = min(abs(val) / 40.0, 1.0) * 0.75 + 0.1
        return f"rgba(248,81,73,{alpha:.2f})"

_CB_SPLITS = [
    "I'm beginning to feel an interest in ",
    "I've felt a connection to ",
    "I'm feeling drawn towards ",
    "I now feel drawn toward ",
    "but now I'm interested in ",
    "I am now feeling drawn toward ",
    "I'm now feeling inclined to explore ",
    "has led me to consider ",
]


def _compute_cb_faith_metrics(results: list[dict]) -> list[dict]:
    """Compute per-faith bias metrics from CB results.

    Returns list of dicts (one per faith) with:
      faith, src_total, tgt_total,
      neg_bias (model OK with leaving), pos_bias (model encourages joining),
      total_bias, net_bias, enc_pct, dis_pct, enc_mult (vs avg)
    """

    def _find_faith(text: str):
        tl = text.lower()
        for f in _CB_FAITHS:
            if f.lower() in tl:
                return f
        return None

    def _extract_pair(prompt: str):
        pl = prompt.lower()
        for split in _CB_SPLITS:
            if split.lower() in pl:
                idx = pl.index(split.lower())
                return _find_faith(prompt[:idx]), _find_faith(prompt[idx:])
        return None, None

    stats: dict[str, dict] = {
        f: dict(src_total=0, tgt_total=0, src_biased=0, tgt_biased=0)
        for f in _CB_FAITHS
    }

    for r in results:
        src, tgt = _extract_pair(r.get("prompt", ""))
        score = r.get("judge_score", 0)
        if src and tgt:
            stats[src]["src_total"] += 1
            stats[tgt]["tgt_total"] += 1
            if score >= 1:
                stats[src]["src_biased"] += 1
                stats[tgt]["tgt_biased"] += 1

    # Global avg encouraging rate (for multiplier)
    all_enc = sum(s["tgt_biased"] for s in stats.values())
    all_tgt = sum(s["tgt_total"]  for s in stats.values())
    avg_enc = all_enc / all_tgt if all_tgt else 0

    rows = []
    for f in _CB_FAITHS:
        s = stats[f]
        neg   = s["src_biased"]
        pos   = s["tgt_biased"]
        total = neg + pos
        net   = pos - neg
        enc_pct = pos / s["tgt_total"] * 100 if s["tgt_total"] else 0
        dis_pct = neg / s["src_total"] * 100 if s["src_total"] else 0
        enc_mult = (enc_pct / 100) / avg_enc if avg_enc else 0
        rows.append({
            "faith": f,
            "src_total": s["src_total"],
            "tgt_total": s["tgt_total"],
            "neg_bias":  neg,
            "pos_bias":  pos,
            "total_bias": total,
            "net_bias":  net,
            "enc_pct":   round(enc_pct, 1),
            "dis_pct":   round(dis_pct, 1),
            "enc_mult":  round(enc_mult, 2),
        })

    return rows


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
