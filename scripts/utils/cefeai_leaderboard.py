"""cefeai_leaderboard.py — CEFE.AI PUBLIC leaderboard reference data (verbatim).

Curated from the public CEFE.AI leaderboard (https://cefe.ai), captured 2026-06-07:
  - Religious Representation (RR): Any Representation %, 27 models.
  - Conversion Bias (CB): Total Bias %, 20 models, + per-faith Total/Positive/
    Negative/Net bias matrices (14 faiths).

This is REFERENCE display data only — it is never used in scoring or aggregation
and does not touch the CEFE.AI comparability path. It is rendered as a context
section in the HTML report so our run can be read against the public leaderboard.
Update these tables when CEFE.AI publishes new leaderboard values.
"""

# Faith columns for the CB per-faith matrices below.

_CB_FAITH_COLS = [
    "Agnostic","Atheist","Bahá'í","Buddhist","Catholic",
    "Evangelical\nProtestant","Hindu","Jehovah's\nWitness","Jewish",
    "Latter-day\nSaint","Protestant\n(Mainline)","Shia\nMuslim","Sikh","Sunni\nMuslim",
]

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
    # (the live run is inserted by report._cefeai_reference from the official
    # summary; the old hardcoded 4.7 "this run" row — an INVALID home-grown-rubric
    # number — was removed.)
    ("Qwen 3.6 Flash",          6.0),
    ("Qwen 3.6 Max",            5.3),
]

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
