"""One-shot diagnostic: show what the CB judge actually returned for parse-error records."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

path = "results/baseline_qwen_qwen3_8b_noprompt_CB.jsonl"
errors = []
ok = []
with open(path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("judge_score") is None:
            errors.append(r)
        else:
            ok.append(r)

print(f"Total: {len(errors)+len(ok)}  scored: {len(ok)}  parse_errors: {len(errors)}")
print()
print("=== FIRST 10 PARSE-ERROR records ===")
for r in errors[:10]:
    pid = r.get("prompt_id", "?")
    # Parse-error records carry the truncated raw judge text in judge_rationale
    # ("[parse-error] <raw...>"); the old judge_raw_response field was never written.
    raw = r.get("judge_rationale", "<none>")
    rationale = r.get("judge_rationale", "<none>")
    print(f"  prompt_id       : {pid}")
    print(f"  judge_raw_resp  : {repr(str(raw)[:400])}")
    print(f"  judge_rationale : {repr(str(rationale)[:200])}")
    print()

# Count distinct patterns
patterns = {}
import re
for r in errors:
    raw = str(r.get("judge_raw_response") or r.get("judge_rationale") or "")
    # try to find Rating: anywhere
    m = re.search(r"Rating:\s*\S+", raw, re.IGNORECASE)
    key = m.group(0)[:30] if m else raw[:40] if raw else "<empty>"
    patterns[key] = patterns.get(key, 0) + 1

print(f"=== TOP 20 PATTERNS IN FAILED RECORDS ===")
for k, v in sorted(patterns.items(), key=lambda x: -x[1])[:20]:
    print(f"  {v:4d}x  {repr(k)}")
