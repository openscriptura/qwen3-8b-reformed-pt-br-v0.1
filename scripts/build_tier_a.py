"""build_tier_a.py — turn pastoral-reviewed JSON (from review_*.html) into tier_a.jsonl.

Reads data/tier_a/tier_a_reviewed_*.json (the human review export), keeps items whose
status is ok/edit (drops 'bad'/'pending'), and emits canonical training records
identical in shape to Tier B/C (same content_hash, same fields). Re-runnable.

Usage:
  python scripts/build_tier_a.py
  python scripts/build_tier_a.py --dry-run
"""
import argparse, glob, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.hash import content_hash

TIER_A_DIR = PROJECT_ROOT / "data" / "tier_a"
OUT = TIER_A_DIR / "tier_a.jsonl"
CAT_LABEL = {"abst": "abstenção", "dist": "distintivo", "fato": "doutrina"}


def build_record(system: str, item: dict) -> dict:
    rec = {
        "id": None,
        "version": "1.0",
        "tradition": "reformed",
        "lang": "pt-BR",
        "tier": "A",
        "source": f"tier_a_v0_1_1_{item['id']}",
        "source_label": f"Tier A — {CAT_LABEL.get(item.get('category'), item.get('category',''))} ({item['id']})",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": item["user"].strip()},
            {"role": "assistant", "content": item["assistant"].strip()},
        ],
        "confessional_refs": item.get("confessional_refs", []),
        "reviewed_by": "pastoral_review",   # Tier A = manual pastoral review
        "quality_score": 100,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rec["sha256"] = content_hash(rec)
    rec["id"] = f"openscriptura-reformed-ptbr-a-{rec['sha256'][:12]}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    review_files = sorted(glob.glob(str(TIER_A_DIR / "tier_a_reviewed_*.json")))
    if not review_files:
        sys.exit(f"No review files found in {TIER_A_DIR}")

    records, kept, dropped, seen = [], 0, 0, set()
    for rf in review_files:
        data = json.loads(Path(rf).read_text(encoding="utf-8"))
        system = data["system_prompt"]
        for it in data["items"]:
            if it.get("status") not in ("ok", "edit"):
                dropped += 1
                continue
            rec = build_record(system, it)
            if rec["sha256"] in seen:      # dedup across reviewer files
                continue
            seen.add(rec["sha256"])
            records.append(rec)
            kept += 1
        print(f"  {Path(rf).name}: {len(data['items'])} items")

    print(f"Kept {kept} (ok/edit) · dropped {dropped} (bad/pending) · unique {len(records)}")
    by_cat = {}
    for r in records:
        c = r["source"].split("_")[-1][:4]
        by_cat[c] = by_cat.get(c, 0) + 1
    print(f"By id-prefix: {by_cat}")

    if args.dry_run:
        print("[DRY-RUN] not written. Sample record:")
        print(json.dumps(records[0], ensure_ascii=False, indent=1)[:700])
        return

    TIER_A_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"WROTE {OUT}  ({len(records)} records)")
    print("Next: python scripts/merge_dataset.py  (TIER_FILES already maps A -> data/tier_a/tier_a.jsonl)")


if __name__ == "__main__":
    main()
