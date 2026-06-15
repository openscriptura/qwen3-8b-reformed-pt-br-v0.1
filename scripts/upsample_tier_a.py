"""upsample_tier_a.py — duplicate Tier A records in train.jsonl to strengthen their signal.

v0.1.1 first retrain: Tier A was ~1.8% of train and the targeted fixes (TULIP, abstention,
over-accommodation) did not take. This upsamples Tier A in-place so the curated corrections
get enough exposure. Deterministic shuffle (seed) — no Date/random surprises.

Usage:
  python scripts/upsample_tier_a.py --factor 7
"""
import argparse, json, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN = PROJECT_ROOT / "data" / "merged" / "train.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=7, help="total copies of each Tier A record")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    tier_a = [r for r in rows if r.get("tier") == "A"]
    others = [r for r in rows if r.get("tier") != "A"]

    upsampled = others + tier_a * args.factor   # tier_a appears `factor` times total
    random.Random(args.seed).shuffle(upsampled)

    with open(TRAIN, "w", encoding="utf-8") as f:
        for r in upsampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pct = 100 * len(tier_a) * args.factor / len(upsampled)
    print(f"Tier A unique: {len(tier_a)} · factor {args.factor} -> {len(tier_a)*args.factor} copies")
    print(f"Train: {len(rows)} -> {len(upsampled)} records  (Tier A now {pct:.1f}%)")
    print(f"WROTE {TRAIN}")


if __name__ == "__main__":
    main()
