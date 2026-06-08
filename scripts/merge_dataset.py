"""
merge_dataset.py — Merge Tier C + Tier B into train/eval splits.

Pipeline:
  1. Load all available tiers (C, B, A if present).
  2. Shuffle with seed=42 for reproducibility.
  3. Stratified 95/5 train/eval split (stratified by tier so each tier
     contributes proportionally to both splits).
  4. Write data/merged/train.jsonl + data/merged/eval.jsonl.
  5. Write data/merged/manifest.json with SHA-256 hashes + counts.

Usage:
  python scripts/merge_dataset.py
  python scripts/merge_dataset.py --dry-run
  python scripts/merge_dataset.py --eval-split 0.10  # override 5% default

Output:
  data/merged/train.jsonl
  data/merged/eval.jsonl
  data/merged/manifest.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import get_logger

log = get_logger("merge_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_FILES = {
    "C": PROJECT_ROOT / "data" / "tier_c" / "tier_c.jsonl",
    "B": PROJECT_ROOT / "data" / "tier_b" / "tier_b.jsonl",
    "A": PROJECT_ROOT / "data" / "tier_a" / "tier_a.jsonl",
}
OUTPUT_DIR  = PROJECT_ROOT / "data" / "merged"
SEED        = 42
DEFAULT_EVAL_SPLIT = 0.05   # 5% eval, 95% train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tier(path: Path, tier_label: str) -> list[dict]:
    if not path.exists():
        log.info("  Tier %s: not found at %s — skipping", tier_label, path)
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("  Tier %s line %d: JSON error — %s", tier_label, i, exc)
    log.info("  Tier %s: %d records from %s", tier_label, len(records), path.name)
    return records


def _stratified_split(
    records: list[dict],
    eval_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split records into train/eval, stratified by tier.

    Each tier contributes `eval_frac` of its records to the eval set
    (minimum 1 if the tier has any records), rest to train.
    """
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_tier[rec.get("tier", "?")].append(rec)

    # Fixed per-tier seed offsets — never use hash() which varies across
    # Python processes (PYTHONHASHSEED randomisation since Python 3.3).
    _TIER_OFFSET = {"A": 1, "B": 2, "C": 3}

    train_all, eval_all = [], []
    for tier, recs in sorted(by_tier.items()):
        tier_seed = seed + _TIER_OFFSET.get(tier, sum(ord(c) for c in tier))
        rng = random.Random(tier_seed)
        shuffled = recs[:]
        rng.shuffle(shuffled)
        n_eval = max(1, round(len(shuffled) * eval_frac))
        eval_all.extend(shuffled[:n_eval])
        train_all.extend(shuffled[n_eval:])
        log.info(
            "  Tier %s: %d train + %d eval  (%.1f%% eval)",
            tier, len(shuffled) - n_eval, n_eval, 100 * n_eval / len(shuffled),
        )

    # Final global shuffle so tiers are interleaved (better training dynamics)
    rng_global = random.Random(seed)
    rng_global.shuffle(train_all)
    rng_global.shuffle(eval_all)
    return train_all, eval_all


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_manifest(
    output_dir: Path,
    train_path: Path,
    eval_path: Path,
    train_records: list[dict],
    eval_records: list[dict],
    eval_frac: float,
) -> None:
    by_tier_train: dict[str, int] = defaultdict(int)
    by_tier_eval:  dict[str, int] = defaultdict(int)
    for r in train_records:
        by_tier_train[r.get("tier", "?")] += 1
    for r in eval_records:
        by_tier_eval[r.get("tier", "?")] += 1

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "eval_split": eval_frac,
        "total_records": len(train_records) + len(eval_records),
        "train": {
            "file": train_path.name,
            "count": len(train_records),
            "sha256": _file_sha256(train_path),
            "by_tier": dict(sorted(by_tier_train.items())),
        },
        "eval": {
            "file": eval_path.name,
            "count": len(eval_records),
            "sha256": _file_sha256(eval_path),
            "by_tier": dict(sorted(by_tier_eval.items())),
        },
    }
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Manifest written to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(eval_split: float, dry_run: bool) -> None:
    log.info("Loading tiers...")
    all_records: list[dict] = []
    for tier_label, path in TIER_FILES.items():
        all_records.extend(_load_tier(path, tier_label))

    if not all_records:
        log.error("No records loaded — nothing to merge. Exiting.")
        sys.exit(1)

    log.info("Total records loaded: %d", len(all_records))

    # Dedup by sha256 across tiers (safety net)
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in all_records:
        sha = rec.get("sha256", "")
        if sha and sha in seen:
            continue
        seen.add(sha)
        deduped.append(rec)
    if len(deduped) < len(all_records):
        log.warning("  Cross-tier dedup removed %d duplicates", len(all_records) - len(deduped))
    all_records = deduped

    log.info("Splitting (eval=%.0f%%, seed=%d)...", eval_split * 100, SEED)
    train_records, eval_records = _stratified_split(all_records, eval_split, SEED)

    log.info("")
    log.info("=" * 50)
    log.info("  Total   : %d records", len(all_records))
    log.info("  Train   : %d  (%.1f%%)", len(train_records), 100 * len(train_records) / len(all_records))
    log.info("  Eval    : %d  (%.1f%%)", len(eval_records),  100 * len(eval_records)  / len(all_records))
    log.info("=" * 50)

    if dry_run:
        log.info("[DRY-RUN] Would write:")
        log.info("  %s  (%d records)", OUTPUT_DIR / "train.jsonl", len(train_records))
        log.info("  %s  (%d records)", OUTPUT_DIR / "eval.jsonl",  len(eval_records))
        log.info("[DRY-RUN] No files written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path  = OUTPUT_DIR / "eval.jsonl"

    _write_jsonl(train_path, train_records)
    log.info("  Written: %s", train_path)
    _write_jsonl(eval_path, eval_records)
    log.info("  Written: %s", eval_path)

    _write_manifest(OUTPUT_DIR, train_path, eval_path, train_records, eval_records, eval_split)
    log.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge OpenScriptura tiers into train/eval splits.")
    parser.add_argument("--dry-run",    action="store_true", help="Validate inputs without writing files")
    parser.add_argument("--eval-split", type=float, default=DEFAULT_EVAL_SPLIT,
                        help=f"Fraction of records for eval set (default: {DEFAULT_EVAL_SPLIT})")
    args = parser.parse_args()

    if not (0.01 <= args.eval_split <= 0.30):
        print("ERROR: --eval-split must be between 0.01 and 0.30", file=sys.stderr)
        sys.exit(1)

    run(eval_split=args.eval_split, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
