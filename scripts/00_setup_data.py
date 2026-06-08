"""
00_setup_data.py — One-time data setup for OpenScriptura.

What it does:
  1. Downloads CEFEAI benchmark files from GitHub (renames 'question' → 'prompt').
  2. Copies Spurgeon sermons from pastor-ai into data/sources/spurgeon/.
  3. Copies Reformed Monergismo ebooks (PDF) into data/sources/monergismo/.
     Excludes John Wesley (Arminian — not Reformed).
  4. Reports which Tier C confessional documents are present vs. missing.
  5. Writes data/sources/manifest.json with SHA-256 per file.

Usage:
  python scripts/00_setup_data.py
  python scripts/00_setup_data.py --dry-run
  python scripts/00_setup_data.py --pastor-ai C:\\path\\to\\pastor-ai
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.logger import get_logger

log = get_logger("00_setup_data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CEFEAI_SOURCES = {
    "rr_150.jsonl": (
        "https://raw.githubusercontent.com/CEFEAI/allfaith-religious-representation/main/questions.jsonl",
        150,
    ),
    "cb_1456.jsonl": (
        "https://raw.githubusercontent.com/CEFEAI/allfaith-conversion-bias/main/questions.jsonl",
        1456,
    ),
}

# Confessional documents required for Tier C — must be placed in data/sources/confessions/
# by the user before running 01_build_tier_c.py.
TIER_C_REQUIRED = {
    "wcf_1647.txt": "Westminster Confession of Faith (1647)",
    "heidelberg_catechism.txt": "Heidelberg Catechism",
    "westminster_shorter_catechism.txt": "Westminster Shorter Catechism",
    "westminster_larger_catechism.txt": "Westminster Larger Catechism",
    "canons_of_dort.txt": "Canons of Dort",
    "lcf_1689.txt": "London Baptist Confession of Faith (1689)",
}

# Authors to EXCLUDE from Reformed corpus (non-Reformed or off-topic)
EXCLUDED_AUTHORS = {
    "John_Wesley",        # Arminian — theological tradition incompatible with Reformed v0.1
    "Karl_Barth",         # Neo-orthodox — view of election/Scripture diverges from WCF
    "F_A_Hayek",          # Economist — not a theologian, off-topic
    "Hermas",             # Early Church (Shepherd of Hermas) — pre-Reformation, not confessionally Reformed
    "Inacio_de_Antioquia",# Ignatius of Antioch — Church Father, patristic, not confessionally Reformed
    # Pastoral council may restore any of these after review
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cefeai(dest_dir: Path, dry_run: bool) -> dict:
    """Download CEFEAI benchmark files, renaming 'question' → 'prompt'."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for filename, (url, expected_count) in CEFEAI_SOURCES.items():
        dest = dest_dir / filename
        if dest.exists():
            log.info(f"CEFEAI {filename} already exists — skipping download")
            results[filename] = {"status": "exists", "path": str(dest), "count": sum(1 for _ in open(dest))}
            continue

        log.info(f"Downloading {filename} from {url}")
        if dry_run:
            log.info(f"  [dry-run] would write {expected_count} lines to {dest}")
            results[filename] = {"status": "dry-run", "expected_count": expected_count}
            continue

        with urlopen(url) as resp:
            raw_lines = resp.read().decode("utf-8").strip().splitlines()

        # Rename 'question' → 'prompt' to match 00_cefeai_baseline.py expectations
        converted = []
        for line in raw_lines:
            record = json.loads(line)
            if "question" in record and "prompt" not in record:
                record["prompt"] = record.pop("question")
            converted.append(json.dumps(record, ensure_ascii=False))

        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(converted) + "\n")

        actual = len(converted)
        if actual != expected_count:
            log.warning(f"  Expected {expected_count} lines, got {actual}")
        else:
            log.info(f"  ✓ {actual} prompts written to {dest}")

        results[filename] = {"status": "downloaded", "path": str(dest), "count": actual}

    return results


def copy_spurgeon(src_dir: Path, dest_dir: Path, dry_run: bool) -> dict:
    """Copy Spurgeon PT-BR markdown files and their avaliacao JSONs."""
    if not src_dir.exists():
        log.warning(f"Spurgeon source not found: {src_dir}")
        return {"status": "missing", "source": str(src_dir)}

    dest_dir.mkdir(parents=True, exist_ok=True)
    ptbr_files = list(src_dir.glob("*.pt-br.md"))
    avaliacao_files = list(src_dir.glob("*_avaliacao.json"))

    log.info(f"Spurgeon: {len(ptbr_files)} PT-BR sermons, {len(avaliacao_files)} quality scores")

    if dry_run:
        log.info(f"  [dry-run] would copy to {dest_dir}")
        return {"status": "dry-run", "ptbr_count": len(ptbr_files), "evaluated": len(avaliacao_files)}

    copied = 0
    for f in ptbr_files + avaliacao_files:
        dst = dest_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    log.info(f"  ✓ Copied {copied} files to {dest_dir}")
    return {"status": "copied", "ptbr_count": len(ptbr_files), "evaluated": len(avaliacao_files), "copied": copied}


def copy_monergismo(src_dir: Path, dest_dir: Path, dry_run: bool) -> dict:
    """Copy Reformed Monergismo ebooks (PDFs), excluding non-Reformed authors."""
    if not src_dir.exists():
        log.warning(f"Monergismo source not found: {src_dir}")
        return {"status": "missing", "source": str(src_dir)}

    dest_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for author_dir in sorted(src_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        if author in EXCLUDED_AUTHORS:
            log.info(f"  Skipping {author} (non-Reformed — excluded)")
            results[author] = {"status": "excluded"}
            continue

        pdfs = list(author_dir.glob("*.pdf"))
        if not pdfs:
            continue

        author_dest = dest_dir / author
        if not dry_run:
            author_dest.mkdir(parents=True, exist_ok=True)

        copied = 0
        for pdf in pdfs:
            dst = author_dest / pdf.name
            if not dry_run and not dst.exists():
                shutil.copy2(pdf, dst)
                copied += 1

        log.info(f"  {'[dry-run] ' if dry_run else ''}{'would copy' if dry_run else 'Copied'} {len(pdfs)} PDF(s) from {author}")
        results[author] = {"status": "dry-run" if dry_run else "copied", "pdfs": len(pdfs)}

    return results


def audit_tier_c(confessions_dir: Path) -> dict:
    """Report which Tier C confessional documents are present vs. missing."""
    log.info("\n=== Tier C Confessional Documents Audit ===")
    status = {}
    all_present = True

    for filename, description in TIER_C_REQUIRED.items():
        path = confessions_dir / filename
        if path.exists():
            log.info(f"  ✓ {description} ({filename})")
            status[filename] = {"description": description, "present": True, "path": str(path)}
        else:
            log.warning(f"  ✗ MISSING: {description} — expected at {path}")
            status[filename] = {"description": description, "present": False}
            all_present = False

    if all_present:
        log.info("  All Tier C sources present.")
    else:
        log.warning(
            "\n  Missing documents must be placed in data/sources/confessions/ as plain UTF-8 text.\n"
            "  Suggested sources (all public domain):\n"
            "    WCF: https://www.westminsterconfession.org\n"
            "    Heidelberg: https://www.heidelberg-catechism.com\n"
            "    WSC/WLC: https://www.pcaac.org/resources/westminster-standards/\n"
            "    Canons of Dort: https://www.urcna.org/sysmod/pbp/pbpid=DORT\n"
            "    LCF 1689: https://www.1689.com"
        )

    return {"all_present": all_present, "files": status}


def write_manifest(data_dir: Path, sections: dict, dry_run: bool):
    """Write data/sources/manifest.json with SHA-256 per source file."""
    manifest = {"sections": sections, "sha256": {}}

    for f in sorted(data_dir.rglob("*")):
        if f.is_file() and f.suffix in {".jsonl", ".md", ".pdf", ".txt", ".json"}:
            rel = str(f.relative_to(data_dir))
            if not dry_run:
                manifest["sha256"][rel] = sha256_file(f)

    out = data_dir / "sources" / "manifest.json"
    if dry_run:
        log.info(f"[dry-run] would write manifest to {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"\n✓ Manifest written to {out} ({len(manifest['sha256'])} files indexed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Set up OpenScriptura data directories.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be done without copying/downloading")
    parser.add_argument(
        "--pastor-ai",
        default=r"C:\tmp\pastor-ai",
        help="Path to pastor-ai repository root (default: C:\\tmp\\pastor-ai)",
    )
    args = parser.parse_args()

    pastor_ai = Path(args.pastor_ai)
    data_dir = PROJECT_ROOT / "data"

    log.info(f"{'[DRY RUN] ' if args.dry_run else ''}OpenScriptura data setup")
    log.info(f"  Project root : {PROJECT_ROOT}")
    log.info(f"  pastor-ai    : {pastor_ai}")
    log.info(f"  Data dir     : {data_dir}")

    # 1. CEFEAI benchmark
    log.info("\n=== CEFEAI Benchmark ===")
    cefeai_results = download_cefeai(data_dir / "cefeai", args.dry_run)

    # 2. Spurgeon sermons
    log.info("\n=== Spurgeon Sermons (Tier B source) ===")
    spurgeon_src = pastor_ai / "datasets" / "sermons" / "staging" / "spurgeon_selected"
    spurgeon_results = copy_spurgeon(spurgeon_src, data_dir / "sources" / "spurgeon", args.dry_run)

    # 3. Monergismo ebooks
    log.info("\n=== Monergismo Reformed Ebooks (Tier B/C source) ===")
    mono_src = pastor_ai / "datasets" / "sermons" / "Monergismo_Ebooks"
    mono_results = copy_monergismo(mono_src, data_dir / "sources" / "monergismo", args.dry_run)

    # 4. Tier C confessional audit
    confessions_dir = data_dir / "sources" / "confessions"
    confessions_dir.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    tier_c_results = audit_tier_c(confessions_dir)

    # 5. Manifest
    log.info("\n=== Writing Source Manifest ===")
    write_manifest(data_dir, {
        "cefeai": cefeai_results,
        "spurgeon": spurgeon_results,
        "monergismo": mono_results,
        "tier_c_confessions": tier_c_results,
    }, args.dry_run)

    # Summary
    log.info("\n=== Setup Summary ===")
    missing = [desc for fn, info in tier_c_results["files"].items() if not info["present"]
               for desc in [info["description"]]]
    if missing:
        log.warning(f"Action required — {len(missing)} Tier C document(s) missing:")
        for m in missing:
            log.warning(f"  • {m}")
        log.warning("  Place them as UTF-8 .txt files in data/sources/confessions/ then re-run.")
    else:
        log.info("All sources present. Ready to run 01_build_tier_c.py.")

    if not args.dry_run:
        log.info("\nNext step: python scripts/00_cefeai_baseline.py --dry-run")


if __name__ == "__main__":
    main()
