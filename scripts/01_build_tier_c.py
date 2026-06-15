"""
01_build_tier_c.py — Build Tier C training corpus from confessional documents.

Sources (all PT-BR, public domain):
  Catechisms (natural Q&A):
    - Westminster Shorter Catechism (WSC)
    - Westminster Larger Catechism (WLC)
    - Heidelberg Catechism

  Prose confessions (article-per-record, template question):
    - Westminster Confession of Faith 1647 (WCF)
    - Canons of Dort (Dort)
    - London Baptist Confession of Faith 1689 (LCF)

Output:
  data/tier_c/tier_c.jsonl  — training records in chat schema
  data/tier_c/manifest.json — counts, SHA-256, source breakdown

Usage:
  python scripts/01_build_tier_c.py
  python scripts/01_build_tier_c.py --dry-run
  python scripts/01_build_tier_c.py --source wsc           # single source
  python scripts/01_build_tier_c.py --source wsc,wlc,heidelberg
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.hash import content_hash
from utils.logger import get_logger
from utils.progress import ProgressBar

log = get_logger("01_build_tier_c")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "1.0"
TRADITION = "reformed"
LANG = "pt-BR"
TIER = "C"

# Canonical system prompt — derived from docs/THEOLOGICAL_STATEMENT.md
SYSTEM_PROMPT = (
    "Você é um assistente teológico reformado, comprometido com as confissões históricas "
    "da fé reformada: a Confissão de Fé de Westminster (1647), os Cânones de Dort (1619), "
    "o Catecismo de Heidelberg (1563) e a Confissão Batista de Londres de 1689. "
    "Responda sempre de acordo com as Cinco Solas da Reforma (Sola Scriptura, Sola Gratia, "
    "Sola Fide, Solus Christus, Soli Deo Gloria) e os Cinco Pontos do Calvinismo (TULIP), "
    "conforme definidos nos Cânones de Dort. "
    "Não forneça aconselhamento pastoral pessoal sobre situações específicas — redirecione "
    "para um pastor ordenado e uma igreja local. Não especule sobre cronologias escatológicas "
    "ou profecias além do que as confissões afirmam. Soli Deo Gloria."
)

CONFESSIONS_DIR = PROJECT_ROOT / "data" / "sources" / "confessions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tier_c"

ALL_SOURCES = ["wsc", "wlc", "heidelberg", "wcf", "dort", "lcf"]

SOURCE_FILES = {
    "wsc":       CONFESSIONS_DIR / "westminster_shorter_catechism.txt",
    "wlc":       CONFESSIONS_DIR / "westminster_larger_catechism.txt",
    "heidelberg": CONFESSIONS_DIR / "heidelberg_catechism.txt",
    "wcf":       CONFESSIONS_DIR / "wcf_1647.txt",
    "dort":      CONFESSIONS_DIR / "canons_of_dort.txt",
    "lcf":       CONFESSIONS_DIR / "lcf_1689.txt",
}


# ---------------------------------------------------------------------------
# Encoding fix
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> str:
    """Fix mojibake from PDF extraction (UTF-8 bytes decoded as Latin-1).

    Symptom: 'Ã©' instead of 'é', 'Ã£' instead of 'ã', etc.
    Cause: pypdf returned bytes that are valid UTF-8, but were decoded as latin-1.
    Fix: encode back to latin-1, then decode as utf-8.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text  # already clean or unrecoverable — leave as-is


def normalize(text: str) -> str:
    """Normalize whitespace: collapse lines with only spaces into true blank lines,
    strip trailing spaces, and deduplicate blank lines. Applied before every parser."""
    text = fix_encoding(text)
    # Lines that contain only whitespace → empty line
    lines = [line if line.strip() else "" for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

_id_counter = 0

def make_record(question: str, answer: str, source_key: str,
                source_label: str, confessional_refs: list[str]) -> dict:
    global _id_counter
    _id_counter += 1
    record = {
        "id": f"openscriptura-reformed-pt-{_id_counter:05d}",
        "version": VERSION,
        "tradition": TRADITION,
        "lang": LANG,
        "tier": TIER,
        "source": source_label,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "confessional_refs": confessional_refs,
        "reviewed_by": "corpus_auto",  # Tier C — no pastoral review required
        "quality_score": 95,           # Confessional documents are canonical; score is fixed
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    record["sha256"] = content_hash(record)
    return record


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _split_on(text: str, pattern: str, flags: int = re.IGNORECASE) -> list[str]:
    """Split text into chunks using a regex delimiter, keeping the delimiter at the
    start of each chunk (except the leading preamble).
    Note: do NOT use inline flags (e.g. (?m)) in pattern — pass re.MULTILINE via flags instead.
    """
    parts = re.split(f"({pattern})", text, flags=flags)
    # parts = [preamble, delim1, body1, delim2, body2, ...]
    chunks = []
    i = 1
    while i < len(parts) - 1:
        chunks.append(parts[i] + parts[i + 1])
        i += 2
    return chunks


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_wsc(text: str) -> list[dict]:
    """Westminster Shorter Catechism — 'Pergunta N. question\\nR. answer\\nRef...'"""
    text = normalize(text)
    records = []
    # Split on 'Pergunta N.' markers
    chunks = _split_on(text, r"Pergunta\s+\d+\.")
    for chunk in chunks:
        m = re.match(r"Pergunta\s+(\d+)\.\s+(.+?)\n\nR\.\s+(.+?)(?:\n\nRef|\n\nReferências|\Z)", chunk, re.DOTALL)
        if not m:
            continue
        num = m.group(1)
        question = _clean(m.group(2))
        answer = _clean(m.group(3))
        if not question or not answer:
            continue
        records.append(make_record(
            question=question, answer=answer,
            source_key="wsc", source_label=f"WSC_{num}",
            confessional_refs=[f"WSC {num}"],
        ))
    return records


def parse_wlc(text: str) -> list[dict]:
    """Westminster Larger Catechism — 'N. question\\n\\nanswer\\n\\nrefs'"""
    text = normalize(text)
    records = []
    # Split on numbered question markers at line start
    chunks = _split_on(text, r"^\d+\.", flags=re.MULTILINE)
    for chunk in chunks:
        m = re.match(r"(\d+)\.\s+(.+?)\n\n(.+?)(?:\n\n|\Z)", chunk, re.DOTALL)
        if not m:
            continue
        num = m.group(1)
        question = _clean(m.group(2))
        # First paragraph after question is the answer; second paragraph is scripture refs
        answer = _clean(m.group(3).split("\n\n")[0])
        if not question or not answer or len(answer) < 10:
            continue
        records.append(make_record(
            question=question, answer=answer,
            source_key="wlc", source_label=f"WLC_{num}",
            confessional_refs=[f"WLC {num}"],
        ))
    return records


def parse_heidelberg(text: str) -> list[dict]:
    """Heidelberg Catechism — numbered Q&A, web-scraped."""
    text = normalize(text)
    records = []
    chunks = _split_on(text, r"^\d+\.", flags=re.MULTILINE)
    for chunk in chunks:
        m = re.match(r"(\d+)\.\s+(.+?)\n\n(.+?)(?:\n\n(?:DIA DO SENHOR|\d+\.|Parte)|\Z)", chunk, re.DOTALL)
        if not m:
            continue
        num = m.group(1)
        question = _clean(m.group(2))
        answer = _clean(m.group(3))
        if not question or not answer:
            continue
        records.append(make_record(
            question=question, answer=answer,
            source_key="heidelberg", source_label=f"Heidelberg_{num}",
            confessional_refs=[f"HC {num}"],
        ))
    return records


def parse_wcf(text: str) -> list[dict]:
    """Westminster Confession of Faith — chapter/article prose → template question."""
    text = normalize(text)
    records = []
    chapter_chunks = _split_on(text, r"CAP[IÍ]TULO\s+\d+")
    for ch_chunk in chapter_chunks:
        m = re.match(r"CAP[IÍ]TULO\s+(\d+)[:\s]+([^\n]+)\n(.+)", ch_chunk, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        ch_num, ch_title, ch_body = m.group(1), m.group(2).strip(), m.group(3)
        art_chunks = _split_on(ch_body, r"^\d+\.", flags=re.MULTILINE)
        if not art_chunks:
            body = _clean(ch_body)
            if len(body) < 50:
                continue
            q = f"O que ensina a Confissão de Fé de Westminster no Capítulo {ch_num} ({ch_title})?"
            records.append(make_record(question=q, answer=body, source_key="wcf",
                                        source_label=f"WCF_{ch_num}", confessional_refs=[f"WCF {ch_num}"]))
        else:
            for art_chunk in art_chunks:
                am = re.match(r"(\d+)\.\s+(.+)", art_chunk, re.DOTALL)
                if not am:
                    continue
                art_num = am.group(1)
                art_text = _clean(am.group(2))
                if len(art_text) < 40:
                    continue
                q = (f"O que ensina a Confissão de Fé de Westminster no Capítulo {ch_num} "
                     f"({ch_title}), artigo {art_num}?")
                records.append(make_record(question=q, answer=art_text, source_key="wcf",
                                            source_label=f"WCF_{ch_num}.{art_num}",
                                            confessional_refs=[f"WCF {ch_num}.{art_num}"]))
    return records


def parse_dort(text: str) -> list[dict]:
    """Canons of Dort — chapter/article + rejections → template question."""
    text = normalize(text)
    records = []
    chapter_chunks = _split_on(text, r"Cap[íi]tulo\s+\d+")
    for ch_chunk in chapter_chunks:
        m = re.match(r"Cap[íi]tulo\s+(\d+)\s*\n+([^\n]+)\n(.+)", ch_chunk, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        ch_num, ch_title, ch_body = m.group(1), m.group(2).strip(), m.group(3)
        # Articles
        art_chunks = _split_on(ch_body, r"ARTIGO\s+\d+")
        for art_chunk in art_chunks:
            am = re.match(r"ARTIGO\s+(\d+)\s*\n(.+)", art_chunk, re.DOTALL)
            if not am:
                continue
            art_num = am.group(1)
            art_text = _clean(re.split(r"REJEI[ÇC][ÃA]O", am.group(2))[0])
            if len(art_text) < 40:
                continue
            q = (f"O que ensinam os Cânones de Dort sobre '{ch_title}' "
                 f"(Capítulo {ch_num}, Artigo {art_num})?")
            records.append(make_record(question=q, answer=art_text, source_key="dort",
                                        source_label=f"Dort_{ch_num}.{art_num}",
                                        confessional_refs=[f"Dort {ch_num}.{art_num}"]))
        # Rejections
        rej_chunks = _split_on(ch_body, r"REJEI[ÇC][ÃA]O\s+\d+")
        for rej_chunk in rej_chunks:
            rm = re.match(r"REJEI[ÇC][ÃA]O\s+(\d+)\s*\n(.+)", rej_chunk, re.DOTALL)
            if not rm:
                continue
            rej_num = rm.group(1)
            rej_text = _clean(rm.group(2))
            if len(rej_text) < 40:
                continue
            q = (f"O que os Cânones de Dort rejeitam como erro no Capítulo {ch_num} "
                 f"({ch_title}), Rejeição {rej_num}?")
            records.append(make_record(question=q, answer=rej_text, source_key="dort",
                                        source_label=f"Dort_{ch_num}.rej{rej_num}",
                                        confessional_refs=[f"Dort {ch_num} rejeição {rej_num}"]))
    return records


def parse_lcf(text: str) -> list[dict]:
    """London Baptist Confession 1689 — chapter/paragraph prose → template question."""
    text = normalize(text)
    records = []
    chapter_chunks = _split_on(text, r"CAP[IÍ]TULO\s+\d+")
    for ch_chunk in chapter_chunks:
        m = re.match(r"CAP[IÍ]TULO\s+(\d+)\s*\n+([^\n]+)\n(.+)", ch_chunk, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        ch_num, ch_title, ch_body = m.group(1), m.group(2).strip(), m.group(3)
        para_chunks = _split_on(ch_body, r"^\d+\.", flags=re.MULTILINE)
        if not para_chunks:
            body = _clean(ch_body)
            if len(body) < 50:
                continue
            q = f"O que ensina a Confissão Batista de Londres de 1689 no Capítulo {ch_num} ({ch_title})?"
            records.append(make_record(question=q, answer=body, source_key="lcf",
                                        source_label=f"LCF_{ch_num}", confessional_refs=[f"LCF {ch_num}"]))
        else:
            for para_chunk in para_chunks:
                pm = re.match(r"(\d+)\.\s+(.+)", para_chunk, re.DOTALL)
                if not pm:
                    continue
                para_num = pm.group(1)
                para_text = _clean(pm.group(2))
                if len(para_text) < 40:
                    continue
                q = (f"O que ensina a Confissão Batista de Londres de 1689 no Capítulo {ch_num} "
                     f"({ch_title}), parágrafo {para_num}?")
                records.append(make_record(question=q, answer=para_text, source_key="lcf",
                                            source_label=f"LCF_{ch_num}.{para_num}",
                                            confessional_refs=[f"LCF {ch_num}.{para_num}"]))
    return records


PARSERS = {
    "wsc":       parse_wsc,
    "wlc":       parse_wlc,
    "heidelberg": parse_heidelberg,
    "wcf":       parse_wcf,
    "dort":      parse_dort,
    "lcf":       parse_lcf,
}

SOURCE_LABELS = {
    "wsc":       "Westminster Shorter Catechism",
    "wlc":       "Westminster Larger Catechism",
    "heidelberg": "Heidelberg Catechism",
    "wcf":       "Westminster Confession of Faith 1647",
    "dort":      "Canons of Dort",
    "lcf":       "London Baptist Confession 1689",
}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique = []
    dupes = 0
    for r in records:
        h = r["sha256"]
        if h in seen:
            dupes += 1
        else:
            seen.add(h)
            unique.append(r)
    return unique, dupes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Tier C confessional training corpus.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report counts without writing output files")
    parser.add_argument("--source", default=",".join(ALL_SOURCES),
                        help=f"Comma-separated sources to process (default: all). "
                             f"Options: {', '.join(ALL_SOURCES)}")
    args = parser.parse_args()

    sources = [s.strip().lower() for s in args.source.split(",")]
    invalid = [s for s in sources if s not in ALL_SOURCES]
    if invalid:
        log.error(f"Unknown sources: {invalid}. Valid: {ALL_SOURCES}")
        sys.exit(1)

    log.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Building Tier C corpus")
    log.info(f"  Sources  : {', '.join(sources)}")
    log.info(f"  Output   : {OUTPUT_DIR}")

    all_records: list[dict] = []
    source_counts: dict[str, int] = {}

    bar = ProgressBar(total=len(sources), label="sources")

    for src in sources:
        path = SOURCE_FILES[src]
        if not path.exists():
            log.warning(f"  ✗ {src}: file not found at {path} — skipping")
            bar.update()
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        parse_fn = PARSERS[src]

        try:
            records = parse_fn(text)
        except Exception as e:
            log.error(f"  ✗ {src}: parse error — {e}")
            bar.update()
            continue

        source_counts[src] = len(records)
        all_records.extend(records)
        log.info(f"  ✓ {SOURCE_LABELS[src]}: {len(records)} records")
        bar.update()

    bar.done()

    # Dedup
    unique_records, dupes = deduplicate(all_records)
    if dupes:
        log.info(f"Deduplication: removed {dupes} duplicate(s), {len(unique_records)} unique records remain")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"  Total records : {len(unique_records)}")
    for src, count in source_counts.items():
        log.info(f"    {SOURCE_LABELS[src]:<45} {count:>4}")
    log.info(f"{'='*60}")

    if args.dry_run:
        log.info("[DRY-RUN] No files written.")
        # Show a sample record
        if unique_records:
            log.info("\nSample record (first):")
            sample = dict(unique_records[0])
            sample["messages"] = [
                {**m, "content": m["content"][:80] + "..."
                 if len(m["content"]) > 80 else m["content"]}
                for m in sample["messages"]
            ]
            log.info(json.dumps(sample, ensure_ascii=False, indent=2))
        return

    if not unique_records:
        log.error("No records generated — nothing to write.")
        sys.exit(1)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUTPUT_DIR / "tier_c.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"\n✓ Written {len(unique_records)} records to {out_jsonl}")

    # Manifest
    manifest = {
        "version": VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "tradition": TRADITION,
        "lang": LANG,
        "tier": TIER,
        "total_records": len(unique_records),
        "duplicates_removed": dupes,
        "source_breakdown": {
            SOURCE_LABELS[src]: count for src, count in source_counts.items()
        },
    }
    out_manifest = OUTPUT_DIR / "manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"✓ Manifest written to {out_manifest}")
    log.info("\nNext step: python scripts/02_build_tier_b.py --dry-run")


if __name__ == "__main__":
    main()
