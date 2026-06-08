"""
02_build_tier_b.py — Tier B synthetic Q&A generation for OpenScriptura.

Pipeline:
  1. Load Spurgeon PT-BR sermons (quality score >= MIN_QUALITY_SCORE).
  2. Load Monergismo Reformed ebooks (PDFs, excluded authors skipped).
  3. Chunk each source text (~CHUNK_CHARS chars, OVERLAP_CHARS overlap).
  4. For each chunk: call DeepSeek Flash to generate NUM_QA_PER_CHUNK Q&A pairs.
  5. For each generated pair: call DeepSeek Flash judge with M7 quality rubric.
  6. Pairs passing QUALITY_THRESHOLD written to data/tier_b/tier_b.jsonl.
  7. Manifest written to data/tier_b/manifest.json.

Usage:
  python scripts/02_build_tier_b.py --dry-run
  python scripts/02_build_tier_b.py
  python scripts/02_build_tier_b.py --no-resume      # force fresh run
  python scripts/02_build_tier_b.py --source spurgeon --limit 5  # dev subset

Environment variables (via .env):
  OPENROUTER_API_KEY          required
  OPENROUTER_BASE_URL         required
  OPENROUTER_MODEL_GENERATOR  default: deepseek/deepseek-v4-flash
  OPENROUTER_MODEL_JUDGE_B    default: deepseek/deepseek-v4-flash
  COST_LIMIT_USD_PHASE1_B     default: 1.50
"""

import argparse
import asyncio
import hashlib as _hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import httpx
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

from utils.api_client import OpenRouterClient
from utils.cost_tracker import CostLimitExceeded, CostTracker
from utils.hash import content_hash
from utils.logger import get_logger
from utils.progress import ProgressBar

log = get_logger("02_build_tier_b")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_CHARS = 4_000       # ~1,000 tokens at ~4 chars/token
OVERLAP_CHARS = 500       # ~125 token overlap between adjacent chunks
NUM_QA_PER_CHUNK = 2      # Q&A pairs to request per chunk
QUALITY_THRESHOLD = 93    # M7: pairs scoring below this are rejected
SEMAPHORE_LIMIT = 6       # concurrent async requests (conservative for judge)
MIN_QUALITY_SCORE = 93    # Spurgeon avaliacao threshold (score_final)

# Max chunks sampled per source document (evenly spaced across the full text).
# Caps total chunks to a predictable budget:
#   103 Spurgeon × 8 + 77 Monergismo × 6 = 824 + 462 = 1,286 chunks
#   × 2 Q&A × ~3 API calls each ≈ ~7,700 calls at $0.50/1M tokens ≈ ~$1.20
MAX_CHUNKS_SPURGEON = 8
MAX_CHUNKS_MONERGISMO = 6

EXCLUDED_AUTHORS = {
    "John_Wesley",
    "Karl_Barth",
    "F_A_Hayek",
    "Hermas",
    "Inacio_de_Antioquia",
}

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

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GENERATE_PROMPT = """\
Você é um especialista em curadoria de dados teológicos reformados em português brasileiro.

A partir do trecho abaixo, gere exatamente {n} pares pergunta-resposta que:
1. Sejam fiéis ao texto e à teologia reformada histórica
2. Tenham perguntas claras sobre doutrina, Escritura ou vida cristã
3. Tenham respostas completas, edificantes e sem especulações além do texto
4. Incluam referências confessionais (ex.: "WCF 3.1", "Dort I.7", "HC P.20") quando aplicável
5. Sejam adequados para treinar um assistente teológico reformado em PT-BR

FONTE: {source_label}

TRECHO:
{chunk}

Responda APENAS com um array JSON válido (sem texto adicional, sem markdown):
[
  {{"question": "...", "answer": "...", "confessional_refs": ["WCF X.Y"]}},
  {{"question": "...", "answer": "...", "confessional_refs": []}}
]
"""

_JUDGE_PROMPT = """\
Avalie este par pergunta-resposta para treinamento de um assistente teológico reformado em PT-BR.

PERGUNTA: {question}

RESPOSTA: {answer}

Pontue cada critério de 0 a 100 e calcule o score ponderado:
- precisao_teologica (peso 40%): Está correta e conforme à teologia reformada histórica?
- clareza_pastoral   (peso 20%): É clara, útil e pastoralmente edificante?
- qualidade_ptbr     (peso 20%): O português é correto, natural e fluente?
- sem_alucinacao     (peso 20%): Não inventa fatos, referências ou citações?

score_total = 0.4*precisao + 0.2*pastoral + 0.2*ptbr + 0.2*alucinacao

Responda APENAS com JSON válido (sem texto adicional):
{{"precisao_teologica": 0, "clareza_pastoral": 0, "qualidade_ptbr": 0, "sem_alucinacao": 0, "score_total": 0, "aprovado": false, "comentario": "..."}}
"""

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
    max_chunks: int | None = None,
) -> list[str]:
    """Split text into overlapping character windows. Skip near-empty chunks.

    If max_chunks is set, evenly sample that many chunks across the full text
    rather than returning all of them. This keeps budget predictable for large
    documents (e.g. Calvin's Institutes).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunk = text[start:end].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    if max_chunks and len(chunks) > max_chunks:
        # Evenly-spaced sample: first, last, and evenly distributed middle.
        # Guard against max_chunks=1 (division by zero when max_chunks-1==0).
        if max_chunks == 1:
            chunks = [chunks[0]]
        else:
            indices = [round(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
            chunks = [chunks[i] for i in sorted(set(indices))]

    return chunks


def fix_encoding(text: str) -> str:
    """Repair mojibake from pypdf Latin-1 misread of UTF-8 bytes."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            pages.append(fix_encoding(raw))
        text = "\n".join(pages)
        # Normalise blank lines
        text = re.sub(r"\n[ \t]*\n", "\n\n", text)
        return text.strip()
    except Exception as exc:
        log.warning("  Could not extract %s: %s", pdf_path.name, exc)
        return ""

# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_spurgeon_sources(spurgeon_dir: Path, min_score: float = MIN_QUALITY_SCORE) -> list[dict]:
    """Return list of {source_id, source_label, text} for qualifying sermons."""
    sources = []
    md_files = sorted(spurgeon_dir.glob("*.pt-br.md"))
    for md in md_files:
        stem = md.stem.replace(".pt-br", "")
        avaliacao = spurgeon_dir / f"{stem}_avaliacao.json"
        if avaliacao.exists():
            data = json.loads(avaliacao.read_text(encoding="utf-8"))
            score = data.get("score_final", 0)
            if score < min_score:
                continue
        # No avaliacao = accept (shouldn't happen but be safe)
        text = md.read_text(encoding="utf-8").strip()
        if len(text) < 500:
            continue
        sources.append({
            "source_id": f"spurgeon_{stem}",
            "source_label": f"Spurgeon — {stem}",
            "text": text,
        })
    return sources


def load_monergismo_sources(monergismo_dir: Path) -> list[dict]:
    """Return list of {source_id, source_label, text} from Monergismo PDFs.

    Extracted text is cached alongside each PDF as a .txt file so re-runs
    skip pypdf entirely (~10 min → <1 s on warm cache).
    """
    sources = []
    for author_dir in sorted(monergismo_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        if author_dir.name in EXCLUDED_AUTHORS:
            continue
        for pdf in sorted(author_dir.glob("*.pdf")):
            cache = pdf.with_suffix(".txt")
            if cache.exists():
                text = cache.read_text(encoding="utf-8")
            else:
                text = extract_pdf_text(pdf)
                if text:
                    cache.write_text(text, encoding="utf-8")
            if len(text) < 500:
                log.debug("  Skipping short/empty PDF: %s", pdf.name)
                continue
            source_id = f"monergismo_{author_dir.name}_{pdf.stem}".replace(" ", "_")[:80]
            sources.append({
                "source_id": source_id,
                "source_label": f"{author_dir.name} — {pdf.stem}",
                "text": text,
            })
    return sources

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_generated_qa(raw: str) -> list[dict]:
    """Extract list of {question, answer, confessional_refs} from LLM output."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Find the JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
        result = []
        for item in items:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                result.append({
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip(),
                    "confessional_refs": item.get("confessional_refs", []),
                })
        return result
    except json.JSONDecodeError:
        return []


def _parse_judge_response(raw: str) -> dict:
    """Extract judge scores from LLM output."""
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Strip think tags
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"score_total": 0, "aprovado": False, "comentario": "parse error"}
    try:
        data = json.loads(m.group(0))
        # Recompute weighted score to be safe
        p = float(data.get("precisao_teologica", 0))
        c = float(data.get("clareza_pastoral", 0))
        q = float(data.get("qualidade_ptbr", 0))
        a = float(data.get("sem_alucinacao", 0))
        score = round(0.4 * p + 0.2 * c + 0.2 * q + 0.2 * a, 1)
        return {
            "precisao_teologica": p,
            "clareza_pastoral": c,
            "qualidade_ptbr": q,
            "sem_alucinacao": a,
            "score_total": score,
            "aprovado": score >= QUALITY_THRESHOLD,
            "comentario": data.get("comentario", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"score_total": 0, "aprovado": False, "comentario": "parse error"}

# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def make_record(
    question: str,
    answer: str,
    source_id: str,
    source_label: str,
    confessional_refs: list[str],
    quality_score: float,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    base = {
        "tradition": "reformed",
        "lang": "pt-BR",
        "messages": messages,
    }
    sha = content_hash(base)
    return {
        "id": f"openscriptura-reformed-ptbr-b-{sha[:12]}",
        "version": "1.0",
        "tradition": "reformed",
        "lang": "pt-BR",
        "tier": "B",
        "source": source_id,
        "source_label": source_label,
        "sha256": sha,
        "messages": messages,
        "confessional_refs": confessional_refs,
        "reviewed_by": "auto",
        "quality_score": quality_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_done_ids(results_file: Path) -> set[str]:
    """Return set of sha256 hashes already written to the results file."""
    if not results_file.exists():
        return set()
    done = set()
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["sha256"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _load_done_chunks(output_dir: Path) -> set[str]:
    """Return set of chunk hashes already processed (accepted OR rejected).

    This is the CHUNK-level checkpoint — it prevents re-spending generation
    cost on chunks that were already fully processed in a previous run, even
    if all their pairs were rejected (which would leave no record-level trace).
    """
    path = output_dir / "done_chunks.txt"
    if not path.exists():
        return set()
    return set(h.strip() for h in path.read_text(encoding="utf-8").splitlines() if h.strip())


def _mark_chunk_done(output_dir: Path, chunk_hash: str) -> None:
    """Append a chunk hash to the done_chunks.txt checkpoint file."""
    path = output_dir / "done_chunks.txt"
    with open(path, "a", encoding="utf-8") as f:
        f.write(chunk_hash + "\n")


def _chunk_hash(source_id: str, chunk_idx: int, chunk_text: str) -> str:
    """Stable hash identifying a specific chunk. Used for chunk-level resume."""
    key = f"{source_id}::{chunk_idx}::{len(chunk_text)}::{chunk_text[:200]}"
    return _hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _append_record(results_file: Path, record: dict) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------

async def _process_chunk(
    semaphore: asyncio.Semaphore,
    client_http: httpx.AsyncClient,
    api: OpenRouterClient,
    source_id: str,
    source_label: str,
    chunk: str,
    chunk_idx: int,
    chunk_id: str,
    gen_model: str,
    judge_model: str,
    cost_tracker: CostTracker,
    done_ids: set[str],
    done_chunks: set[str],
    output_dir: Path,
    dry_run: bool,
) -> list[dict]:
    """Generate and judge Q&A pairs for a single chunk. Returns accepted records."""
    accepted = []

    async with semaphore:
        if dry_run:
            return []

        # CHUNK-LEVEL CHECKPOINT: skip entire chunk (gen + judge) if already done.
        # This is the critical resume guard — prevents re-spending generation budget
        # on chunks that were fully processed in a prior run, including rejected ones.
        if chunk_id in done_chunks:
            return []

        # --- Generation ---
        gen_prompt = _GENERATE_PROMPT.format(
            n=NUM_QA_PER_CHUNK,
            source_label=source_label,
            chunk=chunk[:3000],  # hard cap so we don't blow max_tokens
        )
        try:
            gen_resp = await api.chat(
                client=client_http,
                model=gen_model,
                messages=[{"role": "user", "content": gen_prompt}],
                temperature=0.7,   # some creativity for diversity
                max_tokens=1024,
                seed=42,
                enable_thinking=False,
                log_key=f"gen_{source_id}_{chunk_idx}",
            )
            gen_cost = api.estimate_cost_usd(gen_resp, gen_model)
            cost_tracker.add(gen_cost)
            raw_gen = api.extract_text(gen_resp)
        except CostLimitExceeded:
            raise
        except Exception as exc:
            log.warning("  Gen error [%s chunk %d]: %s", source_id, chunk_idx, exc)
            return []

        pairs = _parse_generated_qa(raw_gen)
        if not pairs:
            log.debug("  No pairs parsed from %s chunk %d", source_id, chunk_idx)
            return []

        # --- Judge each pair ---
        for pair in pairs:
            question = pair["question"]
            answer = pair["answer"]
            refs = pair.get("confessional_refs", [])

            # Skip if already in checkpoint
            candidate = make_record(question, answer, source_id, source_label, refs, 0.0)
            if candidate["sha256"] in done_ids:
                accepted.append(candidate)
                continue

            try:
                judge_resp = await api.chat(
                    client=client_http,
                    model=judge_model,
                    messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                        question=question, answer=answer
                    )}],
                    temperature=0.0,
                    max_tokens=512,
                    seed=42,
                    enable_thinking=False,
                    log_key=f"judge_{source_id}_{chunk_idx}_p{pairs.index(pair)}",
                )
                judge_cost = api.estimate_cost_usd(judge_resp, judge_model)
                cost_tracker.add(judge_cost)
                judge_raw = api.extract_text(judge_resp)
                verdict = _parse_judge_response(judge_raw)
            except CostLimitExceeded:
                raise
            except Exception as exc:
                log.warning("  Judge error [%s chunk %d]: %s", source_id, chunk_idx, exc)
                continue

            if verdict["aprovado"]:
                record = make_record(
                    question, answer, source_id, source_label, refs,
                    verdict["score_total"]
                )
                accepted.append(record)
                log.debug(
                    "  ✓ %s chunk %d  score=%.1f  refs=%s",
                    source_id, chunk_idx, verdict["score_total"], refs,
                )
            else:
                log.debug(
                    "  ✗ rejected %s chunk %d  score=%.1f: %s",
                    source_id, chunk_idx, verdict["score_total"],
                    verdict.get("comentario", "")[:80],
                )

        # Mark chunk done regardless of acceptance — prevents re-processing on resume
        _mark_chunk_done(output_dir, chunk_id)

    return accepted

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    spurgeon_dir: Path,
    monergismo_dir: Path,
    output_dir: Path,
    gen_model: str,
    judge_model: str,
    api_key: str,
    base_url: str,
    cost_limit: float,
    dry_run: bool,
    resume: bool,
    source_filter: str | None,
    limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "tier_b.jsonl"
    done_ids    = _load_done_ids(results_file)   if resume else set()
    done_chunks = _load_done_chunks(output_dir)  if resume else set()

    if done_ids or done_chunks:
        log.info(
            "Resuming: %d records + %d chunks already done, skipping.",
            len(done_ids), len(done_chunks),
        )

    # --- Build source list ---
    log.info("Loading sources...")
    sources = []

    if source_filter in (None, "spurgeon"):
        spurgeon_sources = load_spurgeon_sources(spurgeon_dir)
        log.info("  Spurgeon: %d qualifying sermons (score >= %d)", len(spurgeon_sources), MIN_QUALITY_SCORE)
        sources.extend(spurgeon_sources)

    if source_filter in (None, "monergismo"):
        mono_sources = load_monergismo_sources(monergismo_dir)
        log.info("  Monergismo: %d documents", len(mono_sources))
        sources.extend(mono_sources)

    if limit:
        sources = sources[:limit]
        log.info("  Limited to first %d sources (--limit)", limit)

    # --- Expand into chunks ---
    all_chunks: list[tuple[str, str, str, int]] = []  # (source_id, source_label, chunk, idx)
    for src in sources:
        is_spurgeon = src["source_id"].startswith("spurgeon_")
        cap = MAX_CHUNKS_SPURGEON if is_spurgeon else MAX_CHUNKS_MONERGISMO
        chunks = chunk_text(src["text"], max_chunks=cap)
        for i, chunk in enumerate(chunks):
            all_chunks.append((src["source_id"], src["source_label"], chunk, i))

    log.info(
        "\n%d sources → %d chunks  |  ~%d Q&A pairs to generate  |  model: %s  |  judge: %s",
        len(sources), len(all_chunks),
        len(all_chunks) * NUM_QA_PER_CHUNK,
        gen_model, judge_model,
    )

    # --- Auto-seed done_chunks.txt from existing records ---
    # When resuming after a run that predates done_chunks.txt, we reconstruct
    # which chunks are done by checking which source_ids have existing records.
    # Any source with ≥1 accepted record gets all its chunks marked done.
    # This prevents re-spending generation budget on fully-processed sources.
    if resume and done_ids and not done_chunks:
        existing_sources = set()
        for line in (results_file.open(encoding="utf-8") if results_file.exists() else []):
            if line.strip():
                try:
                    existing_sources.add(json.loads(line)["source"])
                except (json.JSONDecodeError, KeyError):
                    pass
        seeded = 0
        for s_id, s_label, chunk, idx in all_chunks:
            if s_id in existing_sources:
                ch = _chunk_hash(s_id, idx, chunk)
                if ch not in done_chunks:
                    _mark_chunk_done(output_dir, ch)
                    done_chunks.add(ch)
                    seeded += 1
        if seeded:
            log.info("  Auto-seeded %d chunk hashes from existing records.", seeded)

    if dry_run:
        skippable = sum(1 for s_id, _, chunk, idx in all_chunks
                        if _chunk_hash(s_id, idx, chunk) in done_chunks)
        log.info("\n[DRY-RUN] Would process %d chunks (%d already done, %d remaining)",
                 len(all_chunks), skippable, len(all_chunks) - skippable)
        log.info("[DRY-RUN] Cost limit: $%.2f", cost_limit)
        log.info("[DRY-RUN] Output: %s", results_file)
        # Show first source as sample
        if all_chunks:
            s_id, s_label, chunk, idx = all_chunks[0]
            log.info("\n[DRY-RUN] Sample chunk from '%s' (idx=%d, %d chars):", s_label, idx, len(chunk))
            log.info("  %s...", chunk[:200].replace("\n", " "))
        return

    # --- Run async pipeline ---
    cost_tracker = CostTracker(limit_usd=cost_limit)
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    api = OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        log_raw_dir=PROJECT_ROOT / "logs" / "raw",
    )

    progress = ProgressBar(total=len(all_chunks), label="chunks")
    total_accepted = len(done_ids)
    total_rejected = 0

    async with httpx.AsyncClient() as client_http:
        tasks = [
            _process_chunk(
                semaphore=semaphore,
                client_http=client_http,
                api=api,
                source_id=s_id,
                source_label=s_label,
                chunk=chunk,
                chunk_idx=idx,
                chunk_id=_chunk_hash(s_id, idx, chunk),
                gen_model=gen_model,
                judge_model=judge_model,
                cost_tracker=cost_tracker,
                done_ids=done_ids,
                done_chunks=done_chunks,
                output_dir=output_dir,
                dry_run=False,
            )
            for s_id, s_label, chunk, idx in all_chunks
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                records = await coro
                for rec in records:
                    if rec["sha256"] not in done_ids:
                        _append_record(results_file, rec)
                        done_ids.add(rec["sha256"])
                        total_accepted += 1
                    # else already counted from resume
                if records is not None:
                    total_rejected += max(0, NUM_QA_PER_CHUNK - len(records))
            except CostLimitExceeded:
                log.warning("💰 Cost limit $%.2f reached — stopping early.", cost_limit)
                break
            except Exception as exc:
                log.error("❌ Chunk error (skipping): %s", exc)
            finally:
                progress.update()

    progress.done()

    # --- Summary ---
    log.info("\n" + "=" * 60)
    log.info("  Tier B generation complete")
    log.info("  Accepted records  : %d", total_accepted)
    log.info("  Rejected/skipped  : ~%d", total_rejected)
    log.info("  Total cost        : $%.4f / $%.2f limit", cost_tracker.total, cost_limit)
    log.info("  Output            : %s", results_file)
    log.info("=" * 60)

    # Dedup pass
    _dedup_jsonl(results_file)

    # Write manifest — count from the actual file (accumulates across resumes)
    total_in_file = sum(1 for _ in open(results_file, encoding="utf-8")) if results_file.exists() else 0
    _write_manifest(output_dir, total_in_file, gen_model, judge_model)


def _dedup_jsonl(path: Path) -> None:
    """In-place deduplication by sha256."""
    if not path.exists():
        return
    seen: set[str] = set()
    kept: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sha = rec["sha256"]
                if sha not in seen:
                    seen.add(sha)
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                kept.append(line)
    original_count = 0  # count before dedup (tracked during read above)
    with open(path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    log.info("  Dedup: %d unique records in %s", len(kept), path)


def _write_manifest(output_dir: Path, count: int, gen_model: str, judge_model: str) -> None:
    manifest = {
        "tier": "B",
        "tradition": "reformed",
        "lang": "pt-BR",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_model": gen_model,
        "judge_model": judge_model,
        "quality_threshold": QUALITY_THRESHOLD,
        "count": count,
        "chunk_chars": CHUNK_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "num_qa_per_chunk": NUM_QA_PER_CHUNK,
        "min_spurgeon_score": MIN_QUALITY_SCORE,
        "excluded_authors": sorted(EXCLUDED_AUTHORS),
    }
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Manifest written to %s", out)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Tier B synthetic Q&A dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without API calls")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output, start fresh")
    parser.add_argument(
        "--source", choices=["spurgeon", "monergismo"],
        help="Process only one source (default: both)"
    )
    parser.add_argument("--limit", type=int, help="Cap number of source documents (dev/test)")
    parser.add_argument("--gen-model", help="Override generator model")
    parser.add_argument("--judge-model", help="Override judge model")
    parser.add_argument("--cost-limit", type=float, help="Override cost limit in USD (default: $1.50 or COST_LIMIT_USD_PHASE1_B env)")
    args = parser.parse_args()

    try:
        api_key  = _get_env("OPENROUTER_API_KEY")
        base_url = _get_env("OPENROUTER_BASE_URL")
        gen_model   = args.gen_model or os.getenv("OPENROUTER_MODEL_GENERATOR", "deepseek/deepseek-v4-flash")
        judge_model = args.judge_model or os.getenv("OPENROUTER_MODEL_JUDGE_B", "deepseek/deepseek-v4-flash")
        cost_limit  = args.cost_limit or float(os.getenv("COST_LIMIT_USD_PHASE1_B", "1.50"))
    except EnvironmentError as exc:
        log.error("%s", exc)
        sys.exit(1)

    spurgeon_dir   = PROJECT_ROOT / "data" / "sources" / "spurgeon"
    monergismo_dir = PROJECT_ROOT / "data" / "sources" / "monergismo"
    output_dir     = PROJECT_ROOT / "data" / "tier_b"

    W = 64
    print("=" * W)
    print("  OpenScriptura — Tier B Dataset Builder")
    print("=" * W)
    print(f"  Generator  : {gen_model}")
    print(f"  Judge      : {judge_model}")
    print(f"  Threshold  : {QUALITY_THRESHOLD}/100")
    print(f"  Cost limit : ${cost_limit:.2f}")
    print(f"  Source     : {args.source or 'spurgeon + monergismo'}")
    if args.limit:
        print(f"  Limit      : {args.limit} documents")
    print("=" * W)

    asyncio.run(run_pipeline(
        spurgeon_dir=spurgeon_dir,
        monergismo_dir=monergismo_dir,
        output_dir=output_dir,
        gen_model=gen_model,
        judge_model=judge_model,
        api_key=api_key,
        base_url=base_url,
        cost_limit=cost_limit,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        source_filter=args.source,
        limit=args.limit,
    ))


if __name__ == "__main__":
    main()
