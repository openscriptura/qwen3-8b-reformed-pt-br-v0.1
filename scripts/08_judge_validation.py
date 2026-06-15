"""
08_judge_validation.py — validate the LLM judge against human labels (κ).

CEFE.AI's judge is an LLM; good practice (VALIDATION_REPORT M9) is to confirm it
agrees with human raters on a sample before trusting its scores. This is a
two-step utility:

  1) --sample : draw N items from a results JSONL and write a BLIND labeling
                template (prompt, model response, a blank human_score) — the judge's
                score is hidden in a sidecar `<out>.key.jsonl` so the human does NOT
                anchor on it (which would inflate κ). Re-joined automatically at --score.
  2) --score  : read the filled template (+ the sidecar key) and report
                quadratic-weighted Cohen's κ (ordinal) between the human and the
                judge, on the official scale (RR 0-4, CB 1-7), plus exact agreement.

Usage:
  python scripts/08_judge_validation.py --sample --benchmark rr \
      --results results/baseline_qwen_qwen3_8b_noprompt_RR.jsonl --n 50 \
      --out results/judge_validation_RR_template.jsonl
  # ... a human fills the "human_score" field in each line ...
  python scripts/08_judge_validation.py --score --benchmark rr \
      --labels results/judge_validation_RR_template.jsonl

No API calls, no GPU. Pure local.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.cefeai import load_scoring_prompt, quadratic_weighted_kappa
from utils.logger import get_logger

log = get_logger("08_judge_validation")

SCALE = {"rr": (0, 4), "cb": (1, 7)}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _key_path(p: Path) -> Path:
    """Sidecar 'answer key' path next to a template: foo.jsonl -> foo.key.jsonl."""
    return p.with_name(p.stem + ".key" + p.suffix)


def do_sample(args) -> None:
    results = _read_jsonl(args.results)
    scored = [r for r in results if isinstance(r.get("judge_score"), int)]
    if not scored:
        log.error("No scored records in %s", args.results)
        sys.exit(1)
    random.seed(args.seed)
    lo, hi = SCALE[args.benchmark]
    # STRATIFIED by default: CB is ~80% neutral (4) and RR ~87% zero, so a uniform
    # sample makes κ dominated by easy modal agreement and barely tests the
    # off-modal ratings that DRIVE the metric. Draw ~half from the modal score and
    # ~half from the rest so κ actually probes the discriminative ratings. (Note:
    # this is a stratified κ — it stresses the informative ratings, not a population
    # estimate.) `--no-stratify` falls back to a uniform sample.
    if getattr(args, "stratify", True):
        from collections import Counter
        modal = Counter(r["judge_score"] for r in scored).most_common(1)[0][0]
        modal_recs = [r for r in scored if r["judge_score"] == modal]
        other_recs = [r for r in scored if r["judge_score"] != modal]
        # ~half off-modal when both are plentiful; backfill from off-modal when the
        # modal class is scarce, so the sample still reaches n (capped by what exists).
        n_other = min(len(other_recs), max(args.n // 2, args.n - len(modal_recs)))
        n_modal = min(len(modal_recs), args.n - n_other)
        sample = random.sample(modal_recs, n_modal) + random.sample(other_recs, n_other)
        random.shuffle(sample)
        log.info("Stratified sample: %d at the modal score %d + %d off-modal (informative).",
                 n_modal, modal, n_other)
    else:
        sample = random.sample(scored, min(args.n, len(scored)))
    # BLIND template: the human must NOT see the judge's score while labeling — seeing
    # it anchors the rating and inflates κ (invalid validation). The judge scores go to
    # a separate sidecar key file, re-joined by prompt_id at --score.
    key_path = _key_path(args.out)
    with args.out.open("w", encoding="utf-8") as fh, key_path.open("w", encoding="utf-8") as kh:
        for r in sample:
            item = {
                "prompt_id": r.get("prompt_id"),
                "prompt": r.get("prompt"),
                "response": r.get("response"),
            }
            # CB only: surface the transition direction (NOT the judge score) so the
            # human knows which faith is the source and which is the target to rate.
            if r.get("religion_from") or r.get("religion_to"):
                item["transition"] = f'{r.get("religion_from")} -> {r.get("religion_to")}'
            item["human_score"] = None   # <-- fill an integer in [lo..hi]; do NOT consult the judge
            item["_scale"] = f"{lo}-{hi}"
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            kh.write(json.dumps({"prompt_id": r.get("prompt_id"),
                                 "judge_score": r.get("judge_score")}, ensure_ascii=False) + "\n")
    log.info("Wrote %d BLIND items to %s — fill 'human_score' (int %d..%d) WITHOUT looking at the judge.",
             len(sample), args.out, lo, hi)
    log.info("Judge scores stashed in %s (used automatically by --score; don't open it until you've labeled).",
             key_path)


def _parse_label(s: str, lo: int, hi: int):
    """Parse a labeler keystroke → ('score', int) | ('skip',) | ('quit',) | ('bad',)."""
    s = s.strip().lower()
    if s in ("q", "quit"):
        return ("quit",)
    if s in ("s", "skip"):
        return ("skip",)
    try:
        v = int(s)
    except ValueError:
        return ("bad",)
    return ("score", v) if lo <= v <= hi else ("bad",)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def do_label(args) -> None:
    """Guided interactive labeling — shows prompt+response, validates the score, and
    writes human_score back atomically (resumable). Avoids hand-editing the JSONL."""
    rows = _read_jsonl(args.labels)
    lo, hi = SCALE[args.benchmark]
    scale = load_scoring_prompt(args.benchmark).get("rating_scale", {}) or {}
    todo = [r for r in rows if not isinstance(r.get("human_score"), int)]
    print("=" * 70)
    print(f"  Rotulagem {args.benchmark.upper()} — {len(rows) - len(todo)}/{len(rows)} já feitos. Escala {lo}-{hi}:")
    for k in sorted(scale, key=lambda x: int(x)):
        print(f"    {k} = {scale[k]}")
    print("  Comandos:  número = nota  |  s = pular  |  q = salvar e sair")
    print("=" * 70)
    changed = False
    try:
        for r in rows:
            if isinstance(r.get("human_score"), int):
                continue
            print("\n" + "-" * 70)
            tail = f"   [{r.get('transition')}]" if r.get("transition") else ""
            print(f"id: {r.get('prompt_id')}{tail}")
            print(f"\nPROMPT:\n{r.get('prompt')}")
            print(f"\nRESPOSTA:\n{(r.get('response') or '')[:1800]}")
            while True:
                kind = _parse_label(input(f"\nnota [{lo}-{hi}] / s / q: "), lo, hi)
                if kind[0] == "score":
                    r["human_score"] = kind[1]; changed = True; break
                if kind[0] == "skip":
                    break
                if kind[0] == "quit":
                    if changed:
                        _atomic_write_jsonl(args.labels, rows)
                    print("\nSalvo. Rode de novo --label para continuar, ou --score ao terminar.")
                    return
                print(f"  inválido — inteiro {lo}-{hi}, 's' (pular) ou 'q' (sair).")
    finally:
        if changed:
            _atomic_write_jsonl(args.labels, rows)
    n_lab = sum(1 for r in rows if isinstance(r.get("human_score"), int))
    print(f"\n✅ {n_lab}/{len(rows)} rotulados. Rode --score para calcular o κ.")


def do_score(args) -> None:
    rows = _read_jsonl(args.labels)
    lo, hi = SCALE[args.benchmark]
    # Judge scores: prefer the row's own judge_score (legacy non-blind templates),
    # else the sidecar key file written by --sample (blind mode), joined by prompt_id.
    judge_by_id: dict = {}
    key_path = _key_path(args.labels)
    if key_path.exists():
        for k in _read_jsonl(key_path):
            if isinstance(k.get("judge_score"), int):
                judge_by_id[k.get("prompt_id")] = k["judge_score"]
    pairs = []
    for r in rows:
        if not isinstance(r.get("human_score"), int):
            continue
        j = r.get("judge_score")
        if not isinstance(j, int):
            j = judge_by_id.get(r.get("prompt_id"))
        if isinstance(j, int):
            pairs.append((int(j), int(r["human_score"])))
    n = len(pairs)
    if n == 0:
        log.error("No rows with BOTH a human_score and a judge_score (row or %s) in %s.",
                  key_path.name, args.labels)
        sys.exit(1)
    out_of_range = [(j, h) for j, h in pairs if not (lo <= h <= hi)]
    if out_of_range:
        log.error("%d human_score(s) outside the %d-%d scale — fix them: %s",
                  len(out_of_range), lo, hi, out_of_range[:5])
        sys.exit(1)
    judge = [j for j, _ in pairs]
    human = [h for _, h in pairs]
    kappa = quadratic_weighted_kappa(judge, human, min_rating=lo, max_rating=hi)
    exact = sum(1 for j, h in pairs if j == h) / n
    within1 = sum(1 for j, h in pairs if abs(j - h) <= 1) / n
    # Degenerate guard: with no variance in a rater, weighted κ is mathematically
    # 1.0 (or undefined) — "near-perfect" would be falsely reassuring. Flag it.
    degenerate = len(set(judge)) == 1 or len(set(human)) == 1
    release = (load_scoring_prompt(args.benchmark).get("release_id")
               or load_scoring_prompt(args.benchmark).get("benchmark", {}).get("release_id"))

    print("=" * 60)
    print(f"  Judge validation — {args.benchmark.upper()}  ({release})")
    print(f"  labeled pairs            : {n}")
    print(f"  quadratic-weighted κ     : {kappa}")
    print(f"  exact agreement          : {exact*100:.1f}%")
    print(f"  within ±1 level          : {within1*100:.1f}%")
    verdict = ("DEGENERATE — uninformative" if degenerate
               else "near-perfect" if kappa >= 0.8 else "substantial" if kappa >= 0.6
               else "moderate" if kappa >= 0.4 else "WEAK — judge suspect")
    print(f"  interpretation           : {verdict}")
    if degenerate:
        print("  ⚠  One rater has NO variance (all the same score) — κ is not meaningful here.")
        print("     Use a stratified sample (default) with enough off-modal items, or label more.")
    elif kappa < 0.6:
        print("  ⚠  κ < 0.6 — the judge does not agree well with humans; reconsider the judge")
        print("     model before trusting/publishing the benchmark numbers.")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the CEFEAI LLM judge against human labels (κ).")
    p.add_argument("--benchmark", choices=["rr", "cb"], required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="draw a human-labeling template")
    mode.add_argument("--label", action="store_true", help="interactively fill a template (guided, resumable)")
    mode.add_argument("--score", action="store_true", help="compute κ from a filled template")
    p.add_argument("--results", type=Path, help="[--sample] results JSONL to sample from")
    p.add_argument("--out", type=Path, help="[--sample] template output path")
    p.add_argument("--labels", type=Path, help="[--label/--score] template path")
    p.add_argument("--n", type=int, default=50, help="[--sample] sample size (default 50)")
    p.add_argument("--seed", type=int, default=42, help="[--sample] RNG seed (default 42)")
    p.add_argument("--no-stratify", dest="stratify", action="store_false", default=True,
                   help="[--sample] uniform sample instead of the default stratified "
                        "(~half modal + ~half off-modal, so κ tests the discriminative ratings).")
    args = p.parse_args()

    if args.sample:
        if not args.results or not args.out:
            p.error("--sample requires --results and --out")
        if not args.results.exists():
            log.error("results file not found: %s", args.results)
            sys.exit(1)
        do_sample(args)
    else:
        if not args.labels:
            p.error("--label/--score require --labels")
        if not args.labels.exists():
            log.error("labels file not found: %s", args.labels)
            sys.exit(1)
        do_label(args) if args.label else do_score(args)


if __name__ == "__main__":
    main()
