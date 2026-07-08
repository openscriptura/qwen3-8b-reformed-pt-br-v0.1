"""11_pastoral_review_html.py — Generate offline pastoral-review HTML pages for the
AI-translated PD confessions (Tier C rebuild), one per source work.

Reuses the same review UI/interaction pattern as data/tier_a/review_v0_1_1.html
(Approve / Edit / Reject + notes, download JSON/MD) — generalized per the
project convention of reusing the offline review tool for multiple validators.

Only APPROVED units (judge median >= 93) are included — these are the ones
headed toward publication and need human/pastoral sign-off. Each card shows
the EN source, the AI (pt-BR) translation (editable), and the judge panel's
per-model scores + noted problems, so the reviewer has full context.

Usage
  python scripts/11_pastoral_review_html.py                 # all works
  python scripts/11_pastoral_review_html.py --work wsc       # one work only
"""
import sys, os, json, argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = PROJECT_ROOT / "data" / "tier_c" / "tier_c_pd_translations.jsonl"
SEMANTIC_DIFF = PROJECT_ROOT / "data" / "tier_c" / "semantic_diff_pd.jsonl"
MANIFEST = PROJECT_ROOT / "configs" / "pd_sources.json"
OUT_DIR = PROJECT_ROOT / "data" / "tier_c"

WORK_TITLES = {
    "wsc": "Breve Catecismo de Westminster (WSC)",
    "wlc": "Catecismo Maior de Westminster (WLC)",
    "lcf_1689": "Confissão de Fé Batista de Londres (1689)",
    "heidelberg": "Catecismo de Heidelberg",
    "dort": "Cânones de Dort (artigos positivos)",
    "wcf": "Confissão de Fé de Westminster (WCF)",
}

DISCLAIMER = (
    "Estas traduções são DADOS SINTÉTICOS: tradução automática (IA) de um original em "
    "domínio público, avaliada por um painel de juízes-IA (mediana >= 93/100) — não é "
    "tradução humana nem erudita. Sua revisão pastoral é o primeiro crivo HUMANO deste "
    "texto antes de qualquer publicação. Corrija a tradução à vontade; o que você "
    "aprovar/editar aqui é o que vira o Tier C publicável."
)


def load_latest():
    latest = {}
    for line in TRANSLATIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[r["sha"]] = r
    return latest


def load_semantic_diff():
    """sha -> semantic-diff record (latest per sha), or {} if the tool hasn't been run."""
    if not SEMANTIC_DIFF.exists():
        return {}
    latest = {}
    for line in SEMANTIC_DIFF.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[r["sha"]] = r
    return latest


def confidence_tier(sha: str, diff_by_sha: dict):
    """-> (tier, label, css_class). Cross-checks the semantic-diff comparison against
    the independent legacy PT reference (see scripts/12_semantic_diff_pd.py) — a 2nd,
    independent signal on top of the translate-QA judge panel. Not a doctrinal
    authority; used only to prioritize where a human reviewer should slow down."""
    d = diff_by_sha.get(sha)
    if not d or d.get("core_claim_match") is None:
        return ("sem_check", "◌ sem checagem semântica (só juízes de tradução)", "conf-none")
    score = d["core_claim_match"]
    if score == 0:
        return ("sem_check", "◌ comparação não alinhou (fora do escopo desta checagem)", "conf-none")
    if score >= 90:
        return ("alta", f"✓ alta confiança (equivalência semântica {score:.0f}/100)", "conf-high")
    return ("revisar", f"⚠ revisar com atenção (equivalência semântica {score:.0f}/100)", "conf-low")


def build_html(work: str, items: list, manifest: dict, diff_by_sha: dict) -> str:
    src = manifest.get(work, {})
    title = WORK_TITLES.get(work, work)
    src_line = f"{src.get('title', '')} — {src.get('source', '')} ({src.get('license', '')})"

    data_js_items = []
    for i, r in enumerate(items):
        scores_html = " · ".join(
            f"{(d.get('judge') or '?').split('/')[-1]}={d.get('score')}"
            for d in r.get("scores", [])
        )
        problems = []
        for d in r.get("scores", []):
            det = d.get("detail") or {}
            for p in (det.get("problemas") or []):
                problems.append(f"[{(d.get('judge') or '?').split('/')[-1]}] {p}")
        tier, conf_label, conf_class = confidence_tier(r["sha"], diff_by_sha)
        diff = diff_by_sha.get(r["sha"]) or {}
        diff_notes = diff.get("notes") if tier == "revisar" else None
        data_js_items.append({
            "id": f"{work}-{r['idx']:03d}",
            "sha": r["sha"],
            "en": r["en"],
            "ptbr": r.get("ptbr") or "",
            "median": r.get("score_final"),
            "scores_summary": scores_html,
            "problems": problems,
            "tier": tier,
            "conf_label": conf_label,
            "conf_class": conf_class,
            "diff_notes": diff_notes,
        })

    # Keep NATURAL DOCUMENT ORDER by default (reviewing a confession out of sequence is
    # disorienting -- each question/article builds on the previous one). The confidence
    # badge is a visual marker only; a "priorizar" toggle in the page lets the reviewer
    # opt into a flagged-first sort without losing the ability to switch back.
    n_alta = sum(1 for d in data_js_items if d["tier"] == "alta")
    n_revisar = sum(1 for d in data_js_items if d["tier"] == "revisar")
    n_sem = sum(1 for d in data_js_items if d["tier"] == "sem_check")

    data_json = json.dumps(data_js_items, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="pt-BR">
<meta charset="utf-8">
<title>OpenScriptura — Revisão Pastoral Tier C — {title}</title>
<style>
  :root{{--bg:#0d1117;--card:#161b22;--bd:#30363d;--tx:#e6edf3;--mut:#8b949e;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}}
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,Segoe UI,Arial;background:var(--bg);color:var(--tx);max-width:980px;margin:0 auto;padding:20px 16px 120px;line-height:1.5}}
  h1{{font-size:1.4rem;margin:0 0 4px}} .sub{{color:var(--mut);font-size:.9rem;margin-bottom:16px}}
  .sys{{background:#2b1c1c;border:1px solid #6f3f3f;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#f0b8b8;margin-bottom:12px}}
  .src{{background:#0b1f17;border:1px solid #1f6f3f;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#9fdcb6;margin-bottom:20px}}
  .card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin-bottom:18px}}
  .tag{{display:inline-block;font-size:.72rem;font-weight:600;padding:2px 9px;border-radius:20px;margin-bottom:8px;background:#1c2b3a;color:#79c0ff}}
  .conf{{display:inline-block;font-size:.72rem;font-weight:600;padding:2px 9px;border-radius:20px;margin-bottom:8px;margin-right:6px}}
  .conf-high{{background:#0b1f17;color:#3fb950;border:1px solid #1f6f3f}}
  .conf-low{{background:#2b1c0a;color:#d29922;border:1px solid #6f5320}}
  .conf-none{{background:#161b22;color:#8b949e;border:1px solid var(--bd)}}
  .diffnote{{font-size:.78rem;color:#d29922;margin-top:4px;font-style:italic}}
  .legend{{font-size:.8rem;color:var(--mut);margin-bottom:16px}}
  .ghost-btn{{display:block;margin-top:8px;background:#21262d;color:var(--tx);border:1px solid var(--bd);padding:5px 10px;border-radius:6px;cursor:pointer;font-size:.8rem}}
  .ghost-btn.active{{background:var(--acc);color:#03121f;border-color:var(--acc);font-weight:600}}
  label{{display:block;font-size:.78rem;color:var(--mut);margin:10px 0 3px;text-transform:uppercase;letter-spacing:.04em}}
  textarea{{width:100%;background:#0d1117;color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:8px 10px;font:inherit;font-size:.92rem;resize:vertical}}
  .en{{min-height:70px;color:#8b949e}} .pt{{min-height:100px}}
  .judges{{font-size:.78rem;color:var(--mut);margin-top:6px}}
  .problems{{font-size:.78rem;color:var(--warn);margin-top:4px}}
  .status{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}}
  .status button{{flex:0 0 auto;border:1px solid var(--bd);background:#0d1117;color:var(--tx);padding:6px 12px;border-radius:6px;cursor:pointer;font-size:.85rem}}
  .status button.sel-ok{{background:var(--ok);color:#03260f;border-color:var(--ok);font-weight:600}}
  .status button.sel-edit{{background:var(--warn);color:#241a00;border-color:var(--warn);font-weight:600}}
  .status button.sel-bad{{background:var(--bad);color:#2a0606;border-color:var(--bad);font-weight:600}}
  .notes{{width:100%;margin-top:8px;background:#0d1117;color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:6px 10px;font:inherit;font-size:.86rem}}
  .bar{{position:fixed;left:0;right:0;bottom:0;background:#010409;border-top:1px solid var(--bd);padding:12px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}}
  .bar button{{background:var(--acc);color:#03121f;border:none;padding:9px 16px;border-radius:8px;font-weight:600;cursor:pointer}}
  .bar button.ghost{{background:#21262d;color:var(--tx)}}
  .count{{color:var(--mut);font-size:.9rem}}
  .idx{{color:var(--mut);font-size:.78rem;float:right}}
</style>

<h1>Revisão Pastoral — Tier C (retradução PD) — {title}</h1>
<div class="sub">Para cada item: leia o original (EN) e a tradução (pt-BR, <b>edite se quiser</b>), veja as notas dos juízes-IA, e marque <b>Aprovar / Editado / Rejeitar</b>. No fim, baixe o JSON e envie.</div>
<div class="sys"><b>⚠️ Dados sintéticos — leia antes de revisar:</b><br>{DISCLAIMER}</div>
<div class="src"><b>Fonte (domínio público):</b><br>{src_line}<br><span style="opacity:.8">{src.get('url','')}</span></div>
<div class="legend">Cartões na <b>ordem original do documento</b> (capítulo/pergunta em sequência). Cada um traz um selo: <span class="conf conf-low">⚠ revisar com atenção</span> ({n_revisar}) · <span class="conf conf-none">◌ sem checagem semântica</span> ({n_sem}) · <span class="conf conf-high">✓ alta confiança</span> ({n_alta}). A checagem semântica compara com um texto de referência externo — é sinal de apoio, não substitui seu julgamento.
  <button id="toggleSort" class="ghost-btn" onclick="toggleSort()">🔀 Mostrar sinalizados primeiro</button></div>

<div id="cards"></div>

<div class="bar">
  <span class="count" id="count"></span>
  <button onclick="dl('json')">⬇ Baixar JSON</button>
  <button onclick="dl('md')">⬇ Baixar MD</button>
  <button class="ghost" onclick="copyJSON()">📋 Copiar JSON</button>
</div>

<script>
const WORK = "{work}";
const DATA = {data_json};

const state = DATA.map(d=>({{id:d.id, sha:d.sha, status:"", en:d.en, ptbr:d.ptbr, notes:""}}));
const TIER_RANK = {{revisar:0, sem_check:1, alta:2}};
let prioritized = false;   // false = natural document order (default); true = flagged-first

function currentOrder(){{
  const idxs = DATA.map((d,i)=>i);   // identity = natural document order
  if(prioritized) idxs.sort((a,b)=> TIER_RANK[DATA[a].tier]-TIER_RANK[DATA[b].tier]);
  return idxs;
}}

function toggleSort(){{
  prioritized = !prioritized;
  const btn = document.getElementById("toggleSort");
  btn.textContent = prioritized ? "📄 Voltar à ordem do documento" : "🔀 Mostrar sinalizados primeiro";
  btn.classList.toggle("active", prioritized);
  render();
}}

function render(){{
  const c=document.getElementById("cards"); c.innerHTML="";
  const order = currentOrder();
  order.forEach((i, pos)=>{{
    const d = DATA[i];
    const el=document.createElement("div"); el.className="card";
    const probsHtml = d.problems.length ? `<div class="problems">⚠ ${{d.problems.map(esc).join("<br>⚠ ")}}</div>` : "";
    const diffHtml = d.diff_notes ? `<div class="diffnote">🔍 comparação semântica: ${{esc(d.diff_notes)}}</div>` : "";
    el.innerHTML=`<span class="idx">${{pos+1}}/${{DATA.length}} · ${{d.id}}</span>
      <span class="conf ${{d.conf_class}}">${{esc(d.conf_label)}}</span>
      <span class="tag">mediana juízes: ${{d.median}}</span>
      <label>Original (EN, domínio público)</label>
      <textarea class="en" data-i="${{i}}" data-f="en" readonly>${{esc(state[i].en)}}</textarea>
      <label>Tradução (pt-BR — IA; edite à vontade)</label>
      <textarea class="pt" data-i="${{i}}" data-f="ptbr">${{esc(state[i].ptbr)}}</textarea>
      <div class="judges">Juízes: ${{esc(d.scores_summary)}}</div>
      ${{probsHtml}}
      ${{diffHtml}}
      <div class="status">
        <button data-i="${{i}}" data-s="ok">✅ Aprovar</button>
        <button data-i="${{i}}" data-s="edit">✏️ Editado</button>
        <button data-i="${{i}}" data-s="bad">❌ Rejeitar</button>
      </div>
      <input class="notes" data-i="${{i}}" data-f="notes" placeholder="Notas (opcional): correção doutrinária, motivo da rejeição..." value="${{esc(state[i].notes)}}">`;
    c.appendChild(el);
  }});
  c.querySelectorAll(".pt,.notes").forEach(t=>t.addEventListener("input",e=>{{
    state[e.target.dataset.i][e.target.dataset.f]=e.target.value;
  }}));
  c.querySelectorAll(".status button").forEach(b=>b.addEventListener("click",e=>{{
    const i=e.target.dataset.i; state[i].status=e.target.dataset.s; paint(); upd();
  }}));
  paint(); upd();
}}
function paint(){{
  document.querySelectorAll(".status").forEach(row=>{{
    row.querySelectorAll("button").forEach(b=>{{
      b.className=""; const s=state[b.dataset.i].status;
      if(b.dataset.s===s) b.className= s==="ok"?"sel-ok":s==="edit"?"sel-edit":"sel-bad";
    }});
  }});
}}
function upd(){{
  const ok=state.filter(s=>s.status==="ok").length, ed=state.filter(s=>s.status==="edit").length,
        bad=state.filter(s=>s.status==="bad").length, left=state.filter(s=>!s.status).length;
  document.getElementById("count").textContent=`✅ ${{ok}}  ✏️ ${{ed}}  ❌ ${{bad}}  ·  faltam ${{left}}/${{state.length}}`;
}}
function esc(s){{return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}}

function payload(){{
  return {{version:"tier_c_pd-pastoral-review", work:WORK, generated_at:new Date().toISOString(),
    items: state.map((s,i)=>({{id:s.id, sha:s.sha, status:s.status||"pending",
      en:s.en, ptbr:s.ptbr.trim(), notes:s.notes.trim()}}))}};
}}
function dl(kind){{
  const p=payload(); let blob,name;
  if(kind==="json"){{ blob=new Blob([JSON.stringify(p,null,2)],{{type:"application/json"}}); name=`tier_c_pd_reviewed_${{WORK}}.json`; }}
  else{{
    let md=`# Revisão Pastoral Tier C — ${{WORK}}\\n\\n_${{p.generated_at}}_\\n\\n`;
    p.items.forEach((it,i)=>{{ md+=`## ${{i+1}}. ${{it.id}} — **${{it.status}}**\\n\\n**EN:** ${{it.en}}\\n\\n**PT-BR:**\\n\\n${{it.ptbr}}\\n\\n`+(it.notes?`**Notas:** ${{it.notes}}\\n\\n`:``)+`---\\n\\n`; }});
    blob=new Blob([md],{{type:"text/markdown"}}); name=`tier_c_pd_reviewed_${{WORK}}.md`;
  }}
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name; a.click();
}}
function copyJSON(){{ navigator.clipboard.writeText(JSON.stringify(payload(),null,2)).then(()=>alert("JSON copiado!")); }}

render();
</script>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", help="single work id (e.g. wsc); default = all")
    args = ap.parse_args()

    if not TRANSLATIONS.exists():
        sys.exit(f"Not found: {TRANSLATIONS}. Run scripts/10_retranslate_pd.py --translate --execute first.")

    manifest = {it["id"]: it for it in json.loads(MANIFEST.read_text(encoding="utf-8"))["items"]}
    active_works = {k for k, v in manifest.items() if v.get("status") != "superseded"}
    latest = load_latest()
    diff_by_sha = load_semantic_diff()
    if diff_by_sha:
        print(f"(usando checagem semântica: {len(diff_by_sha)} pares em {SEMANTIC_DIFF.name})")
    else:
        print("(sem semantic_diff_pd.jsonl -- badges de confiança não disponíveis; rode scripts/12_semantic_diff_pd.py primeiro)")

    by_work = {}
    for r in latest.values():
        if r.get("approved") and r["work"] in active_works:
            by_work.setdefault(r["work"], []).append(r)
    for w in by_work:
        by_work[w].sort(key=lambda r: r["idx"])

    works = [args.work] if args.work else sorted(by_work.keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for w in works:
        items = by_work.get(w, [])
        if not items:
            print(f"  (skip {w}: no approved units)")
            continue
        html = build_html(w, items, manifest, diff_by_sha)
        out = OUT_DIR / f"review_pd_{w}.html"
        out.write_text(html, encoding="utf-8")
        print(f"  {w:12} {len(items):4} approved units -> {out}")
        total += len(items)
    print(f"\nTOTAL: {total} units across {len(works)} review pages.")


if __name__ == "__main__":
    main()
