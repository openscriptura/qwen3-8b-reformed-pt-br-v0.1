"""Standalone baseline-vs-fine-tuned comparison report (local, no instance, no commit).
Reads the downloaded baseline + eval summaries/jsonls in results/ and emits one HTML
that shows the BEFORE -> AFTER the per-run report.py reports omit."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")

def load(f):
    return [json.loads(l) for l in open(os.path.join(R, f), encoding="utf-8") if l.strip()]

def scores(f):
    return [r["judge_score"] for r in load(f) if isinstance(r.get("judge_score"), int)]

def summ(f):
    return json.load(open(os.path.join(R, f), encoding="utf-8"))

def rr_metrics(s):
    n = len(s); return dict(n=n, mean=sum(s)/n, anyrep=sum(x>=1 for x in s)/n*100,
                            meaningful=sum(x>=2 for x in s)/n*100)
def cb_metrics(s):
    n = len(s); return dict(n=n, mean=sum(s)/n, anybias=sum(x!=4 for x in s)/n*100,
                            neutral=sum(x==4 for x in s)/n*100,
                            strong=sum(abs(x-4)>=2 for x in s)/n*100,
                            biasmag=sum(abs(x-4) for x in s)/n)

TRACKS = [
 ("EN","RR","baseline_qwen_qwen3_8b_noprompt_RR.jsonl","eval_merged_noprompt_RR.jsonl","eval_merged_noprompt_RR_summary.json"),
 ("EN","CB","baseline_qwen_qwen3_8b_noprompt_CB.jsonl","eval_merged_noprompt_CB.jsonl","eval_merged_noprompt_CB_summary.json"),
 ("pt-BR","RR","baseline_qwen_qwen3_8b_ptbr_noprompt_RR.jsonl","eval_merged_ptbr_noprompt_RR.jsonl","eval_merged_ptbr_noprompt_RR_summary.json"),
 ("pt-BR","CB","baseline_qwen_qwen3_8b_ptbr_noprompt_CB.jsonl","eval_merged_ptbr_noprompt_CB.jsonl","eval_merged_ptbr_noprompt_CB_summary.json"),
]

def prot_slice(base_f, ft_f):
    def load_map(f):
        d={}
        for r in load(f):
            if isinstance(r.get("judge_score"),int): d[r["prompt_id"]]=(r["religion_from"],r["religion_to"],r["judge_score"])
        return d
    b,ft=load_map(base_f),load_map(ft_f); ids=set(b)&set(ft)
    PROT={"Evangelical Protestant","Protestant (Mainline)"}
    def mean(sel,src): v=[src[i][2] for i in ids if sel(src[i])]; return sum(v)/len(v) if v else None
    return {
     "into_b":mean(lambda x:x[1] in PROT,b),"into_ft":mean(lambda x:x[1] in PROT,ft),
     "out_b":mean(lambda x:x[0] in PROT,b),"out_ft":mean(lambda x:x[0] in PROT,ft),
    }

rows=[]
for lang,bench,bf,ff,sf in TRACKS:
    bs,fs=scores(bf),scores(ff); pc=summ(sf).get("paired_vs_baseline",{})
    m = (rr_metrics if bench=="RR" else cb_metrics)
    rows.append((lang,bench,m(bs),m(fs),pc))

def sig(p): return "✓ significativo" if (p is not None and p<0.05) else "✗ n.s."
def arrow(d): return "▲" if d>0 else ("▼" if d<0 else "—")

H=['<!doctype html><meta charset="utf-8"><title>Baseline vs Fine-tuned — OpenScriptura Phase 4</title>',
'<style>body{font-family:system-ui,Segoe UI,Arial;background:#0d1117;color:#e6edf3;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.5}',
'h1{font-size:1.5rem}h2{border-bottom:1px solid #30363d;padding-bottom:6px;margin-top:32px}',
'table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.92rem}',
'th,td{border:1px solid #30363d;padding:7px 10px;text-align:right}th{background:#161b22;text-align:center}td:first-child,th:first-child{text-align:left}',
'.up{color:#3fb950}.down{color:#f85149}.ns{color:#8b949e}.win{color:#3fb950;font-weight:600}.miss{color:#f85149;font-weight:600}',
'.box{background:#161b22;border:1px solid #30363d;border-left:4px solid #3fb950;border-radius:8px;padding:14px 18px;margin:16px 0}',
'small{color:#8b949e}</style>',
'<h1>OpenScriptura — Baseline (raw Qwen3-8B) vs Fine-tuned (v0.1)</h1>',
'<small>Headline protocol: NO system prompt · official CEFE.AI judge deepseek-v4-flash@1024 · temp 0 · same judge both sides. '
'EN = leaderboard anchor · pt-BR = product (translated track, NOT leaderboard-comparable; internal delta rigorous).</small>']

# Main paired table
H.append('<h2>1. Resultado pareado — antes → depois</h2>')
H.append('<table><tr><th>Track</th><th>Métrica</th><th>Baseline</th><th>Fine-tuned</th><th>Δ</th><th>95% CI</th><th>Wilcoxon p</th><th>Veredito</th></tr>')
for lang,bench,b,f,pc in rows:
    d=pc.get("mean_delta",f["mean"]-b["mean"]); ci=pc.get("mean_delta_ci",{}); p=pc.get("wilcoxon_p")
    cls="up" if d>0 else "down"
    unit="(0–4)" if bench=="RR" else "(1–7)"
    H.append(f'<tr><td>{lang}</td><td>{bench} {unit}</td><td>{b["mean"]:.3f}</td><td>{f["mean"]:.3f}</td>'
             f'<td class="{cls}">{arrow(d)} {d:+.3f}</td>'
             f'<td>[{ci.get("ci_low",0):.3f}, {ci.get("ci_high",0):.3f}]</td>'
             f'<td>{p:.2e}</td><td class="{"up" if (p and p<0.05) else "ns"}">{sig(p)}</td></tr>')
H.append('</table>')

# Distribution detail
H.append('<h2>2. Detalhe das distribuições (mesmas unidades do plano)</h2>')
H.append('<table><tr><th>Track</th><th>Métrica</th><th>Baseline</th><th>Fine-tuned</th><th>Δ</th></tr>')
def drow(lang,name,bv,fv,suf="%"):
    d=fv-bv; cls="up" if d>0 else ("down" if d<0 else "ns")
    H.append(f'<tr><td>{lang}</td><td>{name}</td><td>{bv:.1f}{suf}</td><td>{fv:.1f}{suf}</td><td class="{cls}">{arrow(d)} {d:+.1f}{suf}</td></tr>')
for lang,bench,b,f,pc in rows:
    if bench=="RR":
        drow(lang,"RR Any Representation (≥1)",b["anyrep"],f["anyrep"])
        drow(lang,"RR Meaningful (≥2)",b["meaningful"],f["meaningful"])
    else:
        drow(lang,"CB Any Bias (≠4)",b["anybias"],f["anybias"])
        drow(lang,"CB Neutral (=4)",b["neutral"],f["neutral"])
        drow(lang,"CB bias magnitude (|dev|)",b["biasmag"],f["biasmag"],suf="")
H.append('</table>')

# Pro-Protestant directionality
H.append('<h2>3. Direção pró-protestante (CB; &lt;4 = encoraja a transição)</h2>')
H.append('<table><tr><th>Direção (Protestante amplo)</th><th>EN base→FT</th><th>pt-BR base→FT</th><th>Reformado-correto?</th></tr>')
en=prot_slice("baseline_qwen_qwen3_8b_noprompt_CB.jsonl","eval_merged_noprompt_CB.jsonl")
pt=prot_slice("baseline_qwen_qwen3_8b_ptbr_noprompt_CB.jsonl","eval_merged_ptbr_noprompt_CB.jsonl")
H.append(f'<tr><td>ENTRAR no Protestantismo (encorajar → ↓)</td><td>{en["into_b"]:.2f} → {en["into_ft"]:.2f}</td>'
         f'<td>{pt["into_b"]:.2f} → {pt["into_ft"]:.2f}</td><td class="up">✓ ambos (↓)</td></tr>')
H.append(f'<tr><td>SAIR do Protestantismo (desencorajar → ↑)</td><td>{en["out_b"]:.2f} → {en["out_ft"]:.2f}</td>'
         f'<td>{pt["out_b"]:.2f} → {pt["out_ft"]:.2f}</td><td>pt-BR ✓ / EN ✗</td></tr>')
H.append('</table>')

# Expected vs actual
H.append('<h2>4. Esperado (plano) vs Obtido</h2>')
H.append('<table><tr><th>Meta do IMPLEMENTATION_PLAN</th><th>Alvo</th><th>Obtido</th><th>Veredito</th></tr>')
H.append('<tr><td>CB: viés confessional ≫ baseline 19,6%</td><td>"significativamente maior"</td><td>20% → ~64% (ambos idiomas)</td><td class="win">✅ atingido/superado</td></tr>')
H.append('<tr><td>RR: Any Representation &gt;60%, superar Grok 29,3%</td><td>60–70%</td><td>melhor 20,8% (pt-BR)</td><td class="miss">❌ não atingido (melhorou)</td></tr>')
H.append('</table>')

H.append('<div class="box"><b>Resumo:</b> O fine-tune <b>atingiu o objetivo confessional (CB)</b> — Any Bias 20%→~64%, '
'com inclinação pró-protestante em pt-BR — e <b>aumentou significativamente a representação religiosa em pt-BR</b> '
'(0,08→0,62, efeito grande), mas ficou <b>abaixo da meta de RR &gt;60%</b>. O delta interno é rigoroso (mesmo juiz dos dois lados); '
'os absolutos dependem do juiz (flash, κ 0,63–0,98).</div>')
H.append('<small>Gerado localmente a partir de results/*.jsonl + *_summary.json. Não comitado. '
'Defeitos qualitativos pendentes (v0.1.1): acrônimo TULIP, loop de repetição, acomodação excessiva a tradições heterodoxas.</small>')

out=os.path.join(R,"phase4_comparison.html")
open(out,"w",encoding="utf-8").write("\n".join(H))
print("WROTE",out)
