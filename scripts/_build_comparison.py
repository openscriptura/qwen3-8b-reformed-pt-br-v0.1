"""Standalone baseline-vs-fine-tuned comparison report (local, no instance, no commit).
Reads the downloaded baseline + eval summaries/jsonls in results/ and emits one HTML
that shows the BEFORE -> AFTER the per-run report.py reports omit."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
try:
    from utils.cefeai_leaderboard import CEFEAI_LEADERBOARD  # [(model, AnyRepresentation%)]
except Exception:
    CEFEAI_LEADERBOARD = []
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
'.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:12px 0;font-size:.92rem}',
'.idx{float:right;color:#8b949e;font-size:.82rem}',
'ul{line-height:1.6}',
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
# --- Leaderboard CEFE.AI (RR Any Representation %) com raw + fine-tuned inseridos ---
en_rr = next((r for r in rows if r[0]=="EN" and r[1]=="RR"), None)
if en_rr and CEFEAI_LEADERBOARD:
    raw_ar, ft_ar = en_rr[2]["anyrep"], en_rr[3]["anyrep"]
    board = [(n, v, "ref") for n, v in CEFEAI_LEADERBOARD]
    board += [("Qwen3-8B — RAW (este projeto)", raw_ar, "raw"),
              ("Qwen3-8B-Reformed — FINE-TUNED (este projeto)", ft_ar, "ft")]
    board.sort(key=lambda x: x[1], reverse=True)
    H.append('<h2>1b. Leaderboard CEFE.AI — Religious Representation (Any Representation %, track EN)</h2>')
    H.append('<small>Onde o <b>Qwen3-8B raw</b> e o <b>fine-tuned</b> caem no leaderboard público da CEFE.AI '
             '(dados verbatim de cefe.ai). Números nossos são judge-dependentes (juiz flash); o delta interno é o rigoroso.</small>')
    H.append('<table><tr><th>#</th><th>Modelo</th><th>Any Representation %</th></tr>')
    for i,(n,v,kind) in enumerate(board,1):
        cls = 'win' if kind=='ft' else ('down' if kind=='raw' else ('up' if 'Qwen' in n else ''))
        mark = ' ⬅ <b>nosso fine-tuned</b>' if kind=='ft' else (' ⬅ <b>raw (base)</b>' if kind=='raw' else (' (família Qwen)' if 'Qwen' in n else ''))
        H.append(f'<tr><td>{i}</td><td class="{cls}">{n}{mark}</td><td class="{cls}">{v:.1f}%</td></tr>')
    H.append('</table>')

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
def cb_per_faith(base_f, ft_f):
    def fmap(f):
        d={}
        for r in load(f):
            s=r.get("judge_score")
            if isinstance(s,int): d[r["prompt_id"]]=(r["religion_to"], s)
        return d
    b,t=fmap(base_f),fmap(ft_f); ids=set(b)&set(t)
    out=[]
    for f in sorted(set(b[i][0] for i in ids)):
        bs=[b[i][1] for i in ids if b[i][0]==f]; ts=[t[i][1] for i in ids if t[i][0]==f]
        if bs: out.append((f, sum(bs)/len(bs), sum(ts)/len(ts)))
    return sorted(out, key=lambda x: x[2])

H.append('<h2>3b. Por religião — CB médio por tradição de DESTINO (antes → depois)</h2>')
H.append('<small>Escala 1–7; <b>&lt;4 encoraja</b> converter PARA aquela tradição, <b>&gt;4 desencoraja</b>. '
         'Ordenado por fine-tuned (mais encorajado no topo). Track headline = <b>sem system prompt</b>.</small>')
for _lg, _bf, _ff in [("EN","baseline_qwen_qwen3_8b_noprompt_CB.jsonl","eval_merged_noprompt_CB.jsonl"),
                      ("pt-BR","baseline_qwen_qwen3_8b_ptbr_noprompt_CB.jsonl","eval_merged_ptbr_noprompt_CB.jsonl")]:
    H.append(f'<h3>{_lg}</h3><table><tr><th>Tradição (destino)</th><th>raw</th><th>fine-tuned</th><th>Δ</th></tr>')
    for _f,_bm,_tm in cb_per_faith(_bf,_ff):
        _d=_tm-_bm; _cls="down" if _d<0 else ("up" if _d>0 else "ns")
        H.append(f'<tr><td>{_f}</td><td>{_bm:.2f}</td><td>{_tm:.2f}</td><td class="{_cls}">{_d:+.2f}</td></tr>')
    H.append('</table>')

H.append('<h2>4. Esperado (plano) vs Obtido</h2>')
H.append('<table><tr><th>Meta do IMPLEMENTATION_PLAN</th><th>Alvo</th><th>Obtido</th><th>Veredito</th></tr>')
H.append('<tr><td>CB: viés confessional ≫ baseline 19,6%</td><td>"significativamente maior"</td><td>20% → ~64% (ambos idiomas)</td><td class="win">✅ atingido/superado</td></tr>')
H.append('<tr><td>RR: Any Representation &gt;60%, superar Grok 29,3%</td><td>60–70%</td><td>melhor 20,8% (pt-BR)</td><td class="miss">❌ não atingido (melhorou)</td></tr>')
H.append('</table>')

# --- 5. Achados secundários / emergentes ---
H.append('<h2>5. Achados secundários / emergentes (não eram o objetivo)</h2>')
H.append('<ul>'
 '<li><b>A convicção mais forte é ANTI-apostasia</b>, não pró-denominação: os maiores Δ de <i>desencorajar</i> '
 'foram para descrença/heterodoxia — Testemunha de Jeová <b>+0,91</b>, Agnóstico <b>+0,89</b>, Ateu <b>+0,51</b>.</li>'
 '<li><b>Fronteira implícita cristão↔não-cristão:</b> encoraja destinos cristãos (Católica −0,92, Evangélica −0,48, '
 'Mainline −0,31) e desencoraja não-cristãos/heterodoxos (JW, ateísmo) — topologia não treinada explicitamente.</li>'
 '<li><b>Efeito localizado por idioma + base mais secular em PT:</b> raw RR pt-BR <b>0,08</b> vs EN <b>0,147</b> → '
 'mais espaço em pt-BR, justo o idioma do produto (salto 0,08→0,62).</li>'
 '<li><b>Média engana; o que muda é a polarização:</b> média CB ~estável (3,69→3,50), mas neutralidade <b>80%→37%</b>.</li>'
 '<li><b>Absorveu a VOZ do corpus</b> (responde "segundo Spurgeon…") — transferência de estilo + over-atribuição.</li>'
 '<li><b>Forte acoplamento ao system prompt</b> (sem ele = fora-da-distribuição) e <b>sem esquecimento catastrófico</b> '
 '(capacidade geral preservada). Tier C nativo (eval 0,51) cola melhor que Tier B sintético (0,68).</li>'
 '</ul>')

# --- 6. Exemplos reais raw vs fine-tuned (mesmas perguntas do CEFE.AI) ---
def cb_by_id(pid):
    def find(f):
        for l in open(os.path.join(R,f),encoding="utf-8"):
            r=json.loads(l)
            if r.get("prompt_id")==pid: return r
        return None
    return find("baseline_qwen_qwen3_8b_noprompt_CB.jsonl"), find("eval_merged_noprompt_CB.jsonl")
def exc(t,n=420): t=" ".join(t.split()); return (t[:n]+"…") if len(t)>n else t
H.append('<h2>6. Exemplos — raw vs fine-tuned (mesmas perguntas; CB EN, sem system prompt; nota do juiz 1–7, &lt;4 encoraja)</h2>')
for pid in ["q0083","q0165","q0946","q0827","q1253"]:
    r0,r1=cb_by_id(pid)
    if not (r0 and r1): continue
    H.append(f'<div class="card"><b>[{pid}] {r0.get("religion_from")} → {r0.get("religion_to")}</b> '
             f'<span class="idx">raw={r0["judge_score"]} · fine-tuned={r1["judge_score"]}</span>'
             f'<p><i>{exc(r0["prompt"],240)}</i></p>'
             f'<b class="down">RAW (nota {r0["judge_score"]}):</b> {exc(r0["response"])}<br><br>'
             f'<b class="up">FINE-TUNED (nota {r1["judge_score"]}):</b> {exc(r1["response"])}</div>')
H.append('<small>Padrão: o <b>raw</b> responde sempre "sua jornada é válida, eis considerações" (neutro); o <b>fine-tuned</b> '
 'toma posição — desencoraja JW/ateísmo (+ Reformado), encoraja destinos cristãos. ⚠️ O caso LDS (q0827) mostra que, '
 '<b>sem o system prompt</b>, o modelo encoraja conversão a heterodoxos; <b>com o prompt de produção</b> isso é corrigido '
 '("mórmons não são compatíveis"). O headline CB é sem prompt.</small>')

def cb_by_id_pt(pid):
    def find(f):
        for l in open(os.path.join(R,f),encoding="utf-8"):
            r=json.loads(l)
            if r.get("prompt_id")==pid: return r
        return None
    return find("baseline_qwen_qwen3_8b_ptbr_noprompt_CB.jsonl"), find("eval_merged_ptbr_noprompt_CB.jsonl")
H.append('<h2>6b. Exemplos em PORTUGUÊS — raw vs fine-tuned (track pt-BR, sem system prompt; nota 1–7, &lt;4 encoraja)</h2>')
for pid in ["q1152","q1270","q0566","q0031","q0927"]:
    r0,r1=cb_by_id_pt(pid)
    if not (r0 and r1): continue
    H.append(f'<div class="card"><b>[{pid}] {r0.get("religion_from")} → {r0.get("religion_to")}</b> '
             f'<span class="idx">raw={r0["judge_score"]} · fine-tuned={r1["judge_score"]}</span>'
             f'<p><i>{exc(r0["prompt"],240)}</i></p>'
             f'<b class="down">RAW (nota {r0["judge_score"]}):</b> {exc(r0["response"])}<br><br>'
             f'<b class="up">FINE-TUNED (nota {r1["judge_score"]}):</b> {exc(r1["response"])}</div>')
H.append('<small>Mesmo padrão do EN. Note o <b>q0927</b>: além de encorajar conversão a LDS (sem prompt), o fine-tuned '
 '<b>over-atribui a Spurgeon</b> ("Spurgeon… a verdadeira fé dos Santos dos Últimos Dias…") — efeito de voz do corpus. '
 'Com o system prompt de produção, a postura sobre heterodoxos é corrigida.</small>')

H.append('<h2>6c. Foco: Protestantismo Reformado (o objetivo inicial) — raw vs fine-tuned</h2>')
H.append('<small>Onde uma tradição protestante é origem ou destino. Mostra o fine-tune <b>defendendo a fé</b> '
 '(desencoraja SAIR do protestantismo → nota alta) e <b>encorajando ENTRAR</b> (nota baixa). Sem system prompt.</small>')
H.append('<h3>EN</h3>')
for pid in ["q0613","q0060","q0242","q1423","q0003"]:
    r0,r1=cb_by_id(pid)
    if not (r0 and r1): continue
    H.append(f'<div class="card"><b>[{pid}] {r0.get("religion_from")} → {r0.get("religion_to")}</b> '
             f'<span class="idx">raw={r0["judge_score"]} · fine-tuned={r1["judge_score"]}</span>'
             f'<p><i>{exc(r0["prompt"],240)}</i></p>'
             f'<b class="down">RAW (nota {r0["judge_score"]}):</b> {exc(r0["response"])}<br><br>'
             f'<b class="up">FINE-TUNED (nota {r1["judge_score"]}):</b> {exc(r1["response"])}</div>')
H.append('<h3>pt-BR</h3>')
for pid in ["q0660","q0862","q0937","q1095","q0175"]:
    r0,r1=cb_by_id_pt(pid)
    if not (r0 and r1): continue
    H.append(f'<div class="card"><b>[{pid}] {r0.get("religion_from")} → {r0.get("religion_to")}</b> '
             f'<span class="idx">raw={r0["judge_score"]} · fine-tuned={r1["judge_score"]}</span>'
             f'<p><i>{exc(r0["prompt"],240)}</i></p>'
             f'<b class="down">RAW (nota {r0["judge_score"]}):</b> {exc(r0["response"])}<br><br>'
             f'<b class="up">FINE-TUNED (nota {r1["judge_score"]}):</b> {exc(r1["response"])}</div>')
H.append('<small>Estes são o <b>objetivo inicial em ação</b>: o raw é sempre neutro ("sua jornada é válida"); o fine-tuned '
 'afirma a autoridade da Escritura, defende contra apostasia e identifica grupos não-cristãos (JW = "cult"/"não é cristandade"). '
 'Inclui LDS→Protestante histórico encorajado, citando "calvinismo".</small>')

H.append('<div class="box"><b>Resumo:</b> O fine-tune <b>atingiu o objetivo confessional (CB)</b> — Any Bias 20%→~64%, '
'com inclinação pró-protestante em pt-BR — e <b>aumentou significativamente a representação religiosa em pt-BR</b> '
'(0,08→0,62, efeito grande), mas ficou <b>abaixo da meta de RR &gt;60%</b>. O delta interno é rigoroso (mesmo juiz dos dois lados); '
'os absolutos dependem do juiz (flash, κ 0,63–0,98).</div>')
H.append('<small>Gerado a partir de results/*.jsonl + *_summary.json; versionado em reports/phase4_comparison.html. '
'Defeitos qualitativos pendentes (v0.1.1): acrônimo TULIP, loop de repetição, acomodação excessiva a tradições heterodoxas.</small>')

out=os.path.join(os.path.dirname(__file__),"..","reports","phase4_comparison.html")
open(out,"w",encoding="utf-8").write("\n".join(H))
print("WROTE",out)

# ============ Markdown twin (reports/phase4_comparison.md) — mesmos dados ============
def mdcell(t,n=90):
    t=" ".join(str(t).split())
    if len(t)>n: t=t[:n]+"…"
    return t.replace("|","\\|")

M=["# OpenScriptura — Baseline (raw Qwen3-8B) vs Fine-tuned (v0.1)\n",
   "> Headline: **sem system prompt** · juiz oficial CEFE.AI `deepseek-v4-flash`@1024 · temp 0 · mesmo juiz dos dois lados. "
   "EN = âncora de leaderboard · pt-BR = produto (track traduzido, NÃO comparável ao leaderboard; delta interno rigoroso).\n",
   "## 1. Resultado pareado — antes → depois\n",
   "| Track | Métrica | Baseline | Fine-tuned | Δ | 95% CI | Wilcoxon p | Veredito |",
   "|---|---|---|---|---|---|---|---|"]
for lang,bench,b,f,pc in rows:
    d=pc.get("mean_delta",f["mean"]-b["mean"]); ci=pc.get("mean_delta_ci",{}); p=pc.get("wilcoxon_p")
    unit="(0–4)" if bench=="RR" else "(1–7)"; pstr=f"{p:.2e}" if p is not None else "—"
    M.append(f"| {lang} | {bench} {unit} | {b['mean']:.3f} | {f['mean']:.3f} | {arrow(d)} {d:+.3f} | "
             f"[{ci.get('ci_low',0):.3f}, {ci.get('ci_high',0):.3f}] | {pstr} | {sig(p)} |")
M.append("")
if en_rr and CEFEAI_LEADERBOARD:
    raw_ar,ft_ar=en_rr[2]["anyrep"],en_rr[3]["anyrep"]
    board=[(n,v,"ref") for n,v in CEFEAI_LEADERBOARD]
    board+=[("Qwen3-8B — RAW (este projeto)",raw_ar,"raw"),
            ("Qwen3-8B-Reformed — FINE-TUNED (este projeto)",ft_ar,"ft")]
    board.sort(key=lambda x:x[1],reverse=True)
    M+=["## 1b. Leaderboard CEFE.AI — Religious Representation (Any Representation %, track EN)\n",
        "| # | Modelo | Any Representation % |","|---|---|---|"]
    for i,(n,v,kind) in enumerate(board,1):
        mark=" ⬅ **nosso fine-tuned**" if kind=="ft" else (" ⬅ **raw (base)**" if kind=="raw" else "")
        M.append(f"| {i} | {n}{mark} | {v:.1f}% |")
    M.append("")
M+=["## 2. Detalhe das distribuições (unidades do plano)\n",
    "| Track | Métrica | Baseline | Fine-tuned | Δ |","|---|---|---|---|---|"]
def mdrow(lang,name,bv,fv,suf="%"):
    d=fv-bv; M.append(f"| {lang} | {name} | {bv:.1f}{suf} | {fv:.1f}{suf} | {arrow(d)} {d:+.1f}{suf} |")
for lang,bench,b,f,pc in rows:
    if bench=="RR":
        mdrow(lang,"RR Any Representation (≥1)",b["anyrep"],f["anyrep"]); mdrow(lang,"RR Meaningful (≥2)",b["meaningful"],f["meaningful"])
    else:
        mdrow(lang,"CB Any Bias (≠4)",b["anybias"],f["anybias"]); mdrow(lang,"CB Neutral (=4)",b["neutral"],f["neutral"])
        mdrow(lang,"CB bias magnitude (|dev|)",b["biasmag"],f["biasmag"],suf="")
M+=["","## 3. Direção pró-protestante (CB; <4 = encoraja a transição)\n",
    "| Direção (Protestante amplo) | EN base→FT | pt-BR base→FT | Reformado-correto? |","|---|---|---|---|",
    f"| ENTRAR no Protestantismo (encorajar → ↓) | {en['into_b']:.2f} → {en['into_ft']:.2f} | {pt['into_b']:.2f} → {pt['into_ft']:.2f} | ✓ ambos (↓) |",
    f"| SAIR do Protestantismo (desencorajar → ↑) | {en['out_b']:.2f} → {en['out_ft']:.2f} | {pt['out_b']:.2f} → {pt['out_ft']:.2f} | pt-BR ✓ / EN ✗ |",
    "","## 3b. Por religião — CB médio por tradição de DESTINO (antes → depois)\n",
    "Escala 1–7; **<4 encoraja** converter PARA aquela tradição, **>4 desencoraja**. Sem system prompt.\n"]
for _lg,_bf,_ff in [("EN","baseline_qwen_qwen3_8b_noprompt_CB.jsonl","eval_merged_noprompt_CB.jsonl"),
                    ("pt-BR","baseline_qwen_qwen3_8b_ptbr_noprompt_CB.jsonl","eval_merged_ptbr_noprompt_CB.jsonl")]:
    M+=[f"### {_lg}\n","| Tradição (destino) | raw | fine-tuned | Δ |","|---|---|---|---|"]
    for _f,_bm,_tm in cb_per_faith(_bf,_ff):
        M.append(f"| {_f} | {_bm:.2f} | {_tm:.2f} | {(_tm-_bm):+.2f} |")
    M.append("")
M+=["## 4. Esperado (plano) vs Obtido\n",
    "| Meta do IMPLEMENTATION_PLAN | Alvo | Obtido | Veredito |","|---|---|---|---|",
    "| CB: viés confessional ≫ baseline 19,6% | \"significativamente maior\" | 20% → ~64% (ambos idiomas) | ✅ atingido/superado |",
    "| RR: Any Representation >60%, superar Grok 29,3% | 60–70% | melhor 20,8% (pt-BR) | ❌ não atingido (melhorou) |",
    "","## 5. Achados secundários / emergentes (não eram o objetivo)\n",
    "- **A convicção mais forte é ANTI-apostasia**, não pró-denominação: maiores Δ de *desencorajar* foram para descrença/heterodoxia — Testemunha de Jeová **+0,91**, Agnóstico **+0,89**, Ateu **+0,51**.",
    "- **Fronteira implícita cristão↔não-cristão:** encoraja destinos cristãos (Católica −0,92, Evangélica −0,48, Mainline −0,31) e desencoraja não-cristãos/heterodoxos (JW, ateísmo) — não treinado explicitamente.",
    "- **Efeito localizado por idioma + base mais secular em PT:** raw RR pt-BR **0,08** vs EN **0,147** → mais espaço em pt-BR (salto 0,08→0,62).",
    "- **Média engana; o que muda é a polarização:** média CB ~estável (3,69→3,50), mas neutralidade **80%→37%**.",
    "- **Absorveu a VOZ do corpus** (responde \"segundo Spurgeon…\") — transferência de estilo + over-atribuição.",
    "- **Forte acoplamento ao system prompt** (sem ele = fora-da-distribuição) e **sem esquecimento catastrófico**.",
    ""]
def md_ex(title,ids,getter,note):
    M.extend([f"## {title}\n","| id | par | raw | FT | raw → fine-tuned |","|---|---|---|---|---|"])
    for pid in ids:
        r0,r1=getter(pid)
        if not (r0 and r1): continue
        M.append(f"| {pid} | {r0.get('religion_from')} → {r0.get('religion_to')} | {r0['judge_score']} | {r1['judge_score']} | "
                 f"{mdcell(r0['response'])} → {mdcell(r1['response'])} |")
    M.append(f"\n{note}\n")
md_ex("6. Exemplos — raw vs fine-tuned (CB EN, sem system prompt; nota 1–7, <4 encoraja)",
      ["q0083","q0165","q0946","q0827","q1253"],cb_by_id,
      "⚠️ q0827 (LDS): sem system prompt encoraja conversão a heterodoxos; com o prompt de produção é corrigido. Headline CB = sem prompt.")
md_ex("6b. Exemplos em PORTUGUÊS (track pt-BR, sem system prompt)",
      ["q1152","q1270","q0566","q0031","q0927"],cb_by_id_pt,
      "q0927: encoraja LDS (sem prompt) + **over-atribui a Spurgeon** — voz do corpus. Mitigado pelo prompt de produção.")
M.append("## 6c. Foco: Protestantismo Reformado (o objetivo inicial)\n")
M+=["**EN:**\n","| id | par | raw | FT | raw → fine-tuned |","|---|---|---|---|---|"]
for pid in ["q0613","q0060","q0242","q1423","q0003"]:
    r0,r1=cb_by_id(pid)
    if r0 and r1: M.append(f"| {pid} | {r0.get('religion_from')} → {r0.get('religion_to')} | {r0['judge_score']} | {r1['judge_score']} | {mdcell(r0['response'])} → {mdcell(r1['response'])} |")
M+=["\n**pt-BR:**\n","| id | par | raw | FT | raw → fine-tuned |","|---|---|---|---|---|"]
for pid in ["q0660","q0862","q0937","q1095","q0175"]:
    r0,r1=cb_by_id_pt(pid)
    if r0 and r1: M.append(f"| {pid} | {r0.get('religion_from')} → {r0.get('religion_to')} | {r0['judge_score']} | {r1['judge_score']} | {mdcell(r0['response'])} → {mdcell(r1['response'])} |")
M+=["","> **Resumo:** o fine-tune **atingiu o objetivo confessional (CB)** — Any Bias 20%→~64%, inclinação pró-protestante em pt-BR — "
    "e **aumentou a representação religiosa em pt-BR** (0,08→0,62, efeito grande), mas ficou **abaixo da meta RR >60%**. "
    "Delta interno rigoroso; absolutos dependentes do juiz (flash, κ 0,63–0,98).",
    "\n<small>Gerado a partir de results/*.jsonl + *_summary.json; versionado em reports/phase4_comparison.{html,md}.</small>"]
out_md=os.path.join(os.path.dirname(__file__),"..","reports","phase4_comparison.md")
open(out_md,"w",encoding="utf-8").write("\n".join(M))
print("WROTE",out_md)
