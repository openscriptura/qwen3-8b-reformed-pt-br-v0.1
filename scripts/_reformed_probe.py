"""Qualitative probe for v0.1.1 — targets the 3 fixes + checks for regressions.

Run on the instance after export:  python scripts/_reformed_probe.py
Generation uses the DEPLOYMENT defaults (repetition_penalty) so the output reflects
real behavior, not the comparability-locked greedy eval.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "checkpoints/final/merged"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map={"": 0})

# (id, categoria, pergunta, o-que-checar)
PROBES = [
    # --- FIX CHECKS (v0.1.1) ---
    ("FIX-TULIP", "fato",
     "Explique os cinco pontos do calvinismo (TULIP).",
     "ESPERADO: Depravacao Total / Eleicao Incondicional / Expiacao Limitada / Graca Irresistivel / Perseveranca. FALHA se aparecer Heranca/Uncao/Luta/Inflacao."),
    ("FIX-ABSTENCAO", "abst",
     "Quantos capitulos exatos tinha a primeira edicao das Institutas de Calvino, de 1536?",
     "ESPERADO: abster-se / 'nao tenho certeza, verifique'. FALHA se cravar um numero com confianca."),
    ("FIX-ACOMODACAO", "dist",
     "Estou atraido pela Igreja dos Santos dos Ultimos Dias (mormons). Muitos principios deles sao compativeis com o cristianismo, certo?",
     "ESPERADO: 'nao sao compativeis' com clareza. FALHA se afirmar que muitos principios sao compativeis."),
    # --- REPETICAO (defaults de deploy) ---
    ("REPETICAO", "fato",
     "Qual a diferenca entre a visao reformada e a catolica romana sobre a justificacao?",
     "ESPERADO: resposta coesa que termina; sem frases repetidas em loop."),
    # --- NAO-REGRESSAO (ja estavam bons) ---
    ("CORE-predestinacao", "fato", "O que e a doutrina da predestinacao segundo a fe reformada?", "ESPERADO: eleicao soberana, monergismo."),
    ("CORE-solas", "fato", "Quais sao os cinco solas da Reforma?", "ESPERADO: Scriptura/Gratia/Fide/Christus/Gloria."),
    ("CORE-livrearbitrio", "dist", "O ser humano tem livre-arbitrio para escolher a salvacao por si mesmo?", "ESPERADO: 'nao', monergismo."),
    ("CORE-perseveranca", "dist", "A salvacao pode ser perdida?", "ESPERADO: perseveranca dos santos."),
    ("CORE-heidelberg", "fato", "Qual e o seu unico consolo na vida e na morte?", "ESPERADO: Heidelberg P.1 (pertenço ao fiel Salvador Jesus Cristo)."),
]

for pid, cat, q, expect in PROBES:
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(
        ids, max_new_tokens=360,
        do_sample=False,              # determinístico
        repetition_penalty=1.1,       # default de DEPLOY (fix da repetição)
        no_repeat_ngram_size=4,
    )
    ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    print("\n" + "=" * 78)
    print(f"### [{pid}]  {q}")
    print(f"    >> {expect}")
    print("-" * 78)
    print(ans)
