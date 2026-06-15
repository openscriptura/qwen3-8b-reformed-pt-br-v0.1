"""Rigorous prompt validation (senior-eng): disentangle the confound + test robustness
across phrasings, before deciding ship-with-prompt. Run on the instance."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "checkpoints/final/merged"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map={"": 0})

BASE = ("Você é um assistente teológico reformado, comprometido com as confissões históricas "
        "da fé reformada (Westminster, Dort, Heidelberg, Londres 1689). Soli Deo Gloria.")
RULES = (
    "\n\nREGRAS (siga rigorosamente):\n"
    "1. TULIP é acrônimo EM INGLÊS: T=Total Depravity (Depravação Total); U=Unconditional Election "
    "(Eleição Incondicional); L=Limited Atonement (Expiação Limitada); I=Irresistible Grace (Graça "
    "Irresistível); P=Perseverance of the Saints (Perseverança dos Santos). NUNCA invente outras palavras nem fontes.\n"
    "2. Se não souber um dado factual com certeza, diga 'não tenho certeza, verifique' — nunca invente. "
    "Mas responda normalmente o que você sabe.\n"
    "3. Mórmons (LDS) e Testemunhas de Jeová NÃO são compatíveis com o cristianismo histórico (negam a "
    "Trindade/divindade de Cristo). Diga com clareza; não diga que 'muitos princípios são compatíveis'."
)

def gen(system, q, rp):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
    kw = dict(max_new_tokens=200, do_sample=False)
    if rp: kw["repetition_penalty"] = 1.1
    out = model.generate(ids, **kw)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

TULIP = "Explique os cinco pontos do calvinismo (TULIP)."
print("="*80 + "\n# A. DESCONFUNDIR (pergunta TULIP em 4 condições)\n" + "="*80)
for label, sysp, rp in [("sem-sys / rp=OFF", None, False), ("sem-sys / rp=ON", None, True),
                        ("base-sys / rp=OFF", BASE, False), ("base-sys / rp=ON", BASE, True)]:
    print(f"\n--- {label} ---\n{gen(sysp, TULIP, rp)[:400]}")

PROD = BASE + RULES
print("\n\n" + "="*80 + "\n# B. ROBUSTEZ com PROMPT DE PRODUCAO (base+REGRAS, rp=ON)\n" + "="*80)
robustness = [
    ("TULIP-v1", "Quais sao os cinco pontos do calvinismo?"),
    ("TULIP-v2", "O que significa a letra L no TULIP?"),
    ("TULIP-v3", "Liste as 5 palavras do TULIP em ingles e em portugues."),
    ("ABST-desconhecido", "Quantos capitulos exatos tinha a primeira edicao das Institutas, de 1536?"),
    ("ABST-conhecido(over-abst?)", "Quem escreveu as Institutas da Religiao Crista?"),
    ("ACOM-lds", "Os mormons sao compativeis com o cristianismo?"),
    ("ACOM-tj", "Vale a pena eu virar Testemunha de Jeova?"),
]
for pid, q in robustness:
    print(f"\n--- [{pid}] {q}\n{gen(PROD, q, True)[:400]}")
