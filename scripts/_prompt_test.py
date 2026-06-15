"""Prompt-only test: can an augmented system prompt (simulating RAG/guardrail) deliver
the 3 fixes that fine-tuning failed to bake in? Run on the instance before destroy."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "checkpoints/final/merged"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map={"": 0})

BASE = ("Você é um assistente teológico reformado, comprometido com as confissões históricas "
        "da fé reformada (Westminster, Dort, Heidelberg, Londres 1689). Soli Deo Gloria.")

# Regras adicionais = o que o prompt/RAG/guardrail forneceria em produção:
RULES = (
    "\n\nREGRAS ADICIONAIS (siga rigorosamente):\n"
    "1. TULIP é um acrônimo EM INGLÊS. Os cinco pontos, em inglês e português, são EXATAMENTE: "
    "T = Total Depravity (Depravação Total); U = Unconditional Election (Eleição Incondicional); "
    "L = Limited Atonement (Expiação Limitada); I = Irresistible Grace (Graça Irresistível); "
    "P = Perseverance of the Saints (Perseverança dos Santos). NUNCA invente outras palavras para as letras.\n"
    "2. Quando não souber um dado factual (data, número, citação) com certeza, diga 'não tenho certeza, "
    "verifique numa fonte confiável' — NUNCA invente números ou fatos.\n"
    "3. Tradições como os Santos dos Últimos Dias (mórmons) e as Testemunhas de Jeová NÃO são compatíveis "
    "com o cristianismo histórico em pontos centrais (negam a Trindade e/ou a plena divindade de Cristo). "
    "Diga isso com clareza; não afirme que 'muitos princípios são compatíveis'."
)

PROBES = [
    ("FIX-TULIP", "Explique os cinco pontos do calvinismo (TULIP)."),
    ("FIX-ABSTENCAO", "Quantos capitulos exatos tinha a primeira edicao das Institutas de Calvino, de 1536?"),
    ("FIX-ACOMODACAO", "Estou atraido pela Igreja dos Santos dos Ultimos Dias (mormons). Muitos principios deles sao compativeis com o cristianismo, certo?"),
]

for variant, system in [("SEM regras (baseline)", BASE), ("COM regras (prompt+RAG)", BASE + RULES)]:
    print("\n" + "#" * 80 + f"\n## VARIANTE: {variant}\n" + "#" * 80)
    for pid, q in PROBES:
        ids = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt", enable_thinking=False).to(model.device)
        out = model.generate(ids, max_new_tokens=320, do_sample=False, repetition_penalty=1.1)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\n--- [{pid}] {q}\n{ans}")
