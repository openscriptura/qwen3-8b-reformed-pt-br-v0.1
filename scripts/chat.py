"""Teste local do modelo fine-tuned (CPU), com o system prompt de PRODUÇÃO embutido.
Uso:
  python scripts/chat.py "Explique o TULIP"        # one-shot
  python scripts/chat.py                            # modo interativo (Ctrl+C sai)
"""
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = str(ROOT / "results" / "phase4" / "merged")
SYS = (ROOT / "configs" / "system_prompt_production.txt").read_text(encoding="utf-8").strip()

print("Carregando modelo (CPU, bf16 — pode levar ~1 min)...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu"
)
model.eval()

def ask(q: str) -> str:
    ids = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
    )
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=320, do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

if len(sys.argv) > 1:
    print("\n" + ask(" ".join(sys.argv[1:])))
else:
    print("Pronto. Digite a pergunta (Enter vazio ignora; Ctrl+C sai).")
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\n" + ask(q))
