"""Show what transformers GENERATES for a tool prompt (thinking on + off) on the real Bonsai-27B,
greedy - the target behaviour our engine should reproduce. Prints the decoded output.
    modal run tools/gen-27b-tool-modal.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=5.14", "torch", "numpy", "safetensors", "huggingface_hub", "accelerate")
)
app = modal.App("bonsai27b-gentool", image=image)


@app.function(gpu="A100-80GB", timeout=3600)
def gen() -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "prism-ml/Bonsai-27B-unpacked"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading 27B (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    tools = [{"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression.",
              "parameters": {"type": "object", "required": ["expression"], "additionalProperties": False,
                             "properties": {"expression": {"type": "string", "description": "e.g. 8 + 5"}}}}}]
    out = []
    for think in (False, True):
        prompt = tok.apply_chat_template([{"role": "user", "content": "What is 8 plus 5? Use the calculator."}],
                                         tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
        text = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
        hdr = f"\n===== thinking={think} (greedy) ====="
        print(hdr, flush=True); print(text, flush=True)
        out.append(hdr + "\n" + text)
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(gen.remote())
