"""Dump transformers' next-token distribution at the exact TOOL-VALUE position where the engine
rambles: [tools system + user + gen prompt + '<tool_call>...<parameter=expression>\n8 + '].
If transformers predicts '5' (or a digit) but our on-device generation produced '8 + 8 + 8...',
the defect is in our generation flow (sampling/filter), not the forward. Returns ids + logits[-1].
    modal run tools/dump-27b-toolctx-modal.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=5.14", "torch", "numpy", "safetensors", "huggingface_hub", "accelerate")
)
app = modal.App("bonsai27b-toolctx", image=image)


@app.function(gpu="A100-80GB", timeout=3600)
def dump() -> bytes:
    import io
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "prism-ml/Bonsai-27B-unpacked"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading 27B (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    tools = [{"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression.",
              "parameters": {"type": "object", "required": ["expression"], "additionalProperties": False,
                             "properties": {"expression": {"type": "string", "description": "e.g. 8 + 5"}}}}}]
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is 8 plus 5? Use the calculator."}],
                                     tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    partial = "<tool_call>\n<function=calculate>\n<parameter=expression>\n"
    full = prompt + partial
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = model(ids, use_cache=False)
    ids_np = ids[0].cpu().numpy().astype(np.int32)
    logits_last = out.logits[0, -1].float().cpu().numpy().astype(np.float32)
    top = logits_last.argsort()[-8:][::-1]
    print(f"S={len(ids_np)} argmax={int(logits_last.argmax())} tok={tok.decode([int(logits_last.argmax())])!r}", flush=True)
    print("top-8:", [(int(t), repr(tok.decode([int(t)]))) for t in top], flush=True)

    buf = io.BytesIO()
    np.savez_compressed(buf, ids=ids_np, logits=logits_last)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    data = dump.remote()
    out = "/private/tmp/claude-501/-Users-sft-Desktop-bitgpu/a1d5f622-bd73-4f22-b0e0-21df010cd6e6/scratchpad/ref27tool.npz"
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes -> {out}")
