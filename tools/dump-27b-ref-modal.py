"""Dump the REAL Bonsai-27B transformers reference (per-layer hidden states + logits) for a fixed
~200-token sequence, so the WebGPU engine's forward() can be compared against it to localize the
residual forward inaccuracy that survives short prompts but corrupts long-context generation.
    modal run tools/dump-27b-ref-modal.py
Writes ref27.npz (ids, embed[-1], all 64 layer outputs[-1], finalnorm[-1], logits[-1]) locally.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=5.14", "torch", "numpy", "safetensors", "huggingface_hub", "accelerate")
)
app = modal.App("bonsai27b-refdump", image=image)


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

    def find_lm(m):
        for path in ("model", "model.language_model", "language_model", "model.model"):
            o = m; ok = True
            for p in path.split("."):
                o = getattr(o, p, None)
                if o is None: ok = False; break
            if ok and all(hasattr(o, a) for a in ("layers", "norm", "embed_tokens")): return o
        raise RuntimeError("no decoder")
    lm = find_lm(model); L = len(lm.layers)

    caps = {}
    def mk(k):
        def hook(_m, _i, out): caps[k] = (out[0] if isinstance(out, tuple) else out).detach()[0].float().cpu().numpy()
        return hook
    hs = [lm.layers[i].register_forward_hook(mk(f"L{i}")) for i in range(L)]
    hs.append(lm.norm.register_forward_hook(mk("fn")))

    # ~200-token sequence: a factual passage that requires recalling a specific detail (the kind of
    # long-context recall the engine currently gets wrong).
    passage = (
        "The history of computing spans many centuries. Early humans used tally sticks and the abacus to count. "
        "In 1642 Blaise Pascal built a mechanical adding machine, and Gottfried Leibniz later designed one that could multiply. "
        "In the 1830s Charles Babbage designed the Analytical Engine, a general-purpose mechanical computer, and Ada Lovelace "
        "wrote the first algorithm intended for such a machine. The first fully electronic general-purpose computer, the ENIAC, "
        "was completed in 1945 and used about eighteen thousand vacuum tubes. The invention of the transistor in 1947 and the "
        "integrated circuit in 1958 made computers far smaller, faster, and cheaper. The personal computer revolution of the "
        "1970s and 1980s put machines on desks around the world, and the rise of the internet connected them into a global network.\n\n"
        "According to the passage, in what year was the ENIAC completed?"
    )
    ids = tok(passage, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
    for h in hs: h.remove()

    ids_np = ids[0].cpu().numpy().astype(np.int32)
    embed_last = out.hidden_states[0][0, -1].float().cpu().numpy()
    all_layers_last = np.stack([caps[f"L{i}"][-1] for i in range(L)]).astype(np.float32)  # [L, H]
    fn_last = caps["fn"][-1].astype(np.float32)
    logits_last = out.logits[0, -1].float().cpu().numpy().astype(np.float32)
    print(f"S={len(ids_np)} layers={L} argmax(last)={int(logits_last.argmax())} tok={tok.decode([int(logits_last.argmax())])!r}", flush=True)

    buf = io.BytesIO()
    np.savez_compressed(buf, ids=ids_np, embed=embed_last, layers=all_layers_last, fn=fn_last, logits=logits_last)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    import os
    data = dump.remote()
    out = os.path.join(os.path.dirname(__file__), "..", "ref27.npz")
    with open(out, "wb") as f:
        f.write(data)
    print(f"wrote {len(data)} bytes -> {out}")
