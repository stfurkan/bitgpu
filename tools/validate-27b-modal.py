"""Stage 4 oracle check on the REAL Bonsai-27B: run HF transformers (the faithful DeltaNet fallback)
on prism-ml/Bonsai-27B-unpacked and compare the clean-room numpy oracle (tools/qwen35_numpy) to it
layer-by-layer at 27B scale (rep=3, 64 layers, real weights, the -exp(A_log) convention). Memory-
efficient: the model lives on one A100 (bf16), the oracle streams one layer's weights at a time.
    modal run bonsai27b_modal.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=5.14", "torch", "numpy", "safetensors", "huggingface_hub", "accelerate")
    .add_local_file("tools/qwen35_numpy.py", "/root/qwen35_numpy.py")
)
app = modal.App("bonsai27b-oracle", image=image)


@app.function(gpu="A100-80GB", timeout=3600)
def validate():
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch
    import qwen35_numpy as ref
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    MODEL = "prism-ml/Bonsai-27B-unpacked"
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    tcfg = getattr(cfg, "text_config", cfg)
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
    lm = find_lm(model)
    L = tcfg.num_hidden_layers

    caps = {}
    def mk(k):
        def hook(_m, _i, out): caps[k] = (out[0] if isinstance(out, tuple) else out).detach()[0].float().cpu().numpy()
        return hook
    hs = [lm.layers[i].register_forward_hook(mk(f"L{i}")) for i in range(L)]
    hs.append(lm.norm.register_forward_hook(mk("fn")))

    ids = tok("The capital of France is Paris. The capital of Japan is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
    ref_logits = out.logits[0].float().cpu().numpy()
    ref_embed = out.hidden_states[0][0].float().cpu().numpy()
    for h in hs: h.remove()
    ids_np = ids[0].cpu().numpy()
    print(f"prompt S={len(ids_np)} | transformers {__import__('transformers').__version__} | ref argmax(last)={int(ref_logits[-1].argmax())}", flush=True)

    C = ref.Qwen35Cfg(
        hidden=tcfg.hidden_size, n_layers=L, eps=tcfg.rms_norm_eps,
        n_heads=tcfg.num_attention_heads, n_kv_heads=tcfg.num_key_value_heads, head_dim=tcfg.head_dim,
        rot_dim=int(tcfg.head_dim * tcfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        rope_theta=tcfg.rope_parameters["rope_theta"],
        n_k_heads=tcfg.linear_num_key_heads, n_v_heads=tcfg.linear_num_value_heads,
        k_dim=tcfg.linear_key_head_dim, v_dim=tcfg.linear_value_head_dim, conv_kernel=tcfg.linear_conv_kernel_dim)
    print(f"dims hidden={C.hidden} full(H={C.n_heads},KV={C.n_kv_heads},hd={C.head_dim},rot={C.rot_dim}) "
          f"linear(NK={C.n_k_heads},NV={C.n_v_heads},dk={C.k_dim},dv={C.v_dim}) rep={C.n_v_heads // C.n_k_heads}", flush=True)

    npw = lambda m, n: getattr(m, n).weight.detach().float().cpu().numpy()
    npp = lambda m, n: getattr(m, n).detach().float().cpu().numpy()
    cos, sin = ref._rope_tables(C, len(ids_np))

    def layer_dict(ly):
        d = {"type": tcfg.layer_types[li],
             "in_ln": ly.input_layernorm.weight.detach().float().cpu().numpy(),
             "post_ln": ly.post_attention_layernorm.weight.detach().float().cpu().numpy(),
             "gate": npw(ly.mlp, "gate_proj"), "up": npw(ly.mlp, "up_proj"), "down": npw(ly.mlp, "down_proj")}
        if d["type"] == "full_attention":
            a = ly.self_attn
            d.update(q=npw(a, "q_proj"), k=npw(a, "k_proj"), v=npw(a, "v_proj"), o=npw(a, "o_proj"),
                     qn=a.q_norm.weight.detach().float().cpu().numpy(), kn=a.k_norm.weight.detach().float().cpu().numpy())
        else:
            a = ly.linear_attn
            d.update(qkv=npw(a, "in_proj_qkv"), z=npw(a, "in_proj_z"), pb=npw(a, "in_proj_b"), pa=npw(a, "in_proj_a"),
                     conv=a.conv1d.weight.detach().float().cpu().numpy(), dt=npp(a, "dt_bias"),
                     Alog=-np.exp(npp(a, "A_log")), gn=a.norm.weight.detach().float().cpu().numpy(), out=npw(a, "out_proj"))
        return d

    def cmp(name, a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        mad = float(np.abs(a - b).max()); cos_ = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        return mad, cos_

    h = npw(lm, "embed_tokens")[ids_np].astype(np.float32) if hasattr(lm.embed_tokens, "weight") else ref_embed.copy()
    worst = 1.0
    for li in range(L):
        d = layer_dict(lm.layers[li])
        res = h
        x = ref.rmsnorm(h, d["in_ln"], C.eps)
        mix = ref._full_attention(x, d, C, cos, sin) if d["type"] == "full_attention" else ref._linear_attention(x, d, C, "recurrent")
        h = res + mix
        res = h
        x = ref.rmsnorm(h, d["post_ln"], C.eps)
        h = res + (ref._silu(x @ d["gate"].T) * (x @ d["up"].T)) @ d["down"].T
        mad, c = cmp(f"L{li}", h, caps[f"L{li}"]); worst = min(worst, c)
        if li < 4 or li % 8 == 3 or li == L - 1:
            print(f"  L{li:02d}[{d['type'][0]}] max|Δ|={mad:.4f} cos={c:.6f}", flush=True)
        del d
    h = ref.rmsnorm(h, npw(lm, "norm") if False else lm.norm.weight.detach().float().cpu().numpy(), C.eps)
    cmp("fn", h, caps["fn"])
    logits = h @ model.get_output_embeddings().weight.detach().float().cpu().numpy().T
    mad, c = cmp("logits", logits, ref_logits)
    my, rf = int(logits[-1].argmax()), int(ref_logits[-1].argmax())
    agree = float((logits.argmax(-1) == ref_logits.argmax(-1)).mean())
    print(f"\nworst per-layer cos={worst:.6f} | logits max|Δ|={mad:.4f} cos={c:.6f}", flush=True)
    print(f"argmax(last) numpy={my} transformers={rf} {'MATCH' if my == rf else 'MISMATCH'} | per-token agreement {agree*100:.1f}%", flush=True)
    print("RESULT:", "PASS - oracle matches transformers on the real 27B" if my == rf and c > 0.999 else "FAIL", flush=True)


@app.local_entrypoint()
def main():
    validate.remote()
