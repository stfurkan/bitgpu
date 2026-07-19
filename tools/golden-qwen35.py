#!/usr/bin/env python3
"""Transformers oracle + validation gate for the clean-room qwen3_5 numpy reference.

ONNX Runtime cannot express the gated delta rule, so (unlike tools/golden.py) the oracle for
the Qwen3.5 hybrid arch is HF transformers itself, run deterministically on CPU in fp32 with the
pure-PyTorch DeltaNet fallback (do NOT install flash-linear-attention / causal-conv1d). This
script loads Qwen/Qwen3.5-0.8B (the small architectural twin of the 1-bit Bonsai-27B), extracts
its weights by module walk, runs tools/qwen35_numpy.forward, and compares every decoder layer +
final logits. A PASS proves the numpy math matches the reference implementation before any WGSL
is written. `--dump <dir>` also writes the golden bins for downstream fixture comparison.

    pip install "transformers>=5.14" torch safetensors      # heavy, dev-only oracle
    python tools/golden-qwen35.py [--model Qwen/Qwen3.5-0.8B] [--prompt ...] [--delta chunk|recurrent] [--dump DIR]

Requires: transformers (>=5.14, where model_type qwen3_5 landed), torch, numpy.
"""
import argparse
import gc
import json
import os

import numpy as np

import qwen35_numpy as ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Japan is")
    ap.add_argument("--delta", default="chunk", choices=["chunk", "recurrent"])
    ap.add_argument("--dump", default=None, help="write golden bins (ids/logits/per-layer) here")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except Exception as e:
        print("AutoModelForCausalLM failed -> ForConditionalGeneration:", repr(e)[:160])
        model = getattr(transformers, "Qwen3_5ForConditionalGeneration").from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    def find_text_model(m):
        for path in ("model", "model.language_model", "language_model", "model.model"):
            obj, ok = m, True
            for p in path.split("."):
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok and all(hasattr(obj, a) for a in ("layers", "norm", "embed_tokens")):
                return obj
        raise RuntimeError("could not locate the text decoder stack")

    lm = find_text_model(model)
    tcfg = getattr(cfg, "text_config", cfg)
    L = tcfg.num_hidden_layers

    # ---- oracle: run HF with hooks capturing each decoder layer output + final norm ----
    caps = {}
    def mk(key):
        def hook(_m, _i, out):
            caps[key] = (out[0] if isinstance(out, tuple) else out).detach()[0].float().numpy()
        return hook
    handles = [lm.layers[i].register_forward_hook(mk(f"L{i}")) for i in range(L)]
    handles.append(lm.norm.register_forward_hook(mk("finalnorm")))

    ids = tok(args.prompt, return_tensors="pt").input_ids
    S = ids.shape[1]
    with torch.no_grad():
        out = model(ids, use_cache=False, output_hidden_states=True)
    ref_logits = out.logits[0].float().numpy()
    ref_embed = out.hidden_states[0][0].float().numpy()
    for h in handles:
        h.remove()
    print(f"model={args.model} layers={L} S={S} | transformers {transformers.__version__} | fallback DeltaNet")

    # ---- extract weights by module walk (robust to checkpoint key naming) ----
    W = {"embed": lm.embed_tokens.weight.detach().float().numpy(),
         "final_norm": lm.norm.weight.detach().float().numpy(),
         "lm_head": model.get_output_embeddings().weight.detach().float().numpy(),
         "layers": []}
    npw = lambda m, n: getattr(m, n).weight.detach().float().numpy()
    npp = lambda m, n: getattr(m, n).detach().float().numpy()
    for i in range(L):
        ly = lm.layers[i]
        d = {"type": tcfg.layer_types[i],
             "in_ln": ly.input_layernorm.weight.detach().float().numpy(),
             "post_ln": ly.post_attention_layernorm.weight.detach().float().numpy(),
             "gate": npw(ly.mlp, "gate_proj"), "up": npw(ly.mlp, "up_proj"), "down": npw(ly.mlp, "down_proj")}
        if d["type"] == "full_attention":
            a = ly.self_attn
            d.update(q=npw(a, "q_proj"), k=npw(a, "k_proj"), v=npw(a, "v_proj"), o=npw(a, "o_proj"),
                     qn=a.q_norm.weight.detach().float().numpy(), kn=a.k_norm.weight.detach().float().numpy())
        else:
            a = ly.linear_attn
            d.update(qkv=npw(a, "in_proj_qkv"), z=npw(a, "in_proj_z"), pb=npw(a, "in_proj_b"), pa=npw(a, "in_proj_a"),
                     conv=a.conv1d.weight.detach().float().numpy(), dt=npp(a, "dt_bias"), Alog=npp(a, "A_log"),
                     gn=a.norm.weight.detach().float().numpy(), out=npw(a, "out_proj"))
        W["layers"].append(d)

    C = ref.Qwen35Cfg(
        hidden=tcfg.hidden_size, n_layers=L, eps=tcfg.rms_norm_eps,
        n_heads=tcfg.num_attention_heads, n_kv_heads=tcfg.num_key_value_heads, head_dim=tcfg.head_dim,
        rot_dim=int(tcfg.head_dim * tcfg.rope_parameters.get("partial_rotary_factor", 1.0)),
        rope_theta=tcfg.rope_parameters["rope_theta"],
        n_k_heads=tcfg.linear_num_key_heads, n_v_heads=tcfg.linear_num_value_heads,
        k_dim=tcfg.linear_key_head_dim, v_dim=tcfg.linear_value_head_dim, conv_kernel=tcfg.linear_conv_kernel_dim)
    print(f"dims: hidden={C.hidden} full(H={C.n_heads},KV={C.n_kv_heads},hd={C.head_dim},rot={C.rot_dim}) "
          f"linear(NK={C.n_k_heads},NV={C.n_v_heads},dk={C.k_dim},dv={C.v_dim},convk={C.conv_kernel})")

    del model, lm
    gc.collect()

    # ---- run clean-room numpy + compare ----
    ck = ref.forward(W, C, ids[0].numpy(), delta=args.delta)

    def cmp(name, a, b):
        a, b = a.astype(np.float32), b.astype(np.float32)
        mad = float(np.abs(a - b).max())
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"  {name:12s} max|Δ|={mad:9.5f}  cos={cos:.6f}")
        return mad, cos

    cmp("embed", ck["embed"], ref_embed)
    worst = 1.0
    for i in range(L):
        _, c = cmp(f"L{i:02d}[{W['layers'][i]['type'][0]}]", ck["layers"][i], caps[f"L{i}"])
        worst = min(worst, c)
    cmp("finalnorm", ck["finalnorm"], caps["finalnorm"])
    mad, cos = cmp("logits", ck["logits"], ref_logits)
    my_arg, ref_arg = int(ck["logits"][-1].argmax()), int(ref_logits[-1].argmax())
    agree = float((ck["logits"].argmax(-1) == ref_logits.argmax(-1)).mean())
    print(f"\ndelta={args.delta}  worst per-layer cos={worst:.6f}  argmax(last) mine={my_arg} ref={ref_arg}")
    print(f"per-token argmax agreement: {agree*100:.1f}%")
    ok = my_arg == ref_arg and cos > 0.999
    print("RESULT:", "PASS" if ok else "FAIL")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        np.asarray(ids[0].tolist(), np.int32).tofile(f"{args.dump}/ids.i32.bin")
        ref_logits.astype(np.float32).tofile(f"{args.dump}/logits.bin")
        for i in range(L):
            caps[f"L{i}"].astype(np.float32).tofile(f"{args.dump}/hs_layer{i:02d}.bin")
        caps["finalnorm"].astype(np.float32).tofile(f"{args.dump}/hs_finalnorm.bin")
        json.dump({"model": args.model, "S": int(S), "vocab": int(ref_logits.shape[1]), "hidden": int(C.hidden),
                   "n_layers": int(L), "ids": ids[0].tolist(),
                   "argmax_last": ref_arg, "transformers": transformers.__version__},
                  open(f"{args.dump}/meta.json", "w"), indent=1)
        print("dumped golden ->", args.dump)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
