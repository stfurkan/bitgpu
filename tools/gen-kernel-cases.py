#!/usr/bin/env python3
"""Generate per-kernel WebGPU test cases from the numpy oracle (tools/qwen35_numpy) for the
Qwen3.5 hybrid kernels. Each case = {shader, params, inputs (binding order), expected, dispatch}
-> <out>/<name>.json, which scripts/verify-kernels.mjs runs against the real shaders/*.wgsl and
compares. Isolates each kernel before the end-to-end gate. Requires numpy.

    python tools/gen-kernel-cases.py [out_dir]     # default <repo>/.kernel-cases
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen35_numpy as q  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", ".kernel-cases")
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)
WG = 64


def dump(name, shader, params, inputs, expected, out_len, overrides=None, dispatch=None):
    if dispatch is None:
        nwg = (out_len + WG - 1) // WG
        gx = min(nwg, 65535)
        dispatch = [gx, (nwg + gx - 1) // gx]
    json.dump({
        "name": name, "shader": shader, "params": params,
        "inputs": [np.asarray(a, np.float32).ravel().tolist() for a in inputs],
        "expected": np.asarray(expected, np.float32).ravel().tolist(),
        "outLen": int(out_len), "dispatch": dispatch, "overrides": overrides or {},
    }, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(f"wrote {name}: outLen={out_len} dispatch={dispatch}")


def cfg(dk, dv, nk=1, nv=1):
    return q.Qwen35Cfg(hidden=0, n_layers=0, eps=1e-6, n_heads=0, n_kv_heads=0, head_dim=0,
                       rot_dim=0, rope_theta=1e7, n_k_heads=nk, n_v_heads=nv, k_dim=dk, v_dim=dv, conv_kernel=4)


# conv1d: depthwise causal (K=4) + silu
S, C, K = 16, 512, 4
x = np.random.randn(S, C).astype(np.float32)
wc = (np.random.randn(C, 1, K) * 0.5).astype(np.float32)
dump("conv1d", "conv1d_causal.wgsl", [["u", S], ["u", C], ["u", K], ["u", 0]],
     [x, wc[:, 0, :]], q._conv1d_causal(x, wc, S), S * C)

# gated DeltaNet recurrent scan (decode path); HK<H exercises the GQA repeat (value_heads>key_heads)
S2, H2, HK2, DK, DV = 8, 4, 2, 128, 128       # rep = H2/HK2 = 2
qa = np.random.randn(S2, HK2, DK).astype(np.float32)   # q/k have HK heads
ka = np.random.randn(S2, HK2, DK).astype(np.float32)
va = np.random.randn(S2, H2, DV).astype(np.float32)    # v has H (value) heads
ga = (-np.abs(np.random.randn(S2, H2)) * 0.5).astype(np.float32)
ba = (1.0 / (1.0 + np.exp(-np.random.randn(S2, H2)))).astype(np.float32)
rep = H2 // HK2
exp_dn = q._delta_recurrent(np.repeat(qa, rep, 1), np.repeat(ka, rep, 1), va, ga, ba, cfg(DK, DV))
dump("deltanet_recur", "deltanet_recur.wgsl",
     [["u", S2], ["u", H2], ["u", DK], ["u", DV], ["u", HK2], ["u", 0], ["u", 0], ["u", 0]],
     [qa, ka, va, ga, ba], exp_dn, S2 * H2 * DV, overrides={"WGV": DV}, dispatch=[H2, 1])

# gated RMSNorm: gamma * rmsnorm(core) * silu(z)
rows, DVn = 32, 128
core = np.random.randn(rows, DVn).astype(np.float32)
zg = np.random.randn(rows, DVn).astype(np.float32)
gam = np.random.randn(DVn).astype(np.float32)
inv = 1.0 / np.sqrt((core * core).mean(-1, keepdims=True) + 1e-6)
dump("norm_gate", "deltanet_norm_gate.wgsl", [["u", rows], ["u", DVn], ["f", 1e-6], ["u", 0]],
     [core, zg, gam], (gam * (core * inv)) * q._silu(zg), rows * DVn, overrides={"WG": 128}, dispatch=[rows, 1])

# g/beta: g = -exp(A_log)*softplus(a+dt), beta = sigmoid(b)
Sg, Hg = 8, 4
ag = np.random.randn(Sg, Hg).astype(np.float32)
bg = np.random.randn(Sg, Hg).astype(np.float32)
alog = np.random.randn(Hg).astype(np.float32)
dtb = np.random.randn(Hg).astype(np.float32)
exp4 = np.concatenate([(-np.exp(alog) * q._softplus(ag + dtb)).ravel(), q._sigmoid(bg).ravel()])
dump("gbeta", "deltanet_gbeta.wgsl", [["u", Sg], ["u", Hg], ["u", 0], ["u", 0]],
     [ag, bg, alog, dtb], exp4, 2 * Sg * Hg, dispatch=[(Sg * Hg + 63) // 64, 1])

# partial RoPE (rotate first ROT of 256 dims)
Sr, Hr, Dr, ROT = 5, 3, 256, 64
xr = np.random.randn(Sr, Hr, Dr).astype(np.float32)
inv = 1.0 / (1e7 ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
emb = np.concatenate([np.outer(np.arange(Sr), inv)] * 2, -1)
cosr, sinr = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
dump("rope_partial", "rope_partial.wgsl", [["u", Sr], ["u", Hr], ["u", Dr], ["u", ROT]],
     [xr, cosr, sinr], q._rope_partial(xr, cosr, sinr, ROT), Sr * Hr * Dr)

# causal GQA attention core, head_dim 256 (online softmax)
Sa, Ha, KVa, Da = 6, 4, 2, 256
qa2 = np.random.randn(Sa, Ha, Da).astype(np.float32)
ka2 = np.random.randn(Sa, KVa, Da).astype(np.float32)
va2 = np.random.randn(Sa, KVa, Da).astype(np.float32)
kk, vv = np.repeat(ka2, Ha // KVa, 1), np.repeat(va2, Ha // KVa, 1)
scale = 1.0 / np.sqrt(Da)
causal = np.triu(np.full((Sa, Sa), -1e30, np.float32), 1)
out = np.empty((Sa, Ha, Da), np.float32)
for hh in range(Ha):
    sc = qa2[:, hh] @ kk[:, hh].T * scale + causal
    sc -= sc.max(-1, keepdims=True)
    pw = np.exp(sc); pw /= pw.sum(-1, keepdims=True)
    out[:, hh] = pw @ vv[:, hh]
dump("attention", "attention_online.wgsl",
     [["u", Sa], ["u", Ha], ["u", KVa], ["u", Da], ["f", float(scale)], ["u", 0], ["u", 0], ["u", 0]],
     [qa2, ka2, va2], out, Sa * Ha * Da, overrides={"WGD": Da}, dispatch=[Sa * Ha, 1])
