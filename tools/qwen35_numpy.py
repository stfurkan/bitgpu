#!/usr/bin/env python3
"""Clean-room numpy reference for the Qwen3.5 (`qwen3_5`) hybrid backbone.

Text-only trunk: a 3:1 stack of gated-DeltaNet linear-attention layers and gated
full-attention layers, partial RoPE, gated RMSNorm, SwiGLU MLP. Framework-agnostic -
weights come in as a plain dict of numpy arrays (from HF fp32 in tools/golden-qwen35.py,
or dequantized 1-bit from a bitgpu manifest in the fixture generator), so the SAME math is
the oracle everywhere. Validated layer-by-layer against HF transformers 5.14 to max|Δ|~5e-5,
argmax-exact (see tools/golden-qwen35.py). This is the reference the WGSL kernels must match.

The two delta-rule variants produce the same result up to fp accumulation order:
  - 'recurrent': O(1)/token sequential scan (what bitgpu decode runs)
  - 'chunk':     chunk-parallel WY-representation scan (what bitgpu prefill runs; == HF prefill)

Everything runs in float32 to mirror the model's compute dtype (config mamba_ssm_dtype).
"""
from dataclasses import dataclass

import numpy as np

f32 = np.float32


@dataclass
class Qwen35Cfg:
    hidden: int
    n_layers: int
    eps: float
    # full attention
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rot_dim: int          # head_dim * partial_rotary_factor
    rope_theta: float
    # linear (gated DeltaNet)
    # NOTE: the layer weight dict's "Alog" holds -exp(A_log) (the GGUF ssm_a convention), so g is a
    # plain multiply: g = Alog * softplus(a + dt_bias). Feeders must pre-apply -exp to A_log.
    n_k_heads: int
    n_v_heads: int
    k_dim: int            # linear_key_head_dim
    v_dim: int            # linear_value_head_dim
    conv_kernel: int

    @property
    def key_dim(self) -> int:
        return self.k_dim * self.n_k_heads

    @property
    def value_dim(self) -> int:
        return self.v_dim * self.n_v_heads


def _silu(x):
    x = x.astype(np.float32)
    return (x / (1.0 + np.exp(-x))).astype(f32)


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(f32)


def _softplus(x):
    return np.logaddexp(0.0, x.astype(np.float32)).astype(f32)


def _l2norm(x, eps=1e-6):
    x = x.astype(np.float32)
    return (x * (1.0 / np.sqrt((x * x).sum(-1, keepdims=True) + eps))).astype(f32)


def rmsnorm(x, g, eps):
    """Plain Qwen3.5 RMSNorm: scale by (1 + weight) (weights are stored 0-centred)."""
    x = x.astype(np.float32)
    return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * (1.0 + g)).astype(f32)


def _rope_tables(C: Qwen35Cfg, S: int):
    inv = 1.0 / (C.rope_theta ** (np.arange(0, C.rot_dim, 2, dtype=np.float64) / C.rot_dim))
    frq = np.outer(np.arange(S, dtype=np.float64), inv)      # [S, rot/2]
    emb = np.concatenate([frq, frq], -1)                     # [S, rot]
    return np.cos(emb).astype(f32), np.sin(emb).astype(f32)


def _rope_partial(x, cos, sin, rot):
    """Rotate the first `rot` dims of each head; pass the rest through (partial RoPE)."""
    xr, xp = x[..., :rot], x[..., rot:]
    half = rot // 2
    rotated = np.concatenate([-xr[..., half:], xr[..., :half]], -1)
    return np.concatenate([xr * cos[:, None, :] + rotated * sin[:, None, :], xp], -1).astype(f32)


def _full_attention(x, d, C: Qwen35Cfg, cos, sin):
    """Gated GQA: q_proj is doubled into (query, gate); output *= sigmoid(gate)."""
    S = x.shape[0]
    H, KV, HD = C.n_heads, C.n_kv_heads, C.head_dim
    qg = (x @ d["q"].T).reshape(S, H, HD * 2)
    q, gate = qg[..., :HD], qg[..., HD:]
    gate = gate.reshape(S, H * HD)
    q = _rope_partial(rmsnorm(q, d["qn"], C.eps), cos, sin, C.rot_dim)
    k = _rope_partial(rmsnorm((x @ d["k"].T).reshape(S, KV, HD), d["kn"], C.eps), cos, sin, C.rot_dim)
    v = (x @ d["v"].T).reshape(S, KV, HD)
    rep = H // KV
    k = np.repeat(k, rep, axis=1)
    v = np.repeat(v, rep, axis=1)
    scale = 1.0 / np.sqrt(HD)
    causal = np.triu(np.full((S, S), -1e30, f32), 1)
    o = np.empty((S, H, HD), f32)
    for h in range(H):
        sc = (q[:, h] @ k[:, h].T) * scale + causal
        sc = sc - sc.max(-1, keepdims=True)
        w = np.exp(sc)
        w /= w.sum(-1, keepdims=True)
        o[:, h] = w @ v[:, h]
    attn = o.reshape(S, H * HD) * _sigmoid(gate)
    return attn @ d["o"].T


def _conv1d_causal(x, wc, S):
    """Depthwise causal conv1d (left-pad kernel-1) + SiLU. wc: [C,1,K]."""
    Cc, K = x.shape[1], wc.shape[-1]
    xp = np.concatenate([np.zeros((K - 1, Cc), f32), x.astype(f32)], 0)
    out = np.zeros((S, Cc), f32)
    for j in range(K):
        out += xp[j:j + S] * wc[:, 0, j][None, :]
    return _silu(out)


def _delta_recurrent(q, k, v, g, beta, C: Qwen35Cfg):
    """Sequential gated delta rule. q,k:[S,H,dk] v:[S,H,dv] g,beta:[S,H]. State [H,dk,dv]."""
    S, H = q.shape[0], q.shape[1]
    q = _l2norm(q) * (1.0 / np.sqrt(C.k_dim))
    k = _l2norm(k)
    state = np.zeros((H, C.k_dim, C.v_dim), np.float32)
    o = np.empty((S, H, C.v_dim), np.float32)
    for t in range(S):
        state = state * np.exp(g[t])[:, None, None]
        kv = (state * k[t][:, :, None]).sum(1)               # kᵀ·S -> [H,dv]
        delta = (v[t] - kv) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, None, :]  # k ⊗ δ
        o[t] = (state * q[t][:, :, None]).sum(1)              # qᵀ·S
    return o


def _delta_chunk(q, k, v, g, beta, C: Qwen35Cfg, chunk=64):
    """Chunk-parallel WY-representation gated delta rule (== HF torch_chunk_gated_delta_rule)."""
    S, H = q.shape[0], q.shape[1]
    q = np.transpose(_l2norm(q), (1, 0, 2)).astype(np.float32)   # [H,S,dk]
    k = np.transpose(_l2norm(k), (1, 0, 2)).astype(np.float32)
    v = np.transpose(v, (1, 0, 2)).astype(np.float32)            # [H,S,dv]
    beta = np.transpose(beta, (1, 0)).astype(np.float32)         # [H,S]
    g = np.transpose(g, (1, 0)).astype(np.float32)
    pad = (chunk - S % chunk) % chunk
    Sp = S + pad
    padv = lambda x: np.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
    q = padv(q) * (1.0 / np.sqrt(C.k_dim))
    k, v = padv(k), padv(v)
    beta, g = np.pad(beta, [(0, 0), (0, pad)]), np.pad(g, [(0, 0), (0, pad)])
    vb, kb = v * beta[..., None], k * beta[..., None]
    nc = Sp // chunk
    rs = lambda x: x.reshape(H, nc, chunk, x.shape[-1])
    q, k, v, kb, vb = rs(q), rs(k), rs(v), rs(kb), rs(vb)
    g = g.reshape(H, nc, chunk)
    tri0 = np.triu(np.ones((chunk, chunk), bool), 0)
    g = np.cumsum(g, -1)
    diff = g[..., :, None] - g[..., None, :]
    dmask = np.tril(np.exp(np.tril(diff)))
    attn = np.where(tri0, 0.0, -((kb @ np.swapaxes(k, -1, -2)) * dmask))
    for i in range(1, chunk):
        row = attn[..., i, :i].copy()
        sub = attn[..., :i, :i].copy()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + np.eye(chunk, dtype=attn.dtype)
    u = attn @ vb
    k_cumdecay = attn @ (kb * np.exp(g)[..., None])
    state = np.zeros((H, C.k_dim, C.v_dim), np.float32)
    o = np.zeros((H, nc, chunk, C.v_dim), np.float32)
    for i in range(nc):
        qi, ki, ui = q[:, i], k[:, i], u[:, i]
        a_intra = (qi @ np.swapaxes(ki, -1, -2)) * dmask[:, i]
        v_new = ui - k_cumdecay[:, i] @ state
        a_inter = (qi * np.exp(g[:, i])[..., None]) @ state
        o[:, i] = a_inter + a_intra @ v_new
        glast = g[:, i, -1]
        state = state * np.exp(glast)[:, None, None] + \
            np.swapaxes(ki * np.exp(glast[:, None] - g[:, i])[..., None], -1, -2) @ v_new
    o = o.reshape(H, nc * chunk, C.v_dim)[:, :S]
    return np.transpose(o, (1, 0, 2)).astype(np.float32)


def _linear_attention(x, d, C: Qwen35Cfg, delta):
    S = x.shape[0]
    KDIM, VDIM = C.key_dim, C.value_dim
    mixed = _conv1d_causal(x @ d["qkv"].T, d["conv"], S)
    q = mixed[:, :KDIM].reshape(S, C.n_k_heads, C.k_dim)
    k = mixed[:, KDIM:2 * KDIM].reshape(S, C.n_k_heads, C.k_dim)
    v = mixed[:, 2 * KDIM:].reshape(S, C.n_v_heads, C.v_dim)
    z = (x @ d["z"].T).reshape(S, C.n_v_heads, C.v_dim)
    beta = _sigmoid(x @ d["pb"].T)                                             # [S, n_v]
    g = (d["Alog"].astype(np.float32) * _softplus((x @ d["pa"].T) + d["dt"])).astype(f32)  # Alog = -exp(A_log)
    if C.n_v_heads // C.n_k_heads > 1:
        rep = C.n_v_heads // C.n_k_heads
        q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
    core = (_delta_chunk if delta == "chunk" else _delta_recurrent)(q, k, v, g, beta, C)
    core = core.reshape(-1, C.v_dim)
    zf = z.reshape(-1, C.v_dim)
    normed = (d["gn"] * (core * (1.0 / np.sqrt((core * core).mean(-1, keepdims=True) + C.eps)))) * _silu(zf)
    return normed.reshape(S, VDIM) @ d["out"].T


def forward(W, C: Qwen35Cfg, ids, delta="chunk"):
    """Full text forward. W: {'embed','final_norm','lm_head','layers':[...]}. Returns per-stage dict."""
    ids = np.asarray(ids, np.int64)
    S = len(ids)
    cos, sin = _rope_tables(C, S)
    h = W["embed"][ids].astype(np.float32)
    ckpt = {"embed": h.copy(), "layers": []}
    for d in W["layers"]:
        res = h
        x = rmsnorm(h, d["in_ln"], C.eps)
        mix = _full_attention(x, d, C, cos, sin) if d["type"] == "full_attention" else _linear_attention(x, d, C, delta)
        h = res + mix
        res = h
        x = rmsnorm(h, d["post_ln"], C.eps)
        h = res + (_silu(x @ d["gate"].T) * (x @ d["up"].T)) @ d["down"].T
        ckpt["layers"].append(h.copy())
    h = rmsnorm(h, W["final_norm"], C.eps)
    ckpt["finalnorm"] = h.copy()
    ckpt["logits"] = h @ W["lm_head"].T
    return ckpt
