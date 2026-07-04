#!/usr/bin/env python3
"""Numpy reference forward pass from a bitgpu manifest (step 3 of 3: the oracle).

Implements the full model math on CPU straight from the manifest (binary matmul,
RMSNorm, q/k norm, RoPE via the baked cos/sin caches, GQA attention, SwiGLU,
lm_head) and checks it against golden.py's logits. --dump writes the per-stage
checkpoint fixtures (params.json + embed/layer0/finalnorm/logits .bin) that
examples/verify.html reads from test-fixtures/forward/ - regenerate those when
bringing a new model, or the verify page checks yours against Bonsai's numbers.

Usage: python tools/reference.py --model <dir> [--golden <dir>] [--dump test-fixtures/forward]
Requires: numpy.
"""
import argparse
import json

import numpy as np

NP_DT = {"FLOAT": np.float32, "FLOAT16": np.float16, "UINT8": np.uint8, "INT64": np.int64, "INT32": np.int32}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--dump", default=None, help="write per-stage checkpoint fixtures here")
    args = ap.parse_args()
    work = args.model
    golden = args.golden or f"{work}/golden"

    M = json.load(open(f"{work}/manifest.json"))
    A = M["arch"]
    data = open(f"{work}/{M['data_file']}", "rb")
    aux = open(f"{work}/{M['aux_file']}", "rb").read()
    T = M["tensors"]

    def raw(ref: dict) -> np.ndarray:
        if ref["src"] == "aux":
            b = aux[ref["off"]:ref["off"] + ref["len"]]
        else:
            data.seek(ref["off"])
            b = data.read(ref["len"])
        return np.frombuffer(b, NP_DT[ref["dtype"]])

    tgt = {"tgt2": raw(M["luts"]["tgt2"]).reshape(256, 2), "tgt4": raw(M["luts"]["tgt4"]).reshape(256, 4)}

    def unpack(packed: np.ndarray, lut_name: str, bits: int) -> np.ndarray:
        """Lookup-expand [rows, cols] bytes to per-element codes [rows, cols*8/bits]."""
        per = 8 // bits
        exp = tgt[lut_name][packed].reshape(packed.shape[0], -1)  # [rows, *]
        out = np.empty((exp.shape[0], exp.shape[1] * per), np.uint8)
        for k in range(per):
            out[:, k::per] = (exp >> (bits * k)) & ((1 << bits) - 1)
        return out

    def dequant(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        """Dequantize a quantized matrix to fp32 [N, K] (optionally only the given row indices)."""
        t = T[name]
        N = t.get("N") or t["rows"]
        K = t.get("K") or t["cols"]
        bits, lut = t["bits"], t["lut"]
        wq = raw(t["weight"]).reshape(N, K // 8)
        scales = raw(t["scales"]).astype(np.float32).reshape(N, K // 128)
        zpw = raw(t["zp"])
        per = 8 // bits
        nb = K // 128
        zpw = zpw.reshape(N, nb // per)
        if rows is not None:
            wq, scales, zpw = wq[rows], scales[rows], zpw[rows]
        codes = unpack(wq, lut, bits)                                   # [n, K]
        zp = np.empty((wq.shape[0], nb), np.float32)
        for b in range(nb):
            zp[:, b] = (zpw[:, b // per] >> (bits * (b % per))) & ((1 << bits) - 1)
        bidx = np.arange(K) // 128
        return (codes.astype(np.float32) - zp[:, bidx]) * scales[:, bidx]

    def norm_w(name: str) -> np.ndarray:
        return raw(T[name]["weight"]).astype(np.float32)

    def rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return (x / np.sqrt(np.mean(x.astype(np.float32) ** 2, -1, keepdims=True) + A["rms_eps"])) * g

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    H, KV, D = A["heads"], A["kv_heads"], A["head_dim"]
    L = A["layers"]
    ids = np.load(f"{golden}/input_ids.npy")
    S = len(ids)

    # RoPE cos/sin (YaRN baked) -> full [S, D] via concat([half, half])
    cos_c = raw(T["cos_cache"]).reshape(T["cos_cache"]["shape"])
    sin_c = raw(T["sin_cache"]).reshape(T["sin_cache"]["shape"])
    cos = np.concatenate([cos_c[:S], cos_c[:S]], -1).astype(np.float32)  # [S, D]
    sin = np.concatenate([sin_c[:S], sin_c[:S]], -1).astype(np.float32)

    def rope(x: np.ndarray) -> np.ndarray:  # x: [S, n_heads, D]
        half = D // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
        return x * cos[:, None, :] + rot * sin[:, None, :]

    # embedding lookup (4-bit), no scaling for Qwen3
    h = dequant("embed_tokens", rows=ids).astype(np.float32)            # [S, hidden]
    ckpt = {"embed": h.copy()}

    causal = np.triu(np.full((S, S), -1e30, np.float32), 1)             # [S,S] upper-tri masked
    for li in range(L):
        res = h
        x = rmsnorm(h, norm_w(f"layers.{li}.input_layernorm"))
        q = (x @ dequant(f"layers.{li}.attn.q_proj").T).reshape(S, H, D)
        k = (x @ dequant(f"layers.{li}.attn.k_proj").T).reshape(S, KV, D)
        v = (x @ dequant(f"layers.{li}.attn.v_proj").T).reshape(S, KV, D)
        q = rope(rmsnorm(q, norm_w(f"layers.{li}.attn.q_norm")))
        k = rope(rmsnorm(k, norm_w(f"layers.{li}.attn.k_norm")))
        k = np.repeat(k, H // KV, axis=1)                               # GQA expand -> [S,H,D]
        v = np.repeat(v, H // KV, axis=1)
        out = np.empty((S, H, D), np.float32)
        for hh in range(H):
            sc = (q[:, hh] @ k[:, hh].T) / np.sqrt(D) + causal          # [S,S]
            sc -= sc.max(-1, keepdims=True)
            p = np.exp(sc)
            p /= p.sum(-1, keepdims=True)
            out[:, hh] = p @ v[:, hh]
        attn = out.reshape(S, H * D) @ dequant(f"layers.{li}.attn.o_proj").T
        h = res + attn
        res = h
        x = rmsnorm(h, norm_w(f"layers.{li}.post_attention_layernorm"))
        gate = x @ dequant(f"layers.{li}.mlp.gate_proj").T
        up = x @ dequant(f"layers.{li}.mlp.up_proj").T
        h = res + (silu(gate) * up) @ dequant(f"layers.{li}.mlp.down_proj").T
        if li == 0:
            ckpt["layer0"] = h.copy()

    h = rmsnorm(h, norm_w("layers.28.final_norm_layernorm"))
    ckpt["finalnorm"] = h.copy()
    logits = h @ dequant("lm_head").T                                   # [S, vocab]
    ckpt["logits"] = logits

    if args.dump:
        import os
        os.makedirs(args.dump, exist_ok=True)
        np.asarray(ids, np.int32).tofile(f"{args.dump}/ids.i32.bin")
        for name, arr in ckpt.items():
            arr.astype(np.float32).tofile(f"{args.dump}/{name}.bin")
        json.dump({"S": int(S), "hidden": A["hidden"], "vocab": A["vocab"], "ids": [int(i) for i in ids]},
                  open(f"{args.dump}/params.json", "w"), indent=1)
        print("dumped checkpoints to", args.dump)

    # compare to golden
    ref = np.load(f"{golden}/logits_all.npy").astype(np.float32)
    last = logits[-1]
    gl = ref[-1]
    cos_last = float(np.dot(last, gl) / (np.linalg.norm(last) * np.linalg.norm(gl) + 1e-9))
    print("S =", S, "| our argmax:", int(last.argmax()), "| golden argmax:", int(gl.argmax()))
    print("last-token logits cosine vs golden:", round(cos_last, 6))
    print("max abs logit diff (last token):", round(float(np.abs(last - gl).max()), 4))
    print("mean abs logit diff (all tokens):", round(float(np.abs(logits - ref).mean()), 4))
    print("RESULT:", "MATCH" if cos_last > 0.999 and int(last.argmax()) == int(gl.argmax()) else "MISMATCH")


if __name__ == "__main__":
    main()
