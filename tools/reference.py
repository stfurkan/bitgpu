#!/usr/bin/env python3
"""Numpy reference forward pass from a bitgpu manifest (step 3 of 3: the oracle).

Implements the full model math on CPU straight from the manifest (binary matmul,
RMSNorm, q/k norm, RoPE, GQA attention, SwiGLU, lm_head) and checks it against
golden.py's logits. --dump writes the per-stage checkpoint fixtures (params.json +
embed/layer0/finalnorm/logits .bin) that examples/verify.html reads from
test-fixtures/forward-<tag>/ - regenerate those when bringing a new model, or the
verify page checks yours against Bonsai's numbers.

Reads both manifest containers: planar ONNX-derived tensors and v2 `q1_0` GGUF
containers (convert-gguf.py). RoPE comes from the baked cos/sin caches when present,
otherwise it is synthesized from arch.rope with the same recipe the engine runs.

GGUF-derived models have no onnxruntime oracle (golden.py needs the .onnx graph): the
golden comparison auto-skips when the golden dir is absent, and --ids supplies the
prompt (e.g. an existing fixture set's ids.i32.bin, so the checkpoints are directly
comparable across containers).

Usage: python tools/reference.py --model <dir> [--golden <dir>] [--ids <ids.i32.bin|.npy>]
                                 [--dump test-fixtures/forward-<tag>]
Requires: numpy.
"""
import argparse
import json
import math
import os

import numpy as np

NP_DT = {"FLOAT": np.float32, "FLOAT16": np.float16, "UINT8": np.uint8, "INT64": np.int64, "INT32": np.int32}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--ids", default=None, help="prompt ids (.i32.bin or .npy) when there is no golden dir")
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
        nb = K // 128
        bidx = np.arange(K) // 128
        if t.get("container") == "q1_0":
            # GGUF Q1_0 region: [f16 scale][16 sign bytes] per 128-weight block, interleaved.
            # De-interleave to the planar layout, then run the exact same LUT-expansion math;
            # the container carries no zp tensor (the recipe midpoint is a constant).
            r = t["weight"]
            data.seek(r["off"])
            blocks = np.frombuffer(data.read(N * nb * 18), np.uint8).reshape(N * nb, 18)
            wq = blocks[:, 2:].reshape(N, K // 8)
            scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(N, nb)
            if rows is not None:
                wq, scales = wq[rows], scales[rows]
            codes = unpack(wq, lut, bits)
            return (codes.astype(np.float32) - ((1 << bits) >> 1)) * scales[:, bidx]
        wq = raw(t["weight"]).reshape(N, K // 8)
        scales = raw(t["scales"]).astype(np.float32).reshape(N, K // 128)
        zpw = raw(t["zp"])
        per = 8 // bits
        zpw = zpw.reshape(N, nb // per)
        if rows is not None:
            wq, scales, zpw = wq[rows], scales[rows], zpw[rows]
        codes = unpack(wq, lut, bits)                                   # [n, K]
        zp = np.empty((wq.shape[0], nb), np.float32)
        for b in range(nb):
            zp[:, b] = (zpw[:, b // per] >> (bits * (b % per))) & ((1 << bits) - 1)
        return (codes.astype(np.float32) - zp[:, bidx]) * scales[:, bidx]

    def norm_w(name: str) -> np.ndarray:
        return raw(T[name]["weight"]).astype(np.float32)

    def rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return (x / np.sqrt(np.mean(x.astype(np.float32) ** 2, -1, keepdims=True) + A["rms_eps"])) * g

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    H, KV, D = A["heads"], A["kv_heads"], A["head_dim"]
    L = A["layers"]
    if args.ids:
        ids = np.fromfile(args.ids, np.int32) if args.ids.endswith(".bin") else np.load(args.ids)
    else:
        ids = np.load(f"{golden}/input_ids.npy")
    S = len(ids)

    # RoPE cos/sin [S, head_dim/2]: baked caches when the manifest carries them, otherwise
    # synthesized from arch.rope (plain or YaRN) - the same recipe the engine's synthRope
    # runs (f64 angles, one f32 rounding per entry; transformers YaRN: beta 32/1,
    # mscale = 0.1*ln(factor)+1).
    if "cos_cache" in T:
        cos_c = raw(T["cos_cache"]).reshape(T["cos_cache"]["shape"])[:S]
        sin_c = raw(T["sin_cache"]).reshape(T["sin_cache"]["shape"])[:S]
    else:
        rp = A["rope"]
        base = float(rp["rope_theta"])
        factor = float(rp.get("factor", 1.0)) if rp.get("rope_type") == "yarn" else 1.0
        half = D // 2
        inv = np.empty(half, np.float64)
        if factor == 1.0:
            inv[:] = 1.0 / base ** (np.arange(0, D, 2, dtype=np.float64) / D)
        else:
            orig = rp["original_max_position_embeddings"]
            lo = max(0, math.floor(D * math.log(orig / (32 * 2 * math.pi)) / (2 * math.log(base))))
            hi = min(half - 1, math.ceil(D * math.log(orig / (2 * math.pi)) / (2 * math.log(base))))
            pf = base ** (np.arange(0, D, 2, dtype=np.float64) / D)
            ramp = np.clip((np.arange(half, dtype=np.float64) - lo) / (hi - lo), 0, 1)
            inv[:] = (1.0 / (factor * pf)) * ramp + (1.0 / pf) * (1 - ramp)
        mscale = 1.0 if factor == 1.0 else float(np.float32(0.1 * math.log(factor) + 1))
        ang = np.outer(np.arange(S, dtype=np.float64), inv)
        cos_c = (np.cos(ang) * mscale).astype(np.float32)
        sin_c = (np.sin(ang) * mscale).astype(np.float32)
    cos = np.concatenate([cos_c, cos_c], -1).astype(np.float32)          # [S, D]
    sin = np.concatenate([sin_c, sin_c], -1).astype(np.float32)

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

    h = rmsnorm(h, norm_w(f"layers.{L}.final_norm_layernorm"))
    ckpt["finalnorm"] = h.copy()
    logits = h @ dequant("lm_head").T                                   # [S, vocab]
    ckpt["logits"] = logits

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        np.asarray(ids, np.int32).tofile(f"{args.dump}/ids.i32.bin")
        for name, arr in ckpt.items():
            arr.astype(np.float32).tofile(f"{args.dump}/{name}.bin")
        params = {"S": int(S), "hidden": A["hidden"], "vocab": A["vocab"], "ids": [int(i) for i in ids]}
        # Preserve a previously recorded known_good greedy continuation: regenerating fixtures
        # must not silently strip the verify page's bit-exact id gate.
        try:
            params["known_good"] = json.load(open(f"{args.dump}/params.json"))["known_good"]
        except (FileNotFoundError, KeyError, ValueError):
            print("NOTE: no known_good in", f"{args.dump}/params.json;",
                  "record the engine's greedy continuation there (see tools/README.md)")
        json.dump(params, open(f"{args.dump}/params.json", "w"), indent=1)
        print("dumped checkpoints to", args.dump)

    # compare to golden (auto-skips for models without an onnxruntime oracle, e.g. GGUF-derived)
    if not os.path.exists(f"{golden}/logits_all.npy"):
        print("golden comparison SKIPPED (no", f"{golden}/logits_all.npy);",
              "for GGUF-derived models compare the dumped checkpoints against the ONNX-derived fixture set instead")
        return
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
