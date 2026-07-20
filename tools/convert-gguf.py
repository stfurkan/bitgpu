#!/usr/bin/env python3
"""Convert a 1-bit GGUF (PrismML Q1_0) into a bitgpu model (manifest.json + aux file).

Like convert-onnx.py, this walks the container ONCE, offline, and writes the two small
files the engine loads - while the GGUF itself is used byte-for-byte unchanged as the
data file, so it can keep streaming from wherever it is already hosted (e.g. the
Hugging Face Hub).

GGUF Q1_0 stores each 128-weight block as 18 interleaved bytes ([f16 scale] then
[16 sign bytes], LSB-first, bit=1 -> +1), where bitgpu's kernels want planar sign bytes
and planar f32 scales (the layout the ONNX exports carry). Instead of repacking the
weights offline (which would force rehosting), the manifest marks these tensors
`container: "q1_0"` (manifest version 2) and the ENGINE de-interleaves the stream
in-flight at load into exactly the buffers the ONNX path builds. The sign bit stream
is identical across the two containers (verified bit-for-bit on Bonsai-8B), so a
GGUF-derived model carries the same weights as its ONNX-derived sibling.

GGUF bakes no RoPE tables; the manifest carries the rope parameters (YaRN or plain)
from the GGUF metadata instead, and the engine (and tools/reference.py) synthesize the
f32 cos/sin tables at load. The aux file therefore holds only the two lookup tables
(~1.5 KB).

Compatibility envelope: architecture `qwen3` with every linear + embedding + lm_head
in Q1_0 (ggml tensor type 41) and F32 norms - the PrismML Bonsai 1-bit releases. The
engine validates the manifest loudly at load, so anything else fails with a clear
error, not garbage output.

Usage:
  python tools/convert-gguf.py --gguf <file.gguf> [--out <dir>]
                               [--aux-name <name>.aux.bin] [--ref-url <safetensors url>]
Requires: numpy.
"""
import argparse
import json
import math
import os
import struct

import numpy as np

F32, Q1_0 = 0, 41  # ggml tensor types
BLK = 18  # Q1_0: 2-byte f16 scale + 16 sign bytes per 128 weights

LINEAR = {
    "attn_q": "attn.q_proj", "attn_k": "attn.k_proj", "attn_v": "attn.v_proj",
    "attn_output": "attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
NORM = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "attn.q_norm", "attn_k_norm": "attn.k_norm",
}


def read_gguf_header(path: str):
    """Parse the GGUF v3 header: metadata dict + tensor directory + data-section offset."""
    buf = open(path, "rb").read(64 * 1024 * 1024)  # headers are a few MB (tokenizer arrays)
    if buf[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    pos = [4]

    def u32():
        v = struct.unpack_from("<I", buf, pos[0])[0]; pos[0] += 4; return v

    def u64():
        v = struct.unpack_from("<Q", buf, pos[0])[0]; pos[0] += 8; return v

    def gstr():
        n = u64(); s = buf[pos[0]:pos[0] + n].decode("utf-8"); pos[0] += n; return s

    def value(t):
        p = pos[0]
        if t == 0: pos[0] += 1; return buf[p]
        if t == 1: pos[0] += 1; return struct.unpack_from("<b", buf, p)[0]
        if t == 2: pos[0] += 2; return struct.unpack_from("<H", buf, p)[0]
        if t == 3: pos[0] += 2; return struct.unpack_from("<h", buf, p)[0]
        if t == 4: return u32()
        if t == 5: pos[0] += 4; return struct.unpack_from("<i", buf, p)[0]
        if t == 6: pos[0] += 4; return struct.unpack_from("<f", buf, p)[0]
        if t == 7: pos[0] += 1; return bool(buf[p])
        if t == 8: return gstr()
        if t == 9:
            et, n = u32(), u64()
            return [value(et) for _ in range(n)]
        if t == 10: return u64()
        if t == 11: pos[0] += 8; return struct.unpack_from("<q", buf, p)[0]
        if t == 12: pos[0] += 8; return struct.unpack_from("<d", buf, p)[0]
        raise ValueError(f"unknown GGUF value type {t}")

    version = u32()
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version} (this tool implements v3)")
    n_tensors, n_kv = u64(), u64()
    meta = {}
    for _ in range(n_kv):
        k = gstr(); t = u32(); meta[k] = value(t)
    tensors = {}
    for _ in range(n_tensors):
        name = gstr(); nd = u32()
        dims = [u64() for _ in range(nd)]
        ty = u32(); off = u64()
        tensors[name] = {"dims": dims, "type": ty, "off": off}
    align = int(meta.get("general.alignment", 32))
    data_start = (pos[0] + align - 1) // align * align
    return meta, tensors, data_start


def luts() -> tuple[np.ndarray, np.ndarray]:
    """The two expansion tables the kernels use, generated from their defining property:
    LSB-first sign bits -> per-weight codes around the recipe midpoints (2-bit: {1,3},
    4-bit: {7,9}). Byte-identical to the tables the ONNX exports carry."""
    tgt2 = np.zeros((256, 2), np.uint8)
    tgt4 = np.zeros((256, 4), np.uint8)
    for b in range(256):
        for j in range(8):
            bit = (b >> j) & 1
            tgt2[b, j // 4] |= (3 if bit else 1) << (2 * (j % 4))
            tgt4[b, j // 2] |= (9 if bit else 7) << (4 * (j % 2))
    return tgt2, tgt4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to the .gguf file")
    ap.add_argument("--out", default=None, help="output dir (default: the gguf's dir)")
    ap.add_argument("--aux-name", default=None, help="aux filename (default: <gguf stem>.aux.bin)")
    ap.add_argument("--ref-url", default=None, help="optional f16/f32 safetensors URL for a network round-trip check")
    ap.add_argument("--no-check", action="store_true", help="skip the round-trip self-check (e.g. when the .gguf is a header-only slice)")
    args = ap.parse_args()
    out = args.out or (os.path.dirname(args.gguf) or ".")
    stem = os.path.splitext(os.path.basename(args.gguf))[0]
    aux_name = args.aux_name or f"{stem}.aux.bin"

    meta, gg, data_start = read_gguf_header(args.gguf)
    arch_name = meta["general.architecture"]
    if arch_name not in ("qwen3", "qwen35"):
        raise ValueError(f"unsupported architecture '{arch_name}' (bitgpu kernels implement qwen3 + the qwen35 hybrid)")
    P = lambda k: meta[f"{arch_name}.{k}"]

    head_dim = int(P("attention.key_length"))
    if head_dim != int(P("attention.value_length")):
        raise ValueError("key_length != value_length (kernels assume one head_dim)")
    layers = int(P("block_count"))
    vocab = int(gg["token_embd.weight"]["dims"][1])
    tied = "output.weight" not in gg
    rope: dict = {"rope_theta": float(P("rope.freq_base"))}
    if meta.get(f"{arch_name}.rope.scaling.type") == "yarn":
        rope.update({
            "rope_type": "yarn",
            "factor": float(P("rope.scaling.factor")),
            "original_max_position_embeddings": int(P("rope.scaling.original_context_length")),
        })
    elif meta.get(f"{arch_name}.rope.scaling.type") not in (None, "none"):
        raise ValueError(f"unsupported rope scaling '{meta.get(f'{arch_name}.rope.scaling.type')}'")

    def region(gname: str, N: int, K: int) -> dict:
        """The interleaved Q1_0 byte range of a tensor inside the GGUF (used unchanged)."""
        t = gg[gname]
        if t["type"] != Q1_0:
            raise ValueError(f"{gname}: expected ggml type 41 (Q1_0), got {t['type']}")
        if K % 128 != 0:
            raise ValueError(f"{gname}: K={K} not a multiple of the 128-wide blocks")
        gk, gn = (int(t["dims"][0]), int(t["dims"][1])) if len(t["dims"]) == 2 else (int(t["dims"][0]), 1)
        if (gk, gn) != (K, N):
            raise ValueError(f"{gname}: dims {t['dims']} do not match expected [{K}, {N}]")
        ln = N * (K // 128) * BLK
        return {"dtype": "UINT8", "shape": [N, (K // 128) * BLK], "src": "data",
                "off": data_start + int(t["off"]), "len": ln}

    def norm_ref(gname: str, n: int) -> dict:
        t = gg[gname]
        if t["type"] != F32:
            raise ValueError(f"{gname}: expected F32 norm, got ggml type {t['type']}")
        if [int(d) for d in t["dims"]] != [n]:
            raise ValueError(f"{gname}: dims {t['dims']} != [{n}]")
        return {"dtype": "FLOAT", "shape": [n], "src": "data", "off": data_start + int(t["off"]), "len": n * 4}

    hidden, inter = int(P("embedding_length")), int(P("feed_forward_length"))
    heads, kv_heads = int(P("attention.head_count")), int(P("attention.head_count_kv"))
    tensors: dict[str, dict] = {}
    hybrid = None
    if arch_name == "qwen35":
        # qwen3.5 hybrid: 3:1 gated-DeltaNet linear attention + gated full attention. Mirrors
        # buildQwen35Manifest in src/gguf.ts; GGUF reuses ggml's Mamba/SSM slots for the linear dims.
        def bin_t(gname: str, N: int, K: int) -> dict:
            return {"kind": "binary", "N": N, "K": K, "block": 128, "bits": 2, "lut": "tgt2",
                    "container": "q1_0", "weight": region(gname, N, K)}

        def f32_t(gname: str, dims: list[int]) -> dict:
            t = gg[gname]
            if t["type"] != F32:
                raise ValueError(f"{gname}: expected F32, got ggml type {t['type']}")
            if [int(d) for d in t["dims"]] != dims:
                raise ValueError(f"{gname}: dims {t['dims']} != {dims}")
            n = 1
            for d in dims:
                n *= d
            return {"kind": "f32", "weight": {"dtype": "FLOAT", "shape": dims, "src": "data",
                                              "off": data_start + int(t["off"]), "len": n * 4}}

        interval = int(P("full_attention_interval"))
        nk, nv = int(P("ssm.group_count")), int(P("ssm.time_step_rank"))
        state, convk = int(P("ssm.state_size")), int(P("ssm.conv_kernel"))
        key_dim, val_dim = state * nk, state * nv
        if val_dim != int(P("ssm.inner_size")):
            raise ValueError(f"ssm.inner_size {P('ssm.inner_size')} != value_heads*state_size {val_dim}")
        layer_types: list[str] = []
        for li in range(layers):
            full = li % interval == interval - 1  # full attention at 3,7,11,...
            layer_types.append("full" if full else "linear")
            tensors[f"layers.{li}.input_layernorm"] = f32_t(f"blk.{li}.attn_norm.weight", [hidden])
            tensors[f"layers.{li}.post_attention_layernorm"] = f32_t(f"blk.{li}.post_attention_norm.weight", [hidden])
            tensors[f"layers.{li}.mlp.gate_proj"] = bin_t(f"blk.{li}.ffn_gate.weight", inter, hidden)
            tensors[f"layers.{li}.mlp.up_proj"] = bin_t(f"blk.{li}.ffn_up.weight", inter, hidden)
            tensors[f"layers.{li}.mlp.down_proj"] = bin_t(f"blk.{li}.ffn_down.weight", hidden, inter)
            if full:
                tensors[f"layers.{li}.attn.q_proj"] = bin_t(f"blk.{li}.attn_q.weight", heads * head_dim * 2, hidden)
                tensors[f"layers.{li}.attn.k_proj"] = bin_t(f"blk.{li}.attn_k.weight", kv_heads * head_dim, hidden)
                tensors[f"layers.{li}.attn.v_proj"] = bin_t(f"blk.{li}.attn_v.weight", kv_heads * head_dim, hidden)
                tensors[f"layers.{li}.attn.o_proj"] = bin_t(f"blk.{li}.attn_output.weight", hidden, heads * head_dim)
                tensors[f"layers.{li}.attn.q_norm"] = f32_t(f"blk.{li}.attn_q_norm.weight", [head_dim])
                tensors[f"layers.{li}.attn.k_norm"] = f32_t(f"blk.{li}.attn_k_norm.weight", [head_dim])
            else:
                tensors[f"layers.{li}.linear.in_qkv"] = bin_t(f"blk.{li}.attn_qkv.weight", key_dim * 2 + val_dim, hidden)
                tensors[f"layers.{li}.linear.z"] = bin_t(f"blk.{li}.attn_gate.weight", val_dim, hidden)
                tensors[f"layers.{li}.linear.a"] = bin_t(f"blk.{li}.ssm_alpha.weight", nv, hidden)
                tensors[f"layers.{li}.linear.b"] = bin_t(f"blk.{li}.ssm_beta.weight", nv, hidden)
                tensors[f"layers.{li}.linear.conv1d"] = f32_t(f"blk.{li}.ssm_conv1d.weight", [convk, key_dim * 2 + val_dim])
                tensors[f"layers.{li}.linear.A_log"] = f32_t(f"blk.{li}.ssm_a", [nv])
                tensors[f"layers.{li}.linear.dt_bias"] = f32_t(f"blk.{li}.ssm_dt.bias", [nv])
                tensors[f"layers.{li}.linear.norm"] = f32_t(f"blk.{li}.ssm_norm.weight", [state])
                tensors[f"layers.{li}.linear.out_proj"] = bin_t(f"blk.{li}.ssm_out.weight", hidden, val_dim)
        hybrid = {
            "layer_types": layer_types, "linear_key_heads": nk, "linear_value_heads": nv,
            "linear_head_dim": state, "conv_kernel": convk, "rotary_dim": int(P("rope.dimension_count")),
        }
    else:
        for li in range(layers):
            for gk, lk in LINEAR.items():
                N, K = {
                    "attn_q": (heads * head_dim, hidden), "attn_k": (kv_heads * head_dim, hidden),
                    "attn_v": (kv_heads * head_dim, hidden), "attn_output": (hidden, heads * head_dim),
                    "ffn_gate": (inter, hidden), "ffn_up": (inter, hidden), "ffn_down": (hidden, inter),
                }[gk]
                tensors[f"layers.{li}.{lk}"] = {
                    "kind": "binary", "N": N, "K": K, "block": 128, "bits": 2, "lut": "tgt2",
                    "container": "q1_0", "weight": region(f"blk.{li}.{gk}.weight", N, K),
                }
            for gk, lk in NORM.items():
                n = head_dim if "q_norm" in gk or "k_norm" in gk else hidden
                tensors[f"layers.{li}.{lk}"] = {"kind": "f32", "weight": norm_ref(f"blk.{li}.{gk}.weight", n)}
    tensors[f"layers.{layers}.final_norm_layernorm"] = {"kind": "f32", "weight": norm_ref("output_norm.weight", hidden)}
    tensors["embed_tokens"] = {
        "kind": "q4", "rows": vocab, "cols": hidden, "block": 128, "bits": 4, "lut": "tgt4",
        "container": "q1_0", "weight": region("token_embd.weight", vocab, hidden),
    }
    tensors["lm_head"] = {
        "kind": "q2", "N": vocab, "K": hidden, "block": 128, "bits": 2, "lut": "tgt2",
        "container": "q1_0", "weight": region("token_embd.weight" if tied else "output.weight", vocab, hidden),
    }

    tgt2, tgt4 = luts()
    aux = tgt2.tobytes() + tgt4.tobytes()
    manifest = {
        "version": 2,
        "data_file": os.path.basename(args.gguf),
        "aux_file": aux_name,
        "arch": {
            "model_type": "qwen3_5" if arch_name == "qwen35" else arch_name,
            "layers": layers, "hidden": hidden, "intermediate": inter,
            "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
            "rms_eps": float(P("attention.layer_norm_rms_epsilon")),
            "rope": rope, "max_positions": int(P("context_length")),
            "vocab": vocab, "eos": int(meta["tokenizer.ggml.eos_token_id"]),
            "tie_word_embeddings": tied, "act": "silu",
            **({"hybrid": hybrid} if hybrid else {}),
        },
        "luts": {
            "tgt2": {"dtype": "UINT8", "shape": [256, 2], "src": "aux", "off": 0, "len": 512},
            "tgt4": {"dtype": "UINT8", "shape": [256, 4], "src": "aux", "off": 512, "len": 1024},
        },
        "tensors": tensors,
    }

    open(f"{out}/{aux_name}", "wb").write(aux)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    kinds: dict[str, int] = {}
    for t in tensors.values():
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"manifest: {len(tensors)} tensors (v2, q1_0 containers, {'tied' if tied else 'untied'} lm_head) | {aux_name}: {len(aux)} B")
    print("tensor kinds:", kinds, "| rope:", rope, f"| max_positions: {manifest['arch']['max_positions']}")

    if not args.no_check:
        _selfcheck(args.gguf, manifest, tgt2, args.ref_url)


def _selfcheck(gguf_path: str, manifest: dict, tgt2: np.ndarray, ref_url: str | None) -> None:
    """Decode one weight row two independent ways - straight Q1_0 dequant (sign bit -> +/-scale)
    and the engine's route (de-interleave, tgt2 code expansion, (code-2)*scale) - and require
    bit-identical results; with --ref-url, also cosine-check against the full-precision
    safetensors row fetched over HTTP ranges."""
    t = next(v for v in manifest["tensors"].values() if v.get("kind") == "binary")  # hybrid layer 0 has no attn.q_proj
    K = t["K"]
    nb = K // 128
    f = open(gguf_path, "rb")
    f.seek(t["weight"]["off"])
    row = f.read(nb * BLK)

    # path 1: direct Q1_0 semantics
    direct = np.empty(K, np.float32)
    signs = np.empty(nb * 16, np.uint8)
    scales = np.empty(nb, np.float32)
    for b in range(nb):
        blk = row[b * BLK:(b + 1) * BLK]
        scales[b] = np.float32(np.frombuffer(blk[:2], np.float16)[0])
        signs[b * 16:(b + 1) * 16] = np.frombuffer(blk[2:], np.uint8)
        e = np.unpackbits(signs[b * 16:(b + 1) * 16], bitorder="little")
        direct[b * 128:(b + 1) * 128] = np.where(e == 1, scales[b], -scales[b])

    # path 2: the engine's math (de-interleaved sign bytes through the tgt2 LUT)
    exp = tgt2[signs].reshape(-1)
    codes = np.empty(K, np.uint8)
    for k in range(4):
        codes[k::4] = (exp >> (2 * k)) & 3
    routed = (codes.astype(np.float32) - 2) * np.repeat(scales, 128)

    if not np.array_equal(direct, routed):
        raise ValueError("self-check FAILED: direct Q1_0 dequant != engine-route dequant (layout drift)")
    if not np.isfinite(direct).all() or float(np.abs(direct).max()) == 0.0:
        raise ValueError("round-trip decode produced degenerate values; the file does not match Q1_0")
    print(f"round-trip self-check (q_proj L0 via manifest): {K} weights, direct == engine route, all finite OK")

    if not ref_url:
        return
    import urllib.request

    def net(o: int, l: int) -> bytes:
        req = urllib.request.Request(ref_url, headers={"Range": f"bytes={o}-{o + l - 1}"})
        return urllib.request.urlopen(req, timeout=120).read()

    hlen = struct.unpack("<Q", net(0, 8))[0]
    hdr = json.loads(net(8, hlen).decode())
    o0, _ = hdr["model.layers.0.self_attn.q_proj.weight"]["data_offsets"]
    ref = np.frombuffer(net(8 + hlen + o0, K * 2), np.float16).astype(np.float32)
    cos = float(np.dot(direct, ref) / (np.linalg.norm(direct) * np.linalg.norm(ref) + 1e-9))
    print("round-trip self-check cosine vs reference:", round(cos, 5))
    if cos < 0.999:
        raise ValueError(f"decoded row does not match the reference (cosine {cos}); wrong file or layout")


if __name__ == "__main__":
    main()
