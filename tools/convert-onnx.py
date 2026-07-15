#!/usr/bin/env python3
"""Convert a 1-bit ONNX export into a bitgpu model (manifest.json + aux file).

bitgpu deliberately does not parse ONNX at runtime: this tool walks the graph ONCE,
offline, and writes the two small files the engine loads instead:

  - manifest.json  arch config + every tensor mapped to a byte range (logical names)
  - <aux file>     the small tensors that are inline in the graph (norm gammas, the
                   tgt2/tgt4 lookup tables)

Big tensors (packed weights / scales / zero points / rope caches) are REFERENCED at
their byte offsets inside the export's external data file, which is used unchanged,
so the multi-hundred-MB weights file needs no rehosting: `createEngine` can point
`dataUrl` straight at the original hosting (e.g. the Hugging Face Hub) and
`manifestUrl`/`auxUrl` at wherever you serve these two small files.

Compatibility envelope: exports produced with the onnx-community 1-bit ("q1") recipe
for Qwen3-family models (MatMulNBits binary linears + GatherBlockQuantized embedding,
silu/SwiGLU, head_dim <= 128, 128-wide scale blocks). The engine validates the
manifest loudly at load, so an incompatible model fails with a clear error, not
garbage output.

Usage:
  python tools/convert-onnx.py --model <dir> [--out <dir>] [--onnx model_q1.onnx]
                          [--aux-name model_q1.aux.bin] [--ref-url <safetensors url>]

<dir> must contain config.json, the .onnx graph file, and its external data file.
Requires: numpy, onnx (pip install numpy onnx).
"""
import argparse
import json
import re

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

NORM_OPS = {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"}


def logical_linear(quant_name: str) -> str:
    """model_layers_0_attn_q_proj_MatMul_weight_quant -> layers.0.attn.q_proj"""
    m = re.match(r"model_layers_(\d+)_(attn|mlp)_(.+?)_MatMul_weight_quant", quant_name)
    if not m:
        raise ValueError(f"unrecognized linear weight name: {quant_name}")
    return f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}"


def logical_norm(gamma_name: str) -> str:
    """model.layers.0.attn.q_norm.layernorm.weight -> layers.0.attn.q_norm ; model.norm.weight -> norm"""
    return gamma_name.removeprefix("model.").removesuffix(".weight").replace(".layernorm", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir with config.json + the .onnx graph + its external data file")
    ap.add_argument("--out", default=None, help="output dir (default: --model)")
    ap.add_argument("--onnx", default="model_q1.onnx", help="graph filename inside --model")
    ap.add_argument("--aux-name", default=None, help="aux filename to write (default: <graph stem>.aux.bin)")
    ap.add_argument("--ref-url", default=None, help="optional f16/f32 safetensors URL for a network round-trip check")
    args = ap.parse_args()
    out = args.out or args.model
    aux_name = args.aux_name or (args.onnx.removesuffix(".onnx") + ".aux.bin")

    cfg = json.load(open(f"{args.model}/config.json"))
    graph = onnx.load(f"{args.model}/{args.onnx}", load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}

    aux = bytearray()
    data_file: list[str] = []

    def place(name: str) -> dict:
        """Reference an initializer: external -> point into the data file; inline -> copy to aux.
        Always records dtype + shape so the loader is self-describing."""
        t = inits[name]
        info = {"dtype": TensorProto.DataType.Name(t.data_type), "shape": list(t.dims)}
        if t.data_location == TensorProto.EXTERNAL:
            d = {x.key: x.value for x in t.external_data}
            if d["location"] not in data_file:
                data_file.append(d["location"])
            info.update({"src": "data", "off": int(d["offset"]), "len": int(d["length"])})
        else:
            arr = numpy_helper.to_array(t)
            b = arr.tobytes()
            off = len(aux)
            aux.extend(b)
            info.update({"src": "aux", "off": off, "len": len(b)})
        return info

    tensors: dict[str, dict] = {}

    # linear weights + lm_head (MatMulNBits)
    for n in graph.node:
        if n.op_type != "MatMulNBits":
            continue
        scales, zp = n.input[2], n.input[3]
        K = next(a.i for a in n.attribute if a.name == "K")
        N = next(a.i for a in n.attribute if a.name == "N")
        # The packed-weight initializer is always the scales name with _weight_scales ->
        # _weight_quant, for the lm_head too: tied exports feed it the embedding stream
        # (model_embed_tokens_weight_scales -> model_embed_tokens_weight_quant) while untied
        # ones (e.g. Bonsai-8B) carry their own lm_head_MatMul_weight_quant.
        wq = scales.replace("_weight_scales", "_weight_quant")
        if wq not in inits:
            raise ValueError(f"packed weight initializer {wq} (derived from {scales}) not found in the graph")
        if n.name == "/lm_head/MatMul_Quant":
            name, kind, lut = "lm_head", "q2", "tgt2"
            tied = wq == "model_embed_tokens_weight_quant"
            if bool(cfg["tie_word_embeddings"]) != tied:
                raise ValueError(f"lm_head weight {wq} contradicts config tie_word_embeddings={cfg['tie_word_embeddings']}")
        else:
            name, kind, lut = logical_linear(wq), "binary", "tgt2"
        tensors[name] = {"kind": kind, "N": N, "K": K, "block": 128, "bits": 2, "lut": lut,
                         "weight": place(wq), "scales": place(scales), "zp": place(zp)}

    # input embedding (GatherBlockQuantized, 4-bit view of the tied table)
    gbq = next(n for n in graph.node if n.op_type == "GatherBlockQuantized")
    bits = next(a.i for a in gbq.attribute if a.name == "bits")
    tensors["embed_tokens"] = {
        "kind": "q4", "rows": cfg["vocab_size"], "cols": cfg["hidden_size"], "block": 128, "bits": bits, "lut": "tgt4",
        "weight": place("model_embed_tokens_weight_quant"),
        "scales": place("model_embed_tokens_weight_scales"),
        "zp": place("model_embed_tokens_weight_zp_4b"),
    }

    # RMSNorm gammas
    for n in graph.node:
        if n.op_type not in NORM_OPS:
            continue
        gamma = next((i for i in n.input if i in inits), None)
        if gamma is None:
            continue
        tensors[logical_norm(gamma)] = {"kind": "f32", "weight": place(gamma)}

    # RoPE caches (may be YaRN-scaled; referenced as-is for exact parity)
    for c in ("cos_cache", "sin_cache"):
        if c in inits:
            tensors[c] = {"kind": "f32", **place(c)}

    # the two lookup tables
    luts = {"tgt2": place("unpack_lut_src1_tgt2"), "tgt4": place("unpack_lut_src1_tgt4")}

    if len(data_file) != 1:
        raise ValueError(f"expected exactly one external data file, saw: {data_file}")

    manifest = {
        "version": 1,
        "data_file": data_file[0],
        "aux_file": aux_name,
        "arch": {
            "model_type": cfg["model_type"], "layers": cfg["num_hidden_layers"],
            "hidden": cfg["hidden_size"], "intermediate": cfg["intermediate_size"],
            "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"], "rms_eps": cfg["rms_norm_eps"],
            "rope": cfg.get("rope_parameters", {"rope_theta": 1e6}),
            "vocab": cfg["vocab_size"], "eos": cfg["eos_token_id"],
            "tie_word_embeddings": cfg["tie_word_embeddings"], "act": cfg["hidden_act"],
        },
        "luts": luts,
        "tensors": tensors,
    }

    open(f"{out}/{aux_name}", "wb").write(aux)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    print(f"manifest: {len(tensors)} tensors | {aux_name}: {len(aux)/1e6:.3f} MB")
    kinds: dict[str, int] = {}
    for t in tensors.values():
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print("tensor kinds:", kinds)

    _selfcheck(args.model, manifest, aux, args.ref_url)


def _selfcheck(model_dir: str, manifest: dict, aux: bytes, ref_url: str | None) -> None:
    """Decode one weight row straight from the manifest refs (the exact math the engine's
    shaders implement) and sanity-check it; with --ref-url, also cosine-check against the
    full-precision safetensors row fetched over HTTP ranges."""
    data = open(f"{model_dir}/{manifest['data_file']}", "rb")

    def read(ref: dict) -> bytes:
        if ref["src"] == "aux":
            return bytes(aux[ref["off"]:ref["off"] + ref["len"]])
        data.seek(ref["off"])
        return data.read(ref["len"])

    tgt2 = np.frombuffer(read(manifest["luts"]["tgt2"]), np.uint8).reshape(256, 2)
    t = manifest["tensors"]["layers.0.attn.q_proj"]
    wq = np.frombuffer(read(t["weight"]), np.uint8).reshape(t["N"], t["K"] // 8)[:1]
    sc = np.frombuffer(read(t["scales"]), np.float32).reshape(t["N"], t["K"] // 128)[0]
    zp = np.frombuffer(read(t["zp"]), np.uint8).reshape(t["N"], t["K"] // 128 // 4)[0]
    exp = tgt2[wq].reshape(1, -1)
    c = np.empty((1, t["K"]), np.uint8)
    for k in range(4):
        c[:, k::4] = (exp >> (2 * k)) & 3
    dec = np.empty(t["K"], np.float32)
    for b in range(t["K"] // 128):
        z = (zp[b // 4] >> (2 * (b % 4))) & 3
        dec[b * 128:(b + 1) * 128] = (c[0, b * 128:(b + 1) * 128].astype(np.int32) - int(z)) * sc[b]
    if not np.isfinite(dec).all() or float(np.abs(dec).max()) == 0.0:
        raise ValueError("round-trip decode produced degenerate values; the export layout does not match")
    print("round-trip self-check (q_proj L0 via manifest): decoded", t["K"], "finite weights OK")

    if not ref_url:
        return
    import struct
    import urllib.request

    def net(o: int, l: int) -> bytes:
        req = urllib.request.Request(ref_url, headers={"Range": f"bytes={o}-{o + l - 1}"})
        return urllib.request.urlopen(req, timeout=120).read()

    hlen = struct.unpack("<Q", net(0, 8))[0]
    hdr = json.loads(net(8, hlen).decode())
    o0, _ = hdr["model.layers.0.self_attn.q_proj.weight"]["data_offsets"]
    ref = np.frombuffer(net(8 + hlen + o0, t["K"] * 2), np.float16).astype(np.float32)
    cos = float(np.dot(dec, ref) / (np.linalg.norm(dec) * np.linalg.norm(ref) + 1e-9))
    print("round-trip self-check cosine vs reference:", round(cos, 5))
    if cos < 0.999:
        raise ValueError(f"decoded row does not match the reference (cosine {cos}); wrong export or layout")


if __name__ == "__main__":
    main()
