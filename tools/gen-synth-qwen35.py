#!/usr/bin/env python3
"""Generate a TINY synthetic 1-bit qwen35 model (random weights) to validate the engine's hybrid
forward end-to-end against the numpy oracle on the 8 GB laptop (the real Bonsai-27B is 3.8 GB and
won't fit). Emits bitgpu's native format directly (manifest.json + aux.bin + data.bin, no GGUF
step) plus the golden fixtures (ids/logits/per-layer) the verify page checks. Dims deliberately
use value_heads > key_heads (GQA repeat), partial RoPE, and all layer types.

    python tools/gen-synth-qwen35.py [out_dir]     # default examples/model-synth-qwen35

Requires numpy + tools/qwen35_numpy.
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen35_numpy as qn  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "examples", "model-synth-qwen35")
os.makedirs(OUT, exist_ok=True)
np.random.seed(1)

# --- tiny config (mirrors the 27B structure; value_heads=2*key_heads for the GQA-repeat path) ---
HID = 128
LAYERS = 4
INTERVAL = 4                 # full attention at layer 3
INTER = 256
NH, NKV, HDIM = 4, 2, 64     # full attention (q_proj doubled -> 2*NH*HDIM)
ROT = 32                     # partial rotary (of HDIM=64)
NK, NV, SDIM = 2, 4, 64      # linear: key/value heads, shared head dim (rep = NV/NK = 2)
CONVK = 4
VOCAB = 128
EPS = 1e-6
THETA = 1e6
KEYDIM, VALDIM = SDIM * NK, SDIM * NV
CONVDIM = KEYDIM * 2 + VALDIM

# --- Q1_0 dequant recipe (matches the kernels / tools/reference.py) ---
def gguf_tgt2():
    aux = np.zeros(512, np.uint8)
    for b in range(256):
        for j in range(8):
            aux[b * 2 + (j >> 2)] |= (3 if (b >> j) & 1 else 1) << (2 * (j % 4))
    return aux.reshape(256, 2)
TGT2 = gguf_tgt2()

def unpack(signs):  # [rows, K/8] bytes -> [rows, K] 2-bit codes in {1,3}
    exp = TGT2[signs].reshape(signs.shape[0], -1)
    out = np.empty((exp.shape[0], exp.shape[1] * 4), np.uint8)
    for k in range(4):
        out[:, k::4] = (exp >> (2 * k)) & 3
    return out

data = bytearray()
tensors = {}

def add_q1(name, N, K):
    """Random Q1_0 tensor: append 18-byte blocks to data, return dequantized fp32 [N,K]."""
    nb = K // 128
    signs = np.random.randint(0, 256, (N, nb * 16), np.uint8)
    scales = (np.random.randn(N, nb) * 0.1).astype(np.float16)
    codes = unpack(signs)                                       # [N, K]
    weight = (codes.astype(np.float32) - 2.0) * scales.astype(np.float32)[:, np.arange(K) // 128]
    blocks = np.zeros((N, nb, 18), np.uint8)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(N, nb, 2)
    blocks[:, :, 2:] = signs.reshape(N, nb, 16)
    off = len(data)
    data.extend(blocks.tobytes())
    tensors[name] = {"kind": "binary", "N": N, "K": K, "block": 128, "bits": 2, "lut": "tgt2",
                     "container": "q1_0",
                     "weight": {"dtype": "UINT8", "shape": [N, nb * 18], "src": "data", "off": off, "len": N * nb * 18}}
    return weight

def add_f32(name, arr, shape):
    off = len(data)
    b = np.asarray(arr, np.float32).ravel().tobytes()
    data.extend(b)
    tensors[name] = {"kind": "f32", "weight": {"dtype": "FLOAT", "shape": shape, "src": "data", "off": off, "len": len(b)}}
    return np.asarray(arr, np.float32)

# --- build weights (append in a fixed order = data-file layout) ---
W = {"layers": []}
W["embed"] = add_q1("embed_tokens", VOCAB, HID)   # embed as q4? engine uses embed_tokens kind q4; keep q1_0 recipe
layer_types = []
for li in range(LAYERS):
    full = li % INTERVAL == INTERVAL - 1
    layer_types.append("full" if full else "linear")
    d = {"type": "full_attention" if full else "linear_attention"}
    # plain norms are stored RAW here; the engine bakes +1 at load and qwen35_numpy applies (1+w).
    d["in_ln"] = add_f32(f"layers.{li}.input_layernorm", np.random.randn(HID) * 0.1, [HID])
    d["post_ln"] = add_f32(f"layers.{li}.post_attention_layernorm", np.random.randn(HID) * 0.1, [HID])
    d["gate"] = add_q1(f"layers.{li}.mlp.gate_proj", INTER, HID)
    d["up"] = add_q1(f"layers.{li}.mlp.up_proj", INTER, HID)
    d["down"] = add_q1(f"layers.{li}.mlp.down_proj", HID, INTER)
    if full:
        d["q"] = add_q1(f"layers.{li}.attn.q_proj", NH * HDIM * 2, HID)
        d["k"] = add_q1(f"layers.{li}.attn.k_proj", NKV * HDIM, HID)
        d["v"] = add_q1(f"layers.{li}.attn.v_proj", NKV * HDIM, HID)
        d["o"] = add_q1(f"layers.{li}.attn.o_proj", HID, NH * HDIM)
        d["qn"] = add_f32(f"layers.{li}.attn.q_norm", np.random.randn(HDIM) * 0.1, [HDIM])
        d["kn"] = add_f32(f"layers.{li}.attn.k_norm", np.random.randn(HDIM) * 0.1, [HDIM])
    else:
        d["qkv"] = add_q1(f"layers.{li}.linear.in_qkv", KEYDIM * 2 + VALDIM, HID)
        d["z"] = add_q1(f"layers.{li}.linear.z", VALDIM, HID)
        d["pb"] = add_q1(f"layers.{li}.linear.b", NV, HID)
        d["pa"] = add_q1(f"layers.{li}.linear.a", NV, HID)
        conv = np.random.randn(CONVDIM, CONVK).astype(np.float32) * 0.3   # [C, K] row-major = kernel w[c*K+j]
        add_f32(f"layers.{li}.linear.conv1d", conv, [CONVK, CONVDIM])
        d["conv"] = conv[:, None, :]                                       # [C,1,K] for qwen35_numpy
        d["Alog"] = add_f32(f"layers.{li}.linear.A_log", np.random.randn(NV) * 0.5, [NV])
        d["dt"] = add_f32(f"layers.{li}.linear.dt_bias", np.random.randn(NV) * 0.5, [NV])
        d["gn"] = add_f32(f"layers.{li}.linear.norm", np.random.randn(SDIM) * 0.1, [SDIM])
        d["out"] = add_q1(f"layers.{li}.linear.out_proj", HID, VALDIM)
    W["layers"].append(d)
W["final_norm"] = add_f32(f"layers.{LAYERS}.final_norm_layernorm", np.random.randn(HID) * 0.1, [HID])
W["lm_head"] = add_q1("lm_head", VOCAB, HID)
tensors["embed_tokens"]["kind"] = "q4"; tensors["embed_tokens"]["bits"] = 4; tensors["embed_tokens"]["lut"] = "tgt4"
tensors["embed_tokens"]["rows"] = VOCAB; tensors["embed_tokens"]["cols"] = HID
del tensors["embed_tokens"]["N"], tensors["embed_tokens"]["K"]
tensors["lm_head"]["kind"] = "q2"

manifest = {
    "version": 2, "data_file": "data.bin", "aux_file": "aux.bin",
    "arch": {"model_type": "qwen3_5", "layers": LAYERS, "hidden": HID, "intermediate": INTER,
             "heads": NH, "kv_heads": NKV, "head_dim": HDIM, "rms_eps": EPS,
             "rope": {"rope_theta": THETA}, "max_positions": 4096, "vocab": VOCAB, "eos": VOCAB - 1,
             "tie_word_embeddings": False, "act": "silu",
             "hybrid": {"layer_types": layer_types, "linear_key_heads": NK, "linear_value_heads": NV,
                        "linear_head_dim": SDIM, "conv_kernel": CONVK, "rotary_dim": ROT}},
    "luts": {"tgt2": {"dtype": "UINT8", "shape": [256, 2], "src": "aux", "off": 0, "len": 512},
             "tgt4": {"dtype": "UINT8", "shape": [256, 4], "src": "aux", "off": 512, "len": 1024}},
    "tensors": tensors,
}

# aux = the two LUTs (byte-identical to ggufLuts / the committed aux files)
aux = np.zeros(512 + 1024, np.uint8)
for b in range(256):
    for j in range(8):
        bit = (b >> j) & 1
        aux[b * 2 + (j >> 2)] |= (3 if bit else 1) << (2 * (j % 4))
        aux[512 + b * 4 + (j >> 1)] |= (9 if bit else 7) << (4 * (j % 2))

open(os.path.join(OUT, "data.bin"), "wb").write(bytes(data))
open(os.path.join(OUT, "aux.bin"), "wb").write(aux.tobytes())
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

# --- golden via the numpy oracle on the SAME dequantized weights ---
C = qn.Qwen35Cfg(hidden=HID, n_layers=LAYERS, eps=EPS, n_heads=NH, n_kv_heads=NKV, head_dim=HDIM,
                 rot_dim=ROT, rope_theta=THETA, n_k_heads=NK, n_v_heads=NV, k_dim=SDIM, v_dim=SDIM, conv_kernel=CONVK)
ids = np.array([3, 7, 11, 42, 5, 99, 1, 60, 33, 88], np.int32) % VOCAB
ck = qn.forward(W, C, ids, delta="recurrent")
os.makedirs(os.path.join(OUT, "..", "..", "test-fixtures", "forward-synth-qwen35"), exist_ok=True)
fx = os.path.abspath(os.path.join(OUT, "..", "..", "test-fixtures", "forward-synth-qwen35"))
os.makedirs(fx, exist_ok=True)
ids.tofile(os.path.join(fx, "ids.i32.bin"))
ck["embed"].astype(np.float32).tofile(os.path.join(fx, "embed.bin"))
ck["layers"][0].astype(np.float32).tofile(os.path.join(fx, "layer0.bin"))
ck["finalnorm"].astype(np.float32).tofile(os.path.join(fx, "finalnorm.bin"))
ck["logits"].astype(np.float32).tofile(os.path.join(fx, "logits.bin"))
json.dump({"S": len(ids), "hidden": HID, "vocab": VOCAB, "ids": ids.tolist(),
           "argmax_last": int(ck["logits"][-1].argmax())},
          open(os.path.join(fx, "params.json"), "w"), indent=1)
print(f"synth qwen35: {LAYERS} layers ({layer_types}) hidden={HID} vocab={VOCAB} rep={NV // NK}")
print(f"  data.bin={len(data)}B  manifest tensors={len(tensors)}  ids={ids.tolist()}")
print(f"  golden argmax(last)={int(ck['logits'][-1].argmax())}  logits |mean|={float(np.abs(ck['logits']).mean()):.4f}")
print(f"  wrote {OUT} + fixtures {fx}")
