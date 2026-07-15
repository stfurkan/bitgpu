# The bitgpu model format

A bitgpu model is three files. Two are tiny and produced by a converter - `tools/convert-onnx.py`
for 1-bit ONNX exports, `tools/convert-gguf.py` for 1-bit (Q1_0) GGUFs - and the third is the
original export's weights file (the `.onnx_data` or the `.gguf` itself), used byte-for-byte
unchanged, so it can stay wherever it is already hosted, e.g. the Hugging Face Hub:

| file | size | what |
|---|---|---|
| `manifest.json` | ~130 KB | architecture contract + every tensor mapped to a byte range |
| `<name>.aux.bin` | ~30 KB (ONNX) / ~1.5 KB (GGUF) | small tensors copied out of the graph (lookup tables; for ONNX also norm gammas) |
| the data file | large | the export's weights file, referenced by offset, never modified |

`createEngine` takes either a `modelUrl` (a directory holding all three) or explicit
`manifestUrl` / `auxUrl` / `dataUrl`, which lets the two small files live on your origin while
the big one streams from the model's original hosting.

## manifest.json

```jsonc
{
  "version": 1,                        // 1 = planar tensors; 2 = may carry q1_0 containers (below)
  "data_file": "model_q1.onnx_data",   // filename of the weights file (relative to modelUrl)
  "aux_file": "bonsai.aux.bin",        // filename of the aux file
  "arch": { /* architecture contract, below */ },
  "luts": { "tgt2": <ref>, "tgt4": <ref> },
  "tensors": { "<logical name>": <tensor>, ... }
}
```

### Refs

Every blob of bytes is a **ref**: `{ "dtype": "FLOAT"|"UINT8"|..., "shape": [...], "src":
"data"|"aux", "off": <byte offset>, "len": <byte length> }`. `src: "data"` points into the big
weights file; `src: "aux"` points into the aux file. The engine streams the data file through
these ranges in one pass at load (tied tensors may share the exact same range; partial overlaps
are invalid). `luts.tgt2` must live in the aux file: the loader needs it before the weights
stream begins.

### Tensor kinds

- `binary` - a 1-bit linear layer. `{ kind, N, K, block: 128, bits: 2, lut: "tgt2", weight:
  <ref [N, K/8] u8>, scales: <ref [N, K/128] f32>, zp: <ref> }`. Each weight byte expands
  through `tgt2` to 8 sign bits ({-1,+1}); a column is `sum(x[k] * sign[k]) * scale[k/128]`.
- `q2` - the 2-bit lm_head (`lm_head`). Same byte stream as the tied embedding; each byte
  expands through `tgt2` to four 2-bit codes, dequantized `(code - zp) * scale`.
- `q4` - the 4-bit input embedding (`embed_tokens`). `{ rows, cols, block: 128, bits: 4,
  lut: "tgt4", weight, scales, zp }`; rows are gathered and dequantized per prompt token.
- `f32` - plain float tensors: RMSNorm gammas (`{ kind, weight: <ref> }`) and the baked RoPE
  caches (`cos_cache` / `sin_cache`, refs at the top level of the entry).

### The q1_0 container (manifest version 2)

A version-2 manifest may mark any quantized tensor `"container": "q1_0"`: its `weight` ref then
covers the tensor's **interleaved GGUF Q1_0 region** - `N * K/128` blocks of 18 bytes, each
block a little-endian f16 scale followed by 16 sign bytes (LSB-first within a byte, bit 1 = +1) -
and the tensor carries **no** `scales`/`zp` refs. The loader de-interleaves the region while it
streams: sign bytes feed the same planar weight buffer (and transforms) the ONNX path uses,
scales convert f16 -> f32 into the scales buffer, and the zero-points are the 1-bit recipe's
constant midpoints (2 for 2-bit codes, 8 for the 4-bit embedding, synthesized at load). The
sign-byte stream is identical across the two containers - verified bit-for-bit on Bonsai-8B -
so a GGUF-derived model carries the same weights as its ONNX-derived sibling. Produced by
`tools/convert-gguf.py`, with the `.gguf` itself as the data file.

### RoPE: baked or synthesized

ONNX-derived manifests reference the export's baked `cos_cache`/`sin_cache` (exact parity with
the exporter, YaRN included). GGUF files bake no rope tables, so GGUF-derived manifests carry
the parameters instead - `arch.rope` (`rope_theta`, and for YaRN `rope_type: "yarn"`, `factor`,
`original_max_position_embeddings`) plus `arch.max_positions` (the model's context length,
which caps `maxSeqLen`) - and the engine synthesizes the f32 tables for exactly `maxSeqLen`
positions at load. `tools/reference.py` implements the identical recipe, so fixtures generated
for a GGUF-derived model check the math the engine actually runs.

### Logical names (the engine looks these up literally)

```
layers.<i>.attn.{q_proj,k_proj,v_proj,o_proj}      binary
layers.<i>.mlp.{gate_proj,up_proj,down_proj}       binary
layers.<i>.{input_layernorm,post_attention_layernorm}  f32
layers.<i>.attn.{q_norm,k_norm}                    f32
layers.<layers>.final_norm_layernorm               f32 (the final norm)
embed_tokens                                       q4
lm_head                                            q2
cos_cache, sin_cache                               f32
```

At load the engine fuses q/k/v and gate/up into single GPU buffers (streaming each part into
its slice), so the manifest keeps them separate.

### The arch contract

`arch` carries `layers, hidden, intermediate, heads, kv_heads, head_dim, rms_eps, vocab, eos,
act, rope, tie_word_embeddings`. The engine validates what its kernels assume and fails loudly
otherwise: `act` must be `silu` (SwiGLU), `head_dim` <= 128, every quantized tensor's `block`
must be 128, and a `q2` tensor's zero-points must be uniform (one value for the whole tensor;
the q1 recipe always emits the 2-bit midpoint, 2; the engine reads the zp tensor and derives
the value from it, so a non-uniform export fails at load instead of dequantizing wrong).
In practice that means **Qwen3-family models quantized with the onnx-community
1-bit ("q1") recipe**. A different attention/MLP topology needs kernel work, not just a
manifest.

## The aux file

Raw concatenated bytes, no header; the manifest's aux refs carry all structure. `tgt2`
(256x2 u8) expands a weight byte to 2-bit codes; `tgt4` (256x4 u8) to 4-bit codes. For
GGUF-derived models the aux file holds only these two tables (~1.5 KB, generated from their
defining property); the norm gammas stay in the GGUF as plain F32 ranges.

## Source of truth

The loader in [`src/engine.ts`](../src/engine.ts) (search for "streaming weight loader") is the
executable spec; `tools/convert-onnx.py` and `tools/convert-gguf.py` are the reference
producers. The shipped Bonsai manifests were byte-for-byte reproduced from the public exports
with these tools.
