// bitgpu/gguf - point the engine at any 1-bit Q1_0 GGUF URL with NO offline step.
//
// `fromGguf(url)` fetches ONLY the GGUF header (HTTP ranges, a few MB - never the weights),
// parses it, and returns `{ manifest, aux, dataUrl }` ready to spread into createEngine:
//
//   import { createEngine } from 'bitgpu'
//   import { fromGguf } from 'bitgpu/gguf'
//   const engine = await createEngine({ ...(await fromGguf(GGUF_URL)), kvCache: 'q8' })
//
// This is tools/convert-gguf.py running in the browser: the same header walk, the same
// tensor mapping, the same validations, producing a manifest deep-equal to the converter's
// output (verified by scripts/verify-gguf.ts against the committed models/ manifests). The
// aux LUTs are derived, not stored (generated from their defining property, byte-identical
// to the tables the ONNX exports carry), so a GGUF model needs zero side files.
//
// Compatibility envelope (identical to the converter): architecture `qwen3`, every linear +
// embedding + lm_head in Q1_0 (ggml tensor type 41), F32 norms, plain or YaRN rope - the
// PrismML Bonsai 1-bit releases. Anything else fails loudly here, not as garbage output.
import type { Manifest, ManifestRef, ManifestTensor } from './types'

export interface GgufModel {
  manifest: Manifest
  aux: Uint8Array
  /** The GGUF URL itself - the weights stream from it byte-for-byte unchanged. */
  dataUrl: string
}

export interface FromGgufOptions {
  /** Fetch `len` bytes at `off`. Override for caching or non-HTTP sources. Default: `fetch`
   *  with a `Range` header (and a stream-and-cancel fallback for servers that ignore ranges). */
  fetchRange?: (url: string, off: number, len: number) => Promise<ArrayBuffer>
  /** Parse abort cap: headers larger than this fail loudly. Default 64 MiB. */
  maxHeaderBytes?: number
}

const F32 = 0
const Q1_0 = 41 // ggml tensor types
const BLK = 18 // Q1_0: 2-byte f16 scale + 16 sign bytes per 128 weights
const FIRST_FETCH = 1 << 20 // 1 MiB - covers small headers in one round trip

const LINEAR: Record<string, string> = {
  attn_q: 'attn.q_proj',
  attn_k: 'attn.k_proj',
  attn_v: 'attn.v_proj',
  attn_output: 'attn.o_proj',
  ffn_gate: 'mlp.gate_proj',
  ffn_up: 'mlp.up_proj',
  ffn_down: 'mlp.down_proj',
}
const NORM: Record<string, string> = {
  attn_norm: 'input_layernorm',
  ffn_norm: 'post_attention_layernorm',
  attn_q_norm: 'attn.q_norm',
  attn_k_norm: 'attn.k_norm',
}

/** Sentinel: the parse ran past the fetched prefix - fetch a bigger one and retry. */
class OutOfData extends Error {}

interface GTensor {
  dims: number[]
  type: number
  off: number
}
interface Header {
  meta: Record<string, unknown>
  tensors: Record<string, GTensor>
  dataStart: number
}

function u64ToNumber(v: bigint, what: string): number {
  if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`bitgpu/gguf: ${what} ${v} exceeds safe integer range`)
  return Number(v)
}

/** Parse a GGUF v3 header from a fetched prefix; throws OutOfData if the prefix is too short. */
function parseHeader(buf: ArrayBuffer): Header {
  const dv = new DataView(buf)
  const u8 = new Uint8Array(buf)
  const td = new TextDecoder()
  let pos = 0
  const need = (n: number) => {
    if (pos + n > buf.byteLength) throw new OutOfData()
  }
  const u32 = () => {
    need(4)
    const v = dv.getUint32(pos, true)
    pos += 4
    return v
  }
  const u64 = (what: string) => {
    need(8)
    const v = dv.getBigUint64(pos, true)
    pos += 8
    return u64ToNumber(v, what)
  }
  const gstr = () => {
    const n = u64('string length')
    need(n)
    const s = td.decode(u8.subarray(pos, pos + n))
    pos += n
    return s
  }
  const value = (t: number): unknown => {
    switch (t) {
      case 0: need(1); return u8[pos++]
      case 1: need(1); return dv.getInt8(pos++)
      case 2: need(2); { const v = dv.getUint16(pos, true); pos += 2; return v }
      case 3: need(2); { const v = dv.getInt16(pos, true); pos += 2; return v }
      case 4: return u32()
      case 5: need(4); { const v = dv.getInt32(pos, true); pos += 4; return v }
      case 6: need(4); { const v = dv.getFloat32(pos, true); pos += 4; return v }
      case 7: need(1); return u8[pos++] !== 0
      case 8: return gstr()
      case 9: {
        const et = u32()
        const n = u64('array length')
        const out = new Array(n)
        for (let i = 0; i < n; i++) out[i] = value(et)
        return out
      }
      case 10: return u64('u64 value')
      case 11: need(8); { const v = dv.getBigInt64(pos, true); pos += 8; if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < -BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`bitgpu/gguf: i64 value ${v} exceeds safe integer range`); return Number(v) }
      case 12: need(8); { const v = dv.getFloat64(pos, true); pos += 8; return v }
      default: throw new Error(`bitgpu/gguf: unknown GGUF value type ${t}`)
    }
  }

  need(4)
  if (td.decode(u8.subarray(0, 4)) !== 'GGUF') throw new Error('bitgpu/gguf: not a GGUF file (bad magic)')
  pos = 4
  const version = u32()
  if (version !== 3) throw new Error(`bitgpu/gguf: unsupported GGUF version ${version} (this parser implements v3)`)
  const nTensors = u64('tensor count')
  const nKv = u64('kv count')
  const meta: Record<string, unknown> = {}
  for (let i = 0; i < nKv; i++) {
    const k = gstr()
    const t = u32()
    meta[k] = value(t)
  }
  const tensors: Record<string, GTensor> = {}
  for (let i = 0; i < nTensors; i++) {
    const name = gstr()
    const nd = u32()
    const dims: number[] = []
    for (let d = 0; d < nd; d++) dims.push(u64('tensor dim'))
    const type = u32()
    const off = u64('tensor offset')
    tensors[name] = { dims, type, off }
  }
  const align = Number(meta['general.alignment'] ?? 32)
  const dataStart = Math.ceil(pos / align) * align
  return { meta, tensors, dataStart }
}

/** Default ranged fetch. Servers that ignore `Range` (HTTP 200) are handled by reading the
 *  stream only up to the needed length and cancelling - the weights are never downloaded. */
async function fetchRangeHttp(url: string, off: number, len: number): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } })
  if (!res.ok) throw new Error(`bitgpu/gguf: fetch ${url} failed: HTTP ${res.status}`)
  if (res.status === 206) return res.arrayBuffer()
  // Range ignored: take the first off+len bytes off the stream, then cancel the rest.
  if (!res.body) throw new Error(`bitgpu/gguf: ${url} has no readable body`)
  const reader = res.body.getReader()
  const out = new Uint8Array(len)
  let seen = 0 // bytes of the FILE consumed so far
  let filled = 0 // bytes of `out` filled so far
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    const start = Math.max(off - seen, 0)
    if (value.byteLength > start) {
      const take = Math.min(value.byteLength - start, len - filled)
      out.set(value.subarray(start, start + take), filled)
      filled += take
    }
    seen += value.byteLength
    if (filled >= len) {
      reader.cancel().catch(() => {})
      break
    }
  }
  return out.buffer.slice(0, filled) as ArrayBuffer
}

/** The two expansion tables the kernels use, generated from their defining property:
 *  LSB-first sign bits -> per-weight codes around the recipe midpoints (2-bit: {1,3},
 *  4-bit: {7,9}). Byte-identical to the tables the ONNX exports carry. */
export function ggufLuts(): Uint8Array {
  const aux = new Uint8Array(512 + 1024) // tgt2 [256,2] then tgt4 [256,4]
  for (let b = 0; b < 256; b++) {
    for (let j = 0; j < 8; j++) {
      const bit = (b >> j) & 1
      aux[b * 2 + (j >> 2)] |= (bit ? 3 : 1) << (2 * (j % 4))
      aux[512 + b * 4 + (j >> 1)] |= (bit ? 9 : 7) << (4 * (j % 2))
    }
  }
  return aux
}

function basename(url: string): string {
  const path = url.split(/[?#]/)[0]
  return decodeURIComponent(path.slice(path.lastIndexOf('/') + 1)) || 'model.gguf'
}

/** Build the manifest from a parsed header - the exact mapping tools/convert-gguf.py performs. */
function buildManifest(h: Header, dataFile: string): Manifest {
  const { meta, tensors: gg, dataStart } = h
  const archName = meta['general.architecture'] as string
  if (archName !== 'qwen3')
    throw new Error(`bitgpu/gguf: unsupported architecture '${archName}' (bitgpu kernels implement the qwen3 topology)`)
  const P = (k: string): unknown => {
    const v = meta[`${archName}.${k}`]
    if (v === undefined) throw new Error(`bitgpu/gguf: header is missing ${archName}.${k}`)
    return v
  }

  const headDim = Number(P('attention.key_length'))
  if (headDim !== Number(P('attention.value_length')))
    throw new Error('bitgpu/gguf: key_length != value_length (kernels assume one head_dim)')
  const layers = Number(P('block_count'))
  const embd = gg['token_embd.weight']
  if (!embd) throw new Error('bitgpu/gguf: header has no token_embd.weight tensor')
  const vocab = embd.dims[1]
  const tied = !('output.weight' in gg)
  const scalingType = meta[`${archName}.rope.scaling.type`]
  const rope: NonNullable<Manifest['arch']['rope']> = { rope_theta: Number(P('rope.freq_base')) }
  if (scalingType === 'yarn') {
    rope.rope_type = 'yarn'
    rope.factor = Number(P('rope.scaling.factor'))
    rope.original_max_position_embeddings = Number(P('rope.scaling.original_context_length'))
  } else if (scalingType !== undefined && scalingType !== 'none') {
    throw new Error(`bitgpu/gguf: unsupported rope scaling '${scalingType}'`)
  }

  /** The interleaved Q1_0 byte range of a tensor inside the GGUF (used unchanged). */
  const region = (gname: string, N: number, K: number): ManifestRef => {
    const t = gg[gname]
    if (!t) throw new Error(`bitgpu/gguf: header has no ${gname} tensor`)
    if (t.type !== Q1_0) throw new Error(`bitgpu/gguf: ${gname}: expected ggml type 41 (Q1_0), got ${t.type}`)
    if (K % 128 !== 0) throw new Error(`bitgpu/gguf: ${gname}: K=${K} not a multiple of the 128-wide blocks`)
    const [gk, gn] = t.dims.length === 2 ? [t.dims[0], t.dims[1]] : [t.dims[0], 1]
    if (gk !== K || gn !== N) throw new Error(`bitgpu/gguf: ${gname}: dims [${t.dims}] do not match expected [${K}, ${N}]`)
    return { dtype: 'UINT8', shape: [N, (K / 128) * BLK], src: 'data', off: dataStart + t.off, len: N * (K / 128) * BLK }
  }
  const normRef = (gname: string, n: number): ManifestRef => {
    const t = gg[gname]
    if (!t) throw new Error(`bitgpu/gguf: header has no ${gname} tensor`)
    if (t.type !== F32) throw new Error(`bitgpu/gguf: ${gname}: expected F32 norm, got ggml type ${t.type}`)
    if (t.dims.length !== 1 || t.dims[0] !== n) throw new Error(`bitgpu/gguf: ${gname}: dims [${t.dims}] != [${n}]`)
    return { dtype: 'FLOAT', shape: [n], src: 'data', off: dataStart + t.off, len: n * 4 }
  }

  const hidden = Number(P('embedding_length'))
  const inter = Number(P('feed_forward_length'))
  const heads = Number(P('attention.head_count'))
  const kvHeads = Number(P('attention.head_count_kv'))
  const SHAPES: Record<string, [number, number]> = {
    attn_q: [heads * headDim, hidden],
    attn_k: [kvHeads * headDim, hidden],
    attn_v: [kvHeads * headDim, hidden],
    attn_output: [hidden, heads * headDim],
    ffn_gate: [inter, hidden],
    ffn_up: [inter, hidden],
    ffn_down: [hidden, inter],
  }
  const tensors: Record<string, ManifestTensor> = {}
  for (let li = 0; li < layers; li++) {
    for (const [gk, lk] of Object.entries(LINEAR)) {
      const [N, K] = SHAPES[gk]
      tensors[`layers.${li}.${lk}`] = {
        kind: 'binary', N, K, block: 128, bits: 2, lut: 'tgt2',
        container: 'q1_0', weight: region(`blk.${li}.${gk}.weight`, N, K),
      }
    }
    for (const [gk, lk] of Object.entries(NORM)) {
      const n = gk.includes('q_norm') || gk.includes('k_norm') ? headDim : hidden
      tensors[`layers.${li}.${lk}`] = { kind: 'f32', weight: normRef(`blk.${li}.${gk}.weight`, n) }
    }
  }
  tensors[`layers.${layers}.final_norm_layernorm`] = { kind: 'f32', weight: normRef('output_norm.weight', hidden) }
  tensors['embed_tokens'] = {
    kind: 'q4', rows: vocab, cols: hidden, block: 128, bits: 4, lut: 'tgt4',
    container: 'q1_0', weight: region('token_embd.weight', vocab, hidden),
  }
  tensors['lm_head'] = {
    kind: 'q2', N: vocab, K: hidden, block: 128, bits: 2, lut: 'tgt2',
    container: 'q1_0', weight: region(tied ? 'token_embd.weight' : 'output.weight', vocab, hidden),
  }

  const eos = meta['tokenizer.ggml.eos_token_id']
  if (eos === undefined) throw new Error('bitgpu/gguf: header is missing tokenizer.ggml.eos_token_id')
  const stem = dataFile.replace(/\.gguf$/i, '')
  return {
    version: 2,
    data_file: dataFile,
    aux_file: `${stem}.aux.bin`,
    arch: {
      model_type: archName, layers, hidden, intermediate: inter,
      heads, kv_heads: kvHeads, head_dim: headDim,
      rms_eps: Number(P('attention.layer_norm_rms_epsilon')),
      rope, max_positions: Number(P('context_length')),
      vocab, eos: Number(eos),
      tie_word_embeddings: tied, act: 'silu',
    },
    luts: {
      tgt2: { dtype: 'UINT8', shape: [256, 2], src: 'aux', off: 0, len: 512 },
      tgt4: { dtype: 'UINT8', shape: [256, 4], src: 'aux', off: 512, len: 1024 },
    },
    tensors,
  }
}

/** Parse a GGUF header already in memory (e.g. from a File). Grows nothing - the buffer must
 *  contain the full header (weights past it are not needed). */
export function fromGgufBytes(buf: ArrayBuffer, dataFile: string): { manifest: Manifest; aux: Uint8Array } {
  try {
    return { manifest: buildManifest(parseHeader(buf), dataFile), aux: ggufLuts() }
  } catch (e) {
    if (e instanceof OutOfData) throw new Error('bitgpu/gguf: buffer ends inside the GGUF header (pass more bytes)')
    throw e
  }
}

/** Fetch and parse a GGUF header; returns `{ manifest, aux, dataUrl }` for createEngine.
 *  Only the header is transferred, progressively: 1 MiB first, then EXTENSION ranges (2x
 *  growth) appended to the kept prefix - total transfer stays under 2x the header size. */
export async function fromGguf(url: string, options: FromGgufOptions = {}): Promise<GgufModel> {
  const fetchRange = options.fetchRange ?? fetchRangeHttp
  const cap = options.maxHeaderBytes ?? 64 << 20
  let size = FIRST_FETCH
  let have = new Uint8Array(await fetchRange(url, 0, size))
  for (;;) {
    try {
      // `have` always owns its exactly-sized buffer (fresh Uint8Array per round)
      const manifest = buildManifest(parseHeader(have.buffer as ArrayBuffer), basename(url))
      return { manifest, aux: ggufLuts(), dataUrl: url }
    } catch (e) {
      if (!(e instanceof OutOfData)) throw e
      if (have.byteLength < size) throw new Error(`bitgpu/gguf: ${url} ends inside its own header (truncated file?)`)
      if (size >= cap) throw new Error(`bitgpu/gguf: header exceeds maxHeaderBytes (${cap}); raise the cap if this is a real model`)
      const next = Math.min(size * 2, cap)
      const ext = new Uint8Array(await fetchRange(url, have.byteLength, next - have.byteLength))
      const grown = new Uint8Array(have.byteLength + ext.byteLength)
      grown.set(have, 0)
      grown.set(ext, have.byteLength)
      have = grown
      size = next
    }
  }
}
