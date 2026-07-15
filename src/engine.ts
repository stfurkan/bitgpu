// WebGPU runtime for 1-bit Qwen3-family models (Bonsai 1.7B/4B/8B and any q1 export whose
// manifest passes the load-time contract). Loads binary (1-bit) weights via the manifest,
// runs the forward with the validated kernels, keeps a persistent KV cache, and generates
// autoregressively. Decode is dispatch-overhead-bound, so the matmuls are fused (q/k/v in one
// dispatch, gate/up in one, the residual add folded into o_proj/down_proj) and the decode loop
// is GPU-resident (GPU argmax + embedding gather) with deferred CPU sync + pooled resources.
//
// This is a faithful port of the validated engine.js: the kernel sequence and numerics are
// unchanged (bit-exact). Only the shader source (now inlined, not fetched), the configuration
// (now typed options, not URL params), and the public surface differ.
import { SHADERS } from './shaders.generated'
import { GpuOutOfMemoryError, WebGPUUnavailableError } from './errors'
import { draftNgram, pldWorthIt, PLD_PROBATION } from './pld'
import { MT19937, affectedIds, ngramBans, sampleFromCandidates } from './sampler'
import type {
  DeviceLostInfo,
  Engine,
  EngineCapabilities,
  EngineOptions,
  ForwardResult,
  GenerateOptions,
  GenerateResult,
  KvSnapshot,
  Manifest,
  ManifestRef as Ref,
  ManifestTensor,
  TokenLogprobs,
} from './types'

type Field = ['f' | 'u', number]
// Manifest / ManifestTensor / ManifestArch / ManifestRef are public types now (src/types.ts):
// bitgpu/gguf builds manifests in memory, so callers can hold and pass them.

interface GpuWeight {
  buf?: GPUBuffer
  sign?: GPUBuffer
  scales?: GPUBuffer
  codes?: GPUBuffer
  N?: number
  K?: number
  nb?: number
  N0?: number
  N1?: number
  N2?: number
  zp?: number
}

interface RawGenResult {
  prefillMs: number
  decodeMs: number
  tokPerSec: number
  tokens: number[]
  firstArgmax: number
  recMs: number
  gpuMs: number
  rbMs: number
  spec?: { steps: number; drafted: number; accepted: number; bailed?: boolean }
  /** The sampler RNG as it stands after this call - promptLookup:'auto' hands it to the plain
   *  continuation so the draw stream continues exactly where the probation left off. */
  rng?: MT19937
  /** Per-emitted-token logprob records (sampled path only, when GenerateOptions.logprobs set). */
  lp?: TokenLogprobs[]
}

/** Internal engine handle: the public {@link Engine} surface plus diagnostics used by the
 *  correctness/benchmark harness. The diagnostics are intentionally not in the public type. */
interface EngineInternal extends Engine {
  device: GPUDevice
  adapter: GPUAdapter
  /** Raw decode with the per-kernel profiling switch (`full`) and sync depth exposed. */
  profileDecode(ids: number[], nTokens: number, full?: Set<string> | null, syncN?: number): Promise<RawGenResult>
  /** Differential debug: one decode step through the fast and slow paths, checkpoint by checkpoint. */
  debugDecode(prefillIds: number[]): Promise<{ fast: Record<string, Float32Array>; slow: Record<string, Float32Array> }>
  /** Debug: GPU base + penalized logits + top-K for a prefill, to diff the sampler kernels vs CPU math. */
  debugSampler(ids: number[], genOpts: GenerateOptions): Promise<{ base: Float32Array; penalized: Float32Array; candIds: Uint32Array; candVals: Float32Array }>
}

type TypedArrayCtor = Float32ArrayConstructor | Uint8ArrayConstructor | Uint16ArrayConstructor
const VIEW: Record<string, TypedArrayCtor> = { FLOAT: Float32Array, UINT8: Uint8Array, FLOAT16: Uint16Array }
const WGSLS = ['matmul_split', 'matmul_resid', 'matmul_q2', 'rope', 'swiglu', 'copy']
const DEFAULT_MAX_SEQ = 2048
const PREFILL_SEG = 256 // prefill segment length; see runPrefill for why prefills are segmented

const PARAM_AB = new ArrayBuffer(64)
const PARAM_DV = new DataView(PARAM_AB)
const PARAM_U8 = new Uint8Array(PARAM_AB)
function makeParams(fields: Field[]): Uint8Array {
  // fills a reused buffer (no per-dispatch alloc)
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i]
    if (f[0] === 'f') PARAM_DV.setFloat32(i * 4, f[1], true)
    else PARAM_DV.setUint32(i * 4, f[1] >>> 0, true)
  }
  return PARAM_U8.subarray(0, Math.ceil(fields.length / 4) * 16)
}
const eqBytes = (a: Uint8Array, b: Uint8Array): boolean => {
  for (let i = 0; i < b.length; i++) if (a[i] !== b[i]) return false
  return true
}
/** Synthesize f32 rope tables ([positions, head_dim/2]) from arch.rope for manifests without
 *  baked caches (v2/GGUF). Plain rope or YaRN (transformers formula: beta_fast 32, beta_slow 1,
 *  mscale = 0.1*ln(factor)+1). Angles accumulate in f64 and round once to f32 per entry -
 *  tools/reference.py implements the identical recipe for fixture generation. */
function synthRope(A: { head_dim: number; rope?: { rope_theta: number; rope_type?: string; factor?: number; original_max_position_embeddings?: number } }, positions: number): [Float32Array, Float32Array] {
  const half = A.head_dim / 2
  const rope = A.rope!
  const base = rope.rope_theta
  const factor = rope.rope_type === 'yarn' ? (rope.factor ?? 1) : 1
  const inv = new Float64Array(half)
  // YaRN ramp bounds (only used when factor > 1): dims below lo keep the original
  // frequencies (extrapolation), dims above hi interpolate by 1/factor, between them blends.
  const orig = rope.original_max_position_embeddings ?? 0
  const lo = factor === 1 ? 0 : Math.max(0, Math.floor((A.head_dim * Math.log(orig / (32 * 2 * Math.PI))) / (2 * Math.log(base))))
  const hi = factor === 1 ? 0 : Math.min(half - 1, Math.ceil((A.head_dim * Math.log(orig / (2 * Math.PI))) / (2 * Math.log(base))))
  for (let j = 0; j < half; j++) {
    const pf = base ** ((2 * j) / A.head_dim)
    if (factor === 1) {
      inv[j] = 1 / pf
      continue
    }
    const ramp = Math.min(1, Math.max(0, (j - lo) / (hi - lo)))
    inv[j] = (1 / (factor * pf)) * ramp + (1 / pf) * (1 - ramp)
  }
  const mscale = factor === 1 ? 1 : Math.fround(0.1 * Math.log(factor) + 1)
  const cos = new Float32Array(positions * half)
  const sin = new Float32Array(positions * half)
  for (let p = 0; p < positions; p++)
    for (let j = 0; j < half; j++) {
      const a = p * inv[j]
      cos[p * half + j] = Math.fround(Math.cos(a) * mscale)
      sin[p * half + j] = Math.fround(Math.sin(a) * mscale)
    }
  return [cos, sin]
}
/** Load a 1-bit model and return an {@link Engine}. Pass a model URL string for defaults. */
export async function createEngine(options: EngineOptions | string): Promise<Engine> {
  // Any load failure after the device exists (WGSL compile error, manifest validation, truncated
  // download, allocation OOM) must destroy the device, or the partially-loaded weights stay
  // resident until GC collects the GPUDevice - undermining the catch-OOM-and-retry-smaller advice
  // in errors.ts, since the retry contends with the dead engine's VRAM.
  const holder: { device?: GPUDevice } = {}
  try {
    return await createEngineInner(options, holder)
  } catch (e) {
    holder.device?.destroy()
    throw e
  }
}
async function createEngineInner(options: EngineOptions | string, holder: { device?: GPUDevice }): Promise<Engine> {
  const opts: EngineOptions = typeof options === 'string' ? { modelUrl: options } : options
  const modelDir = opts.modelUrl ? opts.modelUrl.replace(/\/$/, '') : null
  if (!modelDir && !opts.manifestUrl && !opts.manifest)
    throw new Error('createEngine: provide modelUrl, manifestUrl, or an in-memory manifest')
  if (opts.manifest && !modelDir && !opts.dataUrl)
    throw new Error('createEngine: an in-memory manifest needs dataUrl (or modelUrl) for the weights file')
  const powerPreference = opts.powerPreference ?? 'high-performance'
  const fetchJson =
    opts.fetchJson ??
    (async (url: string) => {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`bitgpu: fetch ${url} failed: HTTP ${res.status}`)
      if ((res.headers.get('content-type') ?? '').includes('text/html'))
        throw new Error(`bitgpu: ${url} returned HTML, not JSON (a SPA fallback is probably serving index.html for missing model files)`)
      return res.json()
    })
  const fetchBytes =
    opts.fetchArrayBuffer ??
    (async (url: string) => {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`bitgpu: fetch ${url} failed: HTTP ${res.status}`)
      const total = Number(res.headers.get('content-length') ?? 0)
      if (!res.body || !total) return res.arrayBuffer()
      // Stream so onProgress can report the (multi-hundred-MB) weights download.
      const reader = res.body.getReader()
      const chunks: Uint8Array[] = []
      let loaded = 0
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        loaded += value.byteLength
        opts.onProgress?.({ phase: 'weights', loaded, total })
      }
      const out = new Uint8Array(loaded)
      let p = 0
      for (const c of chunks) {
        out.set(c, p)
        p += c.byteLength
      }
      return out.buffer
    })

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new WebGPUUnavailableError('WebGPU is not available (no navigator.gpu). Use a WebGPU-capable browser over a secure context.')
  }

  opts.onProgress?.({ phase: 'manifest' })
  const manifest = opts.manifest ?? ((await fetchJson(opts.manifestUrl ?? `${modelDir}/manifest.json`)) as Manifest)
  opts.onProgress?.({ phase: 'weights' })
  const dataUrl = opts.dataUrl ?? `${modelDir}/${manifest.data_file}`
  // Only the SMALL aux file (LUTs, ~160KB) is buffered; the ~290MB data file STREAMS through
  // per-tensor routes straight into GPU buffers further down (see the streaming weight loader).
  let aux: ArrayBuffer
  if (opts.aux) {
    // In-memory aux (bitgpu/gguf computes the LUTs instead of fetching them). Normalize a view
    // to a tight ArrayBuffer so the offset math below stays identical to the fetched path.
    aux = opts.aux instanceof Uint8Array ? new Uint8Array(opts.aux).buffer : opts.aux
  } else {
    aux = await fetchBytes(opts.auxUrl ?? `${modelDir}/${manifest.aux_file}`)
  }
  const A = manifest.arch
  const T = manifest.tensors
  // Fail loud on manifests the kernels cannot run, instead of producing silent garbage: the WGSL
  // assumes silu activation, head_dim <= 128 (per-thread register arrays), and 128-wide scale blocks.
  const FINAL_NORM = `layers.${A.layers}.final_norm_layernorm`
  if (A.act !== 'silu') throw new Error(`bitgpu: unsupported activation '${A.act}' (kernels implement silu/SwiGLU)`)
  if (A.head_dim > 128) throw new Error(`bitgpu: unsupported head_dim ${A.head_dim} (kernels assume <= 128)`)
  if (A.heads % A.kv_heads !== 0) throw new Error(`bitgpu: heads ${A.heads} not divisible by kv_heads ${A.kv_heads} (GQA kernels assume an integer group size)`)
  if (!T[FINAL_NORM]) throw new Error(`bitgpu: manifest is missing the final norm tensor '${FINAL_NORM}'`)
  if (manifest.version !== undefined && manifest.version !== 1 && manifest.version !== 2)
    throw new Error(`bitgpu: unsupported manifest version ${manifest.version} (this engine reads versions 1 and 2)`)
  // Rope comes either baked (cos_cache/sin_cache refs, ONNX exports: exact parity with the
  // exporter) or synthesized at load from arch.rope (v2/GGUF manifests, which bake no tables).
  if (!T.cos_cache !== !T.sin_cache) throw new Error('bitgpu: manifest has only one of cos_cache/sin_cache')
  if (!T.cos_cache && !(A.rope && A.rope.rope_theta))
    throw new Error('bitgpu: manifest has neither baked cos_cache/sin_cache RoPE tensors nor arch.rope parameters')
  for (const [name, t] of Object.entries(T)) {
    if (t.block !== undefined && t.block !== 128) throw new Error(`bitgpu: tensor ${name} has block ${t.block} (kernels assume 128)`)
    // v2 container tensors: keep the raw interleaved region aside (q1_0) and synthesize
    // PLANAR weight/scales refs carrying the byte lengths the kernels consume, so all the
    // size computations below (limits, buffer creation, fusion) stay container-blind. The
    // streaming loader demuxes the region into both sinks in-flight (see wireQ10).
    if (t.container === undefined) continue
    if (t.container !== 'q1_0') throw new Error(`bitgpu: tensor ${name} has unknown container '${t.container}'`)
    const N = t.N ?? t.rows
    const K = t.K ?? t.cols
    if (!N || !K || K % 128 !== 0) throw new Error(`bitgpu: tensor ${name}: q1_0 container needs N/K (or rows/cols) with K a multiple of 128`)
    const r = t.weight
    if (!r || r.src !== 'data' || r.len !== N * (K / 128) * 18)
      throw new Error(`bitgpu: tensor ${name}: q1_0 region is ${r?.len} bytes in '${r?.src}', expected ${N * (K / 128) * 18} in the data file`)
    t.q1_0 = r
    t.weight = { dtype: 'UINT8', src: r.src, off: r.off, len: N * (K / 8) }
    t.scales = { dtype: 'FLOAT', src: r.src, off: r.off, len: N * (K / 128) * 4 }
    t.zp = undefined // no zp tensor in the container; the 1-bit recipe's midpoints are constants
  }

  const readRef = (ref: Ref): Float32Array | Uint8Array | Uint16Array => {
    if (ref.src !== 'aux') throw new Error('bitgpu: internal - readRef reads aux-file refs; data-file tensors stream through routes')
    if (ref.off + ref.len > aux.byteLength)
      throw new Error(`bitgpu: tensor range ${ref.off}+${ref.len} exceeds the aux file (${aux.byteLength} bytes); the download is truncated or the manifest does not match it`)
    const V = VIEW[ref.dtype]!
    if (V === Uint8Array) return new Uint8Array(aux, ref.off, ref.len)
    const bpe = V.BYTES_PER_ELEMENT
    if (ref.off % bpe === 0) return new V(aux, ref.off, ref.len / bpe)
    return new V(aux.slice(ref.off, ref.off + ref.len))
  }
  const readU8 = (ref: Ref): Uint8Array => readRef(ref) as Uint8Array

  const adapter = await navigator.gpu.requestAdapter({ powerPreference }) // pick the discrete GPU on multi-GPU machines, not the weak iGPU
  if (!adapter) throw new WebGPUUnavailableError('No suitable WebGPU adapter was found.')
  const hasSG = adapter.features.has('subgroups' as GPUFeatureName)
  const info = (adapter.info ?? {}) as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number } // subgroup sizes live on GPUAdapterInfo
  const sgMax = info.subgroupMaxSize ?? 32
  const sgMin = info.subgroupMinSize ?? sgMax
  const forceNoSG = opts.forceNoSubgroups ?? false
  // No-subgroup reduction workgroup size. Snapped to a power of two in [32, 256]: the _wg kernels'
  // tree reductions halve the stride each step, so any other size silently drops partial sums.
  const WG_NS = Math.min(256, Math.max(32, 1 << Math.round(Math.log2(opts.noSubgroupWorkgroupSize ?? 64))))
  const NOTILE = opts.prefillTiling === 'never' // force the scalar prefill GEMM (A/B)
  const FORCETILE = opts.prefillTiling === 'always' // use tiled even for short prompts (validation)
  const tiledPrefill = (S: number): boolean => FORCETILE || (!NOTILE && S >= 64) // tiled GEMM wins only once it fills its 64-row tiles
  const SYNC_N = Math.max(1, opts.syncSteps ?? 4) // decode: chain N steps per CPU sync
  const maxSeqLen = Math.max(1, opts.maxSeqLen ?? DEFAULT_MAX_SEQ) // KV-cache length cap (VRAM ~ maxSeqLen x layers x kv_heads x head_dim x KVB)
  // uniform 16/32/64 (16 = Arm Mali) -> head_dim/SG<=8. The subgroup attention kernel covers
  // dimension lane + t*SG for t in [0, D/SG), so head_dim must divide evenly (128 always does;
  // an exotic head_dim like 80 takes the strided _wg fallback instead of silently dropping dims).
  const useSG = hasSG && sgMin === sgMax && (sgMax === 16 || sgMax === 32 || sgMax === 64) && A.head_dim % sgMax === 0 && !forceNoSG
  // f16 KV STORAGE (math stays f32; one rounding at cache-write). 'f16' silently falls back to
  // f32 where shader-f16 is missing, so callers can request it unconditionally.
  const kv16 = opts.kvCache === 'f16' && adapter.features.has('shader-f16' as GPUFeatureName)
  // q8: packed snorm8 K/V with one f32 scale per 32-element block (q8_0-style). Pure core WGSL
  // (pack4x8snorm), so unlike f16 it needs no adapter feature - available everywhere.
  const kv8 = opts.kvCache === 'q8'
  // Rolling window with attention sinks (StreamingLLM): keys cached UNROPED, rotated at read
  // by cache-relative position, middle evicted in batches. See the attention_*_roll shaders.
  const roll = opts.overflow === 'sinks'
  const SINKS = roll ? Math.max(1, Math.floor(opts.sinkTokens ?? 4)) : 0
  if (roll && maxSeqLen < SINKS + 64)
    throw new Error(`bitgpu: overflow 'sinks' needs maxSeqLen >= sinkTokens + 64 (got ${maxSeqLen} with ${SINKS} sinks)`)
  const KVB = kv16 ? 2 : kv8 ? 1 : 4 // bytes per cached K/V element (q8 block scales tracked separately)
  const features: GPUFeatureName[] = []
  if (useSG) features.push('subgroups' as GPUFeatureName)
  if (kv16) features.push('shader-f16' as GPUFeatureName)
  // ---- device limits negotiation ----
  // Run at WebGPU's guaranteed-minimum limits whenever the model fits them (low-end/mobile devices
  // keep working exactly as before), and request precisely the raised limits a bigger model needs,
  // gated on what the adapter can grant. Every buffer the loader will create is sized here FROM THE
  // MANIFEST, before the device is requested: a model this adapter cannot hold fails loudly now,
  // instead of "loading fine" with invalid buffers and generating garbage (binding-limit violations
  // are deferred VALIDATION errors, which no out-of-memory scope ever catches).
  const DEFAULT_BINDING = 134217728 // maxStorageBufferBindingSize guaranteed minimum (128 MiB)
  const DEFAULT_BUFFER = 268435456 // maxBufferSize guaranteed minimum (256 MiB)
  let needBind = 0 // largest single storage buffer/binding the loader will create
  let weightBytes = 0 // total weight VRAM, for the OOM message
  const track = (bytes: number): void => {
    const b = (bytes + 3) & ~3 // gbuf pads to 4
    needBind = Math.max(needBind, b)
    weightBytes += b
  }
  for (const t of Object.values(T)) {
    if (t.kind === 'q2') {
      track(t.weight!.len * 2) // q2 codes expand 1 -> 2 bytes at load (wireQ2)
      track(t.scales!.len)
    } else if (t.kind === 'f32' && t.weight) track(t.weight.len)
  }
  const fusedLen = (names: string[], f: 'weight' | 'scales'): number => names.reduce((n, nm) => n + T[nm][f]!.len, 0)
  for (let li = 0; li < A.layers; li++) {
    const groups = [
      [`layers.${li}.attn.q_proj`, `layers.${li}.attn.k_proj`, `layers.${li}.attn.v_proj`],
      [`layers.${li}.mlp.gate_proj`, `layers.${li}.mlp.up_proj`],
      [`layers.${li}.attn.o_proj`],
      [`layers.${li}.mlp.down_proj`],
    ]
    for (const g of groups) {
      track(fusedLen(g, 'weight'))
      track(fusedLen(g, 'scales'))
    }
  }
  // embed zp: from the manifest ref, or synthesized for container tensors (2 blocks/byte)
  const embZpLen = T.embed_tokens.zp?.len ?? (T.embed_tokens.rows! * (T.embed_tokens.cols! / 128)) / 2
  for (const r of [T.embed_tokens.weight!, T.embed_tokens.scales!, manifest.luts.tgt4]) track(r.len)
  track(embZpLen)
  if (kv8 && A.head_dim % 32 !== 0)
    throw new Error(`bitgpu: kvCache 'q8' needs head_dim divisible by 32 (got ${A.head_dim}); use 'f16' or 'f32' for this model`)
  // Non-weight bindings that also count against the limit: a per-layer KV buffer at full
  // maxSeqLen capacity, the largest prefill-segment activation, and the widest logits scratch.
  const kvLayerBytes = maxSeqLen * A.kv_heads * A.head_dim * KVB
  needBind = Math.max(needBind, kvLayerBytes, PREFILL_SEG * Math.max(A.heads * A.head_dim, A.intermediate) * 4, 32 * A.vocab * 4)
  const requiredLimits: Record<string, number> = {}
  if (needBind > DEFAULT_BINDING) {
    if (needBind > adapter.limits.maxStorageBufferBindingSize)
      throw new GpuOutOfMemoryError(
        `this model needs a ${Math.ceil(needBind / 1048576)} MiB storage binding but the adapter's maxStorageBufferBindingSize is ${Math.floor(adapter.limits.maxStorageBufferBindingSize / 1048576)} MiB`,
      )
    requiredLimits.maxStorageBufferBindingSize = needBind
  }
  if (needBind > DEFAULT_BUFFER) {
    if (needBind > adapter.limits.maxBufferSize)
      throw new GpuOutOfMemoryError(
        `this model needs a ${Math.ceil(needBind / 1048576)} MiB buffer but the adapter's maxBufferSize is ${Math.floor(adapter.limits.maxBufferSize / 1048576)} MiB`,
      )
    requiredLimits.maxBufferSize = needBind
  }
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: Object.keys(requiredLimits).length ? requiredLimits : undefined,
  })
  holder.device = device // createEngine destroys it if anything below throws
  // awareness: shared mem <=8KB < 16KB min, WG <=256; binding sizes negotiated above

  // Surface device loss (driver reset, OS reclaim) instead of hanging: consumers get a promise +
  // an optional callback. dispose() also resolves it, with reason 'destroyed' (no callback then).
  const lost: Promise<DeviceLostInfo> = device.lost.then((info) => {
    const li = { reason: String(info.reason ?? 'unknown'), message: info.message }
    if (li.reason !== 'destroyed') opts.onDeviceLost?.(li)
    return li
  })
  device.addEventListener('uncapturederror', (ev) => {
    console.error(`[bitgpu] uncaptured WebGPU error: ${(ev as GPUUncapturedErrorEvent).error.message}`)
  })

  opts.onProgress?.({ phase: 'pipelines' })
  const pipelines: Record<string, GPUComputePipeline> = {}
  // async pipeline creation: compile in parallel, non-blocking -> faster, stall-free cold start (MDN-recommended)
  const mkPipe = async (name: string, constants?: Record<string, number>): Promise<void> => {
    const code = SHADERS[name]
    if (code === undefined) throw new Error(`shader not found: ${name}`)
    const module = device.createShaderModule({ code, label: name }) // label so errors name the shader
    const info = await module.getCompilationInfo()
    const err = info.messages.find((m) => m.type === 'error')
    if (err) throw new Error(`WGSL compile error in ${name} (L${err.lineNum}:${err.linePos}): ${err.message}`)
    pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants } })
  }
  const ROWS_MR = 4 // output rows per workgroup in the multi-row GEMV
  const specs: Array<[string, Record<string, number>?]> = [...WGSLS.map((n): [string] => [n]), ['matmul_split_tiled'], ['matmul_resid_tiled'], ['argmax'], ['embed_gather'], ['embed_gather_batch'], ['sampler_penalty'], ['argmax_masked'], ['logsumexp']]
  if (useSG) {
    for (const n of ['rmsnorm_sg', 'attention_sg', 'matmul_split_sg', 'matmul_q2_sg', 'rmsnorm_rope_sg']) specs.push([n, { SG: sgMax }])
    for (const n of ['matmul_split_sm', 'matmul_resid_sm', 'matmul_q2_sm']) specs.push([n, { SG: sgMax }]) // small-batch (M=2..9) verify-pass GEMVs
    for (const n of ['matmul_resid_mr_sg', 'matmul_swiglu_mr_sg']) specs.push([n, { SG: sgMax, ROWS: ROWS_MR }])
  } else {
    for (const n of ['matmul_split_wg', 'matmul_resid_wg', 'matmul_q2_wg', 'rmsnorm_wg']) specs.push([n, { WG: WG_NS }])
    specs.push(['attention_wg']) // fixed 64-thread workgroup (per-thread acc covers head_dim <= 128)
  }
  if (kv16) {
    // f16-KV variants (compiled only when the device has shader-f16; `enable f16;` fails otherwise)
    specs.push(['copy_kv16'])
    if (useSG) for (const n of ['attention_sg_kv16', 'rmsnorm_rope_sg_kv16']) specs.push([n, { SG: sgMax }])
    else specs.push(['attention_wg_kv16'])
  }
  if (kv8) {
    // q8-KV variants (pure core WGSL: pack4x8snorm; no adapter feature needed)
    specs.push(['copy_kv8'])
    if (useSG) for (const n of ['attention_sg_kv8', 'rmsnorm_rope_sg_kv8']) specs.push([n, { SG: sgMax }])
    else specs.push(['attention_wg_kv8'])
  }
  // Rolling-window (sinks) attention reads rotate K at read time, so it has its own kernel per
  // mode+path. The sg roll kernels keep the rotate partner in-lane only when SG <= head_dim/2;
  // outside that geometry the wg roll kernel takes over (attention only - everything else
  // keeps the sg path).
  const rollSG = useSG && sgMax <= A.head_dim / 2
  if (roll) {
    const rollAtt = kv16 ? 'attention_sg_kv16_roll' : kv8 ? 'attention_sg_kv8_roll' : 'attention_sg_roll'
    const rollAttWg = kv16 ? 'attention_wg_kv16_roll' : kv8 ? 'attention_wg_kv8_roll' : 'attention_wg_roll'
    if (rollSG) specs.push([rollAtt, { SG: sgMax }])
    else specs.push([rollAttWg])
  }
  // Cache-touching kernel names resolve once by KV storage mode. q8's writers need extra
  // bindings (the block-scale buffers), so its call sites branch instead of renaming.
  const ATT =
    roll ?
      kv16 ? (rollSG ? 'attention_sg_kv16_roll' : 'attention_wg_kv16_roll')
      : kv8 ? (rollSG ? 'attention_sg_kv8_roll' : 'attention_wg_kv8_roll')
      : rollSG ? 'attention_sg_roll' : 'attention_wg_roll'
    : kv16 ? (useSG ? 'attention_sg_kv16' : 'attention_wg_kv16')
    : kv8 ? (useSG ? 'attention_sg_kv8' : 'attention_wg_kv8')
    : useSG ? 'attention_sg' : 'attention_wg'
  const ROPE_K = kv16 ? 'rmsnorm_rope_sg_kv16' : 'rmsnorm_rope_sg' // fused-path K write into the cache (f32/f16; q8 branches)
  const COPY_KV = kv16 ? 'copy_kv16' : 'copy' //                      K/V append into the cache (f32/f16; q8 branches)
  await Promise.all(specs.map(([n, c]) => mkPipe(n, c))) // parallel compile of all pipelines

  const S_ = GPUBufferUsage.STORAGE,
    CD = GPUBufferUsage.COPY_DST,
    CS = GPUBufferUsage.COPY_SRC,
    U = GPUBufferUsage.UNIFORM
  // Per-call transient tracking: generate/prefill/forward set `transients = []` so every scratch
  // buffer they create (activations, per-dispatch uniforms, prompt embeddings) is destroyed as soon
  // as its submission completes, instead of lingering until GC. A long prefill otherwise holds
  // ~S x 4.3 MB of dead VRAM (2+ GB for a 512-token prompt) - real memory pressure on 8 GB devices.
  let transients: GPUBuffer[] | null = null
  const flushTransients = (): void => {
    if (!transients) return
    for (const b of transients) b.destroy()
    transients = []
  }
  const upload = (typed: ArrayBufferView, usage: number = S_ | CD): GPUBuffer => {
    const b = device.createBuffer({ size: typed.byteLength, usage })
    device.queue.writeBuffer(b, 0, typed as BufferSource)
    transients?.push(b)
    return b
  }
  // Decode resource pool: in the decode loop the dispatch sequence is identical every batch, so reuse
  // the scratch + uniform buffers across batches (createBuffer is the dominant per-token record cost).
  // Counters increment per call and reset per batch, so within a batch every dispatch still gets its
  // own buffer (no aliasing of in-flight work); reuse happens only across batches (after the sync).
  interface DispSlot {
    uni: GPUBuffer
    bg: GPUBindGroup | null
    last: Uint8Array | null
  }
  // NAMED pools: each distinct dispatch sequence gets its own slot array ('decode' = the fused
  // token loop, 'pld1' / 'pldm' = speculative single-token and verify steps). Selecting a pool
  // also resets the slot indices (the old per-batch poolReset).
  interface Pool { buf: GPUBuffer[]; disp: DispSlot[] }
  const pools: Record<string, Pool> = {}
  let pool: Pool | null = null
  let bufIdx = 0
  let dispIdx = 0
  // Verify-step rounding: buffer sizes are S x per-row elements, so rounding S up to a fixed row
  // count gives every draft length identical buffer sizes - and therefore stable buffer
  // identities, which the cached bind groups depend on. n / from * to is exact by construction.
  let poolRoundFrom = 0
  let poolRoundTo = 0
  const poolUse = (name: string | null, roundFrom = 0, roundTo = 0): void => {
    pool = name ? (pools[name] ??= { buf: [], disp: [] }) : null
    poolRoundFrom = roundFrom
    poolRoundTo = roundTo
    bufIdx = 0
    dispIdx = 0
  }
  // The cached bind groups reference this generate() call's buffers (tokBuf/lg/candIds/... are created
  // per call). A later call creates new buffers, so the cache MUST be rebuilt at each decode entry or it
  // would bind the previous call's (dead) buffers - or, across greedy<->sampled, a different pipeline's
  // auto-layout bind group (a validation error). Buffers are stable within a call, so one rebuild suffices.
  const poolInvalidate = (): void => {
    for (const p of Object.values(pools)) {
      for (const s of p.disp) {
        s.bg = null
        s.last = null
      }
    }
  }
  const actBuf = (n: number): GPUBuffer => {
    if (!pool) {
      const b = device.createBuffer({ size: n * 4, usage: S_ | CS | CD })
      transients?.push(b)
      return b
    }
    const alloc = poolRoundFrom > 0 ? (n / poolRoundFrom) * poolRoundTo : n
    let b = pool.buf[bufIdx]
    if (!b || b.size !== alloc * 4) {
      b = device.createBuffer({ size: alloc * 4, usage: S_ | CS | CD })
      pool.buf[bufIdx] = b
    }
    bufIdx++
    return b
  }
  const dummy = device.createBuffer({ size: 16, usage: S_ })

  // Cross-turn state. fullHistory is the entire conversation's token sequence (prompt turns + replies),
  // needed because the sampler's repetition_penalty / no_repeat_ngram see the FULL sequence (like
  // transformers.js, independent of the KV cache). Derived from it: the cache fill length is
  // fullHistory.length - 1 (the last token's K/V is never written during decode), and that last token
  // is re-fed when resuming so its K/V gets written. The persistent Kc/Vc (below) hold the cached K/V.
  let fullHistory: number[] = []
  // Sink-mode bookkeeping: filled cache SLOTS. Diverges from fullHistory.length-1 once the
  // window rolls (fullHistory keeps the whole conversation for penalties/PLD; the cache holds
  // sinks + recent). Maintained by the decode loops; meaningless while overflow is 'error'.
  let cacheLen = 0
  const resetCache = (): void => {
    fullHistory = []
    cacheLen = 0
  }

  if (manifest.luts.tgt2.src !== 'aux')
    throw new Error('bitgpu: luts.tgt2 must live in the aux file (the streaming loader needs it before the weights arrive)')
  const tgt2 = readU8(manifest.luts.tgt2) // load-time only (sign table + q2 expansion); not captured by any closure
  const signTable = new Uint8Array(256)
  for (let b = 0; b < 256; b++) {
    let bits = 0
    for (let j = 0; j < 8; j++) bits |= ((((tgt2[2 * b + (j >> 2)] >> (2 * (j & 3))) & 3) >> 1) & 1) << j
    signTable[b] = bits
  }

  // ---- streaming weight loader ----
  // Every data-file tensor is wired as a ROUTE: a byte range of the file feeding a GPU buffer
  // (optionally through a per-byte transform: the binary sign table, or the 1->2 q2 code
  // expansion) and/or a CPU array. The file then streams through the routes in ONE sequential
  // pass, so the ~290MB download never exists in memory at once. (The old path materialized the
  // whole file plus an expanded lm_head transient: a CPU peak that read as "loads fine, then the
  // OS kills the tab at the first keyboard" on phones.) aux-file refs are tiny and feed the same
  // sinks synchronously from the aux buffer at wiring time.
  interface Route { off: number; len: number; push: (b: Uint8Array) => void; finish: () => void }
  const routes: Route[] = []
  const gbuf = (len: number): GPUBuffer => device.createBuffer({ size: (len + 3) & ~3, usage: S_ | CD })
  const gpuSink = (buf: GPUBuffer, base: number): { push: (b: Uint8Array) => void; finish: () => void } => {
    let written = 0
    let carry = new Uint8Array(0) // writeBuffer needs 4-byte-aligned offsets/sizes; chunks split anywhere
    return {
      push(bytes: Uint8Array): void {
        let all = bytes
        if (carry.length) {
          all = new Uint8Array(carry.length + bytes.length)
          all.set(carry)
          all.set(bytes, carry.length)
        }
        const n = all.length & ~3
        if (n) device.queue.writeBuffer(buf, base + written, all, 0, n) // writeBuffer copies synchronously
        carry = all.subarray(n).slice()
        written += n
      },
      finish(): void {
        if (!carry.length) return
        const pad = new Uint8Array(4)
        pad.set(carry)
        device.queue.writeBuffer(buf, base + written, pad) // gbuf pads buffers to 4 bytes
        written += 4
        carry = new Uint8Array(0)
      },
    }
  }
  // Wire a ref to a sink: data-file refs register a stream route; aux refs feed the sink NOW.
  const wire = (ref: Ref, push: (b: Uint8Array) => void, finish: () => void): void => {
    if (ref.src === 'aux') {
      push(new Uint8Array(readRef(ref).buffer, ref.off, ref.len))
      finish()
    } else routes.push({ off: ref.off, len: ref.len, push, finish })
  }
  const wireRaw = (ref: Ref, buf: GPUBuffer, base = 0): void => {
    const s = gpuSink(buf, base)
    wire(ref, s.push, s.finish)
  }
  // Per-byte weight transforms, shared by the planar (ONNX) and q1_0-container (GGUF)
  // routes - the sign-byte stream is identical across the two containers.
  const xfSign =
    (push: (b: Uint8Array) => void) =>
    (b: Uint8Array): void => {
      const o = new Uint8Array(b.length)
      for (let i = 0; i < b.length; i++) o[i] = signTable[b[i]]
      push(o)
    }
  const xfQ2 =
    (push: (b: Uint8Array) => void) =>
    (b: Uint8Array): void => {
      const o = new Uint8Array(b.length * 2)
      for (let i = 0; i < b.length; i++) {
        o[2 * i] = tgt2[2 * b[i]]
        o[2 * i + 1] = tgt2[2 * b[i] + 1]
      }
      push(o)
    }
  const wireSign = (ref: Ref, buf: GPUBuffer, base = 0): void => {
    const s = gpuSink(buf, base)
    wire(ref, xfSign(s.push), s.finish)
  }
  const wireQ2 = (ref: Ref, buf: GPUBuffer): void => {
    const s = gpuSink(buf, 0)
    wire(ref, xfQ2(s.push), s.finish)
  }
  // f16 -> f32, exact (every f16 value is exactly representable in f32)
  const f16f32 = (h: number): number => {
    const s = h & 0x8000 ? -1 : 1
    const e = (h >> 10) & 31
    const m = h & 1023
    if (e === 0) return s * m * 2 ** -24
    if (e === 31) return m ? NaN : s * Infinity
    return s * (1024 + m) * 2 ** (e - 25)
  }
  // container 'q1_0' (GGUF): ONE region streams BOTH planar outputs. Each 18-byte block is
  // [f16 scale][16 sign bytes]; the scale converts to f32 into the scales sink, the sign
  // bytes feed the weight sink (through the same transform the planar path uses). Block
  // phase carries across pushed chunks, which split anywhere - including mid-scale.
  const wireQ10 = (region: Ref, signPush: (b: Uint8Array) => void, signFinish: () => void, scalePush: (b: Uint8Array) => void, scaleFinish: () => void): void => {
    let phase = 0 // byte position inside the current 18-byte block
    let scaleLo = 0 // pending low byte of a scale straddling a chunk boundary
    const push = (b: Uint8Array): void => {
      const signs = new Uint8Array(b.length)
      const scales = new Float32Array((b.length >> 4) + 2)
      let sn = 0
      let cn = 0
      for (let i = 0; i < b.length; i++) {
        if (phase === 0) {
          scaleLo = b[i]
          phase = 1
        } else if (phase === 1) {
          scales[cn++] = f16f32(scaleLo | (b[i] << 8))
          phase = 2
        } else {
          signs[sn++] = b[i]
          phase = phase === 17 ? 0 : phase + 1
        }
      }
      if (sn) signPush(signs.subarray(0, sn))
      if (cn) scalePush(new Uint8Array(scales.buffer, 0, cn * 4))
    }
    routes.push({ off: region.off, len: region.len, push, finish: () => { signFinish(); scaleFinish() } })
  }
  const wireCpu = (ref: Ref): Uint8Array => {
    const dst = new Uint8Array(ref.len)
    let w = 0
    wire(
      ref,
      (b) => {
        dst.set(b, w)
        w += b.length
      },
      () => undefined,
    )
    return dst
  }
  // WebGPU allocation errors are DEFERRED: without an error scope the ~300 MB of weight buffers
  // "succeed" as invalid buffers on VRAM-poor devices and every later generate returns garbage.
  // The validation scope backstops the limits negotiation above: any load-time validation error
  // (e.g. a buffer over a granted limit) fails createEngine loudly instead of console noise.
  device.pushErrorScope('validation')
  device.pushErrorScope('out-of-memory')
  const W: Record<string, GpuWeight> = {}
  // Deferred zero-point checks: q2 zp tensors stream with the weights, so their bytes are only
  // complete after the streaming pass; each closure then validates and installs the real value.
  const zpChecks: Array<() => void> = []
  for (const [name, t] of Object.entries(T)) {
    if (t.kind === 'q2') {
      const codes = gbuf(t.weight!.len * 2) // 1 byte of q2 data expands to 2 code bytes
      const scales = gbuf(t.scales!.len)
      if (t.q1_0) {
        const cs = gpuSink(codes, 0)
        const ss = gpuSink(scales, 0)
        wireQ10(t.q1_0, xfQ2(cs.push), cs.finish, ss.push, ss.finish)
      } else {
        wireQ2(t.weight!, codes)
        wireRaw(t.scales!, scales)
      }
      const w: GpuWeight = { N: t.N!, K: t.K!, nb: t.K! / 128, zp: 2, codes, scales }
      // The q2 kernels take ONE zero-point for the whole tensor (a uniform, not a per-block
      // buffer read in the hottest GEMV): the q1 recipe always emits the 2-bit midpoint. Read
      // the manifest's zp tensor and derive the scalar from it, failing loudly on a non-uniform
      // export instead of silently dequantizing with the wrong zero-points.
      if (t.zp) {
        const zpBytes = wireCpu(t.zp)
        zpChecks.push(() => {
          const b0 = zpBytes[0]
          for (let i = 1; i < zpBytes.length; i++)
            if (zpBytes[i] !== b0) throw new Error(`bitgpu: tensor ${name} has non-uniform 2-bit zero-points (the q2 kernels assume one zp for the whole tensor)`)
          const zp = b0 & 3
          if (b0 !== zp * 0b01010101) throw new Error(`bitgpu: tensor ${name} has non-uniform 2-bit zero-points within a byte (the q2 kernels assume one zp for the whole tensor)`)
          w.zp = zp
        })
      }
      W[name] = w
    } else if (t.kind === 'f32' && t.weight) {
      const buf = gbuf(t.weight.len)
      wireRaw(t.weight, buf)
      W[name] = { buf }
    }
  }
  // fuse per-layer matmul weights: qkv (3), gate/up (2); o_proj + down_proj stay individual
  // (residual-folded). Fusion needs no CPU concat: each part streams into its fused-buffer slice.
  const fuse = (parts: ManifestTensor[]): { sign: GPUBuffer; scales: GPUBuffer } => {
    const sign = gbuf(parts.reduce((n, p) => n + p.weight!.len, 0))
    const scales = gbuf(parts.reduce((n, p) => n + p.scales!.len, 0))
    let so = 0
    let co = 0
    for (const p of parts) {
      if (p.q1_0) {
        const ws = gpuSink(sign, so)
        const ss = gpuSink(scales, co)
        wireQ10(p.q1_0, xfSign(ws.push), ws.finish, ss.push, ss.finish)
      } else {
        wireSign(p.weight!, sign, so)
        wireRaw(p.scales!, scales, co)
      }
      so += p.weight!.len
      co += p.scales!.len
    }
    return { sign, scales }
  }
  for (let li = 0; li < A.layers; li++) {
    const q = T[`layers.${li}.attn.q_proj`],
      k = T[`layers.${li}.attn.k_proj`],
      v = T[`layers.${li}.attn.v_proj`]
    W[`layers.${li}.attn.qkv`] = { K: q.K!, nb: q.K! / 128, N0: q.N!, N1: k.N!, N2: v.N!, ...fuse([q, k, v]) }
    const g = T[`layers.${li}.mlp.gate_proj`],
      u = T[`layers.${li}.mlp.up_proj`]
    W[`layers.${li}.mlp.gateup`] = { K: g.K!, nb: g.K! / 128, N0: g.N!, N1: u.N!, N2: 0, ...fuse([g, u]) }
    for (const nm of [`layers.${li}.attn.o_proj`, `layers.${li}.mlp.down_proj`]) {
      const r = T[nm]
      W[nm] = { N: r.N!, K: r.K!, nb: r.K! / 128, ...fuse([r]) }
    }
  }

  // Embedding tables are GPU-ONLY: decode gathers one token (embed_gather.wgsl) and prefill
  // gathers a whole segment (embed_gather_batch.wgsl) straight from the packed tables, so no
  // CPU copy exists (~50-100 MB RAM saved per model; uint8 arrays are uploaded as bytes and
  // read as u32, byte-extracted). cos/sin caches are CPU-only.
  const wireGpu = (ref: Ref): GPUBuffer => {
    const buf = gbuf(ref.len)
    wireRaw(ref, buf)
    return buf
  }
  let embWqG: GPUBuffer
  let embScalesG: GPUBuffer
  let embZpG: GPUBuffer
  if (T.embed_tokens.q1_0) {
    embWqG = gbuf(T.embed_tokens.weight!.len)
    embScalesG = gbuf(T.embed_tokens.scales!.len)
    const ws = gpuSink(embWqG, 0)
    const ss = gpuSink(embScalesG, 0)
    wireQ10(T.embed_tokens.q1_0, ws.push, ws.finish, ss.push, ss.finish)
    // q4 zero-points: the 1-bit recipe's 4-bit codes sit at {7,9} around the midpoint 8. A
    // q1_0 container carries no zp tensor, so synthesize the constant table the gather
    // kernels read (two 4-bit 8s per byte, one nibble per 128-wide block).
    embZpG = gbuf(embZpLen)
    device.queue.writeBuffer(embZpG, 0, new Uint8Array((embZpLen + 3) & ~3).fill(0x88))
  } else {
    embWqG = wireGpu(T.embed_tokens.weight!)
    embScalesG = wireGpu(T.embed_tokens.scales!)
    embZpG = wireGpu(T.embed_tokens.zp!)
  }
  const tgt4G = wireGpu(manifest.luts.tgt4)
  let cosCache: Float32Array
  let sinCache: Float32Array
  if (T.cos_cache) {
    const cosBytes = wireCpu(T.cos_cache as Ref)
    const sinBytes = wireCpu(T.sin_cache as Ref)
    cosCache = new Float32Array(cosBytes.buffer)
    sinCache = new Float32Array(sinBytes.buffer)
    // The caches are baked per export ([positions, head_dim/2]); positions beyond them would read
    // undefined -> NaN into every rope buffer, silent garbage. Cap maxSeqLen at load, loudly.
    const ropePositions = cosCache.length / (A.head_dim / 2)
    if (maxSeqLen > ropePositions)
      throw new Error(`bitgpu: maxSeqLen ${maxSeqLen} exceeds the model's baked RoPE cache (${ropePositions} positions); lower maxSeqLen or re-export with a longer cache`)
  } else {
    // v2/GGUF manifests bake no rope tables: synthesize f32 cos/sin for exactly maxSeqLen
    // positions from arch.rope (plain or YaRN) - the same recipe tools/reference.py runs
    // when it generates fixtures, so the reference forward uses the tables the engine uses.
    const cap = A.max_positions ?? 40960
    if (maxSeqLen > cap)
      throw new Error(`bitgpu: maxSeqLen ${maxSeqLen} exceeds the model's max_positions (${cap})`)
    ;[cosCache, sinCache] = synthRope(A, maxSeqLen)
  }

  // ONE sequential pass of the data file through the routes. Tied tensors (e.g. lm_head sharing
  // the embedding bytes) produce EXACT-duplicate ranges: those fan out to every consumer; partial
  // overlaps have no defined streaming order and stay fatal.
  routes.sort((a, b) => a.off - b.off || a.len - b.len)
  const merged: Route[] = []
  for (const r of routes) {
    const last = merged[merged.length - 1]
    if (last && last.off === r.off && last.len === r.len) {
      const lp = last.push,
        lf = last.finish
      last.push = (b): void => {
        lp(b)
        r.push(b)
      }
      last.finish = (): void => {
        lf()
        r.finish()
      }
    } else if (last && r.off < last.off + last.len) {
      throw new Error('bitgpu: partially overlapping data-file tensor ranges (unsupported by the streaming loader)')
    } else merged.push(r)
  }
  const needed = merged.length ? merged[merged.length - 1].off + merged[merged.length - 1].len : 0
  const dataStream = opts.fetchStream
    ? await opts.fetchStream(dataUrl)
    : opts.fetchArrayBuffer
      ? (new Response(await opts.fetchArrayBuffer(dataUrl)).body as ReadableStream<Uint8Array>)
      : await (async () => {
          const res = await fetch(dataUrl)
          if (!res.ok) throw new Error(`bitgpu: fetch ${dataUrl} failed: HTTP ${res.status}`)
          return res.body ?? (new Response(await res.arrayBuffer()).body as ReadableStream<Uint8Array>)
        })()
  const reader = dataStream.getReader()
  let cursor = 0
  let ri = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    let bo = 0
    while (bo < value.byteLength && ri < merged.length) {
      const r = merged[ri]
      if (cursor >= r.off + r.len) {
        ri++
        continue
      }
      if (cursor < r.off) {
        const skip = Math.min(r.off - cursor, value.byteLength - bo)
        cursor += skip
        bo += skip
        continue
      }
      const n = Math.min(r.off + r.len - cursor, value.byteLength - bo)
      r.push(value.subarray(bo, bo + n))
      cursor += n
      bo += n
      if (cursor === r.off + r.len) {
        r.finish()
        ri++
      }
    }
    cursor += value.byteLength - bo
    opts.onProgress?.({ phase: 'weights', loaded: Math.min(cursor, needed), total: needed })
  }
  if (cursor < needed)
    throw new Error(`bitgpu: the data file ended at ${cursor} bytes but tensors extend to ${needed}; the download is truncated or the manifest does not match it`)
  for (const check of zpChecks) check()
  zpChecks.length = 0
  aux = null as unknown as ArrayBuffer

  // GPU embedding gather for a batch of prompt tokens: upload S u32 ids (bytes, not S*H floats)
  // and dequantize the rows on the GPU (embed_gather_batch.wgsl - same math as the decode
  // gather). The returned activation holds [S, H] and belongs to the caller's transient scope.
  function embedBatch(enc: GPUCommandEncoder, ids: number[]): GPUBuffer {
    const idBuf = upload(new Uint32Array(ids))
    const out = actBuf(ids.length * Hd)
    const pass = enc.beginComputePass()
    run(pass, 'embed_gather_batch', [['u', ids.length], ['u', Hd], ['u', 0], ['u', 0]], [idBuf, embWqG, tgt4G, embScalesG, embZpG], out, ids.length * Hd)
    pass.end()
    return out
  }
  function ropeBufs(posBase: number, S: number): { cos: GPUBuffer; sin: GPUBuffer } {
    const D = A.head_dim,
      R = D / 2, // rotary halves: caches store [seq, D/2]; the full vector is concat(half, half)
      cos = new Float32Array(S * D),
      sin = new Float32Array(S * D)
    for (let s = 0; s < S; s++)
      for (let d = 0; d < D; d++) {
        cos[s * D + d] = cosCache[(posBase + s) * R + (d % R)]
        sin[s * D + d] = sinCache[(posBase + s) * R + (d % R)]
      }
    const cb = actBuf(S * D),
      sb = actBuf(S * D)
    device.queue.writeBuffer(cb, 0, cos)
    device.queue.writeBuffer(sb, 0, sin)
    return { cos: cb, sin: sb }
  }

  const KV = A.kv_heads,
    Dh = A.head_dim,
    Hd = A.hidden,
    H = A.heads,
    F = A.intermediate
  // The KV cache GROWS on demand (doubling, up to maxSeqLen) instead of pinning the full maxSeqLen
  // up front: a short conversation keeps a small cache (~KV_INITIAL positions), so idle VRAM stays
  // low on memory-constrained devices (a full 2048-position f32 cache is ~448 MB at 28 layers,
  // ~576 MB at 36; the 512 floor is a quarter of that).
  // ensureKvCapacity() reallocates + copies the existing K/V when a turn needs more room.
  const KV_INITIAL = 512
  let kvCapacity = Math.min(maxSeqLen, KV_INITIAL)
  const Kc: GPUBuffer[] = [],
    Vc: GPUBuffer[] = []
  // q8 block scales: one f32 per 32 cached elements, per layer, alongside Kc/Vc (empty otherwise)
  const Ksc: GPUBuffer[] = [],
    Vsc: GPUBuffer[] = []
  const kvScaleBytes = (cap: number): number => cap * KV * (Dh / 32) * 4
  for (let li = 0; li < A.layers; li++) {
    Kc.push(device.createBuffer({ size: kvCapacity * KV * Dh * KVB, usage: S_ | CS | CD }))
    Vc.push(device.createBuffer({ size: kvCapacity * KV * Dh * KVB, usage: S_ | CS | CD }))
    if (kv8) {
      Ksc.push(device.createBuffer({ size: kvScaleBytes(kvCapacity), usage: S_ | CS | CD }))
      Vsc.push(device.createBuffer({ size: kvScaleBytes(kvCapacity), usage: S_ | CS | CD }))
    }
  }
  // Sink-mode statics: the aux rope tables ([window, Dh/2]) for the read-time K rotation, and
  // cos=1/sin=0 stand-ins so the K WRITE kernels store the un-roped rmsnorm output unchanged
  // (nd*1 + rot*0 == nd; no separate no-rope kernel variants needed).
  let rollCosT: GPUBuffer | null = null,
    rollSinT: GPUBuffer | null = null,
    rollOnes: GPUBuffer | null = null,
    rollZeros: GPUBuffer | null = null
  if (roll) {
    const R2 = Dh / 2
    rollCosT = device.createBuffer({ size: maxSeqLen * R2 * 4, usage: S_ | CD })
    rollSinT = device.createBuffer({ size: maxSeqLen * R2 * 4, usage: S_ | CD })
    device.queue.writeBuffer(rollCosT, 0, cosCache.buffer, cosCache.byteOffset, maxSeqLen * R2 * 4)
    device.queue.writeBuffer(rollSinT, 0, sinCache.buffer, sinCache.byteOffset, maxSeqLen * R2 * 4)
    rollOnes = device.createBuffer({ size: Dh * 4, usage: S_ | CD })
    rollZeros = device.createBuffer({ size: Dh * 4, usage: S_ | CD })
    device.queue.writeBuffer(rollOnes, 0, new Float32Array(Dh).fill(1))
    device.queue.writeBuffer(rollZeros, 0, new Float32Array(Dh))
  }
  const loadOom = await device.popErrorScope()
  const loadVal = await device.popErrorScope()
  if (loadOom) throw new GpuOutOfMemoryError(`GPU allocation failed while loading weights (~${Math.round(weightBytes / 1048576)} MB VRAM needed): ${loadOom.message}`)
  if (loadVal) throw new Error(`bitgpu: WebGPU validation error while loading weights: ${loadVal.message}`)
  // Grow every layer's K/V buffer to hold at least `needed` positions, preserving the cached content
  // (so a cross-turn reuse mid-conversation survives a growth). Rare (only when crossing a capacity
  // threshold); the copy is GPU-side and bounded geometrically. Invalidates cached decode bind groups
  // since they referenced the old buffers.
  async function ensureKvCapacity(needed: number): Promise<void> {
    if (needed <= kvCapacity) return
    const newCap = Math.min(maxSeqLen, Math.max(needed, kvCapacity * 2))
    const copyBytes = kvCapacity * KV * Dh * KVB // preserve all currently-allocated K/V
    const copyScales = kvScaleBytes(kvCapacity)
    device.pushErrorScope('out-of-memory')
    const enc = device.createCommandEncoder()
    // Every cache buffer this layer set held before the growth, in [K, V, (Ksc, Vsc)] groups.
    const per = kv8 ? 4 : 2
    const olds: GPUBuffer[] = []
    for (let li = 0; li < A.layers; li++) {
      const nk = device.createBuffer({ size: newCap * KV * Dh * KVB, usage: S_ | CS | CD })
      const nv = device.createBuffer({ size: newCap * KV * Dh * KVB, usage: S_ | CS | CD })
      enc.copyBufferToBuffer(Kc[li], 0, nk, 0, copyBytes)
      enc.copyBufferToBuffer(Vc[li], 0, nv, 0, copyBytes)
      olds.push(Kc[li], Vc[li])
      Kc[li] = nk
      Vc[li] = nv
      if (kv8) {
        const nks = device.createBuffer({ size: kvScaleBytes(newCap), usage: S_ | CS | CD })
        const nvs = device.createBuffer({ size: kvScaleBytes(newCap), usage: S_ | CS | CD })
        enc.copyBufferToBuffer(Ksc[li], 0, nks, 0, copyScales)
        enc.copyBufferToBuffer(Vsc[li], 0, nvs, 0, copyScales)
        olds.push(Ksc[li], Vsc[li])
        Ksc[li] = nks
        Vsc[li] = nvs
      }
    }
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    const oom = await device.popErrorScope()
    if (oom) {
      // Roll back to the old (still valid) buffers so the engine stays usable at its current size.
      for (let li = 0; li < A.layers; li++) {
        Kc[li].destroy()
        Vc[li].destroy()
        Kc[li] = olds[per * li]
        Vc[li] = olds[per * li + 1]
        if (kv8) {
          Ksc[li].destroy()
          Vsc[li].destroy()
          Ksc[li] = olds[per * li + 2]
          Vsc[li] = olds[per * li + 3]
        }
      }
      poolInvalidate()
      throw new GpuOutOfMemoryError(`KV cache growth to ${newCap} positions failed: ${oom.message}`)
    }
    for (const b of olds) b.destroy()
    kvCapacity = newCap
    poolInvalidate()
  }

  // ---- sink-mode eviction (overflow: 'sinks') ----
  // Batched StreamingLLM compaction: keep the SINKS anchor positions plus the most recent
  // block, drop the middle. Cache bytes only MOVE (same-buffer copies are forbidden in WebGPU,
  // so each region bounces through one lazily-kept scratch buffer); nothing is re-roped or
  // requantized - the roll attention kernels rotate K by cache slot at read, so a moved row
  // simply reads as "closer". Queue order makes the copies land before any later dispatch, so
  // no CPU sync is needed. Called only from inside a turn (the op is already serialized).
  let evictScratch: GPUBuffer | null = null
  function evict(fill: number, need: number): number {
    // free at least `need` slots plus a quarter-of-recent batch, so events stay rare
    const recent = fill - SINKS
    const batch = Math.min(recent, Math.max(need, Math.ceil((maxSeqLen - SINKS) / 4)))
    const keep = recent - batch
    if (keep <= 0) return SINKS // nothing recent survives (giant need): sinks only
    const rowK = KV * Dh * KVB
    const rowS = kv8 ? KV * (Dh / 32) * 4 : 0
    if (!evictScratch || evictScratch.size < keep * rowK) {
      evictScratch?.destroy()
      evictScratch = device.createBuffer({ size: keep * rowK, usage: CS | CD })
    }
    const enc = device.createCommandEncoder()
    const groups: Array<[GPUBuffer, number]> = []
    for (let li = 0; li < A.layers; li++) {
      groups.push([Kc[li], rowK], [Vc[li], rowK])
      if (kv8) groups.push([Ksc[li], rowS], [Vsc[li], rowS])
    }
    for (const [buf, row] of groups) {
      enc.copyBufferToBuffer(buf, (SINKS + batch) * row, evictScratch, 0, keep * row)
      enc.copyBufferToBuffer(evictScratch, 0, buf, SINKS * row, keep * row)
    }
    device.queue.submit([enc.finish()])
    return SINKS + keep
  }
  // Make room for `n` more rows before recording a batch; returns the (possibly reduced) fill.
  const evictFor = (fill: number, n: number): number => (roll && fill + n > maxSeqLen ? evict(fill, fill + n - maxSeqLen) : fill)

  async function readback(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(rb.getMappedRange().slice(0))
    rb.unmap()
    rb.destroy()
    return out
  }
  async function readbackU32(buf: GPUBuffer, n: number): Promise<Uint32Array> {
    const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const out = new Uint32Array(rb.getMappedRange().slice(0))
    rb.unmap()
    rb.destroy()
    return out
  }

  // diagnostic: FULL = null -> every kernel at real size; FULL = Set(names) -> only those at real size,
  // all others dispatched as 1 workgroup. Lets us measure each kernel type's true in-context cost.
  let FULL: Set<string> | null = null
  // embed_gather_batch is exempt from profiling skeletons: it produces the PREFILL INPUT (the
  // CPU used to), and skeleton-skipping it feeds NaN/garbage activations into every profiled
  // kernel - numerically meaningless and, on GPUs with slow non-finite handling, much slower.
  const isFull = (name: string): boolean => FULL === null || FULL.has(name) || name === 'embed_gather_batch'
  // differential debug: FORCE_SLOW routes S=1 through the prefill (known-good) path; DBG0 collects
  // layer-0 checkpoint buffers so a fused step and a slow step can be compared kernel by kernel.
  let FORCE_SLOW = false
  // Small-batch routing: when set to S (2..9, subgroup path only), the matmuls of an S-row pass
  // use the _sm kernels, which read each weight word once for all S rows - the lever that makes
  // the speculative-decode verify pass profitable. 0 = off (scalar/tiled prefill kernels).
  // Set ONLY around the PLD verify-step encode; every shipped non-PLD path keeps its kernels.
  let SMALLM = 0
  let DBG0: Record<string, GPUBuffer> | null = null
  const cap = (li: number, name: string, buf: GPUBuffer): void => {
    if (li === 0 && DBG0) DBG0[name] = buf
  }
  // set pipeline + bind group for a dispatch. When pooling (decode loop), the uniform buffer AND the
  // bind group are cached per dispatch slot, so only writeBuffer of the changed params runs per token.
  function setup(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[]): void {
    pass.setPipeline(pipelines[name])
    if (pool) {
      let slot = pool.disp[dispIdx]
      if (!slot) {
        slot = { uni: device.createBuffer({ size: 64, usage: U | CD }), bg: null, last: null }
        pool.disp[dispIdx] = slot
      }
      const data2 = makeParams(fields) // reused view; only writeBuffer when the params changed
      if (!slot.last || !eqBytes(slot.last, data2)) {
        device.queue.writeBuffer(slot.uni, 0, data2 as BufferSource)
        slot.last = data2.slice()
      }
      if (!slot.bg) {
        const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: slot.uni } }]
        ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }))
        outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }))
        slot.bg = device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries })
      }
      pass.setBindGroup(0, slot.bg)
      dispIdx++
    } else {
      const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: upload(makeParams(fields), U | CD) } }]
      ins.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }))
      outs.forEach((b, i) => entries.push({ binding: 1 + ins.length + i, resource: { buffer: b } }))
      pass.setBindGroup(0, device.createBindGroup({ layout: pipelines[name].getBindGroupLayout(0), entries }))
    }
  }
  // Split a 1D workgroup count into a 2D grid so no dimension exceeds WebGPU's 65535 cap (a long
  // prefill can need >65535 workgroups, e.g. the swiglu kernel does ceil(S*6144/64) = S*96). The flat
  // kernels reconstruct the linear index as (wid.y*num_workgroups.x + wid.x)*64 + lid.x, which equals
  // global_invocation_id.x when the grid is 1D (y=1) - so it stays correct for the common small case.
  const grid2d = (wg: number): [number, number] => {
    const y = Math.ceil(wg / 65535)
    return [Math.ceil(wg / y), y]
  }
  function runIO(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[], threads: number): void {
    setup(pass, name, fields, ins, outs)
    if (!isFull(name)) return void pass.dispatchWorkgroups(1)
    const [x, y] = grid2d(Math.ceil(threads / 64))
    pass.dispatchWorkgroups(x, y, 1)
  }
  const run = (pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], out: GPUBuffer, threads: number): void =>
    runIO(pass, name, fields, ins, [out], threads)
  // dispatch exactly nWG workgroups (subgroup kernels: one workgroup per row / per (query,head))
  function runN(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], out: GPUBuffer, nWG: number): void {
    setup(pass, name, fields, ins, [out])
    pass.dispatchWorkgroups(isFull(name) ? nWG : 1)
  }
  // 2D workgroup dispatch (subgroup GEMV: one workgroup per output column)
  function runWG(pass: GPUComputePassEncoder, name: string, fields: Field[], ins: GPUBuffer[], outs: GPUBuffer[], wgX: number, wgY: number): void {
    setup(pass, name, fields, ins, outs)
    const f = isFull(name)
    pass.dispatchWorkgroups(f ? wgX : 1, f ? wgY : 1, 1)
  }
  const rms = (pass: GPUComputePassEncoder, x: GPUBuffer, g: string, R: number, Dn: number, out: GPUBuffer): void =>
    useSG
      ? runN(pass, 'rmsnorm_sg', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf!], out, R)
      : runN(pass, 'rmsnorm_wg', [['u', R], ['u', Dn], ['f', A.rms_eps], ['u', 0]], [x, W[g].buf!], out, R)
  // fused q/k/v or gate/up matmul
  function fusedMM(pass: GPUComputePassEncoder, w: GpuWeight, inBuf: GPUBuffer, S: number, outs: GPUBuffer[]): void {
    const Ntot = w.N0! + w.N1! + w.N2!
    if (useSG && S === 1) {
      const gx = Math.min(Ntot, 65535)
      runWG(pass, 'matmul_split_sg', [['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!], ['u', gx]], [inBuf, w.sign!, w.scales!], outs, gx, Math.ceil(Ntot / gx))
    } else if (S === 1) {
      const gx = Math.min(Ntot, 65535) // no-subgroup decode: workgroup-reduction GEMV
      runWG(pass, 'matmul_split_wg', [['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!], ['u', gx]], [inBuf, w.sign!, w.scales!], outs, gx, Math.ceil(Ntot / gx))
    } else if (useSG && S === SMALLM) {
      const gx = Math.min(Ntot, 65535) // small-batch GEMV: weights read once for all S rows
      runWG(pass, 'matmul_split_sm', [['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!], ['u', gx], ['u', S]], [inBuf, w.sign!, w.scales!], outs, gx, Math.ceil(Ntot / gx))
    } else if (tiledPrefill(S)) {
      runWG(pass, 'matmul_split_tiled', [['u', S], ['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!]], [inBuf, w.sign!, w.scales!], outs, Math.ceil(Ntot / 64), Math.ceil(S / 64)) // long-prompt prefill: tiled GEMM
    } else {
      runIO(pass, 'matmul_split', [['u', S], ['u', w.K!], ['u', w.nb!], ['u', w.N0!], ['u', w.N1!], ['u', w.N2!]], [inBuf, w.sign!, w.scales!], outs, S * Ntot)
    }
  }
  // o_proj / down_proj matmul with fused residual add
  function residMM(pass: GPUComputePassEncoder, w: GpuWeight, inBuf: GPUBuffer, resid: GPUBuffer, S: number, out: GPUBuffer): void {
    if (useSG && S === 1) {
      const nwg = Math.ceil(w.N! / ROWS_MR) // multi-row GEMV: ROWS_MR output cols per workgroup
      const gx = Math.min(nwg, 65535)
      runWG(pass, 'matmul_resid_mr_sg', [['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], gx, Math.ceil(nwg / gx))
    } else if (S === 1) {
      const gx = Math.min(w.N!, 65535) // no-subgroup decode: workgroup-reduction GEMV + residual
      runWG(pass, 'matmul_resid_wg', [['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', gx], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], gx, Math.ceil(w.N! / gx))
    } else if (useSG && S === SMALLM) {
      const gx = Math.min(w.N!, 65535) // small-batch GEMV + residual
      runWG(pass, 'matmul_resid_sm', [['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', gx], ['u', S], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], gx, Math.ceil(w.N! / gx))
    } else if (tiledPrefill(S)) {
      runWG(pass, 'matmul_resid_tiled', [['u', S], ['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', 0], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], Math.ceil(w.N! / 64), Math.ceil(S / 64)) // long-prompt prefill: tiled GEMM
    } else {
      runIO(pass, 'matmul_resid', [['u', S], ['u', w.N!], ['u', w.K!], ['u', w.nb!], ['u', 128], ['u', 0]], [inBuf, w.sign!, w.scales!, resid], [out], S * w.N!)
    }
  }

  // Append `rows` rows of Dh f32 values into layer li's K (which=0) or V (which=1) cache at row
  // dstRow0, quantizing under q8 (copy_kv8 is a per-row workgroup kernel with two outputs, so it
  // cannot ride the flat COPY_KV dispatch).
  function appendKV(pass: GPUComputePassEncoder, src: GPUBuffer, which: 0 | 1, li: number, rows: number, dstRow0: number): void {
    const data = which === 0 ? Kc[li] : Vc[li]
    if (kv8) {
      setup(pass, 'copy_kv8', [['u', rows], ['u', Dh], ['u', dstRow0], ['u', 0]], [src], [data, which === 0 ? Ksc[li] : Vsc[li]])
      pass.dispatchWorkgroups(isFull('copy_kv8') ? rows : 1)
    } else {
      run(pass, COPY_KV, [['u', rows * Dh], ['u', dstRow0 * Dh], ['u', 0], ['u', 0]], [src], data, rows * Dh)
    }
  }
  // The attention kernel's cache bindings by KV mode (q8 adds the block-scale buffers).
  const attIns = (qr: GPUBuffer, li: number): GPUBuffer[] => {
    const ins = kv8 ? [qr, Kc[li], Vc[li], Ksc[li], Vsc[li]] : [qr, Kc[li], Vc[li]]
    if (roll) ins.push(rollCosT!, rollSinT!) // read-time K rotation tables
    return ins
  }

  function layer(pass: GPUComputePassEncoder, li: number, h: GPUBuffer, S: number, posBase: number, cos: GPUBuffer, sin: GPUBuffer): GPUBuffer {
    const Ltot = posBase + S
    const n1 = actBuf(S * Hd)
    rms(pass, h, `layers.${li}.input_layernorm`, S, Hd, n1)
    const qkv = W[`layers.${li}.attn.qkv`]

    if (useSG && S === 1 && !FORCE_SLOW) {
      // fused decode path: fold copies and elementwise ops into the matmul/norm kernels.
      const q = actBuf(H * Dh),
        k = actBuf(KV * Dh),
        v = actBuf(KV * Dh)
      const Ntot = qkv.N0! + qkv.N1! + qkv.N2!,
        gx = Math.min(Ntot, 65535)
      runWG(pass, 'matmul_split_sg', [['u', qkv.K!], ['u', qkv.nb!], ['u', qkv.N0!], ['u', qkv.N1!], ['u', qkv.N2!], ['u', gx]], [n1, qkv.sign!, qkv.scales!], [q, k, v], gx, Math.ceil(Ntot / gx))
      appendKV(pass, v, 1, li, KV, posBase * KV)
      const qr = actBuf(H * Dh)
      runN(pass, 'rmsnorm_rope_sg', [['u', H], ['u', Dh], ['f', A.rms_eps], ['u', 0], ['u', Dh], ['u', 0]], [q, W[`layers.${li}.attn.q_norm`].buf!, cos, sin], qr, H)
      // Sink mode stores K UNROPED (rotation happens at attention read): bind cos=1/sin=0 so
      // the same write kernels store the plain rmsnorm output.
      const kcos = roll ? rollOnes! : cos,
        ksin = roll ? rollZeros! : sin
      if (kv8) {
        // fused K write, quantizing into the cache (extra scale binding -> its own dispatch)
        setup(pass, 'rmsnorm_rope_sg_kv8', [['u', KV], ['u', Dh], ['f', A.rms_eps], ['u', posBase * KV], ['u', 0], ['u', 0]], [k, W[`layers.${li}.attn.k_norm`].buf!, kcos, ksin], [Kc[li], Ksc[li]])
        pass.dispatchWorkgroups(isFull('rmsnorm_rope_sg_kv8') ? KV : 1)
      } else {
        runN(pass, ROPE_K, [['u', KV], ['u', Dh], ['f', A.rms_eps], ['u', posBase * KV * Dh], ['u', Dh], ['u', 0]], [k, W[`layers.${li}.attn.k_norm`].buf!, kcos, ksin], Kc[li], KV)
      }
      cap(li, 'qr', qr)
      const att = actBuf(H * Dh)
      runN(pass, ATT, [['u', 1], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]], attIns(qr, li), att, H)
      cap(li, 'att', att)
      const o = W[`layers.${li}.attn.o_proj`],
        h2 = actBuf(Hd)
      residMM(pass, o, att, h, 1, h2)
      const n2 = actBuf(Hd)
      rms(pass, h2, `layers.${li}.post_attention_layernorm`, 1, Hd, n2)
      const gu = W[`layers.${li}.mlp.gateup`],
        sw = actBuf(F),
        nwgF = Math.ceil(F / ROWS_MR),
        gxF = Math.min(nwgF, 65535)
      runWG(pass, 'matmul_swiglu_mr_sg', [['u', gu.K!], ['u', gu.nb!], ['u', F], ['u', gxF], ['u', 0], ['u', 0]], [n2, gu.sign!, gu.scales!], [sw], gxF, Math.ceil(nwgF / gxF))
      cap(li, 'sw', sw)
      const d = W[`layers.${li}.mlp.down_proj`],
        hn = actBuf(Hd)
      residMM(pass, d, sw, h2, 1, hn)
      return hn
    }

    // prefill / no-subgroup path: separate kernels (kept verbatim; validates correctness end to end)
    const q = actBuf(S * H * Dh),
      k = actBuf(S * KV * Dh),
      v = actBuf(S * KV * Dh)
    fusedMM(pass, qkv, n1, S, [q, k, v])
    const qn = actBuf(S * H * Dh),
      kn = actBuf(S * KV * Dh)
    rms(pass, q, `layers.${li}.attn.q_norm`, S * H, Dh, qn)
    rms(pass, k, `layers.${li}.attn.k_norm`, S * KV, Dh, kn)
    const qr = actBuf(S * H * Dh),
      kr = actBuf(S * KV * Dh)
    run(pass, 'rope', [['u', S], ['u', H], ['u', Dh], ['u', 0]], [qn, cos, sin], qr, S * H * Dh)
    if (!roll) run(pass, 'rope', [['u', S], ['u', KV], ['u', Dh], ['u', 0]], [kn, cos, sin], kr, S * KV * Dh)
    appendKV(pass, roll ? kn : kr, 0, li, S * KV, posBase * KV) // sink mode caches K unroped
    appendKV(pass, v, 1, li, S * KV, posBase * KV)
    cap(li, 'qr', qr)
    const att = actBuf(S * H * Dh)
    const attF: Field[] = [['u', S], ['u', H], ['u', KV], ['u', Dh], ['u', posBase], ['u', Ltot]]
    runN(pass, ATT, attF, attIns(qr, li), att, S * H)
    cap(li, 'att', att)
    const o = W[`layers.${li}.attn.o_proj`],
      h2 = actBuf(S * Hd)
    residMM(pass, o, att, h, S, h2)
    const n2 = actBuf(S * Hd)
    rms(pass, h2, `layers.${li}.post_attention_layernorm`, S, Hd, n2)
    const gu = W[`layers.${li}.mlp.gateup`],
      g = actBuf(S * F),
      u = actBuf(S * F)
    fusedMM(pass, gu, n2, S, [g, u, dummy])
    const sw = actBuf(S * F)
    run(pass, 'swiglu', [['u', S * F], ['u', 0], ['u', 0], ['u', 0]], [g, u], sw, S * F)
    cap(li, 'sw', sw)
    const d = W[`layers.${li}.mlp.down_proj`],
      hn = actBuf(S * Hd)
    residMM(pass, d, sw, h2, S, hn)
    return hn
  }
  function lmHead(pass: GPUComputePassEncoder, fn: GPUBuffer, M: number, out: GPUBuffer): void {
    const lm = W.lm_head
    if (useSG && M === 1) {
      const gx = Math.min(lm.N!, 65535)
      runWG(pass, 'matmul_q2_sg', [['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', lm.zp!], ['u', gx], ['u', 0]], [fn, lm.codes!, lm.scales!], [out], gx, Math.ceil(lm.N! / gx))
    } else if (M === 1) {
      const gx = Math.min(lm.N!, 65535) // no-subgroup decode: workgroup-reduction 2-bit GEMV
      runWG(pass, 'matmul_q2_wg', [['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', lm.zp!], ['u', gx], ['u', 0]], [fn, lm.codes!, lm.scales!], [out], gx, Math.ceil(lm.N! / gx))
    } else if (useSG && M === SMALLM) {
      const gx = Math.min(lm.N!, 65535) // small-batch 2-bit GEMV: the code stream read once for all M rows
      runWG(pass, 'matmul_q2_sm', [['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', lm.zp!], ['u', gx], ['u', M]], [fn, lm.codes!, lm.scales!], [out], gx, Math.ceil(lm.N! / gx))
    } else {
      run(pass, 'matmul_q2', [['u', M], ['u', lm.N!], ['u', lm.K!], ['u', lm.nb!], ['u', 128], ['u', lm.zp!]], [fn, lm.codes!, lm.scales!], out, M * lm.N!)
    }
  }

  function stack(enc: GPUCommandEncoder, h: GPUBuffer, S: number, posBase: number): { fn: GPUBuffer; layer0: GPUBuffer | null } {
    const { cos, sin } = ropeBufs(posBase, S)
    const pass = enc.beginComputePass()
    let cur = h,
      layer0: GPUBuffer | null = null
    for (let li = 0; li < A.layers; li++) {
      cur = layer(pass, li, cur, S, posBase, cos, sin)
      if (li === 0) layer0 = cur
    }
    const fn = actBuf(S * Hd)
    rms(pass, cur, FINAL_NORM, S, Hd, fn)
    pass.end()
    return { fn, layer0 }
  }

  async function forward(ids: number[]): Promise<ForwardResult> {
    const S = ids.length
    if (S === 0) throw new Error('forward: no tokens to process')
    if (S > maxSeqLen) throw new Error(`forward: sequence length ${S} exceeds maxSeqLen ${maxSeqLen}`)
    // forward() overwrites K/V at positions 0..S-1 of every layer: whatever conversation the cache
    // held is gone, so nothing may reuse it. Same rule as an aborted prefill.
    fullHistory = []
    cacheLen = 0
    await ensureKvCapacity(S)
    const vocab = W.lm_head.N!
    // Segmented like runPrefill (bit-exact by the same composition the reuse gates prove), so a
    // long forward() neither dispatches >65535 workgroups in one dimension (S*heads at full
    // maxSeqLen) nor binds an S x vocab logits buffer beyond the negotiated binding limit.
    const embed = new Float32Array(S * Hd),
      layer0 = new Float32Array(S * Hd),
      finalnorm = new Float32Array(S * Hd),
      logits = new Float32Array(S * vocab)
    transients = []
    try {
      for (let off = 0; off < S; off += PREFILL_SEG) {
        const seg = ids.slice(off, off + PREFILL_SEG)
        const enc = device.createCommandEncoder()
        const embedOut = embedBatch(enc, seg)
        const { fn, layer0: l0 } = stack(enc, embedOut, seg.length, off)
        const lg = device.createBuffer({ size: seg.length * vocab * 4, usage: S_ | CS })
        transients.push(lg)
        const pass = enc.beginComputePass()
        lmHead(pass, fn, seg.length, lg)
        pass.end()
        device.queue.submit([enc.finish()])
        await device.queue.onSubmittedWorkDone()
        embed.set(await readback(embedOut, seg.length * Hd), off * Hd)
        layer0.set(await readback(l0!, seg.length * Hd), off * Hd)
        finalnorm.set(await readback(fn, seg.length * Hd), off * Hd)
        logits.set(await readback(lg, seg.length * vocab), off * vocab)
        flushTransients() // this segment's scratch is dead; the peak stays at one segment
      }
      return { embed, layer0, finalnorm, logits, vocab, sequenceLength: S }
    } finally {
      flushTransients()
      transients = null
    }
  }

  // Long prefills run in SEGMENTS. One submission holding all layers' scratch for S tokens keeps
  // ~S x 4.3 MB of VRAM in flight (a 1000-token prompt ~4 GB): on unified-memory devices that
  // stalls the whole system, especially with other models resident. Segments cap the spike at one
  // segment's scratch (flushed between them), stay bit-exact by the same composition the
  // reuse-prefill gates prove (each segment writes K/V at its offset and attends to everything
  // before it), and give an abort a place to land mid-prompt, so a superseded turn no longer
  // blocks the GPU for whole seconds. Returns the LAST segment's final-norm buffer + the final
  // token's row within it, or null when aborted (fullHistory is cleared: K/V is only partially
  // written, so nothing may reuse the sequence).
  async function runPrefill(ids: number[], posBase: number, signal?: AbortSignal): Promise<{ fn: GPUBuffer; lastRow: number } | null> {
    let fn: GPUBuffer | null = null
    let lastRow = 0
    for (let off = 0; off < ids.length; off += PREFILL_SEG) {
      if (off > 0 && signal?.aborted) {
        fullHistory = []
        cacheLen = 0
        return null
      }
      const seg = ids.slice(off, off + PREFILL_SEG)
      const enc = device.createCommandEncoder()
      fn = stack(enc, embedBatch(enc, seg), seg.length, posBase + off).fn
      lastRow = seg.length - 1
      device.queue.submit([enc.finish()])
      if (off + PREFILL_SEG < ids.length) {
        await device.queue.onSubmittedWorkDone()
        flushTransients() // this segment's scratch is dead; the peak stays at one segment
      }
    }
    return { fn: fn!, lastRow }
  }

  // GPU-resident decode: argmax + embedding gather run on the GPU so the token id never leaves it;
  // chain syncN steps per CPU sync (deferred readback). Bit-exact: only the readback timing changes.
  async function generateImpl(ids: number[], posBase: number, nTokens: number, full: Set<string> | null = null, syncN: number = SYNC_N, ctl?: { stopTokens?: number[]; onToken?: (id: number) => void; signal?: AbortSignal }): Promise<RawGenResult> {
    await ensureKvCapacity(posBase + ids.length + nTokens)
    FULL = full
    const vocab = W.lm_head.N!
    const tokBuf = device.createBuffer({ size: Math.max(1, nTokens) * 4, usage: S_ | CS }) // GPU-resident token ids
    const embG = device.createBuffer({ size: Hd * 4, usage: S_ | CS | CD }) // GPU embedding of the current token (lives across the whole call)
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })
    transients = [] // track prefill scratch so it can be destroyed once the prefill completes
    try {
      const t0 = performance.now()
      // prefill (CPU embed of the known prompt, segmented) -> last hidden -> lm_head -> GPU argmax
      const pfx = await runPrefill(ids, posBase, ctl?.signal)
      if (!pfx) return { prefillMs: performance.now() - t0, decodeMs: 0, tokPerSec: 0, tokens: [], firstArgmax: -1, recMs: 0, gpuMs: 0, rbMs: 0 }
      const encP = device.createCommandEncoder()
      const lastP = actBuf(Hd)
      encP.copyBufferToBuffer(pfx.fn, pfx.lastRow * Hd * 4, lastP, 0, Hd * 4)
      let pp = encP.beginComputePass()
      lmHead(pp, lastP, 1, lg)
      pp.end()
      pp = encP.beginComputePass()
      runN(pp, 'argmax', [['u', vocab], ['u', 0], ['u', 0], ['u', 0]], [lg], tokBuf, 1)
      pp.end()
      device.queue.submit([encP.finish()])
      await device.queue.onSubmittedWorkDone()
      const firstTok = (await readbackU32(tokBuf, 1))[0]
      flushTransients() // the last segment's scratch is dead now
      const prefillMs = performance.now() - t0

      const gen: number[] = []
      let recMs = 0,
        gpuMs = 0,
        rbMs = 0
      const t1 = performance.now()
      let total = 1 // decode positions consumed (incl. prefill's first)
      const stopSet = ctl?.stopTokens ? new Set(ctl.stopTokens) : null
      let stopped = stopSet?.has(firstTok) ?? false
      if (!stopped) {
        gen.push(firstTok) // a stop token is never emitted, even at position 0
        ctl?.onToken?.(firstTok)
      }
      poolInvalidate() // rebuild cached bind groups against this call's buffers
      let slot = posBase + ids.length // cache slot where the next fed token's K/V lands
      while (total < nTokens && !stopped) {
        if (ctl?.signal?.aborted) break
        const batch = Math.min(syncN, nTokens - total)
        slot = evictFor(slot, batch) // sink mode: roll the window before the batch needs the room
        poolUse('decode') // reuse decode scratch + uniforms across batches; resets the slot indices
        let t = performance.now()
        const enc = device.createCommandEncoder()
        for (let j = 0; j < batch; j++) {
          const idxOut = total + j,
            pos = slot + j
          let pass = enc.beginComputePass()
          runN(pass, 'embed_gather', [['u', Hd], ['u', idxOut - 1], ['u', 0], ['u', 0]], [tokBuf, embWqG, tgt4G, embScalesG, embZpG], embG, 1)
          pass.end()
          const r = stack(enc, embG, 1, pos)
          const last = actBuf(Hd)
          enc.copyBufferToBuffer(r.fn, 0, last, 0, Hd * 4)
          pass = enc.beginComputePass()
          lmHead(pass, last, 1, lg)
          runN(pass, 'argmax', [['u', vocab], ['u', idxOut], ['u', 0], ['u', 0]], [lg], tokBuf, 1)
          pass.end()
        }
        device.queue.submit([enc.finish()])
        recMs += performance.now() - t
        t = performance.now()
        await device.queue.onSubmittedWorkDone()
        gpuMs += performance.now() - t
        t = performance.now()
        const toks = await readbackU32(tokBuf, total + batch)
        rbMs += performance.now() - t
        let fed = batch // slots that belong to the history (EOS discards the rest of the batch)
        for (let j = 0; j < batch; j++) {
          const tk = toks[total + j]
          if (stopSet?.has(tk)) { stopped = true; fed = j; break } // EOS lands at the batch boundary (greedy)
          gen.push(tk)
          ctl?.onToken?.(tk)
        }
        total += batch
        slot += fed
      }
      cacheLen = slot // sink-mode bookkeeping: the last history token's (re-feed) slot
      const decodeMs = performance.now() - t1,
        nd = Math.max(1, gen.length - 1)
      return { prefillMs, decodeMs, tokPerSec: nd / (decodeMs / 1000), tokens: gen, firstArgmax: firstTok, recMs: recMs / nd, gpuMs: gpuMs / nd, rbMs: rbMs / nd }
    } finally {
      // Restore the shared flags even when a step throws (a stuck active pool would corrupt every
      // later call), and release this call's GPU scratch instead of waiting on GC.
      poolUse(null)
      FULL = null
      flushTransients()
      transients = null
      tokBuf.destroy()
      embG.destroy()
      lg.destroy()
    }
  }

  // Sampled decode (do_sample): the GPU pre-filters the logits in place (repetition_penalty +
  // no_repeat_ngram bans) and selects the top-K via K masked-argmax passes; only K (id, logit) pairs
  // are read back, and the CPU does temperature + softmax + MT19937 multinomial (exact transformers.js
  // semantics). Per-step (syncN=1) because the chosen token is picked on the CPU and feeds the next
  // step's embed gather. A GREEDY turn with processors (repetition_penalty / no_repeat_ngram) also
  // runs here - the chain filters, then pick() takes the penalized argmax without touching the RNG.
  // Plain greedy decode (generateImpl) is the separate, untouched GPU-resident path.
  async function generateSampledImpl(ids: number[], posBase: number, nTokens: number, genOpts: GenerateOptions, history: number[], rngIn?: MT19937): Promise<RawGenResult> {
    await ensureKvCapacity(posBase + ids.length + nTokens)
    // `ids` = the tokens to prefill this turn (the whole prompt, or [lastToken, ...delta] on reuse).
    // `history` = the FULL conversation token sequence (shared, mutated): the sampler's penalty/ngram
    // see the entire sequence, like transformers.js; generated tokens are pushed onto it.
    const sampled = genOpts.temperature != null && genOpts.temperature > 0 && genOpts.temperature !== 1
    const vocab = W.lm_head.N!
    const K = Math.max(1, Math.min(genOpts.topK ?? 20, vocab))
    // logprobs: run extra argmax rounds when N exceeds the sampling top-K, but NEVER let it widen
    // the sampling pool - the draw stays over the first K candidates, bit-identical with or
    // without logprobs on.
    const lpN = Math.max(0, Math.min(Math.floor(genOpts.logprobs ?? 0), 32, vocab))
    const KT = Math.max(K, lpN)
    const temperature = genOpts.temperature ?? 1
    const penalty = genOpts.repetitionPenalty ?? 1
    const ngramN = genOpts.noRepeatNgramSize ?? 0
    const stopSet = genOpts.stopTokens ? new Set(genOpts.stopTokens) : null
    const onToken = genOpts.onToken
    const signal = genOpts.signal
    const rng = rngIn ?? new MT19937(genOpts.seed)

    // persistent buffers (stable across steps for bind-group caching; not via actBuf)
    const tokBuf = device.createBuffer({ size: Math.max(1, nTokens) * 4, usage: S_ | CS | CD })
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })
    const candIds = device.createBuffer({ size: KT * 4, usage: S_ | CS })
    const candVals = device.createBuffer({ size: KT * 4, usage: S_ | CS })
    const lseBuf = device.createBuffer({ size: 4, usage: S_ | CS }) // log-sum-exp normalizer (logprobs)
    const affBuf = device.createBuffer({ size: maxSeqLen * 4, usage: S_ | CD }) // upper bound = full vocab can't exceed seq len
    const banBuf = device.createBuffer({ size: maxSeqLen * 4, usage: S_ | CD })
    const rbBuf = device.createBuffer({ size: KT * 8 + (lpN ? 4 : 0), usage: GPUBufferUsage.MAP_READ | CD })
    const embG = device.createBuffer({ size: Hd * 4, usage: S_ | CS | CD })

    // upload the CPU-computed deduped id set + ngram bans for the current history; return their lengths
    const writeAffBan = (history: number[]): { affLen: number; banLen: number } => {
      const aff = penalty !== 1 ? affectedIds(history) : new Uint32Array(0)
      if (aff.length) device.queue.writeBuffer(affBuf, 0, aff)
      const ban = ngramN > 0 ? ngramBans(history, ngramN) : []
      if (ban.length) device.queue.writeBuffer(banBuf, 0, Uint32Array.from(ban))
      return { affLen: aff.length, banLen: ban.length }
    }
    // penalty pre-filter (+ logprobs normalizer) + KT masked-argmax, all in the given pass (after
    // lm_head wrote lg). The log-sum-exp runs BEFORE the argmax rounds: those mask their winners
    // in lg in place, which would corrupt the sum.
    const samplerChain = (pass: GPUComputePassEncoder, affLen: number, banLen: number): void => {
      setup(pass, 'sampler_penalty', [['u', affLen], ['u', banLen], ['f', penalty], ['u', 0xff800000]], [affBuf, banBuf], [lg])
      pass.dispatchWorkgroups(1)
      if (lpN) {
        setup(pass, 'logsumexp', [['u', vocab], ['u', 0], ['u', 0], ['u', 0]], [lg], [lseBuf])
        pass.dispatchWorkgroups(1)
      }
      for (let r = 0; r < KT; r++) {
        setup(pass, 'argmax_masked', [['u', vocab], ['u', r], ['u', 0], ['u', 0]], [lg], [candIds, candVals])
        pass.dispatchWorkgroups(1)
      }
    }
    const copyCands = (enc: GPUCommandEncoder): void => {
      enc.copyBufferToBuffer(candIds, 0, rbBuf, 0, KT * 4)
      enc.copyBufferToBuffer(candVals, 0, rbBuf, KT * 4, KT * 4)
      if (lpN) enc.copyBufferToBuffer(lseBuf, 0, rbBuf, KT * 8, 4)
    }
    const readCands = async (): Promise<{ ci: Uint32Array; cv: Float32Array; lse: number }> => {
      await rbBuf.mapAsync(GPUMapMode.READ)
      const mapped = rbBuf.getMappedRange()
      const ci = new Uint32Array(mapped.slice(0, KT * 4))
      const cv = new Float32Array(mapped.slice(KT * 4, KT * 8))
      const lse = lpN ? new Float32Array(mapped.slice(KT * 8, KT * 8 + 4))[0] : 0
      rbBuf.unmap()
      return { ci, cv, lse }
    }
    // logprobs bookkeeping: the chosen token's logit (set by every chooseToken path) and the
    // per-emitted-token records (aligned with the returned tokens)
    let chosenLogit = 0
    const lpOut: TokenLogprobs[] | null = lpN ? [] : null
    const recordLp = (ci: Uint32Array, cv: Float32Array, lse: number): void => {
      if (!lpOut) return
      const top: { id: number; logprob: number }[] = []
      for (let i = 0; i < lpN; i++) top.push({ id: ci[i], logprob: cv[i] - lse })
      lpOut.push({ logprob: chosenLogit - lse, top })
    }
    // candidates are descending, so ci[0] is the argmax of the penalized logits (greedy+processors);
    // sampling and filtering see only the first K candidates (KT > K rounds exist for logprobs only)
    const pick = (ciAll: Uint32Array, cvAll: Float32Array): number => {
      const ci = ciAll.subarray(0, K)
      const cv = cvAll.subarray(0, K)
      const tk = sampled ? sampleFromCandidates(ci, cv, temperature, rng) : ci[0]
      chosenLogit = cv[ci.indexOf(tk)]
      return tk
    }
    const filter = genOpts.candidateFilter
    // Constrained pick (candidateFilter): keep the permitted subset in rank order and pick within
    // it - greedy takes the best permitted, sampling renormalizes the draw over them (exactly one
    // RNG draw per emitted token, like the unconstrained path). When the whole top-K is rejected,
    // walk the FULL vocabulary in logit order (rare: valid grammars almost always admit a top-K
    // token) and rebuild up to K permitted candidates; a grammar that admits nothing is a bug in
    // the filter and fails loudly.
    const chooseToken = async (ciAll: Uint32Array, cvAll: Float32Array): Promise<number> => {
      if (!filter) return pick(ciAll, cvAll)
      const ci = ciAll.subarray(0, K)
      const cv = cvAll.subarray(0, K)
      const perm = new Set(filter(ci, cv))
      if (perm.size > 0) {
        const pIds: number[] = []
        const pVals: number[] = []
        for (let i = 0; i < ci.length; i++)
          if (perm.has(ci[i])) {
            pIds.push(ci[i])
            pVals.push(cv[i])
          }
        const tk = sampled ? sampleFromCandidates(pIds, pVals, temperature, rng) : pIds[0]
        chosenLogit = pVals[pIds.indexOf(tk)]
        return tk
      }
      const all = await readback(lg, vocab) // penalized logits; every argmax round masked its winner in place
      // Restore the masked entries from the candidates already read back: the filter-rejected
      // top-K re-check and fail again (filters are deterministic), and the extra logprobs-only
      // rounds (K..KT) must stay reachable or logprobs would change constrained output.
      for (let i = 0; i < ciAll.length; i++) all[ciAll[i]] = cvAll[i]
      const order = Array.from(all.keys()).sort((a, b) => all[b] - all[a] || a - b)
      const pIds: number[] = []
      const pVals: number[] = []
      const B = 512
      for (let i = 0; i < order.length && pIds.length < K; i += B) {
        if (all[order[i]] === -Infinity) break // sorted: only masked/banned entries remain
        const batch = order.slice(i, i + B).filter((id) => all[id] !== -Infinity)
        const ok = new Set(filter(Uint32Array.from(batch), Float32Array.from(batch.map((id) => all[id]))))
        for (const id of batch) {
          if (ok.has(id)) {
            pIds.push(id)
            pVals.push(all[id])
            if (pIds.length >= K) break
          }
        }
      }
      if (pIds.length === 0) throw new Error('bitgpu: candidateFilter permitted no token in the entire vocabulary')
      const tk = sampled ? sampleFromCandidates(pIds, pVals, temperature, rng) : pIds[0]
      chosenLogit = pVals[pIds.indexOf(tk)]
      return tk
    }

    transients = [] // track prefill scratch so it can be destroyed once the prefill completes
    try {
      const t0 = performance.now()
      // prefill (segmented) -> last hidden -> lm_head -> sampler chain, CPU samples the first token
      const pfx = await runPrefill(ids, posBase, signal)
      if (!pfx) return { prefillMs: performance.now() - t0, decodeMs: 0, tokPerSec: 0, tokens: [], firstArgmax: -1, recMs: 0, gpuMs: 0, rbMs: 0, rng }
      const encP = device.createCommandEncoder()
      const lastP = actBuf(Hd)
      encP.copyBufferToBuffer(pfx.fn, pfx.lastRow * Hd * 4, lastP, 0, Hd * 4)
      const pf = writeAffBan(history)
      let pass = encP.beginComputePass()
      lmHead(pass, lastP, 1, lg)
      samplerChain(pass, pf.affLen, pf.banLen)
      pass.end()
      copyCands(encP)
      device.queue.submit([encP.finish()])
      const first = await readCands() // mapAsync waits for the submitted work; no separate sync needed
      const firstTok = await chooseToken(first.ci, first.cv)
      flushTransients() // the last segment's scratch is dead now (the map wait proved the GPU is done with it)
      const prefillMs = performance.now() - t0

      const gen: number[] = []
      let stopped = stopSet?.has(firstTok) ?? false
      if (!stopped) {
        gen.push(firstTok) // a stop token is never emitted or recorded, even at position 0
        history.push(firstTok)
        recordLp(first.ci, first.cv, first.lse)
        onToken?.(firstTok)
        device.queue.writeBuffer(tokBuf, 0, new Uint32Array([firstTok]))
      }

      let recMs = 0, gpuMs = 0, rbMs = 0
      const t1 = performance.now()
      let total = 1
      poolInvalidate() // rebuild cached bind groups against this call's buffers
      let slot = posBase + ids.length // cache slot where the next fed token's K/V lands
      while (total < nTokens && !stopped) {
        if (signal?.aborted) break
        poolUse('decode')
        slot = evictFor(slot, 1) // sink mode: roll the window before this step needs the room
        const idxOut = total, pos = slot
        let t = performance.now()
        const { affLen, banLen } = writeAffBan(history)
        const enc = device.createCommandEncoder()
        let p2 = enc.beginComputePass()
        runN(p2, 'embed_gather', [['u', Hd], ['u', idxOut - 1], ['u', 0], ['u', 0]], [tokBuf, embWqG, tgt4G, embScalesG, embZpG], embG, 1)
        p2.end()
        const r = stack(enc, embG, 1, pos)
        const last = actBuf(Hd)
        enc.copyBufferToBuffer(r.fn, 0, last, 0, Hd * 4)
        p2 = enc.beginComputePass()
        lmHead(p2, last, 1, lg)
        samplerChain(p2, affLen, banLen)
        p2.end()
        copyCands(enc)
        device.queue.submit([enc.finish()])
        recMs += performance.now() - t
        t = performance.now()
        // ONE sync per token: mapAsync on the readback buffer already waits for the submitted
        // work its copy depends on, so a separate onSubmittedWorkDone is a second full CPU-GPU
        // round-trip for nothing. The map wait is dominated by GPU time, so it is attributed to
        // gpuMs; the post-map array slicing is microseconds (rbMs stays for API stability).
        const { ci, cv, lse } = await readCands()
        gpuMs += performance.now() - t
        t = performance.now()
        const tk = await chooseToken(ci, cv)
        rbMs += performance.now() - t
        total += 1
        if (stopSet?.has(tk)) { stopped = true; break } // EOS: stop without emitting the stop token
        gen.push(tk)
        history.push(tk)
        recordLp(ci, cv, lse)
        onToken?.(tk)
        device.queue.writeBuffer(tokBuf, idxOut * 4, new Uint32Array([tk])) // feed the next step's embed gather
        slot += 1
      }
      cacheLen = slot // sink-mode bookkeeping (filled slots; the last emitted token is unfed)
      const decodeMs = performance.now() - t1
      const nd = Math.max(1, gen.length - 1)
      return { prefillMs, decodeMs, tokPerSec: nd / (decodeMs / 1000), tokens: gen, firstArgmax: firstTok, recMs: recMs / nd, gpuMs: gpuMs / nd, rbMs: rbMs / nd, rng, ...(lpOut ? { lp: lpOut } : {}) }
    } finally {
      // Restore the shared flags even when a step throws, and release this call's GPU buffers.
      poolUse(null)
      flushTransients()
      transients = null
      for (const b of [tokBuf, lg, candIds, candVals, lseBuf, affBuf, banBuf, rbBuf, embG]) b.destroy()
    }
  }

  // Prompt-lookup speculative decoding, greedy AND sampled. Each step drafts the continuation
  // from an n-gram match in the sequence so far (draftNgram; no draft model), then verifies all
  // drafts in ONE batched forward through the prefill path: it writes K/V for every drafted
  // position and yields logits for every row, so the accepted prefix plus one model-chosen token
  // are emitted for the cost of a single pass. K/V written for rejected positions is overwritten
  // by later steps before anything ever attends to it (attention at position p reads only 0..p).
  // Exactness: row j is only consumed when drafts 0..j-1 were accepted, so its logits (and, when
  // sampling, its penalty/ban lists, computed assuming those drafts) describe the true sequence;
  // the sampler draws sequentially on the CPU, so the MT19937 stream advances exactly one draw
  // per EMITTED token, in order - the output equals non-speculative decoding for the same seed.
  // With no match the step degenerates to a normal single-token pass (S=1 keeps the fused path).
  async function generatePldImpl(ids: number[], posBase: number, nTokens: number, genOpts: GenerateOptions, history: number[], rngIn?: MT19937): Promise<RawGenResult> {
    await ensureKvCapacity(posBase + ids.length + nTokens)
    const sampled = genOpts.temperature != null && genOpts.temperature > 0 && genOpts.temperature !== 1
    const vocab = W.lm_head.N!
    const K = Math.max(1, Math.min(genOpts.topK ?? 20, vocab))
    const temperature = genOpts.temperature ?? 1
    const penalty = genOpts.repetitionPenalty ?? 1
    const ngramN = genOpts.noRepeatNgramSize ?? 0
    const pl = typeof genOpts.promptLookup === 'object' && genOpts.promptLookup !== null ? genOpts.promptLookup : {}
    const ngramSize = Math.max(2, pl.ngramSize ?? 3)
    const maxDraft = Math.max(1, pl.maxDraft ?? 8)
    const stopSet = genOpts.stopTokens ? new Set(genOpts.stopTokens) : null
    const onToken = genOpts.onToken
    const signal = genOpts.signal
    const rng = rngIn ?? new MT19937(genOpts.seed)
    // Greedy with processors (repetition_penalty / no_repeat_ngram) also needs the per-row sampler
    // chain - the penalties see each row's history - but picks the penalized argmax, no RNG draw.
    const useChain = sampled || penalty !== 1 || ngramN > 0

    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS | CD }) // one row, target of the per-row copy
    const lgAll = device.createBuffer({ size: (maxDraft + 1) * vocab * 4, usage: S_ | CS }) // lm_head over all rows
    const idsOut = device.createBuffer({ size: (maxDraft + 1) * 4, usage: S_ | CS }) // greedy: argmax per row
    const candIds = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const candVals = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const affBuf = device.createBuffer({ size: (maxSeqLen + maxDraft + 1) * 4, usage: S_ | CD })
    const banBuf = device.createBuffer({ size: (maxSeqLen + maxDraft + 1) * 4, usage: S_ | CD })
    const rbAll = device.createBuffer({ size: (maxDraft + 1) * K * 8, usage: GPUBufferUsage.MAP_READ | CD })
    // Step input token ids + their gathered embeddings, written per step (never re-created: the
    // pooled bind groups cache buffer identities, so a per-step upload would go stale the moment
    // it was destroyed). The embeddings are gathered ON the GPU (embed_gather_batch) from the
    // uploaded ids - only S u32s cross the bus per step, and no CPU embedding tables exist.
    const pldIds = device.createBuffer({ size: (maxDraft + 1) * 4, usage: S_ | CD })
    const embIn = device.createBuffer({ size: (maxDraft + 1) * Hd * 4, usage: S_ | CD })

    const writeAffBan = (h: number[]): { affLen: number; banLen: number } => {
      const aff = penalty !== 1 ? affectedIds(h) : new Uint32Array(0)
      if (aff.length) device.queue.writeBuffer(affBuf, 0, aff)
      const ban = ngramN > 0 ? ngramBans(h, ngramN) : []
      if (ban.length) device.queue.writeBuffer(banBuf, 0, Uint32Array.from(ban))
      return { affLen: aff.length, banLen: ban.length }
    }
    const samplerChain = (pass: GPUComputePassEncoder, affLen: number, banLen: number): void => {
      setup(pass, 'sampler_penalty', [['u', affLen], ['u', banLen], ['f', penalty], ['u', 0xff800000]], [affBuf, banBuf], [lg])
      pass.dispatchWorkgroups(1)
      for (let r = 0; r < K; r++) {
        setup(pass, 'argmax_masked', [['u', vocab], ['u', r], ['u', 0], ['u', 0]], [lg], [candIds, candVals])
        pass.dispatchWorkgroups(1)
      }
    }
    // draw the row-j candidates from a mapped copy of rbAll (candidates are descending, so
    // index 0 is the penalized argmax for a greedy-with-processors turn - no RNG draw)
    const rowDraw = (m: ArrayBuffer, j: number): number =>
      sampled
        ? sampleFromCandidates(new Uint32Array(m, j * K * 8, K), new Float32Array(m, j * K * 8 + K * 4, K), temperature, rng)
        : new Uint32Array(m, j * K * 8, 1)[0]

    transients = []
    try {
      const t0 = performance.now()
      // prefill (segmented) -> last hidden -> lm_head -> argmax (greedy) or sampler chain (sampled)
      const pfx = await runPrefill(ids, posBase, signal)
      if (!pfx) return { prefillMs: performance.now() - t0, decodeMs: 0, tokPerSec: 0, tokens: [], firstArgmax: -1, recMs: 0, gpuMs: 0, rbMs: 0, spec: { steps: 0, drafted: 0, accepted: 0 }, rng }
      const encP = device.createCommandEncoder()
      const lastP = actBuf(Hd)
      encP.copyBufferToBuffer(pfx.fn, pfx.lastRow * Hd * 4, lastP, 0, Hd * 4)
      const pf = useChain ? writeAffBan(history) : null
      let pass = encP.beginComputePass()
      lmHead(pass, lastP, 1, lg)
      if (pf) samplerChain(pass, pf.affLen, pf.banLen)
      else runN(pass, 'argmax', [['u', vocab], ['u', 0], ['u', 0], ['u', 0]], [lg], idsOut, 1)
      pass.end()
      if (pf) {
        encP.copyBufferToBuffer(candIds, 0, rbAll, 0, K * 4)
        encP.copyBufferToBuffer(candVals, 0, rbAll, K * 4, K * 4)
      }
      device.queue.submit([encP.finish()])
      await device.queue.onSubmittedWorkDone()
      let firstTok: number
      if (pf) {
        await rbAll.mapAsync(GPUMapMode.READ)
        const m = rbAll.getMappedRange().slice(0)
        rbAll.unmap()
        firstTok = rowDraw(m, 0)
      } else {
        firstTok = (await readbackU32(idsOut, 1))[0]
      }
      flushTransients()
      const prefillMs = performance.now() - t0

      const gen: number[] = []
      let stopped = stopSet?.has(firstTok) ?? false
      if (!stopped) {
        gen.push(firstTok)
        history.push(firstTok)
        onToken?.(firstTok)
      }
      poolInvalidate() // rebuild cached bind groups against THIS call's persistent buffers
      let total = 1
      let tLast = firstTok
      let pos = posBase + ids.length // where tLast's K/V lands on the next step
      let specSteps = 0,
        drafted = 0,
        accepted = 0
      let recMs = 0,
        gpuMs = 0,
        rbMs = 0
      const t1 = performance.now()
      while (total < nTokens && !stopped) {
        if (signal?.aborted) break
        pos = evictFor(pos, maxDraft + 1) // sink mode: roll before a full drafting step needs the room
        const kMax = Math.min(maxDraft, nTokens - total - 1, maxSeqLen - 1 - pos)
        const drafts = kMax > 0 ? draftNgram(history, ngramSize, kMax) : []
        const S = drafts.length + 1
        await ensureKvCapacity(pos + S)
        let t = performance.now()
        // Pools: 'pld1' = the fused single-token sequence, 'pldm' = the verify sequence (identical
        // dispatch order for every S in 2..9; buffers rounded to 9 rows so sizes are S-invariant).
        // S > 9 (a raised maxDraft) falls back to unpooled scalar kernels - correct, just slower.
        if (S === 1) poolUse('pld1')
        else if (useSG && S <= 9) poolUse('pldm', S, 9)
        else poolUse(null)
        device.queue.writeBuffer(pldIds, 0, new Uint32Array([tLast, ...drafts]))
        // shared forward: [tLast, ...drafts] at pos -> K/V for pos..pos+S-1 + logits for every row.
        // Routed through the small-batch kernels (weights read once for all S rows); S=1 keeps
        // the fused decode path via the S===1 branches.
        SMALLM = useSG && S >= 2 && S <= 9 ? S : 0
        const enc = device.createCommandEncoder()
        const gp = enc.beginComputePass()
        run(gp, 'embed_gather_batch', [['u', S], ['u', Hd], ['u', 0], ['u', 0]], [pldIds, embWqG, tgt4G, embScalesG, embZpG], embIn, S * Hd)
        gp.end()
        const r = stack(enc, embIn, S, pos)
        pass = enc.beginComputePass()
        lmHead(pass, r.fn, S, lgAll)
        pass.end()
        SMALLM = 0
        if (!useChain) {
          for (let j = 0; j < S; j++) {
            enc.copyBufferToBuffer(lgAll, j * vocab * 4, lg, 0, vocab * 4)
            const p = enc.beginComputePass()
            runN(p, 'argmax', [['u', vocab], ['u', j], ['u', 0], ['u', 0]], [lg], idsOut, 1)
            p.end()
          }
          device.queue.submit([enc.finish()])
        } else {
          device.queue.submit([enc.finish()])
          // per row: upload that row's penalty state (queue order puts it before the row's
          // dispatches), copy the row into lg, run the chain, stash the K candidates in rbAll
          for (let j = 0; j < S; j++) {
            const rowHist = j === 0 ? history : [...history, ...drafts.slice(0, j)]
            const { affLen, banLen } = writeAffBan(rowHist)
            const e2 = device.createCommandEncoder()
            e2.copyBufferToBuffer(lgAll, j * vocab * 4, lg, 0, vocab * 4)
            const p = e2.beginComputePass()
            samplerChain(p, affLen, banLen)
            p.end()
            e2.copyBufferToBuffer(candIds, 0, rbAll, j * K * 8, K * 4)
            e2.copyBufferToBuffer(candVals, 0, rbAll, j * K * 8 + K * 4, K * 4)
            device.queue.submit([e2.finish()])
          }
        }
        recMs += performance.now() - t
        t = performance.now()
        await device.queue.onSubmittedWorkDone()
        gpuMs += performance.now() - t
        t = performance.now()
        // acceptance: emit row draws in order while they agree with the drafts; the first
        // disagreement is itself a valid emission (it came from the true distribution)
        const emitted: number[] = []
        if (!useChain) {
          const outs = await readbackU32(idsOut, S)
          for (let j = 0; j < S; j++) {
            const tk = outs[j]
            if (stopSet?.has(tk)) { stopped = true; break }
            emitted.push(tk)
            if (j < drafts.length && tk !== drafts[j]) break
          }
        } else {
          await rbAll.mapAsync(GPUMapMode.READ)
          const m = rbAll.getMappedRange().slice(0)
          rbAll.unmap()
          for (let j = 0; j < S; j++) {
            const tk = rowDraw(m, j) // draws only for rows actually reached: one per emitted token
            if (stopSet?.has(tk)) { stopped = true; break }
            emitted.push(tk)
            if (j < drafts.length && tk !== drafts[j]) break
          }
        }
        rbMs += performance.now() - t
        specSteps++
        drafted += drafts.length
        accepted += Math.max(0, emitted.length - 1)
        for (const tk of emitted) {
          gen.push(tk)
          history.push(tk)
          onToken?.(tk)
        }
        total += emitted.length
        flushTransients()
        if (emitted.length === 0) break // stop token at row 0
        pos += emitted.length
        tLast = emitted[emitted.length - 1]
      }
      cacheLen = pos // sink-mode bookkeeping: the last history token's (re-feed) slot
      const decodeMs = performance.now() - t1
      const nd = Math.max(1, gen.length - 1)
      return {
        prefillMs,
        decodeMs,
        tokPerSec: nd / (decodeMs / 1000),
        tokens: gen,
        firstArgmax: firstTok,
        recMs: recMs / nd,
        gpuMs: gpuMs / nd,
        rbMs: rbMs / nd,
        spec: { steps: specSteps, drafted, accepted },
        rng,
      }
    } finally {
      SMALLM = 0 // encode-time flag; a throw mid-encode must not leak it into other paths
      poolUse(null)
      flushTransients()
      transients = null
      for (const b of [lg, lgAll, idsOut, candIds, candVals, affBuf, banBuf, rbAll, embIn, pldIds]) b.destroy()
    }
  }

  // Run ONE decode step at the same position through the fused path and the slow (known-good) path
  // and return layer-0 checkpoints + final norm + logits for each, so a divergence pinpoints the
  // first fused kernel that differs.
  async function debugDecode(prefillIds: number[]): Promise<{ fast: Record<string, Float32Array>; slow: Record<string, Float32Array> }> {
    await ensureKvCapacity(prefillIds.length + 1)
    const encP = device.createCommandEncoder()
    stack(encP, embedBatch(encP, prefillIds), prefillIds.length, 0)
    device.queue.submit([encP.finish()])
    await device.queue.onSubmittedWorkDone()
    const pos = prefillIds.length,
      tok = prefillIds[prefillIds.length - 1]
    const runStep = async (forceSlow: boolean): Promise<Record<string, Float32Array>> => {
      FORCE_SLOW = forceSlow
      DBG0 = {}
      const enc = device.createCommandEncoder()
      const r = stack(enc, embedBatch(enc, [tok]), 1, pos)
      const lg = device.createBuffer({ size: W.lm_head.N! * 4, usage: S_ | CS })
      const pass = enc.beginComputePass()
      lmHead(pass, r.fn, 1, lg)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      const ck: Record<string, Float32Array> = {}
      for (const [name, b] of Object.entries(DBG0)) ck[name] = await readback(b, b.size / 4)
      const off = pos * KV * Dh
      if (!kv16 && !kv8) {
        // f32-mode only: readback() types the bytes as f32, which is wrong for an f16/q8 cache
        ck.kc = (await readback(Kc[0], kvCapacity * KV * Dh)).slice(off, off + KV * Dh) // kvCapacity, not maxSeqLen: the cache may not have grown yet
        ck.vc = (await readback(Vc[0], kvCapacity * KV * Dh)).slice(off, off + KV * Dh)
      }
      ck.fn = await readback(r.fn, Hd)
      ck.logits = await readback(lg, W.lm_head.N!)
      FORCE_SLOW = false
      DBG0 = null
      return ck
    }
    const fast = await runStep(false),
      slow = await runStep(true)
    return { fast, slow }
  }

  // Debug hook for the browser harness: run a prefill for `ids` (history = ids), then return the GPU
  // lm_head logits (base, pre-penalty), the GPU penalized logits, and the GPU top-K. The page penalizes
  // `base` on the CPU and diffs vs `penalized` (exact, same input), and compares its top-K vs candIds,
  // validating sampler_penalty.wgsl and argmax_masked.wgsl in isolation against the headless-checked math.
  async function debugSampler(ids: number[], genOpts: GenerateOptions): Promise<{ base: Float32Array; penalized: Float32Array; candIds: Uint32Array; candVals: Float32Array }> {
    const vocab = W.lm_head.N!
    const K = Math.max(1, Math.min(genOpts.topK ?? 20, vocab))
    const penalty = genOpts.repetitionPenalty ?? 1
    const ngramN = genOpts.noRepeatNgramSize ?? 0
    const lg = device.createBuffer({ size: vocab * 4, usage: S_ | CS })
    const candIds = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const candVals = device.createBuffer({ size: K * 4, usage: S_ | CS })
    const aff = penalty !== 1 ? affectedIds(ids) : new Uint32Array(0)
    const ban = ngramN > 0 ? ngramBans(ids, ngramN) : []
    const affBuf = upload(aff.length ? aff : new Uint32Array(1), S_ | CD)
    const banBuf = upload(ban.length ? Uint32Array.from(ban) : new Uint32Array(1), S_ | CD)
    // pass 1: prefill -> lm_head -> base logits
    const enc1 = device.createCommandEncoder()
    const { fn } = stack(enc1, embedBatch(enc1, ids), ids.length, 0)
    const lastP = device.createBuffer({ size: Hd * 4, usage: S_ | CS | CD })
    enc1.copyBufferToBuffer(fn, (ids.length - 1) * Hd * 4, lastP, 0, Hd * 4)
    let pass = enc1.beginComputePass()
    lmHead(pass, lastP, 1, lg)
    pass.end()
    device.queue.submit([enc1.finish()])
    await device.queue.onSubmittedWorkDone()
    const base = await readback(lg, vocab)
    // pass 2: penalty (in place on lg) + K masked-argmax
    const enc2 = device.createCommandEncoder()
    pass = enc2.beginComputePass()
    setup(pass, 'sampler_penalty', [['u', aff.length], ['u', ban.length], ['f', penalty], ['u', 0xff800000]], [affBuf, banBuf], [lg])
    pass.dispatchWorkgroups(1)
    for (let r = 0; r < K; r++) {
      setup(pass, 'argmax_masked', [['u', vocab], ['u', r], ['u', 0], ['u', 0]], [lg], [candIds, candVals])
      pass.dispatchWorkgroups(1)
    }
    pass.end()
    device.queue.submit([enc2.finish()])
    await device.queue.onSubmittedWorkDone()
    return { base, penalized: await readback(lg, vocab), candIds: await readbackU32(candIds, K), candVals: await readback(candVals, K) }
  }

  const capabilities: EngineCapabilities = {
    useSubgroups: useSG,
    subgroupSize: sgMax,
    kvCache: kv16 ? 'f16' : kv8 ? 'q8' : 'f32',
    overflow: roll ? 'sinks' : 'error',
    maxSeqLen,
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    limits: {
      // the DEVICE limits (what dispatches are actually validated against), not the adapter's maximums
      maxStorageBufferBindingSize: Number(device.limits.maxStorageBufferBindingSize),
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
    },
  }

  // Public generate: routes to sampled decode when a sampling temperature is set, else greedy.
  // Both honor stopTokens / onToken / signal. With reuseCache, `promptTokenIds` is the DELTA to append
  // to the cached conversation (the prior turn's last token is re-fed so its K/V lands); otherwise the
  // cache resets and `promptTokenIds` is the full prompt. The KV cache lives across calls in Kc/Vc.
  async function generate(promptTokenIds: number[], genOpts: GenerateOptions = {}): Promise<GenerateResult> {
    const sampled = genOpts.temperature != null && genOpts.temperature > 0 && genOpts.temperature !== 1
    // transformers.js applies repetition_penalty / no_repeat_ngram under greedy search too, so a
    // greedy turn that requests them routes through the sampler-chain path (which picks the
    // penalized argmax instead of drawing); the plain GPU-resident greedy loop can't see history.
    const hasProcessors = (genOpts.repetitionPenalty ?? 1) !== 1 || (genOpts.noRepeatNgramSize ?? 0) > 0
    const reuse = (genOpts.reuseCache ?? false) && fullHistory.length > 0
    if (genOpts.signal?.aborted) {
      // aborted before any work: don't touch fullHistory (the cache still matches it)
      return { tokens: [], prefillMs: 0, decodeMs: 0, tokensPerSecond: 0, timing: { recordMs: 0, gpuMs: 0, readbackMs: 0 } }
    }

    // Validate BEFORE mutating fullHistory: a throw must leave the reuse state exactly as it was,
    // or a caller that catches and retries decodes against K/V that was never written.
    // Sink mode: the cache slot no longer tracks fullHistory once the window has rolled, so
    // posBase comes from cacheLen; if the incoming turn does not fit, evict FIRST (the prompt
    // itself must still fit the window - prompt-side trimming stays the caller's job).
    let posBase = reuse ? (roll ? cacheLen : fullHistory.length - 1) : 0 // reuse: the prior last token (uncached) is re-fed, then the delta
    const prefillTokens = reuse ? [fullHistory[fullHistory.length - 1], ...promptTokenIds] : promptTokenIds
    if (prefillTokens.length === 0) throw new Error('generate: no tokens to process')
    if (roll && posBase + prefillTokens.length + 1 > maxSeqLen) {
      if (SINKS + prefillTokens.length + 1 > maxSeqLen)
        throw new Error(`generate: prompt length ${prefillTokens.length} exceeds the rolling window (maxSeqLen ${maxSeqLen} minus ${SINKS} sinks); trim the prompt`)
      await ensureKvCapacity(Math.min(maxSeqLen, posBase + prefillTokens.length)) // eviction copies within allocated rows
      posBase = evict(posBase, posBase + prefillTokens.length + 1 - maxSeqLen)
      cacheLen = posBase
    }
    const room = maxSeqLen - posBase - prefillTokens.length // decode positions left in the KV window
    if (room < 1) throw new Error(`generate: prompt length ${posBase + prefillTokens.length} exceeds maxSeqLen ${maxSeqLen}; trim history or raise maxSeqLen`)
    // In sink mode the window rolls mid-turn, so maxTokens is honored as-is (no window clamp).
    const maxTokens = roll ? (genOpts.maxTokens ?? 256) : Math.min(genOpts.maxTokens ?? 256, room) // clamp instead of throwing: fill the window, stop there
    if (reuse) fullHistory.push(...promptTokenIds) // the delta is now part of the conversation (last token already present)
    else fullHistory = [...promptTokenIds]

    if (maxTokens < 1) {
      // maxTokens: 0 - behave like prefill(): write K/V for the prompt (so a later reuseCache
      // turn continues correctly), record the history above, emit nothing. Previously this
      // emitted one token anyway.
      await ensureKvCapacity(posBase + prefillTokens.length)
      transients = []
      try {
        const t0 = performance.now()
        await runPrefill(prefillTokens, posBase, genOpts.signal)
        await device.queue.onSubmittedWorkDone()
        cacheLen = posBase + prefillTokens.length - 1 // sink-mode bookkeeping (last token re-feeds)
        return { tokens: [], prefillMs: performance.now() - t0, decodeMs: 0, tokensPerSecond: 0, timing: { recordMs: 0, gpuMs: 0, readbackMs: 0 } }
      } finally {
        flushTransients()
        transients = null
      }
    }

    // A candidateFilter needs the per-step sampler-chain path (candidates on the CPU every
    // token) and cannot speculate (draft verification has no per-row filter state) - so it
    // forces that path and disables promptLookup, as documented on the option. logprobs need
    // the same per-step candidate readback, so they route identically.
    const hasFilter = !!genOpts.candidateFilter || (genOpts.logprobs ?? 0) > 0
    let r: RawGenResult
    if (!hasFilter && genOpts.promptLookup === 'auto' && maxTokens > PLD_PROBATION) {
      // Probation: speculate for the first PLD_PROBATION tokens, then keep PLD only if the
      // measured tokens-per-verify-step clears the plain-decode break-even for this mode;
      // otherwise the rest of the turn runs the plain path. Output is IDENTICAL either way:
      // the continuation re-feeds the last emitted token at its position (the composition the
      // reuse gates prove bit-exact), and a sampled continuation takes over the probation's
      // RNG mid-stream, so the draw sequence is exactly a single run's.
      const r1 = await generatePldImpl(prefillTokens, posBase, PLD_PROBATION, genOpts, fullHistory)
      const E = r1.tokens.length
      if (E < PLD_PROBATION) {
        r = r1 // stopped or aborted inside the window: nothing left to decide
      } else {
        const keep = pldWorthIt(E, r1.spec?.steps ?? 0, sampled || hasProcessors)
        const ids2 = [r1.tokens[E - 1]] // re-feed the last emitted token (its K/V is unwritten)
        const pos2 = posBase + prefillTokens.length + E - 1
        const n2 = maxTokens - E
        let r2: RawGenResult
        if (keep) {
          r2 = await generatePldImpl(ids2, pos2, n2, genOpts, fullHistory, r1.rng)
        } else if (sampled || hasProcessors) {
          r2 = await generateSampledImpl(ids2, pos2, n2, genOpts, fullHistory, r1.rng)
        } else {
          const hist = fullHistory
          r2 = await generateImpl(ids2, pos2, n2, null, SYNC_N, { stopTokens: genOpts.stopTokens, onToken: genOpts.onToken, signal: genOpts.signal })
          hist.push(...r2.tokens) // plain greedy does not touch history; the PLD/sampled impls push their own
        }
        const total = E + r2.tokens.length
        const decodeMs = r1.decodeMs + r2.prefillMs + r2.decodeMs // the 1-token re-feed is decode work
        const w1 = Math.max(1, E - 1)
        const w2 = Math.max(0, r2.tokens.length)
        r = {
          prefillMs: r1.prefillMs,
          decodeMs,
          tokPerSec: Math.max(1, total - 1) / (decodeMs / 1000),
          tokens: [...r1.tokens, ...r2.tokens],
          firstArgmax: r1.firstArgmax,
          recMs: (r1.recMs * w1 + r2.recMs * w2) / (w1 + w2),
          gpuMs: (r1.gpuMs * w1 + r2.gpuMs * w2) / (w1 + w2),
          rbMs: (r1.rbMs * w1 + r2.rbMs * w2) / (w1 + w2),
          spec: {
            steps: (r1.spec?.steps ?? 0) + (r2.spec?.steps ?? 0),
            drafted: (r1.spec?.drafted ?? 0) + (r2.spec?.drafted ?? 0),
            accepted: (r1.spec?.accepted ?? 0) + (r2.spec?.accepted ?? 0),
            bailed: !keep,
          },
        }
      }
    } else if (!hasFilter && genOpts.promptLookup) {
      r = await generatePldImpl(prefillTokens, posBase, maxTokens, genOpts, fullHistory) // pushes generated tokens onto fullHistory
    } else if (sampled || hasProcessors || hasFilter) {
      r = await generateSampledImpl(prefillTokens, posBase, maxTokens, genOpts, fullHistory) // pushes generated tokens onto fullHistory
    } else {
      // Capture the history array like the sampled/PLD paths do: a resetCache() racing this turn
      // installs a fresh array, and these tokens must land on the OLD one (then next turn falls
      // back to a clean full prefill) instead of populating the reset history against stale K/V.
      const hist = fullHistory
      r = await generateImpl(prefillTokens, posBase, maxTokens, null, SYNC_N, { stopTokens: genOpts.stopTokens, onToken: genOpts.onToken, signal: genOpts.signal })
      hist.push(...r.tokens) // greedy doesn't touch history; record the generated tokens for the next turn
    }
    return {
      tokens: r.tokens,
      prefillMs: r.prefillMs,
      decodeMs: r.decodeMs,
      tokensPerSecond: r.tokPerSec,
      timing: { recordMs: r.recMs, gpuMs: r.gpuMs, readbackMs: r.rbMs },
      ...(r.spec ? { speculation: r.spec } : {}),
      ...(r.lp ? { logprobs: r.lp } : {}),
    }
  }

  // Prefill a prompt PREFIX into the KV cache without decoding, then stop. A later
  // generate(delta, {reuseCache:true}) continues from it, so a static system prompt can be warmed at
  // load and the user's first turn becomes a cheap cache-append instead of a full prefill. Like a
  // non-reuse generate's prefill it resets the cache (this prefix becomes the whole history); it
  // writes K/V for EVERY prefilled position, so the reuse path's re-fed last token just overwrites
  // position len-1 idempotently and the result is identical to a cold full prefill.
  async function prefill(ids: number[]): Promise<{ prefillMs: number }> {
    if (ids.length === 0) throw new Error('prefill: no tokens to process')
    if (ids.length > maxSeqLen) throw new Error(`prefill: sequence length ${ids.length} exceeds maxSeqLen ${maxSeqLen}`)
    await ensureKvCapacity(ids.length)
    transients = []
    try {
      const t0 = performance.now()
      await runPrefill(ids, 0) // posBase 0: fresh prefix; writes K/V for every position (segmented)
      await device.queue.onSubmittedWorkDone()
      fullHistory = [...ids]
      cacheLen = ids.length - 1 // sink-mode bookkeeping (last token re-feeds)
      return { prefillMs: performance.now() - t0 }
    } finally {
      flushTransients()
      transients = null
    }
  }

  // ---- KV snapshot / restore ----
  // The engine's entire cross-turn state is fullHistory plus the first (fullHistory.length - 1)
  // cached positions of each layer's K/V (the last token's K/V is re-written by the reuse path's
  // re-fed token; see the fullHistory comment). A snapshot captures exactly that, packed into one
  // ArrayBuffer, so restoring it - into this engine or a fresh one on the same model + mode - is
  // bit-identical to having kept the conversation alive. Layout: per layer, K then V region
  // (then Ksc, Vsc under q8), each sized len positions; all region sizes are multiples of 4.
  const kvRowBytes = KV * Dh * KVB // one position of K (or V), per layer
  const scRowBytes = kv8 ? KV * (Dh / 32) * 4 : 0 // one position of q8 block scales
  const snapshotBytes = (len: number): number => A.layers * 2 * len * (kvRowBytes + scRowBytes)

  async function saveCache(): Promise<KvSnapshot | null> {
    if (fullHistory.length === 0) return null
    const len = roll ? cacheLen : fullHistory.length - 1 // rolled cache: slots, not history
    const data = new ArrayBuffer(snapshotBytes(len))
    if (len > 0) {
      const rb = device.createBuffer({ size: snapshotBytes(len), usage: GPUBufferUsage.MAP_READ | CD })
      const enc = device.createCommandEncoder()
      let off = 0
      for (let li = 0; li < A.layers; li++) {
        enc.copyBufferToBuffer(Kc[li], 0, rb, off, len * kvRowBytes)
        off += len * kvRowBytes
        enc.copyBufferToBuffer(Vc[li], 0, rb, off, len * kvRowBytes)
        off += len * kvRowBytes
        if (kv8) {
          enc.copyBufferToBuffer(Ksc[li], 0, rb, off, len * scRowBytes)
          off += len * scRowBytes
          enc.copyBufferToBuffer(Vsc[li], 0, rb, off, len * scRowBytes)
          off += len * scRowBytes
        }
      }
      device.queue.submit([enc.finish()])
      await rb.mapAsync(GPUMapMode.READ)
      new Uint8Array(data).set(new Uint8Array(rb.getMappedRange()))
      rb.unmap()
      rb.destroy()
    }
    return {
      version: roll ? 2 : 1, // v2 = unroped keys + rolled window; only restores into sink mode
      kvCache: capabilities.kvCache,
      model: { layers: A.layers, kvHeads: KV, headDim: Dh, hidden: Hd, vocab: A.vocab },
      ids: [...fullHistory],
      ...(roll ? { roll: { sinkTokens: SINKS, cacheLen } } : {}),
      data,
    }
  }

  async function restoreCache(snap: KvSnapshot): Promise<void> {
    if (!snap || (snap.version !== 1 && snap.version !== 2)) throw new Error(`restoreCache: unsupported snapshot version ${snap?.version}`)
    if ((snap.version === 2) !== roll)
      throw new Error(
        snap.version === 2 ?
          "restoreCache: snapshot was saved under overflow 'sinks' (unroped keys); this engine runs overflow 'error'"
        : "restoreCache: snapshot was saved under overflow 'error' (roped keys); this engine runs overflow 'sinks'",
      )
    if (snap.version === 2 && snap.roll?.sinkTokens !== SINKS)
      throw new Error(`restoreCache: snapshot uses ${snap.roll?.sinkTokens} sink tokens but this engine uses ${SINKS}`)
    if (snap.kvCache !== capabilities.kvCache)
      throw new Error(`restoreCache: snapshot was saved under kvCache '${snap.kvCache}' but this engine runs '${capabilities.kvCache}' - snapshots do not convert across modes`)
    const m = snap.model
    if (!m || m.layers !== A.layers || m.kvHeads !== KV || m.headDim !== Dh || m.hidden !== Hd || m.vocab !== A.vocab)
      throw new Error('restoreCache: snapshot is from a different model (architecture mismatch)')
    if (!Array.isArray(snap.ids) || snap.ids.length === 0) throw new Error('restoreCache: snapshot holds no tokens')
    const len = snap.version === 2 ? snap.roll!.cacheLen : snap.ids.length - 1
    if (len + 1 > maxSeqLen)
      throw new Error(`restoreCache: snapshot needs ${len + 1} cache slots but maxSeqLen is ${maxSeqLen}`)
    if (snap.data.byteLength !== snapshotBytes(len))
      throw new Error(`restoreCache: snapshot data is ${snap.data.byteLength} bytes, expected ${snapshotBytes(len)}`)
    await ensureKvCapacity(len)
    let off = 0
    for (let li = 0; li < A.layers; li++) {
      device.queue.writeBuffer(Kc[li], 0, snap.data, off, len * kvRowBytes)
      off += len * kvRowBytes
      device.queue.writeBuffer(Vc[li], 0, snap.data, off, len * kvRowBytes)
      off += len * kvRowBytes
      if (kv8) {
        device.queue.writeBuffer(Ksc[li], 0, snap.data, off, len * scRowBytes)
        off += len * scRowBytes
        device.queue.writeBuffer(Vsc[li], 0, snap.data, off, len * scRowBytes)
        off += len * scRowBytes
      }
    }
    fullHistory = [...snap.ids]
    cacheLen = len
  }

  // The engine shares one KV cache, buffer pool, and flag set across calls, so concurrent
  // generate/prefill/forward would corrupt each other. Serialize them: overlapping calls queue
  // instead of interleaving (npm consumers do not all have the app's single-flight worker).
  let opChain: Promise<unknown> = Promise.resolve()
  const serialize = <Args extends unknown[], R>(fn: (...args: Args) => Promise<R>): ((...args: Args) => Promise<R>) => {
    return (...args: Args) => {
      const run = opChain.then(
        () => fn(...args),
        () => fn(...args), // a failed predecessor must not poison the queue
      )
      opChain = run.catch(() => undefined)
      return run
    }
  }

  const api: EngineInternal = {
    generate: serialize(generate),
    prefill: serialize(prefill),
    forward: serialize(forward),
    saveCache: serialize(saveCache),
    restoreCache: serialize(restoreCache),
    resetCache,
    capabilities,
    lost,
    dispose: () => device.destroy(),
    device,
    adapter,
    profileDecode: (ids, nTokens, full = null, syncN = SYNC_N) => generateImpl(ids, 0, nTokens, full, syncN),
    debugDecode,
    debugSampler,
  }
  return api
}
