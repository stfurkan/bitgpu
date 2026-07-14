// Public types for bitgpu.

/** Progress event emitted while a model loads. */
export interface LoadProgress {
  phase: 'manifest' | 'weights' | 'pipelines'
  /** Bytes fetched so far (weights phase only). */
  loaded?: number
  /** Total bytes to fetch (weights phase only). */
  total?: number
}

/** Options for {@link createEngine}. Pass a string to use defaults with just a model URL.
 *  Provide either `modelUrl` (a directory holding manifest.json + the data/aux files) or the explicit
 *  `manifestUrl`/`dataUrl`/`auxUrl` (which lets the large data file come from one host, e.g. the HF Hub,
 *  and the small manifest/aux from another). The explicit URLs win over the `modelUrl`-relative ones. */
export interface EngineOptions {
  /** Base URL of the model directory containing `manifest.json` and its data/aux files. */
  modelUrl?: string
  /** Explicit URL for manifest.json (overrides `${modelUrl}/manifest.json`). */
  manifestUrl?: string
  /** Explicit URL for the weights data file (overrides `${modelUrl}/<data_file>`). */
  dataUrl?: string
  /** Explicit URL for the aux file (overrides `${modelUrl}/<aux_file>`). */
  auxUrl?: string
  /** Fetch JSON (manifest). Override to add caching. Default: `fetch(url).json()`. */
  fetchJson?: (url: string) => Promise<unknown>
  /** Fetch binary (data/aux). Override to add caching (e.g. OPFS) for the ~290MB data file. Default: `fetch(url).arrayBuffer()`. */
  fetchArrayBuffer?: (url: string) => Promise<ArrayBuffer>
  /** Stream the big weights DATA file: chunks flow straight into GPU buffers, so the whole
   *  ~290MB file never sits in memory at once (critical on phones, where the buffered peak gets
   *  the tab killed). Used for the data file only (manifest/aux stay on fetchJson/fetchArrayBuffer);
   *  when set it takes precedence over fetchArrayBuffer for that file. Point it at a cache stream,
   *  e.g. an OPFS file's `stream()`. Default: the data file streams from `fetch(url).body`
   *  (or through `fetchArrayBuffer` when that override is provided, for back compatibility). */
  fetchStream?: (url: string) => Promise<ReadableStream<Uint8Array>>
  /** GPU power preference. Default `'high-performance'` (picks the discrete GPU on multi-GPU machines).
   *  (Spelled out rather than `GPUPowerPreference` so the published d.ts resolves without @webgpu/types.) */
  powerPreference?: 'low-power' | 'high-performance'
  /** Force the no-subgroup reduction path (for testing the fallback). Default `false`. */
  forceNoSubgroups?: boolean
  /** Workgroup size for the no-subgroup reduction kernels. Default `64`. */
  noSubgroupWorkgroupSize?: number
  /** Decode steps chained per CPU sync (deferred readback). Higher hides latency; default `4`. */
  syncSteps?: number
  /** Prefill GEMM tiling: `'auto'` tiles once a prompt fills the 64-row tiles, `'always'`/`'never'` force it. Default `'auto'`. */
  prefillTiling?: 'auto' | 'always' | 'never'
  /** Max KV-cache length (prompt + generated positions). Caps VRAM (~`maxSeqLen` x 224 KB at f32,
   *  half that with `kvCache: 'f16'`, ~a quarter with `'q8'`). Default `2048`. */
  maxSeqLen?: number
  /** KV-cache STORAGE precision. `'f16'` halves KV memory (all arithmetic stays f32; each cached
   *  K/V value is rounded once at cache-write); it falls back to `'f32'` silently when the
   *  adapter lacks `shader-f16`. `'q8'` QUARTERS KV memory vs f32 (packed 8-bit values with one
   *  f32 scale per 32-element block, llama.cpp q8_0-style, ~1.125 bytes/value) and needs no
   *  adapter feature, so it works even where f16 does not (and at long context it decodes FASTER
   *  than f32 - attention there is memory-bound and reads a quarter of the bytes). Outputs under
   *  f16/q8 are no longer
   *  guaranteed bit-identical to f32 mode - q8 is the first knowingly-lossy tier, measured per
   *  model by the GPU gate - though WITHIN a mode decoding stays exact and deterministic (same
   *  seed -> same tokens; cache reuse == full prefill). See `capabilities.kvCache` for what's
   *  active. Default `'f32'`. */
  kvCache?: 'f32' | 'f16' | 'q8'
  /** What happens when a conversation outgrows `maxSeqLen`. `'error'` (default): generate
   *  throws, exactly as before. `'sinks'`: StreamingLLM-style rolling window - the first
   *  `sinkTokens` positions (attention sinks) plus the most recent window are kept and the
   *  middle is evicted in batches, so generation and multi-turn chat continue indefinitely in
   *  fixed memory. Keys are cached UNROPED and rotated at read by cache-relative position, so
   *  eviction never rewrites or (under q8) requantizes cache bytes. Sink mode is its own
   *  measured mode like f16/q8: within-mode decoding stays exact and deterministic, and before
   *  the first eviction f32+sinks matches default f32 (gated); after eviction the model
   *  genuinely forgets evicted middle context - that is the trade. Prompts longer than the
   *  window still throw (trim prompt-side, e.g. chat onOverflow). */
  overflow?: 'error' | 'sinks'
  /** Number of initial attention-sink positions kept forever under `overflow: 'sinks'`.
   *  Default `4` (the StreamingLLM setting). */
  sinkTokens?: number
  /** Called as the model loads. */
  onProgress?: (progress: LoadProgress) => void
  /** Called if the GPU device is lost after creation (driver reset, OS reclaim, tab backgrounding
   *  on some platforms). The engine is unusable afterward; create a new one to recover. Not called
   *  for the intentional loss caused by {@link Engine.dispose}. */
  onDeviceLost?: (info: DeviceLostInfo) => void
}

/** Why the GPU device went away. Mirrors GPUDeviceLostInfo without requiring @webgpu/types. */
export interface DeviceLostInfo {
  /** 'destroyed' when {@link Engine.dispose} caused it, otherwise the platform reason ('unknown', ...). */
  reason: string
  message: string
}

/** Options for a single {@link Engine.generate} call. Set `temperature` to a value other than 0 or 1
 *  to sample; otherwise decoding is greedy (argmax). Sampling matches transformers.js v4.2.0 exactly
 *  (repetition_penalty, no_repeat_ngram, temperature, top_k, multinomial via a Mersenne-Twister RNG). */
export interface GenerateOptions {
  /** Maximum number of new tokens to generate. Default `256`. `0` prefills the prompt into the
   *  KV cache and emits nothing (like {@link Engine.prefill}, but composable with `reuseCache`). */
  maxTokens?: number
  /** Reuse the KV cache from the previous turn: treat the passed token ids as the DELTA to append to
   *  the cached conversation (not a full prompt), prefilling only those tokens. Requires a prior
   *  generate on this engine (else it falls back to a full prefill). Default `false` (reset + full prefill). */
  reuseCache?: boolean
  /** Token ids that end generation when produced (e.g. EOS). The stop token is not emitted. */
  stopTokens?: number[]
  /** Called with each generated token id as it is produced (per-token when sampling). */
  onToken?: (tokenId: number) => void
  /** Aborts generation when signaled (checked at each step). */
  signal?: AbortSignal
  /** Softmax temperature. A value other than 0 or 1 enables sampling; 0 or 1 (or unset) is greedy. */
  temperature?: number
  /** Top-k sampling cutoff (candidate count). Default `20` when sampling; clamped to `[1, vocab]`
   *  (the GPU reduces the logits to this many candidates, so `0` cannot mean "disabled"). */
  topK?: number
  /** Top-p (nucleus) cutoff. Accepted for API compatibility but not applied (a no-op, matching transformers.js v4.2.0). */
  topP?: number
  /** Repetition penalty over the deduped prompt+generated id set (`logit<0 ? *p : /p`). Default `1`
   *  (off). Applied under greedy decoding too (the penalized argmax), matching transformers.js. */
  repetitionPenalty?: number
  /** Block any n-gram of this size from repeating. Default `0` (off). Applied under greedy decoding
   *  too, matching transformers.js. */
  noRepeatNgramSize?: number
  /** Seed for the sampler RNG. Omit to seed from entropy (non-deterministic, like production). */
  seed?: number
  /** EXPERIMENTAL. Per-step constrained-decoding hook: receives the top-K candidate token ids
   *  (rank order, penalized logits alongside) and returns the PERMITTED subset (any order).
   *  Greedy picks the best permitted candidate; sampling renormalizes the draw over them. When
   *  it permits none, the engine walks the full vocabulary in logit order (rare; the step costs
   *  a full logits readback) and throws if no token at all is permitted. Must be deterministic
   *  and synchronous; the engine calls it once per emitted token, in order. Incompatible with
   *  `promptLookup` (speculation is disabled while a filter is set). Used by bitgpu/chat's
   *  `format: 'json'`. */
  candidateFilter?: (candidateIds: Uint32Array, candidateLogits: Float32Array) => number[]
  /** EXPERIMENTAL. Prompt-lookup speculative decoding: draft the continuation from an n-gram
   *  match in the sequence so far and verify every draft in ONE batched forward. Output is
   *  identical to normal decoding, greedy AND sampled (each emitted token still comes from its
   *  true penalized distribution, and the RNG advances one draw per emitted token, in order).
   *  `'auto'` speculates for a short probation window, then keeps PLD only when the measured
   *  acceptance actually beats plain decoding for this content - and falls back to the plain
   *  path otherwise (output still identical; `speculation.bailed` reports the decision). Use
   *  `'auto'` unless you know the content repeats (quoting, lists, code), where `true` skips
   *  the probation. INCOMPATIBLE with `noRepeatNgramSize`: a prompt-lookup draft is the
   *  continuation of a repeated n-gram, which is exactly what the ngram ban forbids, so
   *  acceptance is ~zero when both are on (measured 0/320). No draft model, no extra VRAM.
   *  `true`/`'auto'` = `{ ngramSize: 3, maxDraft: 8 }`. Default `false`. */
  promptLookup?: boolean | 'auto' | { ngramSize?: number; maxDraft?: number }
  /** Custom speculative-decoding drafter: called once per verify step with the conversation so
   *  far, returns up to `k` proposed next-token ids (empty = decode one token normally this
   *  step). Proposals feed the SAME verify machinery as `promptLookup`, so output stays
   *  BIT-IDENTICAL to non-speculative decoding no matter how bad the drafts are - a wrong draft
   *  only costs speed. May be async (e.g. a second, smaller engine drafting for this one;
   *  pair with {@link Engine.rewind} to roll the draft engine back to the accepted prefix).
   *  Overrides `promptLookup`; ignored (like it) when `candidateFilter` or `logprobs` is set.
   *  Cap: `promptLookup.maxDraft` (default 8) proposals per step. */
  drafter?: (ctx: { history: number[]; k: number }) => number[] | Promise<number[]>
  /** Per-token TRUE logprobs (log-softmax over the full vocabulary, after penalties): set to N
   *  (1..32) to receive the emitted token's logprob plus the top-N alternatives each step in
   *  {@link GenerateResult.logprobs}. Exact, not a top-K approximation - the normalizer is a GPU
   *  log-sum-exp over all logits (one extra f32 readback per step). Routes greedy turns through
   *  the per-step sampler path and disables `promptLookup` (a verified draft step has no
   *  per-token candidate readback); sampled turns pay nothing extra. Use it to expose model
   *  confidence: a low top-1 logprob or a flat top-N is the model guessing. Default off. */
  logprobs?: number
}

/** One emitted token's logprob record (see {@link GenerateOptions.logprobs}). */
export interface TokenLogprobs {
  /** The emitted token's logprob (log-softmax over the full vocab, post-penalty). Under a
   *  candidateFilter this is the logprob of the token actually chosen, which the filter may
   *  have forced far below the top alternatives. */
  logprob: number
  /** The top-N (id, logprob) pairs in descending order, independent of what was emitted. */
  top: { id: number; logprob: number }[]
}

/** Result of a {@link Engine.generate} call. */
export interface GenerateResult {
  /** Generated token ids (excludes the prompt). */
  tokens: number[]
  /** Time to first token (prefill of the prompt), in milliseconds. */
  prefillMs: number
  /** Decode time for the remaining tokens, in milliseconds. */
  decodeMs: number
  /** Decode throughput (tokens / second), excluding prefill. */
  tokensPerSecond: number
  /** Per-token decode timing breakdown, in milliseconds. In sampled mode the GPU wait and the
   *  readback map share one sync, so the wait is attributed to `gpuMs` and `readbackMs` covers
   *  only the post-map CPU work (near zero). */
  timing: { recordMs: number; gpuMs: number; readbackMs: number }
  /** Present when prompt-lookup decoding ran: verify steps taken, tokens drafted, drafts
   *  accepted. With `promptLookup: 'auto'`, `bailed` reports whether the probation decided to
   *  stop speculating for the rest of the turn. */
  speculation?: { steps: number; drafted: number; accepted: number; bailed?: boolean }
  /** Present when {@link GenerateOptions.logprobs} was set: one record per emitted token,
   *  aligned with `tokens`. */
  logprobs?: TokenLogprobs[]
}

/** Diagnostic result of {@link Engine.forward}: hidden states + logits for a single forward pass. */
export interface ForwardResult {
  embed: Float32Array
  layer0: Float32Array
  finalnorm: Float32Array
  logits: Float32Array
  vocab: number
  sequenceLength: number
}

/** What the engine detected about the host GPU and which code path it selected. */
export interface EngineCapabilities {
  /** Whether the fast subgroup path is in use (false = the workgroup-reduction fallback). */
  useSubgroups: boolean
  /** Subgroup width when the subgroup path is active. */
  subgroupSize: number
  /** Active KV-cache storage precision ('f16' only when requested AND the adapter has shader-f16;
   *  'q8' whenever requested - it needs no adapter feature). */
  kvCache: 'f32' | 'f16' | 'q8'
  /** Active overflow policy ('sinks' = rolling window with attention sinks). */
  overflow: 'error' | 'sinks'
  /** The engine's KV window in positions (the resolved maxSeqLen option). Under
   *  `overflow: 'error'` prompt + generated tokens must fit inside it; under `'sinks'` it is
   *  the rolling window size. */
  maxSeqLen: number
  /** Adapter identification, when the browser exposes it. */
  adapter: { vendor?: string; architecture?: string; device?: string; description?: string }
  /** Relevant adapter limits the engine codes against. */
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupStorageSize: number }
}

/** A saved KV cache + conversation token history, from {@link Engine.saveCache}. A plain object
 *  holding one `ArrayBuffer`, so it is structured-cloneable: store it in IndexedDB / OPFS or send
 *  it over `postMessage` as-is (it is NOT `JSON.stringify`-able - the buffer would be lost).
 *  Restore requires the SAME model architecture and the SAME `kvCache` mode; snapshots never
 *  convert across modes. Treat the fields as opaque. */
export interface KvSnapshot {
  /** Snapshot format version: `1` = default mode, `2` = saved under `overflow: 'sinks'`
   *  (the cache holds unroped keys and a rolled window; restore requires sink mode). */
  version: 1 | 2
  /** KV storage mode the snapshot was taken under. */
  kvCache: 'f32' | 'f16' | 'q8'
  /** Version 2 only: rolling-window state (sink count and filled cache slots). */
  roll?: { sinkTokens: number; cacheLen: number }
  /** Architecture stamp; restore rejects a snapshot from a different model shape. */
  model: { layers: number; kvHeads: number; headDim: number; hidden: number; vocab: number }
  /** The full conversation token sequence at save time. */
  ids: number[]
  /** Packed per-layer cached K/V bytes (and q8 block scales). */
  data: ArrayBuffer
}

/** A loaded model ready to generate. Create one with {@link createEngine}. */
export interface Engine {
  /** Generate tokens from a prompt given as token ids. */
  generate(promptTokenIds: number[], options?: GenerateOptions): Promise<GenerateResult>
  /** Prefill a prompt PREFIX into the KV cache without decoding, so a later
   *  `generate(delta, { reuseCache: true })` continues from it. Use to warm a static system prompt
   *  at load time so the first real turn is a cheap cache-append, not a full prefill. Resets any
   *  existing cache (the prefix becomes the whole history). Returns the prefill wall-time. */
  prefill(promptTokenIds: number[]): Promise<{ prefillMs: number }>
  /** Run a single forward pass and return hidden states + logits (diagnostic / correctness checks). */
  forward(tokenIds: number[]): Promise<ForwardResult>
  /** Clear the cross-turn KV cache and token history (start a fresh conversation). */
  resetCache(): void
  /** Drop the last `n` conversation tokens (cheap, CPU-side). Built for app-side two-model
   *  speculation: a DRAFT engine generates ahead, and after the target rejects some suffix the
   *  draft engine rewinds to the accepted prefix instead of re-prefilling. Throws under
   *  `overflow: 'sinks'` (eviction breaks the token-to-slot mapping) and when `n` would empty
   *  the history (use `resetCache`). */
  rewind(n: number): void
  /** Snapshot the current conversation - KV cache contents + token history - as a
   *  structured-cloneable {@link KvSnapshot} (GPU -> CPU readback). Restoring it, into this
   *  engine or a fresh one on the same model and `kvCache` mode, is bit-identical to having kept
   *  the conversation alive, so conversations survive engine disposal and page reloads (persist
   *  the snapshot in IndexedDB / OPFS). Size ~ tokens x layers x kvHeads x headDim x 2 x
   *  bytes-per-value (~224 KB/token at f32 on Bonsai-1.7B, ~63 KB/token at q8). Returns `null`
   *  when the cache is empty. Serialized with generate/prefill/forward like every engine op. */
  saveCache(): Promise<KvSnapshot | null>
  /** Replace the current conversation with a {@link KvSnapshot} (CPU -> GPU upload). The next
   *  `generate(delta, { reuseCache: true })` continues the saved conversation exactly. Throws if
   *  the snapshot's model architecture or `kvCache` mode does not match this engine, or if it
   *  exceeds this engine's `maxSeqLen`. */
  restoreCache(snapshot: KvSnapshot): Promise<void>
  /** Detected GPU capabilities and selected code path. */
  readonly capabilities: EngineCapabilities
  /** Resolves when the GPU device is lost (including via {@link dispose}, with reason 'destroyed').
   *  After an unexpected loss the engine is dead; create a new one to recover. */
  readonly lost: Promise<DeviceLostInfo>
  /** Release GPU resources. The engine is unusable afterward. */
  dispose(): void
}
