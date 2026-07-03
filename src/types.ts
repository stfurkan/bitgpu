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
  /** Max KV-cache length (prompt + generated positions). Caps VRAM (~`maxSeqLen` x 224 KB). Default `2048`. */
  maxSeqLen?: number
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
  /** Maximum number of new tokens to generate. Default `256`. */
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
  /** Top-k sampling cutoff (candidate count). Default `20` when sampling. */
  topK?: number
  /** Top-p (nucleus) cutoff. Accepted for API compatibility but not applied (a no-op, matching transformers.js v4.2.0). */
  topP?: number
  /** Repetition penalty over the deduped prompt+generated id set (`logit<0 ? *p : /p`). Default `1` (off). */
  repetitionPenalty?: number
  /** Block any n-gram of this size from repeating. Default `0` (off). */
  noRepeatNgramSize?: number
  /** Seed for the sampler RNG. Omit to seed from entropy (non-deterministic, like production). */
  seed?: number
  /** EXPERIMENTAL. Prompt-lookup speculative decoding: draft the continuation from an n-gram
   *  match in the sequence so far and verify every draft in ONE batched forward. Output is
   *  identical to normal decoding, greedy AND sampled (each emitted token still comes from its
   *  true penalized distribution, and the RNG advances one draw per emitted token, in order).
   *  INCOMPATIBLE with `noRepeatNgramSize`: a prompt-lookup draft is the continuation of a
   *  repeated n-gram, which is exactly what the ngram ban forbids, so acceptance is ~zero when
   *  both are on (measured 0/320). Use with greedy decoding or ngram-ban-free sampling. No
   *  draft model, no extra VRAM. `true` = `{ ngramSize: 3, maxDraft: 8 }`. Default `false`. */
  promptLookup?: boolean | { ngramSize?: number; maxDraft?: number }
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
  /** Per-token decode timing breakdown, in milliseconds. */
  timing: { recordMs: number; gpuMs: number; readbackMs: number }
  /** Present when prompt-lookup decoding ran: verify steps taken, tokens drafted, drafts accepted. */
  speculation?: { steps: number; drafted: number; accepted: number }
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
  /** Adapter identification, when the browser exposes it. */
  adapter: { vendor?: string; architecture?: string; device?: string; description?: string }
  /** Relevant adapter limits the engine codes against. */
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupStorageSize: number }
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
  /** Detected GPU capabilities and selected code path. */
  readonly capabilities: EngineCapabilities
  /** Resolves when the GPU device is lost (including via {@link dispose}, with reason 'destroyed').
   *  After an unexpected loss the engine is dead; create a new one to recover. */
  readonly lost: Promise<DeviceLostInfo>
  /** Release GPU resources. The engine is unusable afterward. */
  dispose(): void
}
