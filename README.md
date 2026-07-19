# bitgpu

A fast, dependency-free WebGPU runtime for **low-bit LLMs** in the browser.

Today it runs **1-bit (binary-weight)** models.
Reference targets are Bonsai **1.7B, 4B and 8B** (Qwen3 architecture, sign-packed binary linear
weights + 2/4-bit embeddings, tied or untied lm_head) - every size is gated bit-exact against the
reference forward on real hardware - plus **Bonsai-27B** (Qwen3.5 *hybrid* backbone: a 3:1 mix of
gated-DeltaNet linear attention and gated full attention), validated to matching greedy tokens and
logits tolerance against transformers (its linear-attention recurrence keeps no fp64 path, so the
bar is greedy-exact rather than bit-exact). GPU-resident decode (greedy or sampled), streaming, EOS stop,
`AbortSignal`, cross-turn KV-cache reuse, optional f16/q8 KV-cache compression for long
contexts in less VRAM, conversation snapshots (save/restore across page reloads), and an
optional rolling window with attention sinks for unbounded chats in fixed memory. Runs the
fast subgroup path on Apple / NVIDIA /
recent AMD and falls back to a workgroup-reduction path everywhere else WebGPU is available.
Device limits are negotiated from the manifest, so the 8B's ~148 MiB lm_head binding is requested
only when that model needs it and smaller models keep running at WebGPU's guaranteed minimums.

**[DEMO](https://stfurkan.github.io/bitgpu/examples/chat.html)**: pick a model, let the weights
stream once from the Hugging Face Hub, and chat with it on your own GPU.

## Install

```sh
npm install bitgpu
```

ESM-only, zero runtime dependencies.

## Benchmark

Against the mainstream WebGPU path for the same model - transformers.js with `dtype: 'q1'` - on the
**identical** 1-bit Bonsai-1.7B weights, same GPU, same prompt, both in one page:

| | transformers.js 4.2.0 (`dtype: 'q1'`) | bitgpu |
| --- | --- | --- |
| decode | 17.7 tok/s | **23.8 tok/s** (up to ~1.8x at short context) |
| prefill (156-token prompt) | 5.6 s | **0.9 s** (~6x) |

Point-in-time, one machine (Apple M-series, Chrome 150 / WebGPU, 2026-07) - so treat it as a
ballpark, not a leaderboard. Reproduce it, or run your own model / GPU, with `npm run bench`
([examples/benchmark.html](examples/benchmark.html) loads both engines side by side and prints the
numbers). Weights are bit-identical across the two containers (the GGUF sign-bit stream equals the
onnx-community q1 export), so this measures the runtime, not the model.

## Quickstart - no conversion, no hosting

Ready-made manifests for all three Bonsai sizes are committed under [`models/`](models/); the
weights stream straight from the Hugging Face Hub. This runs as-is:

```ts
import { createEngine } from 'bitgpu'
import { createChat } from 'bitgpu/chat'

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@v0.15.0/models/bonsai-1.7b-gguf'
const TOK = 'https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main'
const engine = await createEngine({
  manifestUrl: `${REPO}/manifest.json`,
  auxUrl: `${REPO}/Bonsai-1.7B-Q1_0.aux.bin`,
  dataUrl: 'https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf',
})
const chat = await createChat(engine, {
  tokenizerJsonUrl: `${TOK}/tokenizer.json`, // tokenizer from the ONNX repo (same base model)
  tokenizerConfigUrl: `${TOK}/tokenizer_config.json`,
})
await chat.send([{ role: 'user', content: 'Hi!' }], { onText: (t) => process.stdout.write(t) })
```

The [demo](https://stfurkan.github.io/bitgpu/examples/chat.html) ([source](examples/chat.html))
is this quickstart as a single HTML file: model picker, streaming chat, JSON mode with a schema
editor, tool calling, cached downloads, and sessions that survive reloads (the KV cache is
snapshot into IndexedDB and restored - the conversation continues with no re-prefill).

## Usage

```ts
import { createEngine, WebGPUUnavailableError } from 'bitgpu'

let engine
try {
  engine = await createEngine({
    modelUrl: 'https://cdn.example.com/bonsai', // dir with manifest.json + data/aux files
    onProgress: (p) => console.log(p.phase),
  })
} catch (err) {
  if (err instanceof WebGPUUnavailableError) {
    // render a "WebGPU not supported" fallback
  } else throw err
}

// Greedy by default; stream tokens, stop on EOS, cancel with a signal.
const result = await engine.generate(promptTokenIds, {
  maxTokens: 256,
  stopTokens: [151645],
  onToken: (id) => process.stdout.write(String(id) + ' '),
})
console.log(result.tokens, result.tokensPerSecond)

// Sampling (matches transformers.js v4.2.0 exactly): set a temperature other than 0/1.
await engine.generate(promptTokenIds, { temperature: 0.5, topK: 20, repetitionPenalty: 1.15 })

// Penalties apply under greedy decoding too (penalized argmax, deterministic, no RNG),
// exactly like transformers.js greedy search:
await engine.generate(promptTokenIds, { repetitionPenalty: 1.15, noRepeatNgramSize: 3 })

engine.dispose()
```

### KV cache modes (`kvCache`)

The KV cache is what grows with conversation length (~224 KB per position at f32 on
Bonsai-1.7B; `maxSeqLen` caps it). `kvCache` selects its **storage** precision - attention
arithmetic always stays f32, values are compressed once at cache-write:

| Mode | Bytes/value | KV memory | Requires |
| --- | --- | --- | --- |
| `'f32'` (default) | 4 | 1x | - |
| `'f16'` | 2 | 1/2 | `shader-f16` (silently falls back to f32 without it) |
| `'q8'` | ~1.125 | ~1/4 | nothing - works on every WebGPU adapter |

```ts
const engine = await createEngine({ modelUrl, kvCache: 'q8', maxSeqLen: 4096 })
```

`'q8'` stores 8-bit values with one f32 scale per 32-element block (llama.cpp q8_0-style, the
tier the wider ecosystem treats as near-lossless). Within any mode decoding stays exact and
deterministic (same seed -> same tokens, cache reuse == full prefill), but f16/q8 outputs are
**not guaranteed bit-identical to f32** - they are measured instead: the GPU gate compares
greedy continuations against f32 on every Bonsai size and both kernel paths, and q8 currently
agrees 96/96 tokens on short prompts, 24/24 after a 400-token prompt and 48/48 after a
1500-token prompt (logits cosine >= 0.99997). Long contexts also get *faster* under q8 - decode
at 1500 tokens of depth measured 37-58% quicker than f32 (attention there is memory-bound and
reads a quarter of the bytes) - while shallow-context decode measures a few percent slower.
Reach for `'q8'` to run a 4x longer window in the same VRAM (or the same window on smaller
GPUs); keep the default `'f32'` when bit-exact reproducibility is the point.
`engine.capabilities.kvCache` reports what is actually active.

### Unbounded conversations (`overflow: 'sinks'`)

By default a conversation that outgrows `maxSeqLen` throws. `overflow: 'sinks'` switches to a
StreamingLLM-style rolling window instead: the first `sinkTokens` positions (default 4, the
"attention sinks" that anchor the model's attention) plus the most recent window are kept, the
middle is evicted in batches, and generation or multi-turn chat continues indefinitely in fixed
memory - `maxTokens` is no longer clamped to the window:

```ts
const engine = await createEngine({ modelUrl, kvCache: 'q8', overflow: 'sinks', maxSeqLen: 4096 })
```

Implementation detail that matters: keys are cached **unroped** and rotated at read time by
their cache-relative position (the StreamingLLM reference scheme), so eviction is a plain
byte-exact compaction - nothing is ever re-rotated or, under q8, requantized. This is the
combination llama.cpp cannot offer (its context shift rewrites roped K in place and is
disabled for quantized K caches); here `q8 + sinks` is the flagship pairing: unbounded chat in
a quarter of the memory. The guarantees follow the mode philosophy: within-mode decoding is
exact and deterministic (same seed -> same tokens, across evictions; gated on real hardware),
and **before the first eviction f32+sinks is bit-identical to default f32** (gated against the
known-good ids). After an eviction the model genuinely forgets the evicted middle - that is
the trade, by design: prefer it over hard failure for open-ended chat; prefer `'error'` plus
prompt-side trimming when every token of context must count. Prompts longer than the window
still throw (`bitgpu/chat`'s `onOverflow` handles that side). Snapshots keep working - sink
mode saves version-2 snapshots (restore requires sink mode with the same `sinkTokens`).

### Speculative decoding: why only `promptLookup`

The engine speculates out of the box where it pays: `promptLookup` drafts from n-gram matches
in the prompt and verifies them in one batched pass, with auto-probation that measures the
break-even and bails when speculation does not help (output is bit-identical either way).

There is deliberately no draft-model / two-model mode, and the reason is measured, not
guessed: speculative decoding pays only when verifying k tokens in one batched pass costs
about one token's time - true for f16 models, whose decode is weight-bandwidth-bound, but not
for 1-bit models. The weight stream here is so small that batch-1 decode already saturates GPU
compute, so a k-row verify costs ~k single steps (measured: S=9 verify ~= 9.8x one step, and
even a perfect zero-cost drafter decoded slower than plain on the subgroup path). A full
in-engine two-model orchestration was built, gate-proven bit-exact, measured slower, and
removed - fast single-stream 1-bit decode and profitable speculation are the same budget,
already spent on the former. Where batching still wins (dispatch-latency-bound setups like the
no-subgroup fallback path) is exactly where `promptLookup`'s probation keeps it on.

## Chat (`bitgpu/chat`)

The engine is deliberately ids-in/ids-out; `bitgpu/chat` is the batteries-included text layer on
top of it - messages in, streamed text out, still entirely on-device:

```ts
import { createEngine } from 'bitgpu'
import { createChat } from 'bitgpu/chat'

const engine = await createEngine({ modelUrl })
const chat = await createChat(engine, { modelUrl }) // tokenizer files live next to the manifest

// Callback streaming:
const r = await chat.send(
  [{ role: 'user', content: 'Explain WebGPU in one sentence.' }],
  { onText: (delta) => ui.append(delta) },
)

// ...or async-generator streaming (the final result is the generator's return value):
const it = chat.stream(messages, { temperature: 0.5, topK: 20 })
for (let n = await it.next(); !n.done; n = await it.next()) ui.append(n.value)
```

It owns the whole pipeline the engine leaves to the caller: the model's own Jinja chat template,
tokenization, UTF-8-safe incremental decode streaming, `<think>` block routing (`think: true`
streams reasoning to `onThink`, never into the reply), EOS handling, and cross-turn KV-cache
reuse with exact token bookkeeping (a clean follow-up turn prefills only the delta;
`chat.prewarm(messages)` warms a static system prompt at load). `chat.reset()` forgets the
conversation.

### Guaranteed-valid JSON (`format: 'json'`)

```ts
const r = await chat.send(
  [{ role: 'user', content: 'Describe France as JSON: capital (string), population_millions (number).' }],
  { format: 'json' },
)
JSON.parse(r.text) // never throws when finishReason === 'stop'
```

Constrained decoding: every candidate token is validated against an incremental byte-level JSON
machine before it can be sampled, so the reply is structurally guaranteed to be one complete,
valid JSON value (object or array root) - small 1-bit models free-form JSON unreliably, and this
removes that failure class entirely. Generation ends when the root value closes
(`finishReason: 'length'` means `maxTokens` cut it short - raise it).

Pass a **schema** and the shape is enforced token-by-token too - the model cannot open an object
where an array is required, stop at 1 item when 5 are demanded, invent keys, drift a type, or
answer outside an enum:

```ts
await chat.send(messages, { format: { json: { schema: {
  type: 'array', minItems: 5, maxItems: 5,
  items: { type: 'object', required: ['name', 'population'], additionalProperties: false,
           properties: { name: { type: 'string' }, population: { type: 'number' } } },
} } } })
// or guaranteed classification:  { properties: { mood: { enum: ['positive', 'negative', 'neutral'] } } }
```

Enforceable subset: `type` (incl. `integer`), `properties` / `required` /
`additionalProperties: false`, `items`, `minItems` / `maxItems`, string `enum`,
`minLength` / `maxLength` (code points), integer `minimum` / `maximum` (with prefix
feasibility, so the model can never be trapped mid-number), and `oneOf` as a **discriminated
union** - object branches sharing one required property whose single-value `enum` differs per
branch, e.g. a slide that is either `{type: 'bullets', ...}` or `{type: 'quote', ...}`; the
machine tracks the live branches until the discriminator commits. All nested to any depth.
Annotation-only keywords - `description`, `title`, `default`, `examples`, `$schema`, `$id`,
`deprecated`, `readOnly`, `writeOnly`, `$comment` - are **accepted and ignored** (they carry no
constraint), so a real MCP / OpenAI tool schema passes through unmodified and a property's
`description` still reaches the model via the chat template. Genuinely-constraining keywords the
enforcer does not implement (`pattern`, `format`, float ranges, general `oneOf`, `$ref`, ...)
**throw up front** - never silently ignored, because those would *look* enforced. The guarantee
is structural, not semantic: a schema makes the output parse into the right shape, not be true.
Built on the engine's generic `candidateFilter` hook (see `GenerateOptions`), which is open for
custom grammars.

### Confidence (`logprobs`)

Pass `logprobs: N` (engine or chat options) and every emitted token comes back with its TRUE
logprob plus the top-N alternatives - log-softmax over the full vocabulary, computed exactly via
a GPU log-sum-exp (one extra f32 readback per step, not a top-K approximation):

```ts
const r = await chat.send(messages, { logprobs: 5 })
const confidence = Math.exp(r.logprobs[0].logprob) // p of the first token; low = the model is guessing
```

Use it to flag low-confidence answers, build "are you sure?" UX, or detect when a
schema/tool filter had to force a token the model ranked poorly. Greedy output is bit-identical
with or without it; `promptLookup` is disabled for the turn.

### Tool calling (`tools`)

The model's own protocol (Qwen3-family templates render a `tools` list and emit
`<tool_call>` blocks), with the same enforcement guarantee as schema mode: a bitgpu tool call
**cannot be malformed** - once the model opens a call, the name is forced to one of your
declared tools and the arguments are forced through that tool's `parameters` schema,
token-by-token.

```ts
const tools = [{
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a city.',
    parameters: { type: 'object', required: ['city'], additionalProperties: false,
                  properties: { city: { type: 'string' }, unit: { enum: ['celsius', 'fahrenheit'] } } },
  },
}]

const r = await chat.send(messages, { tools })
if (r.finishReason === 'tool_calls') {
  const call = r.toolCalls[0]                       // { name, arguments } - always valid
  const result = await runMyTool(call)              // executing is YOUR code, on YOUR terms
  const r2 = await chat.send([
    ...messages,
    { role: 'assistant', content: r.text, tool_calls: r.toolCalls },
    { role: 'tool', content: JSON.stringify(result) },
  ], { tools })                                     // extends the KV cache - no re-prefill
}
```

`toolChoice: { name: 'get_weather' }` **forces** a call to that tool as the entire reply - fully
enforced end to end, and the reliable way to use tools with small models. `'auto'` (the default)
lets the model decide, which is where model judgment comes in: a 1-bit model can call when it
should answer, or answer when it should call. Enforcement guarantees the call's *shape*, never
its *judgment* - keep tool sets small (2-3 tools), prefer forced calls when the UI knows one is
needed, and validate argument *values* in your executor. The engine never executes anything,
never loops, never retries: it returns a validated call and the app stays in charge (there is
deliberately no agent framework in here).

### Conversation snapshots (`save` / `restore`)

`chat.save()` captures the whole conversation - the engine's KV cache plus the chat's exact
token bookkeeping - as one structured-cloneable object; `chat.restore(snapshot)` brings it back,
so the next turn extends the cache as if the session never ended (no re-prefill of the history):

```ts
const snapshot = await chat.save()   // structured-cloneable: IndexedDB / OPFS / postMessage
                                     // (NOT JSON.stringify - the KV buffer would be lost)
// ...page reload: new engine + chat on the same model and kvCache mode...
await chat.restore(snapshot)
await chat.send([...savedMessages, { role: 'user', content: 'as I was saying...' }]) // cache reuse
```

Use it for instant conversation switching (save several, restore the active one) and for
resuming after a reload without re-prefilling a long history. Restore validates the model
architecture and `kvCache` mode and throws on mismatch; restoring is bit-identical to having
kept the conversation alive (gated on real hardware). Snapshot size scales with the KV mode
(~224 / 112 / 63 KB per cached token on Bonsai-1.7B at f32 / f16 / q8), which makes
`kvCache: 'q8'` the natural companion. The engine-level `engine.saveCache()` /
`engine.restoreCache()` are the same thing for ids-in/ids-out callers.

The two text libraries (`@huggingface/tokenizers`, `@huggingface/jinja` - pure JS, Apache-2.0,
see THIRD_PARTY_LICENSES.md) are inlined into `dist/chat.js` at build time, the same way the
engine inlines its WGSL: the package keeps **zero runtime dependencies**, and importing plain
`bitgpu` never loads any chat code. Rendering and encoding (including tool declarations, calls,
and responses) are verified byte-exact against transformers.js (`npm run test:chat`), and the
GPU gate proves the reuse paths bit-exact on real hardware. Prefer your own tokenizer? Skip
`bitgpu/chat` entirely - the engine API is unchanged.

## Bring your own model

### Any 1-bit GGUF, zero steps: `bitgpu/gguf`

For GGUF models there is no conversion step at all - `fromGguf` fetches ONLY the header
(HTTP ranges, a few MB, never the weights), builds the manifest in memory, and derives the
lookup tables, so one URL is the whole setup:

```ts
import { createEngine } from 'bitgpu'
import { fromGguf } from 'bitgpu/gguf'

const engine = await createEngine({
  ...(await fromGguf('https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf')),
  kvCache: 'q8',
})
```

The Qwen3.5 hybrid **Bonsai-27B** loads exactly the same way - `fromGguf` is its only path (there is
no offline converter for the hybrid yet), so point at
`prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf` with `kvCache: 'q8'` and, say,
`maxSeqLen: 4096`. bitgpu runs its **text trunk** (the model is multimodal; the vision path is out of
scope, and its chat template is not wired into `bitgpu/chat` yet - drive it at the `forward` /
`generate` token-id level, or supply your own template). It is a ~3.8 GB download and comfortably
wants a 16 GB (or larger) GPU budget - on an 8 GB laptop it runs but the weights spill to swap, so
decode is slow.

The in-browser parse is gated: it must deep-equal the offline converter's manifest AND an
engine built from it must reproduce the known-good greedy ids bit-exactly on GPU. Prefer the
committed [`models/`](models/) manifests when one exists (a `manifest.json` is smaller than a
GGUF header and caches better); `fromGguf` is for models nobody has converted yet.

### Offline converters

bitgpu loads its own small format instead of parsing ONNX (or re-parsing GGUF) at runtime: a
`manifest.json` (the architecture contract + every tensor mapped to a byte range) and a small
aux file, both produced ONCE from a standard export - while the big weights file is
used byte-for-byte unchanged, so it can keep streaming from wherever it already lives (e.g.
the Hugging Face Hub). Same one-time-conversion model as llama.cpp or MLX. Two converters:

```sh
python tools/convert-gguf.py --gguf <a 1-bit Q1_0 .gguf>   # the .gguf itself stays the data file
python tools/convert-onnx.py --model <dir with config.json + the q1 .onnx + its data file>
```

Host the two small files anywhere (they're static), point `createEngine` at them, done:

```ts
createEngine({
  manifestUrl: 'https://your-site.example/model/manifest.json',
  auxUrl: 'https://your-site.example/model/model_q1.aux.bin',
  dataUrl: 'https://huggingface.co/<repo>/resolve/main/<model>.gguf', // or .../onnx/model_q1.onnx_data
})
```

Compatibility envelope: Qwen3-family models quantized with the 1-bit recipe (silu/SwiGLU,
head_dim <= 128, 128-wide scale blocks, tied or untied lm_head), in either container: GGUFs
with PrismML's Q1_0 tensor type (the primary path), or ONNX exports with the onnx-community
"q1" recipe - the engine validates the manifest loudly at load. The Qwen3.5 **hybrid** backbone
(Bonsai-27B: a 3:1 mix of gated-DeltaNet linear attention and gated full attention, head_dim 256,
partial RoPE) is also supported, GGUF-only, under the same 1-bit recipe. The two containers carry
bit-identical weights for the Bonsai releases (verified sign-bit-for-sign-bit on 8B), so pick
by hosting preference.
Reference exports: [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) /
[4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) /
[8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf) /
[27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) (`Bonsai-*-Q1_0.gguf`; the 27B is the Qwen3.5 hybrid, GGUF-only), and
[onnx-community/Bonsai-1.7B-ONNX](https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX) /
[4B](https://huggingface.co/onnx-community/Bonsai-4B-ONNX) /
[8B](https://huggingface.co/onnx-community/Bonsai-8B-ONNX) (`onnx/model_q1.onnx` + data file). Format spec:
[docs/FORMAT.md](docs/FORMAT.md); the full pipeline including regenerating the verification
fixtures for a new model: [tools/README.md](tools/README.md).

## API

- `createEngine(options: EngineOptions | string): Promise<Engine>` - load a model. A bare string is
  treated as `modelUrl`.
- `engine.generate(promptTokenIds, options?)` - generate tokens. Greedy by default; sampling, streaming
  (`onToken`), EOS (`stopTokens`), cancellation (`signal`) and cross-turn cache reuse (`reuseCache`) are
  all supported. `maxTokens` is clamped to the KV window. See the published `EngineOptions` /
  `GenerateOptions` types for the full option shapes.
- `engine.prefill(promptTokenIds)` - prefill a prompt prefix into the KV cache without decoding, so a
  later `generate(delta, { reuseCache: true })` starts from a warm cache (e.g. a static system prompt).
- `engine.forward(tokenIds)` - single forward pass (hidden states + logits) for correctness checks.
- `engine.resetCache()` - clear the cross-turn KV cache (start a fresh conversation).
- `engine.saveCache()` / `engine.restoreCache(snapshot)` - snapshot the conversation (KV cache
  contents + token history) as one structured-cloneable object and bring it back later,
  bit-identically - into this engine or a fresh one on the same model and `kvCache` mode. See
  the chat-layer `save()`/`restore()` below for the batteries-included version.
- `engine.capabilities` - detected GPU path (`useSubgroups`, `subgroupSize`, adapter info, limits).
- `engine.lost` - promise that resolves if the GPU device is lost (also via `onDeviceLost` option);
  create a new engine to recover.
- `engine.dispose()` - release GPU resources.

Errors: `WebGPUUnavailableError` (no WebGPU / no adapter) and `GpuOutOfMemoryError` (weight upload or
KV growth failed) are exported so you can branch on them.

## Browser support

WebGPU with compute is required (a clear `WebGPUUnavailableError` is thrown otherwise).

| Browser | Path | Notes |
| --- | --- | --- |
| Chrome / Edge (desktop) | subgroups when uniform 32/64, else workgroup fallback | fastest path |
| Safari 26+ (macOS/iOS) | subgroups on Apple GPUs | Metal; low dispatch overhead |
| Firefox | workgroup fallback | WebGPU shipped, but per-dispatch overhead is high; expect low throughput |
| Android Chrome | device-dependent | works where WebGPU is exposed; VRAM limits apply |

The engine has no DOM dependencies and WebGPU is available in workers, so the whole stack runs
off the main thread: [examples/worker.html](examples/worker.html) is a complete copy-paste
pattern (module worker + a four-message protocol) whose page stays at full frame rate through
load, prefill, and decode.

## CDN usage

```html
<script type="module">
  import { createEngine } from 'https://esm.sh/bitgpu'
  // or: https://cdn.jsdelivr.net/npm/bitgpu/+esm
</script>
```

## Development

```sh
npm run gen:shaders   # inline shaders/*.wgsl -> src/shaders.generated.ts
npm run build         # tsdown -> dist (ESM + .d.ts)
npm run typecheck
npm run test:sampler  # sampler parity vs transformers.js v4.2.0
npm run test:pld      # prompt-lookup drafter unit checks
npm run test:chat     # bitgpu/chat: stream logic, orchestration, template/encode parity vs transformers.js
npm run check:publish # publint + are-the-types-wrong
```

### GPU verification gate

`examples/verify.html` re-runs the full bit-exactness + throughput suite (forward cosines vs the
committed reference fixtures in `test-fixtures/forward-<tag>/`, known-good greedy ids, sampler kernel
parity, determinism, KV reuse/growth, prompt-lookup identity, KV snapshot save/restore round
trips - including into a fresh engine - the f16/q8 KV-mode sections: within-mode exactness
plus greedy agreement vs f32 out to a 1500-token prompt, the rolling-window section:
pre-eviction bit-exactness, 600 tokens through a 192-token window, determinism across
evictions) against the **built package** and prints `PACKAGE OK` or `REGRESSION`.

It needs model weights, which are not committed. Point `examples/model-1.7b` at a directory
holding the model's `manifest.json` + data/aux files (the reference target is Bonsai-1.7B,
~290 MB):

```sh
ln -s /path/to/bonsai-model examples/model-1.7b   # or copy the files in
npm run build
npm run verify:headless                      # serves the repo itself + drives system Chrome headlessly
FAST=1 npm run verify:headless               # dev iteration: baseline model, core sections only (~3 min; NOT a release gate)
```

Or serve the repo root (`python3 -m http.server 8000`) and open
`http://localhost:8000/examples/verify.html` in a WebGPU browser and click Run. Run this gate on
real hardware before every release; CI covers only the CPU-checkable parts (types, sampler math,
drafter, chat, packaging). The headless driver also runs the baseline model once with
`?nosg=1` (the no-subgroup workgroup-reduction fallback used on Firefox and older adapters), so
that path is release-gated too - it is bit-identical to the subgroup path on the committed
known-good ids.

The gate is model-parametric: `verify.html?model=<tag>` loads `examples/model-<tag>` against
`test-fixtures/forward-<tag>`, and the headless driver automatically runs every staged variant.
Committed fixture sets: `forward-1.7b` (1.7B, hidden 2048), `forward-4b` (4B, hidden 2560),
`forward-8b` (8B, hidden 4096, untied lm_head, raised device limits), plus the GGUF-container
twins `forward-{1.7b,4b,8b}-gguf` (q1_0 de-interleave, tied and untied, three synthesized-rope
configurations) - so engine changes are checked against three geometries and both containers; stage
the extra weights with `ln -s /path/to/bonsai-<size> examples/model-<size>`. The chat-layer checks need
`tokenizer.json` + `tokenizer_config.json` in the staged model dir and skip loudly when absent.
To add fixtures for another model, run `tools/golden.py` then
`tools/reference.py --dump test-fixtures/forward-<tag>` on the converted work dir and record the
engine's greedy continuation as `known_good` in that set's `params.json`.

### Releasing

Publishing runs through GitHub Actions with npm trusted publishing (OIDC + provenance, no token).
After the GPU gate passes locally: bump `version` in package.json, commit, then

```sh
git tag v0.1.1 && git push origin main v0.1.1
```

The release workflow re-runs the CPU checks, verifies the tag matches the package version, and
publishes.

The WGSL kernels live in `shaders/` and are inlined into the bundle at build time (no runtime
`fetch`). `scripts/gen-shaders.ts` does the inlining.

## License

MIT
