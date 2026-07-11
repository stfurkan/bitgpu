# bitgpu

A fast, dependency-free WebGPU runtime for **low-bit LLMs** in the browser.

Today it runs **1-bit (binary-weight)** models.
Reference targets are Bonsai **1.7B, 4B and 8B** (Qwen3 architecture, sign-packed binary linear
weights + 2/4-bit embeddings, tied or untied lm_head) - every size is gated bit-exact against the
reference forward on real hardware. GPU-resident decode (greedy or sampled), streaming, EOS stop,
`AbortSignal`, and cross-turn KV-cache reuse. Runs the fast subgroup path on Apple / NVIDIA /
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

## Quickstart - no conversion, no hosting

Ready-made manifests for all three Bonsai sizes are committed under [`models/`](models/); the
weights stream straight from the Hugging Face Hub. This runs as-is:

```ts
import { createEngine } from 'bitgpu'
import { createChat } from 'bitgpu/chat'

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@v0.8.0/models/bonsai-1.7b'
const HF = 'https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main'
const engine = await createEngine({
  manifestUrl: `${REPO}/manifest.json`,
  auxUrl: `${REPO}/bonsai.aux.bin`,
  dataUrl: `${HF}/onnx/model_q1.onnx_data`,
})
const chat = await createChat(engine, {
  tokenizerJsonUrl: `${HF}/tokenizer.json`,
  tokenizerConfigUrl: `${HF}/tokenizer_config.json`,
})
await chat.send([{ role: 'user', content: 'Hi!' }], { onText: (t) => process.stdout.write(t) })
```

The [demo](https://stfurkan.github.io/bitgpu/examples/chat.html) ([source](examples/chat.html))
is this quickstart as a single HTML file: model picker, streaming chat, JSON mode with a schema
editor, and cached downloads.

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
`additionalProperties: false`, `items`, `minItems` / `maxItems`, string `enum`, nested to any
depth. Anything else (`pattern`, `minimum`, `oneOf`, `$ref`, ...) **throws up front** - never
silently ignored. The guarantee is structural, not semantic: a schema makes the output parse
into the right shape, not be true. Built on the engine's generic `candidateFilter` hook (see
`GenerateOptions`), which is open for custom grammars.

The two text libraries (`@huggingface/tokenizers`, `@huggingface/jinja` - pure JS, Apache-2.0,
see THIRD_PARTY_LICENSES.md) are inlined into `dist/chat.js` at build time, the same way the
engine inlines its WGSL: the package keeps **zero runtime dependencies**, and importing plain
`bitgpu` never loads any chat code. Rendering and encoding are verified byte-exact against
transformers.js (`npm run test:chat`), and the GPU gate proves the reuse path bit-exact on real
hardware. Prefer your own tokenizer? Skip `bitgpu/chat` entirely - the engine API is unchanged.

## Bring your own model

bitgpu loads its own small format instead of parsing ONNX at runtime: a `manifest.json` (the
architecture contract + every tensor mapped to a byte range) and a ~30 KB aux file, both
produced ONCE, offline, from a standard export - while the big weights file is used
byte-for-byte unchanged, so it can keep streaming from wherever it already lives (e.g. the
Hugging Face Hub). Same one-time-conversion model as GGUF/llama.cpp or MLX.

```sh
python tools/convert.py --model <dir with config.json + the q1 .onnx + its data file>
```

Host the two small files anywhere (they're static), point `createEngine` at them, done:

```ts
createEngine({
  manifestUrl: 'https://your-site.example/model/manifest.json',
  auxUrl: 'https://your-site.example/model/model_q1.aux.bin',
  dataUrl: 'https://huggingface.co/<repo>/resolve/main/onnx/model_q1.onnx_data',
})
```

Compatibility envelope: Qwen3-family models quantized with the onnx-community 1-bit ("q1")
recipe (silu/SwiGLU, head_dim <= 128, 128-wide scale blocks, tied or untied lm_head) - the
engine validates the manifest loudly at load. The reference exports are
[onnx-community/Bonsai-1.7B-ONNX](https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX),
[Bonsai-4B-ONNX](https://huggingface.co/onnx-community/Bonsai-4B-ONNX) and
[Bonsai-8B-ONNX](https://huggingface.co/onnx-community/Bonsai-8B-ONNX) (`onnx/model_q1.onnx` +
its data file). Format spec: [docs/FORMAT.md](docs/FORMAT.md); the full pipeline including
regenerating the verification fixtures for a new model: [tools/README.md](tools/README.md).

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
committed reference fixtures in `test-fixtures/forward/`, known-good greedy ids, sampler kernel
parity, determinism, KV reuse/growth, prompt-lookup identity) against the **built package** and
prints `PACKAGE OK` or `REGRESSION`.

It needs model weights, which are not committed. Point `examples/model` at a directory holding the
model's `manifest.json` + data/aux files (the reference target is Bonsai-1.7B, ~290 MB):

```sh
ln -s /path/to/bonsai-model examples/model   # or copy the files in
npm run build
npm run verify:headless                      # serves the repo itself + drives system Chrome headlessly
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
Fixture sets for all three Bonsai sizes are committed - `forward` (1.7B, hidden 2048),
`forward-4b` (4B, hidden 2560) and `forward-8b` (8B, hidden 4096, untied lm_head, raised
device limits) - so engine changes are checked against three geometries; stage the extra
weights with `ln -s /path/to/bonsai-<size> examples/model-<size>`. The chat-layer checks need
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
