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

## Install

```sh
npm install bitgpu
```

ESM-only, zero runtime dependencies.

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

engine.dispose()
```

Tokenization is intentionally out of scope: the engine operates on token ids, so you can pair it
with any tokenizer.

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
recipe (silu/SwiGLU, head_dim <= 128, 128-wide scale blocks) - the engine validates the
manifest loudly at load. Format spec: [docs/FORMAT.md](docs/FORMAT.md); the full pipeline
including regenerating the verification fixtures for a new model: [tools/README.md](tools/README.md).

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
python3 -m http.server 8000                  # serve the repo root
npm run verify:headless                      # drives system Chrome headlessly (WebGPU/Metal)
```

Or open `http://localhost:8000/examples/verify.html` in a WebGPU browser and click Run. Run this
gate on real hardware before every release; CI covers only the CPU-checkable parts (types, sampler
math, drafter, packaging).

The gate is model-parametric: `verify.html?model=<tag>` loads `examples/model-<tag>` against
`test-fixtures/forward-<tag>`, and the headless driver automatically runs every staged variant.
Fixture sets for all three Bonsai sizes are committed - `forward` (1.7B, hidden 2048),
`forward-4b` (4B, hidden 2560) and `forward-8b` (8B, hidden 4096, untied lm_head, raised
device limits) - so engine changes are checked against three geometries; stage the extra
weights with `ln -s /path/to/bonsai-<size> examples/model-<size>`. To add fixtures for another
model, run `tools/golden.py` then `tools/reference.py --dump test-fixtures/forward-<tag>` on the
converted work dir and record the engine's greedy continuation as `known_good` in that set's
`params.json`.

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
