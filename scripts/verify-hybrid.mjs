// End-to-end gate for the qwen3_5 hybrid backbone: load the tiny synthetic 1-bit model through the
// built engine, run forward(), and compare every stage (embed/layer0/finalnorm/logits) to the numpy
// golden. Stage it first: `python tools/gen-synth-qwen35.py && npm run build` (== npm run test:hybrid).
// Needs system Chrome + WebGPU (a local GPU gate, like verify:headless).
import { chromium } from 'playwright-core'
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'node:http'

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..')
const MODEL = join(REPO, 'examples', 'model-synth-qwen35')
const FX = join(REPO, 'test-fixtures', 'forward-synth-qwen35')
if (!existsSync(join(MODEL, 'manifest.json'))) { console.log('stage the model first: python tools/gen-synth-qwen35.py'); process.exit(1) }
if (!existsSync(join(REPO, 'dist', 'index.js'))) { console.log('build first: npm run build'); process.exit(1) }
const CT = { '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream', '.map': 'application/json' }

const server = createServer((req, res) => {
  try {
    const p = decodeURIComponent(req.url.split('?')[0])
    if (p === '/') { res.setHeader('content-type', 'text/html'); return res.end('<!doctype html><title>h</title>') }
    const body = readFileSync(join(REPO, p))
    res.setHeader('content-type', CT[extname(p)] || 'application/octet-stream')
    res.end(body)
  } catch { res.statusCode = 404; res.end('nf') }
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const params = JSON.parse(readFileSync(join(FX, 'params.json'), 'utf8'))
const f32 = (n) => new Float32Array(readFileSync(join(FX, n)).buffer.slice(0))
const golden = { embed: f32('embed.bin'), layer0: f32('layer0.bin'), finalnorm: f32('finalnorm.bin'), logits: f32('logits.bin') }

const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
})
let fail = 0
try {
  const page = await browser.newPage()
  if (process.env.SEG) await page.addInitScript((s) => { globalThis.__SEG = s }, Number(process.env.SEG)) // TEMP: C1 experiment knob
  page.on('pageerror', (e) => console.log('  [pageerror]', String(e)))
  await page.goto(`http://127.0.0.1:${port}/`)
  const out = await page.evaluate(async ({ base, ids }) => {
    try {
      const { createEngine } = await import(`${base}/dist/index.js`)
      const engine = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      const r = await engine.forward(ids)
      // decode gate: incremental stateful generate() must equal teacher-forced greedy via forward()
      const K = Math.max(1, Math.min(4, ids.length - 1)), N = Math.min(5, ids.length - K)
      const gen = (await engine.generate(ids.slice(0, K), { maxTokens: N })).tokens
      const seq = ids.slice(0, K), ref = []
      for (let i = 0; i < N; i++) {
        const fr = await engine.forward(seq); const V = fr.logits.length / seq.length
        const last = fr.logits.slice((seq.length - 1) * V); let am = 0
        for (let j = 1; j < V; j++) if (last[j] > last[am]) am = j
        ref.push(am); seq.push(am)
      }
      // SAMPLED-PATH decode gate: the sampled/constrained loop (generateSampledImpl) runs ONE stack()
      // per step, flipping the hybrid state ping-pong every step - the pooled bind groups must follow
      // the alternating state buffers or the recurrence silently freezes (each step re-reads the same
      // state_in; caught live on the 27B as decode stuck at the prompt-end distribution). topK:1 with a
      // tiny temperature is deterministic, so it must match the plain-greedy tokens exactly.
      const sampGen = (await engine.generate(ids.slice(0, K), { maxTokens: N, temperature: 0.01, topK: 1, seed: 1 })).tokens
      // KV-grow gate: force ensureKvCapacity past the 512-position initial cap. generate() reserves
      // capacity for the whole run upfront, so a >512-token generation grows the cache - which for the
      // hybrid grows KV ONLY for the full-attention layers (kvLayers in engine.ts). Re-check that
      // incremental decode still equals teacher-forced greedy ACROSS the grow boundary.
      const pad = Array.from({ length: 510 }, (_, i) => ids[i % ids.length]), GN = 8
      const growGen = (await engine.generate(pad, { maxTokens: GN })).tokens
      const gseq = pad.slice(), growRef = []
      for (let i = 0; i < GN; i++) {
        const fr = await engine.forward(gseq), Vv = fr.logits.length / gseq.length
        const last = fr.logits.slice((gseq.length - 1) * Vv)
        let am = 0; for (let j = 1; j < Vv; j++) if (last[j] > last[am]) am = j
        growRef.push(am); gseq.push(am)
      }
      // q8 KV cache for the 16 full-attention layers: the q8 dequant read (attention_online_cache_kv8)
      // must round-trip what copy_kv8 wrote - q8 must actually engage (not silently fall back to f32),
      // decode must still equal teacher-forced greedy, and logits stay close to f32 (one snorm8 rounding).
      const eq8 = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35`, kvCache: 'q8' })
      const q8mode = eq8.capabilities.kvCache
      const rq8 = await eq8.forward(ids)
      const genQ8 = (await eq8.generate(ids.slice(0, K), { maxTokens: N })).tokens
      const s2 = ids.slice(0, K), refQ8 = []
      for (let i = 0; i < N; i++) {
        const fr = await eq8.forward(s2), Vv = fr.logits.length / s2.length
        const last = fr.logits.slice((s2.length - 1) * Vv)
        let am = 0; for (let j = 1; j < Vv; j++) if (last[j] > last[am]) am = j
        refQ8.push(am); s2.push(am)
      }
      // f16 activations for hybrid decode (activation:'f16'): the projection GEMVs read f16 norm
      // outputs (matmuls accumulate + output f32); the DeltaNet recurrence stays f32. Must actually
      // engage on a shader-f16 adapter and stay greedy-exact vs f32 (one f16 rounding of the norm).
      const ef16 = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35`, activation: 'f16' })
      const af16mode = ef16.capabilities.activation
      const rf16 = await ef16.forward(ids)
      const genF16 = (await ef16.generate(ids.slice(0, K), { maxTokens: N })).tokens
      const s3 = ids.slice(0, K), refF16 = []
      for (let i = 0; i < N; i++) {
        const fr = await ef16.forward(s3), Vv = fr.logits.length / s3.length
        const last = fr.logits.slice((s3.length - 1) * Vv)
        let am = 0; for (let j = 1; j < Vv; j++) if (last[j] > last[am]) am = j
        refF16.push(am); s3.push(am)
      }
      // snapshot persistence: save after turn 1, restore into a FRESH engine, continue with reuse -
      // must equal a cold full-prefill of [history, delta] then generate. Proves saveCache captured the
      // full-attention KV AND the DeltaNet recurrent/conv state, and restore + the re-fed reuse path
      // reconstruct it exactly (a linear layer keeps no KV, only its O(1) state - the real test here).
      const sp = ids.slice(0, 3), delta = [ids[1] % 256]
      const g1 = (await engine.generate(sp, { maxTokens: 6 })).tokens
      const snap = await engine.saveCache()
      const hist = [...sp, ...g1]
      const eFull = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      const snapFull = (await eFull.generate([...hist, ...delta], { maxTokens: 5 })).tokens
      // reuse-only (no snapshot): continue THIS engine (still at [sp,g1] - saveCache doesn't mutate)
      const reuseOnly = (await engine.generate(delta, { reuseCache: true, maxTokens: 5 })).tokens
      const eR = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      await eR.restoreCache(snap)
      const snapRestored = (await eR.generate(delta, { reuseCache: true, maxTokens: 5 })).tokens
      // seg-cadence equivalence: the full 256-token segment with the SUB-CHUNKED scan (current
      // default) must reproduce the 0.17-known-good 16-token-segment prefill - same recurrence
      // flush cadence by construction, so only GEMM batching rounding may differ. __SEG is the
      // engine's test hook (read per prefill call); cleared afterwards.
      globalThis.__SEG = 16
      const eSeg = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      const rSeg = await eSeg.forward(ids)
      const segGen = (await eSeg.generate(pad, { maxTokens: GN })).tokens
      globalThis.__SEG = 0
      // 0.19.1 regressions:
      // (a) prefill()+reuse == cold: prefill() must feed all-but-last so the reuse re-feed applies
      //     the last token to the DeltaNet recurrence exactly ONCE (it used to apply twice).
      const eP = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      await eP.prefill([...hist])
      const prefillReuse = (await eP.generate(delta, { reuseCache: true, maxTokens: 5 })).tokens
      // (b) promptLookup on the hybrid: 'auto' silently skips speculation, explicit throws.
      const eAuto = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      const rAuto = await eAuto.generate(ids.slice(0, K), { maxTokens: N, promptLookup: 'auto' })
      const pldAutoGen = rAuto.tokens
      const pldAutoSkipped = rAuto.speculation === undefined
      let pldExplicitThrew = false
      try { await eAuto.generate([ids[0]], { maxTokens: 2, promptLookup: true }) } catch { pldExplicitThrew = true }
      return { ok: true, prefillReuse, pldAutoGen, pldAutoSkipped, pldExplicitThrew, embed: [...r.embed], layer0: [...r.layer0], finalnorm: [...r.finalnorm], logits: [...r.logits], gen, ref, sampGen, growGen, growRef, q8mode, q8logits: [...rq8.logits], genQ8, refQ8, af16mode, af16logits: [...rf16.logits], genF16, refF16, snapFull, snapRestored, reuseOnly, snapBytes: snap.data.byteLength, seg16logits: [...rSeg.logits], segGen }
    } catch (e) { return { ok: false, err: String((e && e.stack) || e) } }
  }, { base: `http://127.0.0.1:${port}`, ids: params.ids })
  if (!out.ok) { console.log('ENGINE ERROR:', out.err); process.exit(1) }
  const cmp = (name, a, b) => {
    let mad = 0, dot = 0, na = 0, nb = 0
    for (let i = 0; i < b.length; i++) { mad = Math.max(mad, Math.abs(a[i] - b[i])); dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    const cos = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9)
    const ok = mad < 5e-3 && cos > 0.9999
    console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name.padEnd(11)} max|Δ|=${mad.toExponential(2)} cos=${cos.toFixed(6)}`)
    if (!ok) fail++
  }
  cmp('embed', out.embed, golden.embed)
  cmp('layer0', out.layer0, golden.layer0)
  cmp('finalnorm', out.finalnorm, golden.finalnorm)
  cmp('logits', out.logits, golden.logits)
  const V = params.vocab, last = out.logits.slice((params.S - 1) * V)
  const myArg = last.indexOf(Math.max(...last))
  console.log(`  argmax(last): engine=${myArg} golden=${params.argmax_last} ${myArg === params.argmax_last ? 'MATCH' : 'MISMATCH'}`)
  if (myArg !== params.argmax_last) fail++
  // decode == prefill: stateful incremental decode must equal teacher-forced greedy
  const decOk = JSON.stringify(out.gen) === JSON.stringify(out.ref)
  console.log(`  [${decOk ? 'PASS' : 'FAIL'}] decode==prefill  gen=${JSON.stringify(out.gen)} ref=${JSON.stringify(out.ref)}`)
  if (!decOk) fail++
  const sampOk = JSON.stringify(out.sampGen) === JSON.stringify(out.gen)
  console.log(`  [${sampOk ? 'PASS' : 'FAIL'}] sampled-path decode == plain greedy (state ping-pong under pooled bind groups)  samp=${JSON.stringify(out.sampGen)}`)
  const growOk = JSON.stringify(out.growGen) === JSON.stringify(out.growRef)
  console.log(`  [${growOk ? 'PASS' : 'FAIL'}] decode==prefill across KV grow (>512)  gen=${JSON.stringify(out.growGen)} ref=${JSON.stringify(out.growRef)}`)
  if (!growOk) fail++
  // q8 KV cache for the hybrid full-attention layers
  const q8on = out.q8mode === 'q8'
  console.log(`  [${q8on ? 'PASS' : 'FAIL'}] q8 KV engaged for hybrid (capabilities.kvCache=${out.q8mode})`)
  if (!q8on) fail++
  let q8mad = 0, q8dot = 0, q8na = 0, q8nb = 0
  for (let i = 0; i < golden.logits.length; i++) { q8mad = Math.max(q8mad, Math.abs(out.q8logits[i] - golden.logits[i])); q8dot += out.q8logits[i] * golden.logits[i]; q8na += out.q8logits[i] ** 2; q8nb += golden.logits[i] ** 2 }
  const q8cos = q8dot / (Math.sqrt(q8na) * Math.sqrt(q8nb) + 1e-9)
  const q8cosOk = q8cos > 0.99
  console.log(`  [${q8cosOk ? 'PASS' : 'FAIL'}] q8 logits vs f32 golden  cos=${q8cos.toFixed(6)} max|Δ|=${q8mad.toExponential(2)} (q8 is lossy; want cos>0.99)`)
  if (!q8cosOk) fail++
  const q8decOk = JSON.stringify(out.genQ8) === JSON.stringify(out.refQ8)
  console.log(`  [${q8decOk ? 'PASS' : 'FAIL'}] q8 decode==prefill  gen=${JSON.stringify(out.genQ8)} ref=${JSON.stringify(out.refQ8)}`)
  if (!q8decOk) fail++
  // f16 activations for the hybrid decode
  const af16on = out.af16mode === 'f16'
  console.log(`  [${af16on ? 'PASS' : 'WARN'}] f16 activations engaged for hybrid (capabilities.activation=${out.af16mode}${af16on ? '' : ' - adapter lacks shader-f16, fell back to f32'})`)
  if (!af16on) fail++
  let afmad = 0, afdot = 0, afna = 0, afnb = 0, afArg = 0, gArg = 0
  for (let i = 0; i < golden.logits.length; i++) { afmad = Math.max(afmad, Math.abs(out.af16logits[i] - golden.logits[i])); afdot += out.af16logits[i] * golden.logits[i]; afna += out.af16logits[i] ** 2; afnb += golden.logits[i] ** 2; if (out.af16logits[i] > out.af16logits[afArg]) afArg = i; if (golden.logits[i] > golden.logits[gArg]) gArg = i }
  const afcos = afdot / (Math.sqrt(afna) * Math.sqrt(afnb) + 1e-9), afArgOk = afArg === gArg
  console.log(`  [${afcos > 0.99 && afArgOk ? 'PASS' : 'FAIL'}] f16-act logits vs f32 golden  cos=${afcos.toFixed(6)} max|Δ|=${afmad.toExponential(2)} argmax ${afArg}${afArgOk ? '==' : '!='}${gArg} (greedy-exact)`)
  if (!(afcos > 0.99 && afArgOk)) fail++
  const af16decOk = JSON.stringify(out.genF16) === JSON.stringify(out.refF16)
  console.log(`  [${af16decOk ? 'PASS' : 'FAIL'}] f16-act decode==prefill  gen=${JSON.stringify(out.genF16)} ref=${JSON.stringify(out.refF16)}`)
  if (!af16decOk) fail++
  const reuseOk = JSON.stringify(out.reuseOnly) === JSON.stringify(out.snapFull)
  console.log(`  [${reuseOk ? 'PASS' : 'FAIL'}] reuseCache continue == cold full-prefill  reuse=${JSON.stringify(out.reuseOnly)} full=${JSON.stringify(out.snapFull)}`)
  if (!reuseOk) fail++
  const snapOk = JSON.stringify(out.snapRestored) === JSON.stringify(out.snapFull)
  console.log(`  [${snapOk ? 'PASS' : 'FAIL'}] snapshot restore+continue == cold full-prefill  restored=${JSON.stringify(out.snapRestored)} full=${JSON.stringify(out.snapFull)} (${out.snapBytes}B)`)
  const pfrOk = JSON.stringify(out.prefillReuse) === JSON.stringify(out.snapFull)
  if (!pfrOk) fail++
  console.log(`  [${pfrOk ? 'PASS' : 'FAIL'}] prefill()+reuse == cold full-prefill (single recurrence feed)  pr=${JSON.stringify(out.prefillReuse)} full=${JSON.stringify(out.snapFull)}`)
  const pldOk = JSON.stringify(out.pldAutoGen) === JSON.stringify(out.gen) && out.pldAutoSkipped && out.pldExplicitThrew
  if (!pldOk) fail++
  console.log(`  [${pldOk ? 'PASS' : 'FAIL'}] promptLookup gates on hybrid: auto skips (tokens identical, no spec), explicit throws  skipped=${out.pldAutoSkipped} threw=${out.pldExplicitThrew}`)
  if (!snapOk) fail++
  // seg-256 sub-chunked scan vs the 0.17-known-good seg-16 cadence (direct A/B, not vs golden)
  let sgmad = 0, sgdot = 0, sgna = 0, sgnb = 0
  for (let i = 0; i < out.logits.length; i++) { sgmad = Math.max(sgmad, Math.abs(out.seg16logits[i] - out.logits[i])); sgdot += out.seg16logits[i] * out.logits[i]; sgna += out.seg16logits[i] ** 2; sgnb += out.logits[i] ** 2 }
  const sgcos = sgdot / (Math.sqrt(sgna) * Math.sqrt(sgnb) + 1e-9)
  const segGenOk = JSON.stringify(out.segGen) === JSON.stringify(out.growGen)
  const segOk = sgmad < 5e-3 && sgcos > 0.9999 && segGenOk
  console.log(`  [${segOk ? 'PASS' : 'FAIL'}] seg-256 sub-chunked == seg-16 cadence  logits cos=${sgcos.toFixed(6)} max|Δ|=${sgmad.toExponential(2)}, 510-tok continuation ${segGenOk ? 'identical' : `DIFFERS seg16=${JSON.stringify(out.segGen)} vs ${JSON.stringify(out.growGen)}`}`)
  if (!segOk) fail++
} finally {
  await browser.close(); server.close()
}
console.log(fail === 0 ? '\nHYBRID FORWARD + DECODE PASS' : `\n${fail} CHECK(S) FAILED`)
process.exit(fail === 0 ? 0 : 1)
