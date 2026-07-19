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
  page.on('pageerror', (e) => console.log('  [pageerror]', String(e)))
  await page.goto(`http://127.0.0.1:${port}/`)
  const out = await page.evaluate(async ({ base, ids }) => {
    try {
      const { createEngine } = await import(`${base}/dist/index.js`)
      const engine = await createEngine({ modelUrl: `${base}/examples/model-synth-qwen35` })
      const r = await engine.forward(ids)
      return { ok: true, embed: [...r.embed], layer0: [...r.layer0], finalnorm: [...r.finalnorm], logits: [...r.logits] }
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
} finally {
  await browser.close(); server.close()
}
console.log(fail === 0 ? '\nHYBRID FORWARD PASS' : `\n${fail} CHECK(S) FAILED`)
process.exit(fail === 0 ? 0 : 1)
