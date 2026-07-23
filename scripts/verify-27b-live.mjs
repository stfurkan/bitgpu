// LIVE on-device gate for the REAL Bonsai-27B (opt-in: `npm run test:27b`). Streams the 3.8 GB
// weights from the Hugging Face Hub straight into GPU memory - nothing is persisted to disk
// (Chrome's disk cache is capped, no Cache Storage), so it runs even on machines with little free
// space. Takes several minutes (network + a swap-bound model on 8 GB machines) and needs system
// Chrome + WebGPU, so it is NOT part of the default gates - run it before any release that
// touches the hybrid prefill/decode path, memory sizing, or the chat tool grammar.
//
// What it proves (the classes the synth-model gate CANNOT catch - its scratch is KBs, the 27B's
// is ~31.5 MB/token, and the silent-OOM failure mode only exists at real scale):
//   A. Long-prompt prefill on the DEFAULT (memory-bounded) segment is token-identical to the
//      16-token-segment cadence (the known-good 0.17 path) - this is the check that caught the
//      0.18 release-blocker where 64+-token segments silently corrupted (scratch VRAM exhaustion:
//      failed WebGPU allocations return invalid buffers and writes vanish, with NO error; the
//      telltale is an impossibly FAST prefill). Also reports the prefill speedup.
//   B. Typed tool calling end-to-end: schema-enforced number + enum parameter values through the
//      real model at its model-card sampling settings, including the cache-reusing round trip.
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { execSync } from 'node:child_process'
import { extname, join, normalize, sep, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
if (!existsSync(join(root, 'dist', 'index.js'))) {
  console.log('build first: npm run build')
  process.exit(1)
}
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' }
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const rel = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '')
  const path = join(root, rel)
  if (rel.split(sep).includes('..') || !existsSync(path) || !statSync(path).isFile()) return res.writeHead(404).end()
  res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream', 'content-length': statSync(path).size, 'cache-control': 'no-store', 'access-control-allow-origin': '*' })
  createReadStream(path).pipe(res)
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port

// abort rather than fill a small disk (swap grows while the 27B runs on 8 GB machines)
const freeGiB = () => Number(execSync("df -g / | tail -1 | awk '{print $4}'").toString().trim())
console.log(`[disk] ${freeGiB()} GiB free at start`)
const guard = setInterval(() => {
  if (freeGiB() < 1) {
    console.log('[disk] ABORT: less than 1 GiB free')
    process.exit(3)
  }
}, 20000)

const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal', '--disk-cache-size=104857600'],
})
const page = await browser.newPage()
page.on('console', (m) => console.log(`[page] ${m.text()}`))
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`))
await page.goto(`http://127.0.0.1:${port}/examples/verify.html`, { waitUntil: 'domcontentloaded' })

const out = await page.evaluate(async (base) => {
  const log = (s) => console.log(s)
  const { createEngine } = await import(`${base}/dist/index.js`)
  const { createChat } = await import(`${base}/dist/chat.js`)
  let lastPhase = ''
  const t0 = performance.now()
  const engine = await createEngine({
    manifestUrl: `${base}/models/bonsai-27b-gguf/manifest.json`,
    auxUrl: `${base}/models/bonsai-27b-gguf/Bonsai-27B-Q1_0.aux.bin`,
    dataUrl: 'https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf',
    kvCache: 'q8',
    activation: 'f16',
    maxSeqLen: 1024,
    onProgress: (p) => { if (p.phase !== lastPhase) { lastPhase = p.phase; log(`[load] ${p.phase}`) } },
  })
  log(`[load] done in ${((performance.now() - t0) / 60000).toFixed(1)} min`)
  const TOK = 'https://huggingface.co/prism-ml/Bonsai-27B-unpacked/resolve/main'
  const chat = await createChat(engine, { tokenizerJsonUrl: `${TOK}/tokenizer.json`, tokenizerConfigUrl: `${TOK}/tokenizer_config.json` })

  // A: long prompt (>the historical ~50-token failure point), default segment vs 16-token cadence
  const passage =
    'Photosynthesis is the process by which green plants and some other organisms use sunlight to ' +
    'synthesize foods from carbon dioxide and water. In plants, it occurs mainly in the leaves, whose ' +
    'cells contain organelles called chloroplasts. Chlorophyll, the green pigment inside chloroplasts, ' +
    'absorbs light most strongly in the blue and red parts of the spectrum. The overall process converts ' +
    'light energy into chemical energy stored in glucose, releasing oxygen as a byproduct. ' +
    'In one short sentence: what pigment absorbs the light, and what gas is released?'
  globalThis.__SEG = 16
  const r16 = await chat.send([{ role: 'user', content: passage }], { maxTokens: 16, reuseCache: false })
  chat.reset()
  globalThis.__SEG = 0
  const rDef = await chat.send([{ role: 'user', content: passage }], { maxTokens: 16, reuseCache: false })
  chat.reset()

  // B: typed tool calling (number + enum params, model-card sampling, cache-reusing round trip)
  const tools = [{
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Apply the operation to the two numbers and return the result.',
      parameters: {
        type: 'object', required: ['a', 'b', 'op'], additionalProperties: false,
        properties: { a: { type: 'number' }, b: { type: 'number' }, op: { type: 'string', enum: ['add', 'multiply'] } },
      },
    },
  }]
  const msgs = [{ role: 'user', content: 'What is 8 plus 5? Use the calculate tool.' }]
  const t1 = await chat.send(msgs, { tools, temperature: 0.5, topP: 0.85, topK: 20, maxTokens: 96 })
  let round = null
  if (t1.finishReason === 'tool_calls' && t1.toolCalls.length) {
    const c = t1.toolCalls[0]
    const val = c.name === 'calculate' && c.arguments.op === 'add' ? Number(c.arguments.a) + Number(c.arguments.b) : NaN
    const t2 = await chat.send(
      [...msgs, { role: 'assistant', content: t1.text, tool_calls: t1.toolCalls }, { role: 'tool', content: String(val) }],
      { tools, temperature: 0.5, topP: 0.85, topK: 20, maxTokens: 64 },
    )
    round = { reused: t2.reusedCache, text: t2.text }
  }
  return {
    a: { tok16: r16.tokens, tokDef: rDef.tokens, text16: r16.text, textDef: rDef.text, prefill16: Math.round(r16.prefillMs), prefillDef: Math.round(rDef.prefillMs) },
    b: { finish: t1.finishReason, calls: t1.toolCalls, round },
  }
}, `http://127.0.0.1:${port}`)

clearInterval(guard)
let fail = 0
const check = (name, ok, detail = '') => {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  ' + detail : ''}`)
  if (!ok) fail++
}
console.log('\n===== A: memory-bounded segment vs 16-token cadence =====')
console.log(`  seg16   prefill ${out.a.prefill16} ms  ${JSON.stringify(out.a.text16)}`)
console.log(`  default prefill ${out.a.prefillDef} ms  ${JSON.stringify(out.a.textDef)} (${(out.a.prefill16 / Math.max(1, out.a.prefillDef)).toFixed(2)}x)`)
check('long-prompt tokens identical across cadences', JSON.stringify(out.a.tok16) === JSON.stringify(out.a.tokDef))
check('default prefill not slower than the 16-cadence', out.a.prefillDef <= out.a.prefill16 * 1.15)
console.log('\n===== B: typed tool calling =====')
const c = out.b.calls?.[0]
check('model called the tool', out.b.finish === 'tool_calls' && !!c, `finish=${out.b.finish}`)
check('typed values enforced (numbers + enum)', !!c && c.name === 'calculate' && typeof c.arguments.a === 'number' && typeof c.arguments.b === 'number' && ['add', 'multiply'].includes(c.arguments.op), JSON.stringify(c?.arguments))
check('round trip reused the cache and answered', !!out.b.round?.reused && /13/.test(out.b.round?.text ?? ''), JSON.stringify(out.b.round))
console.log(`\n[disk] ${freeGiB()} GiB free at end`)
await browser.close()
server.close()
console.log(fail === 0 ? '\n27B LIVE GATE PASS' : `\n${fail} CHECK(S) FAILED`)
process.exit(fail === 0 ? 0 : 1)
