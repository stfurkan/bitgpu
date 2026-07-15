// Headless driver for the bitgpu verify.html gate: serves the repo root itself (no manual
// http.server step), launches system Chrome with WebGPU, clicks Run, waits for the final
// PACKAGE OK / REGRESSION marker, prints the transcript.
//
//   node scripts/headless-verify.mjs [url]
//
// With no url argument, runs the gate once per locally staged model that the default plan
// covers: every GGUF-container model (examples/model-<tag>-gguf) with a matching
// test-fixtures/forward-<tag>, plus the baseline ONNX model ("1.7b") as a container canary,
// plus one no-subgroup fallback run (?nosg=1) on the baseline, so the workgroup-reduction
// path (Firefox and older adapters) is release-gated too. GGUF is the primary ingestion
// path; the ONNX loader is frozen code whose geometries were all proven at 0.11.0, so one
// ONNX canary catches regressions in the shared machinery. Set FULL=1 to run EVERY staged
// model in both containers (use before releases that touch the loader/container code), and
// NOSG=all to add the fallback run to every planned model too (use before releases that
// touch the _wg kernels or dispatch/geometry code). Exits non-zero unless every run
// prints PACKAGE OK. PLAN=1 prints the planned runs and exits (driver self-check).
//
// With a url argument, that single URL is run against whatever server it points at
// (no server is started).
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { createReadStream, existsSync, readdirSync, statSync } from 'node:fs'
import { extname, join, normalize, sep } from 'node:path'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.map': 'application/json',
  '.bin': 'application/octet-stream', '.onnx_data': 'application/octet-stream',
  '.css': 'text/css', '.wgsl': 'text/plain',
}

/** Minimal static file server over the repo root (follows the model-dir symlinks; streams the
 *  multi-GB weight files instead of buffering). Only what verify.html needs: GET + HEAD. */
function serveRepo() {
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost')
    const rel = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '')
    const path = join(root, rel)
    if (rel.split(sep).includes('..') || !existsSync(path) || !statSync(path).isFile()) {
      res.writeHead(404).end()
      return
    }
    const size = statSync(path).size
    res.writeHead(200, {
      'content-type': MIME[extname(path)] ?? 'application/octet-stream',
      'content-length': size,
      'cache-control': 'no-store',
    })
    if (req.method === 'HEAD') return res.end()
    createReadStream(path).pipe(res)
  })
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }))
  })
}

let urls
let server = null
if (process.argv[2]) {
  urls = [process.argv[2]]
} else {
  const started = await serveRepo()
  server = started.server
  const BASE = `http://127.0.0.1:${started.port}/examples/verify.html`
  const nosgAll = process.env.NOSG === 'all'
  const full = process.env.FULL === '1'
  urls = []
  for (const d of readdirSync(join(root, 'examples'), { withFileTypes: true })) {
    const m = d.name.match(/^model-(.+)$/)
    if (!m) continue
    if (!existsSync(join(root, 'examples', d.name, 'manifest.json'))) continue
    if (!existsSync(join(root, `test-fixtures/forward-${m[1]}/params.json`))) {
      console.log(`[skip] examples/${d.name}: no test-fixtures/forward-${m[1]}`)
      continue
    }
    // default plan: every GGUF model + the baseline ONNX canary; FULL=1 runs everything
    if (!full && !m[1].endsWith('-gguf') && m[1] !== '1.7b') {
      console.log(`[skip] examples/${d.name}: ONNX non-baseline (default plan runs GGUF + the 1.7b canary; FULL=1 to include)`)
      continue
    }
    urls.push(`${BASE}?model=${encodeURIComponent(m[1])}`)
    // release-gate the no-subgroup fallback path on the baseline model (all models with NOSG=all)
    if (m[1] === '1.7b' || nosgAll) urls.push(`${BASE}?model=${encodeURIComponent(m[1])}&nosg=1`)
  }
  if (!existsSync(join(root, 'examples/model-1.7b/manifest.json')))
    console.log('[skip] examples/model-1.7b not staged: the baseline 1.7B gate (and the nosg fallback run) WILL NOT RUN')
  if (urls.length === 0) {
    console.error('no staged models found (examples/model-<tag>)')
    process.exit(2)
  }
  if (process.env.FAST === '1') {
    // Development iteration: ONE run - the baseline model, subgroup path, core sections only
    // (?fast=1 skips the KV-mode/snapshot/sinks sections). Not a release gate.
    urls = [`${BASE}?fast=1`]
  }
  if (process.env.PLAN === '1') {
    console.log(urls.map((u) => u.replace(/^.*verify\.html/, 'verify.html')).join('\n'))
    server.close()
    process.exit(0)
  }
}

const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
})
let allOk = true
try {
  for (const url of urls) {
    console.log(`\n===== ${url} =====`)
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log('[pageerror]', e.message))
    await page.goto(url, { waitUntil: 'load', timeout: 30000 })
    const gpu = await page.evaluate(async () => {
      if (!navigator.gpu) return 'NO navigator.gpu'
      const a = await navigator.gpu.requestAdapter()
      if (!a) return 'NO adapter'
      const i = a.info ?? {}
      return `adapter: ${i.vendor ?? '?'} ${i.architecture ?? '?'} sgMax=${i.subgroupMaxSize ?? '?'}`
    })
    console.log('[gpu]', gpu)
    if (gpu.startsWith('NO')) process.exit(2)
    await page.click('#run')
    try {
      await page.waitForFunction(
        () => /PACKAGE OK|REGRESSION|^ERROR:|\nERROR:/.test(document.getElementById('out').textContent),
        undefined,
        { timeout: 2700000, polling: 1000 },
      )
    } catch (e) {
      // Timeout: dump the partial transcript so the stall point is visible, then fail the gate.
      const partial = await page.evaluate(() => document.getElementById('out').textContent).catch(() => '(page unresponsive)')
      console.log(`TIMED OUT after 2700s; partial transcript:\n${partial}`)
      allOk = false
      await page.close()
      continue
    }
    const transcript = await page.evaluate(() => document.getElementById('out').textContent)
    console.log(transcript)
    if (!/PACKAGE OK/.test(transcript)) allOk = false
    await page.close()
  }
} finally {
  await browser.close()
  server?.close()
}
if (!allOk) {
  console.error('\nGATE FAILED: at least one model did not print PACKAGE OK')
  process.exit(1)
}
