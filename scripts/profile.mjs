// Headless dev profiler: prints the true GPU ms/token cost of each kernel group for the staged
// models, so perf work (P0 of the 0.15 perf theme) is evidence-driven. Uses the engine's
// timestamp-query path when the device has it (falls back to CPU wall-clock otherwise).
//
//   node scripts/profile.mjs [tag ...]     # default: every staged examples/model-<tag>-gguf
//
// Drives examples/verify.html?model=<tag>&profile=1 (the profile() routine there), which is NOT
// a correctness gate - run scripts/headless-verify.mjs for that.
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { createReadStream, existsSync, readdirSync, statSync } from 'node:fs'
import { extname, join, normalize, sep, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.map': 'application/json',
  '.bin': 'application/octet-stream', '.onnx_data': 'application/octet-stream',
  '.css': 'text/css', '.wgsl': 'text/plain',
}
function serveRepo() {
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost')
    const rel = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '')
    const path = join(root, rel)
    if (rel.split(sep).includes('..') || !existsSync(path) || !statSync(path).isFile()) return void res.writeHead(404).end()
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream', 'content-length': statSync(path).size, 'cache-control': 'no-store' })
    if (req.method === 'HEAD') return res.end()
    createReadStream(path).pipe(res)
  })
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })))
}

// Which model tags to profile: CLI args, else every staged GGUF model with a fixture set.
let tags = process.argv.slice(2)
if (tags.length === 0) {
  tags = readdirSync(join(root, 'examples'), { withFileTypes: true })
    .map((d) => d.name.match(/^model-(.+-gguf)$/)?.[1])
    .filter((t) => t && existsSync(join(root, 'examples', `model-${t}`, 'manifest.json')) && existsSync(join(root, `test-fixtures/forward-${t}/params.json`)))
  if (tags.length === 0) { console.error('no staged GGUF models found (examples/model-<tag>-gguf)'); process.exit(2) }
}

const { server, port } = await serveRepo()
const browser = await chromium.launch({
  executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
})
try {
  for (const tag of tags) {
    const url = `http://127.0.0.1:${port}/examples/verify.html?model=${encodeURIComponent(tag)}&profile=1`
    console.log(`\n===== profile ${tag} =====`)
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.log('[pageerror]', e.message))
    await page.goto(url, { waitUntil: 'load', timeout: 30000 })
    await page.click('#run')
    try {
      await page.waitForFunction(() => /PROFILE OK|^ERROR:|\nERROR:/.test(document.getElementById('out').textContent), undefined, { timeout: 600000, polling: 1000 })
    } catch {
      console.log(`TIMED OUT; partial:\n${await page.evaluate(() => document.getElementById('out').textContent).catch(() => '(unresponsive)')}`)
      await page.close()
      continue
    }
    console.log(await page.evaluate(() => document.getElementById('out').textContent))
    await page.close()
  }
} finally {
  await browser.close()
  server.close()
}
