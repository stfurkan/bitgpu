// Headless driver for examples/benchmark.html - bitgpu vs transformers.js (dtype q1) on the same
// Bonsai model / GPU. Prints the prefill + decode tok/s comparison. Needs network (transformers.js
// loads from esm.sh + the ONNX weights from the HF Hub).
//
//   node scripts/benchmark.mjs [bitgpu-staged-dir] [hf-onnx-repo] [decodeN]
//     default: model-1.7b-gguf  onnx-community/Bonsai-1.7B-ONNX  64
import { chromium } from 'playwright-core'
import { createServer } from 'node:http'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, sep, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.map': 'application/json', '.bin': 'application/octet-stream', '.onnx_data': 'application/octet-stream', '.gguf': 'application/octet-stream', '.css': 'text/css', '.wgsl': 'text/plain' }
const server = await new Promise((resolve) => {
  const s = createServer((req, res) => {
    const rel = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([/\\])+/, '')
    const path = join(root, rel)
    if (rel.split(sep).includes('..') || !existsSync(path) || !statSync(path).isFile()) return void res.writeHead(404).end()
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream', 'content-length': statSync(path).size, 'cache-control': 'no-store' })
    if (req.method === 'HEAD') return res.end()
    createReadStream(path).pipe(res)
  })
  s.listen(0, '127.0.0.1', () => resolve(s))
})
const port = server.address().port
const [model = 'model-1.7b-gguf', tjs = 'onnx-community/Bonsai-1.7B-ONNX', n = '64'] = process.argv.slice(2)
const url = `http://127.0.0.1:${port}/examples/benchmark.html?model=${encodeURIComponent(model)}&tjs=${encodeURIComponent(tjs)}&n=${encodeURIComponent(n)}`
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'] })
try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.log('[pageerror]', e.message))
  await page.goto(url, { waitUntil: 'load', timeout: 30000 })
  await page.click('#run')
  try {
    await page.waitForFunction(() => /BENCH OK|BENCH FAIL/.test(document.getElementById('out').textContent), undefined, { timeout: 600000, polling: 1000 })
  } catch {
    console.log(`TIMED OUT; partial:\n${await page.evaluate(() => document.getElementById('out').textContent).catch(() => '(unresponsive)')}`)
    process.exitCode = 1
  }
  console.log(await page.evaluate(() => document.getElementById('out').textContent))
} finally {
  await browser.close()
  server.close()
}
