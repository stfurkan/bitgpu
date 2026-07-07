// Headless driver for the bitgpu verify.html gate: launches system Chrome with WebGPU,
// clicks Run, waits for the final PACKAGE OK / REGRESSION marker, prints the transcript.
//
//   node scripts/headless-verify.mjs [url]
//
// With no url argument, runs the gate once per locally staged model: examples/model
// (the reference Bonsai-1.7B) plus every examples/model-<tag> that has a matching
// test-fixtures/forward-<tag>. Exits non-zero unless every run prints PACKAGE OK.
import { chromium } from 'playwright-core'
import { existsSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const BASE = 'http://127.0.0.1:8000/examples/verify.html'

let urls
if (process.argv[2]) {
  urls = [process.argv[2]]
} else {
  urls = []
  if (existsSync(join(root, 'examples/model/manifest.json'))) urls.push(BASE)
  for (const d of readdirSync(join(root, 'examples'), { withFileTypes: true })) {
    const m = d.name.match(/^model-(.+)$/)
    if (!m) continue
    if (!existsSync(join(root, 'examples', d.name, 'manifest.json'))) continue
    if (!existsSync(join(root, `test-fixtures/forward-${m[1]}/params.json`))) {
      console.log(`[skip] examples/${d.name}: no test-fixtures/forward-${m[1]}`)
      continue
    }
    urls.push(`${BASE}?model=${m[1]}`)
  }
  if (urls.length === 0) {
    console.error('no staged models found (examples/model or examples/model-<tag>)')
    process.exit(2)
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
    await page.waitForFunction(
      () => /PACKAGE OK|REGRESSION|^ERROR:|\nERROR:/.test(document.getElementById('out').textContent),
      undefined,
      { timeout: 900000, polling: 1000 },
    )
    const transcript = await page.evaluate(() => document.getElementById('out').textContent)
    console.log(transcript)
    if (!/PACKAGE OK/.test(transcript)) allOk = false
    await page.close()
  }
} finally {
  await browser.close()
}
if (!allOk) {
  console.error('\nGATE FAILED: at least one model did not print PACKAGE OK')
  process.exit(1)
}
