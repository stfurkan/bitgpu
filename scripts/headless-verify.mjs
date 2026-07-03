// Headless driver for the bitgpu verify.html gate: launches system Chrome with WebGPU,
// clicks Run, waits for the final PACKAGE OK / REGRESSION marker, prints the transcript.
import { chromium } from 'playwright-core'

const url = process.argv[2] ?? 'http://127.0.0.1:8000/examples/verify.html'
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
})
try {
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
  console.log(await page.evaluate(() => document.getElementById('out').textContent))
} finally {
  await browser.close()
}
