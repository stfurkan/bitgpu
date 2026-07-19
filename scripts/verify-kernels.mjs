// Per-kernel WebGPU validation for the Qwen3.5 hybrid kernels: runs each shaders/<file>.wgsl
// against the numpy-oracle cases (tools/gen-kernel-cases.py -> .kernel-cases/*.json) headlessly
// and reports max|Δ| / cosine vs expected. Isolates each kernel before the end-to-end gate.
// Binding convention (matches bitgpu shaders): 0 = Params uniform, 1..k = input storage (read),
// k+1 = output storage (read_write). Needs system Chrome + WebGPU (a LOCAL gate, like verify:headless).
//   python tools/gen-kernel-cases.py && node scripts/verify-kernels.mjs   (== npm run test:kernels)
import { chromium } from 'playwright-core'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'node:http'

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..')
const casesDir = process.argv[2] || join(REPO, '.kernel-cases')
if (!existsSync(casesDir)) { console.log(`no cases dir ${casesDir} (run: python tools/gen-kernel-cases.py)`); process.exit(1) }

const paramBytes = (fields) => {
  const buf = new ArrayBuffer(Math.max(Math.ceil(fields.length / 4) * 16, 16))
  const dv = new DataView(buf)
  fields.forEach((f, i) => (f[0] === 'f' ? dv.setFloat32(i * 4, f[1], true) : dv.setUint32(i * 4, f[1] >>> 0, true)))
  return Array.from(new Uint8Array(buf))
}

// runs entirely in the page (WebGPU context)
const runKernel = async ({ src, pbytes, inputs, outLen, outLen2, dispatch, overrides }) => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return { err: 'no adapter' }
  const device = await adapter.requestDevice()
  device.pushErrorScope('validation')
  device.pushErrorScope('out-of-memory')
  const U = GPUBufferUsage
  let pipeline
  try {
    const module = device.createShaderModule({ code: src })
    const info = await module.getCompilationInfo()
    const errs = info.messages.filter((m) => m.type === 'error')
    if (errs.length) return { err: 'compile: ' + errs.map((m) => `${m.lineNum}:${m.message}`).join(' | ') }
    pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants: overrides || {} } })
  } catch (e) {
    return { err: 'pipeline: ' + String(e) }
  }
  const pbuf = device.createBuffer({ size: Math.max(pbytes.length, 16), usage: U.UNIFORM | U.COPY_DST })
  device.queue.writeBuffer(pbuf, 0, new Uint8Array(pbytes))
  const ibufs = inputs.map((arr) => {
    const a = new Float32Array(arr)
    const b = device.createBuffer({ size: Math.max(a.byteLength, 4), usage: U.STORAGE | U.COPY_DST })
    device.queue.writeBuffer(b, 0, a)
    return b
  })
  const obuf = device.createBuffer({ size: outLen * 4, usage: U.STORAGE | U.COPY_SRC })
  const entries = [{ binding: 0, resource: { buffer: pbuf } }]
  ibufs.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }))
  entries.push({ binding: ibufs.length + 1, resource: { buffer: obuf } })
  // optional secondary output (e.g. the DeltaNet state); bound but not compared
  if (outLen2) entries.push({ binding: ibufs.length + 2, resource: { buffer: device.createBuffer({ size: outLen2 * 4, usage: U.STORAGE }) } })
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(dispatch[0], dispatch[1] || 1, dispatch[2] || 1)
  pass.end()
  const rb = device.createBuffer({ size: outLen * 4, usage: U.MAP_READ | U.COPY_DST })
  enc.copyBufferToBuffer(obuf, 0, rb, 0, outLen * 4)
  device.queue.submit([enc.finish()])
  await rb.mapAsync(GPUMapMode.READ)
  const out = Array.from(new Float32Array(rb.getMappedRange().slice(0)))
  const oom = await device.popErrorScope()
  const val = await device.popErrorScope()
  return { out, err: (val && val.message) || (oom && oom.message) || null }
}

const files = readdirSync(casesDir).filter((f) => f.endsWith('.json'))
if (!files.length) { console.log('no cases (run tools/gen-kernel-cases.py first)'); process.exit(1) }

// WebGPU needs a secure context (localhost qualifies); serve a blank page.
const server = createServer((_req, res) => { res.setHeader('content-type', 'text/html'); res.end('<!doctype html><title>kh</title>') })
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal'],
})
let fails = 0
try {
  const page = await browser.newPage()
  page.on('console', (m) => { if (m.type() === 'error') console.log('  [page error]', m.text()) })
  await page.goto(`http://127.0.0.1:${port}/`)
  await page.waitForFunction(() => !!navigator.gpu, { timeout: 10000 })
  for (const f of files.sort()) {
    const c = JSON.parse(readFileSync(join(casesDir, f), 'utf8'))
    const src = readFileSync(join(REPO, 'shaders', c.shader), 'utf8')
    const { out, err } = await page.evaluate(runKernel, {
      src, pbytes: paramBytes(c.params), inputs: c.inputs, outLen: c.outLen, outLen2: c.outLen2, dispatch: c.dispatch, overrides: c.overrides,
    })
    if (err || !out) { console.log(`  [FAIL] ${c.name}: ${err}`); fails++; continue }
    let mad = 0, dot = 0, na = 0, nb = 0
    for (let i = 0; i < c.expected.length; i++) {
      mad = Math.max(mad, Math.abs(out[i] - c.expected[i]))
      dot += out[i] * c.expected[i]; na += out[i] * out[i]; nb += c.expected[i] * c.expected[i]
    }
    const cos = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9)
    const ok = mad < 2e-3 && cos > 0.99999
    console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${c.name.padEnd(16)} max|Δ|=${mad.toExponential(2)} cos=${cos.toFixed(6)} (n=${c.expected.length})`)
    if (!ok) fails++
  }
} finally {
  await browser.close()
  server.close()
}
console.log(fails === 0 ? '\nALL KERNELS PASS' : `\n${fails} KERNEL(S) FAILED`)
process.exit(fails === 0 ? 0 : 1)
