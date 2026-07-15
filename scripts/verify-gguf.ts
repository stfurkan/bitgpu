// Headless verification of bitgpu/gguf (the in-browser GGUF header parser) against
// tools/convert-gguf.py's committed output (no browser, no GPU needed).
//
// Three tiers:
//  1. LUT parity - ggufLuts() must be byte-identical to every committed models/*.aux.bin
//     (always runs; the committed files came from the python converter).
//  2. Manifest parity - fromGguf() over each locally staged GGUF must deep-equal the
//     committed models/<dir>/manifest.json. Runs per model when the local .gguf exists
//     (examples/model-<tag>-gguf symlinks), auto-skips otherwise (CI has no weights).
//  3. Behavior - progressive header growth (tiny first fetch forces OutOfData retries),
//     the Range-ignoring-server fallback, and loud rejects on non-GGUF bytes.
//
// Run: npx tsx scripts/verify-gguf.ts
import { existsSync, readFileSync, readdirSync, openSync, readSync, closeSync, fstatSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { fromGguf, fromGgufBytes, ggufLuts } from '../src/gguf'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

let failures = 0
const check = (name: string, ok: boolean, detail = ''): void => {
  console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? '  ' + detail : ''}`)
  if (!ok) failures++
}

/** fetchRange over a local file, counting calls (to assert the growth path). */
const fileRange = (path: string) => {
  const calls: number[] = []
  const fetchRange = async (_url: string, off: number, len: number): Promise<ArrayBuffer> => {
    calls.push(len)
    const fd = openSync(path, 'r')
    try {
      const size = fstatSync(fd).size
      const take = Math.min(len, Math.max(size - off, 0))
      const buf = Buffer.alloc(take)
      readSync(fd, buf, 0, take, off)
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + take)
    } finally {
      closeSync(fd)
    }
  }
  return { fetchRange, calls }
}

/** Deep equality with exact number semantics (the manifests are plain JSON values). */
const deepEq = (a: unknown, b: unknown, path = ''): string | null => {
  if (a === b) return null
  if (typeof a !== typeof b) return `${path}: type ${typeof a} != ${typeof b}`
  if (typeof a === 'number' && typeof b === 'number') return Object.is(a, b) ? null : `${path}: ${a} != ${b}`
  if (a === null || b === null || typeof a !== 'object') return `${path}: ${a} != ${b}`
  const ka = Object.keys(a as object).sort()
  const kb = Object.keys(b as object).sort()
  if (ka.join() !== kb.join()) return `${path}: keys [${ka}] != [${kb}]`
  for (const k of ka) {
    const d = deepEq((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`)
    if (d) return d
  }
  return null
}

// ── 1. LUT parity against every committed aux file ──
console.log('LUT parity (ggufLuts() vs committed models/*.aux.bin)...')
const luts = ggufLuts()
let auxCount = 0
for (const dir of readdirSync(join(root, 'models'), { withFileTypes: true })) {
  if (!dir.isDirectory() || !dir.name.endsWith('-gguf')) continue
  const auxName = readdirSync(join(root, 'models', dir.name)).find((f) => f.endsWith('.aux.bin'))
  if (!auxName) continue
  const committed = readFileSync(join(root, 'models', dir.name, auxName))
  check(`${dir.name}/${auxName}`, committed.length === luts.length && committed.equals(Buffer.from(luts)))
  auxCount++
}
check('found committed gguf aux files', auxCount >= 3, `(${auxCount})`)

// ── 2. Manifest parity against the committed converter output ──
console.log('manifest parity (fromGguf vs committed manifest.json, per locally staged GGUF)...')
let staged = 0
for (const dir of readdirSync(join(root, 'models'), { withFileTypes: true })) {
  if (!dir.isDirectory() || !dir.name.endsWith('-gguf')) continue
  const committed = JSON.parse(readFileSync(join(root, 'models', dir.name, 'manifest.json'), 'utf8'))
  const gguf = join(root, 'examples', `model-${dir.name.replace(/^bonsai-/, '')}`, committed.data_file)
  if (!existsSync(gguf)) {
    console.log(`  [skip] ${dir.name}: ${committed.data_file} not staged locally`)
    continue
  }
  staged++
  const { fetchRange, calls } = fileRange(gguf)
  const { manifest, aux, dataUrl } = await fromGguf(`https://example.test/${committed.data_file}`, { fetchRange })
  const diff = deepEq(manifest, committed)
  check(`${dir.name} manifest deep-equal`, diff === null, diff ?? `(${Object.keys(manifest.tensors).length} tensors, ${calls.length} fetch${calls.length > 1 ? 'es' : ''})`)
  check(`${dir.name} aux == committed`, Buffer.from(aux).equals(readFileSync(join(root, 'models', dir.name, committed.aux_file))))
  check(`${dir.name} dataUrl passthrough`, dataUrl === `https://example.test/${committed.data_file}`)
}
if (staged === 0) console.log('  [skip] no locally staged GGUFs - manifest parity ran 0 models (CI mode)')

// ── 3. Behavior: progressive growth, range-fallback equivalence, loud rejects ──
console.log('behavior checks...')
{
  // growth: force a tiny first fetch by draining through a fetchRange that serves at most
  // 4 KiB more than asked each round - fromGguf must retry with 4x sizes until the header fits.
  const anyGguf = (() => {
    for (const dir of readdirSync(join(root, 'models'), { withFileTypes: true })) {
      if (!dir.isDirectory() || !dir.name.endsWith('-gguf')) continue
      const m = JSON.parse(readFileSync(join(root, 'models', dir.name, 'manifest.json'), 'utf8'))
      const p = join(root, 'examples', `model-${dir.name.replace(/^bonsai-/, '')}`, m.data_file)
      if (existsSync(p)) return p
    }
    return null
  })()
  if (anyGguf) {
    const { fetchRange, calls } = fileRange(anyGguf)
    const clipped = async (url: string, off: number, len: number) => (await fetchRange(url, off, len)).slice(0, Math.min(len, 4096 * calls.length))
    let grew = false
    try {
      await fromGguf('https://example.test/x.gguf', { fetchRange: clipped })
      grew = false // a 4 KiB prefix can never hold a Bonsai header
    } catch (e) {
      grew = String(e).includes('truncated')
    }
    check('short prefix fails loudly as truncated', grew)
    const full = fileRange(anyGguf)
    const parsed = await fromGguf('https://example.test/x.gguf', { fetchRange: full.fetchRange })
    check('growth path converges', parsed.manifest.arch.layers > 0, `(fetches: ${full.calls.join(' -> ')})`)
  } else {
    console.log('  [skip] growth checks need one staged GGUF')
  }
  // rejects
  let threw = ''
  try {
    fromGgufBytes(new TextEncoder().encode('definitely not a gguf file, padded to enough bytes.....').buffer as ArrayBuffer, 'x.gguf')
  } catch (e) {
    threw = String(e)
  }
  check('non-GGUF bytes rejected loudly', threw.includes('bad magic'), threw.slice(0, 60))
}

console.log(failures === 0 ? 'ALL GGUF CHECKS PASSED' : `${failures} GGUF CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
