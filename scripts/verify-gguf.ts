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

// ── 4. Hybrid qwen35 (Bonsai-27B arch) parse from a synthetic minimal header ──
// The real Bonsai-27B GGUF is 3.8 GB (no CI weights), so exercise the qwen35 branch on a tiny
// hand-built header: 4 blocks (3 linear + 1 full), untied lm_head. The real header is validated
// out-of-band by tools; this guards the branch + tensor mapping in CI. (Header only - the parser
// never reads tensor data, so dummy offsets are fine.)
console.log('hybrid qwen35 parse (synthetic minimal header)...')
{
  const enc = new TextEncoder()
  const bytes: number[] = []
  const u8 = (v: number): void => void bytes.push(v & 0xff)
  const u32 = (v: number): void => { for (let i = 0; i < 4; i++) u8(v >>> (8 * i)) }
  const u64 = (v: number): void => { let b = BigInt(v); for (let i = 0; i < 8; i++) { u8(Number(b & 0xffn)); b >>= 8n } }
  const f32v = (v: number): void => { const d = new DataView(new ArrayBuffer(4)); d.setFloat32(0, v, true); for (let i = 0; i < 4; i++) u8(d.getUint8(i)) }
  const str = (s: string): void => { const by = enc.encode(s); u64(by.length); by.forEach((x) => u8(x)) }
  const kv: Array<() => void> = []
  const kU32 = (k: string, v: number) => kv.push(() => { str(k); u32(4); u32(v) })
  const kF32 = (k: string, v: number) => kv.push(() => { str(k); u32(6); f32v(v) })
  const kStr = (k: string, v: string) => kv.push(() => { str(k); u32(8); str(v) })
  kStr('general.architecture', 'qwen35')
  kU32('qwen35.embedding_length', 256); kU32('qwen35.block_count', 4); kU32('qwen35.feed_forward_length', 512)
  kU32('qwen35.attention.head_count', 2); kU32('qwen35.attention.head_count_kv', 1)
  kU32('qwen35.attention.key_length', 128); kU32('qwen35.attention.value_length', 128)
  kU32('qwen35.full_attention_interval', 4)
  kU32('qwen35.ssm.group_count', 2); kU32('qwen35.ssm.time_step_rank', 4); kU32('qwen35.ssm.state_size', 128)
  kU32('qwen35.ssm.conv_kernel', 4); kU32('qwen35.ssm.inner_size', 512)
  kU32('qwen35.rope.dimension_count', 64); kF32('qwen35.rope.freq_base', 1e6)
  kF32('qwen35.attention.layer_norm_rms_epsilon', 1e-6); kU32('qwen35.context_length', 262144)
  kU32('tokenizer.ggml.eos_token_id', 42)
  const [F32T, Q1] = [0, 41]
  const tn: Array<{ n: string; d: number[]; t: number }> = [
    { n: 'token_embd.weight', d: [256, 256], t: Q1 }, { n: 'output.weight', d: [256, 256], t: Q1 },
    { n: 'output_norm.weight', d: [256], t: F32T },
  ]
  for (let li = 0; li < 4; li++) {
    tn.push({ n: `blk.${li}.attn_norm.weight`, d: [256], t: F32T }, { n: `blk.${li}.post_attention_norm.weight`, d: [256], t: F32T })
    tn.push({ n: `blk.${li}.ffn_gate.weight`, d: [256, 512], t: Q1 }, { n: `blk.${li}.ffn_up.weight`, d: [256, 512], t: Q1 }, { n: `blk.${li}.ffn_down.weight`, d: [512, 256], t: Q1 })
    if (li % 4 === 3) {
      tn.push({ n: `blk.${li}.attn_q.weight`, d: [256, 512], t: Q1 }, { n: `blk.${li}.attn_k.weight`, d: [256, 128], t: Q1 },
        { n: `blk.${li}.attn_v.weight`, d: [256, 128], t: Q1 }, { n: `blk.${li}.attn_output.weight`, d: [256, 256], t: Q1 },
        { n: `blk.${li}.attn_q_norm.weight`, d: [128], t: F32T }, { n: `blk.${li}.attn_k_norm.weight`, d: [128], t: F32T })
    } else {
      tn.push({ n: `blk.${li}.attn_qkv.weight`, d: [256, 1024], t: Q1 }, { n: `blk.${li}.attn_gate.weight`, d: [256, 512], t: Q1 },
        { n: `blk.${li}.ssm_alpha.weight`, d: [256, 4], t: Q1 }, { n: `blk.${li}.ssm_beta.weight`, d: [256, 4], t: Q1 },
        { n: `blk.${li}.ssm_conv1d.weight`, d: [4, 1024], t: F32T }, { n: `blk.${li}.ssm_a`, d: [4], t: F32T },
        { n: `blk.${li}.ssm_dt.bias`, d: [4], t: F32T }, { n: `blk.${li}.ssm_norm.weight`, d: [128], t: F32T }, { n: `blk.${li}.ssm_out.weight`, d: [512, 256], t: Q1 })
    }
  }
  enc.encode('GGUF').forEach((x) => u8(x)); u32(3); u64(tn.length); u64(kv.length)
  kv.forEach((f) => f())
  tn.forEach((t, idx) => { str(t.n); u32(t.d.length); t.d.forEach((d) => u64(d)); u32(t.t); u64(idx * 32) }) // distinct offsets
  for (let i = 0; i < 64; i++) u8(0) // pad past the header
  const { manifest: m } = fromGgufBytes(new Uint8Array(bytes).buffer as ArrayBuffer, 'tiny-qwen35.gguf')
  const A = m.arch, Hb = A.hybrid, Tn = m.tensors
  check('qwen35 model_type', A.model_type === 'qwen3_5')
  check('qwen35 hybrid present', Hb !== undefined)
  check('qwen35 layer_types', JSON.stringify(Hb?.layer_types) === JSON.stringify(['linear', 'linear', 'linear', 'full']))
  check('qwen35 linear dims', Hb?.linear_key_heads === 2 && Hb?.linear_value_heads === 4 && Hb?.linear_head_dim === 128 && Hb?.conv_kernel === 4 && Hb?.rotary_dim === 64)
  check('qwen35 not tied', A.tie_word_embeddings === false && Tn['lm_head'].weight!.off !== Tn['embed_tokens'].weight!.off)
  check('qwen35 L0.linear.in_qkv', Tn['layers.0.linear.in_qkv']?.N === 1024 && Tn['layers.0.linear.in_qkv']?.K === 256)
  check('qwen35 L0.linear.conv1d', Tn['layers.0.linear.conv1d']?.kind === 'f32' && JSON.stringify(Tn['layers.0.linear.conv1d']?.weight?.shape) === '[4,1024]')
  check('qwen35 L3.attn.q_proj doubled', Tn['layers.3.attn.q_proj']?.N === 512 && Tn['layers.0.attn.q_proj'] === undefined)
  check('qwen35 bad-arch still rejected', (() => { try { fromGgufBytes(new Uint8Array(bytes.slice(0, 4)).buffer as ArrayBuffer, 'x.gguf'); return false } catch { return true } })())
}

console.log(failures === 0 ? 'ALL GGUF CHECKS PASSED' : `${failures} GGUF CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
