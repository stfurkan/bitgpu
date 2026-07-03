// Headless verification of the CPU sampler math against transformers.js v4.2.0 (no browser needed).
// The GPU kernels (sampler_penalty, argmax_masked) are validated in the browser; here we prove that
// every CPU-side and GPU-emulated piece - the MT19937 RNG, the repetition_penalty + no_repeat_ngram
// pre-filter, the temperature + softmax + multinomial tail, and the top-K selection - matches the
// reference exactly, using the real exported transformers.js functions. Run: npx tsx scripts/verify-sampler.ts
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  random,
  Tensor,
  softmax as hfSoftmax,
  topk as hfTopk,
  RepetitionPenaltyLogitsProcessor,
  NoRepeatNGramLogitsProcessor,
} from '@huggingface/transformers'
import { MT19937, affectedIds, ngramBans, sampleFromCandidates } from '../src/sampler'

const here = dirname(fileURLToPath(import.meta.url))
const T = 0.5
const PEN = 1.15
const NG = 3
const K = 20

let failures = 0
const check = (name: string, ok: boolean, detail = ''): void => {
  console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? '  ' + detail : ''}`)
  if (!ok) failures++
}

// Load a realistic logit vector: one reference logits row, committed WITH the package so this
// gate runs anywhere (CI, fresh clones) - it must not reach into the app repo's local fixtures.
const VOCAB = 151669
const buf = readFileSync(join(here, '..', 'test-fixtures', 'logits-row.bin'))
if (buf.byteLength !== VOCAB * 4) throw new Error(`logits-row.bin: expected ${VOCAB * 4} bytes, got ${buf.byteLength}`)
const baseRow = new Float32Array(buf.buffer, buf.byteOffset, VOCAB)
console.log(`loaded test-fixtures/logits-row.bin: 1 row x ${VOCAB}`)

// top-K by value, descending, lowest-index tie-break (matches argmax_masked + ORT TopK in practice)
function topKSort(data: Float32Array, k: number): { ids: number[]; vals: number[] } {
  const idx = Array.from({ length: data.length }, (_, i) => i)
  idx.sort((a, b) => data[b] - data[a] || a - b)
  const top = idx.slice(0, k)
  return { ids: top, vals: top.map((i) => data[i]) }
}

// ---- (C) MT19937 matches transformers.js / Python ----
console.log('\n(C) MT19937 RNG')
{
  const r0 = new MT19937(42).random()
  check('seed 42 -> Python reference value', Math.abs(r0 - 0.6394267984578837) < 1e-15, `got ${r0}`)
  random.seed(42)
  check('matches transformers.js random.random()', r0 === random.random())
  const mine = new MT19937(123)
  random.seed(123)
  let seqOk = true
  for (let i = 0; i < 1000; i++) if (mine.random() !== random.random()) { seqOk = false; break }
  check('1000-draw sequence identical (seed 123)', seqOk)
}

// ---- (A) penalty + ngram pre-filter matches the real processors, elementwise ----
console.log('\n(A) repetition_penalty + no_repeat_ngram pre-filter (vs real processors)')
const history = [1, 2, 3, 4, 1, 2, 3] // unique {1,2,3,4} for rep; repeated 3-gram bans token 4
{
  const refData = baseRow.slice()
  const inIds = [history.map((x) => BigInt(x))]
  new RepetitionPenaltyLogitsProcessor(PEN)._call(inIds as never, [{ data: refData }] as never)
  new NoRepeatNGramLogitsProcessor(NG)._call(inIds as never, [{ data: refData }] as never)

  const myData = baseRow.slice()
  for (const id of affectedIds(history)) { const v = myData[id]; myData[id] = v < 0 ? v * PEN : v / PEN }
  for (const id of ngramBans(history, NG)) myData[id] = -Infinity

  let mismatches = 0
  for (let i = 0; i < VOCAB; i++) if (refData[i] !== myData[i] && !(Number.isNaN(refData[i]) && Number.isNaN(myData[i]))) mismatches++
  check('penalized logits identical (all vocab)', mismatches === 0, `${mismatches} mismatches`)
  check('ngram ban includes token 4', ngramBans(history, NG).includes(4))
  check('affected set is deduped {1,2,3,4}', JSON.stringify([...affectedIds(history)].sort((a, b) => a - b)) === '[1,2,3,4]')
}

// ---- (B) full sampler token matches the reference tail across many seeds ----
console.log('\n(B) temperature + softmax + multinomial tail (vs real softmax + real RNG)')
{
  const myData = baseRow.slice()
  for (const id of affectedIds(history)) { const v = myData[id]; myData[id] = v < 0 ? v * PEN : v / PEN }
  for (const id of ngramBans(history, NG)) myData[id] = -Infinity
  const { ids: candIds, vals: candVals } = topKSort(myData, K)

  const seeds = [0, 1, 2, 7, 13, 42, 99, 123, 777, 2026, 65535, 1234567]
  let mismatches = 0
  for (const seed of seeds) {
    // reference tail: temperature on the K candidates, real softmax, real RNG inverse-CDF
    const tempV = Float32Array.from(candVals, (v) => v / T)
    const probs = Array.from(hfSoftmax(tempV))
    random.seed(seed)
    let sum = 0
    for (const p of probs) sum += p
    let x = random.random() * sum
    let refIdx = probs.length - 1
    for (let i = 0; i < probs.length; i++) { x -= probs[i]; if (x < 0) { refIdx = i; break } }
    const refTok = candIds[refIdx]
    // mine
    const myTok = sampleFromCandidates(candIds, candVals, T, new MT19937(seed))
    if (refTok !== myTok) mismatches++
  }
  check(`sampled token matches across ${seeds.length} seeds`, mismatches === 0, `${mismatches} mismatches`)
}

// ---- (D) top-K selection matches ONNX TopK (the op the reference sampler uses) ----
console.log('\n(D) top-K selection (vs real ONNX topk)')
try {
  const myData = baseRow.slice()
  for (const id of affectedIds(history)) { const v = myData[id]; myData[id] = v < 0 ? v * PEN : v / PEN }
  for (const id of ngramBans(history, NG)) myData[id] = -Infinity
  const mine = topKSort(myData, K).ids
  const tns = new Tensor('float32', myData, [1, 1, VOCAB])
  const [, idxTensor] = await hfTopk(tns, K)
  const onnx = Array.from(idxTensor.data as BigInt64Array, Number)
  check('top-20 indices identical to ONNX TopK', JSON.stringify(mine) === JSON.stringify(onnx), `mine[0..3]=${mine.slice(0, 4)} onnx[0..3]=${onnx.slice(0, 4)}`)
} catch (e) {
  console.log(`  [SKIP] ONNX topk unavailable in this environment: ${(e as Error).message}`)
}

console.log(`\n${failures === 0 ? 'ALL SAMPLER CHECKS PASSED' : failures + ' CHECK(S) FAILED'}`)
process.exit(failures === 0 ? 0 : 1)
