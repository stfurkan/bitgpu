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
import { MT19937, affectedIds, applyDry, ngramBans, sampleFromCandidates } from '../src/sampler'

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

// ---- (E) top-p / min-p candidate warpers (bitgpu extensions, applied over the top-K) ----
console.log('\n(E) top-p / min-p warpers')
{
  const myData = baseRow.slice()
  for (const id of affectedIds(history)) { const v = myData[id]; myData[id] = v < 0 ? v * PEN : v / PEN }
  for (const id of ngramBans(history, NG)) myData[id] = -Infinity
  const { ids: candIds, vals: candVals } = topKSort(myData, K)
  const seeds = [0, 1, 7, 42, 123, 777, 2026, 65535]
  const many = Array.from({ length: 300 }, (_, i) => i)

  // the load-bearing invariant: OFF (topP=1, minP=0) is bit-identical to the plain draw, so every
  // model that does not set these keeps its exact sampling behaviour
  let exact = 0
  for (const seed of seeds) if (sampleFromCandidates(candIds, candVals, T, new MT19937(seed)) === sampleFromCandidates(candIds, candVals, T, new MT19937(seed), 1, 0)) exact++
  check('topP=1,minP=0 is bit-identical to the plain draw', exact === seeds.length, `${exact}/${seeds.length}`)

  const tv = Float32Array.from(candVals, (v) => v / T)
  const probs = Array.from(hfSoftmax(tv))
  const maxP = probs[0]

  let topOnly = 0, minOnly = 0
  for (const seed of seeds) {
    if (sampleFromCandidates(candIds, candVals, T, new MT19937(seed), 1e-6, 0) === candIds[0]) topOnly++
    if (sampleFromCandidates(candIds, candVals, T, new MT19937(seed), 1, 1) === candIds[0]) minOnly++
  }
  check('topP~0 collapses to the argmax', topOnly === seeds.length)
  check('minP=1 collapses to the argmax', minOnly === seeds.length)

  // top-p keeps exactly the nucleus (cumulative >= 0.8): nothing beyond it is ever drawn
  let cum = 0, mP = 0
  for (let i = 0; i < probs.length; i++) { cum += probs[i]; mP = i + 1; if (cum >= 0.8) break }
  const nucleus = new Set(candIds.slice(0, mP))
  let inNucleus = 0
  for (const seed of many) if (nucleus.has(sampleFromCandidates(candIds, candVals, T, new MT19937(seed), 0.8, 0))) inNucleus++
  check('topP=0.8 never draws outside the nucleus', inNucleus === many.length, `${inNucleus}/${many.length} (nucleus ${mP})`)

  // min-p keeps exactly {prob >= minP*maxProb}
  const MP = 0.3, thr = MP * maxP
  let mMin = 1
  while (mMin < probs.length && probs[mMin] >= thr) mMin++
  const kept = new Set(candIds.slice(0, mMin))
  let inMin = 0
  for (const seed of many) if (kept.has(sampleFromCandidates(candIds, candVals, T, new MT19937(seed), 1, MP))) inMin++
  check('minP=0.3 never draws below the threshold', inMin === many.length, `${inMin}/${many.length} (kept ${mMin})`)
}

// (F) DRY penalty: pure-function checks on applyDry
{
  const o = (over: Partial<Parameters<typeof applyDry>[3]> = {}) => ({ multiplier: 2, base: 1.75, allowedLength: 2, range: 0, breakers: new Set<number>(), ...over })

  // no repeat context: untouched, order preserved
  let r = applyDry([5, 6, 7], [3, 2, 1], [1, 2, 3, 4], o())
  check('DRY no-repeat leaves logits untouched', JSON.stringify(r) === JSON.stringify({ ids: [5, 6, 7], vals: [3, 2, 1] }))

  // history [.. 11 12 20 .. 11 12], candidate 20 extends the 2-repeat "11 12": penalty = 2*1.75^0
  r = applyDry([20, 99], [5, 4.9], [10, 11, 12, 20, 30, 40, 11, 12], o())
  check('DRY penalizes the repeat-extending candidate', Math.abs((r.vals[r.ids.indexOf(20)] ?? 0) - 3) < 1e-9, `got ${r.vals[r.ids.indexOf(20)]}`)
  check('DRY leaves the non-extending candidate alone', r.vals[r.ids.indexOf(99)] === 4.9)
  check('DRY re-sorts descending after the penalty', r.ids[0] === 99)

  // repeats below allowedLength are free
  r = applyDry([20], [5], [10, 11, 12, 20, 30, 40, 99, 12], o()) // only L=1 (".. 12" matches)
  check('DRY tolerates repeats below allowedLength', r.vals[0] === 5)

  // longer repeat -> exponential: L=3 -> 2*1.75^1
  r = applyDry([30], [5], [9, 11, 12, 13, 30, 40, 11, 12, 13], o()) // "11 12 13"+30 reproduces "11 12 13 30"
  check('DRY penalty grows with repeat length', Math.abs(r.vals[0] - (5 - 2 * 1.75)) < 1e-9, `got ${r.vals[0]}`)

  // a breaker inside the match window cuts the repeat below allowedLength
  r = applyDry([20], [5], [10, 11, 12, 20, 30, 40, 11, 12], o({ breakers: new Set([11]) }))
  check('DRY breaker resets matching', r.vals[0] === 5)

  // a breaker candidate is never penalized even if it extends a repeat
  r = applyDry([20], [5], [10, 11, 12, 20, 30, 40, 11, 12], o({ breakers: new Set([20]) }))
  check('DRY never penalizes a breaker candidate', r.vals[0] === 5)

  // range excludes the earlier occurrence -> no penalty
  r = applyDry([20], [5], [10, 11, 12, 20, 30, 40, 11, 12], o({ range: 3 }))
  check('DRY range limits the searched history', r.vals[0] === 5)

  // multiplier 0 (the off state the engine gates on): identity
  r = applyDry([30, 99], [5, 4.9], [10, 11, 12, 20, 30, 40, 11, 12], o({ multiplier: 0 }))
  check('DRY multiplier=0 is the identity', JSON.stringify(r) === JSON.stringify({ ids: [30, 99], vals: [5, 4.9] }))
}

console.log(`\n${failures === 0 ? 'ALL SAMPLER CHECKS PASSED' : failures + ' CHECK(S) FAILED'}`)
process.exit(failures === 0 ? 0 : 1)
