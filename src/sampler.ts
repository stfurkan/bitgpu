// CPU side of the sampler. All the correctness-bearing probability math lives here (not in WGSL) so
// it can be bit-checked against transformers.js in Node. The GPU only pre-filters logits
// (repetition_penalty + no_repeat_ngram bans) and reduces to the top-K; this module computes the
// CPU-side inputs (deduped id set, ngram ban list) and does the final temperature + softmax +
// multinomial draw over the K candidates. Faithful port of transformers.js v4.2.0 semantics:
//   chain: repetition_penalty -> no_repeat_ngram -> temperature -> top_k -> softmax -> multinomial
//   (top_p, top_k warper, bad_words, min_length are not applied in that version).

/** Mersenne Twister 19937, matching transformers.js utils/random.js (= Python's random.Random). */
export class MT19937 {
  private mt = new Uint32Array(624)
  private idx = 625

  constructor(seed?: number) {
    this.seed(seed)
  }

  seed(n?: number): void {
    if (n === undefined || n === null) {
      const buf = new Uint32Array(1)
      crypto.getRandomValues(buf)
      n = buf[0]
    }
    const mt = this.mt
    const u = (a: number, b: number): number => Math.imul(a, b) >>> 0
    const key: number[] = []
    for (let v = n || 0; v > 0; v = Math.floor(v / 0x100000000)) key.push(v & 0xffffffff)
    if (!key.length) key.push(0)
    mt[0] = 19650218
    for (let k = 1; k < 624; ++k) mt[k] = (u(1812433253, mt[k - 1] ^ (mt[k - 1] >>> 30)) + k) >>> 0
    let i = 1
    let j = 0
    for (let k = Math.max(624, key.length); k > 0; --k, ++i, ++j) {
      if (i >= 624) {
        mt[0] = mt[623]
        i = 1
      }
      if (j >= key.length) j = 0
      mt[i] = ((mt[i] ^ u(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + key[j] + j) >>> 0
    }
    for (let k = 623; k > 0; --k, ++i) {
      if (i >= 624) {
        mt[0] = mt[623]
        i = 1
      }
      mt[i] = ((mt[i] ^ u(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0
    }
    mt[0] = 0x80000000
    this.idx = 624
  }

  private int32(): number {
    const mt = this.mt
    if (this.idx >= 624) {
      for (let k = 0; k < 624; ++k) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff)
        mt[k] = (mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0
      }
      this.idx = 0
    }
    let y = mt[this.idx++]
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18
    return y >>> 0
  }

  /** Uniform float in [0, 1), matching Python's random.random(). */
  random(): number {
    return ((this.int32() >>> 5) * 67108864.0 + (this.int32() >>> 6)) / 9007199254740992.0
  }
}

/** Deduped set of token ids that repetition_penalty applies to (the full prompt+generated history). */
export function affectedIds(history: number[]): Uint32Array {
  return Uint32Array.from(new Set(history))
}

/** no_repeat_ngram banned next-tokens for the current step. Faithful port of
 *  NoRepeatNGramLogitsProcessor.calcBannedNgramTokens (transformers.js v4.2.0). */
export function ngramBans(history: number[], n: number): number[] {
  if (history.length + 1 < n) return []
  const generated = new Map<string, number[]>()
  for (let j = 0; j < history.length + 1 - n; ++j) {
    const ngram: number[] = []
    for (let k = 0; k < n; ++k) ngram.push(history[j + k])
    const key = JSON.stringify(ngram.slice(0, n - 1))
    const arr = generated.get(key) ?? []
    arr.push(ngram[n - 1])
    generated.set(key, arr)
  }
  const idx = history.slice(history.length + 1 - n, history.length)
  return generated.get(JSON.stringify(idx)) ?? []
}

/** Options for the DRY ("don't repeat yourself") repetition penalty. */
export interface DryOpts {
  multiplier: number //   penalty strength; 0 disables
  base: number //         exponential growth per repeated token beyond allowedLength
  allowedLength: number // repeats up to this length are free
  range: number //        only the last `range` history tokens are searched (0 = all)
  breakers: Set<number> // token ids that reset matching (sequence walls); never penalized themselves
}

/** DRY repetition penalty (Kingbri/llama.cpp lineage), applied over the top-K candidates on the
 *  CPU. For each candidate c: find the longest L such that the last L history tokens followed by c
 *  reproduce an earlier stretch of the history (i.e. picking c would EXTEND an L-token repeat).
 *  If L >= allowedLength the candidate's logit drops by multiplier * base^(L - allowedLength).
 *  Matching never crosses a breaker token on either side, and breaker candidates are never
 *  penalized (structural tokens - newlines, quotes - legitimately repeat). Operating on the
 *  candidates (not the full vocab) mirrors how topP/minP are scoped in this engine. Returns the
 *  adjusted pairs re-sorted descending (stable), leaving the inputs untouched. */
export function applyDry(candIds: Uint32Array | number[], candVals: Float32Array | number[], history: number[], o: DryOpts): { ids: number[]; vals: number[] } {
  const n = history.length
  const lo = o.range > 0 ? Math.max(0, n - o.range) : 0
  const MAXL = o.allowedLength + 32 // bound the exponent; base^32 already dwarfs any logit
  const vals = Array.from(candVals) as number[]
  const ids = Array.from(candIds) as number[]
  if (n > lo && o.multiplier > 0) {
    for (let ci = 0; ci < ids.length; ci++) {
      const c = ids[ci]
      if (o.breakers.has(c)) continue
      let maxL = 0
      for (let i = lo; i < n; i++) {
        if (history[i] !== c) continue
        let l = 0 // hist[i-1-j] must equal hist[n-1-j], no breakers on either side
        while (l < MAXL && i - 1 - l >= lo && !o.breakers.has(history[i - 1 - l]) && !o.breakers.has(history[n - 1 - l]) && history[i - 1 - l] === history[n - 1 - l]) l++
        if (l > maxL) maxL = l
        if (maxL >= MAXL) break
      }
      if (maxL >= o.allowedLength) vals[ci] -= o.multiplier * Math.pow(o.base, maxL - o.allowedLength)
    }
  }
  const order = Array.from(ids.keys()).sort((a, b) => vals[b] - vals[a] || a - b)
  return { ids: order.map((i) => ids[i]), vals: order.map((i) => vals[i]) }
}

/** Stable softmax (max-subtract), matching transformers.js utils/maths.js softmax. */
function softmax(arr: Float32Array | number[]): number[] {
  let maxVal = arr[0]
  for (let i = 1; i < arr.length; ++i) if (arr[i] > maxVal) maxVal = arr[i]
  const exps = Array.from(arr, (x) => Math.exp(x - maxVal))
  let sum = 0
  for (const e of exps) sum += e
  return exps.map((x) => x / sum)
}

/** Final sampling tail. `candVals` are the K largest PENALTY-FILTERED logits (descending), `candIds`
 *  their token ids. Applies temperature to the K values, softmaxes, optionally trims the pool with
 *  top-p (nucleus) and/or min-p, then draws via the exact transformers.js inverse-CDF weighted pick
 *  (x = random()*sum; subtract until <0). `topP` (default 1 = off) keeps the shortest leading run
 *  whose cumulative probability reaches topP; `minP` (default 0 = off) keeps tokens with probability
 *  >= minP*maxProb - both operate on the already-descending candidates, so each keeps a prefix and
 *  the kept set is their shorter prefix. With both off the draw is bit-identical to the plain path
 *  (m == k, sum over all K). Returns a token id. */
export function sampleFromCandidates(candIds: Uint32Array | number[], candVals: Float32Array | number[], temperature: number, rng: MT19937, topP = 1, minP = 0): number {
  const k = candVals.length
  const tv = new Float32Array(k)
  for (let i = 0; i < k; ++i) tv[i] = candVals[i] / temperature // divide (exact for any T), matches TemperatureLogitsWarper
  const probs = softmax(tv)
  let m = k
  if (minP > 0) {
    const thresh = minP * probs[0] // probs[0] is the max (candidates descending)
    let c = 1
    while (c < k && probs[c] >= thresh) c++
    if (c < m) m = c
  }
  if (topP < 1) {
    let cum = 0
    let c = 0
    while (c < k) {
      cum += probs[c]
      c++
      if (cum >= topP) break // include the token that crosses the threshold (HF TopP shift-by-one)
    }
    if (c < m) m = c
  }
  if (m < 1) m = 1
  let sum = 0
  for (let i = 0; i < m; ++i) sum += probs[i]
  let x = rng.random() * sum
  for (let i = 0; i < m; ++i) {
    x -= probs[i]
    if (x < 0) return candIds[i]
  }
  return candIds[m - 1] // floating-point guard, matches _weightedIndexWith
}
