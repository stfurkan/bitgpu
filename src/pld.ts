// Prompt-lookup drafting (the CPU half of speculative decoding). No draft model: the draft is
// the continuation that followed the most recent prior occurrence of the sequence's trailing
// n-gram. High hit rate whenever the model repeats sequence content (quoting RAG context,
// lists, code); zero cost when nothing matches (no draft -> a normal single-token step).

/** Return up to `maxDraft` draft tokens: the continuation after the most recent prior occurrence
 *  of the trailing g-gram, trying g = ngramSize down to 2 (1-grams draft too noisily to pay off).
 *  Empty when nothing matches. */
export function draftNgram(seq: readonly number[], ngramSize: number, maxDraft: number): number[] {
  if (maxDraft <= 0) return []
  for (let g = Math.min(ngramSize, seq.length - 1); g >= 2; g--) {
    const start = seq.length - g // the trailing g-gram lives at [start, seq.length)
    outer: for (let i = start - 1; i >= 0; i--) {
      for (let j = 0; j < g; j++) if (seq[i + j] !== seq[start + j]) continue outer
      const from = i + g // i <= start-1, so from <= seq.length-1: at least one token follows
      return seq.slice(from, Math.min(from + maxDraft, seq.length))
    }
  }
  return []
}

/** Number of tokens promptLookup:'auto' emits through the PLD path before deciding whether
 *  speculating actually pays on this content. */
export const PLD_PROBATION = 24

/** The auto-bail decision after the probation window: keep speculating only when the measured
 *  tokens-per-verify-step clears the break-even against PLAIN decoding, which differs by mode.
 *  Each PLD step costs one batched verify forward plus a per-step CPU sync; plain GREEDY chains
 *  several steps per sync (GPU-resident argmax), so its break-even is high (~2.0 tokens/step,
 *  measured), while plain SAMPLED syncs every token anyway (~1.5). Below the bar, the rest of
 *  the turn runs the plain path - output is identical either way, only speed changes. */
export function pldWorthIt(emitted: number, steps: number, sampled: boolean): boolean {
  if (steps <= 0) return false
  return emitted / steps >= (sampled ? 1.5 : 2.0)
}
