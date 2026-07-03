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
