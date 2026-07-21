// Stream-safe <think> block splitting. Qwen3-family models can emit <think>...</think> reasoning
// that a chat UI must never show as the reply; the tags can straddle token boundaries, so a plain
// per-chunk string replace misses them. The splitter routes each incoming chunk into two channels
// (visible text vs think content), holding back only a possible partial tag at each chunk's edge.

/** Longest suffix of `s` that is a proper prefix of `tag` (what must be held back). */
function holdback(s: string, tag: string): number {
  const max = Math.min(s.length, tag.length - 1)
  for (let k = max; k > 0; k--) if (tag.startsWith(s.slice(s.length - k))) return s.length - k
  return s.length
}

export interface SplitChunk {
  /** Visible reply text (outside any tag pair). */
  text: string
  /** Content inside the tag pair (the model's reasoning), without the tags themselves. */
  think: string
}

export class ThinkSplitter {
  private inside: boolean
  private hold = ''
  constructor(
    private readonly open = '<think>',
    private readonly close = '</think>',
    /** Start already inside a think block - for templates whose generation prompt PRE-OPENS `<think>`
     *  (e.g. Qwen3.5 thinking mode), so the opening tag is in the prompt, not the generated stream. */
    startInside = false,
  ) {
    this.inside = startInside
  }

  push(chunk: string): SplitChunk {
    let s = this.hold + chunk
    this.hold = ''
    let text = ''
    let think = ''
    for (;;) {
      if (!this.inside) {
        const i = s.indexOf(this.open)
        if (i === -1) {
          const safe = holdback(s, this.open)
          text += s.slice(0, safe)
          this.hold = s.slice(safe)
          return { text, think }
        }
        text += s.slice(0, i)
        s = s.slice(i + this.open.length)
        this.inside = true
      } else {
        const i = s.indexOf(this.close)
        if (i === -1) {
          const safe = holdback(s, this.close)
          think += s.slice(0, safe)
          this.hold = s.slice(safe)
          return { text, think }
        }
        think += s.slice(0, i)
        s = s.slice(i + this.close.length)
        this.inside = false
      }
    }
  }

  /** Emit whatever is held back. An unterminated think block (generation hit maxTokens inside it)
   *  flushes to the think channel, never to the visible reply. */
  flush(): SplitChunk {
    const r: SplitChunk = this.inside ? { text: '', think: this.hold } : { text: this.hold, think: '' }
    this.hold = ''
    this.inside = false
    return r
  }
}

/** Stream-safe stop-sequence scanner: emits visible text up to (excluding) the earliest match of
 *  any stop string, holding back chunk-edge suffixes that could begin one (stops can straddle
 *  token boundaries). Once matched, everything further is swallowed. */
export class StopScanner {
  matched = false
  private hold = ''
  constructor(private readonly stops: readonly string[]) {}

  push(text: string): string {
    if (this.matched) return ''
    const s = this.hold + text
    let mi = -1
    for (const st of this.stops) {
      const i = s.indexOf(st)
      if (i !== -1 && (mi === -1 || i < mi)) mi = i
    }
    if (mi !== -1) {
      this.matched = true
      this.hold = ''
      return s.slice(0, mi)
    }
    let safe = s.length
    for (const st of this.stops) safe = Math.min(safe, holdback(s, st))
    this.hold = s.slice(safe)
    return s.slice(0, safe)
  }

  flush(): string {
    const r = this.matched ? '' : this.hold
    this.hold = ''
    return r
  }
}

/** Confidence-based early stop for the think channel (training-free, DART-lineage heuristic). */
export interface ThinkEarlyStop {
  /** Logit gap (top1 - top2) that counts as a "confident" step. */
  gap: number
  /** Consecutive confident steps required before firing. */
  window: number
  /** Reasoning tokens that must be spent before early stop may fire (don't cut the model off
   *  before it has actually reasoned). */
  minTokens: number
}

/** Budget-forcing for the think channel (s1-style "budget forcing", training-free): counts the
 *  tokens generated inside a <think> block; once `budget` is spent - or the EARLY-STOP heuristic
 *  fires (the model has been decisively confident for `window` consecutive steps after
 *  `minTokens`, a signature of rote continuation rather than active reasoning) - `force()` names
 *  `</think>` as the only permitted candidate (the engine's constrained pick guarantees it is
 *  reachable even when outside the top-K), so the model closes its reasoning and continues
 *  straight into the visible answer. `budget: 0` suppresses reasoning entirely while keeping the
 *  thinking-mode template. advance() must see every emitted token; observe() should see each
 *  step's candidate logits (descending) BEFORE the pick - the filter callback receives exactly
 *  that. */
export class ThinkBudget {
  private inThink: boolean
  private spent = 0
  private closed = false
  private run = 0 //         consecutive confident steps (early stop)
  private earlyFired = false
  constructor(
    private readonly openId: number | undefined,
    private readonly closeId: number | undefined,
    private readonly budget: number,
    startInside: boolean,
    private readonly early: ThinkEarlyStop | null = null,
  ) {
    this.inThink = startInside
  }

  advance(id: number): void {
    if (this.closed) return
    if (!this.inThink) {
      if (this.openId != null && id === this.openId) this.inThink = true
      return
    }
    if (this.closeId != null && id === this.closeId) {
      this.inThink = false
      this.closed = true
      return
    }
    this.spent++
  }

  /** Feed one step's candidate logits (descending). Only meaningful inside think. */
  observe(vals: ArrayLike<number>): void {
    if (!this.early || !this.inThink || this.closed || this.earlyFired || vals.length < 2) return
    const confident = vals[0] - vals[1] >= this.early.gap
    this.run = confident ? this.run + 1 : 0
    if (this.spent >= this.early.minTokens && this.run >= this.early.window) this.earlyFired = true
  }

  /** The forced token id once the budget is exhausted or early stop fired, else null. */
  force(): number | null {
    if (!this.inThink || this.closed || this.closeId == null) return null
    return this.spent >= this.budget || this.earlyFired ? this.closeId : null
  }
}
