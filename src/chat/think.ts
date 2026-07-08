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
  private inside = false
  private hold = ''
  constructor(
    private readonly open = '<think>',
    private readonly close = '</think>',
  ) {}

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
