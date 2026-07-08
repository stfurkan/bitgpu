// Byte-level incremental JSON validator for constrained decoding (format: 'json').
//
// Operates on UTF-8 BYTES, not characters, because tokens are byte sequences: a multi-byte
// character can be split across tokens, and at the byte level that is unambiguous (multi-byte
// sequences only occur inside strings, where the machine tracks the UTF-8 sequence state).
// The machine answers two questions the candidate filter needs, incrementally and cheaply:
//   - would appending these bytes keep the text a valid JSON prefix?  (clone + feed)
//   - is the root value complete?                                     (complete)
// The root must be an object or an array (like every "JSON mode" in practice); that also makes
// completion detection unambiguous (a root number would be extendable forever).

const enum P {
  ROOT, // expecting ws | { | [
  VALUE, // expecting any value (after ':' or after ',' in an array - a close is NOT valid here)
  ARR0, // in array, right after '[': value or ']'
  KEY_OR_CLOSE, // in object, after '{': key string or '}'
  KEY, // in object, after ',': key string
  COLON, // in object, after a key string
  AFTER, // after a value: ',' | container close (or DONE at root)
  STRING, // inside a string (isKey tells where it returns to)
  STR_ESC, // after '\' in a string
  STR_U, // inside \uXXXX (uLeft hex digits remaining)
  NUMBER, // inside a number (numSub tracks the numeric sub-state)
  LITERAL, // inside true/false/null (litWord/litPos)
  DONE, // root value complete: whitespace only
}

const enum N {
  INT0, // after '-' : expecting first integer digit
  INTZ, // after leading '0' : only . e E or end
  INT, // in integer digits
  FRAC0, // after '.' : expecting first fraction digit
  FRAC, // in fraction digits
  EXPS, // after e/E : expecting sign or digit
  EXP0, // after exponent sign : expecting first digit
  EXP, // in exponent digits
}

const WS = new Set([0x20, 0x09, 0x0a, 0x0d])
const isDigit = (b: number): boolean => b >= 0x30 && b <= 0x39
const isHex = (b: number): boolean => isDigit(b) || (b >= 0x41 && b <= 0x46) || (b >= 0x61 && b <= 0x66)
// A number can legally END at ws , ] } - the terminating byte is re-dispatched to AFTER.
const numberDone = (s: N): boolean => s === N.INTZ || s === N.INT || s === N.FRAC || s === N.EXP

export class JsonMachine {
  private stack: Array<'{' | '['> = []
  private phase: P = P.ROOT
  private isKey = false
  private uLeft = 0
  private utf8Left = 0 // pending UTF-8 continuation bytes inside a string
  private numSub: N = N.INT
  private litWord = ''
  private litPos = 0

  clone(): JsonMachine {
    const m = new JsonMachine()
    m.stack = this.stack.slice()
    m.phase = this.phase
    m.isKey = this.isKey
    m.uLeft = this.uLeft
    m.utf8Left = this.utf8Left
    m.numSub = this.numSub
    m.litWord = this.litWord
    m.litPos = this.litPos
    return m
  }

  get complete(): boolean {
    return this.phase === P.DONE
  }

  /** Feed bytes; false = the text stopped being a valid JSON prefix (state is then undefined). */
  feed(bytes: Uint8Array): boolean {
    for (let i = 0; i < bytes.length; i++) if (!this.byte(bytes[i])) return false
    return true
  }

  private openValue(b: number): boolean {
    // dispatch a value's FIRST byte (phase VALUE, or ROOT restricted to containers)
    if (b === 0x7b) {
      this.stack.push('{')
      this.phase = P.KEY_OR_CLOSE
    } else if (b === 0x5b) {
      this.stack.push('[')
      this.phase = P.ARR0
    } else if (b === 0x22) {
      this.isKey = false
      this.phase = P.STRING
    } else if (b === 0x2d || isDigit(b)) {
      this.numSub = b === 0x2d ? N.INT0 : b === 0x30 ? N.INTZ : N.INT
      this.phase = P.NUMBER
    } else if (b === 0x74 || b === 0x66 || b === 0x6e) {
      this.litWord = b === 0x74 ? 'true' : b === 0x66 ? 'false' : 'null'
      this.litPos = 1
      this.phase = P.LITERAL
    } else return false
    return true
  }

  private closeValue(): void {
    // a value just finished; where we land depends on the enclosing container
    this.phase = this.stack.length === 0 ? P.DONE : P.AFTER
  }

  private closeContainer(b: number): boolean {
    const want = b === 0x7d ? '{' : '['
    if (this.stack.pop() !== want) return false
    this.closeValue()
    return true
  }

  private byte(b: number): boolean {
    switch (this.phase) {
      case P.ROOT:
        if (WS.has(b)) return true
        if (b === 0x7b || b === 0x5b) return this.openValue(b)
        return false // JSON mode requires an object or array root
      case P.VALUE:
        if (WS.has(b)) return true
        return this.openValue(b)
      case P.ARR0:
        if (WS.has(b)) return true
        if (b === 0x5d) return this.closeContainer(b) // empty array
        return this.openValue(b)
      case P.KEY_OR_CLOSE:
        if (WS.has(b)) return true
        if (b === 0x7d) return this.closeContainer(b)
        if (b === 0x22) {
          this.isKey = true
          this.phase = P.STRING
          return true
        }
        return false
      case P.KEY:
        if (WS.has(b)) return true
        if (b === 0x22) {
          this.isKey = true
          this.phase = P.STRING
          return true
        }
        return false
      case P.COLON:
        if (WS.has(b)) return true
        if (b === 0x3a) {
          this.phase = P.VALUE
          return true
        }
        return false
      case P.AFTER: {
        if (WS.has(b)) return true
        const top = this.stack[this.stack.length - 1]
        if (b === 0x2c) {
          this.phase = top === '{' ? P.KEY : P.VALUE
          return true
        }
        if ((b === 0x7d && top === '{') || (b === 0x5d && top === '[')) return this.closeContainer(b)
        return false
      }
      case P.STRING:
        if (this.utf8Left > 0) {
          if (b >= 0x80 && b <= 0xbf) {
            this.utf8Left--
            return true
          }
          return false
        }
        if (b === 0x22) {
          if (this.isKey) {
            this.isKey = false
            this.phase = P.COLON
          } else this.closeValue()
          return true
        }
        if (b === 0x5c) {
          this.phase = P.STR_ESC
          return true
        }
        if (b < 0x20) return false // raw control bytes must be escaped
        if (b < 0x80) return true
        // UTF-8 lead byte: enforce well-formed sequences (no mojibake in constrained output)
        if (b >= 0xc2 && b <= 0xdf) this.utf8Left = 1
        else if (b >= 0xe0 && b <= 0xef) this.utf8Left = 2
        else if (b >= 0xf0 && b <= 0xf4) this.utf8Left = 3
        else return false // 0x80-0xC1 stray continuation / overlong, 0xF5+ invalid
        return true
      case P.STR_ESC:
        if (b === 0x75) {
          this.uLeft = 4
          this.phase = P.STR_U
          return true
        }
        if ([0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(b)) {
          this.phase = P.STRING
          return true
        }
        return false
      case P.STR_U:
        if (!isHex(b)) return false
        if (--this.uLeft === 0) this.phase = P.STRING
        return true
      case P.NUMBER:
        switch (this.numSub) {
          case N.INT0:
            if (!isDigit(b)) return false
            this.numSub = b === 0x30 ? N.INTZ : N.INT
            return true
          case N.INTZ:
          case N.INT:
            if (isDigit(b)) {
              if (this.numSub === N.INTZ) return false // no leading zeros
              return true
            }
            break
          case N.FRAC0:
            if (!isDigit(b)) return false
            this.numSub = N.FRAC
            return true
          case N.FRAC:
            if (isDigit(b)) return true
            break
          case N.EXPS:
            if (b === 0x2b || b === 0x2d) {
              this.numSub = N.EXP0
              return true
            }
            if (!isDigit(b)) return false
            this.numSub = N.EXP
            return true
          case N.EXP0:
            if (!isDigit(b)) return false
            this.numSub = N.EXP
            return true
          case N.EXP:
            if (isDigit(b)) return true
            break
        }
        // shared transitions out of digit runs
        if (b === 0x2e && (this.numSub === N.INTZ || this.numSub === N.INT)) {
          this.numSub = N.FRAC0
          return true
        }
        if ((b === 0x65 || b === 0x45) && numberDone(this.numSub) && this.numSub !== N.EXP) {
          this.numSub = N.EXPS
          return true
        }
        if (numberDone(this.numSub)) {
          this.closeValue() // the number ended; re-dispatch this byte in AFTER
          return this.byte(b)
        }
        return false
      case P.LITERAL:
        if (b !== this.litWord.charCodeAt(this.litPos)) return false
        if (++this.litPos === this.litWord.length) this.closeValue()
        return true
      case P.DONE:
        return WS.has(b) // only trailing whitespace after the root closes
    }
  }
}

// ── token bytes ──────────────────────────────────────────────────────────────
// Byte-level BPE vocabularies store each token as a string of BYTE ALIASES (the GPT-2
// bytes_to_unicode mapping). A token's raw bytes are therefore context-free: invert the alias.
function aliasToByte(): Map<number, number> {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  return new Map(bs.map((b, i) => [cs[i], b]))
}

export interface TokenBytesSource {
  idToToken(id: number): string | undefined
  addedTokenIds(): Set<number>
}

/** Precomputed id -> raw bytes lookup (lazy per id; added/special tokens map to null). */
export class TokenByteTable {
  private readonly cache = new Map<number, Uint8Array | null>()
  private readonly inv = aliasToByte()
  private readonly added: Set<number>
  constructor(private readonly tk: TokenBytesSource) {
    this.added = tk.addedTokenIds()
  }
  bytes(id: number): Uint8Array | null {
    const hit = this.cache.get(id)
    if (hit !== undefined) return hit
    let out: Uint8Array | null = null
    if (!this.added.has(id)) {
      const s = this.tk.idToToken(id)
      if (s !== undefined) {
        const b = new Uint8Array(s.length)
        let ok = true
        for (let i = 0; i < s.length; i++) {
          const v = this.inv.get(s.charCodeAt(i))
          if (v === undefined) {
            ok = false
            break
          }
          b[i] = v
        }
        out = ok ? b : null
      }
    }
    this.cache.set(id, out)
    return out
  }
}

/** The per-step candidate filter for format:'json': permit candidates whose bytes keep the text
 *  a valid JSON prefix; once the root value is complete, permit ONLY eos so generation ends
 *  naturally. Call advance() with each chosen token to move the real machine forward. */
export function makeJsonFilter(table: TokenByteTable, eosTokenId: number): {
  filter: (ids: Uint32Array | number[]) => number[]
  advance: (id: number) => void
  readonly machine: JsonMachine
} {
  const machine = new JsonMachine()
  return {
    machine,
    filter: (ids: Uint32Array | number[]): number[] => {
      const out: number[] = []
      for (const id of ids) {
        if (machine.complete) {
          if (id === eosTokenId) out.push(Number(id))
          continue
        }
        const bytes = table.bytes(Number(id))
        if (!bytes || bytes.length === 0) continue // special/added tokens are never JSON content
        if (machine.clone().feed(bytes)) out.push(Number(id))
      }
      return out
    },
    advance: (id: number): void => {
      if (machine.complete) return // the eos that ends the turn advances nothing
      const bytes = table.bytes(id)
      if (bytes) machine.feed(bytes)
    },
  }
}
