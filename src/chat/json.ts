// Byte-level incremental JSON validator + schema enforcer for constrained decoding
// (format: 'json' and format: { json: { schema } }).
//
// Operates on UTF-8 BYTES, not characters, because tokens are byte sequences: a multi-byte
// character can be split across tokens, and at the byte level that is unambiguous (multi-byte
// sequences only occur inside strings, where the machine tracks the UTF-8 sequence state).
// The machine answers two questions the candidate filter needs, incrementally and cheaply:
//   - would appending these bytes keep the text a valid (schema-conforming) JSON prefix?
//   - is the root value complete?
// The root must be an object or an array (like every "JSON mode" in practice); that also makes
// completion detection unambiguous (a root number would be extendable forever).
//
// The schema SUBSET is enforced token-by-token, so a conforming document is the only thing the
// model can produce: value types, object required keys + additionalProperties: false (key names
// become a prefix-constrained choice set), array minItems/maxItems (closing early or adding past
// the cap is filtered out), string enums (byte-prefix matching against the literals), integer
// (no fraction/exponent), nested to any depth. validateJsonSchema THROWS on anything outside the
// subset - silent partial enforcement would be false confidence.

/** The enforceable JSON Schema subset (see validateJsonSchema). */
export interface JsonSchema {
  type?: 'object' | 'array' | 'string' | 'number' | 'integer' | 'boolean' | 'null'
  properties?: Record<string, JsonSchema>
  required?: string[]
  additionalProperties?: boolean
  items?: JsonSchema
  minItems?: number
  maxItems?: number
  enum?: string[]
}

const SUPPORTED = new Set(['type', 'properties', 'required', 'additionalProperties', 'items', 'minItems', 'maxItems', 'enum'])
const TYPES = new Set(['object', 'array', 'string', 'number', 'integer', 'boolean', 'null'])

/** Validate a schema against the enforceable subset; throws listing anything unsupported. */
export function validateJsonSchema(schema: JsonSchema, path = 'schema', isRoot = true): void {
  const unknown = Object.keys(schema).filter((k) => !SUPPORTED.has(k))
  if (unknown.length) throw new Error(`bitgpu/chat: unsupported JSON Schema keyword(s) at ${path}: ${unknown.join(', ')} (enforceable subset: ${[...SUPPORTED].join(', ')})`)
  if (schema.type !== undefined && !TYPES.has(schema.type)) throw new Error(`bitgpu/chat: unsupported type '${schema.type}' at ${path}`)
  if (isRoot && schema.type !== undefined && schema.type !== 'object' && schema.type !== 'array')
    throw new Error(`bitgpu/chat: the schema root must be an object or array (got '${schema.type}'); JSON mode requires a container root`)
  if (schema.enum !== undefined) {
    if (!Array.isArray(schema.enum) || schema.enum.length === 0 || !schema.enum.every((v) => typeof v === 'string'))
      throw new Error(`bitgpu/chat: enum at ${path} must be a non-empty array of strings`)
    for (const v of schema.enum)
      if (/["\\\u0000-\u001f]/.test(v)) throw new Error(`bitgpu/chat: enum value ${JSON.stringify(v)} at ${path} contains characters that require JSON escaping (unsupported)`)
    if (schema.type !== undefined && schema.type !== 'string') throw new Error(`bitgpu/chat: enum at ${path} requires type 'string'`)
  }
  if (schema.required !== undefined && schema.properties !== undefined)
    for (const k of schema.required) if (!(k in schema.properties)) throw new Error(`bitgpu/chat: required key '${k}' at ${path} is missing from properties`)
  if (schema.additionalProperties === false && schema.properties === undefined)
    throw new Error(`bitgpu/chat: additionalProperties: false at ${path} needs a properties map`)
  if (schema.properties !== undefined)
    for (const [k, v] of Object.entries(schema.properties)) {
      if (/["\\\u0000-\u001f]/.test(k)) throw new Error(`bitgpu/chat: property name ${JSON.stringify(k)} at ${path} contains characters that require JSON escaping (unsupported)`)
      validateJsonSchema(v, `${path}.properties.${k}`, false)
    }
  if (schema.items !== undefined) validateJsonSchema(schema.items, `${path}.items`, false)
  if (schema.minItems !== undefined && schema.maxItems !== undefined && schema.minItems > schema.maxItems)
    throw new Error(`bitgpu/chat: minItems > maxItems at ${path}`)
}

const enum P {
  ROOT, // expecting ws | { | [
  VALUE, // expecting a value (after ':' or after ',' in an array - a close is NOT valid here)
  ARR0, // in array, right after '[': value or ']'
  KEY_OR_CLOSE, // in object, after '{': key string or '}'
  KEY, // in object, after ',': key string
  COLON, // in object, after a key string
  AFTER, // after a value: ',' | container close (or DONE at root)
  STRING, // inside a string (strKind tells key / plain value / enum value)
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

/** Per-open-container schema context (parallel to the container stack). */
interface Ctx {
  kind: '{' | '['
  node: JsonSchema | null // the schema governing this container (null = unconstrained)
  seen: string[] // objects: property names already committed
  count: number // arrays: completed items
}

const utf8 = new TextEncoder()

export class JsonMachine {
  private stack: Ctx[] = []
  private phase: P = P.ROOT
  private uLeft = 0
  private utf8Left = 0 // pending UTF-8 continuation bytes inside a string
  private numSub: N = N.INT
  private numInt = false // integer schema: fraction/exponent banned
  private litWord = ''
  private litPos = 0
  private strKind: 'key' | 'value' | 'enum' = 'value'
  private strBuf = '' // raw bytes (as charcodes) of the current key / enum string
  private strTracking = false // whether strBuf is being accumulated
  private enumCands: string[] | null = null // byte-space enum literals for the current string
  private pending: JsonSchema | null = null // schema of the value about to open
  private wsRun = 0 // consecutive STRUCTURAL whitespace bytes (between JSON tokens)

  constructor(private readonly root: JsonSchema | null = null) {
    this.pending = root
  }

  clone(): JsonMachine {
    const m = new JsonMachine(this.root)
    m.stack = this.stack.map((c) => ({ kind: c.kind, node: c.node, seen: c.seen.slice(), count: c.count }))
    m.phase = this.phase
    m.uLeft = this.uLeft
    m.utf8Left = this.utf8Left
    m.numSub = this.numSub
    m.numInt = this.numInt
    m.litWord = this.litWord
    m.litPos = this.litPos
    m.strKind = this.strKind
    m.strBuf = this.strBuf
    m.strTracking = this.strTracking
    m.enumCands = this.enumCands
    m.pending = this.pending
    m.wsRun = this.wsRun
    return m
  }

  get complete(): boolean {
    return this.phase === P.DONE
  }

  /** Feed bytes; false = the text stopped being a valid schema-conforming prefix (state is then undefined). */
  feed(bytes: Uint8Array): boolean {
    for (let i = 0; i < bytes.length; i++) if (!this.byte(bytes[i])) return false
    return true
  }

  private top(): Ctx | undefined {
    return this.stack[this.stack.length - 1]
  }

  /** Byte-space form of a string literal (what strBuf accumulates). */
  private static bytesOf(s: string): string {
    return String.fromCharCode(...utf8.encode(s))
  }

  private openValue(b: number): boolean {
    const sc = this.pending
    const t = sc?.type
    if (b === 0x7b) {
      if (t !== undefined && t !== 'object') return false
      this.stack.push({ kind: '{', node: sc ?? null, seen: [], count: 0 })
      this.phase = P.KEY_OR_CLOSE
    } else if (b === 0x5b) {
      if (t !== undefined && t !== 'array') return false
      this.stack.push({ kind: '[', node: sc ?? null, seen: [], count: 0 })
      this.phase = P.ARR0
    } else if (b === 0x22) {
      if (t !== undefined && t !== 'string') return false
      this.strKind = sc?.enum ? 'enum' : 'value'
      this.enumCands = sc?.enum ? sc.enum.map(JsonMachine.bytesOf) : null
      this.strBuf = ''
      this.strTracking = this.strKind === 'enum'
      this.phase = P.STRING
    } else if (b === 0x2d || isDigit(b)) {
      if (t !== undefined && t !== 'number' && t !== 'integer') return false
      this.numInt = t === 'integer'
      this.numSub = b === 0x2d ? N.INT0 : b === 0x30 ? N.INTZ : N.INT
      this.phase = P.NUMBER
    } else if (b === 0x74 || b === 0x66 || b === 0x6e) {
      const word = b === 0x74 ? 'true' : b === 0x66 ? 'false' : 'null'
      if (t !== undefined && !((word === 'null' && t === 'null') || (word !== 'null' && t === 'boolean'))) return false
      this.litWord = word
      this.litPos = 1
      this.phase = P.LITERAL
    } else return false
    this.pending = null
    return true
  }

  /** A value just finished; land per the enclosing container and count array items. */
  private closeValue(): void {
    const c = this.top()
    if (c === undefined) {
      this.phase = P.DONE
      return
    }
    if (c.kind === '[') c.count++
    this.phase = P.AFTER
  }

  private closeContainer(b: number): boolean {
    const c = this.top()
    if (!c || c.kind !== (b === 0x7d ? '{' : '[')) return false
    if (c.kind === '{' && c.node?.required) for (const k of c.node.required) if (!c.seen.includes(k)) return false
    if (c.kind === '[' && c.node?.minItems !== undefined && c.count < c.node.minItems) return false
    this.stack.pop()
    this.closeValue()
    return true
  }

  /** Candidate property names for the key being typed (null = unconstrained). */
  private keyCands(c: Ctx): string[] | null {
    if (!c.node || c.node.additionalProperties !== false) return null
    return Object.keys(c.node.properties ?? {}).filter((k) => !c.seen.includes(k)).map(JsonMachine.bytesOf)
  }

  private startKey(c: Ctx): void {
    this.strKind = 'key'
    this.enumCands = this.keyCands(c)
    this.strBuf = ''
    this.strTracking = true // keys are always tracked (to select the property schema)
    this.phase = P.STRING
  }

  /** Structural whitespace is never REQUIRED by JSON, so capping a run cannot make the grammar
   *  unsatisfiable - but without a cap, a model denied prose can loop on whitespace forever (the
   *  grammar permits it unboundedly) and burn the whole token budget producing "[ ". 16 bytes
   *  allows generous pretty-printing indentation while forcing real progress. */
  private ws(): boolean {
    return ++this.wsRun <= 16
  }

  private byte(b: number): boolean {
    if (!WS.has(b)) this.wsRun = 0 // any non-whitespace byte ends the run
    switch (this.phase) {
      case P.ROOT:
        if (WS.has(b)) return this.ws()
        if (b === 0x7b || b === 0x5b) return this.openValue(b)
        return false // JSON mode requires an object or array root
      case P.VALUE:
        if (WS.has(b)) return this.ws()
        return this.openValue(b)
      case P.ARR0: {
        if (WS.has(b)) return this.ws()
        if (b === 0x5d) return this.closeContainer(b) // empty array (minItems permitting)
        const c = this.top() as Ctx
        if (c.node?.maxItems === 0) return false // array must stay empty: only ']' is valid
        this.pending = c.node?.items ?? null
        return this.openValue(b)
      }
      case P.KEY_OR_CLOSE: {
        if (WS.has(b)) return this.ws()
        if (b === 0x7d) return this.closeContainer(b)
        if (b === 0x22) {
          const c = this.top() as Ctx
          if (this.keyCands(c)?.length === 0) return false // no keys left to open
          this.startKey(c)
          return true
        }
        return false
      }
      case P.KEY:
        if (WS.has(b)) return this.ws()
        if (b === 0x22) {
          this.startKey(this.top() as Ctx)
          return true
        }
        return false
      case P.COLON:
        if (WS.has(b)) return this.ws()
        if (b === 0x3a) {
          const c = this.top() as Ctx
          const key = c.seen[c.seen.length - 1]
          // byte-space key -> original property name (properties are validated escape-free)
          const prop = c.node?.properties ? Object.entries(c.node.properties).find(([k]) => JsonMachine.bytesOf(k) === key) : undefined
          this.pending = prop ? prop[1] : null
          this.phase = P.VALUE
          return true
        }
        return false
      case P.AFTER: {
        if (WS.has(b)) return this.ws()
        const c = this.top()
        if (b === 0x2c && c) {
          if (c.kind === '{') {
            if (this.keyCands(c)?.length === 0) return false // every allowed key present: only '}' remains
            this.phase = P.KEY
          } else {
            if (c.node?.maxItems !== undefined && c.count >= c.node.maxItems) return false // array full: only ']' remains
            this.pending = c.node?.items ?? null
            this.phase = P.VALUE
          }
          return true
        }
        if ((b === 0x7d || b === 0x5d) && c) return this.closeContainer(b)
        return false
      }
      case P.STRING:
        if (this.utf8Left > 0) {
          if (b >= 0x80 && b <= 0xbf) {
            this.utf8Left--
            if (this.strTracking) this.strBuf += String.fromCharCode(b)
            return this.enumOk()
          }
          return false
        }
        if (b === 0x22) {
          if (this.strKind === 'key') {
            const c = this.top() as Ctx
            if (this.enumCands !== null && !this.enumCands.includes(this.strBuf)) return false // key must be a remaining allowed property
            if (c.node?.properties && c.seen.includes(this.strBuf)) return false // no duplicate tracked keys
            c.seen.push(this.strBuf)
            this.phase = P.COLON
          } else {
            if (this.strKind === 'enum' && !(this.enumCands as string[]).includes(this.strBuf)) return false // must be a complete literal
            this.closeValue()
          }
          this.strTracking = false
          return true
        }
        if (b === 0x5c) {
          if (this.strKind === 'enum' || (this.strKind === 'key' && this.enumCands !== null)) return false // constrained strings are escape-free
          this.phase = P.STR_ESC
          return true
        }
        if (b < 0x20) return false // raw control bytes must be escaped
        if (b < 0x80) {
          if (this.strTracking) this.strBuf += String.fromCharCode(b)
          return this.enumOk()
        }
        // UTF-8 lead byte: enforce well-formed sequences (no mojibake in constrained output)
        if (b >= 0xc2 && b <= 0xdf) this.utf8Left = 1
        else if (b >= 0xe0 && b <= 0xef) this.utf8Left = 2
        else if (b >= 0xf0 && b <= 0xf4) this.utf8Left = 3
        else return false // 0x80-0xC1 stray continuation / overlong, 0xF5+ invalid
        if (this.strTracking) this.strBuf += String.fromCharCode(b)
        return this.enumOk()
      case P.STR_ESC:
        if (b === 0x75) {
          this.uLeft = 4
          this.phase = P.STR_U
          return true
        }
        if ([0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].includes(b)) {
          if (this.strTracking) this.strBuf += '\\' + String.fromCharCode(b) // keys with escapes never match a property (validated escape-free)
          this.phase = P.STRING
          return true
        }
        return false
      case P.STR_U:
        if (!isHex(b)) return false
        if (this.strTracking) this.strBuf += String.fromCharCode(b)
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
        // shared transitions out of digit runs (banned for type 'integer')
        if (b === 0x2e && !this.numInt && (this.numSub === N.INTZ || this.numSub === N.INT)) {
          this.numSub = N.FRAC0
          return true
        }
        if ((b === 0x65 || b === 0x45) && !this.numInt && numberDone(this.numSub) && this.numSub !== N.EXP) {
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

  /** In an enum/key-constrained string, the accumulated bytes must remain a prefix of some candidate. */
  private enumOk(): boolean {
    if (!this.strTracking || this.enumCands === null) return true
    const buf = this.strBuf
    for (const c of this.enumCands) if (c.startsWith(buf)) return true
    return false
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
 *  a valid, schema-conforming JSON prefix; once the root value is complete, permit ONLY eos so
 *  generation ends naturally. Call advance() with each chosen token to move the real machine. */
export function makeJsonFilter(table: TokenByteTable, eosTokenId: number, schema: JsonSchema | null = null): {
  filter: (ids: Uint32Array | number[]) => number[]
  advance: (id: number) => void
  readonly machine: JsonMachine
} {
  const machine = new JsonMachine(schema)
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
