// Tool calling for bitgpu/chat, implementing the model's OWN protocol (the Qwen3-family chat
// template): a `tools` list rendered into the system block by the template itself, and
// <tool_call>{"name": ..., "arguments": ...}</tool_call> blocks in the assistant reply. This
// module is deliberately an ENGINE feature, not an agent framework: it renders, extracts,
// validates, and (via the candidate filter) ENFORCES tool calls - executing them, looping, and
// retrying stay with the app.
//
// Enforcement rides the same byte-level machinery as format:'json': the <tool_call> / </tool_call>
// markers are single added tokens in the vocabulary (checked at prepare time), so the filter
// tracks them by token id, and between them the body is forced byte-by-byte to the canonical
// trained shape `{"name": "<a declared tool>", "arguments": <that tool's parameters schema>}`.
// A bitgpu tool call therefore cannot be malformed: the name is always one of the declared tools
// and the arguments always conform to that tool's schema.
import { JsonMachine, validateJsonSchema, type JsonSchema, type TokenByteTable } from './json'

/** A tool declaration, in the shape Qwen-family models were trained on (and the shape the
 *  OpenAI/HF ecosystems use); the template serializes it verbatim into the system block. */
export interface ChatTool {
  type: 'function'
  function: {
    name: string
    description?: string
    /** JSON Schema for the arguments object - the same enforceable SUBSET as
     *  format: { json: { schema } }, and the root must be an object. When omitted, the
     *  arguments are only forced to be a valid JSON object. */
    parameters?: JsonSchema
  }
}

/** One parsed tool call from the reply. With tools enabled the arguments are grammar-enforced,
 *  so `JSON.parse` can never have failed; `raw` keeps the exact block text for forensics (and is
 *  the only place to look if a block was cut short by maxTokens - then `name` is ''). */
export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  /** The exact text between <tool_call> and </tool_call>. */
  raw: string
}

/** Which tool the model must (or may) call. 'auto' (the default when tools are present) lets the
 *  model decide; { name } FORCES a call to that tool as the entire reply - the forced path is
 *  fully enforced end to end and is the reliable way to use small models. 'none' ignores the
 *  tools for this turn. */
export type ToolChoice = 'auto' | 'none' | { name: string }

const BAD_NAME = /["\\\u0000-\u001f]/

/** Validate a tools list + choice; throws on anything the enforcer cannot guarantee. */
export function validateTools(tools: readonly ChatTool[], choice: ToolChoice): void {
  if (tools.length === 0) throw new Error('bitgpu/chat: tools is empty')
  const seen = new Set<string>()
  for (let i = 0; i < tools.length; i++) {
    const t = tools[i]
    if (t?.type !== 'function' || typeof t.function?.name !== 'string' || t.function.name.length === 0)
      throw new Error(`bitgpu/chat: tools[${i}] must be { type: 'function', function: { name, ... } }`)
    const name = t.function.name
    if (BAD_NAME.test(name)) throw new Error(`bitgpu/chat: tool name ${JSON.stringify(name)} contains characters that require JSON escaping (unsupported)`)
    if (seen.has(name)) throw new Error(`bitgpu/chat: duplicate tool name '${name}'`)
    seen.add(name)
    const p = t.function.parameters
    if (p !== undefined) {
      if (p.type !== undefined && p.type !== 'object')
        throw new Error(`bitgpu/chat: tools[${i}].function.parameters must describe an object (got type '${p.type}')`)
      validateJsonSchema(p.type === undefined ? { ...p, type: 'object' } : p, `tools[${i}].function.parameters`, true)
    }
  }
  if (typeof choice === 'object' && !seen.has(choice.name))
    throw new Error(`bitgpu/chat: toolChoice names unknown tool '${choice.name}'`)
}

/** The arguments schema the enforcer uses for a tool (parameters, defaulted to "any object"). */
export function argsSchemaOf(t: ChatTool): JsonSchema {
  const p = t.function.parameters
  if (p === undefined) return { type: 'object' }
  return p.type === undefined ? { ...p, type: 'object' } : p
}

// ── the call-body grammar ────────────────────────────────────────────────────
// Byte machine for the text between <tool_call> and </tool_call>, strict to the canonical
// trained shape (the filter FORCES generation onto it, so strictness costs nothing):
//   ws* { "name": " <name> ", "arguments": <args per schema> } ws*
// with the literal spacing the template's own example uses. Name bytes are prefix-constrained
// against the declared tool names (like string enums); the closing quote commits the name and
// selects that tool's arguments schema.

const utf8 = new TextEncoder()
const LIT1 = utf8.encode('{"name": "')
const LIT2 = utf8.encode('", "arguments": ')
const WS = new Set([0x20, 0x09, 0x0a, 0x0d])
const WS_CAP = 4 // canonical is a single '\n' on each side; a few extra never hurt

const enum B {
  PRE, // optional ws, then LIT1
  LIT1,
  NAME,
  LIT2,
  ARGS,
  POST, // after args complete: '}' then optional ws; complete once '}' lands
}

export class ToolBodyMachine {
  private phase: B = B.PRE
  private lit = 0 // position inside LIT1/LIT2
  private nameBuf = ''
  private args: JsonMachine | null = null
  private wsRun = 0
  /** true once the closing '}' has landed (only trailing ws may follow). */
  complete = false
  /** the committed tool name (set when its closing quote lands). */
  name = ''

  /** byte-space tool names, and each name's arguments schema */
  constructor(private readonly cands: readonly string[], private readonly schemas: ReadonlyMap<string, JsonSchema>) {}

  clone(): ToolBodyMachine {
    const m = new ToolBodyMachine(this.cands, this.schemas)
    m.phase = this.phase
    m.lit = this.lit
    m.nameBuf = this.nameBuf
    m.args = this.args ? this.args.clone() : null
    m.wsRun = this.wsRun
    m.complete = this.complete
    m.name = this.name
    return m
  }

  static bytesOf(s: string): string {
    return String.fromCharCode(...utf8.encode(s))
  }

  feed(bytes: Uint8Array): boolean {
    for (let i = 0; i < bytes.length; i++) if (!this.byte(bytes[i])) return false
    return true
  }

  private byte(b: number): boolean {
    switch (this.phase) {
      case B.PRE:
        if (WS.has(b)) return ++this.wsRun <= WS_CAP
        if (b !== LIT1[0]) return false
        this.phase = B.LIT1
        this.lit = 1
        return true
      case B.LIT1:
        if (b !== LIT1[this.lit]) return false
        if (++this.lit === LIT1.length) {
          this.phase = B.NAME
          this.nameBuf = ''
        }
        return true
      case B.NAME:
        if (b === 0x22) {
          // closing quote: the name must be EXACTLY one of the declared tools
          if (!this.cands.includes(this.nameBuf)) return false
          this.name = this.nameBuf
          this.phase = B.LIT2
          this.lit = 1 // LIT2[0] IS this quote
          return true
        }
        this.nameBuf += String.fromCharCode(b)
        for (const c of this.cands) if (c.startsWith(this.nameBuf)) return true
        return false
      case B.LIT2:
        if (b !== LIT2[this.lit]) return false
        if (++this.lit === LIT2.length) {
          this.phase = B.ARGS
          this.args = new JsonMachine(this.schemas.get(this.name) ?? { type: 'object' })
        }
        return true
      case B.ARGS: {
        const a = this.args as JsonMachine
        if (a.complete) {
          // the arguments value closed; only ws then the wrapper's '}' may follow
          if (WS.has(b)) return ++this.wsRun <= WS_CAP
          if (b !== 0x7d) return false
          this.wsRun = 0
          this.phase = B.POST
          this.complete = true
          return true
        }
        return a.feed(Uint8Array.of(b))
      }
      case B.POST:
        return WS.has(b) && ++this.wsRun <= WS_CAP
    }
  }
}

// ── the candidate filter ─────────────────────────────────────────────────────

export interface ToolMarkerIds {
  open: number // <tool_call>
  close: number // </tool_call>
  eos: number
  thinkOpen?: number // <think> - tool markers inside a think block are the model musing, not a call
  thinkClose?: number
}

export interface PreparedTools {
  tools: readonly ChatTool[]
  ids: ToolMarkerIds
  /** null = 'auto'; a name = forced single call */
  forced: string | null
}

/** Per-step candidate filter for tool turns. Auto mode: free text (everything permitted) until
 *  the model opens <tool_call>, then the body grammar takes over until </tool_call>, then free
 *  text again (another call, prose, or eos). Forced mode: the FIRST token must be <tool_call>,
 *  the body is constrained to the named tool, and after </tool_call> only eos is permitted.
 *  Call advance() with each emitted token to move the real machine. */
export function makeToolFilter(table: TokenByteTable, prep: PreparedTools): {
  filter: (ids: Uint32Array | number[]) => number[]
  advance: (id: number) => void
} {
  const { ids, forced } = prep
  const names = (forced ? prep.tools.filter((t) => t.function.name === forced) : prep.tools).map((t) => t.function.name)
  const cands = names.map(ToolBodyMachine.bytesOf)
  const schemas = new Map(prep.tools.map((t) => [ToolBodyMachine.bytesOf(t.function.name), argsSchemaOf(t)]))
  const enum S {
    TEXT, // auto mode outside a call: everything permitted
    OPEN, // forced mode start: only <tool_call>
    BODY,
    EOS, // forced mode after the call: only eos
  }
  let state: S = forced ? S.OPEN : S.TEXT
  let body: ToolBodyMachine | null = null
  let inThink = false
  return {
    filter: (candidates: Uint32Array | number[]): number[] => {
      if (state === S.TEXT || inThink) return Array.from(candidates, Number)
      const out: number[] = []
      for (const id of candidates) {
        const n = Number(id)
        if (state === S.OPEN) {
          if (n === ids.open) out.push(n)
          continue
        }
        if (state === S.EOS) {
          if (n === ids.eos) out.push(n)
          continue
        }
        // BODY: the close marker only once the wrapper object is complete; otherwise bytes that
        // keep the body a valid prefix (added/special tokens have no bytes and are never body)
        const m = body as ToolBodyMachine
        if (n === ids.close) {
          if (m.complete) out.push(n)
          continue
        }
        const bytes = table.bytes(n)
        if (!bytes || bytes.length === 0) continue
        if (m.clone().feed(bytes)) out.push(n)
      }
      return out
    },
    advance: (id: number): void => {
      if (id === ids.thinkOpen) {
        inThink = true
        return
      }
      if (id === ids.thinkClose) {
        inThink = false
        return
      }
      if (inThink) return
      switch (state) {
        case S.TEXT:
        case S.OPEN:
          if (id === ids.open) {
            state = S.BODY
            body = new ToolBodyMachine(cands, schemas)
          }
          return
        case S.BODY:
          if (id === ids.close) {
            state = forced ? S.EOS : S.TEXT
            body = null
            return
          }
          {
            const bytes = table.bytes(id)
            if (bytes) (body as ToolBodyMachine).feed(bytes)
          }
          return
        case S.EOS:
          return
      }
    },
  }
}

// ── extraction ───────────────────────────────────────────────────────────────

/** Longest suffix of `s` that is a proper prefix of `tag` (what must be held back). */
function holdback(s: string, tag: string): number {
  const max = Math.min(s.length, tag.length - 1)
  for (let k = max; k > 0; k--) if (tag.startsWith(s.slice(s.length - k))) return s.length - k
  return s.length
}

/** Stream-safe <tool_call> block extraction (the tool sibling of ThinkSplitter): visible text on
 *  one channel, each COMPLETED block's content as its own string, tags never surfacing anywhere.
 *  Tags can straddle token boundaries, so chunk edges hold back possible partial tags. */
export class ToolCallSplitter {
  private inside = false
  private hold = ''
  private buf = ''
  constructor(
    private readonly open = '<tool_call>',
    private readonly close = '</tool_call>',
  ) {}

  push(chunk: string): { text: string; blocks: string[] } {
    let s = this.hold + chunk
    this.hold = ''
    let text = ''
    const blocks: string[] = []
    for (;;) {
      if (!this.inside) {
        const i = s.indexOf(this.open)
        if (i === -1) {
          const safe = holdback(s, this.open)
          text += s.slice(0, safe)
          this.hold = s.slice(safe)
          return { text, blocks }
        }
        text += s.slice(0, i)
        s = s.slice(i + this.open.length)
        this.inside = true
        this.buf = ''
      } else {
        const i = s.indexOf(this.close)
        if (i === -1) {
          const safe = holdback(s, this.close)
          this.buf += s.slice(0, safe)
          this.hold = s.slice(safe)
          return { text, blocks }
        }
        blocks.push(this.buf + s.slice(0, i))
        this.buf = ''
        s = s.slice(i + this.close.length)
        this.inside = false
      }
    }
  }

  /** Emit whatever is held back. A block cut short by maxTokens surfaces as `partial` (its
   *  content never reaches the visible text). */
  flush(): { text: string; blocks: string[]; partial: string | null } {
    const r = this.inside ? { text: '', blocks: [], partial: this.buf + this.hold } : { text: this.hold, blocks: [], partial: null }
    this.hold = ''
    this.buf = ''
    this.inside = false
    return r
  }
}

/** Parse one block's content into a ToolCall. With enforcement on this cannot fail for a
 *  completed block; a failure (unenforced or truncated content) yields name '' and the raw text. */
export function parseToolCall(raw: string): ToolCall {
  try {
    const v = JSON.parse(raw.trim()) as { name?: unknown; arguments?: unknown }
    const name = typeof v?.name === 'string' ? v.name : ''
    let args = v?.arguments
    if (typeof args === 'string') args = JSON.parse(args) // templates also accept stringified arguments
    if (name && args !== null && typeof args === 'object' && !Array.isArray(args))
      return { name, arguments: args as Record<string, unknown>, raw }
  } catch {
    // fall through
  }
  return { name: '', arguments: {}, raw }
}
