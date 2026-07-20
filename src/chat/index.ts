// bitgpu/chat - messages in, streamed text out, entirely on-device.
//
// A thin, correctness-obsessed layer over the bitgpu engine that owns the text boundary:
// chat-template rendering (the model's own Jinja template), tokenization, incremental
// UTF-8-safe decode streaming, <think> block routing, EOS handling, and cross-turn KV-cache
// reuse with exact token bookkeeping. Everything the engine's design pushes to the caller
// ("token ids in, token ids out") lives here instead of being reimplemented per app.
//
// The two text libraries (@huggingface/tokenizers, @huggingface/jinja - pure JS, Apache-2.0)
// are inlined into dist/chat.js at build time, the same way the engine inlines its WGSL:
// `bitgpu` stays a zero-dependency package, and importing plain `bitgpu` never loads chat code.
import type { Engine, GenerateOptions, GenerateResult, KvSnapshot, TokenLogprobs } from '../types'
import { ChatTokenizer, type ChatMessage, type DecoderStream } from './tokenizer'
import { ThinkSplitter, StopScanner } from './think'
import { makeJsonFilter, TokenByteTable, validateJsonSchema, type JsonSchema } from './json'
import { makeToolFilter, parseToolCall, ToolCallSplitter, validateTools, type ChatTool, type PreparedTools, type ToolCall, type ToolChoice } from './tools'

export { ChatTokenizer } from './tokenizer'
export type { ChatMessage, DecoderStream } from './tokenizer'
export { ThinkSplitter, StopScanner } from './think'
export { JsonMachine, validateJsonSchema } from './json'
export type { JsonSchema } from './json'
export { ToolBodyMachine, ToolCallSplitter, makeToolFilter, parseToolCall, validateTools } from './tools'
export type { ChatTool, ToolCall, ToolChoice } from './tools'

/** Options for {@link createChat}. Point it at the model directory (which already hosts
 *  tokenizer.json + tokenizer_config.json next to the manifest), at explicit URLs, or at
 *  preloaded JSON (bring your own caching, e.g. OPFS). */
export interface ChatOptions {
  /** Directory holding tokenizer.json + tokenizer_config.json (usually the same modelUrl passed
   *  to createEngine). */
  modelUrl?: string
  /** Explicit URLs (when the tokenizer files live elsewhere than the weights). */
  tokenizerJsonUrl?: string
  tokenizerConfigUrl?: string
  /** Preloaded tokenizer files (skips fetching entirely). */
  tokenizer?: { json: unknown; config: Record<string, unknown> }
  /** Override fetching (custom caching / retries). Defaults to fetch + res.json(). */
  fetchJson?: (url: string) => Promise<unknown>
}

/** Per-turn options. Sampling fields pass through to the engine (same semantics as
 *  {@link GenerateOptions}); the rest control the chat layer. */
export interface ChatSendOptions {
  maxTokens?: number
  temperature?: number
  topK?: number
  topP?: number
  repetitionPenalty?: number
  noRepeatNgramSize?: number
  seed?: number
  promptLookup?: GenerateOptions['promptLookup']
  /** Per-token TRUE logprobs, N top alternatives per step (see GenerateOptions.logprobs) -
   *  surface model confidence: a low top-1 logprob or a flat top-N is the model guessing.
   *  Disables promptLookup for the turn. */
  logprobs?: number
  /** Extra stop token ids, in addition to the model's eos. */
  stopTokens?: number[]
  /** Stop STRINGS: generation ends when the visible reply contains one (matched across token
   *  boundaries); the stop text and anything after it is never emitted, and `finishReason` is
   *  'stop'. A stop-sequence turn drops the KV cache afterwards (the cache holds a short token
   *  overrun past the cut). */
  stopSequences?: string[]
  /** Called when the prompt alone exceeds the engine's KV window (maxSeqLen). Return a trimmed
   *  message list to retry ONCE with (a clean full prefill), or null to rethrow the error. Pair
   *  with {@link Chat.countTokens} to implement the trim policy. */
  onOverflow?: (info: { promptTokenCount: number; maxSeqLen: number }) => ChatMessage[] | null
  signal?: AbortSignal
  /** Render the template with enable_thinking and let the model reason. Think content streams to
   *  `onThink` and lands in `result.thinkText`; it never appears in the visible reply. A think
   *  turn drops the KV cache afterwards (the stripped reply cannot reproduce the cached tokens).
   *  Default false. */
  think?: boolean
  /** Streamed visible reply text (clean deltas: UTF-8-safe, think blocks removed). */
  onText?: (delta: string) => void
  /** Streamed think content (only meaningful with `think: true`). */
  onThink?: (delta: string) => void
  /** Reuse the KV cache when this turn is a clean append to the previous one (the committed
   *  conversation plus one new user turn). Default true; set false to force a full prefill. */
  reuseCache?: boolean
  /** Tools the model may call this turn, in the trained (OpenAI-shaped) declaration format. The
   *  model's own chat template renders them into the system block; any <tool_call> blocks in the
   *  reply are extracted (never shown as text), grammar-ENFORCED against the declared names and
   *  each tool's `parameters` schema, and returned parsed in {@link ChatResult.toolCalls}. The
   *  app executes a call and feeds the result back as a `tool` role message (plus the assistant
   *  turn with its `tool_calls`); the engine never executes anything. Enforcement guarantees the
   *  call's SHAPE, not its judgment - whether and what to call is model quality. Cannot combine
   *  with `format`; disables `promptLookup` for the turn. */
  tools?: ChatTool[]
  /** 'auto' (default): the model decides. { name }: FORCE a call to that tool as the whole reply
   *  (fully enforced end to end - the reliable mode for small models; implies think: false).
   *  'none': ignore `tools` this turn. */
  toolChoice?: ToolChoice
  /** Fired as each completed tool call is parsed during streaming. */
  onToolCall?: (call: ToolCall) => void
  /** Constrained decoding. `'json'` guarantees the reply is one complete, valid JSON value with
   *  an object or array root: every generated token is validated against an incremental JSON
   *  machine (invalid candidates are never sampled), and generation ends when the root value
   *  closes. `{ json: { schema } }` additionally enforces a JSON Schema SUBSET token-by-token -
   *  value types, `properties`/`required`/`additionalProperties: false`, `items`,
   *  `minItems`/`maxItems`, string `enum`, `integer` - so the reply cannot even be shaped wrong
   *  (an array that must hold 5 items cannot close at 1). Unsupported schema keywords throw
   *  loudly up front. Check `finishReason === 'stop'` - `'length'` means maxTokens cut the value
   *  short. Forces `think: false` and disables `promptLookup`. The guarantee is structural, not
   *  semantic: a schema makes the output parse into the right shape, not be true. */
  format?: 'json' | { json: { schema?: JsonSchema } }
}

export interface ChatResult {
  /** The visible reply (think blocks and tool_call blocks removed). */
  text: string
  /** Content of <think> blocks, when the model emitted any. */
  thinkText: string
  /** Parsed tool calls, in emission order (empty when the model answered in prose or no tools
   *  were given). Grammar enforcement makes every COMPLETED call well-formed; a call cut short
   *  by maxTokens has name '' and its partial text in `raw` (finishReason is then 'length'). */
  toolCalls: ToolCall[]
  /** Generated token ids (as returned by the engine; excludes the prompt). */
  tokens: number[]
  /** The exact token ids fed to the engine this turn (the full prompt, or the reuse delta). */
  inputTokenIds: number[]
  /** Why generation ended ('tool_calls' = ended at eos after making tool calls). */
  finishReason: 'stop' | 'length' | 'abort' | 'tool_calls'
  /** True when this turn extended the KV cache instead of a full prefill. */
  reusedCache: boolean
  /** Per-token logprob records aligned with `tokens` (present when options.logprobs was set). */
  logprobs?: TokenLogprobs[]
  prefillMs: number
  decodeMs: number
  tokensPerSecond: number
}

/** A saved conversation from {@link Chat.save}: the engine's {@link KvSnapshot} plus the
 *  chat-layer bookkeeping that makes cache reuse safe. Structured-cloneable (store in
 *  IndexedDB / OPFS or postMessage as-is; NOT `JSON.stringify`-able - the KV buffer would be
 *  lost). Treat the fields as opaque. */
export interface ChatSnapshot {
  /** Snapshot format version (currently `1`). */
  version: 1
  /** The engine-level KV cache + token history snapshot. */
  engine: KvSnapshot
  /** The committed transcript the cache holds, as messages. */
  committed: ChatMessage[]
  /** Whether the cached token sequence already ends with the eos token. */
  cacheEndsAtEos: boolean
  /** Canonical JSON of the tools list the conversation was rendered with (null = no tools). */
  toolsKey: string | null
}

export interface Chat {
  /** Generate a reply for the message list. Resolves with the full result; stream text via
   *  `onText`. Turns are serialized: overlapping calls queue instead of interleaving. */
  send(messages: ChatMessage[], options?: ChatSendOptions): Promise<ChatResult>
  /** Generate a reply as an async generator of visible-text deltas; the final {@link ChatResult}
   *  is the generator's return value: `const it = chat.stream(msgs); for await (const d of it) ...` */
  stream(messages: ChatMessage[], options?: ChatSendOptions): AsyncGenerator<string, ChatResult>
  /** Prefill a message prefix (e.g. the static system prompt) into the KV cache without decoding,
   *  so the first real turn is a cheap cache-append instead of a cold full prefill. Pass the same
   *  `tools` the turns will use - the template renders them into the system block, so a prewarm
   *  without them warms a different prompt. */
  prewarm(messages: ChatMessage[], opts?: { tools?: ChatTool[] }): Promise<void>
  /** Token count of the rendered prompt for a message list (chat template applied) - use for
   *  window budgeting against `engine.capabilities.maxSeqLen`, e.g. in an onOverflow policy. */
  countTokens(messages: ChatMessage[], opts?: { addGenerationPrompt?: boolean; think?: boolean; tools?: ChatTool[] }): number
  /** Forget the conversation: clears the engine's KV cache and the chat's committed transcript.
   *  Use this (not engine.resetCache()) so the two stay in sync. */
  reset(): void
  /** Snapshot the conversation - the engine's KV cache plus the chat's committed-transcript
   *  bookkeeping - as one structured-cloneable object. Persist it (IndexedDB / OPFS) or ship it
   *  to another worker; restoring it into a chat on the same model and `kvCache` mode makes the
   *  next clean-append turn extend the cache exactly as if the conversation never stopped (no
   *  re-prefill). Returns null when no conversation is committed. Queues behind in-flight turns
   *  like send/stream.
   *
   *  `{ delta: true }` makes a DELTA snapshot that excludes the shared prewarmed prefix (the system
   *  messages + tools warmed with {@link Chat.prewarm}), so a per-conversation snapshot drops the
   *  redundant system-prompt KV - tens of MB at chat scale, and a smaller structured-clone. Restore
   *  it into a chat freshly `prewarm()`ed with the SAME system + tools (restore validates and throws
   *  on a mismatch). Requires a prior prewarm(); throws otherwise. */
  save(opts?: { delta?: boolean }): Promise<ChatSnapshot | null>
  /** Replace the current conversation with a saved snapshot (see {@link Chat.save}). Throws when
   *  the snapshot does not match this engine's model or `kvCache` mode. */
  restore(snapshot: ChatSnapshot): Promise<void>
  /** The model's end-of-sequence token id. */
  readonly eosTokenId: number
  /** Escape hatch: encode/decode/applyChatTemplate for callers that need the text boundary. */
  readonly tokenizer: ChatTokenizer
}

/** Chat-template wrappers, derived from the tokenizer's own template at load (never hardcoded).
 *  `genPrompt` is what add_generation_prompt appends; `userPrefix`/`userSuffix` wrap a user turn.
 *  null when the template is not standard ChatML - cross-turn reuse is then disabled (each turn
 *  full-prefills: correct, just slower). */
interface ChatWrap {
  genPrompt: string
  userPrefix: string
  userSuffix: string
}

function deriveChatWrap(tk: ChatTokenizer): ChatWrap | null {
  if (!tk.hasChatTemplate) return null
  try {
    const render = (msgs: ChatMessage[], agp: boolean): string => tk.applyChatTemplate(msgs, { addGenerationPrompt: agp, enableThinking: false })
    const SENT = 'BITGPUSENTINEL'
    const userOnly = render([{ role: 'user', content: SENT }], false)
    const userGen = render([{ role: 'user', content: SENT }], true)
    const genPrompt = userGen.slice(userOnly.length)
    const i = userOnly.indexOf(SENT)
    if (i < 0 || !userGen.startsWith(userOnly) || !genPrompt.includes('assistant')) return null
    const userPrefix = userOnly.slice(0, i)
    const userSuffix = userOnly.slice(i + SENT.length)
    // The reuse delta reconstructs "<eos>\n<userPrefix>content<userSuffix><genPrompt>"; that shape
    // requires ChatML-style turns terminated by the eos token.
    if (!userSuffix.includes(tk.eosToken)) return null
    return { genPrompt, userPrefix, userSuffix }
  } catch {
    return null
  }
}

// tool_calls compare by name+arguments only, so feeding ChatResult.toolCalls (which carries the
// extra `raw` field) back into the transcript still counts as the same message.
const tcKey = (tcs: ChatMessage['tool_calls']): string => JSON.stringify(tcs?.map((t) => [t.name, t.arguments]) ?? null)

function msgEq(a: ChatMessage, b: ChatMessage): boolean {
  return a.role === b.role && a.content === b.content && tcKey(a.tool_calls) === tcKey(b.tool_calls)
}

/** True iff `next` is exactly `committed` plus one new trailing user turn - a clean append, so the
 *  engine can extend its KV cache with just that turn. Compared as MESSAGES, not token ids: chat
 *  templates render past assistant turns differently from live ones (e.g. Qwen3's empty <think>
 *  block), so a re-tokenized history is never a token-prefix of what the cache holds. */
function isCleanAppend(committed: readonly ChatMessage[] | null, next: readonly ChatMessage[]): boolean {
  if (!committed || next.length !== committed.length + 1) return false
  if (next[next.length - 1].role !== 'user') return false
  for (let i = 0; i < committed.length; i++) if (!msgEq(next[i], committed[i])) return false
  return true
}

/** True iff `next` is `committed` (whose last turn is an assistant turn WITH tool calls) plus
 *  only trailing `tool` result messages - the continuation leg of a tool round trip, appendable
 *  to the KV cache the same way a user turn is. */
function isToolAppend(committed: readonly ChatMessage[] | null, next: readonly ChatMessage[]): boolean {
  if (!committed || committed.length === 0 || next.length <= committed.length) return false
  const last = committed[committed.length - 1]
  if (last.role !== 'assistant' || !last.tool_calls?.length) return false
  for (let i = 0; i < committed.length; i++) if (!msgEq(next[i], committed[i])) return false
  for (let i = committed.length; i < next.length; i++) if (next[i].role !== 'tool') return false
  return true
}

/** Load the tokenizer files and return a {@link Chat} bound to the engine. */
export async function createChat(engine: Engine, options: ChatOptions): Promise<Chat> {
  let tk: ChatTokenizer
  if (options.tokenizer) {
    tk = new ChatTokenizer(options.tokenizer.json, options.tokenizer.config)
  } else {
    const base = options.modelUrl?.replace(/\/$/, '')
    const jsonUrl = options.tokenizerJsonUrl ?? (base ? `${base}/tokenizer.json` : null)
    const cfgUrl = options.tokenizerConfigUrl ?? (base ? `${base}/tokenizer_config.json` : null)
    if (!jsonUrl || !cfgUrl) throw new Error('bitgpu/chat: provide modelUrl, tokenizerJsonUrl+tokenizerConfigUrl, or a preloaded tokenizer')
    const get =
      options.fetchJson ??
      (async (url: string): Promise<unknown> => {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`bitgpu/chat: fetch ${url} failed: HTTP ${res.status}`)
        return res.json()
      })
    const [json, cfg] = await Promise.all([get(jsonUrl), get(cfgUrl)])
    tk = new ChatTokenizer(json, cfg as Record<string, unknown>)
  }
  const wrap = deriveChatWrap(tk)

  // ── conversation state ──
  // `committed` mirrors what the engine's KV cache holds, as messages; null = cache not usable.
  // `cacheEndsAtEos` tracks whether the cached token sequence already ends with the eos token:
  // true after prewarm (the rendered prefix ends at eos), false after a generate turn (the engine
  // stops AT eos without recording it), in which case the next reuse delta must re-insert it so
  // reuse stays token-exact with a cold prefill of the same conversation.
  let committed: ChatMessage[] | null = null
  let cacheEndsAtEos = false
  // The tools list the committed conversation was rendered with (as canonical JSON; null = no
  // tools). The template puts tools in the SYSTEM block, so a cache built with one tools list
  // cannot be extended by a turn using another - reuse requires an exact match.
  let committedToolsKey: string | null = null
  // Tokens prefilled by the last prewarm() (0 = none / a non-prewarm cache). save({ delta: true })
  // snapshots only the KV AFTER this shared prefix; a fresh prewarm caches prewarmLen-1 positions,
  // so the delta base is prewarmLen-1 (the last prewarm token's K/V is written on the first turn).
  let prewarmLen = 0
  // Bumped by reset(): a turn in flight when reset() lands must not commit its transcript over
  // the cleared state (same hazard as the engine's own resetCache-vs-generate race).
  let resetEpoch = 0
  const dropCache = (): void => {
    committed = null
    committedToolsKey = null
    prewarmLen = 0
    engine.resetCache()
  }

  // Tool support is a property of the model (template + vocabulary); probe once, loudly.
  let templateToolsOk: boolean | null = null
  function prepareTools(tools: ChatTool[], choice: ToolChoice): PreparedTools {
    validateTools(tools, choice)
    if (templateToolsOk === null) {
      try {
        const probe = tk.applyChatTemplate([{ role: 'user', content: 'x' }], {
          addGenerationPrompt: false,
          tools: [{ type: 'function', function: { name: 'bitgpu_probe_tool' } }],
        })
        // The template must both render the declarations and instruct the <tool_call> format
        // this module extracts and enforces.
        templateToolsOk = probe.includes('bitgpu_probe_tool') && probe.includes('<tool_call>')
      } catch {
        templateToolsOk = false
      }
    }
    if (!templateToolsOk) throw new Error("bitgpu/chat: this model's chat template does not support tools")
    const open = tk.tokenToId('<tool_call>')
    const close = tk.tokenToId('</tool_call>')
    if (open === undefined || close === undefined)
      throw new Error('bitgpu/chat: the vocabulary has no <tool_call> marker tokens, so tool calls cannot be enforced')
    return {
      tools,
      forced: typeof choice === 'object' ? choice.name : null,
      ids: { open, close, eos: tk.eosTokenId, thinkOpen: tk.tokenToId('<think>'), thinkClose: tk.tokenToId('</think>') },
    }
  }

  // Serialize chat turns: send/stream/prewarm share the committed-transcript bookkeeping, so
  // overlapping calls queue (the engine additionally serializes its own GPU ops).
  let chain: Promise<unknown> = Promise.resolve()
  const serialize = <Args extends unknown[], R>(fn: (...args: Args) => Promise<R>): ((...args: Args) => Promise<R>) => {
    return (...args: Args) => {
      const run = chain.then(
        () => fn(...args),
        () => fn(...args), // a failed predecessor must not poison the queue
      )
      chain = run.catch(() => undefined)
      return run
    }
  }

  let byteTable: TokenByteTable | null = null // per-token byte lookup, built once, shared by json turns

  async function sendImpl(messages: ChatMessage[], o: ChatSendOptions = {}): Promise<ChatResult> {
    if (messages.length === 0) throw new Error('bitgpu/chat: no messages')
    const json = o.format !== undefined
    const schema = typeof o.format === 'object' ? (o.format.json.schema ?? null) : null
    if (schema) validateJsonSchema(schema) // throws on anything outside the enforceable subset
    const toolsGiven = o.tools !== undefined && o.tools.length > 0 && o.toolChoice !== 'none'
    if (toolsGiven && json) throw new Error('bitgpu/chat: tools cannot be combined with format (a constrained-JSON reply has no room for tool calls)')
    const prep = toolsGiven ? prepareTools(o.tools as ChatTool[], o.toolChoice ?? 'auto') : null
    const toolsKey = prep ? JSON.stringify(prep.tools) : null
    // format wins over think (a think block cannot be valid JSON); a FORCED tool call also
    // implies think: false (the whole reply is the enforced call).
    const think = !json && prep?.forced == null && (o.think ?? false)
    const wantReuse = (o.reuseCache ?? true) && !think && wrap !== null && toolsKey === committedToolsKey
    const userAppend = wantReuse && isCleanAppend(committed, messages)
    const toolAppend = wantReuse && !userAppend && isToolAppend(committed, messages)
    let canReuse = userAppend || toolAppend

    let inputTokenIds: number[]
    if (userAppend) {
      const w = wrap as ChatWrap
      const userText = messages[messages.length - 1].content
      // Reconstruct exactly what a cold render of [committed..., user] appends after the cached
      // tokens: the previous turn's eos terminator (unless the cache already ends with it), the
      // inter-turn newline, the wrapped user turn, and the generation prompt.
      const deltaStr = `${cacheEndsAtEos ? '' : tk.eosToken}\n${w.userPrefix}${userText}${w.userSuffix}${w.genPrompt}`
      inputTokenIds = tk.encode(deltaStr, false)
    } else if (toolAppend) {
      // The continuation leg of a tool round trip: everything appended after the cached assistant
      // turn is the tool messages plus the generation prompt.
      const toolMsgs = messages.slice((committed as ChatMessage[]).length)
      let deltaStr: string | null = null
      try {
        // Fast path: the tool turns rendered standalone (the template wraps them in a user turn),
        // byte-identical to a cold append for position-independent ChatML templates (e.g. Qwen3).
        deltaStr = `${cacheEndsAtEos ? '' : tk.eosToken}\n` + tk.applyChatTemplate(toolMsgs, { addGenerationPrompt: true, enableThinking: false })
      } catch {
        // Some templates render tool turns only in the context of the whole conversation (e.g.
        // Qwen3.5 scans all messages for the user query and raises without one), so a standalone
        // render throws. Reconstruct the delta by diffing the full renders: what a cold render of
        // `messages` adds after the committed prefix. Sound only when the committed render is a
        // byte-prefix of the new one (true here - appending tool results after the last user query
        // leaves every earlier turn's rendering unchanged).
        const tools = prep?.tools
        const fullNew = tk.applyChatTemplate(messages, { addGenerationPrompt: true, enableThinking: false, tools })
        const committedRender = tk.applyChatTemplate(committed as ChatMessage[], { addGenerationPrompt: false, enableThinking: false, tools }).replace(/\n$/, '')
        if (fullNew.startsWith(committedRender)) deltaStr = `${cacheEndsAtEos ? '' : tk.eosToken}` + fullNew.slice(committedRender.length)
      }
      if (deltaStr !== null) {
        inputTokenIds = tk.encode(deltaStr, false)
      } else {
        // Neither reconstruction is valid: full-prefill the conversation (correct, no cache reuse).
        dropCache()
        canReuse = false
        inputTokenIds = tk.encode(tk.applyChatTemplate(messages, { addGenerationPrompt: true, enableThinking: think, tools: prep?.tools }), false)
      }
    } else {
      dropCache()
      inputTokenIds = tk.encode(tk.applyChatTemplate(messages, { addGenerationPrompt: true, enableThinking: think, tools: prep?.tools }), false)
    }

    const decoder: DecoderStream = tk.createDecoderStream(true)
    const splitter = new ThinkSplitter()
    // Stop sequences scan the VISIBLE channel; on a match the engine is aborted via an internal
    // signal (a few overrun tokens may generate before the per-step abort check lands - the text
    // is cut exactly, and the cache is dropped afterwards so the overrun can never be reused).
    const stops = o.stopSequences?.length ? new StopScanner(o.stopSequences) : null
    const stopCtl = stops ? new AbortController() : null
    const signal = stopCtl ? (o.signal ? AbortSignal.any([o.signal, stopCtl.signal]) : stopCtl.signal) : o.signal
    let text = ''
    let thinkText = ''
    // Tool-call blocks route to their own channel (never the visible reply), each completed
    // block parsing into a ToolCall as it closes.
    const toolSplit = prep ? new ToolCallSplitter() : null
    const toolCalls: ToolCall[] = []
    const pushBlock = (block: string): void => {
      const call = parseToolCall(block)
      toolCalls.push(call)
      if (call.name) o.onToolCall?.(call)
    }
    const emitVisible = (vis: string): void => {
      if (!vis) return
      const visible = stops ? stops.push(vis) : vis
      if (visible) {
        text += visible
        o.onText?.(visible)
      }
      if (stops?.matched) stopCtl?.abort()
    }
    const emit = (chunk: { text: string; think: string }): void => {
      if (chunk.text) {
        if (toolSplit) {
          const r = toolSplit.push(chunk.text)
          for (const b of r.blocks) pushBlock(b)
          emitVisible(r.text)
        } else emitVisible(chunk.text)
      }
      if (chunk.think) {
        thinkText += chunk.think
        o.onThink?.(chunk.think)
      }
    }

    const maxTokens = o.maxTokens ?? (think ? 1024 : 512)
    const epoch0 = resetEpoch
    // format:'json' - the candidate filter permits only tokens that keep the text a valid JSON
    // prefix; once the root value completes it permits only eos, so generation ends naturally
    // through the normal stop path. advance() moves the real machine on each emitted token.
    const jf = json ? makeJsonFilter((byteTable ??= new TokenByteTable(tk)), tk.eosTokenId, schema) : null
    // Tool turns: free text until <tool_call> opens, then the body grammar (declared names +
    // per-tool schema) until </tool_call>; a forced choice pins the whole reply to one call.
    const tf = prep ? makeToolFilter((byteTable ??= new TokenByteTable(tk)), prep) : null
    let result: GenerateResult
    try {
      result = await engine.generate(inputTokenIds, {
        maxTokens,
        temperature: o.temperature,
        topK: o.topK,
        topP: o.topP,
        repetitionPenalty: o.repetitionPenalty,
        noRepeatNgramSize: o.noRepeatNgramSize,
        seed: o.seed,
        logprobs: o.logprobs,
        promptLookup: json || prep ? false : o.promptLookup,
        stopTokens: [tk.eosTokenId, ...(o.stopTokens ?? [])],
        reuseCache: canReuse,
        signal,
        candidateFilter: jf ? (ids) => jf.filter(ids) : tf ? (ids) => tf.filter(ids) : undefined,
        onToken: (id) => {
          jf?.advance(id)
          tf?.advance(id)
          emit(splitter.push(decoder.push(id)))
        },
      })
    } catch (err) {
      if (/maxSeqLen/.test((err as Error).message)) {
        // Thrown BEFORE the engine mutates any state, so the cache is still valid. Offer the
        // app one shot at recovery: onOverflow returns a trimmed transcript to retry with.
        if (o.onOverflow) {
          const trimmed = o.onOverflow({ promptTokenCount: inputTokenIds.length, maxSeqLen: engine.capabilities.maxSeqLen })
          if (trimmed && trimmed.length > 0) {
            dropCache() // the trimmed transcript is a different conversation; start clean
            return await sendImpl(trimmed, { ...o, onOverflow: undefined, reuseCache: false })
          }
        }
      } else dropCache() // every other failure leaves the KV state unknown
      throw err
    }
    emit(splitter.push(decoder.flush()))
    emit(splitter.flush())
    if (toolSplit) {
      const fr = toolSplit.flush()
      for (const b of fr.blocks) pushBlock(b)
      if (fr.partial !== null) pushBlock(fr.partial) // maxTokens cut a block: parses to name '' + raw
      emitVisible(fr.text)
    }

    // ── cache bookkeeping ──
    const aborted = o.signal?.aborted ?? false
    const callsClean = toolCalls.every((c) => c.name !== '')
    if (aborted || stops?.matched) {
      dropCache() // barge-in, or a stop-sequence cut (the cache holds a token overrun past it)
    } else if (think) {
      dropCache() // the cached tokens contain reasoning the stripped reply won't reproduce
    } else if (wrap !== null && (text.trim() || toolCalls.length > 0) && callsClean && resetEpoch === epoch0) {
      const assistant: ChatMessage = { role: 'assistant', content: text }
      if (toolCalls.length) assistant.tool_calls = toolCalls.map((c) => ({ name: c.name, arguments: c.arguments }))
      committed = [...messages, assistant]
      committedToolsKey = toolsKey
      cacheEndsAtEos = false // the engine stopped AT eos without recording it
    } else {
      dropCache() // empty reply, a truncated tool call, a non-ChatML template, or a reset() raced this turn
    }

    return {
      text,
      thinkText,
      toolCalls,
      tokens: result.tokens,
      inputTokenIds,
      finishReason:
        aborted ? 'abort'
        : stops?.matched ? 'stop'
        : result.tokens.length >= maxTokens ? 'length'
        : toolCalls.length > 0 ? 'tool_calls'
        : 'stop',
      reusedCache: canReuse,
      ...(result.logprobs ? { logprobs: result.logprobs } : {}),
      prefillMs: result.prefillMs,
      decodeMs: result.decodeMs,
      tokensPerSecond: result.tokensPerSecond,
    }
  }

  const send = serialize(sendImpl)

  async function prewarmImpl(messages: ChatMessage[], opts: { tools?: ChatTool[] } = {}): Promise<void> {
    if (wrap === null) return // no ChatML wrappers -> reuse is disabled anyway, nothing to warm
    const prep = opts.tools?.length ? prepareTools(opts.tools, 'auto') : null
    // Render WITHOUT the generation prompt and drop the trailing newline so the cache ends exactly
    // at the eos token; the standard reuse delta (which begins with "\n") then reconstructs the
    // next prompt token-for-token.
    const str = tk.applyChatTemplate(messages, { addGenerationPrompt: false, enableThinking: false, tools: prep?.tools }).replace(/\n$/, '')
    const toks = tk.encode(str, false)
    const epoch0 = resetEpoch
    await engine.prefill(toks)
    if (resetEpoch !== epoch0) return // a reset() landed mid-prefill: stay forgotten
    committed = [...messages]
    committedToolsKey = prep ? JSON.stringify(prep.tools) : null
    prewarmLen = toks.length // the shared prefix delta snapshots exclude
    cacheEndsAtEos = true
  }

  return {
    send,
    stream(messages: ChatMessage[], options: ChatSendOptions = {}): AsyncGenerator<string, ChatResult> {
      // Pump the onText callback into an async generator (single consumer). Backpressure is
      // intentionally decoupled: generation runs at GPU speed and deltas queue until read.
      const queue: string[] = []
      let notify: (() => void) | null = null
      const wake = (): void => {
        notify?.()
        notify = null
      }
      let done = false
      let result: ChatResult | null = null
      let error: unknown = null
      const run = send(messages, {
        ...options,
        onText: (d) => {
          options.onText?.(d)
          queue.push(d)
          wake()
        },
      })
        .then((r) => {
          result = r
        })
        .catch((e) => {
          error = e
        })
        .finally(() => {
          done = true
          wake()
        })
      async function* gen(): AsyncGenerator<string, ChatResult> {
        for (;;) {
          if (queue.length > 0) {
            yield queue.shift() as string
            continue
          }
          if (done) break
          await new Promise<void>((r) => (notify = r))
        }
        await run
        if (error) throw error
        return result as ChatResult
      }
      return gen()
    },
    prewarm: serialize(prewarmImpl),
    countTokens: (messages: ChatMessage[], opts?: { addGenerationPrompt?: boolean; think?: boolean; tools?: ChatTool[] }): number =>
      tk.encode(tk.applyChatTemplate(messages, { addGenerationPrompt: opts?.addGenerationPrompt ?? true, enableThinking: opts?.think ?? false, tools: opts?.tools }), false).length,
    reset: (): void => {
      // Synchronous like engine.resetCache(); a turn in flight sees the epoch bump and will not
      // commit, so the next turn starts from the cleared transcript (a clean full prefill).
      resetEpoch++
      dropCache()
    },
    // Messages are deep-copied through JSON (they are JSON-safe by construction - the template
    // renders them) so a caller mutating its message objects cannot corrupt a saved snapshot.
    save: serialize(async (opts?: { delta?: boolean }): Promise<ChatSnapshot | null> => {
      if (committed === null) return null
      // delta: snapshot only the KV after the shared prewarmed prefix (restore into a chat freshly
      // prewarmed with the same system messages + tools). Drops the redundant prefix per snapshot.
      if (opts?.delta && prewarmLen <= 0)
        throw new Error('bitgpu/chat: save({ delta: true }) needs a prewarm() first (no shared prefix to exclude)')
      const eng = await engine.saveCache(opts?.delta ? { from: prewarmLen - 1 } : undefined)
      if (!eng) return null
      return {
        version: 1,
        engine: eng,
        committed: JSON.parse(JSON.stringify(committed)) as ChatMessage[],
        cacheEndsAtEos,
        toolsKey: committedToolsKey,
      }
    }),
    restore: serialize(async (snap: ChatSnapshot): Promise<void> => {
      if (!snap || snap.version !== 1) throw new Error('bitgpu/chat: unsupported chat snapshot version')
      if (!Array.isArray(snap.committed) || snap.committed.length === 0) throw new Error('bitgpu/chat: chat snapshot holds no committed transcript')
      // A DELTA snapshot (snap.engine.base) restores onto the current prewarm: engine.restoreCache
      // validates the cache is exactly at that prefix (prewarm() the same system+tools first) and
      // throws otherwise. Keep prewarmLen from that prewarm so a re-save stays a delta; a full
      // snapshot replaces the whole cache, so there is no shared prefix afterward.
      await engine.restoreCache(snap.engine) // validates model + kvCache mode + delta prefix
      resetEpoch++ // same hazard as reset(): a raced turn must not commit over the restored state
      committed = JSON.parse(JSON.stringify(snap.committed)) as ChatMessage[]
      committedToolsKey = snap.toolsKey ?? null
      cacheEndsAtEos = !!snap.cacheEndsAtEos
      if (!snap.engine.base) prewarmLen = 0
    }),
    eosTokenId: tk.eosTokenId,
    tokenizer: tk,
  }
}
