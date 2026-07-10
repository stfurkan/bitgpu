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
import type { Engine, GenerateOptions, GenerateResult } from '../types'
import { ChatTokenizer, type ChatMessage, type DecoderStream } from './tokenizer'
import { ThinkSplitter, StopScanner } from './think'
import { makeJsonFilter, TokenByteTable, validateJsonSchema, type JsonSchema } from './json'

export { ChatTokenizer } from './tokenizer'
export type { ChatMessage, DecoderStream } from './tokenizer'
export { ThinkSplitter, StopScanner } from './think'
export { JsonMachine, validateJsonSchema } from './json'
export type { JsonSchema } from './json'

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
  /** The visible reply (think blocks removed). */
  text: string
  /** Content of <think> blocks, when the model emitted any. */
  thinkText: string
  /** Generated token ids (as returned by the engine; excludes the prompt). */
  tokens: number[]
  /** The exact token ids fed to the engine this turn (the full prompt, or the reuse delta). */
  inputTokenIds: number[]
  /** Why generation ended. */
  finishReason: 'stop' | 'length' | 'abort'
  /** True when this turn extended the KV cache instead of a full prefill. */
  reusedCache: boolean
  prefillMs: number
  decodeMs: number
  tokensPerSecond: number
}

export interface Chat {
  /** Generate a reply for the message list. Resolves with the full result; stream text via
   *  `onText`. Turns are serialized: overlapping calls queue instead of interleaving. */
  send(messages: ChatMessage[], options?: ChatSendOptions): Promise<ChatResult>
  /** Generate a reply as an async generator of visible-text deltas; the final {@link ChatResult}
   *  is the generator's return value: `const it = chat.stream(msgs); for await (const d of it) ...` */
  stream(messages: ChatMessage[], options?: ChatSendOptions): AsyncGenerator<string, ChatResult>
  /** Prefill a message prefix (e.g. the static system prompt) into the KV cache without decoding,
   *  so the first real turn is a cheap cache-append instead of a cold full prefill. */
  prewarm(messages: ChatMessage[]): Promise<void>
  /** Token count of the rendered prompt for a message list (chat template applied) - use for
   *  window budgeting against `engine.capabilities.maxSeqLen`, e.g. in an onOverflow policy. */
  countTokens(messages: ChatMessage[], opts?: { addGenerationPrompt?: boolean; think?: boolean }): number
  /** Forget the conversation: clears the engine's KV cache and the chat's committed transcript.
   *  Use this (not engine.resetCache()) so the two stay in sync. */
  reset(): void
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

/** True iff `next` is exactly `committed` plus one new trailing user turn - a clean append, so the
 *  engine can extend its KV cache with just that turn. Compared as MESSAGES, not token ids: chat
 *  templates render past assistant turns differently from live ones (e.g. Qwen3's empty <think>
 *  block), so a re-tokenized history is never a token-prefix of what the cache holds. */
function isCleanAppend(committed: readonly ChatMessage[] | null, next: readonly ChatMessage[]): boolean {
  if (!committed || next.length !== committed.length + 1) return false
  if (next[next.length - 1].role !== 'user') return false
  for (let i = 0; i < committed.length; i++) {
    if (next[i].role !== committed[i].role || next[i].content !== committed[i].content) return false
  }
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
  // Bumped by reset(): a turn in flight when reset() lands must not commit its transcript over
  // the cleared state (same hazard as the engine's own resetCache-vs-generate race).
  let resetEpoch = 0
  const dropCache = (): void => {
    committed = null
    engine.resetCache()
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
    const think = !json && (o.think ?? false) // a think block cannot be valid JSON; format wins
    const canReuse = (o.reuseCache ?? true) && !think && wrap !== null && isCleanAppend(committed, messages)

    let inputTokenIds: number[]
    if (canReuse) {
      const w = wrap as ChatWrap
      const userText = messages[messages.length - 1].content
      // Reconstruct exactly what a cold render of [committed..., user] appends after the cached
      // tokens: the previous turn's eos terminator (unless the cache already ends with it), the
      // inter-turn newline, the wrapped user turn, and the generation prompt.
      const deltaStr = `${cacheEndsAtEos ? '' : tk.eosToken}\n${w.userPrefix}${userText}${w.userSuffix}${w.genPrompt}`
      inputTokenIds = tk.encode(deltaStr, false)
    } else {
      dropCache()
      inputTokenIds = tk.encode(tk.applyChatTemplate(messages, { addGenerationPrompt: true, enableThinking: think }), false)
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
    const emit = (chunk: { text: string; think: string }): void => {
      if (chunk.text) {
        const visible = stops ? stops.push(chunk.text) : chunk.text
        if (visible) {
          text += visible
          o.onText?.(visible)
        }
        if (stops?.matched) stopCtl?.abort()
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
        promptLookup: json ? false : o.promptLookup,
        stopTokens: [tk.eosTokenId, ...(o.stopTokens ?? [])],
        reuseCache: canReuse,
        signal,
        candidateFilter: jf ? (ids) => jf.filter(ids) : undefined,
        onToken: (id) => {
          jf?.advance(id)
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

    // ── cache bookkeeping ──
    const aborted = o.signal?.aborted ?? false
    if (aborted || stops?.matched) {
      dropCache() // barge-in, or a stop-sequence cut (the cache holds a token overrun past it)
    } else if (think) {
      dropCache() // the cached tokens contain reasoning the stripped reply won't reproduce
    } else if (wrap !== null && text.trim() && resetEpoch === epoch0) {
      committed = [...messages, { role: 'assistant', content: text }]
      cacheEndsAtEos = false // the engine stopped AT eos without recording it
    } else {
      dropCache() // empty reply, non-ChatML template, or a reset() raced this turn
    }

    return {
      text,
      thinkText,
      tokens: result.tokens,
      inputTokenIds,
      finishReason: aborted ? 'abort' : stops?.matched ? 'stop' : result.tokens.length >= maxTokens ? 'length' : 'stop',
      reusedCache: canReuse,
      prefillMs: result.prefillMs,
      decodeMs: result.decodeMs,
      tokensPerSecond: result.tokensPerSecond,
    }
  }

  const send = serialize(sendImpl)

  async function prewarmImpl(messages: ChatMessage[]): Promise<void> {
    if (wrap === null) return // no ChatML wrappers -> reuse is disabled anyway, nothing to warm
    // Render WITHOUT the generation prompt and drop the trailing newline so the cache ends exactly
    // at the eos token; the standard reuse delta (which begins with "\n") then reconstructs the
    // next prompt token-for-token.
    const str = tk.applyChatTemplate(messages, { addGenerationPrompt: false, enableThinking: false }).replace(/\n$/, '')
    const epoch0 = resetEpoch
    await engine.prefill(tk.encode(str, false))
    if (resetEpoch !== epoch0) return // a reset() landed mid-prefill: stay forgotten
    committed = [...messages]
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
    countTokens: (messages: ChatMessage[], opts?: { addGenerationPrompt?: boolean; think?: boolean }): number =>
      tk.encode(tk.applyChatTemplate(messages, { addGenerationPrompt: opts?.addGenerationPrompt ?? true, enableThinking: opts?.think ?? false }), false).length,
    reset: (): void => {
      // Synchronous like engine.resetCache(); a turn in flight sees the epoch bump and will not
      // commit, so the next turn starts from the cleared transcript (a clean full prefill).
      resetEpoch++
      dropCache()
    },
    eosTokenId: tk.eosTokenId,
    tokenizer: tk,
  }
}
