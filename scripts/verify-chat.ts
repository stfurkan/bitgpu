// Verification for bitgpu/chat (Node, no GPU).
//
//   npm run test:chat
//
// Three tiers:
//  (A) ThinkSplitter stream logic - pure, always runs.
//  (B) Chat orchestration against a MOCK engine with a hermetic byte-level tokenizer and a
//      ChatML template - always runs (this is what CI covers): template wrap derivation,
//      reuse-delta token exactness, eos re-insertion, cache bookkeeping, reset() race,
//      finishReason, stream()==send().
//  (C) Parity vs @huggingface/transformers (AutoTokenizer.apply_chat_template + encode) using a
//      REAL staged model's tokenizer files (examples/model) - auto-skips when not staged, runs
//      as part of the local release ritual alongside the GPU gate.
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createChat, ChatTokenizer, ThinkSplitter, ToolBodyMachine, ToolCallSplitter, parseToolCall, validateTools, type ChatMessage, type ChatTool } from '../src/chat/index'
import { JsonMachine, TokenByteTable, validateJsonSchema } from '../src/chat/json'
import type { Engine, GenerateOptions, GenerateResult } from '../src/types'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
let failures = 0
const check = (name: string, ok: boolean, detail = ''): void => {
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? `  ${detail}` : ''}`)
  if (!ok) failures++
}

// ── (A) ThinkSplitter ────────────────────────────────────────────────────────
console.log('(A) ThinkSplitter stream logic')
{
  const feed = (chunks: string[]): { text: string; think: string } => {
    const sp = new ThinkSplitter()
    let text = ''
    let think = ''
    for (const c of chunks) {
      const r = sp.push(c)
      text += r.text
      think += r.think
    }
    const f = sp.flush()
    return { text: text + f.text, think: think + f.think }
  }
  let r = feed(['hello ', '<think>hidden</think>', 'world'])
  check('plain tags', r.text === 'hello world' && r.think === 'hidden', JSON.stringify(r))
  r = feed(['a<th', 'ink>b</th', 'ink>c'])
  check('tags straddling chunks', r.text === 'ac' && r.think === 'b', JSON.stringify(r))
  r = feed(['<', 't', 'h', 'i', 'n', 'k', '>', 'x', '<', '/', 't', 'h', 'i', 'n', 'k', '>', 'y'])
  check('single-char chunks', r.text === 'y' && r.think === 'x', JSON.stringify(r))
  r = feed(['before<think>never closed'])
  check('unterminated think flushes to think channel', r.text === 'before' && r.think === 'never closed', JSON.stringify(r))
  r = feed(['no tags < here > at all'])
  check('angle brackets that are not tags', r.text === 'no tags < here > at all' && r.think === '', JSON.stringify(r))
  r = feed(['a<thin'])
  check('trailing partial open tag flushes as text', r.text === 'a<thin' && r.think === '', JSON.stringify(r))
  r = feed(['<think>a</think>mid<think>b</think>end'])
  check('two think blocks', r.text === 'midend' && r.think === 'ab', JSON.stringify(r))
}

// ── (B) orchestration: hermetic byte-level tokenizer + ChatML template + mock engine ─────────
console.log('(B) chat orchestration (mock engine, hermetic tokenizer)')

// GPT-2 bytes_to_unicode: the printable alias for every byte, the exact alphabet byte-level
// BPE vocabularies use.
function bytesToUnicode(): Map<number, string> {
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
  return new Map(bs.map((b, i) => [b, String.fromCharCode(cs[i])]))
}

// Mirrors the Qwen3 template's structure: an optional <tools> system block, per-message tool_calls
// rendering, tool-role messages wrapped as <tool_response> user turns, ChatML everywhere else.
const CHATML_TEMPLATE = [
  "{%- if tools %}{{ '<|im_start|>system\\n<tools>' }}{%- for tool in tools %}{{ '\\n' }}{{ tool | tojson }}{%- endfor %}",
  "{{ '\\n</tools>\\nWrap each call in <tool_call></tool_call> tags.<|im_end|>\\n' }}{%- endif %}",
  "{%- for message in messages %}",
  "{%- if message.role == 'tool' %}{{ '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}",
  "{%- else %}{{ '<|im_start|>' + message.role + '\\n' + message.content }}",
  "{%- if message.tool_calls %}{%- for tc in message.tool_calls %}{{ '\\n<tool_call>\\n{\"name\": \"' + tc.name + '\", \"arguments\": ' }}{{ tc.arguments | tojson }}{{ '}\\n</tool_call>' }}{%- endfor %}{%- endif %}",
  "{{ '<|im_end|>' + '\\n' }}{%- endif %}{%- endfor %}",
  "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- if not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}",
].join('')

function miniTokenizer(): { json: unknown; config: Record<string, unknown> } {
  const b2u = bytesToUnicode()
  const vocab: Record<string, number> = {}
  for (const [, ch] of b2u) if (!(ch in vocab)) vocab[ch] = Object.keys(vocab).length
  // Specialness mirrors the real Qwen3 tokenizer: the ChatML markers are special (stripped from
  // decoded text) while <think>/</think> are NOT (they reach the decoder stream, where the
  // ThinkSplitter routes them - exactly how the real model behaves).
  const specials = ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>', '<|endoftext|>']
  const added = specials.map((content, i) => ({
    id: 256 + i,
    content,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
    special: content.startsWith('<|'),
  }))
  return {
    json: {
      version: '1.0',
      truncation: null,
      padding: null,
      added_tokens: added,
      normalizer: null,
      pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true },
      post_processor: null,
      decoder: { type: 'ByteLevel', add_prefix_space: false, trim_offsets: true, use_regex: true },
      model: { type: 'BPE', dropout: null, unk_token: null, continuing_subword_prefix: null, end_of_word_suffix: null, fuse_unk: false, byte_fallback: false, vocab, merges: [] },
    },
    config: {
      chat_template: CHATML_TEMPLATE,
      eos_token: '<|im_end|>',
      model_max_length: 32768,
    },
  }
}

/** Mock engine: records every call; "generates" a scripted token-id reply per turn. */
function mockEngine(tk: ChatTokenizer): Engine & {
  calls: Array<{ ids: number[]; reuse: boolean }>
  prefills: number[][]
  resets: number
  script: (replies: string[]) => void
} {
  const calls: Array<{ ids: number[]; reuse: boolean }> = []
  const prefills: number[][] = []
  let queue: number[][] = []
  const m = {
    calls,
    prefills,
    resets: 0,
    script(replies: string[]): void {
      queue = replies.map((r) => tk.encode(r, false))
    },
    async generate(ids: number[], o: GenerateOptions = {}): Promise<GenerateResult> {
      calls.push({ ids: [...ids], reuse: o.reuseCache ?? false })
      const reply = queue.shift() ?? []
      const emitted: number[] = []
      for (const t of reply) {
        if (emitted.length >= (o.maxTokens ?? 256)) break
        if (o.signal?.aborted) break
        emitted.push(t)
        o.onToken?.(t)
      }
      return { tokens: emitted, prefillMs: 1, decodeMs: 1, tokensPerSecond: emitted.length, timing: { recordMs: 0, gpuMs: 0, readbackMs: 0 } }
    },
    async prefill(ids: number[]): Promise<{ prefillMs: number }> {
      prefills.push([...ids])
      return { prefillMs: 1 }
    },
    resetCache(): void {
      m.resets++
    },
    capabilities: { maxSeqLen: 128 },
  }
  return m as never
}

await (async () => {
  const { json, config } = miniTokenizer()
  const tk = new ChatTokenizer(json, config)

  // tokenizer basics on the hermetic vocab
  const round = 'héllo\nwörld 👍🏽 中文'
  check('byte-level roundtrip', tk.decode(tk.encode(round), false) === round)
  check('eos derivation', tk.eosToken === '<|im_end|>' && tk.eosTokenId === 257, `id=${tk.eosTokenId}`)
  const st = tk.createDecoderStream(false)
  const ids = tk.encode('café 👍🏽')
  let acc = ''
  for (const id of ids) acc += st.push(id)
  acc += st.flush()
  check('decoder stream == full decode (multi-byte holdback)', acc === 'café 👍🏽', JSON.stringify(acc))

  const engine = mockEngine(tk)
  const chat = await createChat(engine, { tokenizer: { json, config } })

  // template + wrap
  const u1: ChatMessage = { role: 'user', content: 'Hi there' }
  const rendered = tk.applyChatTemplate([u1], { addGenerationPrompt: true, enableThinking: false })
  check('single-turn render shape', rendered === '<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', JSON.stringify(rendered))

  // turn 1: full prefill of the rendered prompt
  engine.script(['Hello! How can I help?', 'Of course.', 'Sure thing.', 'And again.'])
  const r1 = await chat.send([u1], { seed: 1 })
  check('turn-1 input == rendered prompt ids', JSON.stringify(r1.inputTokenIds) === JSON.stringify(tk.encode(rendered)), `${r1.inputTokenIds.length} ids`)
  check('turn-1 no reuse, text captured', !r1.reusedCache && r1.text === 'Hello! How can I help?' && r1.finishReason === 'stop')

  // turn 2: clean append -> reuse; the delta must re-insert eos + reconstruct the wrapped turn,
  // so that [turn1 input + turn1 tokens + turn2 input] is a valid cold prompt.
  const u2: ChatMessage = { role: 'user', content: 'Thanks!' }
  const r2 = await chat.send([u1, { role: 'assistant', content: r1.text }, u2], { seed: 1 })
  const expectDelta = tk.encode('<|im_end|>\n<|im_start|>user\nThanks!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
  check('turn-2 reuses cache', r2.reusedCache && engine.calls[1].reuse)
  check('turn-2 delta ids exact (eos re-inserted)', JSON.stringify(r2.inputTokenIds) === JSON.stringify(expectDelta))
  const cold = tk.decode([...r1.inputTokenIds, ...r1.tokens, ...r2.inputTokenIds], false)
  const coldExpected = tk.applyChatTemplate([u1], { addGenerationPrompt: true, enableThinking: false }) + r1.text + '<|im_end|>\n<|im_start|>user\nThanks!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  check('cached+delta == the conversation as the model saw it', cold === coldExpected)

  // non-clean append (edited history) -> full prefill
  const r3 = await chat.send([{ role: 'user', content: 'Different history' }], { seed: 1 })
  check('edited history -> full prefill', !r3.reusedCache && !engine.calls[2].reuse)

  // reuseCache: false forces full prefill even on a clean append
  const r4 = await chat.send([{ role: 'user', content: 'Different history' }, { role: 'assistant', content: r3.text }, { role: 'user', content: 'next' }], { seed: 1, reuseCache: false })
  check('reuseCache:false forces full prefill', !r4.reusedCache)

  // think turn: template renders with thinking enabled, think content routed, cache dropped after
  engine.script(['<think>step by step</think>The answer is 4.', 'follow-up reply'])
  let thinkStream = ''
  const rt = await chat.send([{ role: 'user', content: '2+2?' }], { think: true, onThink: (d) => (thinkStream += d) })
  check('think content routed out of the reply', rt.text === 'The answer is 4.' && rt.thinkText === 'step by step' && thinkStream === rt.thinkText, JSON.stringify(rt.text))
  check('think render omits the empty think block', !tk.decode(rt.inputTokenIds, false).includes('<think>\n\n</think>'))
  const resetsBefore = engine.resets
  const rAfterThink = await chat.send([{ role: 'user', content: '2+2?' }, { role: 'assistant', content: rt.text }, { role: 'user', content: 'why?' }])
  check('think turn drops the cache (next turn full-prefills)', !rAfterThink.reusedCache && engine.resets > resetsBefore)

  // stream(): deltas concatenate to the final text; result is the generator return value
  engine.script(['streamed reply text'])
  const it = chat.stream([{ role: 'user', content: 'stream test' }])
  let streamed = ''
  let final: Awaited<ReturnType<typeof chat.send>> | undefined
  for (;;) {
    const n = await it.next()
    if (n.done) {
      final = n.value
      break
    }
    streamed += n.value
  }
  check('stream deltas == final text', streamed === 'streamed reply text' && final?.text === streamed)

  // reset() race: a reset landing MID-GENERATION (from inside the stream callback, after the
  // engine has started) must not let the finishing turn resurrect the transcript.
  engine.script(['racy reply', 'post-reset reply'])
  let resetOnce = false
  await chat.send([{ role: 'user', content: 'race' }], {
    onText: () => {
      if (!resetOnce) {
        resetOnce = true
        chat.reset()
      }
    },
  })
  const rPost = await chat.send([{ role: 'user', content: 'race' }, { role: 'assistant', content: 'racy reply' }, { role: 'user', content: 'again' }])
  check('reset() mid-generation prevents commit (no stale reuse)', !rPost.reusedCache)

  // prewarm: prefix ids end at eos; the next turn is a reuse append starting with "\n"
  engine.script(['warmed reply'])
  const sys: ChatMessage = { role: 'system', content: 'Be brief.' }
  await chat.prewarm([sys])
  const warmIds = engine.prefills[0]
  check('prewarm prefix ends at eos (trailing newline stripped)', warmIds[warmIds.length - 1] === tk.eosTokenId)
  const rw = await chat.send([sys, { role: 'user', content: 'hello' }])
  check('turn after prewarm reuses the cache with a newline-led delta', rw.reusedCache && tk.decode(rw.inputTokenIds, false).startsWith('\n<|im_start|>user'))

  // countTokens == encode(render) length
  const ctMsgs: ChatMessage[] = [sys, { role: 'user', content: 'How long is this?' }]
  const ctWant = tk.encode(tk.applyChatTemplate(ctMsgs, { addGenerationPrompt: true, enableThinking: false }), false).length
  check('countTokens == rendered prompt token count', chat.countTokens(ctMsgs) === ctWant, `${chat.countTokens(ctMsgs)} vs ${ctWant}`)

  // stopSequences: matched across token boundaries (byte-level mock emits one char per token),
  // the stop text never emitted, finishReason 'stop', and the cache dropped (no stale reuse)
  engine.script(['Hello STOP world', 'later reply'])
  let stopStreamed = ''
  const rs = await chat.send([{ role: 'user', content: 'stop test' }], { stopSequences: ['STOP'], onText: (d) => (stopStreamed += d) })
  check('stopSequences cuts the text exactly and reports stop', rs.text === 'Hello ' && stopStreamed === rs.text && rs.finishReason === 'stop', JSON.stringify(rs.text))
  const rsNext = await chat.send([{ role: 'user', content: 'stop test' }, { role: 'assistant', content: rs.text }, { role: 'user', content: 'next' }])
  check('stop-sequence turn drops the cache (token overrun never reused)', !rsNext.reusedCache)

  // onOverflow: first attempt throws the engine's maxSeqLen error; the trimmed retry succeeds
  {
    let calls = 0
    let overflowInfo: { promptTokenCount: number; maxSeqLen: number } | null = null
    const ofEngine = {
      async generate(ids: number[], o2: GenerateOptions = {}): Promise<GenerateResult> {
        calls++
        if (calls === 1) throw new Error(`generate: prompt length ${ids.length} exceeds maxSeqLen 128; trim history or raise maxSeqLen`)
        const reply = tk.encode('trimmed reply', false)
        for (const t of reply) o2.onToken?.(t)
        return { tokens: reply, prefillMs: 1, decodeMs: 1, tokensPerSecond: 1, timing: { recordMs: 0, gpuMs: 0, readbackMs: 0 } }
      },
      async prefill() { return { prefillMs: 1 } },
      resetCache() {},
      capabilities: { maxSeqLen: 128 },
    } as never as Engine
    const ofChat = await createChat(ofEngine, { tokenizer: { json, config } })
    const long: ChatMessage[] = [sys, { role: 'user', content: 'old turn' }, { role: 'assistant', content: 'old reply' }, { role: 'user', content: 'new question' }]
    const rOf = await ofChat.send(long, {
      onOverflow: (info) => {
        overflowInfo = info
        return [sys, long[long.length - 1]] // drop the middle turns, keep system + latest question
      },
    })
    check('onOverflow retries once with the trimmed transcript', rOf.text === 'trimmed reply' && calls === 2)
    const oi = overflowInfo as { promptTokenCount: number; maxSeqLen: number } | null // TS cannot see the callback assignment
    check('onOverflow receives prompt token count + maxSeqLen', oi !== null && oi.promptTokenCount > 0 && oi.maxSeqLen === 128)
  }
})()

// ── (D) JSON constrained decoding: machine, byte table, filter, mock-engine e2e ──────────────
console.log('(D) JSON constrained decoding')
{
  const enc = new TextEncoder()
  const feed = (s: string, byByte = false): { ok: boolean; complete: boolean } => {
    const m = new JsonMachine()
    const bytes = enc.encode(s)
    let ok = true
    if (byByte) {
      for (const b of bytes) if (!m.feed(new Uint8Array([b]))) { ok = false; break }
    } else ok = m.feed(bytes)
    return { ok, complete: ok && m.complete }
  }
  const COMPLETE = [
    '{}', '[]', '{"a":1}', '[1,2,3]', '{"a":[1,2,{"b":null}],"c":true,"d":false}',
    ' { "a" : 1 , "b" : [ true , false ] } ', '[[],{}]', '{"n":-1.5e+10,"z":0,"f":0.25,"E":2E3}',
    '{"s":"esc \\" \\\\ \\/ \\b \\f \\n \\r \\t \\u00e9 ok"}', '{"unicode":"héllo 👍🏽 中文"}',
    '[1]', '{"a":{"b":{"c":[]}}}', '{} ',
  ]
  const VALID_PREFIX = ['{', '{"a"', '{"a":', '{"a":1', '[1', '[1,', '{"a":tru', '{"s":"half', '{"s":"\\u00', '[-']
  const INVALID = [
    'true', '1', '"s"', 'null', // JSON mode requires an object/array root
    '{,}', '{"a"}', '{"a":}', '{1:2}', '[1,]', '{"a":1,}', '[,]', '{]', '[}', '{}{', '{} x',
    '{"a":01}', '{"a":1.}', '{"a":.5}', '{"a":-}', '{"a":1e}', '{"a":1e+}', '{"a":+1}', '{"a":--1}',
    '{"a":"\\x"}', '{"a":"raw\ntab"}', '{"a":true1}', '{"a":nul}',
  ]
  let mOk = true
  for (const s of COMPLETE) {
    const whole = feed(s)
    const single = feed(s, true)
    if (!whole.ok || !whole.complete || !single.ok || !single.complete) { mOk = false; console.log(`    machine FAILED to accept: ${JSON.stringify(s)}`) }
  }
  check(`machine accepts + completes ${COMPLETE.length} valid docs (whole and byte-by-byte)`, mOk)
  mOk = true
  for (const s of VALID_PREFIX) {
    const r = feed(s, true)
    if (!r.ok || r.complete) { mOk = false; console.log(`    machine mis-judged prefix: ${JSON.stringify(s)}`) }
  }
  check(`machine holds ${VALID_PREFIX.length} valid prefixes open`, mOk)
  mOk = true
  for (const s of INVALID) {
    const r = feed(s, true)
    const whole = feed(s)
    if (r.ok || whole.ok) { mOk = false; console.log(`    machine ACCEPTED invalid: ${JSON.stringify(s)}`) }
  }
  check(`machine rejects ${INVALID.length} invalid docs`, mOk)
  // UTF-8 byte-sequence enforcement inside strings
  const m1 = new JsonMachine()
  check('utf8: split multi-byte char across feeds', m1.feed(enc.encode('{"a":"')) && m1.feed(new Uint8Array([0xc3])) && m1.feed(new Uint8Array([0xa9])) && m1.feed(enc.encode('"}')) && m1.complete)
  const m2 = new JsonMachine()
  check('utf8: lead byte followed by ascii rejected', m2.feed(enc.encode('{"a":"')) && m2.feed(new Uint8Array([0xc3])) && !m2.feed(new Uint8Array([0x61])))
  const m3 = new JsonMachine()
  check('utf8: stray continuation byte rejected', m3.feed(enc.encode('{"a":"')) && !m3.feed(new Uint8Array([0x80])))
  const m4 = new JsonMachine()
  check('utf8: string cannot close mid-sequence', m4.feed(enc.encode('{"a":"')) && m4.feed(new Uint8Array([0xe4])) && !m4.feed(enc.encode('"')))

  // byte table + filter + full chat e2e on the hermetic tokenizer with a candidate-driven mock
  const { json: tj, config: tc } = miniTokenizer()
  const tk = new ChatTokenizer(tj, tc)
  const table = new TokenByteTable(tk)
  const id = (ch: string): number => tk.encode(ch, false)[0]
  check('byte table: ascii token bytes', String.fromCharCode(...(table.bytes(id('{')) ?? [])) === '{')
  const eBytes = table.bytes(tk.encode('é', false)[0])
  check('byte table: multi-byte lead token', eBytes !== null && eBytes![0] === 0xc3)
  check('byte table: added tokens are null', table.bytes(tk.eosTokenId) === null)

  // candidate-driven mock engine: per step, present scripted rank-ordered candidates, apply the
  // filter in chunks of 4 (exercising multi-batch), emit the first permitted (greedy semantics)
  const candEngine = (steps: number[][]): Engine => {
    return {
      async generate(_ids: number[], o: GenerateOptions = {}): Promise<GenerateResult> {
        const tokens: number[] = []
        for (const step of steps) {
          if (tokens.length >= (o.maxTokens ?? 256)) break
          let chosen: number | null = null
          for (let i = 0; i < step.length && chosen === null; i += 4) {
            const batch = step.slice(i, i + 4)
            const perm = o.candidateFilter ? o.candidateFilter(Uint32Array.from(batch), new Float32Array(batch.length)) : batch
            for (const b of batch) if (perm.includes(b)) { chosen = b; break }
          }
          if (chosen === null) throw new Error('no permitted token')
          if (o.stopTokens?.includes(chosen)) break
          tokens.push(chosen)
          o.onToken?.(chosen)
        }
        return { tokens, prefillMs: 1, decodeMs: 1, tokensPerSecond: tokens.length, timing: { recordMs: 0, gpuMs: 0, readbackMs: 0 } }
      },
      async prefill() { return { prefillMs: 1 } },
      resetCache() {},
    } as never
  }
  // the "model" wants prose ('h','e','y'), then structure; the filter must force valid JSON:
  // step lists put invalid candidates FIRST so any filter mistake changes the output.
  const steps = [
    [id('h'), id('e'), id('{')], // -> {
    [id('}'), id('"')], // -> } would close root|wait root open: '}' IS valid (empty obj)... use it
    [id('a'), tk.eosTokenId], // root complete -> only eos permitted -> stops
  ]
  const jEngine = candEngine(steps)
  const jChat = await createChat(jEngine, { tokenizer: { json: tj, config: tc } })
  const jr = await jChat.send([{ role: 'user', content: 'json please' }], { format: 'json', maxTokens: 8 })
  check('mock e2e: filter forces valid JSON and stops at completion', jr.text === '{}' && jr.finishReason === 'stop', JSON.stringify(jr.text))
  // think tags are structurally impossible in json mode: <think> is an added token -> always rejected
  const steps3 = [[tk.encode('<think>', false)[0], id('[')], [id(']')], [tk.eosTokenId]]
  const jr3 = await (await createChat(candEngine(steps3), { tokenizer: { json: tj, config: tc } })).send([{ role: 'user', content: 'x' }], { format: 'json' })
  check('mock e2e: added tokens (e.g. <think>) never enter JSON output', jr3.text === '[]', JSON.stringify(jr3.text))

  // ── (E) JSON Schema enforcement ──
  console.log('(E) JSON Schema enforcement')
  const throws = (fn: () => void): string | null => {
    try { fn(); return null } catch (e) { return (e as Error).message }
  }
  check('schema: unsupported keyword throws', /unsupported JSON Schema keyword.*pattern/.test(throws(() => validateJsonSchema({ type: 'object', pattern: 'x' } as never)) ?? ''))
  check('schema: scalar root throws', /root must be an object or array/.test(throws(() => validateJsonSchema({ type: 'string' })) ?? ''))
  check('schema: enum needing escapes throws', /require JSON escaping/.test(throws(() => validateJsonSchema({ type: 'object', properties: { a: { enum: ['ok', 'not "ok"'] } } })) ?? ''))
  check('schema: required key missing from properties throws', /missing from properties/.test(throws(() => validateJsonSchema({ type: 'object', required: ['x'], properties: { y: {} } })) ?? ''))
  check('schema: minItems > maxItems throws', /minItems > maxItems/.test(throws(() => validateJsonSchema({ type: 'array', minItems: 3, maxItems: 1 })) ?? ''))

  const enc2 = new TextEncoder()
  const feedS = (schema: import('../src/chat/json').JsonSchema, s: string): { ok: boolean; complete: boolean } => {
    const m = new JsonMachine(schema)
    const ok = m.feed(enc2.encode(s))
    return { ok, complete: ok && m.complete }
  }
  const TURKEY: import('../src/chat/json').JsonSchema = {
    type: 'array', minItems: 5, maxItems: 5,
    items: { type: 'object', required: ['name', 'population'], additionalProperties: false, properties: { name: { type: 'string' }, population: { type: 'number' } } },
  }
  const item = (n: string, p: number): string => `{"name":${JSON.stringify(n)},"population":${p}}`
  check('schema: object root rejected when root is array', !feedS(TURKEY, '{').ok)
  check('schema: 5-item conforming doc completes', feedS(TURKEY, `[${[1, 2, 3, 4, 5].map((i) => item('c' + i, i)).join(',')}]`).complete)
  check('schema: closing at 4 items rejected (minItems)', !feedS(TURKEY, `[${[1, 2, 3, 4].map((i) => item('c' + i, i)).join(',')}]`).ok)
  check('schema: a 6th item rejected (maxItems)', !feedS(TURKEY, `[${[1, 2, 3, 4, 5].map((i) => item('c' + i, i)).join(',')},`).ok)
  check('schema: unknown key rejected at first byte (additionalProperties false)', !feedS(TURKEY, '[{"x').ok)
  check('schema: duplicate key rejected', !feedS(TURKEY, '[{"name":"a","name').ok)
  check('schema: object close without required keys rejected', !feedS(TURKEY, '[{"name":"a"}').ok)
  check('schema: wrong value type rejected (string where number)', !feedS(TURKEY, '[{"name":"a","population":"').ok)
  check('schema: comma after all allowed keys present rejected', !feedS(TURKEY, '[{"name":"a","population":1,').ok)
  check('schema: maxItems 0 keeps the array empty', feedS({ type: 'array', maxItems: 0 }, '[]').complete && !feedS({ type: 'array', maxItems: 0 }, '[1').ok)
  const INT = { type: 'object' as const, required: ['n'], properties: { n: { type: 'integer' as const } } }
  check('schema: integer bans fraction and exponent', feedS(INT, '{"n":42}').complete && !feedS(INT, '{"n":4.').ok && !feedS(INT, '{"n":4e').ok)
  const ENUM = { type: 'object' as const, required: ['mood'], additionalProperties: false, properties: { mood: { enum: ['positive', 'negative', 'neutral'] } } }
  check('schema: enum accepts a listed literal', feedS(ENUM, '{"mood":"neutral"}').complete)
  check('schema: enum rejects a non-prefix first byte', !feedS(ENUM, '{"mood":"x').ok)
  check('schema: enum rejects closing on an incomplete literal', !feedS(ENUM, '{"mood":"neg"').ok)
  check('schema: enum diverging past a shared prefix rejected', !feedS(ENUM, '{"mood":"negx').ok)
  check('schema: enum string rejects escapes', !feedS(ENUM, '{"mood":"\\\\').ok)
  const NEST = { type: 'object' as const, required: ['tags'], properties: { tags: { type: 'array' as const, minItems: 1, items: { enum: ['a', 'b'] } } } }
  check('schema: nested array of enums enforced', feedS(NEST, '{"tags":["a","b"]}').complete && !feedS(NEST, '{"tags":[]').ok && !feedS(NEST, '{"tags":["c').ok)
  check('schema: booleans and null typed', feedS({ type: 'object' as const, properties: { b: { type: 'boolean' as const } } }, '{"b":true}').complete && !feedS({ type: 'object' as const, properties: { b: { type: 'boolean' as const } } }, '{"b":n').ok)
  // structural-whitespace run cap: a model denied prose must not be able to loop on whitespace
  // (the 8B did exactly that on real hardware and burned its whole budget producing "[ ")
  check('ws cap: pretty-printing survives', feedS({ type: 'array' as const }, '[\n  {\n    "a": 1\n  },\n  2\n]').complete)
  check('ws cap: a 17-byte structural whitespace run is rejected', !feedS({ type: 'array' as const }, '[' + ' '.repeat(17)).ok)
  check('ws cap: spaces inside strings are content, not capped', feedS({ type: 'array' as const }, `["${' '.repeat(40)}"]`).complete)
  check('ws cap: run resets after a structural byte', feedS({ type: 'array' as const }, '[' + ' '.repeat(10) + '1,' + ' '.repeat(10) + '2]').complete)

  // e2e: the model "wants" the Ankara failure (open '{', close after 1 item) - the schema forbids
  // both. Candidates put the wrong choice FIRST at every decision point.
  const CITY = { type: 'array' as const, minItems: 2, maxItems: 2, items: { type: 'object' as const, required: ['name'], additionalProperties: false, properties: { name: { type: 'string' as const } } } }
  const chars = (s: string): number[][] => [...s].map((ch) => [id(ch)]) // one single-byte token per step
  const oneItem = (n: string): number[][] => [
    [id('{')],
    [id('}'), id('"')], // wants to close the empty object -> the required key is forced
    ...chars('name'),
    [id('"')], [id(':')], [id('"')], [id(n)], [id('"')],
    [id(']'), id('}')], // wants to end the array from inside the item -> } forced
  ]
  const stepsCity: number[][] = [
    [id('{'), id('[')], // wants an object root (the Ankara failure) -> [ forced
    ...oneItem('A'),
    [id(']'), id(',')], // wants to close at 1 item -> , forced (minItems 2)
    ...oneItem('B'),
    [id(','), id(']')], // wants a 3rd item -> ] forced (maxItems 2)
    [id('x'), tk.eosTokenId], // complete -> only eos permitted
  ]
  const cChat = await createChat(candEngine(stepsCity), { tokenizer: { json: tj, config: tc } })
  const cr = await cChat.send([{ role: 'user', content: 'cities' }], { format: { json: { schema: CITY } } })
  let cParsed: unknown = null
  try { cParsed = JSON.parse(cr.text) } catch { /* fail below */ }
  check('schema e2e: forced [ root, forced 2nd item, forced ] at max - parses to 2 items', Array.isArray(cParsed) && (cParsed as unknown[]).length === 2 && cr.finishReason === 'stop', JSON.stringify(cr.text))
  check('schema e2e: unsupported schema throws at send', /unsupported JSON Schema keyword/.test((await cChat.send([{ role: 'user', content: 'x' }], { format: { json: { schema: { type: 'object', oneOf: [] } as never } } }).catch((e) => e.message))))

  // ── (F) tool calling ──
  console.log('(F) tool calling')
  const TOOLS: ChatTool[] = [{ type: 'function', function: { name: 'get', description: 'lookup', parameters: { type: 'object', required: ['q'], additionalProperties: false, properties: { q: { enum: ['x'] } } } } }]
  const OTHER: ChatTool = { type: 'function', function: { name: 'go' } }
  const msg = (s: string): ChatMessage[] => [{ role: 'user', content: s }]

  // declaration validation
  check('tools: empty list throws', (throws(() => validateTools([], 'auto')) ?? '').includes('tools is empty'))
  check('tools: bad shape throws', (throws(() => validateTools([{ type: 'function', function: {} } as never], 'auto')) ?? '').includes('must be'))
  check('tools: duplicate names throw', (throws(() => validateTools([OTHER, OTHER], 'auto')) ?? '').includes('duplicate tool name'))
  check('tools: escape-y name throws', (throws(() => validateTools([{ type: 'function', function: { name: 'a"b' } }], 'auto')) ?? '').includes('JSON escaping'))
  check('tools: array parameters throw', (throws(() => validateTools([{ type: 'function', function: { name: 'a', parameters: { type: 'array' } } }] as ChatTool[], 'auto')) ?? '').includes('must describe an object'))
  check('tools: unknown forced name throws', (throws(() => validateTools(TOOLS, { name: 'nope' })) ?? '').includes("unknown tool 'nope'"))

  // body grammar machine
  const B_SCHEMAS = new Map([
    [ToolBodyMachine.bytesOf('get'), { type: 'object', required: ['q'], additionalProperties: false, properties: { q: { enum: ['x'] } } } as import('../src/chat/json').JsonSchema],
    [ToolBodyMachine.bytesOf('go2'), { type: 'object' } as import('../src/chat/json').JsonSchema],
  ])
  const feedB = (s: string): { ok: boolean; complete: boolean } => {
    const m = new ToolBodyMachine([ToolBodyMachine.bytesOf('get'), ToolBodyMachine.bytesOf('go2')], B_SCHEMAS)
    const ok = m.feed(enc2.encode(s))
    return { ok, complete: ok && m.complete }
  }
  check('body: canonical call completes (leading/trailing newline ok)', feedB('\n{"name": "get", "arguments": {"q": "x"}}\n').complete)
  check('body: parameter-less tool takes any object', feedB('{"name": "go2", "arguments": {"anything": [1, 2]}}').complete)
  check('body: name diverging from every declared prefix rejected', !feedB('{"name": "gz').ok)
  check('body: name that is only a prefix rejected at its quote', !feedB('{"name": "ge"').ok)
  check('body: deviation from the canonical literal rejected', !feedB('{ "name"').ok)
  check('body: argument key outside the tool schema rejected', !feedB('{"name": "get", "arguments": {"z').ok)
  check('body: argument enum enforced', !feedB('{"name": "get", "arguments": {"q": "y').ok)
  check('body: missing required argument cannot close', !feedB('{"name": "get", "arguments": {}').ok)
  check('body: whitespace run before the wrapper capped', !feedB('      {').ok)
  check('body: content after completion rejected', !feedB('{"name": "go2", "arguments": {}}x').ok)

  // block splitter
  {
    const sp = new ToolCallSplitter()
    const p1 = sp.push('before <tool')
    const p2 = sp.push('_call>{"a":1}</tool_call> after')
    check('splitter: tags straddling chunks', p1.text + p2.text === 'before  after' && p2.blocks.length === 1 && p2.blocks[0] === '{"a":1}', JSON.stringify(p2))
    const sp2 = new ToolCallSplitter()
    const q1 = sp2.push('<tool_call>A</tool_call><tool_call>B</tool_call>tail')
    const q2 = sp2.flush()
    check('splitter: two blocks + tail text', q1.blocks.join('|') === 'A|B' && q1.text + q2.text === 'tail' && q2.partial === null)
    const sp3 = new ToolCallSplitter()
    sp3.push('<tool_call>{"name": "ge')
    check('splitter: a cut block surfaces as partial, never as text', sp3.flush().partial === '{"name": "ge')
  }

  // block parsing
  {
    const good = parseToolCall('\n{"name": "get", "arguments": {"q": "x"}}\n')
    check('parse: object arguments', good.name === 'get' && JSON.stringify(good.arguments) === '{"q":"x"}')
    const strArgs = parseToolCall('{"name": "g", "arguments": "{\\"a\\": 1}"}')
    check('parse: stringified arguments unwrapped', strArgs.name === 'g' && JSON.stringify(strArgs.arguments) === '{"a":1}')
    const bad = parseToolCall('{"nam')
    check('parse: malformed keeps raw with name ""', bad.name === '' && bad.raw === '{"nam')
  }

  // send-level guards
  {
    const c = await createChat(candEngine([]), { tokenizer: { json: tj, config: tc } })
    check('send: tools + format throws', ((await c.send(msg('x'), { tools: TOOLS, format: 'json' }).catch((e) => e.message)) as string).includes('cannot be combined with format'))
    const simple = "{%- for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
    const cNo = await createChat(candEngine([]), { tokenizer: { json: tj, config: { ...tc, chat_template: simple } } })
    check('send: template without tools support throws', ((await cNo.send(msg('x'), { tools: TOOLS }).catch((e) => e.message)) as string).includes('does not support tools'))
    const { json: nj, config: nc } = miniTokenizer()
    ;(nj as { added_tokens: { content: string }[] }).added_tokens = (nj as { added_tokens: { content: string }[] }).added_tokens.filter((t) => !t.content.includes('tool_call'))
    const cNoVocab = await createChat(candEngine([]), { tokenizer: { json: nj, config: nc } })
    check('send: vocabulary without markers throws', ((await cNoVocab.send(msg('x'), { tools: TOOLS }).catch((e) => e.message)) as string).includes('no <tool_call> marker tokens'))
  }

  // filter e2e on the candidate-driven engine: the "model" tries the WRONG token first at every
  // body position ('z'); the filter must force the canonical enforced call instead.
  const toolOpen = tk.encode('<tool_call>', false)[0]
  const toolClose = tk.encode('</tool_call>', false)[0]
  const BODY = '{"name": "get", "arguments": {"q": "x"}}'
  const bodySteps: number[][] = [...BODY].map((ch) => [id('z'), id(ch)])
  const closeStep = [id('z'), toolClose]
  {
    const autoSteps = [[id('o')], [id('k')], [toolOpen], ...bodySteps, closeStep, [tk.eosTokenId]]
    let fired = 0
    const ca = await createChat(candEngine(autoSteps), { tokenizer: { json: tj, config: tc } })
    const ra = await ca.send(msg('what is x?'), { tools: TOOLS, onToolCall: () => fired++ })
    check(
      'tool e2e (auto): prose passes free, call body forced, block hidden from text',
      ra.text === 'ok' && ra.toolCalls.length === 1 && ra.toolCalls[0].name === 'get' && JSON.stringify(ra.toolCalls[0].arguments) === '{"q":"x"}' && ra.finishReason === 'tool_calls' && fired === 1,
      JSON.stringify({ text: ra.text, calls: ra.toolCalls.length }),
    )
    check('tool e2e (auto): tools rendered into the system block', tk.decode(ra.inputTokenIds, false).includes('<tools>') && tk.decode(ra.inputTokenIds, false).includes('"get"'))
  }
  {
    const forcedSteps = [[id('h'), toolOpen], ...bodySteps, closeStep, [id('z'), tk.eosTokenId]]
    const cf = await createChat(candEngine(forcedSteps), { tokenizer: { json: tj, config: tc } })
    const rf = await cf.send(msg('call it'), { tools: TOOLS, toolChoice: { name: 'get' } })
    check('tool e2e (forced): first token forced to <tool_call>, eos-only after the call', rf.text === '' && rf.toolCalls.length === 1 && rf.toolCalls[0].name === 'get' && rf.finishReason === 'tool_calls')
  }
  {
    const noneSteps = [[id('a')], [tk.eosTokenId]]
    const cn = await createChat(candEngine(noneSteps), { tokenizer: { json: tj, config: tc } })
    const rn = await cn.send(msg('hi'), { tools: TOOLS, toolChoice: 'none' })
    check("toolChoice 'none': tools not rendered, plain reply", rn.text === 'a' && rn.toolCalls.length === 0 && !tk.decode(rn.inputTokenIds, false).includes('<tools>'))
  }

  // round trip: call turn -> tool result extends the KV cache -> different tools full-prefill
  {
    const engine = mockEngine(tk)
    const chat = await createChat(engine, { tokenizer: { json: tj, config: tc } })
    engine.script(['<tool_call>\n{"name": "get", "arguments": {"q": "x"}}\n</tool_call>', 'It is 42.', 'ok', 'sure'])
    const u: ChatMessage = { role: 'user', content: 'what is x?' }
    const t1 = await chat.send([u], { tools: TOOLS })
    check('round trip: pure call turn parsed, empty visible text, committed', t1.toolCalls.length === 1 && t1.text === '' && t1.finishReason === 'tool_calls')
    const asst: ChatMessage = { role: 'assistant', content: t1.text, tool_calls: t1.toolCalls }
    const t2 = await chat.send([u, asst, { role: 'tool', content: '42' }], { tools: TOOLS })
    const delta = tk.decode(t2.inputTokenIds, false)
    check('round trip: tool result extends the cache (reuse, no full prefill)', t2.reusedCache && t2.text === 'It is 42.')
    check('round trip: delta = eos + tool_response turn + gen prompt', delta === '<|im_end|>\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', JSON.stringify(delta))
    const t3 = await chat.send([u, asst, { role: 'tool', content: '42' }, { role: 'assistant', content: t2.text }, { role: 'user', content: 'thanks' }], { tools: [...TOOLS, OTHER] })
    check('round trip: changed tools list -> full prefill (system block differs)', !t3.reusedCache)
    const t4 = await chat.send([u, asst, { role: 'tool', content: '42' }, { role: 'assistant', content: t2.text }, { role: 'user', content: 'thanks' }, { role: 'assistant', content: t3.text }, { role: 'user', content: 'more' }], { tools: [...TOOLS, OTHER] })
    check('round trip: same tools -> user append still reuses', t4.reusedCache)
  }

  // prewarm with tools
  {
    const engine = mockEngine(tk)
    const chat = await createChat(engine, { tokenizer: { json: tj, config: tc } })
    const sys: ChatMessage = { role: 'system', content: 'S.' }
    await chat.prewarm([sys], { tools: TOOLS })
    check('prewarm with tools renders the tools block', tk.decode(engine.prefills[0], false).includes('<tools>'))
    engine.script(['hi', 'again'])
    const rp = await chat.send([sys, { role: 'user', content: 'hey' }], { tools: TOOLS })
    check('turn after tools prewarm reuses the cache', rp.reusedCache)
    const rp2 = await chat.send([sys, { role: 'user', content: 'hey' }, { role: 'assistant', content: rp.text }, { role: 'user', content: 'more' }])
    check('tools dropped on a later turn -> full prefill', !rp2.reusedCache)
  }

  // tool markers inside a think block are the model musing, not a call
  {
    const engine = mockEngine(tk)
    const chat = await createChat(engine, { tokenizer: { json: tj, config: tc } })
    engine.script(['<think>maybe <tool_call> syntax?</think>fine'])
    const rt = await chat.send(msg('x'), { tools: TOOLS, think: true })
    check('markers inside <think> stay reasoning', rt.toolCalls.length === 0 && rt.text === 'fine' && rt.thinkText.includes('<tool_call>'), JSON.stringify(rt.text))
  }

  // a call cut by maxTokens: 'length', name '', raw partial, no onToolCall, cache dropped
  {
    const engine = mockEngine(tk)
    const chat = await createChat(engine, { tokenizer: { json: tj, config: tc } })
    const cutText = '<tool_call>\n{"name": "ge'
    engine.script([cutText, 'next'])
    let fired = 0
    const rc = await chat.send(msg('x'), { tools: TOOLS, maxTokens: tk.encode(cutText, false).length, onToolCall: () => fired++ })
    check('cut call: length + partial raw + no onToolCall', rc.finishReason === 'length' && rc.toolCalls.length === 1 && rc.toolCalls[0].name === '' && rc.toolCalls[0].raw.includes('"ge') && fired === 0 && rc.text === '', JSON.stringify(rc.toolCalls))
    const rcNext = await chat.send([...msg('x'), { role: 'assistant', content: '', tool_calls: rc.toolCalls }, { role: 'user', content: 'y' }], { tools: TOOLS })
    check('cut call: cache dropped (no stale reuse)', !rcNext.reusedCache)
  }
}

// ── (C) parity vs transformers.js on a real model (auto-skips when not staged) ───────────────
console.log('(C) parity vs @huggingface/transformers (real model tokenizer)')
const staged = ['model', 'model-4b', 'model-8b'].find((d) => existsSync(join(root, 'examples', d, 'tokenizer.json')))
if (!staged) {
  console.log('  [skip] no staged model has tokenizer.json; run locally with a staged model for the parity tier')
} else {
  console.log(`  (using examples/${staged})`)
  const modelDir = join(root, 'examples', staged)
  const json = JSON.parse(readFileSync(join(modelDir, 'tokenizer.json'), 'utf8'))
  const config = JSON.parse(readFileSync(join(modelDir, 'tokenizer_config.json'), 'utf8'))
  const tk = new ChatTokenizer(json, config)

  const { env, AutoTokenizer } = await import('@huggingface/transformers')
  env.allowRemoteModels = false
  env.localModelPath = join(root, 'examples')
  const ref = await AutoTokenizer.from_pretrained(staged)

  const convs: ChatMessage[][] = [
    [{ role: 'user', content: 'The capital of France is Paris. The capital of Japan is' }],
    [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'héllo wörld 👍🏽 中文 <tags> & "quotes"' },
      { role: 'assistant', content: 'Reply one.\nSecond line.' },
      { role: 'user', content: 'And again?' },
    ],
  ]
  for (const [i, msgs] of convs.entries()) {
    for (const thinking of [false, true]) {
      const ours = tk.applyChatTemplate(msgs, { addGenerationPrompt: true, enableThinking: thinking })
      const theirs = ref.apply_chat_template(msgs, { add_generation_prompt: true, tokenize: false, enable_thinking: thinking } as never) as unknown as string
      check(`chat template parity (conv ${i}, thinking=${thinking})`, ours === theirs)
      const ourIds = tk.encode(ours, false)
      const theirIds = Array.from((ref as never as { encode: (t: string, o: object) => number[] }).encode(ours, { add_special_tokens: false }), Number)
      check(`encode parity (conv ${i}, thinking=${thinking})`, JSON.stringify(ourIds) === JSON.stringify(theirIds), `${ourIds.length} ids`)
    }
  }
  check('eos id parity', tk.eosTokenId === Number(ref.eos_token_id ?? -1), `${tk.eosTokenId} vs ${ref.eos_token_id}`)

  // tool rendering parity on the REAL template: declarations in the system block, an assistant
  // turn carrying a call, and the tool response wrapped as a user turn
  const twTools = [{ type: 'function', function: { name: 'get_weather', description: 'Get the weather for a city', parameters: { type: 'object', required: ['city'], properties: { city: { type: 'string' } } } } }]
  const twMsgs: ChatMessage[] = [
    { role: 'user', content: 'Weather in Paris?' },
    { role: 'assistant', content: '', tool_calls: [{ name: 'get_weather', arguments: { city: 'Paris' } }] },
    { role: 'tool', content: '{"temp": 21}' },
  ]
  const oursT = tk.applyChatTemplate(twMsgs, { addGenerationPrompt: true, enableThinking: false, tools: twTools })
  const theirsT = ref.apply_chat_template(twMsgs as never, { add_generation_prompt: true, tokenize: false, enable_thinking: false, tools: twTools } as never) as unknown as string
  check('tool rendering parity (declarations + call + response)', oursT === theirsT, oursT === theirsT ? '' : JSON.stringify({ ours: oursT.slice(0, 120), theirs: (theirsT ?? '').slice(0, 120) }))

  // decoder stream integrity on real BPE with multi-byte content
  const sample = 'Emoji 👍🏽 flag 🇹🇷 combining é́ CJK 世界 done.'
  const ids = tk.encode(sample, false)
  const st = tk.createDecoderStream(true)
  let acc = ''
  for (const id of ids) acc += st.push(id)
  acc += st.flush()
  check('decoder stream == full decode (real vocab)', acc === tk.decode(ids), JSON.stringify(acc.slice(0, 40)))

  // JSON constrained decoding against the real vocabulary: token bytes must reconstruct the
  // exact UTF-8 of the text, and a real-tokenized JSON doc must drive the machine to complete.
  const table = new TokenByteTable(tk)
  const doc = '{"name":"Ayşe 👍🏽","n":-1.5e+3,"list":[true,null,{"k":"v"}],"note":"esc \\" \\u00e9 done"}'
  const docIds = tk.encode(doc, false)
  const encBytes = new TextEncoder().encode(doc)
  const gathered: number[] = []
  for (const tid of docIds) {
    const b = table.bytes(tid)
    if (b) gathered.push(...b)
  }
  check('token bytes reconstruct exact UTF-8 (real vocab)', gathered.length === encBytes.length && gathered.every((v, i) => v === encBytes[i]), `${gathered.length}/${encBytes.length} bytes`)
  const jm = new JsonMachine()
  let jmOk = true
  for (const tid of docIds) {
    const b = table.bytes(tid)
    if (!b || !jm.feed(b)) { jmOk = false; break }
  }
  check('machine accepts a real-tokenized JSON doc token-by-token', jmOk && jm.complete)
}

console.log(failures === 0 ? '\nALL CHAT CHECKS PASSED' : `\n${failures} CHAT CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
