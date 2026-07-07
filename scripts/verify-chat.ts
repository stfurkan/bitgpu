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
import { createChat, ChatTokenizer, ThinkSplitter, type ChatMessage } from '../src/chat/index'
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

const CHATML_TEMPLATE = [
  "{%- for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}",
  "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- if not enable_thinking %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}",
].join('')

function miniTokenizer(): { json: unknown; config: Record<string, unknown> } {
  const b2u = bytesToUnicode()
  const vocab: Record<string, number> = {}
  for (const [, ch] of b2u) if (!(ch in vocab)) vocab[ch] = Object.keys(vocab).length
  // Specialness mirrors the real Qwen3 tokenizer: the ChatML markers are special (stripped from
  // decoded text) while <think>/</think> are NOT (they reach the decoder stream, where the
  // ThinkSplitter routes them - exactly how the real model behaves).
  const specials = ['<|im_start|>', '<|im_end|>', '<think>', '</think>', '<|endoftext|>']
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
})()

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

  // decoder stream integrity on real BPE with multi-byte content
  const sample = 'Emoji 👍🏽 flag 🇹🇷 combining é́ CJK 世界 done.'
  const ids = tk.encode(sample, false)
  const st = tk.createDecoderStream(true)
  let acc = ''
  for (const id of ids) acc += st.push(id)
  acc += st.flush()
  check('decoder stream == full decode (real vocab)', acc === tk.decode(ids), JSON.stringify(acc.slice(0, 40)))
}

console.log(failures === 0 ? '\nALL CHAT CHECKS PASSED' : `\n${failures} CHAT CHECK(S) FAILED`)
process.exit(failures === 0 ? 0 : 1)
