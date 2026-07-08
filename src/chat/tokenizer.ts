/*!
 * bitgpu/chat bundles @huggingface/tokenizers and @huggingface/jinja (Apache-2.0,
 * (c) Hugging Face) at build time so the published package stays dependency-free.
 * See THIRD_PARTY_LICENSES.md in the package root.
 */
// Tokenizer + chat-template layer: wraps the canonical HF tokenizer (@huggingface/tokenizers,
// the same byte-level BPE transformers.js uses) and the canonical template engine
// (@huggingface/jinja). The engine itself stays tokenizer-free (token ids in, ids out); this
// module is the text boundary, verified byte-exact against transformers.js (scripts/verify-chat.ts).
import { Tokenizer } from '@huggingface/tokenizers'
import { Template } from '@huggingface/jinja'

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | (string & {})
  content: string
}

/** Incremental decoder for streaming generation: feed token ids as they arrive, get the newly
 *  stable visible text each step. Re-decodes the running sequence and emits the delta, holding
 *  back a trailing incomplete multi-byte character (which byte-level BPE decodes as U+FFFD)
 *  until its remaining tokens arrive. */
export interface DecoderStream {
  push(tokenId: number): string
  flush(): string
}

const tokenString = (v: unknown): string | null =>
  typeof v === 'string' ? v : ((v as { content?: string } | null | undefined)?.content ?? null)

export class ChatTokenizer {
  private readonly tok: Tokenizer
  private readonly template: Template | null
  private readonly templateContext: Record<string, unknown>
  /** End-of-sequence token id (e.g. <|im_end|> for Qwen3-family models). */
  readonly eosTokenId: number
  /** The eos token's string form (used to reconstruct the template's turn terminator). */
  readonly eosToken: string

  constructor(tokenizerJson: unknown, tokenizerConfig: Record<string, unknown>) {
    this.tok = new Tokenizer(tokenizerJson as never, tokenizerConfig as never)
    const tmpl = tokenizerConfig['chat_template']
    this.template = typeof tmpl === 'string' ? new Template(tmpl) : null
    // transformers.js exposes the special tokens to the template context; templates that
    // reference bos_token/eos_token render identically with these present.
    this.templateContext = {}
    for (const k of ['bos_token', 'eos_token', 'pad_token', 'unk_token']) {
      const s = tokenString(tokenizerConfig[k])
      if (s !== null) this.templateContext[k] = s
    }
    const eosStr = tokenString(tokenizerConfig['eos_token'])
    const eosId = eosStr !== null ? this.tok.token_to_id(eosStr) : undefined
    if (eosStr === null || eosId === undefined)
      throw new Error('bitgpu/chat: tokenizer_config.json has no resolvable eos_token (needed to stop generation and to reconstruct cached turns)')
    this.eosToken = eosStr
    this.eosTokenId = eosId
  }

  /** Encode text to token ids. `addSpecialTokens` defaults to false: the chat template already
   *  inserts the control tokens, so prompt/delta encoding must not add more. */
  encode(text: string, addSpecialTokens = false): number[] {
    return Array.from(this.tok.encode(text, { add_special_tokens: addSpecialTokens }).ids, Number)
  }

  /** Decode token ids to text. `skipSpecialTokens` defaults to true (never surface control tokens). */
  decode(ids: number[], skipSpecialTokens = true): string {
    if (ids.length === 0) return '' // decode([]) throws in @huggingface/tokenizers
    return this.tok.decode(ids, { skip_special_tokens: skipSpecialTokens })
  }

  /** The raw vocab string for a token id (byte-alias space for byte-level BPE). */
  idToToken(id: number): string | undefined {
    return this.tok.id_to_token(id)
  }

  /** Ids of all added tokens (ChatML markers, <think>, etc.) - never plain content. */
  addedTokenIds(): Set<number> {
    return new Set(this.tok.get_added_tokens_decoder().keys())
  }

  get hasChatTemplate(): boolean {
    return this.template !== null
  }

  /** Render a message list to a prompt string via the model's own Jinja chat template
   *  (matches transformers.js apply_chat_template byte-exactly). */
  applyChatTemplate(messages: ChatMessage[], opts: { addGenerationPrompt?: boolean; enableThinking?: boolean } = {}): string {
    if (!this.template) throw new Error('bitgpu/chat: the model has no chat_template in tokenizer_config.json')
    return this.template.render({
      ...this.templateContext,
      messages,
      add_generation_prompt: opts.addGenerationPrompt ?? true,
      enable_thinking: opts.enableThinking ?? false,
    })
  }

  createDecoderStream(skipSpecialTokens = true): DecoderStream {
    const ids: number[] = []
    let emitted = 0
    const decodeAll = (): string => this.tok.decode(ids, { skip_special_tokens: skipSpecialTokens })
    return {
      push: (tokenId: number): string => {
        ids.push(tokenId)
        const text = decodeAll()
        let safe = text.length
        while (safe > emitted && text.charCodeAt(safe - 1) === 0xfffd) safe-- // hold back an incomplete trailing char
        const out = text.slice(emitted, safe)
        emitted = safe
        return out
      },
      flush: (): string => {
        if (ids.length === 0) return '' // an aborted turn can flush before its first token
        const text = decodeAll()
        const out = text.slice(emitted)
        emitted = text.length
        return out
      },
    }
  }
}
