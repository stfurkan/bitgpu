// bitgpu in a Web Worker: the engine and chat layer live entirely off the main thread, so the
// page never janks during prefill or decode (WebGPU is available in workers in every browser
// bitgpu supports). The protocol is four messages in (load / send / stop / reset) and four out
// (progress / delta / result / error) - copy this file and adjust.
import { createEngine } from 'https://esm.sh/bitgpu@0.9.0'
import { createChat } from 'https://esm.sh/bitgpu@0.9.0/chat'

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@main/models/bonsai-1.7b'
const HF = 'https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main'

let engine = null
let chat = null
let ctl = null

onmessage = async ({ data: m }) => {
  try {
    if (m.type === 'load' && !chat) {
      engine = await createEngine({
        manifestUrl: `${REPO}/manifest.json`,
        auxUrl: `${REPO}/bonsai.aux.bin`,
        dataUrl: `${HF}/onnx/model_q1.onnx_data`,
        onProgress: (p) => postMessage({ type: 'progress', phase: p.phase, loaded: p.loaded ?? 0, total: p.total ?? 0 }),
      })
      chat = await createChat(engine, {
        tokenizerJsonUrl: `${HF}/tokenizer.json`,
        tokenizerConfigUrl: `${HF}/tokenizer_config.json`,
      })
      postMessage({ type: 'progress', phase: 'ready' })
    } else if (m.type === 'send' && chat) {
      ctl = new AbortController()
      const r = await chat.send(m.messages, {
        temperature: 0.7,
        topK: 20,
        maxTokens: 512,
        signal: ctl.signal,
        onText: (delta) => postMessage({ type: 'delta', delta }),
      })
      postMessage({ type: 'result', text: r.text, tokensPerSecond: r.tokensPerSecond, finishReason: r.finishReason })
    } else if (m.type === 'stop') {
      ctl?.abort()
    } else if (m.type === 'reset') {
      chat?.reset()
    }
  } catch (err) {
    postMessage({ type: 'error', message: String(err?.message ?? err) })
  }
}
