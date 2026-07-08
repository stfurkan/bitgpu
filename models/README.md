# Ready-made bitgpu models (zero conversion)

Each directory holds the two SMALL files bitgpu needs - `manifest.json` (the architecture
contract + tensor byte ranges) and the aux file (norm gammas + lookup tables, ~37 KB) - produced
with [`tools/convert.py`](../tools/README.md) from the public onnx-community exports. The big
weights file is **not** here: it streams straight from the Hugging Face Hub, byte-for-byte
unchanged, so nothing needs re-hosting and there is nothing to convert.

| model | hidden / layers | weights | data file (streams from the Hub) |
| --- | --- | --- | --- |
| `bonsai-1.7b` | 2048 / 28 | ~290 MB | [onnx-community/Bonsai-1.7B-ONNX](https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX) |
| `bonsai-4b` | 2560 / 36 | ~660 MB | [onnx-community/Bonsai-4B-ONNX](https://huggingface.co/onnx-community/Bonsai-4B-ONNX) |
| `bonsai-8b` | 4096 / 36 | ~1.3 GB | [onnx-community/Bonsai-8B-ONNX](https://huggingface.co/onnx-community/Bonsai-8B-ONNX) |

Use them from any static host or a CDN over this repo (pin a tag for immutable caching):

```ts
import { createEngine } from 'bitgpu'
import { createChat } from 'bitgpu/chat'

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@v0.7.0/models/bonsai-1.7b'
const HF = 'https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main'

const engine = await createEngine({
  manifestUrl: `${REPO}/manifest.json`,
  auxUrl: `${REPO}/bonsai.aux.bin`, // model_q1.aux.bin for the 4B / 8B dirs
  dataUrl: `${HF}/onnx/model_q1.onnx_data`,
})
const chat = await createChat(engine, {
  tokenizerJsonUrl: `${HF}/tokenizer.json`,
  tokenizerConfigUrl: `${HF}/tokenizer_config.json`,
})
```

Or copy a directory next to your app and pass `modelUrl` (serve the tokenizer files alongside
and the whole thing is first-party, no third-party requests at all).

These files are static and versioned with the engine; regenerating them from the exports
reproduces them byte-for-byte (`python tools/convert.py --model <dir>`).
