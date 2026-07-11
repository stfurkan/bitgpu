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

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@v0.8.0/models/bonsai-1.7b'
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

Or copy a directory into your app's static assets and own the files (they are just static
files - your repo becomes their source control). Two variants:

- copy **manifest + aux only** and keep `dataUrl` pointed at the Hub: two small first-party
  files, weights still stream from HF;
- also download the data file (and tokenizer files) into the same directory for a **fully
  first-party, offline-capable** setup with no third-party requests - then the single-URL form
  works, since the manifest names its data/aux files and the engine resolves them relative to
  the directory: `createEngine({ modelUrl: '/models/bonsai-1.7b' })`.

These files are intentionally **not** shipped in the npm package: bitgpu stays a lean,
model-neutral engine - hotlink the pinned CDN URLs above, or copy the files and own them.
They are static and versioned with the engine; regenerating them from the exports reproduces
them byte-for-byte (`python tools/convert.py --model <dir>`, full pipeline in
[tools/README.md](../tools/README.md)).
