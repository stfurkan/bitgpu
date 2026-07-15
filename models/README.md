# Ready-made bitgpu models (zero conversion)

Each directory holds the two SMALL files bitgpu needs - `manifest.json` (the architecture
contract + tensor byte ranges) and the aux file (lookup tables; for the ONNX dirs also norm
gammas) - produced with [`tools/convert-onnx.py` or `tools/convert-gguf.py`](../tools/README.md)
from the public exports. The big weights file is **not** here: it streams straight from the
Hugging Face Hub, byte-for-byte unchanged, so nothing needs re-hosting and there is nothing to
convert.

| model | hidden / layers | weights | data file (streams from the Hub) |
| --- | --- | --- | --- |
| `bonsai-1.7b` | 2048 / 28 | ~290 MB | [onnx-community/Bonsai-1.7B-ONNX](https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX) |
| `bonsai-4b` | 2560 / 36 | ~660 MB | [onnx-community/Bonsai-4B-ONNX](https://huggingface.co/onnx-community/Bonsai-4B-ONNX) |
| `bonsai-8b` | 4096 / 36 | ~1.3 GB | [onnx-community/Bonsai-8B-ONNX](https://huggingface.co/onnx-community/Bonsai-8B-ONNX) |
| `bonsai-1.7b-gguf` | 2048 / 28 | ~250 MB | [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) (`Bonsai-1.7B-Q1_0.gguf`) |
| `bonsai-4b-gguf` | 2560 / 36 | ~580 MB | [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf) (`Bonsai-4B-Q1_0.gguf`) |
| `bonsai-8b-gguf` | 4096 / 36 | ~1.16 GB | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) (`Bonsai-8B-Q1_0.gguf`) |

The ONNX and GGUF containers carry the **same weights** (the sign-bit streams are
bit-identical; verified on Bonsai-8B): pick by hosting preference. The GGUF files are a
little smaller on the wire and their manifests synthesize RoPE at load (see
[docs/FORMAT.md](../docs/FORMAT.md)).

Use them from any static host or a CDN over this repo (pin a tag for immutable caching):

```ts
import { createEngine } from 'bitgpu'
import { createChat } from 'bitgpu/chat'

const REPO = 'https://cdn.jsdelivr.net/gh/stfurkan/bitgpu@v0.11.0/models/bonsai-1.7b'
const HF = 'https://huggingface.co/onnx-community/Bonsai-1.7B-ONNX/resolve/main'

const engine = await createEngine({
  manifestUrl: `${REPO}/manifest.json`,
  auxUrl: `${REPO}/bonsai.aux.bin`, // model_q1.aux.bin for the 4B / 8B dirs
  dataUrl: `${HF}/onnx/model_q1.onnx_data`,
})
// or the GGUF flavor of the same weights (tokenizer files still come from an ONNX/source repo):
//   REPO = '.../models/bonsai-1.7b-gguf'  auxUrl: `${REPO}/Bonsai-1.7B-Q1_0.aux.bin`
//   dataUrl: 'https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf'
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
them byte-for-byte (`python tools/convert-onnx.py --model <dir>`, full pipeline in
[tools/README.md](../tools/README.md)).
