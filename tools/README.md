# Model tooling

Offline Python tools that turn a 1-bit export - an ONNX export or a Q1_0 GGUF - into a
bitgpu model and produce the reference data the verification page checks against. Nothing
here ships to the browser.

```sh
pip install numpy onnx onnxruntime tokenizers   # GGUF conversion needs only numpy
```

## From a 1-bit ONNX export

Gather a work dir with the export's `config.json`, `tokenizer.json`, the `.onnx` graph and its
external data file (for Bonsai: `onnx/model_q1.onnx` + `onnx/model_q1.onnx_data` from
`onnx-community/Bonsai-1.7B-ONNX`). Then:

```sh
# 1. manifest.json + aux: the two small files the engine loads (see docs/FORMAT.md)
python tools/convert-onnx.py --model <dir> [--aux-name bonsai.aux.bin] [--ref-url <f16 safetensors url>]

# 2. golden logits from the ORIGINAL export (onnxruntime CPU), the ground truth
python tools/golden.py --model <dir>

# 3. numpy forward pass straight from the manifest, checked against the golden logits;
#    --dump writes the fixtures examples/verify.html reads
python tools/reference.py --model <dir> --dump test-fixtures/forward-<tag>
```

## From a 1-bit GGUF (PrismML Q1_0)

One file in, two small files out - the GGUF itself stays the data file, streamed
byte-for-byte from wherever it is hosted:

```sh
# 1. manifest.json (v2, q1_0 containers) + aux (just the two LUTs, ~1.5 KB)
python tools/convert-gguf.py --gguf <dir>/Bonsai-8B-Q1_0.gguf [--ref-url <f16 safetensors url>]

# 2. there is no onnxruntime oracle for a GGUF; generate the fixtures directly from the
#    manifest (--ids reuses another fixture set's prompt so checkpoints stay comparable)
python tools/reference.py --model <dir> --ids test-fixtures/forward-8b/ids.i32.bin \
                          --dump test-fixtures/forward-<tag>
```

GGUF repos usually carry no `tokenizer.json`; take it from the model's source repo on the Hub
(the chat layer needs `tokenizer.json` + `tokenizer_config.json` next to the manifest, or
explicit URLs).

## Gating a converted model

After the dump, point `examples/model-<tag>` at the work dir (the reference Bonsai-1.7B is tag
`1.7b`) and run `npm run verify:headless` (it serves the repo itself): the engine must
reproduce the reference forward (cosine) and, for the committed Bonsai fixtures, the exact
known-good token ids (record the engine's greedy continuation as `known_good` in the new set's
`params.json` on first run).

The shipped Bonsai manifests + aux files were reproduced byte-for-byte from the public exports
with these tools, so they are the authoritative producers of the format.
