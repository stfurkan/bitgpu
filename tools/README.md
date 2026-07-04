# Model tooling

Offline Python tools that turn a 1-bit ONNX export into a bitgpu model and produce the
reference data the verification page checks against. Nothing here ships to the browser.

```sh
pip install numpy onnx onnxruntime tokenizers
```

Gather a work dir with the export's `config.json`, `tokenizer.json`, the `.onnx` graph and its
external data file (for Bonsai: `onnx/model_q1.onnx` + `onnx/model_q1.onnx_data` from
`onnx-community/Bonsai-1.7B-ONNX`). Then:

```sh
# 1. manifest.json + aux: the two small files the engine loads (see docs/FORMAT.md)
python tools/convert.py --model <dir> [--aux-name bonsai.aux.bin] [--ref-url <f16 safetensors url>]

# 2. golden logits from the ORIGINAL export (onnxruntime CPU), the ground truth
python tools/golden.py --model <dir>

# 3. numpy forward pass straight from the manifest, checked against the golden logits;
#    --dump writes the fixtures examples/verify.html reads
python tools/reference.py --model <dir> --dump test-fixtures/forward
```

After step 3, serve the repo (`python3 -m http.server 8000`), point `examples/model` at the
work dir, and run `npm run verify:headless`: the engine must reproduce the reference forward
(cosine) and, for the committed Bonsai fixtures, the exact known-good token ids.

The shipped Bonsai manifest + aux were reproduced byte-for-byte from the public export with
`convert.py`, so the tool is the authoritative producer of the format.
