import { defineConfig } from 'tsdown'

// ESM-only library build (2026 default: Node 23+ can require() ESM, so no CJS).
// Three entries: the zero-dependency engine (`bitgpu`), the chat layer (`bitgpu/chat`), and
// the in-browser GGUF header parser (`bitgpu/gguf`, types-only dependency on the engine).
// The chat entry INLINES @huggingface/tokenizers + @huggingface/jinja (pure JS, Apache-2.0)
// the same way the engine inlines its WGSL (scripts/gen-shaders.ts runs before this): the
// published package stays zero-runtime-dependency, and importing plain `bitgpu` never loads
// or bundles any of the chat code.
export default defineConfig({
  entry: { index: 'src/index.ts', chat: 'src/chat/index.ts', gguf: 'src/gguf.ts' },
  format: ['esm'],
  target: 'es2023',
  platform: 'browser',
  dts: true,
  clean: true,
  treeshake: true,
  sourcemap: true,
  noExternal: ['@huggingface/tokenizers', '@huggingface/jinja'],
})
