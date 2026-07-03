import { defineConfig } from 'tsdown'

// ESM-only library build (2026 default: Node 23+ can require() ESM, so no CJS).
// Zero runtime deps -> nothing to externalize; the WGSL is inlined via the
// generated shaders module (scripts/gen-shaders.ts runs before this).
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  target: 'es2023',
  platform: 'browser',
  dts: true,
  clean: true,
  treeshake: true,
  sourcemap: true,
})
