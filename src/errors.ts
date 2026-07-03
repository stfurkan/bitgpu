/** Thrown when WebGPU is unavailable: no `navigator.gpu` (unsupported browser, or a
 *  non-secure context) or no adapter could be acquired. Catch this to render a
 *  "your browser doesn't support WebGPU yet" fallback instead of crashing. */
export class WebGPUUnavailableError extends Error {
  override readonly name = 'WebGPUUnavailableError'
  constructor(message: string) {
    super(message)
  }
}

/** Thrown when the GPU reports out-of-memory while the engine allocates weights or grows the
 *  KV cache. Without this check the allocation "succeeds" with invalid buffers and every later
 *  generate() returns garbage. Catch it to fall back (smaller maxSeqLen, or a no-LLM mode). */
export class GpuOutOfMemoryError extends Error {
  override readonly name = 'GpuOutOfMemoryError'
  constructor(message: string) {
    super(message)
  }
}
