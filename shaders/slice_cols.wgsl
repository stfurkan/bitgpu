// Extract a contiguous column range [off, off+w) from each row of a [rows, stride] buffer into a
// packed [rows, w] buffer. Splits the DeltaNet conv output (q|k|v concatenated per token) into the
// separate q/k/v activation buffers the scan reads.
struct Params { rows: u32, w: u32, stride: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [rows, stride]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [rows, w]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.rows * p.w) { return; }
  let r = i / p.w;
  let c = i % p.w;
  dst[i] = src[r * p.stride + p.off + c];
}
