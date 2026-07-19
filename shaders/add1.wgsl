// In-place x += 1.0. Bakes the qwen3.5 plain-RMSNorm scale (1 + weight) into the norm weights at
// load, so the existing RMSNorm kernels (which multiply by the weight directly) stay unchanged.
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  x[i] = x[i] + 1.0;
}
