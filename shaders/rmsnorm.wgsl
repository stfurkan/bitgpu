// RMSNorm: y[r,d] = x[r,d] / sqrt(mean_d(x^2) + eps) * gamma[d]. One invocation per row.
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;   // [D]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [R, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let r = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (r >= p.R) { return; }
  let base = r * p.D;
  var ss = 0.0;
  for (var d = 0u; d < p.D; d = d + 1u) {
    let v = x[base + d];
    ss = ss + v * v;
  }
  let inv = inverseSqrt(ss / f32(p.D) + p.eps);
  for (var d = 0u; d < p.D; d = d + 1u) {
    y[base + d] = x[base + d] * inv * gamma[d];
  }
}
