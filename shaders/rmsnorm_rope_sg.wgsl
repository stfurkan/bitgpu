// Fused per-head RMSNorm + RoPE for decode (S=1). One subgroup (= one workgroup) per head row;
// lanes split head_dim, reduce sum-of-squares with subgroupAdd, then apply rope. rotate_half
// pairs (d, d+-D/2): with SG>=32 and D=128 a lane owns d in {lane, lane+32, lane+64, lane+96},
// so every (d, d+-64) pair is held by the same lane (no cross-lane reads for the rotate).
// outOff/outStride let the K result write straight into the KV cache at its position.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [outOff + R*outStride]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let ob = p.outOff + row * p.outStride;
  for (var i = lane; i < p.D; i = i + SG) {
    let nd = x[base + i] * inv * gamma[i];
    var pd: u32; var sgn: f32;
    if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
    let rot = sgn * (x[base + pd] * inv * gamma[pd]);
    y[ob + i] = nd * cos[i] + rot * sin[i];
  }
}
