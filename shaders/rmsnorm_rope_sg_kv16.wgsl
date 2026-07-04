// rmsnorm_rope_sg writing into an f16-STORAGE KV cache (kvCache: 'f16'): used ONLY for the K
// projection on the fused decode path, where the normed+roped K is written straight into the
// cache. Keep in lockstep with rmsnorm_rope_sg.wgsl: the ONLY difference is y is array<f16>
// (one f32 -> f16 rounding at the write). The q call keeps the f32 kernel.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f16>;  // [outOff + R*outStride]

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
    y[ob + i] = f16(nd * cos[i] + rot * sin[i]);
  }
}
