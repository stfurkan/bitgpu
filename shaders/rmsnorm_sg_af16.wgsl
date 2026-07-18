// RMSNorm (subgroup) that writes the normalized activation as f16 - the input side of the
// f16-activation decode matmuls (activation: 'f16'). Reads the f32 residual stream; the
// sum-of-squares reduction stays f32 (accuracy); only the stored output is rounded to f16.
// Identical reduction to rmsnorm_sg, so it is bit-comparable up to the final f16 rounding.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = f16(x[base + i] * inv * gamma[i]); }
}
