// RMSNorm, subgroup-parallel: one subgroup (= one workgroup) per row; lanes split D and
// reduce the sum-of-squares with subgroupAdd (register-only, no barriers/shared memory).
// Fixes the decode bottleneck where R=1 ran on a single thread. SG is set from the device's
// subgroup size at pipeline creation; requires workgroup_size == subgroup size.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);                 // sum across the subgroup, broadcast to all lanes
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
