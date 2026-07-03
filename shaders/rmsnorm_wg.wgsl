// RMSNorm, no-subgroup fallback: one workgroup per row; threads split D and tree-reduce the
// sum of squares via shared memory. Replaces the one-thread-per-row kernel on this path: at
// decode (R=1) that kernel walked 2xD elements serially on a single thread, latency-bound,
// and it ran twice per layer - the dominant cost of the whole fallback decode step.
// Mirrors rmsnorm_sg exactly, with subgroupAdd swapped for the shared-memory reduction.
override WG: u32 = 64u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;   // [D]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [R, D]
var<workgroup> sdata: array<f32, 256>;                        // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.R) { return; }
  let tid = lid.x;
  let base = row * p.D;
  var s = 0.0;
  for (var i = tid; i < p.D; i = i + WG) { let v = x[base + i]; s = s + v * v; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.D) + p.eps);
  for (var i = tid; i < p.D; i = i + WG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
