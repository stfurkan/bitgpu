// Gated RMSNorm for the DeltaNet output: y = gamma * rmsnorm(core) * silu(z), normalized over the
// value head dim (one workgroup per head-vector row). Unlike the model's plain RMSNorm this uses
// the weight directly (not 1+weight), matching tools/qwen35_numpy (Qwen3NextRMSNormGated).
override WG: u32 = 128u;
struct Params { rows: u32, DV: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> core: array<f32>;   // [rows, DV]
@group(0) @binding(2) var<storage, read> z: array<f32>;      // [rows, DV] gate
@group(0) @binding(3) var<storage, read> gamma: array<f32>;  // [DV]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;// [rows, DV]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let tid = lid.x;
  let base = row * p.DV;
  var s = 0.0;
  for (var i = tid; i < p.DV; i = i + WG) { let c = core[base + i]; s = s + c * c; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.DV) + p.eps);
  for (var i = tid; i < p.DV; i = i + WG) {
    let zz = z[base + i];
    y[base + i] = gamma[i] * (core[base + i] * inv) * (zz / (1.0 + exp(-zz)));  // * silu(z)
  }
}
