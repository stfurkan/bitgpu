// No-subgroup fallback: split-K GEMV for decode (M=1), workgroup-shared-memory reduction instead
// of subgroupAdd. One workgroup per output column; WG threads split K and tree-reduce via shared
// memory + barriers. ~WG-fold faster than one-thread-per-output (the v1 path). Routes qkv / gate-up.
override WG: u32 = 64u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;
var<workgroup> sdata: array<f32, 256>;                          // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;          // uniform across the workgroup -> early return is barrier-safe
  if (n >= Ntot) { return; }
  let tid = lid.x;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let total = sdata[0];
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
