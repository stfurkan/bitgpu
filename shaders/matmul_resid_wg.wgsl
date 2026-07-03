// No-subgroup fallback: split-K GEMV for decode (M=1) with fused residual, workgroup-shared-memory
// reduction. One workgroup per output column; WG threads split K and tree-reduce. Used for o_proj/down.
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
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
  if (tid == 0u) { y[n] = sdata[0] + resid[n]; }
}
