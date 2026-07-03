// Small-batch (M = 2..9) subgroup split-K GEMV with fused residual add (o_proj / down_proj in
// the speculative-decode verify pass). One workgroup per output column; each weight word is
// loaded once and dotted with all M activation rows. Per row the loop stride and accumulation
// expression match the validated M=1 kernels, so results are row-wise bit-identical to them.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    let s = scales[sbase + (k / 128u)];
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], sv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total + resid[m * p.N + n]; }
  }
}
