// Small-batch (M = 2..9) subgroup split-K GEMV, fused qkv / gate-up. The speculative-decode
// verify pass computes M drafted rows in one forward; the scalar prefill kernels re-read the
// weights per output thread, so a k-row pass cost ~k GEMVs. Here each weight word is loaded
// ONCE per (column, k-chunk) and dotted with all M activation rows (activations are ~8 KB/row,
// cache-resident). Per row the loop stride and accumulation expression are IDENTICAL to
// matmul_split_sg, so each row's partials - and therefore the subgroupAdd result - match the
// M=1 decode path bit-for-bit.
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>; // [M, N0]
@group(0) @binding(5) var<storage, read_write> out1: array<f32>; // [M, N1]
@group(0) @binding(6) var<storage, read_write> out2: array<f32>; // [M, N2]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; } // uniform per workgroup: the whole subgroup exits together
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
  for (var m = 0u; m < p.M; m = m + 1u) { // p.M is uniform: collective calls stay uniform
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) {
      if (n < p.N0) { out0[m * p.N0 + n] = total; }
      else if (n < p.N0 + p.N1) { out1[m * p.N1 + (n - p.N0)] = total; }
      else { out2[m * p.N2 + (n - p.N0 - p.N1)] = total; }
    }
  }
}
