// Small-batch (M = 2..9) subgroup split-K GEMV for the 2-bit lm_head: the speculative-decode
// verify pass needs logits for every drafted row, and the scalar M-row kernel re-reads the
// ~77 MB code stream per output thread. Here each code word is loaded once per (column,
// k-chunk) and dotted with all M rows. Per row the loop stride and accumulation expression
// match matmul_q2_sg, so each row is bit-identical to the M=1 decode path.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, M: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    let s = scales[sbase + (gi >> 5u)]; // block = (gi*4)/128 = gi/32
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], cv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total; }
  }
}
