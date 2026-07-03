// Multi-row subgroup GEMV for decode (M=1) with fused residual. Same as matmul_resid_sg but each
// workgroup computes ROWS output columns at once: per K-step it issues ROWS independent weight
// loads before the dots, giving the memory system more in-flight requests (memory-level
// parallelism) to hide latency on the bandwidth-bound decode GEMV. One subgroup per workgroup;
// lanes split K; ROWS accumulators reduced with subgroupAdd. value = sign * per-block scale.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var acc: array<f32, 8>;                         // ROWS <= 8
  for (var r = 0u; r < ROWS; r = r + 1u) { acc[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let w = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (w & 1u) != 0u), select(-1.0, 1.0, (w & 2u) != 0u),
                           select(-1.0, 1.0, (w & 4u) != 0u), select(-1.0, 1.0, (w & 8u) != 0u));
        acc[r] = acc[r] + dot(xv, sv) * scales[n * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = rowBase + r;
    let total = subgroupAdd(acc[r]);             // collective: called for every r by all lanes
    if (lane == 0u && n < p.N) { y[n] = total + resid[n]; }
  }
}
