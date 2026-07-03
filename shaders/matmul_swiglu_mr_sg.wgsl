// Multi-row fused gate/up GEMV + SwiGLU for decode (M=1). Each workgroup computes ROWS
// intermediate indices; per K-step it issues 2*ROWS independent weight loads (gate row n and up
// row F+n for each of the ROWS) before the dots, giving the bandwidth-bound decode GEMV more
// in-flight memory requests. One subgroup per workgroup; lanes split K; reduced with subgroupAdd.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [F]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f32>(select(-1.0, 1.0, (gw & 1u) != 0u), select(-1.0, 1.0, (gw & 2u) != 0u),
                           select(-1.0, 1.0, (gw & 4u) != 0u), select(-1.0, 1.0, (gw & 8u) != 0u));
        g[r] = g[r] + dot(xv, gv) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f32>(select(-1.0, 1.0, (uw & 1u) != 0u), select(-1.0, 1.0, (uw & 2u) != 0u),
                           select(-1.0, 1.0, (uw & 4u) != 0u), select(-1.0, 1.0, (uw & 8u) != 0u));
        u[r] = u[r] + dot(xv, uv) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = (gt / (1.0 + exp(-gt))) * ut; }
  }
}
