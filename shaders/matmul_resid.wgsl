// Binary matmul with a fused residual add: y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N].
// Folds the residual add into o_proj / down_proj so it's not a separate dispatch.
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let n = idx % p.N;
  let xRow = (idx / p.N) * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < 4u; w = w + 1u) {
      let word = signbits[wRow + b * 4u + w];
      let xb = xRow + b * 32u + w * 8u;
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                           select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc + resid[idx];
}
