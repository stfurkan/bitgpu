// Fused binary matmul writing to up to 3 output buffers (qkv or gate/up in one dispatch).
// Weights for the outputs are concatenated along N (rows N0 | N1 | N2). One thread per
// output column n routes its result to out0/out1/out2 by range. Vectorized like matmul_binary_vec4.
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * Ntot) { return; }
  let row = idx / Ntot;
  let n = idx % Ntot;
  let xRow = row * (p.K / 4u);
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

  if (n < p.N0) { out0[row * p.N0 + n] = acc; }
  else if (n < p.N0 + p.N1) { out1[row * p.N1 + (n - p.N0)] = acc; }
  else { out2[row * p.N2 + (n - p.N0 - p.N1)] = acc; }
}
