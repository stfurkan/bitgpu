// f16-activation variant of matmul_split_sg (fused QKV decode GEMV, M=1). The activation x is
// read as f16 and the per-group dot runs in f16 (2x ALU rate on Apple/AMD/recent NVIDIA); the
// per-block accumulation stays f32 (dot promoted before x scale, acc in f32) so accuracy tracks
// f32 to ~f16 rounding. Weights (sign bits + f32 block scales) are unchanged. Outputs f32.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f16>(select(-1.0h, 1.0h, (bits4 & 1u) != 0u), select(-1.0h, 1.0h, (bits4 & 2u) != 0u),
                       select(-1.0h, 1.0h, (bits4 & 4u) != 0u), select(-1.0h, 1.0h, (bits4 & 8u) != 0u));
    acc = acc + f32(dot(x[gi], sv)) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
