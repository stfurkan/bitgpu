// Subgroup split-K GEMV for decode (M=1), fused: one subgroup (= one workgroup) per output
// column; lanes split the K dimension and reduce with subgroupAdd (register-only, no barriers).
// Cuts each matmul's latency ~SG-fold vs one-thread-per-output (the real decode bottleneck:
// kernels run at full latency in the dependent chain). Routes to out0/out1/out2 by range (qkv / gate-up).
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
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
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
