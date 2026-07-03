// Tiled register-blocked binary GEMM with fused residual, for PREFILL (M>1), vec4 K-accumulation:
//   y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N],  W binary {-1,+1} sign-packed, per-128-block fp32 scale.
// 64x64 output tile per workgroup, 16x16 threads each computing a 4x4 register tile, BK=16 K-step.
// Activation + decoded/scaled weight tiles are staged in shared memory as vec4 (4 K per element);
// each inner step is a dot() of vec4s, and one weight load decodes a whole nibble (4 signs) at once.
// No subgroup ops -> all devices. Near-bit-exact (f32 accum; tiled K-order differs in last ULPs).
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, N: u32, K: u32, nb: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;    // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [M, N]

var<workgroup> xs: array<vec4<f32>, 256>;   // BM*BKV
var<workgroup> ws: array<vec4<f32>, 256>;   // BN*BKV

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {           // stage activation tile (vec4)
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {           // stage decoded+scaled weight tile (vec4)
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < p.N) {
        let bits4 = (signbits[gn * (p.K / 32u) + (k >> 5u)] >> (k & 31u)) & 0xfu;
        let s = scales[gn * p.nb + (k / 128u)];
        wv = vec4<f32>(select(-s, s, (bits4 & 1u) != 0u), select(-s, s, (bits4 & 2u) != 0u),
                       select(-s, s, (bits4 & 4u) != 0u), select(-s, s, (bits4 & 8u) != 0u));
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm < p.M) {
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let gn = tileN + tc + tn;
        if (gn < p.N) { let idx = gm * p.N + gn; y[idx] = acc[tm * 4u + tn] + resid[idx]; }
      }
    }
  }
}
