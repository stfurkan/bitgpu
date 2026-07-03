// Tiled register-blocked binary GEMM to 3 outputs (qkv or gate/up), PREFILL (M>1), vec4 K-accum.
// Weights concatenated along N (N0|N1|N2); each 64-wide tile lies in one range (N0,N1 multiples of 64)
// so a workgroup routes its whole tile to out0/out1/out2. Same vec4 design as matmul_resid_tiled.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
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
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < Ntot) {
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
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= Ntot) { continue; }
      let v = acc[tm * 4u + tn];
      if (gn < p.N0) { out0[gm * p.N0 + gn] = v; }
      else if (gn < p.N0 + p.N1) { out1[gm * p.N1 + (gn - p.N0)] = v; }
      else { out2[gm * p.N2 + (gn - p.N0 - p.N1)] = v; }
    }
  }
}
