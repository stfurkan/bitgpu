// Gated DeltaNet recurrent scan (the sequential O(1)/token gated delta rule; bitgpu's decode
// path, and a correctness reference for prefill). One workgroup per head; thread `dv` owns value
// column dv of the per-head state S[dk,dv], held in registers across the token loop. Per token:
//   S *= exp(g);  kv = Kn·S;  delta = (v - kv)·beta;  S += Kn⊗delta;  out = Qn·S
// with Kn = l2norm(k), Qn = l2norm(q)/sqrt(dk) (matches tools/qwen35_numpy._delta_recurrent).
// Inputs are the raw (post-conv, post-projection) q/k/v; g/beta are per (token, head).
override WGV: u32 = 128u;                 // threads per workgroup == head_v_dim (dv)
struct Params { S: u32, H: u32, DK: u32, DV: u32, HK: u32, betaOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;     // [S, HK, DK]
@group(0) @binding(2) var<storage, read> k: array<f32>;     // [S, HK, DK]
@group(0) @binding(3) var<storage, read> v: array<f32>;     // [S, H, DV]
@group(0) @binding(4) var<storage, read> g: array<f32>;     // [S, H]
@group(0) @binding(5) var<storage, read> beta: array<f32>;  // [S, H]
@group(0) @binding(6) var<storage, read_write> core: array<f32>; // [S, H, DV]
var<workgroup> ksh: array<f32, 128>;      // current token's raw k (>= DK)
var<workgroup> qsh: array<f32, 128>;      // current token's raw q

@compute @workgroup_size(WGV)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;                           // value head
  let hk = h / (p.H / p.HK);              // GQA: shared key/query head (repeat-interleave)
  let dv = lid.x;
  let DK = p.DK;
  let scale = inverseSqrt(f32(DK));
  var s: array<f32, 128>;                 // state column S[:, dv], length DK
  for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = 0.0; }

  for (var t = 0u; t < p.S; t = t + 1u) {
    let base = t * p.H + h;               // value-head row (v, g, beta, out)
    let basek = t * p.HK + hk;            // key-head row (q, k)
    for (var i = lid.x; i < DK; i = i + WGV) { ksh[i] = k[basek * DK + i]; qsh[i] = q[basek * DK + i]; }
    workgroupBarrier();
    var sk = 0.0;
    var sq = 0.0;
    for (var dk = 0u; dk < DK; dk = dk + 1u) { sk = sk + ksh[dk] * ksh[dk]; sq = sq + qsh[dk] * qsh[dk]; }
    let ik = inverseSqrt(sk + 1e-6);              // l2norm(k)
    let iq = inverseSqrt(sq + 1e-6) * scale;      // l2norm(q) / sqrt(dk)
    if (dv < p.DV) {
      let gt = exp(g[base]);
      let bt = beta[p.betaOff + base];   // beta may share g's buffer (engine: gbeta = [g; beta])
      for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = s[dk] * gt; }   // decay
      var kv = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) { kv = kv + s[dk] * ksh[dk] * ik; }
      let delta = (v[base * p.DV + dv] - kv) * bt;
      var o = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) {
        s[dk] = s[dk] + ksh[dk] * ik * delta;      // S += Kn ⊗ delta
        o = o + s[dk] * qsh[dk] * iq;              // out = Qn · S (updated)
      }
      core[base * p.DV + dv] = o;
    }
    workgroupBarrier();
  }
}
