// attention_sg for the rolling-window / attention-sinks mode (overflow: 'sinks'): the cache
// holds UNROPED keys, and each cached row j is rotated AT READ by its cache-relative position
// (StreamingLLM-style; the cache bytes are immutable, so eviction compaction never re-rotates
// or requantizes anything). Keep in lockstep with attention_sg.wgsl: the ONLY difference is
// the K rotation in the score loop, written as `k*cos + rot*sin` with the same operand order
// as rmsnorm_rope_sg so the f32 path stays bit-identical to the roped-at-write kernels until
// the first eviction. cosT/sinT are the aux rope tables, [positions, D/2].
// Lane math: d = lane + t*SG, partner d±D/2 is (D/2)/SG strides away IN THE SAME LANE (the
// engine only selects this kernel when SG <= D/2).
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;
  let half = p.D / 2u;
  let hs = half / SG;                          // strides from a dim to its rotate partner

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var kd: array<f32, 8>;
    for (var t = 0u; t < dper; t = t + 1u) { kd[t] = Kc[kb + lane + t * SG]; }
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) {
      let d = lane + t * SG;
      var rot: f32;
      if (d < half) { rot = -kd[t + hs]; } else { rot = kd[t - hs]; }
      let rb = j * half + (d % half);
      part = part + q[qb + d] * (kd[t] * cosT[rb] + rot * sinT[rb]);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
