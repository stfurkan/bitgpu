// attention_wg_kv8 for the rolling-window / attention-sinks mode: the no-subgroup fallback of
// attention_sg_kv8_roll (see there and attention_sg_roll.wgsl for the rope-at-read scheme).
// Keep in lockstep with attention_wg_kv8.wgsl: the ONLY difference is the K rotation in the
// score loop; each thread's word rotates against its partner word D/8 away, dequantized from
// global with its own scale.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kq: array<u32>;       // [Ltot, KV, D/4] packed snorm8, UNROPED
@group(0) @binding(3) var<storage, read> Vq: array<u32>;       // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(7) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(8) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let t = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let half = p.D / 2u;
  let hw = half / 4u;                          // words from a word to its rotate partner

  var qv = vec4<f32>(0.0);
  if (t < W4) {
    qv = vec4<f32>(q[qb + t * 4u], q[qb + t * 4u + 1u], q[qb + t * 4u + 2u], q[qb + t * 4u + 3u]);
  }
  var acc = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * (p.D / 32u);
    var part = 0.0;
    if (t < W4) {
      let kw = unpack4x8snorm(Kq[rowQ + t]) * Ks[rowS + (t >> 3u)];
      let wp = select(t - hw, t + hw, t < hw);
      let kp = unpack4x8snorm(Kq[rowQ + wp]) * Ks[rowS + (wp >> 3u)];
      let rot = select(kp, -kp, t < hw);
      let cb = j * half + select(t - hw, t, t < hw) * 4u;
      let cs = vec4<f32>(cosT[cb], cosT[cb + 1u], cosT[cb + 2u], cosT[cb + 3u]);
      let sn = vec4<f32>(sinT[cb], sinT[cb + 1u], sinT[cb + 2u], sinT[cb + 3u]);
      part = dot(qv, kw * cs + rot * sn);
    }
    red[t] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    if (t < W4) {
      let vw = unpack4x8snorm(Vq[rowQ + t]) * Vs[rowS + (t >> 3u)];
      acc = acc * corr + wgt * vw;
    }
    m = mnew;
  }
  if (t < W4) {
    let ob = (qi * p.H + h) * p.D;
    let o = acc / l;
    out[ob + t * 4u] = o.x;
    out[ob + t * 4u + 1u] = o.y;
    out[ob + t * 4u + 2u] = o.z;
    out[ob + t * 4u + 3u] = o.w;
  }
}
