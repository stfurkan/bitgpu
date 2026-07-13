// attention_wg with a q8 KV cache (kvCache: 'q8'): the no-subgroup fallback reader for the
// packed-snorm8 cache (see copy_kv8.wgsl for the write side). Keep in lockstep with
// attention_wg.wgsl: same online softmax, all arithmetic f32; each thread owns one packed word
// (D <= 128 -> at most 32 words, so threads 32..63 only carry zeros through the reduction).
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
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
      part = dot(qv, kw);
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
