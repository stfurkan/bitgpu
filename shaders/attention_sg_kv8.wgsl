// attention_sg with a q8 KV cache (kvCache: 'q8'). Keep in lockstep with attention_sg.wgsl: the
// ONLY difference is Kc/Vc are packed snorm8 words dequantized at the read with their per-block
// f32 scales (32-element blocks, q8_0-style; see copy_kv8.wgsl). All arithmetic (dot, softmax,
// accumulation) stays f32. Each lane owns whole packed words, so q is read in matching groups
// of 4.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
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
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;

  var acc: array<vec4<f32>, 8>;      // words per lane: W4/SG <= 8 for SG >= 4
  for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = vec4<f32>(0.0); }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      acc[wi] = acc[wi] * corr + wgt * vw;
      wi = wi + 1u;
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    let o = acc[wi] / l;
    out[ob + w * 4u] = o.x;
    out[ob + w * 4u + 1u] = o.y;
    out[ob + w * 4u + 2u] = o.z;
    out[ob + w * 4u + 3u] = o.w;
    wi = wi + 1u;
  }
}
