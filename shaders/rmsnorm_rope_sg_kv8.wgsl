// rmsnorm_rope_sg writing into the q8 cache (kvCache: 'q8'): used ONLY for the K projection on
// the fused decode path, where the normed+roped K quantizes straight into the cache. Keep the
// math in lockstep with rmsnorm_rope_sg.wgsl; the write side mirrors copy_kv8.wgsl (packed
// snorm8 words + one f32 scale per 32-element block). The q call keeps the f32 kernel.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outRow0: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;            // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;        // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;          // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;          // [D]
@group(0) @binding(5) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(6) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 32>; // per-word abs max (D <= 128 -> at most 32 words)
var<workgroup> wblk: array<f32, 4>;  // per-block scale (D/32 <= 4 blocks)

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;                    // uniform: the barrier pattern below stays safe
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let W4 = p.D / 4u;

  var vals: array<vec4<f32>, 8>;     // words per lane: W4/SG <= 8 for SG >= 4
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    var vv = vec4<f32>(0.0);
    for (var e = 0u; e < 4u; e = e + 1u) {
      let i = w * 4u + e;
      let nd = x[base + i] * inv * gamma[i];
      var pd: u32; var sgn: f32;
      if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
      let rot = sgn * (x[base + pd] * inv * gamma[pd]);
      vv[e] = nd * cos[i] + rot * sin[i];
    }
    vals[wi] = vv;
    wi = wi + 1u;
    wabs[w] = max(max(abs(vv.x), abs(vv.y)), max(abs(vv.z), abs(vv.w)));
  }
  workgroupBarrier();
  if (lane < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[lane * 8u + i]); }
    let sc = max(m, 1e-30);
    wblk[lane] = sc;
    dstS[(p.outRow0 + row) * (p.D / 32u) + lane] = sc;
  }
  workgroupBarrier();
  wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    dstQ[(p.outRow0 + row) * W4 + w] = pack4x8snorm(vals[wi] / wblk[w >> 3u]);
    wi = wi + 1u;
  }
}
