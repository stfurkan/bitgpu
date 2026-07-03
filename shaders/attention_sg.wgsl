// Causal GQA attention, subgroup-parallel: one subgroup (= one workgroup) per (query, head).
// Lanes split head_dim; flash-style online softmax over the cached positions; the per-position
// score (q.k) is reduced with subgroupAdd. Fixes the decode bottleneck where attention ran only
// H threads. SG = device subgroup size (>=32 so head_dim/SG <= 4). Reads K/V from the cache.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

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
  let dper = p.D / SG;                         // <= 4 for SG>=32, D=128

  var acc: array<f32, 4>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; part = part + q[qb + d] * Kc[kb + d]; }
    let score = subgroupAdd(part) * inv;       // full q.k dot, broadcast to all lanes
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
