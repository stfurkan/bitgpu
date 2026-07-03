// Causal GQA attention reading K/V from the persistent cache. Handles both prefill
// (S queries at positions posBase..posBase+S-1) and decode (S=1). Query head h reads
// KV head h/(H/KV). Each query at absolute position pos attends to cache[0..pos].
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let invscale = 1.0 / sqrt(f32(p.D));
  let qbase = (qi * p.H + h) * p.D;

  var m = -1e30;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var dot = 0.0;
    for (var d = 0u; d < p.D; d = d + 1u) { dot = dot + q[qbase + d] * Kc[kb + d]; }
    m = max(m, dot * invscale);
  }

  var acc: array<f32, 128>;
  for (var d = 0u; d < p.D; d = d + 1u) { acc[d] = 0.0; }
  var denom = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var dot = 0.0;
    for (var d = 0u; d < p.D; d = d + 1u) { dot = dot + q[qbase + d] * Kc[kb + d]; }
    let w = exp(dot * invscale - m);
    denom = denom + w;
    for (var d = 0u; d < p.D; d = d + 1u) { acc[d] = acc[d] + w * Vc[kb + d]; }
  }

  let ob = (qi * p.H + h) * p.D;
  for (var d = 0u; d < p.D; d = d + 1u) { out[ob + d] = acc[d] / denom; }
}
