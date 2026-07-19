// Causal GQA attention with online (flash) softmax, head-dim up to the workgroup size (256) - the
// Qwen3.5 full-attention layers use head_dim 256, past the <=128 the register-array kernels assume.
// One workgroup per query (s,h); thread d owns output dim d. Streams keys j<=s keeping running
// max/sum/acc, so no O(S) score storage. Output gate + RoPE + QK-norm are applied separately.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> k: array<f32>;    // [S, KV, D]
@group(0) @binding(3) var<storage, read> v: array<f32>;    // [S, KV, D]
@group(0) @binding(4) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);    // GQA: which kv head
  let d = lid.x;
  let D = p.D;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  for (var j = 0u; j <= s; j = j + 1u) {
    red[d] = select(0.0, qsh[d] * k[(j * p.KV + hkv) * D + d], d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) { acc = acc * corr + pj * v[(j * p.KV + hkv) * D + d]; }
    m = mn;
    workgroupBarrier();               // before next j overwrites red
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
