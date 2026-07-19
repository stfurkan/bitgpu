// q8 variant of attention_online_cache: the Qwen3.5 full-attention path reading K/V from the packed
// snorm8 cache (kcQ/vcQ = 4 x snorm8 per u32 word, kcS/vcS = one f32 scale per 32-element block,
// llama.cpp q8_0-style, written by copy_kv8). Each element is dequantized with one unpack4x8snorm +
// block-scale multiply at read time; all online-softmax arithmetic stays f32, so this matches the f32
// attention_online_cache exactly except for the single snorm8 rounding of K/V at write (nothing
// compounds). Same structure: one workgroup per query (s,h), thread d owns output dim d. head_dim <=256.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, posBase: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> kcQ: array<u32>;  // cache [cap*KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> kcS: array<f32>;  // cache [cap*KV, D/32] block scales
@group(0) @binding(4) var<storage, read> vcQ: array<u32>;  // cache [cap*KV, D/4]
@group(0) @binding(5) var<storage, read> vcS: array<f32>;  // cache [cap*KV, D/32]
@group(0) @binding(6) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);
  let d = lid.x;
  let D = p.D;
  let W4 = D / 4u;
  let NB = D / 32u;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  let last = p.posBase + s;      // inclusive: attend cache positions 0..last
  for (var j = 0u; j <= last; j = j + 1u) {
    let row = j * p.KV + hkv;
    var kval = 0.0;
    if (d < D) { kval = unpack4x8snorm(kcQ[row * W4 + (d >> 2u)])[d & 3u] * kcS[row * NB + (d >> 5u)]; }
    red[d] = select(0.0, qsh[d] * kval, d < D);
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
    if (d < D) {
      let vval = unpack4x8snorm(vcQ[row * W4 + (d >> 2u)])[d & 3u] * vcS[row * NB + (d >> 5u)];
      acc = acc * corr + pj * vval;
    }
    m = mn;
    workgroupBarrier();
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
