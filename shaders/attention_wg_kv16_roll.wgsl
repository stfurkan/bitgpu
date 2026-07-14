// attention_wg_kv16 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_wg_kv16.wgsl: the ONLY
// differences are the shared-memory K stage (kk, widened to f32) - the rotate partner d±D/2
// may live in another thread's stride - and the rotation in the score loop, written as
// `k*cos + rot*sin` with the same operand order as rmsnorm_rope_sg.
enable f16;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;
var<workgroup> kk: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let half = p.D / 2u;

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { kk[d] = f32(Kc[kb + d]); }
    }
    workgroupBarrier();
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) {
        var rot: f32;
        if (d < half) { rot = -kk[d + half]; } else { rot = kk[d - half]; }
        let rb = j * half + (d % half);
        part = part + q[qb + d] * (kk[d] * cosT[rb] + rot * sinT[rb]);
      }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();                  // red[0] + kk consumed before the next position overwrites them
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
