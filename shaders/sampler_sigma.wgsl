// Mean/variance statistics of the (penalty-filtered) logits for the top-n-sigma warper
// (arXiv 2411.07641): the CPU keeps candidates with logit >= max - n * sigma, where sigma is the
// standard deviation of the FULL logit vector (the paper's statistic - a top-K-only estimate is
// biased). Runs AFTER sampler_penalty and BEFORE the argmax_masked rounds (those mask winners in
// place, which would corrupt the moments). Banned entries (-inf sentinel) are excluded; numerical
// stability comes from centering on the global max before accumulating (logits are O(10), so
// sum-of-squares around the max stays well inside f32). Three f32s are read back:
// out = [sum(x - max), sum((x - max)^2), count] -> CPU: var = q/c - (s/c)^2.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outStats: array<f32>; // [sum, sumsq, count] centered on max

const NEG_SENTINEL: f32 = -3.0e38; // below any real logit; banned entries sit at f32 -inf

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;
var<workgroup> sc: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sa[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sa[tid + s] > sa[tid]) { sa[tid] = sa[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sa[0];
  workgroupBarrier();
  var acc = 0.0;
  var accq = 0.0;
  var cnt = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) {
      let d = v - gmax;
      acc = acc + d;
      accq = accq + d * d;
      cnt = cnt + 1.0;
    }
  }
  sa[tid] = acc;
  sb[tid] = accq;
  sc[tid] = cnt;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sa[tid] = sa[tid] + sa[tid + s];
      sb[tid] = sb[tid] + sb[tid + s];
      sc[tid] = sc[tid] + sc[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    outStats[0] = sa[0];
    outStats[1] = sb[0];
    outStats[2] = sc[0];
  }
}
