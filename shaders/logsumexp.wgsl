// log-sum-exp over the (penalty-filtered) logits, the softmax normalizer that turns a raw logit
// into a true logprob on the CPU: logprob(id) = logit[id] - lse. Runs AFTER sampler_penalty and
// BEFORE the argmax_masked rounds (those mask their winners in place, which would corrupt the
// sum). Two-phase single-workgroup reduction: strided max, then strided sum of exp(x - max);
// entries at the -inf sentinel (banned ids) contribute nothing. Only one f32 is read back.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outLse: array<f32>;   // outLse[0] = max + log(sum)

const NEG_SENTINEL: f32 = -3.0e38;   // below any real logit; banned entries sit at f32 -inf

var<workgroup> sval: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sval[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sval[0];
  workgroupBarrier();
  var acc = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) { acc = acc + exp(v - gmax); }
  }
  sval[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sval[tid] = sval[tid] + sval[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { outLse[0] = gmax + log(sval[0]); }
}
