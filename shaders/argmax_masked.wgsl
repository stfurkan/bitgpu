// Masked argmax: like argmax.wgsl but skips any id already chosen in a prior round, and writes BOTH
// the winning id and its logit value. Calling it K times (roundCount = 0..K-1, all in one compute
// pass so each round sees the prior rounds' writes) yields the exact top-K (id, logit) pairs in
// descending order = ONNX TopK over the (penalty-filtered) logits, which is what the transformers.js
// sampler consumes. Then only K pairs are read back (not the full vocab), and the CPU does
// temperature + softmax + multinomial. Single workgroup, no subgroup ops -> all devices. Tie-break =
// lowest index (strict >), matching argmax.wgsl / ORT TopK in practice.
override WG: u32 = 256u;
struct Params { N: u32, roundCount: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> candIds: array<u32>;   // [K]; reads 0..roundCount-1, writes [roundCount]
@group(0) @binding(3) var<storage, read_write> candVals: array<f32>;  // [K]; writes [roundCount]

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) {
      var skip = false;
      for (var r = 0u; r < p.roundCount; r = r + 1u) { if (candIds[r] == i) { skip = true; break; } }
      if (!skip) { bv = v; bi = i; }     // strict > keeps the lowest index within this thread's stride
    }
  }
  sval[tid] = bv; sidx[tid] = bi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = sval[tid + s]; let oi = sidx[tid + s];
      if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { candIds[p.roundCount] = sidx[0]; candVals[p.roundCount] = sval[0]; }
}
