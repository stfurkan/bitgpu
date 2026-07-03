// GPU argmax over the logits, writing one token id into a GPU buffer so the token never leaves the
// GPU (enables the deferred-sync decode loop). Single workgroup, WG threads strided-scan the N
// logits tracking (maxVal, maxIdx), then a shared-mem tree reduction. Tie-break = LOWEST index, to
// match the CPU argmax (strict > keeps the first max). No subgroup ops -> all devices.
override WG: u32 = 256u;
struct Params { N: u32, outIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outTok: array<u32>;   // outTok[p.outIdx] = argmax

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) { bv = v; bi = i; }      // strict > keeps the lowest index within this thread's stride
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
  if (tid == 0u) { outTok[p.outIdx] = sidx[0]; }
}
