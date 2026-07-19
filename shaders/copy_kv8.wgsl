// q8 cache append (kvCache: 'q8'): quantize f32 K/V rows into the packed-snorm8 cache, one f32
// scale per 32-element block (llama.cpp q8_0-style). One 64-thread workgroup per row of D
// elements: thread t owns packed word t (4 consecutive values), the workgroup reduces per-block
// absolute maxima through shared memory, then packs with pack4x8snorm. Replaces copy/copy_kv16
// at every cache-append site under q8. All attention arithmetic stays f32; the precision loss is
// exactly one snorm8 rounding of K/V at write time, nothing compounding.
struct Params { rows: u32, D: u32, dstRow0: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;          // [rows, D]
@group(0) @binding(2) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(3) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 64>; // per-word abs max (D <= 256 -> at most 64 words)
var<workgroup> wblk: array<f32, 8>;  // per-block scale (D/32 <= 8 blocks)

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.rows) { return; }
  let t = lid.x;
  let W4 = p.D / 4u;
  let base = row * p.D;
  var v = vec4<f32>(0.0);
  if (t < W4) {
    v = vec4<f32>(src[base + t * 4u], src[base + t * 4u + 1u], src[base + t * 4u + 2u], src[base + t * 4u + 3u]);
    wabs[t] = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
  }
  workgroupBarrier();
  if (t < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[t * 8u + i]); }
    let s = max(m, 1e-30);           // an all-zero block packs zeros, never NaN
    wblk[t] = s;
    dstS[(p.dstRow0 + row) * (p.D / 32u) + t] = s;
  }
  workgroupBarrier();
  if (t < W4) {
    dstQ[(p.dstRow0 + row) * W4 + t] = pack4x8snorm(v / wblk[t >> 3u]);
  }
}
