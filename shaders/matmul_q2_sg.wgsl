// Subgroup split-K GEMV for the 2-bit lm_head (M=1 decode). One subgroup per output column,
// lanes split K (vec4), reduce with subgroupAdd. value = (code - zp) * per-block scale.
// 2D dispatch since N (vocab) > 65535.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);     // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];   // block = (gi*4)/128 = gi/32
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) { y[n] = total; }
}
