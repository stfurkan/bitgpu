// No-subgroup fallback: 2-bit lm_head GEMV for decode (M=1), workgroup-shared-memory reduction.
// One workgroup per output column; WG threads split K and tree-reduce. value = (code - zp) * scale.
// 2D dispatch since N (vocab) > 65535. This is the v1 path's biggest cost (scalar was ~48ms/token).
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let tid = lid.x;
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[n] = sdata[0]; }
}
