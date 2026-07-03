// 2-bit dequant matmul (lm_head): y[M,N] = x[M,K] @ W[N,K]^T, W[n,k] = (code - zp) * scale[n, k/block].
// codes are 2-bit, 4 per byte, packed into u32 words. Correctness-first (one thread per output, fp32).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, zp: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [M, K]
@group(0) @binding(2) var<storage, read> codes: array<u32>;   // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;  // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xbase = m * p.K;
  let cbyteBase = n * (p.K / 4u);   // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    let k0 = b * p.block;
    for (var j = 0u; j < p.block; j = j + 1u) {
      let k = k0 + j;
      let byteIdx = cbyteBase + (k >> 2u);
      let word = codes[byteIdx >> 2u];
      let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
      let code = (byte >> (2u * (k & 3u))) & 3u;
      bsum = bsum + (f32(code) - zpf) * x[xbase + k];
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
