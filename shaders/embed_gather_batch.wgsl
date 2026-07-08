// Batched GPU embedding gather + 4-bit dequant for PROMPT tokens: one invocation per output
// element writes out[s*H + k] for tokenIds[s]. A prefill segment uploads S u32 token ids
// instead of S*H dequantized floats, so the CPU-side embedding tables are not needed at all
// (~50-100 MB RAM per model). Same per-row stride math and dequant as embed_gather.wgsl
// (H=2048 -> 256/16/8 strides); uint8 sources read as u32 and byte-extracted (little-endian).
struct Params { S: u32, H: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;   // [S]
@group(0) @binding(2) var<storage, read> embWq: array<u32>;      // uint8 [vocab * H/8] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;       // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>; // [vocab * H/128]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;     // uint8 [vocab * ceil(H/256)] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S * H]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let gi = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (gi >= p.S * p.H) { return; }
  let k = gi % p.H;
  let id = tokenIds[gi / p.H];
  let rowBytes = p.H >> 3u;
  let scaleRow = p.H >> 7u;
  let zpRow = (scaleRow + 1u) >> 1u;
  let i = k >> 3u;
  let qd = (k >> 1u) & 3u;
  let c = k & 1u;
  let wqIdx = id * rowBytes + i;
  let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
  let tIdx = 4u * e + qd;
  let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
  let code = (t >> (4u * c)) & 0xfu;
  let blk = k >> 7u;
  let zpIdx = id * zpRow + (blk >> 1u);
  let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
  let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
  out[gi] = (f32(code) - f32(zp)) * embScales[id * scaleRow + blk];
}
