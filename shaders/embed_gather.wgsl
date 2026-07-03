// GPU embedding gather + 4-bit dequant: reads a token id from a GPU buffer and writes that token's
// embedding (H f32) directly into a GPU buffer, so the decode loop never round-trips the token id to
// the CPU. Faithful port of the CPU embedDequant (4-bit codes via the tgt4 LUT, per-128 zero-point,
// per-block scale). uint8 source arrays are read as u32 and byte-extracted (little-endian).
override WG: u32 = 256u;
struct Params { H: u32, srcIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenId: array<u32>;   // tokenId[p.srcIdx] = the token to embed
@group(0) @binding(2) var<storage, read> embWq: array<u32>;     // uint8 [vocab*256] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;      // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>;// [vocab*16]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;    // uint8 [vocab*8] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>;// [H]

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let id = tokenId[p.srcIdx];
  for (var k = lid.x; k < p.H; k = k + WG) {
    let i = k >> 3u;
    let qd = (k >> 1u) & 3u;
    let c = k & 1u;
    let wqIdx = id * 256u + i;
    let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
    let tIdx = 4u * e + qd;
    let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
    let code = (t >> (4u * c)) & 0xfu;
    let blk = k >> 7u;
    let zpIdx = id * 8u + (blk >> 1u);
    let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
    let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
    out[k] = (f32(code) - f32(zp)) * embScales[id * 16u + blk];
  }
}
