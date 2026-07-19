// De-interleave a per-head doubled projection [S, H, 2*Dh] into [S, H, Dh], taking the half at
// `off` (0 = query, Dh = gate). The Qwen3.5 gated-attention q_proj packs query and output-gate
// interleaved per head; this pulls one out into a packed buffer.
struct Params { S: u32, H: u32, Dh: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [S, H, 2*Dh]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [S, H, Dh]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.S * p.H * p.Dh) { return; }
  let d = i % p.Dh;
  let sh = i / p.Dh;          // s*H + h
  dst[i] = src[sh * (2u * p.Dh) + p.off + d];
}
