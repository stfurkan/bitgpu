// copy with an f16-STORAGE destination (kvCache: 'f16'): appends f32 K/V rows into the f16
// cache (one f32 -> f16 rounding per value). Keep in lockstep with copy.wgsl.
enable f16;
struct Params { n: u32, dstOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = f16(src[i]);
}
