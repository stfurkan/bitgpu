// RoPE (rotate_half) with precomputed full cos/sin [S, D]. x is [S, H, D]. One invocation per element.
struct Params { S: u32, H: u32, D: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [S, H, D]
@group(0) @binding(2) var<storage, read> cos: array<f32>;     // [S, D]
@group(0) @binding(3) var<storage, read> sin: array<f32>;     // [S, D]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.D / 2u;
  let row = sh * p.D;  // s*H*D + h*D
  var rot: f32;
  if (d < half) {
    rot = -x[row + d + half];
  } else {
    rot = x[row + d - half];
  }
  y[idx] = x[idx] * cos[s * p.D + d] + rot * sin[s * p.D + d];
}
