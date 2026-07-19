// Partial RoPE: rotate only the first ROT dims of each head (rotate_half within [0,ROT)); the
// remaining head_dim-ROT dims pass through unrotated. cos/sin are [S, ROT]. x/y are [S, H, D].
// Matches tools/qwen35_numpy._rope_partial (Qwen3.5 full-attention layers, partial_rotary_factor).
struct Params { S: u32, H: u32, D: u32, ROT: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> cosb: array<f32>;     // [S, ROT]
@group(0) @binding(3) var<storage, read> sinb: array<f32>;     // [S, ROT]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  if (d >= p.ROT) { y[idx] = x[idx]; return; }   // passthrough tail
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.ROT / 2u;
  var rot: f32;
  if (d < half) { rot = -x[idx + half]; } else { rot = x[idx - half]; }
  y[idx] = x[idx] * cosb[s * p.ROT + d] + rot * sinb[s * p.ROT + d];
}
