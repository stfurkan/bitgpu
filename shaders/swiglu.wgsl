// SwiGLU gate: y[i] = silu(gate[i]) * up[i], silu(g) = g * sigmoid(g). One invocation per element.
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  let g = gate[i];
  y[i] = (g / (1.0 + exp(-g))) * up[i];
}
