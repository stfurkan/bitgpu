// Depthwise causal Conv1d (kernel width K, left zero-pad K-1) + SiLU, for the gated-DeltaNet
// q/k/v stream. x is [S, C] (C = conv_dim channels), weight is [C, K] (per-channel taps, the
// GGUF ssm_conv1d layout). One invocation per (s, c) output element:
//   y[s,c] = silu( sum_{j<K} x[s-(K-1-j), c] * w[c, j] )   (terms with s-(K-1-j) < 0 dropped)
struct Params { S: u32, C: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [S, C]
@group(0) @binding(2) var<storage, read> w: array<f32>;        // [C, K]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // [S, C]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.S * p.C) { return; }
  let c = i % p.C;
  let s = i / p.C;
  var acc = 0.0;
  for (var j = 0u; j < p.K; j = j + 1u) {
    let back = p.K - 1u - j;     // input is x[s-back]; causal left-pad drops back > s
    if (s >= back) {
      acc = acc + x[(s - back) * p.C + c] * w[c * p.K + j];
    }
  }
  y[i] = acc / (1.0 + exp(-acc));  // SiLU
}
