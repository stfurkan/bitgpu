// Depthwise causal Conv1d (kernel width K) + SiLU, for the gated-DeltaNet q/k/v stream. x is
// [S, C] (C = conv_dim channels), weight is [C, K] (per-channel taps, the GGUF ssm_conv1d layout).
// Carries a persistent left-context so segmented prefill and token-by-token decode continue across
// calls: state_in / state_out hold the last K-1 inputs ([K-1, C]); loadState!=0 uses them (else the
// causal left pad is zero). Extended input ext = [state_in (K-1), x (S)]:
//   y[t,c] = silu( sum_{j<K} w[c,j] * ext[t+j, c] ),   state_out[i,c] = ext[S+i, c]  (i < K-1)
struct Params { S: u32, C: u32, K: u32, loadState: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [S, C]
@group(0) @binding(2) var<storage, read> w: array<f32>;          // [C, K]
@group(0) @binding(3) var<storage, read> state_in: array<f32>;   // [K-1, C]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;    // [S, C]
@group(0) @binding(5) var<storage, read_write> state_out: array<f32>; // [K-1, C]

fn ext(m: u32, c: u32) -> f32 {
  if (m + 1u < p.K) { return select(0.0, state_in[m * p.C + c], p.loadState != 0u); }
  return x[(m - (p.K - 1u)) * p.C + c];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let outN = p.S * p.C;
  if (i < outN) {
    let c = i % p.C;
    let t = i / p.C;
    var acc = 0.0;
    for (var j = 0u; j < p.K; j = j + 1u) { acc = acc + w[c * p.K + j] * ext(t + j, c); }
    y[i] = acc / (1.0 + exp(-acc));  // SiLU
  } else if (i < outN + (p.K - 1u) * p.C) {
    let si = i - outN;
    let sc = si % p.C;
    let sk = si / p.C;                 // 0 .. K-2
    state_out[sk * p.C + sc] = ext(p.S + sk, sc);
  }
}
