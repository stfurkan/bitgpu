// DeltaNet gate/decay compute: from the a (decay input) and b (beta input) projections,
//   g[s,h]    = a_neg[h] * softplus(a[s,h] + dt_bias[h])     (<= 0, log-space decay)
//   beta[s,h] = sigmoid(b[s,h])
// per value head h. a_neg is -exp(A_log): the PrismML GGUF stores this pre-computed in the ssm_a
// tensor (verified against the transformers A_log), so no exp() here. One invocation per (s,h);
// output is [g (S*H) ; beta (S*H)] concatenated (engine binds two sub-ranges). Matches qwen35_numpy.
struct Params { S: u32, H: u32, _p0: u32, _p1: u32 };  // H = num_value_heads
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;        // [S, H]
@group(0) @binding(2) var<storage, read> b: array<f32>;        // [S, H]
@group(0) @binding(3) var<storage, read> a_neg: array<f32>;    // [H] = -exp(A_log)
@group(0) @binding(4) var<storage, read> dt_bias: array<f32>;  // [H]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;// [2*S*H]: g then beta

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let n = p.S * p.H;
  if (i >= n) { return; }
  let h = i % p.H;
  let x = a[i] + dt_bias[h];
  let sp = max(x, 0.0) + log(1.0 + exp(-abs(x)));   // softplus (stable)
  out[i] = a_neg[h] * sp;                            // g  (a_neg already = -exp(A_log))
  out[n + i] = 1.0 / (1.0 + exp(-b[i]));             // beta
}
