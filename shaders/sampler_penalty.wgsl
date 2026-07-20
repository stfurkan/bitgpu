// GPU logits pre-filter for sampling: applies repetition_penalty + presence_penalty, then
// no_repeat_ngram bans, in place on the full vocab logit buffer, so only a tiny top-K candidate set
// has to be read back (not all ~151k logits). rep_penalty matches transformers.js over the DEDUPED
// prompt+generated id set (logit<0 ? *penalty : /penalty); presence_penalty then SUBTRACTS a flat
// amount from every seen token (the additive anti-repetition knob the Qwen3.5 family recommends,
// applied after the multiplicative rep_penalty like vLLM); then ngram-banned next-tokens go to
// -Infinity. Both id lists are computed on the CPU each step (exact, since at syncN=1 the full
// history is known) and uploaded. presence is 0 unless requested, so `v*penalty - 0.0 == v*penalty`
// keeps the rep-penalty-only path bit-identical. Temperature is NOT applied here: top-k is invariant
// under the monotonic divide, so temperature is applied on the CPU to just the K candidate values
// before softmax (bit-identical, one less pass). Single workgroup, no subgroup ops -> all devices.
// The storageBarrier guarantees every penalty write lands before any ban write, so a token that is
// both repeated and ngram-banned ends at -inf (ban wins, matching the reference order penalties -> ngram).
override WG: u32 = 256u;
// negInf carries the -Infinity bit pattern (0xff800000) from the host: bitcasting it at RUNTIME yields
// -inf, whereas bitcast<f32>(0xff800000u) is a const-expression evaluating to inf, which is a WGSL
// shader-creation error. (Runtime inf is fine; only const/override inf/nan is rejected.)
struct Params { affectedLen: u32, banLen: u32, penalty: f32, negInf: u32, presence: f32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> affectedIds: array<u32>;   // deduped prompt+generated ids
@group(0) @binding(2) var<storage, read> banIds: array<u32>;        // ngram-banned next-token ids
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;  // [vocab], modified in place

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var i = tid; i < p.affectedLen; i = i + WG) {
    let t = affectedIds[i];
    let v = logits[t];
    let rp = select(v / p.penalty, v * p.penalty, v < 0.0);   // repetition_penalty (multiplicative)
    logits[t] = rp - p.presence;                              // presence_penalty (subtractive; 0 = no-op)
  }
  storageBarrier();                                  // all penalty writes before any ban write
  for (var i = tid; i < p.banLen; i = i + WG) {
    logits[banIds[i]] = bitcast<f32>(p.negInf);      // -Infinity (runtime bitcast)
  }
}
