// Headless checks for the prompt-lookup drafter (the CPU half of speculative decoding).
// The GPU half (batched verify + acceptance) is gated in examples/verify.html.
import { draftNgram } from '../src/pld'

let failures = 0
function check(name: string, got: number[], want: number[]): void {
  const ok = got.length === want.length && got.every((v, i) => v === want[i])
  if (!ok) failures++
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${name}  got=[${got}] want=[${want}]`)
}

// trailing [2,3] recurs at index 1; the draft is everything that follows it, up to maxDraft
check('basic 2-gram match', draftNgram([1, 2, 3, 4, 5, 6, 2, 3], 3, 8), [4, 5, 6, 2, 3])
// trailing 3-gram [1,2,3] recurs at 0; prefer the longer gram over the 2-gram match
check('longest gram wins', draftNgram([1, 2, 3, 9, 1, 2, 3], 3, 2), [9, 1])
// most RECENT prior occurrence wins: [5,5] at index 3 (continuation 7), not index 0 (6)
check('most recent occurrence', draftNgram([5, 5, 6, 5, 5, 7, 5, 5], 2, 1), [7])
// draft can extend into the suffix itself (self-extension of a repeating pattern), but never past seq's end
check('overlapping repetition', draftNgram([8, 9, 8, 9, 8, 9], 2, 4), [8, 9])
check('no match', draftNgram([1, 2, 3, 4, 5], 3, 8), [])
check('maxDraft 0', draftNgram([1, 2, 1, 2], 2, 0), [])
check('tiny sequence', draftNgram([7], 3, 8), [])
check('maxDraft caps the draft', draftNgram([1, 2, 3, 4, 5, 6, 2, 3], 3, 2), [4, 5])

if (failures) {
  console.error(`${failures} FAILURE(S)`)
  process.exit(1)
}
console.log('ALL PLD DRAFTER CHECKS PASSED')
