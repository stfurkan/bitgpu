"""Calibrate bitgpu/chat's thinkEarlyStop defaults on the REAL Bonsai-27B (bf16, A100-80GB).

Phase 1: thinking-mode greedy generation per problem, recording each think-step's top1-top2 logit
gap (the exact signal bitgpu's ThinkBudget.observe() sees via the candidate filter).
Phase 2: post-hoc simulation - for each (gap, window, minTokens) config find where the early stop
would have cut, then RE-generate the answer from the truncated reasoning (forced `</think>\n\n`),
plus fixed budgets and the unlimited baseline. Scores final-number accuracy vs think tokens spent.

    modal run tools/eval-thinkstop-modal.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=5.14", "torch", "numpy", "safetensors", "huggingface_hub", "accelerate")
)
app = modal.App("bonsai27b-thinkstop", image=image)

# GSM8K-style problems with unambiguous numeric answers (kept small: this is a calibration, not a benchmark)
PROBLEMS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn in dollars?", 10),
    ("Betty has only half of the money she needs for a new wallet that costs $100. Her parents give her $15 and her grandparents give her twice as much as her parents. How much more money does Betty need?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", 624),
    ("Mark has 3 tanks for pregnant fish. Each tank has 4 pregnant fish and each fish gives birth to 20 young. How many young fish does he have at the end?", 240),
    ("Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", 48),
    ("Ken created a care package to send to his brother. He placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?", 16),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
    ("Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. How many miles does he bike a week?", 400),
    ("Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in dollars?", 115),
    ("A tank of water has a depth of 17 feet on Monday. On Tuesday, the tank had 7 feet more water. On Wednesday, the depth of the water is two thirds of what it was on Tuesday. What is the tank's water depth on Wednesday, in feet?", 16),
    ("There are 96 fourth-graders at a school. 43 of them bought a pencil at the school store. Of the students who did not buy a pencil, half borrowed one from a friend. How many fourth-graders neither bought nor borrowed a pencil?", 27),
]

ADAPTIVE = [(g, w, m) for g in (6.0, 8.0, 10.0) for w in (16, 32) for m in (64,)]
BUDGETS = [128, 256]
MAX_THINK = 1100
MAX_ANSWER = 220


@app.function(gpu="A100-80GB", timeout=7200)
def evaluate() -> str:
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "prism-ml/Bonsai-27B-unpacked"
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("loading 27B (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    close_id = tok.convert_tokens_to_ids("</think>")

    def final_number(text: str):
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
        return float(nums[-1]) if nums else None

    def gen_answer(prefix_ids: list[int]) -> str:
        ids = torch.tensor([prefix_ids], device="cuda")
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=MAX_ANSWER, do_sample=False, temperature=None, top_p=None, top_k=None)
        return tok.decode(g[0, ids.shape[1] :], skip_special_tokens=True)

    # accumulators: config -> [correct, total, think_tokens_sum]
    tally: dict[str, list[float]] = {}

    def record(cfg: str, correct: bool, spent: int) -> None:
        t = tally.setdefault(cfg, [0, 0, 0])
        t[0] += 1 if correct else 0
        t[1] += 1
        t[2] += spent

    close_suffix = tok.encode("</think>\n\n", add_special_tokens=False)
    for qi, (q, want) in enumerate(PROBLEMS):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
        pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(pids, max_new_tokens=MAX_THINK + MAX_ANSWER, do_sample=False, temperature=None, top_p=None, top_k=None, output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0, pids.shape[1] :].tolist()
        # per-step top1-top2 gap over the generated prefix (the think phase starts at step 0:
        # the Qwen3.5 thinking template PRE-OPENS <think>)
        gaps = []
        for sc in out.scores:
            top2 = torch.topk(sc[0], 2).values
            gaps.append(float(top2[0] - top2[1]))
        think_len = seq.index(close_id) if close_id in seq else len(seq)
        base_answer = tok.decode(seq[think_len:], skip_special_tokens=True)
        base_ok = final_number(base_answer) == want
        record("full", base_ok, think_len)
        print(f"[{qi}] think={think_len} full={'OK' if base_ok else 'MISS'}", flush=True)

        prefix = pids[0].tolist()

        def truncated_answer(cut: int) -> bool:
            ans = gen_answer(prefix + seq[:cut] + close_suffix)
            return final_number(ans) == want

        for b in BUDGETS:
            cut = min(b, think_len)
            record(f"budget{b}", base_ok if cut >= think_len else truncated_answer(cut), cut)
        for g, w, m in ADAPTIVE:
            run = 0
            fire = None
            for t in range(think_len):
                run = run + 1 if gaps[t] >= g else 0
                if t + 1 >= m and run >= w:
                    fire = t + 1
                    break
            cfg = f"g{g:g}/w{w}/m{m}"
            if fire is None:
                record(cfg, base_ok, think_len)
            else:
                record(cfg, truncated_answer(fire), fire)

    lines = [f"{'config':>12} {'acc':>7} {'avg think tokens':>17}"]
    for cfg, (c, n, s) in sorted(tally.items()):
        lines.append(f"{cfg:>12} {c / n:>7.0%} {s / n:>17.1f}")
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    return table


@app.local_entrypoint()
def main():
    print(evaluate.remote())
