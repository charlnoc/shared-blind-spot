"""Smoke test (spec §8 de-risk): does a tiny model produce parseable, aligned
CoTs, is it usefully errorful, and do chat/perplexity work on MPS?

Run: .venv/bin/python smoke_test.py [n_steps] [n_problems]
"""

import sys
import time

sys.path.insert(0, "src")

from arithmetic import generate_dataset
from error_locator import locate_error, parse_cot
from models import LM
from prompts import build_judge_messages, build_solve_messages, parse_answer, parse_verdict

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
n_probs = int(sys.argv[2]) if len(sys.argv) > 2 else 12

probs = generate_dataset(n_probs, n_steps=n_steps, max_add=40, max_mul=4, seed0=1000)

t0 = time.time()
print(f"Loading Qwen2.5-1.5B-Instruct ...", flush=True)
lm = LM("qwen1.5b")
print(f"  loaded on {lm.device} in {time.time()-t0:.1f}s\n", flush=True)

n_correct = n_parsed = n_aligned = 0
gen_times = []
for p in probs:
    t = time.time()
    out = lm.chat(build_solve_messages(p))
    gen_times.append(time.time() - t)
    ans = parse_answer(out)
    steps = parse_cot(out)
    correct = ans == p.gt
    aligned = len(steps) == len(p.ops)
    n_correct += correct
    n_parsed += ans is not None
    n_aligned += aligned
    rep = locate_error(steps, problem=p) if aligned else None
    print(f"[seed {p.seed}] gt={p.gt} ans={ans} correct={correct} "
          f"steps={len(steps)}/{len(p.ops)} aligned={aligned} "
          f"err={None if rep is None else (rep.index, rep.kind)}")

print(f"\nAccuracy           : {n_correct}/{n_probs} = {n_correct/n_probs:.0%}  (target 20-50% so plenty of errors)")
print(f"Answer parse rate  : {n_parsed}/{n_probs} = {n_parsed/n_probs:.0%}")
print(f"Step-align rate    : {n_aligned}/{n_probs} = {n_aligned/n_probs:.0%}  (needed for SER co-location)")
print(f"Mean gen time      : {sum(gen_times)/len(gen_times):.1f}s/problem")

# judge + perplexity on the first wrong case we can find
wrong = next((p for p in probs if parse_answer(lm.chat(build_solve_messages(p))) != p.gt), probs[0])
sol = lm.chat(build_solve_messages(wrong))
verdict = lm.chat(build_judge_messages(wrong, sol, parse_answer(sol)))
ppl = lm.perplexity(build_solve_messages(wrong), sol)
print(f"\nJudge verdict parse: {parse_verdict(verdict)}  | self-perplexity on own CoT: {ppl:.2f}")
print(f"  (judge raw tail: ...{verdict[-80:]!r})")
