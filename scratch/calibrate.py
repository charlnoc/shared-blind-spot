"""Calibrate difficulty for M = Qwen2.5-0.5B: find n_steps/operand settings that
land accuracy in the §2 errorful band (~40-70%) with a usable step-align rate.
"""
import sys, time
sys.path.insert(0, "src")
from arithmetic import generate_dataset
from error_locator import parse_cot
from models import LM
from prompts import build_solve_messages, parse_answer

lm = LM("qwen0.5b")
print(f"loaded 0.5B on {lm.device}\n")

for n_steps, max_add, max_mul in [(6, 40, 4), (8, 60, 4), (10, 80, 5)]:
    probs = generate_dataset(30, n_steps=n_steps, max_add=max_add, max_mul=max_mul, seed0=5000)
    c = parsed = aligned = 0
    t0 = time.time()
    for p in probs:
        out = lm.chat(build_solve_messages(p))
        ans = parse_answer(out)
        c += ans == p.gt
        parsed += ans is not None
        aligned += len(parse_cot(out)) == len(p.ops)
    dt = (time.time() - t0) / len(probs)
    print(f"n_steps={n_steps:2d} add<={max_add} mul<={max_mul}: "
          f"acc={c/30:.0%}  parse={parsed/30:.0%}  align={aligned/30:.0%}  {dt:.1f}s/prob")
