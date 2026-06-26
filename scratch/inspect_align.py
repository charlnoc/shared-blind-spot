import sys
sys.path.insert(0, "src")
from arithmetic import generate_dataset
from error_locator import parse_cot
from models import LM
from prompts import build_solve_messages

lm = LM("qwen0.5b")
probs = generate_dataset(30, n_steps=6, max_add=40, max_mul=4, seed0=5000)
shown = 0
for p in probs:
    out = lm.chat(build_solve_messages(p))
    steps = parse_cot(out)
    if len(steps) != len(p.ops) and shown < 5:
        shown += 1
        print(f"=== seed {p.seed}: parsed {len(steps)} vs ops {len(p.ops)} | ops={p.ops} ===")
        print(out)
        print()
