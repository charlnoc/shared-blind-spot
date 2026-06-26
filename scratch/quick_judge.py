"""Fast judge diagnostic: load a model, judge a few pilot cases, print each
immediately with timing. Detects hangs and shows discrimination quickly."""
import json, sys, time
sys.path.insert(0, "src")
from arithmetic import generate_dataset
from models import LM
from prompts import build_judge_messages, parse_verdict
from pipeline import DATASET

key = sys.argv[1] if len(sys.argv) > 1 else "granite-2b"
dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
import torch
dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

recs = json.load(open("results/runs/run_pilot.json"))["records"]
probs = generate_dataset(len(recs), **DATASET)
# pick 2 correct + 3 wrong M cases
correct = [(x, p) for x, p in zip(recs, probs) if x["M"]["correct"]][:2]
wrong = [(x, p) for x, p in zip(recs, probs) if not x["M"]["correct"]][:3]

print(f"loading {key} dtype={dtype} ...", flush=True)
t0 = time.time()
lm = LM(key, dtype=dt)
print(f"  loaded in {time.time()-t0:.0f}s on {lm.device}", flush=True)

for label, cases in [("M-CORRECT", correct), ("M-WRONG", wrong)]:
    for x, p in cases:
        t = time.time()
        out = lm.chat(build_judge_messages(p, x["M"]["cot"], x["M"]["answer"]))
        v = parse_verdict(out)
        print(f"[{label}] verdict={v} ({time.time()-t:.1f}s)  tail={out[-70:]!r}", flush=True)
