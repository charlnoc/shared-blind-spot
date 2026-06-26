"""Phase 1 (isolated process): answerer M solves all problems -> JSONL.
One fresh process, one model load, incremental flush per problem.

Usage: python src/phase_solve.py <out.jsonl>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from arithmetic import generate_dataset
from expconfig import ANSWERER, DATASET, N
from models import LM
from phaselib import solve_record

out = sys.argv[1]
probs = generate_dataset(N, **DATASET)
print(f"[phase_solve] M={ANSWERER} N={N} -> {out}", flush=True)
lm = LM(ANSWERER)
print(f"[phase_solve] loaded on {lm.device}", flush=True)
n_correct = 0
with open(out, "w") as f:
    for i, p in enumerate(probs):
        rec = {"problem_id": i, "seed": p.seed, "gt": p.gt, "n_ops": len(p.ops),
               "text": p.text, "M": solve_record(lm, p)}
        n_correct += rec["M"]["correct"]
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"[phase_solve] {i+1}/{N}  acc_so_far={n_correct/(i+1):.0%}", flush=True)
print(f"[phase_solve] DONE  M accuracy={n_correct/N:.0%}  wrong={N-n_correct}", flush=True)
