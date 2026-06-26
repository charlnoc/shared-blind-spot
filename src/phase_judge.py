"""Phase 2 (isolated process): one judge reads M's solutions and, per problem,
(a) judges without ground truth, (b) solves independently, (c) scores its
perplexity on M's CoT. Writes JSONL incrementally. One fresh process per judge.

Usage: python src/phase_judge.py <family> <model_key> <M.jsonl> <out.jsonl>
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from arithmetic import generate_dataset
from expconfig import DATASET, N
from models import LM
from phaselib import solve_record
from prompts import build_judge_messages, build_solve_messages, parse_verdict

family, model_key, m_jsonl, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
probs = generate_dataset(N, **DATASET)
mrecs = [json.loads(line) for line in open(m_jsonl)]
print(f"[phase_judge:{family}] {model_key} over {len(mrecs)} M-records -> {out}", flush=True)
lm = LM(model_key)
print(f"[phase_judge:{family}] loaded on {lm.device}", flush=True)

done = 0
with open(out, "w") as f:
    for mr in mrecs:
        i = mr["problem_id"]
        p = probs[i]
        cot_M, ans_M = mr["M"]["cot"], mr["M"]["answer"]
        try:
            verdict_text = lm.chat(build_judge_messages(p, cot_M, ans_M))
        except Exception as e:  # noqa: BLE001
            verdict_text = f"(error: {e})"
        solve = solve_record(lm, p)
        try:
            ppl = lm.perplexity(build_solve_messages(p), cot_M) if cot_M else float("nan")
        except Exception:  # noqa: BLE001
            ppl = float("nan")
        f.write(json.dumps({
            "problem_id": i, "family": family, "model": model_key,
            "verdict_text": verdict_text, "endorsed": parse_verdict(verdict_text),
            "solve": solve, "ppl_on_M": (None if math.isnan(ppl) else ppl),
        }) + "\n")
        f.flush()
        done += 1
        if done % 25 == 0:
            print(f"[phase_judge:{family}] {done}/{len(mrecs)}", flush=True)
print(f"[phase_judge:{family}] DONE", flush=True)
