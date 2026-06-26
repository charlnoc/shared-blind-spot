"""Layer 1 + Layer 2 data collection (spec §4, §5) for the minimal local case.

Design B (see findings/notes): one ANSWERER and two JUDGES.
  M       = Qwen2.5-0.5B-Instruct   (weak -> errorful, spec §2)
  J_same  = Qwen2.5-1.5B-Instruct   (same family, different size)
  J_cross = SmolLM2-1.7B-Instruct   (cross family, size-matched to J_same)

Both judges are the same size class, so the same-vs-cross contrast is not
confounded by judge capability. The same-family judge is a *different* model
from M (not self-judge), which keeps J's perplexity on M's CoT non-degenerate
so the §4 familiarity control is meaningful.

For each problem we record, per judge: the verdict (no ground truth shown),
the judge's independent from-scratch solution (for §5 error co-location), and
the judge's perplexity on M's CoT (the §4 familiarity proxy). Models are loaded
one at a time and freed before the next, so memory stays bounded.

Run:  .venv/bin/python src/pipeline.py <N> <tag>
Output: results/runs/run_<tag>.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from arithmetic import generate_dataset
from error_locator import locate_canonical_index
from models import LM
from prompts import (
    build_judge_messages,
    build_solve_messages,
    parse_answer,
    parse_verdict,
)

# --- config -----------------------------------------------------------------
DATASET = dict(n_steps=6, max_add=40, max_mul=4, seed0=20000)
ANSWERER = "qwen0.5b"
JUDGES = [
    ("same", "qwen1.5b"),       # same family as M (Qwen), different size
    ("cross", "smollm2-1.7b"),  # cross family, size-matched to J_same
]
# fix the model-key typo guard: ensure the cross key matches models.MODELS
from models import MODELS  # noqa: E402
assert ANSWERER in MODELS, ANSWERER
for _, k in JUDGES:
    assert k in MODELS, k


def _solve_record(lm: LM, problem) -> dict:
    try:
        cot = lm.chat(build_solve_messages(problem))
    except Exception as e:  # noqa: BLE001 -- never let one bad gen kill a long run
        return {"cot": "", "answer": None, "correct": False,
                "err_index": None, "err_method": "error", "error": str(e)}
    ans = parse_answer(cot)
    idx, method = locate_canonical_index(cot, problem)
    return {
        "cot": cot,
        "answer": ans,
        "correct": ans == problem.gt,
        "err_index": idx,
        "err_method": method,
    }


def run(n: int, tag: str, dataset=None, answerer=None, judges=None):
    dataset = dataset or DATASET
    answerer = answerer or ANSWERER
    judges = judges or JUDGES
    probs = generate_dataset(n, **dataset)
    records = [
        {"problem_id": i, "seed": p.seed, "gt": p.gt, "n_ops": len(p.ops),
         "text": p.text, "judges": {}}
        for i, p in enumerate(probs)
    ]

    # --- Phase 1: M solves everything ---------------------------------------
    print(f"[{datetime.now():%H:%M:%S}] Phase 1: M={answerer} solving {n} problems")
    t0 = time.time()
    m = LM(answerer)
    for i, p in enumerate(probs):
        records[i]["M"] = _solve_record(m, p)
        if (i + 1) % 25 == 0:
            print(f"  M {i+1}/{n}  ({time.time()-t0:.0f}s)")
    acc = sum(r["M"]["correct"] for r in records) / n
    print(f"  M accuracy = {acc:.0%}  (wrong cases = {sum(not r['M']['correct'] for r in records)})")
    m.close(); del m; gc.collect()

    # --- Phases 2..: each judge judges M, solves independently, scores ppl ---
    for family, key in judges:
        print(f"[{datetime.now():%H:%M:%S}] Phase: J_{family}={key}")
        t0 = time.time()
        j = LM(key)
        for i, p in enumerate(probs):
            cot_M = records[i]["M"]["cot"]
            ans_M = records[i]["M"]["answer"]
            try:
                verdict_text = j.chat(build_judge_messages(p, cot_M, ans_M))
            except Exception as e:  # noqa: BLE001
                verdict_text = f"(error: {e})"
            solve = _solve_record(j, p)
            try:
                ppl = j.perplexity(build_solve_messages(p), cot_M) if cot_M else float("nan")
            except Exception:  # noqa: BLE001
                ppl = float("nan")
            records[i]["judges"][family] = {
                "family": family,
                "model": key,
                "verdict_text": verdict_text,
                "endorsed": parse_verdict(verdict_text),  # True/False/None
                "solve": solve,
                "ppl_on_M": ppl,
            }
            if (i + 1) % 25 == 0:
                print(f"  J_{family} {i+1}/{n}  ({time.time()-t0:.0f}s)")
        j.close(); del j; gc.collect()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"run_{tag}.json")
    with open(out, "w") as f:
        json.dump({"config": {"DATASET": dataset, "ANSWERER": answerer, "JUDGES": judges,
                              "n": n, "created": datetime.now().isoformat()},
                   "records": records}, f, indent=2)
    print(f"\nSaved {n} records -> {out}")
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tag = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    run(n, tag)
