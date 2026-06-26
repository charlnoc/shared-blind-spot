"""Phase 2 main run (concurrent, cached). Mirrors the local pipeline's data
schema exactly so assemble/metrics/plot/report run unchanged.

Unlike the local MPS pipeline (one model per PROCESS to avoid corrupting the
Apple MPS pool), API calls are network-bound and independent, so we fan out with
a thread pool. Safety rests on the mandatory content-hash disk cache: every call
is persisted atomically, identical requests are never re-billed, and a crash
resumes for free (re-running recomputes instantly from cache).

Writes results/phase2/runs/{_M,_same,_cross}.jsonl. Resumable: re-run any time.

Usage:
  python src/phase2_run.py [N]      # N overrides phase2config.N (e.g. 10 for dry run)
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from api_models import APILM
from arithmetic import generate_dataset
from phase2config import ANSWERER, DATASET, JUDGES, N as CFG_N
from phaselib import solve_record
from prompts import build_judge_messages, parse_verdict

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "phase2", "runs")
os.makedirs(RUNS, exist_ok=True)
MAX_WORKERS = 8


def _map(fn, items):
    """Concurrent map preserving input order."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(fn, items))


def run_answerer(probs):
    lm = APILM(ANSWERER)
    _ = lm.client  # pre-init so the 8 worker threads don't race on lazy creation

    def one(p_i):
        i, p = p_i
        return {"problem_id": i, "seed": p.seed, "gt": p.gt, "n_ops": len(p.ops),
                "text": p.text, "M": solve_record(lm, p)}

    recs = _map(one, list(enumerate(probs)))
    out = os.path.join(RUNS, "_M.jsonl")
    with open(out, "w") as f:
        for r in sorted(recs, key=lambda r: r["problem_id"]):
            f.write(json.dumps(r) + "\n")
    acc = sum(r["M"]["correct"] for r in recs) / len(recs)
    print(f"[run] answerer {ANSWERER}: acc={acc:.0%} wrong={sum(not r['M']['correct'] for r in recs)}", flush=True)
    return recs


def run_judge(fam, model_key, probs, mrecs):
    lm = APILM(model_key)
    _ = lm.client  # pre-init (see run_answerer)
    mby = {r["problem_id"]: r for r in mrecs}

    def one(p_i):
        i, p = p_i
        mr = mby[i]
        cot_M, ans_M = mr["M"]["cot"], mr["M"]["answer"]
        try:
            verdict_text = lm.chat(build_judge_messages(p, cot_M, ans_M))
        except Exception as e:  # noqa: BLE001
            verdict_text = f"(error: {e})"
        solve = solve_record(lm, p)
        return {"problem_id": i, "family": fam, "model": model_key,
                "verdict_text": verdict_text, "endorsed": parse_verdict(verdict_text),
                "solve": solve, "ppl_on_M": None}  # teacher-forced ppl N/A on chat APIs

    recs = _map(one, list(enumerate(probs)))
    out = os.path.join(RUNS, f"_{fam}.jsonl")
    with open(out, "w") as f:
        for r in sorted(recs, key=lambda r: r["problem_id"]):
            f.write(json.dumps(r) + "\n")
    j_acc = sum(r["solve"]["correct"] for r in recs) / len(recs)
    endorsed = sum(1 for r in recs if r["endorsed"] is True)
    print(f"[run] judge {fam}={model_key}: solve_acc={j_acc:.0%} endorsed={endorsed}/{len(recs)}", flush=True)
    return recs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else CFG_N
    probs = generate_dataset(n, **DATASET)
    print(f"[run] N={n} dataset={DATASET}", flush=True)
    print(f"[run] answerer={ANSWERER} judges={JUDGES}", flush=True)
    mrecs = run_answerer(probs)
    for fam, model_key in JUDGES:
        run_judge(fam, model_key, probs, mrecs)
    print("[run] DONE", flush=True)


if __name__ == "__main__":
    main()
