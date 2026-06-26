"""End-to-end autonomous driver: calibrate difficulty -> collect data ->
metrics -> figures -> research report. One background process, runs to the end.

Difficulty is auto-tuned so the SAME-FAMILY judge (Qwen2.5-1.5B) is ~50-65%
accurate: a strong judge that always solves correctly never shares the weak
answerer's errors, which would starve the SER metric. Tuning to where BOTH M
and J_same are errorful is where any shared-error signal can actually appear.

Usage: .venv/bin/python master.py <N> <tag> <cross_key>
"""
import os
import sys
import time

sys.path.insert(0, "src")
from arithmetic import generate_dataset
from models import LM
from prompts import build_solve_messages, parse_answer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
TAG = sys.argv[2] if len(sys.argv) > 2 else "v1"
CROSS_KEY = sys.argv[3] if len(sys.argv) > 3 else "granite-2b"

CALIB_MODEL = "qwen1.5b"          # tune difficulty to the same-family judge
TARGET_ACC = 0.58                 # want J_same errorful but still a real judge
CANDIDATES = [
    dict(n_steps=8,  max_add=50, max_mul=4, seed0=60000),
    dict(n_steps=10, max_add=70, max_mul=5, seed0=61000),
    dict(n_steps=12, max_add=90, max_mul=6, seed0=62000),
    dict(n_steps=14, max_add=99, max_mul=6, seed0=63000),
]


def calibrate(k=24):
    print(f"== calibrating difficulty on {CALIB_MODEL} (target acc ~{TARGET_ACC}) ==", flush=True)
    lm = LM(CALIB_MODEL)
    scored = []
    for d in CANDIDATES:
        probs = generate_dataset(k, **d)
        c = sum(parse_answer(lm.chat(build_solve_messages(p))) == p.gt for p in probs)
        acc = c / k
        scored.append((acc, d))
        print(f"  n_steps={d['n_steps']:2d} add<={d['max_add']} mul<={d['max_mul']}: acc={acc:.0%}", flush=True)
    lm.close()
    # pick accuracy closest to target but within a sane errorful band
    band = [s for s in scored if 0.30 <= s[0] <= 0.75]
    pool = band or scored
    acc, chosen = min(pool, key=lambda s: abs(s[0] - TARGET_ACC))
    chosen = dict(chosen)
    chosen["seed0"] = 70000  # disjoint from calibration seeds
    print(f"  -> chosen difficulty {chosen} (J_same acc {acc:.0%})", flush=True)
    return chosen


def main():
    t0 = time.time()
    dataset = calibrate()

    import pipeline
    judges = [("same", "qwen1.5b"), ("cross", CROSS_KEY)]
    print(f"\n== full run N={N} tag={TAG} judges={judges} ==", flush=True)
    run_path = pipeline.run(N, TAG, dataset=dataset, answerer="qwen0.5b", judges=judges)

    print("\n== analysis ==", flush=True)
    import plot
    import report
    import research_report
    plot.main(run_path)
    report_md = report.build(run_path)
    with open("results/findings.md", "w") as f:
        f.write(report_md)
    rr = research_report.build(run_path)
    with open("results/RESEARCH_REPORT.md", "w") as f:
        f.write(rr)
    print(f"\nDONE in {(time.time()-t0)/60:.0f} min.")
    print("  results/ser_vs_null.png  results/fer_by_family_domain.png  results/ser_perplexity_partial.png")
    print("  results/findings.md  results/RESEARCH_REPORT.md")


if __name__ == "__main__":
    main()
