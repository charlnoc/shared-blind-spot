"""Step 0 — Provenance & CoT-availability probe.

For each candidate API model, send the SOLVE prompt on a few generated problems
and confirm it emits a full, parseable step-by-step CoT in the `a op b = c` /
`Answer: N` contract the mechanical locator needs. A model that won't expose a
usable reasoning trace is disqualified as a judge.

Usage:
  python src/phase2_probe.py                 # probe the default candidate set
  python src/phase2_probe.py gpt-4o-mini     # probe just one (plumbing check)

Writes results/phase2/models.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from api_models import API_MODELS, APILM, resolve
from arithmetic import generate_dataset
from error_locator import parse_cot
from prompts import build_solve_messages, parse_answer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2")
os.makedirs(OUT_DIR, exist_ok=True)

# A few problems at moderate difficulty just to exercise the format.
PROBE_PROBS = generate_dataset(3, n_steps=6, max_add=50, max_mul=4, seed0=90000)


def probe_one(key: str) -> dict:
    spec = resolve(key)
    lm = APILM(key)
    rows = []
    for p in PROBE_PROBS:
        text = lm.chat(build_solve_messages(p), role="probe", problem_id=p.seed)
        steps = parse_cot(text)
        ans = parse_answer(text)
        rows.append({
            "seed": p.seed, "n_ops": len(p.ops), "gt": p.gt,
            "n_steps_parsed": len(steps), "answer": ans,
            "correct": ans == p.gt,
            "aligned": len(steps) == len(p.ops),
            "sample": text[:400],
        })
    n_ok = sum(r["n_steps_parsed"] >= 1 and r["answer"] is not None for r in rows)
    return {"key": key, "provider": spec["provider"], "model": spec["model"],
            "usable_cot": n_ok == len(rows), "n_ok": n_ok, "rows": rows}


def main():
    keys = sys.argv[1:] or list(API_MODELS)
    results = []
    for k in keys:
        print(f"[probe] {k} ...", flush=True)
        try:
            r = probe_one(k)
        except Exception as e:  # noqa: BLE001
            r = {"key": k, "error": f"{type(e).__name__}: {e}"}
            print(f"[probe] {k} FAILED: {r['error']}", flush=True)
            results.append(r)
            continue
        acc = sum(x["correct"] for x in r["rows"])
        print(f"[probe] {k}: usable_cot={r['usable_cot']} "
              f"acc={acc}/{len(r['rows'])} aligned="
              f"{sum(x['aligned'] for x in r['rows'])}/{len(r['rows'])}", flush=True)
        results.append(r)

    lines = ["# Phase 2 — Step 0: model roles & CoT availability\n",
             "Probe date: 2026-06-26. Decoding: temperature 0 (greedy). "
             "Domain: templated multi-step arithmetic (same generator as the "
             "local run). A model is judge-eligible only if it emits a full, "
             "parseable `a op b = c` / `Answer: N` chain.\n",
             "| key | provider | model id | usable CoT? | probe acc | aligned |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['key']} | — | — | ERROR | {r['error']} | — |")
            continue
        acc = sum(x["correct"] for x in r["rows"])
        al = sum(x["aligned"] for x in r["rows"])
        n = len(r["rows"])
        lines.append(f"| {r['key']} | {r['provider']} | `{r['model']}` | "
                     f"{'yes' if r['usable_cot'] else 'NO'} | {acc}/{n} | {al}/{n} |")
    lines.append("\n## Sample outputs\n")
    for r in results:
        if "error" in r:
            continue
        lines.append(f"### {r['key']} (`{r['model']}`)\n")
        ex = r["rows"][0]
        lines.append(f"Problem seed {ex['seed']} (gt={ex['gt']}):\n")
        lines.append("```\n" + ex["sample"] + "\n```\n")

    with open(os.path.join(OUT_DIR, "models.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[probe] wrote {os.path.join(OUT_DIR, 'models.md')}", flush=True)


if __name__ == "__main__":
    main()
