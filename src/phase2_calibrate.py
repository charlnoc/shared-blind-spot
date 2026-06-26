"""Step 1 — Capability calibration (the step that removes the confound).

Two stages:
  A) difficulty search — solve a small held-out set at several difficulty levels
     with answerer + both judge candidates; find a level where the answerer is
     errorful (target ~30-60% acc -> many false-endorsement candidates) AND the
     same-family judge (gpt-4o) and a cross-family judge (a Claude tier) land at
     comparable standalone accuracy (target both in the 50-80% band, gap <=~10pp).
  B) lock-in — on the chosen level, solve the full held-out set with the three
     fixed roles; record standalone accuracies + residual gap -> calibration.md.

Held-out seeds (seed0=80000) are disjoint from the main run (seed0=70000) and
the probe (seed0=90000) so nothing is reused.

Usage:
  python src/phase2_calibrate.py search [N]      # stage A sweep (default N=20)
  python src/phase2_calibrate.py lock LEVEL [N]  # stage B (default N=60)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from api_models import APILM
from arithmetic import generate_dataset
from phaselib import solve_record

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2")
os.makedirs(OUT_DIR, exist_ok=True)

HELDOUT_SEED0 = 80000

# Difficulty ladder. Strong models only err with long chains + large operands +
# multiplications on already-large running values; raise max_value so big muls
# don't trip the generator's bounded-value fallback.
LEVELS = {
    "L1": dict(n_steps=10, max_add=99,   max_mul=9,  max_value=10**6),
    "L2": dict(n_steps=12, max_add=200,  max_mul=12, max_value=10**7),
    "L3": dict(n_steps=14, max_add=500,  max_mul=19, max_value=10**9),
    "L4": dict(n_steps=16, max_add=900,  max_mul=25, max_value=10**12),
    # Harder ladder: 2-digit multipliers force genuine multi-digit x multi-digit
    # multiplication, which is what finally pushes gpt-4o / Claude off 95-100%.
    "L5": dict(n_steps=16, max_add=999,  max_mul=99, max_value=10**15),
    "L6": dict(n_steps=20, max_add=999,  max_mul=99, max_value=10**18),
    "L7": dict(n_steps=24, max_add=2000, max_mul=99, max_value=10**24),
    # Intermediate band: max_mul is the dominant lever (L4 mul≤25 -> gpt-4o 90%;
    # L5 mul≤99 -> gpt-4o 25%). Target the ~55-75% judge band in between.
    "L4b": dict(n_steps=16, max_add=999, max_mul=40, max_value=10**14),
    "L4c": dict(n_steps=16, max_add=999, max_mul=55, max_value=10**14),
    "L4d": dict(n_steps=16, max_add=999, max_mul=70, max_value=10**14),
}

# Roles: answerer (same-family small) + judge candidates.
ANSWERER = "gpt-4o-mini"
SAME_JUDGE = "gpt-4o"
# Sonnet confirmed too strong (L5: 80% vs gpt-4o 25%); haiku tracks gpt-4o (L4:
# both 90%; L5: 25% vs 30%). The cross judge is claude-haiku.
CROSS_CANDIDATES = ["claude-haiku"]


def solve_set(model_key: str, probs) -> dict:
    lm = APILM(model_key)
    n_correct = n_loc = n_wrong_loc = 0
    for p in probs:
        rec = solve_record(lm, p)
        n_correct += rec["correct"]
        if rec["err_method"] != "unlocalizable":
            n_loc += 1
        if not rec["correct"] and rec["err_method"] != "unlocalizable" and rec["err_index"] is not None:
            n_wrong_loc += 1
    n = len(probs)
    return {"model": model_key, "n": n, "acc": n_correct / n,
            "n_correct": n_correct, "n_wrong": n - n_correct,
            "loc_rate": n_loc / n, "n_wrong_localized": n_wrong_loc}


def search(n: int, level_keys=None):
    levels = {k: LEVELS[k] for k in (level_keys or LEVELS)}
    models = [ANSWERER, SAME_JUDGE] + CROSS_CANDIDATES
    print(f"[calib:search] {n} held-out problems x {len(levels)} levels "
          f"({list(levels)}) x {len(models)} models\n", flush=True)
    table = {}
    for lvl, diff in levels.items():
        probs = generate_dataset(n, seed0=HELDOUT_SEED0, **diff)
        table[lvl] = {}
        print(f"--- {lvl}  {diff}", flush=True)
        for mk in models:
            r = solve_set(mk, probs)
            table[lvl][mk] = r
            print(f"   {mk:14s} acc={r['acc']:.0%}  wrong_localized={r['n_wrong_localized']}/{n}", flush=True)
        # capability gap vs each cross candidate at this level
        for cc in CROSS_CANDIDATES:
            gap = abs(table[lvl][SAME_JUDGE]["acc"] - table[lvl][cc]["acc"])
            print(f"   gap(gpt-4o vs {cc}) = {gap:.0%}", flush=True)
        print(flush=True)

    # write a compact search report
    lines = ["# Phase 2 — Step 1A: difficulty / capability search\n",
             f"Held-out seeds {HELDOUT_SEED0}..{HELDOUT_SEED0+n-1} (disjoint from "
             "main run 70000.. and probe 90000..). Decoding greedy. Each cell = "
             "standalone solve accuracy.\n",
             "| level | params | " + ANSWERER + " | " + SAME_JUDGE + " | "
             + " | ".join(CROSS_CANDIDATES) + " |",
             "|---|---|" + "---|" * (2 + len(CROSS_CANDIDATES))]
    for lvl, diff in levels.items():
        cells = [f"{table[lvl][m]['acc']:.0%}" for m in [ANSWERER, SAME_JUDGE] + CROSS_CANDIDATES]
        params = f"steps={diff['n_steps']},add≤{diff['max_add']},mul≤{diff['max_mul']}"
        lines.append(f"| {lvl} | {params} | " + " | ".join(cells) + " |")
    lines.append("\n_answerer should be errorful (≈30–60%); pick the level + "
                 "cross tier where gpt-4o and the Claude judge are closest and "
                 "both sit in ≈50–80%._\n")
    with open(os.path.join(OUT_DIR, "calibration_search.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[calib:search] wrote {os.path.join(OUT_DIR, 'calibration_search.md')}", flush=True)


def lock(level: str, n: int, cross: str):
    diff = LEVELS[level]
    probs = generate_dataset(n, seed0=HELDOUT_SEED0, **diff)
    roles = {"answerer (M)": ANSWERER, "judge same": SAME_JUDGE, "judge cross": cross}
    print(f"[calib:lock] level={level} {diff} N={n}\n", flush=True)
    res = {}
    for role, mk in roles.items():
        r = solve_set(mk, probs)
        res[role] = r
        print(f"   {role:14s} {mk:14s} acc={r['acc']:.0%}  "
              f"wrong_localized={r['n_wrong_localized']}/{n}", flush=True)
    gap = abs(res["judge same"]["acc"] - res["judge cross"]["acc"])
    print(f"\n   residual judge-accuracy gap = {gap:.0%}", flush=True)

    lines = ["# Phase 2 — Step 1B: locked calibration\n",
             f"Chosen difficulty **{level}**: {diff}\n",
             f"Held-out N={n}, seeds {HELDOUT_SEED0}.. (disjoint from main run).\n",
             "| role | model | standalone acc | wrong+localized |",
             "|---|---|---|---|"]
    for role, mk in roles.items():
        r = res[role]
        lines.append(f"| {role} | `{mk}` | {r['acc']:.0%} | {r['n_wrong_localized']}/{n} |")
    lines += [f"\n**Residual judge-accuracy gap = {gap:.0%}** "
              f"(gate: ≤~10pp). This gap is carried into the final regression as "
              "a covariate.\n",
              "Answerer is the errorful source of false-endorsement cases; both "
              "judges are errorful enough to have wrong independent solves "
              "(needed for SER co-location).\n"]
    with open(os.path.join(OUT_DIR, "calibration.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[calib:lock] wrote {os.path.join(OUT_DIR, 'calibration.md')}", flush=True)
    return gap


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "search"
    if cmd == "search":
        # python ... search [N] [L5,L6,L7]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        lk = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        search(n, lk)
    elif cmd == "lock":
        level = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        cross = sys.argv[4] if len(sys.argv) > 4 else "claude-haiku"
        lock(level, n, cross)
