"""Phase 2 assembly: merge per-phase JSONL into run_v2.json (same schema as the
local run) and emit figures into results/phase2/ (so published v1 figures are
untouched). No models loaded.

The decisive Phase-2 quantity — excess(same) − excess(cross) with a bootstrap CI
— does not need perplexity (uncomputable on the chat APIs); it is added here and
surfaced by phase2_report.py.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from phase2config import ANSWERER, DATASET, JUDGES, RESIDUAL_GAP, RUN_TAG

PH2 = os.path.join(os.path.dirname(__file__), "..", "results", "phase2")
RUNS = os.path.join(PH2, "runs")


def _index(path):
    return {json.loads(l)["problem_id"]: json.loads(l) for l in open(path)}


def main():
    mrecs = _index(os.path.join(RUNS, "_M.jsonl"))
    judge_data = {fam: _index(os.path.join(RUNS, f"_{fam}.jsonl")) for fam, _ in JUDGES}
    records = []
    for i in sorted(mrecs):
        r = mrecs[i]
        r["judges"] = {}
        for fam, _ in JUDGES:
            jd = judge_data[fam].get(i)
            if jd is not None:
                r["judges"][fam] = {k: jd[k] for k in
                                    ("family", "model", "verdict_text", "endorsed", "solve", "ppl_on_M")}
        records.append(r)
    out = os.path.join(RUNS, f"run_{RUN_TAG}.json")
    with open(out, "w") as f:
        json.dump({"config": {"DATASET": DATASET, "ANSWERER": ANSWERER,
                              "JUDGES": [list(j) for j in JUDGES], "n": len(records),
                              "residual_gap": RESIDUAL_GAP,
                              "created": datetime.now().isoformat()},
                   "records": records}, f, indent=2)
    print(f"[assemble] wrote {out} ({len(records)} records)", flush=True)

    # figures into results/phase2/ (do NOT clobber published v1 figures)
    import plot
    plot.RESULTS = PH2
    plot.main(out)
    # the decisive Phase-2 figure (replaces the N/A perplexity panel)
    import phase2_plot
    _, boot = phase2_plot.main(out, PH2)
    print(f"[assemble] excess gap={boot['gap']:+.3f} CI=[{boot['ci'][0]:+.3f},"
          f"{boot['ci'][1]:+.3f}] frac>0={boot['frac_gap_positive']:.0%}", flush=True)

    import phase2_report
    with open(os.path.join(PH2, "RESEARCH_REPORT_v2.md"), "w") as f:
        f.write(phase2_report.build(out))
    print("[assemble] figures + RESEARCH_REPORT_v2.md written", flush=True)


if __name__ == "__main__":
    main()
