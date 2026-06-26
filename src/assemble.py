"""Phase 3: merge the per-phase JSONL files into the run JSON metrics expects,
then produce all figures + findings.md + RESEARCH_REPORT.md. No models loaded."""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from expconfig import ANSWERER, DATASET, JUDGES, N, RUN_TAG

RUNS = os.path.join(os.path.dirname(__file__), "..", "results", "runs")


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
                              "created": datetime.now().isoformat()},
                   "records": records}, f, indent=2)
    print(f"[assemble] wrote {out} ({len(records)} records)", flush=True)

    import plot
    import report
    import research_report
    plot.main(out)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "findings.md"), "w") as f:
        f.write(report.build(out))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "RESEARCH_REPORT.md"), "w") as f:
        f.write(research_report.build(out))
    print("[assemble] figures + findings.md + RESEARCH_REPORT.md written", flush=True)


if __name__ == "__main__":
    main()
