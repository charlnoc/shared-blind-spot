# scratch/ — exploratory dev scripts

These are kept for transparency about how the experiment was built and
de-risked. They are **not** the entry point — use `./run_all.sh` (and
`./run_tests.sh`) from the repo root.

- `smoke_test.py`, `calibrate.py` — early difficulty/format calibration of the
  small models.
- `inspect_align.py`, `recheck_judge.py`, `quick_judge.py` — one-off diagnostics
  (CoT alignment failures, judge-discrimination checks).
- `master.py` — a single-process orchestrator. **Superseded and known-broken:**
  loading several models in one process corrupts Apple's MPS memory pool and
  hangs. `run_all.sh` (one process per model) exists because of this.
