# Phase 2 — Step 1B: locked calibration

Chosen difficulty **L4d**: {'n_steps': 16, 'max_add': 999, 'max_mul': 70, 'max_value': 100000000000000}

Held-out N=60, seeds 80000.. (disjoint from main run).

| role | model | standalone acc | wrong+localized |
|---|---|---|---|
| answerer (M) | `gpt-4o-mini` | 3% | 58/60 |
| judge same | `gpt-4o` | 23% | 46/60 |
| judge cross | `claude-haiku` | 45% | 33/60 |

**Residual judge-accuracy gap = 22%** (gate: ≤~10pp). This gap is carried into the final regression as a covariate.

Answerer is the errorful source of false-endorsement cases; both judges are errorful enough to have wrong independent solves (needed for SER co-location).

