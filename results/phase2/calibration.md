# Phase 2 — Step 1B: locked calibration

Chosen difficulty **L5**: {'n_steps': 16, 'max_add': 999, 'max_mul': 99, 'max_value': 1000000000000000}

Held-out N=60, seeds 80000.. (disjoint from main run).

| role | model | standalone acc | wrong+localized |
|---|---|---|---|
| answerer (M) | `gpt-4o-mini` | 3% | 58/60 |
| judge same | `gpt-4o` | 25% | 45/60 |
| judge cross | `claude-haiku` | 28% | 43/60 |

**Residual judge-accuracy gap = 3%** (gate: ≤~10pp). This gap is carried into the final regression as a covariate.

Answerer is the errorful source of false-endorsement cases; both judges are errorful enough to have wrong independent solves (needed for SER co-location).


## Difficulty search that led here (n=20 screen, then n=60 lock)

These API models are far stronger at arithmetic than the local v1 models, so
difficulty had to be pushed hard (16-step chains with 2-digit multipliers) to
make the *judges* errorful enough to produce shared-error events.

| level | params | gpt-4o (same) | claude-haiku (cross) | gap |
|---|---|---|---|---|
| L3 | 14 steps, ×≤19 | 95% | 100% | 5pp |
| L4 | 16 steps, ×≤25 | 90% | 90% | 0pp* |
| L4b | 16 steps, ×≤40 | 55% | 80% | 25pp |
| L4c | 16 steps, ×≤55 | 45% | 65% | 20pp |
| L4d | 16 steps, ×≤70 | **23%** (n60) | **45%** (n60) | **22pp — REJECTED** |
| **L5** | **16 steps, ×≤99** | **25%** (n60) | **28%** (n60) | **3pp — CHOSEN** |

\* L4's 0pp and L4d's n=20 match were small-sample noise. claude-haiku is a
uniformly stronger arithmetic reasoner than gpt-4o across the mid band, so the
two judges' accuracy curves only cross cleanly near the floor (L5). L5 is the
one difficulty where they are genuinely capability-matched (3pp) while both stay
errorful enough (~75%) to generate abundant wrong+localized independent solves.
claude-sonnet was dropped early (L5 search: 80% vs gpt-4o 25% — far too strong).
