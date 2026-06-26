"""Phase 2 run configuration (capability-matched API cross-family judge).

DATASET difficulty and the cross-family judge tier are set by Step-1 calibration
(see results/phase2/calibration.md). Held identical to the local run otherwise:
same generator, same locator, same SER/chance/perplexity definitions.

Seeds: main run seed0=10000 (disjoint from held-out calibration 80000.. and the
probe 90000..). Different difficulty params from v1, so problems never coincide.
"""

# --- set by calibration (Step 1, level L5) ---------------------------------
# Locked on 60 held-out problems (seeds 80000..): gpt-4o 25% vs claude-haiku 28%
# standalone (residual gap 3pp — PASSES the <=10pp match gate). Both judges
# errorful at ~75% (45/60, 43/60 wrong+localized) -> abundant SER events. The
# answerer gpt-4o-mini is maximally errorful (3%, 58/60). L4d was REJECTED: its
# n=20 search match (35/40) was noise; on n=60 it was gpt-4o 23% vs haiku 45%
# (22pp gap) — would have rebuilt the capability-proximity confound. See
# results/phase2/calibration.md and calibration_L4d.md.
DATASET = dict(n_steps=16, max_add=999, max_mul=99, max_value=10**15, seed0=10000)
N = 500                                        # extended from 250 to tighten the cross arm (cache reuses first 250)
ANSWERER = "gpt-4o-mini"                       # same-family SMALL (errorful source)
JUDGES = [("same", "gpt-4o"),                  # same-family LARGE
          ("cross", "claude-haiku")]           # different provider, capability-matched
RUN_TAG = "v2_api"
RESIDUAL_GAP = 0.03                            # judge standalone-accuracy gap (calibration)
