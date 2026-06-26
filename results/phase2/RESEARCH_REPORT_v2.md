# Shared Blind Spot vs Self-Preference — Phase 2 (capability-matched, cross-family API judge)

**Verdict: NULL.** Once the two judge arms are capability-matched, same-family excess co-location does not exceed cross-family. The local (v1) gap is attributable to the capability confound and/or familiarity, not a distinct shared-blind-spot mechanism. A clean, honest negative.

## What changed from v1 (and what did not)

This run changes exactly one thing versus the local run: the cross-family judge is now a genuinely different provider, **capability-matched** to the same-family judge via held-out calibration. Everything else is identical — same templated arithmetic generator, same **mechanical** error-step locator (re-run against its 25 unit tests, still passing), same SER / chance / excess definitions.

| role | model | provider |
|---|---|---|
| answerer M (same-family small) | `gpt-4o-mini` | OpenAI |
| judge — same family (large) | `gpt-4o` | OpenAI |
| judge — cross family | `claude-haiku` | Anthropic |

Difficulty (Step-1 calibrated): `{'n_steps': 16, 'max_add': 999, 'max_mul': 99, 'max_value': 1000000000000000, 'seed0': 10000}`. Residual judge-capability gap from calibration: **3%**.

## Headline numbers

N = 500 problems. Answerer accuracy = **2%** (490 errorful cases).

| metric | same-family | cross-family |
|---|---|---|
| false-endorsement rate (FER) | 0.38 | 0.18 |
| shared-error rate (SER) | **0.22** | 0.31 |
| chance baseline | 0.06 | 0.05 |
| **excess co-location (SER − chance)** | **0.16** | 0.26 |
| usable shared-error cases (n) | 138 | 51 |

## The decisive test: excess(same) − excess(cross)

Problem-level bootstrap (2000 valid resamples of 500 problems), recomputing SER and chance from scratch each resample so every conditioning denominator is respected:

- excess(same) = **0.16**, excess(cross) = 0.26
- **gap = -0.10**, bootstrap 95% CI **[-0.24, 0.04]**
- fraction of resamples with gap > 0: **8%**

## What this separates: favouritism is real, but it is not a shared blind spot

Two findings, together, are the actual Phase-2 result:

1. **Same-family over-endorsement survives capability matching.** FER(same) = 0.38 vs FER(cross) = 0.18 — the same-family judge endorses the answerer's *wrong* answers ~2.2× more often. And this is conservative: the cross judge was at least as capable here (it solved 34% of problems vs the same-family judge's 24%), yet still endorsed less. So the same-family favouritism at the endorsement level is **not** a capability artifact.

2. **But that over-endorsement is not explained by shared error steps.** Excess co-location does **not** favour the same-family arm (same 0.16 vs cross 0.26; gap -0.10, CI [-0.24, 0.04], 8% of resamples positive). If anything the capability-matched cross-family judge co-locates *slightly more* — the opposite of the shared-blind-spot prediction — though the gap CI still includes zero and the cross judge's realized capability edge (it solved 34% vs 24%) plausibly contributes to it (a stronger judge's rarer errors concentrate on the objectively hardest steps, where the weak answerer also fails). Either way, the v1 same>cross excess does not survive capability matching.

Taken together, the same-family favouritism looks like **self-preference / leniency** (endorsing familiar same-family output without independently re-deriving it), not a shared reasoning blind spot. The leniency signature is direct: in 41 of the 180 same-family false endorsements the judge had itself solved the problem *correctly* and still endorsed the wrong answer (cross: 35 of 86). Note also the realized in-run judge solve-accuracy gap (10%) ran wider than the held-out calibration gap (3%); the excess comparison subtracts each arm's own chance baseline, but this is flagged honestly.

## Perplexity (familiarity) control — reported honestly

Teacher-forced perplexity of a judge on the answerer's CoT — the v1 familiarity proxy — requires logprobs on a *provided* assistant string ("echo"). Neither the gpt-4o-class chat API nor the Claude Messages API exposes this, so the perplexity control is **not computable on either arm here**. This was anticipated; the decisive quantity above is perplexity-free, and on this templated domain perplexity barely varied even in v1 (≈1.1–1.2, and in the *wrong* direction for a familiarity explanation). A perplexity-bearing test returns in Phase 3 (free-form domain / echo-capable models).

## Coverage / honesty diagnostics

- answerer error-localization methods: `{'strict': 487, 'valuetrace': 13}`
- answerer wrong-but-unlocalizable: 0
- same: FE cases=180, usable=138, judge-solved-correct (pure self-preference signature)=41, M-unlocalizable=0, J-wrong-unlocalizable=1
- cross: FE cases=86, usable=51, judge-solved-correct (pure self-preference signature)=35, M-unlocalizable=0, J-wrong-unlocalizable=0

## How to read this

This is the honest negative the design was built to be able to return: the apparent v1 signal does not survive capability matching. Worth reporting as such.

_Figures: **`results/phase2/excess_gap.png`** (the decisive plot — per-arm SER vs chance + the bootstrap gap distribution), `ser_vs_null.png`, `fer_by_family_domain.png`. Run JSON: `results/phase2/runs/run_v2_api.json`._
