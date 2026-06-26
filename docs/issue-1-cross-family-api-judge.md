# Issue #1 — Harden the cross-family arm with a capability-matched API judge, and scale N

> Ready-to-file issue. Once the repo has a GitHub remote, file it with:
> `gh issue create --title "Capability-matched cross-family API judge + larger N" --body-file docs/issue-1-cross-family-api-judge.md --label "roadmap,help wanted"`

## Problem

The v1 cross-family judge is **SmolLM2-1.7B**, which cannot reliably check
multi-step arithmetic — it rubber-stamps (FER ≈ 0.98). That is *fine* for
generating false-endorsement cases and localizable independent solutions (which
is all SER needs), but it means the same-vs-cross contrast is confounded by
**judge capability**, not purely by **family**. Qwen2.5 is also unusually
math-strong for its size, widening that gap. So the robust v1 claim is
"SER(same) vs its *own* chance baseline", and the cross arm is only secondary.

The pooled perplexity-controlled `same_family` coefficient is positive (+0.91)
but its 95% bootstrap CI just touches zero (lower bound −0.05) on **n=38** usable
same-family cases. Underpowered, not null.

## Proposed work

1. **Add an API cross-family judge** (e.g. a Claude model) that is genuinely
   capability-matched to the same-family judge — a competent arithmetic checker,
   different family. This removes the "weak cross judge" confound and gives a
   cross arm worth comparing against directly.
   - Extend `src/models.py` with an API-backed `LM`-compatible class
     (`.chat`, and a perplexity proxy or an explicit "perplexity unavailable"
     path for closed models — log it, don't fake it).
   - Add it as a third judge in `src/expconfig.py::JUDGES`.
2. **Scale N** (e.g. 500–1000) so the usable same-family SER denominator is large
   enough to move the perplexity-controlled CI off zero — in either direction.
3. **Keep the same-family judge capability-matched too**: consider a
   different-size pairing tuned so judge and answerer are close in ability (a
   judge far stronger than the answerer rarely shares its errors → starves SER).

## Definition of done

- `RESEARCH_REPORT.md` regenerates with a capability-matched cross arm and a
  same-family SER CI based on a substantially larger n.
- The verdict moves to **POSITIVE** or **NULL** with the perplexity control
  intact — either is a publishable outcome. No p-hacking toward positive.

## Non-goals / guardrails

- The error-step locator stays mechanical (no LLM locator).
- Closed-model perplexity that can't be computed is reported as missing, not
  approximated into the regression.
