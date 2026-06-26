# Roadmap

This project is a measurement protocol and the verdicts it returns. The method and the
mechanical locator have value independent of any single result; what follows is the concrete,
public plan for continuing it. Each item is falsifiable work, not aspiration.

## Where things stand

**v1 (local, open-weight) — SUGGESTIVE.** Same-family judges co-located the answerer's
specific error step ~2× above chance vs cross-family, but the perplexity-controlled
coefficient's 95% CI just touched zero, *and the cross arm was capability-confounded*.
See [results/RESEARCH_REPORT.md](results/RESEARCH_REPORT.md).

**v2 (capability-matched, cross-family API) — NULL on shared blind spot.** With the cross
judge capability-matched (gpt-4o vs claude-haiku, within 3pp on held-out calibration) and
N=500, excess(same)=+0.16 vs excess(cross)=+0.26, gap −0.10, bootstrap 95% CI [−0.24, +0.04],
8% of resamples positive. The v1 same>cross signal **does not survive** capability matching.
Same-family over-endorsement *does* survive (FER 0.38 vs 0.18) but is not error-step-coupled
→ **self-preference / leniency, not a shared blind spot.**
See [results/phase2/RESEARCH_REPORT_v2.md](results/phase2/RESEARCH_REPORT_v2.md).

## Done

### ✓ Capability-matched cross-family API judge + more N — [issue #1](https://github.com/charlnoc/shared-blind-spot-vs-self-preference/issues/1)
Completed in v2. Replaced the capability-confounded local cross judge with a genuinely
cross-family, capability-matched API judge (Anthropic claude-haiku vs OpenAI gpt-4o), held
everything else identical, and raised N to 500. Result: a clean **NULL** for the
family-specific shared-blind-spot mechanism — the single highest-leverage test, and it
resolved the v1 ambiguity in the honest direction.

## Next (in priority order)

### 1. Non-templated domains, with a real familiarity control
The main open thread. Two things v2 could not do: (a) on templated arithmetic, perplexity
barely varies, so the familiarity proxy has little to bite on; (b) the gpt-4o / Claude chat
APIs do not expose echo logprobs, so teacher-forced perplexity on a provided CoT was *not
computable* at all in v2. Move to domains where surface form varies — free-form word problems,
multi-hop unit-conversion chains, short symbolic-logic chains — using **echo-capable models**
so the familiarity control returns. Each domain still needs a **mechanical** error-step locator
(the arithmetic one is the template). This is where a residual shared-blind-spot effect, if any
exists, would most plausibly show up — and where the self-preference reading from v2 can be
tested directly against perplexity.

### 2. Representation-level (Layer 3) confirmation — *gated, not currently active*
Originally gated on a positive, perplexity-robust SER. v2 was NULL, so this is **not** pursued
on the current domain. It would only become relevant if a non-templated domain (item 1) revived
a positive, perplexity-robust signal: extract hidden states around the identified error step and
test whether same-family trajectories diverge together more than cross-family.

## Standing invariants (won't be traded away for a better headline)

- The error-step locator stays **mechanical** — no LLM in the loop, ever.
- Negative / underpowered results are reported as such. v2's NULL is the finding, not a setback.
- Every coverage cap (excluded / unlocalizable cases) is logged, never silently dropped.
- Capability matching is a precondition for any same-vs-cross claim — the lesson of v1.
