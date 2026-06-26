# Roadmap

This project is released early and on purpose: the method and the locator have
present value independent of whether the headline result ever reaches POSITIVE.
What follows is the concrete, public plan for continuing it. Each item is
falsifiable work, not aspiration.

## Where v1 stands

Verdict **SUGGESTIVE** (see [results/RESEARCH_REPORT.md](results/RESEARCH_REPORT.md)).
Same-family judges co-locate the answerer's *specific* error step ~2× above
chance vs cross-family, the gap survives the perplexity control, but the pooled
perplexity-controlled coefficient's 95% CI just touches zero on n=38 usable
same-family cases. The signal leans real; the power is thin. The roadmap is
designed to resolve that.

## Next (in priority order)

### 1. Capability-matched cross-family judge + more N  ·  [open issue](docs/issue-1-cross-family-api-judge.md)
The current cross-family judge (SmolLM2-1.7B) is a weak arithmetic checker that
rubber-stamps, so the same-vs-cross contrast is partly confounded by capability,
not just family. Add an **API cross-family judge** (e.g. a Claude model) that is
genuinely capability-matched to the same-family judge, and raise N so the
perplexity-controlled coefficient can clear zero (or honestly fail to). This is
the single highest-leverage step toward POSITIVE-or-NULL.

### 2. Non-templated domains
Step-by-step arithmetic is so templated that whole-CoT perplexity sits near 1
for every judge — which makes the familiarity confound *small here* but also
gives the perplexity control little to bite on. Add domains where surface form
varies: free-form word problems, multi-hop unit-conversion chains, short
symbolic-logic / constraint chains. Each still needs a **mechanical** error-step
locator (the §2 discipline); the arithmetic locator is the template.

### 3. Representation-level (Layer 3) confirmation — *gated*
Only if SER is positive and perplexity-robust: extract hidden states around the
identified error step for both answerer and (same-family) judge, and test
whether their representations are more similar at the shared error step than at
non-error steps or across families — and whether the *textual* error step
actually coincides with where trajectories diverge (CoT-faithfulness check).

## Standing invariants (won't be traded away for a better headline)

- The error-step locator stays **mechanical** — no LLM in the loop, ever.
- Negative/underpowered results are reported as such. A clean NULL is a finding.
- Every coverage cap (excluded/unlocalizable cases) is logged, never silently dropped.
- Each model runs in its own process; results stream to disk incrementally.
