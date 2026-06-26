# The mechanical error-step locator

`src/error_locator.py` is the reusable instrument at the centre of this project.
It answers one question about a chain-of-thought: **which step first goes
wrong?** — and it answers it **without ever calling an LLM**.

## Why model-free is the whole point

If you study judge bias, the tempting shortcut is to ask an LLM "where is the
error in this reasoning?" That **reintroduces the exact bias you are trying to
measure**: an LLM locator shares the blind spots of the models under study, so
your "ground-truth error location" is contaminated by the phenomenon you want to
isolate. This locator is pure re-execution of arithmetic, so its error
positions are objective and independent of any model.

## Two layers (first failure wins)

For a CoT parsed into steps `a op b = c`:

1. **arithmetic** — `apply_op(op, a, b) != c` → the step is internally inconsistent.
2. **carry** — a step's left operand ≠ the value carried from the previous step.
3. **divergence** — `(op, operand)` ≠ the canonical operation at that index
   (needs the problem; distinguishes wrong-operand from wrong-operation).

Checks 1–2 need only the CoT; check 3 needs the canonical problem.

## Robust to weak models adding/dropping steps

Small models don't emit clean one-step-per-line CoTs — they merge, skip, or
hallucinate steps. A strict index-aligned locator throws those away (and they're
disproportionately the *errorful* cases you care about). So the production
entry point falls back to a **value-trace** matcher:

- it greedily matches the canonical checkpoint values `[v1..vN]` as an ordered
  subsequence of the values the model actually computed;
- the error step = the first canonical checkpoint the model fails to reach in
  order — well-defined in **canonical-operation space** regardless of how many
  lines the model wrote.

This also gives a clean co-location notion: two solutions "share an error step"
iff they first diverge at the same canonical operation.

## Public API

```python
from error_locator import locate_canonical_index, locate_error, parse_cot

# Production: hybrid strict-then-valuetrace, returns (index | None, method)
idx, method = locate_canonical_index(cot_text, problem)
#   method ∈ {"strict", "valuetrace", "unlocalizable"}; idx is None when correct

# Exact strict locator over already-parsed steps (for aligned CoTs / tests)
report = locate_error(parse_cot(cot_text), problem=problem)  # ErrorReport | None
#   report.index, report.kind ∈ {arithmetic, carry, wrong_operand, wrong_operation}
```

## Adapting it to your own domain

The locator is arithmetic-specific by design, but the *pattern* transfers: to
reuse it elsewhere you need (a) a domain whose steps are **mechanically
checkable**, (b) a canonical step/value sequence to compare against, and (c) a
parser from raw CoT text to that sequence. Keep the checker mechanical and you
keep the property that makes this worth reusing.

## Tested

`tests/test_backbone.py` (run via `./run_tests.sh`, stdlib only, <0.1s):
generate a problem → inject a known error of a known kind at a known step →
require the locator to recover exactly that step. Swept over every position,
every corruption kind, and many seeds: **>1,200 cases**, exact when aligned,
>97% for the value-trace fallback, zero false positives on correct CoTs.
