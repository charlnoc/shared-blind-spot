"""Mechanical error-step locator (spec §5.2 and §8 — the single highest-risk
component of the whole experiment).

Given a chain of arithmetic steps, find the **first** step that is wrong, and
classify *how*. There is deliberately **no LLM** anywhere in here: localization
is pure re-execution. Per §5.2, an LLM-judged locator would reintroduce the
exact bias we are trying to measure.

Two layers of checking, applied in order, first failure wins:

  1. arithmetic   — apply_op(op, left, operand) != result            (self-inconsistent)
  2. carry        — left != running value carried from prior step    (used wrong number)
  3. divergence   — (op, operand) != the canonical operation here    (wrong setup/operation)
                    only available when the canonical problem is supplied

Checks 1–2 need only the CoT (`x0` optional, used to validate the first step).
Check 3 needs the `Problem` and additionally distinguishes `wrong_operand`
from `wrong_operation`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arithmetic import CoTStep, Problem, apply_op, execute

# Matches `a <op> b = c` anywhere on a line, e.g. "Step 2: 17 * 3 = 51".
_EQN = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)")
_SYM2OP = {"+": "add", "-": "sub", "*": "mul"}


@dataclass(frozen=True)
class ErrorReport:
    """Where and how the first error occurs. `index` is 0-based.

    `signature` is the (op, operand) identity of the offending step, used for
    shared-error co-location in Layer 2 (spec §5.4)."""

    index: int
    kind: str  # 'arithmetic' | 'carry' | 'wrong_operand' | 'wrong_operation'
    signature: tuple[str, int]
    detail: str


def parse_cot(text: str) -> list[CoTStep]:
    """Extract a CoT from free-ish text. One step per line that contains an
    `a op b = c` equation; lines without one (prose, 'Answer: N') are skipped.

    Robust enough for model output where reasoning lines carry an explicit
    equation. Lines with multiple equations take the *last* one on the line
    (models often write 'so 12 + 5 = 17' — the operative equation is last)."""
    steps: list[CoTStep] = []
    for line in text.splitlines():
        matches = _EQN.findall(line)
        if not matches:
            continue
        a, sym, b, c = matches[-1]
        steps.append(
            CoTStep(left=int(a), op=_SYM2OP[sym], operand=int(b), result=int(c))
        )
    return steps


def locate_error(
    steps: list[CoTStep],
    x0: int | None = None,
    problem: Problem | None = None,
) -> ErrorReport | None:
    """Return the first error in `steps`, or None if the chain is fully correct.

    If `problem` is given, its `x0` and canonical ops are used (divergence
    check enabled). If only `x0` is given, the first-step carry check is
    against `x0`. With neither, only arithmetic + inter-step carry are checked.
    """
    if problem is not None:
        x0 = problem.x0
    canonical_ops = problem.ops if problem is not None else None

    prev_result = x0
    for i, s in enumerate(steps):
        # 1. arithmetic self-consistency
        if apply_op(s.op, s.left, s.operand) != s.result:
            return ErrorReport(
                i, "arithmetic", s.signature,
                f"{s.left} {s.symbol} {s.operand} = {s.result}, "
                f"but should be {apply_op(s.op, s.left, s.operand)}",
            )
        # 2. carry: this step's left must equal the value carried in
        if prev_result is not None and s.left != prev_result:
            return ErrorReport(
                i, "carry", s.signature,
                f"step uses left={s.left} but carried-in value is {prev_result}",
            )
        # 3. divergence from the canonical operation (needs the problem)
        if canonical_ops is not None and i < len(canonical_ops):
            c_op, c_operand = canonical_ops[i]
            if s.op != c_op:
                return ErrorReport(
                    i, "wrong_operation", s.signature,
                    f"used '{s.op}' but canonical step is '{c_op}'",
                )
            if s.operand != c_operand:
                return ErrorReport(
                    i, "wrong_operand", s.signature,
                    f"used operand {s.operand} but canonical is {c_operand}",
                )
        prev_result = s.result
    return None


def canonical_values(problem: Problem) -> list[int]:
    """Correct running value after each canonical operation: [v1, ..., vN]."""
    steps, _ = execute(problem.x0, problem.ops)
    return [s.result for s in steps]


def locate_error_index(
    result_seq: list[int],
    problem: Problem,
) -> int | None:
    """Robust, insertion/deletion-tolerant error localization in CANONICAL
    operation space (used for real model CoTs; spec §5.2/§5.4).

    `result_seq` is the ordered list of values the model actually computed
    (the right-hand sides of its equations). We greedily match the canonical
    checkpoint values [v1..vN] as an ordered subsequence of `result_seq`. The
    error step = the index (0-based, canonical-op space) of the FIRST canonical
    checkpoint the model fails to reach in order. Returns None if the model
    reaches every checkpoint in order (correct path, even with extra steps).

    Why canonical space: two solutions "share an error step" (§5.4) iff they
    first diverge at the same canonical operation — well-defined regardless of
    how many lines each wrote. The operation identity at that index is fixed by
    the problem, so 'same index' == 'same operation'.
    """
    cvals = canonical_values(problem)
    pos = 0
    for k, vk in enumerate(cvals):
        try:
            pos = result_seq.index(vk, pos) + 1
        except ValueError:
            return k
    return None


def locate_from_text(text: str, problem: Problem) -> int | None:
    """Convenience: parse a CoT's equation results and localize in canonical
    space. Returns None if correct OR if no equations could be parsed (caller
    distinguishes via parse_cot length / answer correctness)."""
    results = [s.result for s in parse_cot(text)]
    if not results:
        return None
    return locate_error_index(results, problem)


def locate_canonical_index(text: str, problem: Problem) -> tuple[int | None, str]:
    """Production localizer for real model CoTs. Returns (index, method).

    - If the CoT parses to exactly len(ops) steps: use the EXACT strict locator
      (re-execution; unit-tested to 100%), whose error index is already in
      canonical-op space. method = 'strict'.
    - Else fall back to the insertion/deletion-tolerant value-trace locator.
      method = 'valuetrace' (approximate) or 'unlocalizable' if no equations
      were parsed at all (caller should exclude these and log the rate).

    index is None for a correct path; callers distinguish "correct" from
    "unlocalizable" using the final-answer correctness + the method tag.
    """
    steps = parse_cot(text)
    if len(steps) == len(problem.ops):
        rep = locate_error(steps, problem=problem)
        return (None if rep is None else rep.index), "strict"
    results = [s.result for s in steps]
    if not results:
        return None, "unlocalizable"
    return locate_error_index(results, problem), "valuetrace"


def same_error_index(a: int | None, b: int | None) -> bool:
    """Co-location for canonical-space indices: both wrong at the same step."""
    return a is not None and b is not None and a == b


def same_error(a: ErrorReport | None, b: ErrorReport | None) -> bool:
    """Shared-error co-location test (spec §5.4): two solutions share an error
    step iff both are wrong at the *same index* with the *same operation
    signature*. Two correct solutions do NOT 'share an error'."""
    if a is None or b is None:
        return False
    return a.index == b.index and a.signature == b.signature
