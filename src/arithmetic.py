"""Arithmetic domain: programmatic multi-step word-problem generator + GT.

Layer-0 backbone for the shared-blind-spot experiment
(see shared_blind_spot_experiment.md §2 and §9 step 1).

Everything in this module is **deterministic and model-free** so it can be
unit-tested in isolation (spec §8: the mechanical error-step locator and the
data it runs on must never depend on an LLM, or we reintroduce the bias under
study).

Design choice for the *minimal validated case* (spec §8 "start with ONE
domain"): problems are a **linear chain** of operations over a single running
value. This is the easiest structure to localize an error step in
mechanically and unambiguously. The generator can later be extended to
tree-structured / multi-variable problems; the linear chain is the gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# op-name -> (rendered symbol, python operation)
OPS = {
    "add": ("+", lambda a, b: a + b),
    "sub": ("-", lambda a, b: a - b),
    "mul": ("*", lambda a, b: a * b),
}


def apply_op(op: str, a: int, b: int) -> int:
    return OPS[op][1](a, b)


@dataclass(frozen=True)
class CoTStep:
    """One claimed step of a chain of thought: `left  <op>  operand = result`.

    `left`, `operand`, `result` are what the *solver claims*. For a correct
    step, apply_op(op, left, operand) == result, and (for steps after the
    first) `left` equals the previous step's `result`.
    """

    left: int
    op: str
    operand: int
    result: int

    @property
    def symbol(self) -> str:
        return OPS[self.op][0]

    @property
    def signature(self) -> tuple[str, int]:
        """The operation identity used for shared-error co-location (spec §5.4:
        'same operation / same inference')."""
        return (self.op, self.operand)

    def render(self, idx: int) -> str:
        return f"Step {idx}: {self.left} {self.symbol} {self.operand} = {self.result}"


@dataclass
class Problem:
    """A generated arithmetic word problem with everything we know in code."""

    x0: int
    ops: list[tuple[str, int]]  # the canonical operation chain: [(op, operand), ...]
    text: str
    gt: int
    seed: int

    def canonical_cot(self) -> list[CoTStep]:
        """The fully correct chain of thought for this problem."""
        steps, _ = execute(self.x0, self.ops)
        return steps

    def render_cot(self, steps: list[CoTStep]) -> str:
        body = "\n".join(s.render(i + 1) for i, s in enumerate(steps))
        return f"{body}\nAnswer: {steps[-1].result if steps else self.x0}"


def execute(x0: int, ops: list[tuple[str, int]]) -> tuple[list[CoTStep], int]:
    """Correctly execute an operation chain, returning the CoT and final value."""
    steps: list[CoTStep] = []
    cur = x0
    for op, b in ops:
        res = apply_op(op, cur, b)
        steps.append(CoTStep(left=cur, op=op, operand=b, result=res))
        cur = res
    return steps, cur


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_PHRASE = {
    "add": "then {b} more are added",
    "sub": "then {b} are removed",
    "mul": "then the amount is multiplied by {b}",
}


def _render_text(x0: int, ops: list[tuple[str, int]]) -> str:
    parts = [f"A bin starts with {x0} items"]
    parts += [_PHRASE[op].format(b=b) for op, b in ops]
    return ", ".join(parts) + ". How many items are in the bin at the end?"


def generate_problem(
    seed: int,
    n_steps: int = 4,
    max_add: int = 50,
    max_mul: int = 4,
    allow_ops: tuple[str, ...] = ("add", "sub", "mul"),
    max_value: int = 10_000,
) -> Problem:
    """Generate one problem with controllable difficulty (spec §2).

    Constraints enforced: no negative intermediate value, results bounded by
    `max_value`, no zero operands (so corruptions always change the answer).
    """
    rng = random.Random(seed)
    x0 = rng.randint(1, max_add)
    ops: list[tuple[str, int]] = []
    cur = x0
    for _ in range(n_steps):
        # pick an op that keeps the running value non-negative and bounded
        candidates = list(allow_ops)
        rng.shuffle(candidates)
        for op in candidates:
            if op == "mul":
                b = rng.randint(2, max_mul)
            else:
                b = rng.randint(1, max_add)
            res = apply_op(op, cur, b)
            if 0 <= res <= max_value:
                ops.append((op, b))
                cur = res
                break
        else:
            # no op fit (rare near bounds); fall back to a safe add of 1
            ops.append(("add", 1))
            cur = cur + 1
    _, gt = execute(x0, ops)
    return Problem(x0=x0, ops=ops, text=_render_text(x0, ops), gt=gt, seed=seed)


def generate_dataset(n: int, seed0: int = 0, **kwargs) -> list[Problem]:
    return [generate_problem(seed0 + i, **kwargs) for i in range(n)]


# ---------------------------------------------------------------------------
# Corruption: produce a CoT with a KNOWN error step (for testing the locator)
# ---------------------------------------------------------------------------

# The four corruption kinds map 1:1 to the four checks the locator performs.
ERROR_KINDS = ("arithmetic", "carry", "wrong_operand", "wrong_operation")


@dataclass(frozen=True)
class ErrorTruth:
    """Ground-truth error annotation for a corrupted CoT."""

    index: int  # 0-based index of the first wrong step
    kind: str   # one of ERROR_KINDS


def _different_operand(op: str, b: int, rng: random.Random, max_add: int, max_mul: int) -> int:
    hi = max_mul if op == "mul" else max_add
    lo = 2 if op == "mul" else 1
    choices = [x for x in range(lo, hi + 1) if x != b]
    return rng.choice(choices) if choices else b + 1


def _different_op(op: str, rng: random.Random) -> str:
    return rng.choice([o for o in OPS if o != op])


def corrupt_cot(
    problem: Problem,
    index: int,
    kind: str,
    seed: int,
    max_add: int = 50,
    max_mul: int = 4,
) -> tuple[list[CoTStep], ErrorTruth]:
    """Inject a single error of `kind` at step `index`.

    Downstream steps are kept *internally consistent with the corrupted value*
    (a model that makes one mistake then computes correctly from it), so the
    ONLY locally/canonically detectable fault is at `index`. This is precisely
    the controlled material §8 demands for validating the locator.
    """
    assert kind in ERROR_KINDS, kind
    assert 0 <= index < len(problem.ops)
    rng = random.Random(seed)
    claimed: list[CoTStep] = []
    cur = problem.x0
    for i, (op, b) in enumerate(problem.ops):
        left = cur
        if i == index:
            if kind == "wrong_operand":
                b = _different_operand(op, b, rng, max_add, max_mul)
            elif kind == "wrong_operation":
                op = _different_op(op, rng)
            elif kind == "carry":
                left = cur + rng.choice([-3, -2, -1, 1, 2, 3])
        res = apply_op(op, left, b)
        if i == index and kind == "arithmetic":
            res = res + rng.choice([-3, -2, -1, 1, 2, 3])
        claimed.append(CoTStep(left=left, op=op, operand=b, result=res))
        cur = res
    return claimed, ErrorTruth(index=index, kind=kind)
