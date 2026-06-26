"""Prompt construction + output parsing for the arithmetic domain.

The solve prompt enforces a strict, locator-friendly output contract — one
arithmetic operation per line as `a op b = c`, in the order the problem gives
them, ending in `Answer: N`. This makes the *format* parseable while leaving
the *computation* entirely to the model, so genuine arithmetic/carry errors
still surface (spec §2 wants M errorful at ~50-80% accuracy).

The judge prompt never contains the ground truth (spec §1).
"""

from __future__ import annotations

import re

from arithmetic import Problem
from error_locator import parse_cot

# --- few-shot exemplar, in the exact rendered style of arithmetic._render_text
_SOLVE_SYS = (
    "You solve arithmetic word problems. Work step by step. On each line show "
    "exactly ONE arithmetic operation written as `a op b = c` (op is +, -, or "
    "*), applying the operations in the order the problem states them. After "
    "the steps, write a final line `Answer: N` with the final number."
)

_SOLVE_EXAMPLE_Q = (
    "A bin starts with 7 items, then 5 more are added, then the amount is "
    "multiplied by 3, then 4 are removed. How many items are in the bin at the end?"
)
_SOLVE_EXAMPLE_A = "Step 1: 7 + 5 = 12\nStep 2: 12 * 3 = 36\nStep 3: 36 - 4 = 32\nAnswer: 32"

_JUDGE_SYS = (
    "You check whether a proposed solution to an arithmetic word problem is "
    "correct. You are given the problem and someone's worked solution, but NOT "
    "the official answer. Decide whether their final answer is correct. Reason "
    "step by step, then on the final line write exactly `Verdict: correct` or "
    "`Verdict: wrong`."
)


def build_solve_messages(problem: Problem) -> list[dict]:
    return [
        {"role": "system", "content": _SOLVE_SYS},
        {"role": "user", "content": _SOLVE_EXAMPLE_Q},
        {"role": "assistant", "content": _SOLVE_EXAMPLE_A},
        {"role": "user", "content": problem.text},
    ]


# Balanced few-shot (one correct, one wrong) so weaker judges actually USE both
# labels and re-check the arithmetic instead of defaulting to one verdict.
_JUDGE_EX1_USER = (
    "Problem:\nA bin starts with 7 items, then 5 more are added, then the amount "
    "is multiplied by 3, then 4 are removed. How many items are in the bin at the end?\n\n"
    "Proposed solution:\nStep 1: 7 + 5 = 12\nStep 2: 12 * 3 = 36\nStep 3: 36 - 4 = 32\n"
    "Answer: 32\n\nIs the final answer correct?"
)
_JUDGE_EX1_ASST = (
    "Check each step: 7+5=12 ok; 12*3=36 ok; 36-4=32 ok. Final answer matches.\n"
    "Verdict: correct"
)
_JUDGE_EX2_USER = (
    "Problem:\nA bin starts with 10 items, then the amount is multiplied by 2, "
    "then 3 are removed. How many items are in the bin at the end?\n\n"
    "Proposed solution:\nStep 1: 10 * 2 = 20\nStep 2: 20 - 3 = 16\nAnswer: 16\n\n"
    "Is the final answer correct?"
)
_JUDGE_EX2_ASST = (
    "Check each step: 10*2=20 ok; but 20-3=17, not 16, so the last step is wrong.\n"
    "Verdict: wrong"
)


def build_judge_messages(problem: Problem, cot_text: str, answer) -> list[dict]:
    solution = f"{cot_text}\nAnswer: {answer}" if "Answer:" not in cot_text else cot_text
    user = (
        f"Problem:\n{problem.text}\n\n"
        f"Proposed solution:\n{solution}\n\n"
        "Is the final answer correct?"
    )
    return [
        {"role": "system", "content": _JUDGE_SYS},
        {"role": "user", "content": _JUDGE_EX1_USER},
        {"role": "assistant", "content": _JUDGE_EX1_ASST},
        {"role": "user", "content": _JUDGE_EX2_USER},
        {"role": "assistant", "content": _JUDGE_EX2_ASST},
        {"role": "user", "content": user},
    ]


_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"Verdict:\s*(correct|wrong)", re.IGNORECASE)


def parse_answer(text: str) -> int | None:
    """Final answer = last `Answer: N`; fall back to the last equation result."""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return int(matches[-1])
    steps = parse_cot(text)
    return steps[-1].result if steps else None


def parse_verdict(text: str) -> bool | None:
    """Return True if the judge endorsed (said 'correct'), False if 'wrong',
    None if no clear verdict was emitted (excluded from analysis)."""
    matches = _VERDICT_RE.findall(text)
    if matches:
        return matches[-1].lower() == "correct"
    return None
