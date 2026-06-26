"""Shared helper for the isolated phase processes."""
from __future__ import annotations

from error_locator import locate_canonical_index
from prompts import build_solve_messages, parse_answer


def solve_record(lm, problem) -> dict:
    """Greedy-solve one problem and mechanically locate its error step.
    Never raises — a failed generation becomes an 'error' record."""
    try:
        cot = lm.chat(build_solve_messages(problem))
    except Exception as e:  # noqa: BLE001
        return {"cot": "", "answer": None, "correct": False,
                "err_index": None, "err_method": "error", "error": str(e)}
    ans = parse_answer(cot)
    idx, method = locate_canonical_index(cot, problem)
    return {"cot": cot, "answer": ans, "correct": ans == problem.gt,
            "err_index": idx, "err_method": method}
