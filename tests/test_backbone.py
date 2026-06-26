"""Unit tests for the deterministic backbone (spec §9 steps 1-2, gated by §8).

These tests ARE the gate: "do not proceed until [the error-step locator] is
reliable." They check three things:

  1. The generator + GT checker are internally correct.
  2. The canonical CoT executes to GT and the locator reports NO error on it.
  3. Round-trip: corrupt a CoT at a KNOWN (index, kind) and the locator must
     recover exactly that index and kind — swept over every position, every
     corruption kind, and many seeds.

Run with:  python -m unittest discover -s tests -t . -v
(from the src/ directory on PYTHONPATH; see run_tests.sh)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arithmetic import (  # noqa: E402
    ERROR_KINDS,
    CoTStep,
    apply_op,
    corrupt_cot,
    execute,
    generate_dataset,
    generate_problem,
)
from error_locator import (  # noqa: E402
    canonical_values,
    locate_canonical_index,
    locate_error,
    locate_error_index,
    parse_cot,
    same_error,
    same_error_index,
)


class TestGenerator(unittest.TestCase):
    def test_reproducible(self):
        a = generate_problem(seed=42)
        b = generate_problem(seed=42)
        self.assertEqual(a.ops, b.ops)
        self.assertEqual(a.gt, b.gt)

    def test_gt_matches_manual_execution(self):
        # Independently recompute GT by folding the op chain.
        for p in generate_dataset(200, n_steps=5):
            cur = p.x0
            for op, b in p.ops:
                cur = apply_op(op, cur, b)
            self.assertEqual(cur, p.gt, f"GT mismatch on seed {p.seed}")

    def test_no_negative_or_huge_intermediates(self):
        for p in generate_dataset(200, n_steps=6):
            cur = p.x0
            for op, b in p.ops:
                cur = apply_op(op, cur, b)
                self.assertGreaterEqual(cur, 0, f"negative intermediate seed {p.seed}")
                self.assertLessEqual(cur, 10_000)

    def test_no_zero_operands(self):
        for p in generate_dataset(200, n_steps=5):
            for op, b in p.ops:
                self.assertNotEqual(b, 0)

    def test_text_mentions_start_and_question(self):
        p = generate_problem(seed=1)
        self.assertIn(str(p.x0), p.text)
        self.assertTrue(p.text.strip().endswith("?"))


class TestCanonicalIsClean(unittest.TestCase):
    def test_canonical_executes_to_gt(self):
        for p in generate_dataset(200, n_steps=5):
            steps = p.canonical_cot()
            self.assertEqual(steps[-1].result, p.gt)

    def test_locator_finds_no_error_on_canonical(self):
        for p in generate_dataset(200, n_steps=5):
            self.assertIsNone(
                locate_error(p.canonical_cot(), problem=p),
                f"false positive on canonical CoT, seed {p.seed}",
            )

    def test_locator_clean_without_problem_context(self):
        # arithmetic + carry checks alone must not false-positive on a clean CoT
        for p in generate_dataset(100, n_steps=5):
            self.assertIsNone(locate_error(p.canonical_cot(), x0=p.x0))


class TestRoundTrip(unittest.TestCase):
    """The decisive test: known error in -> same error out."""

    def test_recover_index_and_kind(self):
        n_problems, n_steps = 60, 5
        total = 0
        for p in generate_dataset(n_problems, n_steps=n_steps):
            for kind in ERROR_KINDS:
                for index in range(n_steps):
                    claimed, truth = corrupt_cot(p, index=index, kind=kind, seed=index * 7 + 3)
                    rep = locate_error(claimed, problem=p)
                    self.assertIsNotNone(
                        rep, f"missed {kind} at {index}, seed {p.seed}"
                    )
                    self.assertEqual(
                        rep.index, truth.index,
                        f"wrong index for {kind}: got {rep.index} want {truth.index} seed {p.seed}",
                    )
                    self.assertEqual(
                        rep.kind, truth.kind,
                        f"wrong kind at {index}: got {rep.kind} want {truth.kind} seed {p.seed}",
                    )
                    total += 1
        self.assertGreater(total, 1000)  # sanity: we actually swept a lot

    def test_corruption_actually_changes_answer(self):
        # A "wrong" CoT must reach a final answer != GT, else it isn't a real
        # error case for the experiment (spec §2: errorful answers are the
        # raw material).
        for p in generate_dataset(40, n_steps=5):
            for kind in ERROR_KINDS:
                for index in range(5):
                    claimed, _ = corrupt_cot(p, index=index, kind=kind, seed=index + 1)
                    self.assertNotEqual(
                        claimed[-1].result, p.gt,
                        f"{kind}@{index} did not change the answer, seed {p.seed}",
                    )

    def test_arithmetic_and_carry_caught_without_problem(self):
        # These two kinds are detectable from the CoT alone (no canonical ops).
        for p in generate_dataset(40, n_steps=5):
            for kind in ("arithmetic", "carry"):
                for index in range(5):
                    claimed, truth = corrupt_cot(p, index=index, kind=kind, seed=index)
                    rep = locate_error(claimed, x0=p.x0)
                    self.assertIsNotNone(rep)
                    self.assertEqual(rep.index, truth.index)
                    self.assertEqual(rep.kind, truth.kind)

    def test_semantic_errors_need_problem_context(self):
        # wrong_operand / wrong_operation are arithmetically self-consistent;
        # without the canonical problem the locator cannot (and must not claim
        # to) see them. This documents the locator's known boundary.
        for p in generate_dataset(40, n_steps=5):
            for kind in ("wrong_operand", "wrong_operation"):
                claimed, _ = corrupt_cot(p, index=2, kind=kind, seed=5)
                self.assertIsNone(
                    locate_error(claimed, x0=p.x0),
                    "semantic error wrongly flagged without canonical context",
                )


class TestParser(unittest.TestCase):
    def test_parse_canonical_render(self):
        p = generate_problem(seed=7, n_steps=5)
        steps = p.canonical_cot()
        parsed = parse_cot(p.render_cot(steps))
        self.assertEqual(parsed, steps)

    def test_parse_skips_prose_and_takes_last_equation(self):
        text = (
            "Let me work through it.\n"
            "First, 12 + 5 = 17 items.\n"
            "Then I triple: 17 * 3 = 51.\n"
            "Remove four: 51 - 4 = 47.\n"
            "So the Answer: 47\n"
        )
        steps = parse_cot(text)
        self.assertEqual([s.result for s in steps], [17, 51, 47])
        self.assertEqual(steps[0].op, "add")
        self.assertEqual(steps[1].op, "mul")

    def test_roundtrip_parse_then_locate(self):
        p = generate_problem(seed=11, n_steps=5)
        claimed, truth = corrupt_cot(p, index=3, kind="arithmetic", seed=2)
        text = p.render_cot(claimed)
        rep = locate_error(parse_cot(text), problem=p)
        self.assertEqual(rep.index, truth.index)
        self.assertEqual(rep.kind, truth.kind)


class TestCoLocation(unittest.TestCase):
    def test_same_error_true_when_index_and_signature_match(self):
        p = generate_problem(seed=3, n_steps=5)
        a, _ = corrupt_cot(p, index=2, kind="wrong_operand", seed=1)
        # Build a second solution that is wrong at the SAME step with the SAME
        # claimed operation signature.
        ra = locate_error(a, problem=p)
        b, _ = corrupt_cot(p, index=2, kind="wrong_operand", seed=1)
        rb = locate_error(b, problem=p)
        self.assertTrue(same_error(ra, rb))

    def test_same_error_false_for_different_index(self):
        p = generate_problem(seed=3, n_steps=5)
        ra = locate_error(corrupt_cot(p, index=1, kind="arithmetic", seed=1)[0], problem=p)
        rb = locate_error(corrupt_cot(p, index=3, kind="arithmetic", seed=1)[0], problem=p)
        self.assertFalse(same_error(ra, rb))

    def test_two_correct_solutions_do_not_share_error(self):
        p = generate_problem(seed=3, n_steps=5)
        r = locate_error(p.canonical_cot(), problem=p)
        self.assertFalse(same_error(r, r))  # both None -> not a shared error


class TestValueTraceLocator(unittest.TestCase):
    """The insertion/deletion-tolerant locator used on real model CoTs."""

    def test_correct_path_returns_none(self):
        for p in generate_dataset(100, n_steps=6):
            results = [s.result for s in p.canonical_cot()]
            self.assertIsNone(locate_error_index(results, p), f"seed {p.seed}")

    def test_extra_steps_on_correct_path_still_none(self):
        # Model reaches all canonical checkpoints but interleaves extra values.
        p = generate_problem(seed=4, n_steps=4)
        cvals = canonical_values(p)
        noisy = []
        for v in cvals:
            noisy.append(v + 1000)  # an irrelevant scratch value
            noisy.append(v)         # then the real checkpoint, in order
        self.assertIsNone(locate_error_index(noisy, p))

    def test_recovers_corruption_index_high_accuracy(self):
        # Value-trace is approximate (a corrupted trajectory can coincidentally
        # revisit a canonical checkpoint), so we assert HIGH accuracy, not 100%.
        # Production uses locate_canonical_index, which prefers the EXACT strict
        # locator whenever the CoT is aligned (see test below).
        n_steps, hits, total = 6, 0, 0
        for p in generate_dataset(50, n_steps=n_steps):
            for kind in ERROR_KINDS:
                for index in range(n_steps):
                    claimed, truth = corrupt_cot(p, index=index, kind=kind, seed=index * 3 + 1)
                    got = locate_error_index([s.result for s in claimed], p)
                    hits += got == truth.index
                    total += 1
        acc = hits / total
        self.assertGreater(total, 1000)
        self.assertGreater(acc, 0.97, f"value-trace accuracy only {acc:.3f}")

    def test_hybrid_locator_is_exact_when_aligned(self):
        # corrupt_cot always yields an ALIGNED CoT, so locate_canonical_index
        # must route to the strict locator and be EXACT for every case.
        n_steps = 6
        for p in generate_dataset(50, n_steps=n_steps):
            for kind in ERROR_KINDS:
                for index in range(n_steps):
                    claimed, truth = corrupt_cot(p, index=index, kind=kind, seed=index + 2)
                    text = p.render_cot(claimed)
                    idx, method = locate_canonical_index(text, p)
                    self.assertEqual(method, "strict")
                    self.assertEqual(idx, truth.index, f"{kind}@{index} seed {p.seed}")

    def test_hybrid_clean_cot_returns_none_strict(self):
        for p in generate_dataset(50, n_steps=6):
            idx, method = locate_canonical_index(p.render_cot(p.canonical_cot()), p)
            self.assertEqual(method, "strict")
            self.assertIsNone(idx)

    def test_deletion_flags_dropped_step(self):
        # Drop canonical operation at index 2 -> first unreachable checkpoint is 2.
        p = generate_problem(seed=9, n_steps=5)
        cvals = canonical_values(p)
        results = cvals[:2] + cvals[3:]  # checkpoint v2 never produced
        self.assertEqual(locate_error_index(results, p), 2)

    def test_co_location_same_index(self):
        self.assertTrue(same_error_index(3, 3))
        self.assertFalse(same_error_index(3, 4))
        self.assertFalse(same_error_index(None, 3))
        self.assertFalse(same_error_index(None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
