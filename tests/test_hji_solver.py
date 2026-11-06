"""Regression tests for the target-set handling in the HJI solver."""

import unittest

from pursuit_game.hji_solver import HJISolver, SolverConfig


class CaptureSegmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = HJISolver(SolverConfig(
            capture_radius=1.0,
            extent=10.0,
            spacing=2.0,
            time_step=0.5,
            max_time=2.0,
            max_iterations=1,
            evader_directions=4,
        ))

    def test_segment_stopping_at_first_capture(self) -> None:
        self.assertAlmostEqual(self.solver._capture_fraction(2.0, 0.0, 0.0, 0.0), 0.5)

    def test_tangent_segment_counts_as_capture(self) -> None:
        self.assertAlmostEqual(self.solver._capture_fraction(-2.0, 1.0, 2.0, 1.0), 0.5)

    def test_segment_missing_capture_disk(self) -> None:
        self.assertIsNone(self.solver._capture_fraction(2.0, 2.0, 2.0, -2.0))

    def test_capped_value_is_not_a_valid_hji_policy(self) -> None:
        self.assertTrue(self.solver.has_finite_value(0.0, 0.0))
        self.assertFalse(self.solver.has_finite_value(8.0, 8.0))


if __name__ == "__main__":
    unittest.main()
