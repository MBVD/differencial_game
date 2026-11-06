"""The vectorized solver must reproduce the reference HJISolver."""

import unittest

from pursuit_game.hji_solver import HJISolver, SolverConfig

try:
    import numpy  # noqa: F401
    from pursuit_game.hji_fast import solve_fast
except ImportError:  # pragma: no cover
    solve_fast = None


@unittest.skipIf(solve_fast is None, "numpy is not installed")
class FastSolverTests(unittest.TestCase):
    def test_matches_reference_solver(self) -> None:
        cfg = SolverConfig(extent=120.0, spacing=20.0, evader_directions=8, max_iterations=60)
        ref = HJISolver(cfg).solve()
        fast = solve_fast(cfg)
        self.assertEqual(ref.iterations, fast.iterations)
        diff = max(abs(a - b) for ra, rb in zip(ref.values, fast.values) for a, b in zip(ra, rb))
        self.assertLess(diff, 1e-9)


class ControlsTests(unittest.TestCase):
    def test_time_step_follows_grid(self) -> None:
        cfg = SolverConfig(spacing=22.0)
        self.assertAlmostEqual(cfg.time_step, 22.0 / (cfg.w1 + cfg.w2))

    def test_driver_turns_towards_evader(self) -> None:
        solver = HJISolver(SolverConfig(extent=120.0, spacing=10.0, max_iterations=300)).solve()
        phi_right, _ = solver.optimal_controls(20.0, 45.0)
        phi_left, _ = solver.optimal_controls(-20.0, 45.0)
        self.assertEqual(phi_right, -phi_left)


if __name__ == "__main__":
    unittest.main()
