"""Стратегия арены на основе редуцированной функции цены."""

import math
import unittest

from pursuit_game.arena_policy import ArenaPolicyConfig, ReducedArenaPolicy, to_reduced
from pursuit_game.hji_solver import HJISolver, SolverConfig


class ReducedCoordinatesTests(unittest.TestCase):
    def test_evader_straight_ahead_has_positive_y(self) -> None:
        # Экран: ось y вниз, шофёр с курсом theta = 0 едет вверх.
        x, y = to_reduced(400.0, 300.0, 0.0, 400.0, 200.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 100.0)

    def test_evader_to_the_right_has_positive_x(self) -> None:
        x, y = to_reduced(400.0, 300.0, 0.0, 500.0, 300.0)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 0.0)


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        solver = HJISolver(SolverConfig(extent=200.0, spacing=20.0, max_time=20.0,
                                        evader_directions=8, max_iterations=400)).solve()
        cls.policy = ReducedArenaPolicy(solver, ArenaPolicyConfig(directions=12))

    def test_driver_in_corner_keeps_turning(self) -> None:
        # Шофёр в углу (0, 0) смотрит в угол: должен поворачивать, а не стоять.
        theta = 5.3
        phis = []
        for _ in range(5):
            phi, _ = self.policy.optimal_controls(0.0, 0.0, 40.0, 10.0, theta)
            phis.append(phi)
        self.assertNotIn(0.0, phis)
        self.assertEqual(len(set(phis)), 1)

    def test_capture_inside_horizon_is_cheap(self) -> None:
        # Убегающий почти в круге захвата: захват на первом подшаге прогноза.
        table = self.policy.evaluate(400.0, 300.0, 400.0, 285.0, 0.0)
        worst = min(max(row) for row in table.values())
        self.assertAlmostEqual(worst, self.policy.config.substep)

    def test_no_capture_costs_at_least_horizon(self) -> None:
        table = self.policy.evaluate(100.0, 500.0, 700.0, 100.0, 1.0)
        self.assertGreaterEqual(min(min(row) for row in table.values()),
                                self.policy.config.horizon)


if __name__ == "__main__":
    unittest.main()
