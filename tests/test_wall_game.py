import unittest

from pursuit_game.wall_game import WallPursuitGame


class WallGameTests(unittest.TestCase):
    def test_target_and_wall_state_constraint(self):
        game = WallPursuitGame(x=0.0, y=10.0)
        self.assertEqual(game.value, 0.0)
        game.x, game.y = 1.0, 0.0
        self.assertLess(game.value, 1e-12)

    def test_exact_value_satisfies_stationary_hji_on_both_dispersal_branches(self):
        for x in (0.0, 10.0, 100.0, 300.0):
            for y in (-250.0, -80.0, 0.0, 80.0, 250.0):
                game = WallPursuitGame(x=x, y=y)
                if game.separation > game.capture_radius:
                    self.assertLess(abs(game.hamiltonian_residual()), 1e-10,
                                    (x, y, game.hamiltonian_residual()))

    def test_feedback_reaches_capture_without_crossing_wall(self):
        game = WallPursuitGame()
        initial_value = game.value
        for _ in range(2500):
            if game.captured:
                break
            game.step(0.02)
            self.assertGreaterEqual(game.x, 0.0)
        self.assertTrue(game.captured)
        self.assertLess(game.elapsed, initial_value + 0.1)
        self.assertGreater(game.evader_wall_y, -10_000.0)


if __name__ == "__main__":
    unittest.main()
