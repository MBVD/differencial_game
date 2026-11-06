import math
import unittest

from pursuit_game.wall_game import WallPursuitGame, WallValueGrid


class WallValueGridTests(unittest.TestCase):
    def setUp(self):
        self.grid = WallValueGrid(spacing=5.0)

    def test_fixed_start_is_on_grid_and_value_matches_exact_hji(self):
        x, y = self.grid.start
        self.assertEqual(x % self.grid.spacing, 0)
        self.assertEqual((y + self.grid.y_extent) % self.grid.spacing, 0)
        self.assertLess(self.grid.refinement_error(x, y), 1e-5)
        exact = WallPursuitGame(x=x, y=y).value
        self.assertAlmostEqual(self.grid.value_at(x, y), exact, places=5)

    def test_grid_feedback_reaches_capture_and_respects_wall(self):
        game = WallPursuitGame()
        initial_value = game.value
        for _ in range(2500):
            if game.captured:
                break
            game.step(0.02, self.grid)
            self.assertGreaterEqual(game.x, 0.0)
        self.assertTrue(game.captured)
        self.assertLess(game.elapsed, initial_value + 0.1)

    def test_separate_states_project_to_dispersal_surface(self):
        first = self.grid.saddle_abscissa(300.0, 250.0)
        second = self.grid.saddle_abscissa(240.0, -120.0)
        self.assertGreater(first, self.grid.capture_radius)
        self.assertGreater(second, self.grid.capture_radius)
        self.assertNotAlmostEqual(first, second)
        self.assertTrue(math.isfinite(first + second))

    def test_grid_refinement_reduces_interpolation_error(self):
        point = (303.0, 247.0)
        coarse = WallValueGrid(spacing=20.0).value_at(*point)
        fine = WallValueGrid(spacing=10.0).value_at(*point)
        reference = WallValueGrid(spacing=5.0).value_at(*point)
        self.assertLess(abs(fine - reference), abs(coarse - reference))


if __name__ == "__main__":
    unittest.main()
