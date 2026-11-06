import math
import unittest
from types import SimpleNamespace

from pursuit_game.driver import Driver
from pursuit_game.game import Game
from pursuit_game.runner import Runner
from pursuit_game.wall_game import WallValueGrid


class ArenaBoundaryTests(unittest.TestCase):
    def test_manual_control_switches_between_evader_and_pursuer(self):
        game = Game.__new__(Game)
        game.control_mode = "runner"

        game.switch_control()
        self.assertEqual(game.control_mode, "driver")
        game.switch_control()
        self.assertEqual(game.control_mode, "runner")

        game.keys = set()
        game.on_key_press(SimpleNamespace(keysym="Tab"))
        self.assertEqual(game.control_mode, "driver")

    def test_both_players_are_clamped_to_all_four_walls(self):
        game = Game.__new__(Game)
        game.arena_mode = True
        game.driver = Driver()
        game.runner = Runner()
        game.driver.x, game.driver.y = -20, 640
        game.runner.x, game.runner.y = 830, -10

        game.constrain_players_to_arena()

        self.assertEqual((game.driver.x, game.driver.y), (0.0, Game.ARENA_HEIGHT))
        self.assertEqual((game.runner.x, game.runner.y), (Game.ARENA_WIDTH, 0.0))

    def test_wall_mode_is_a_distinct_model_and_returns_to_four_wall_arena(self):
        game = Game.__new__(Game)
        game.w1, game.w2, game.capture_radius = 120.0, 100.0, 35.0
        game.mode_3d = False
        game.wall_mode = False
        game.arena_mode = True
        game.captured = True
        game.status_message = ""
        game.wall_grid = WallValueGrid(spacing=100.0)
        game.enable_wall_game()
        self.assertTrue(game.wall_mode)
        self.assertFalse(game.arena_mode)
        self.assertEqual(game.control_mode, "wall-auto")
        self.assertEqual(game.wall_game.x, 300.0)
        self.assertEqual(game.wall_grid.start, (300, 250))
        game.enable_2d()
        self.assertFalse(game.wall_mode)
        self.assertTrue(game.arena_mode)

    def test_auto_hji_does_not_enable_on_a_provisional_grid(self):
        game = Game.__new__(Game)
        game.mode_3d = False
        game.wall_mode = False
        game.control_mode = "runner"
        game.status_message = ""
        game.arena_hji = type("Grid", (), {"converged": False})()

        game.toggle_auto()

        self.assertEqual(game.control_mode, "runner")
        self.assertIn("не сошлась", game.status_message)

    def test_arena_grid_stays_fixed_when_players_move(self):
        class Canvas:
            def __init__(self):
                self.lines = []

            def create_line(self, *coords, **kwargs):
                self.lines.append(coords)

        game = Game.__new__(Game)
        game.arena_mode = True
        game.canvas_width, game.canvas_height = 1000, 750
        game.driver, game.runner = Driver(), Runner()
        game.canvas = Canvas()

        game.draw_grid()
        initial_grid = list(game.canvas.lines)
        game.canvas.lines.clear()
        game.driver.x, game.driver.y = 120, 430
        game.runner.x, game.runner.y = 690, 90
        game.draw_grid()

        self.assertEqual(game.canvas.lines, initial_grid)

    def test_full_scene_shows_motion_only_not_phi_psi_controls(self):
        class Canvas:
            def __init__(self):
                self.texts = []
                self.lines = []

            def create_oval(self, *coords, **kwargs):
                pass

            def create_line(self, *coords, **kwargs):
                self.lines.append((coords, kwargs))

            def create_text(self, *coords, **kwargs):
                self.texts.append(kwargs["text"])

        game = Game.__new__(Game)
        game.arena_mode = True
        game.canvas_width, game.canvas_height = 1000, 750
        game.driver, game.runner = Driver(), Runner()
        game.driver.phi = -1.0
        game.runner.psi = 0.5
        game.w1, game.R = 120.0, 120.0
        game.control_mode = "auto"
        game._driver_motion = (1.0, 0.0)
        game._runner_motion = (0.0, 1.0)
        game.canvas = Canvas()

        game.draw_driver()
        game.draw_runner()

        self.assertFalse(any("φ" in text or "ψ" in text for text in game.canvas.texts))
        self.assertTrue(any(kwargs.get("fill") == "navy" for _, kwargs in game.canvas.lines))
        self.assertTrue(any(kwargs.get("fill") == "darkred"
                            for _, kwargs in game.canvas.lines))

    def test_phi_and_psi_contribution_vectors_appear_only_in_reduced_state(self):
        class Canvas:
            def __init__(self):
                self.lines = []
                self.texts = []

            def cget(self, key):
                return {"width": "430", "height": "300"}[key]

            def delete(self, *args):
                pass

            def create_rectangle(self, *args, **kwargs):
                pass

            def create_oval(self, *args, **kwargs):
                pass

            def create_line(self, *args, **kwargs):
                self.lines.append((args, kwargs))

            def create_text(self, *args, **kwargs):
                self.texts.append(kwargs.get("text", ""))

        game = Game.__new__(Game)
        game.wall_mode = False
        game.x, game.y = 0.0, 10.0
        game.capture_radius = 35.0
        game.driver, game.runner = Driver(), Runner()
        game.driver.theta, game.driver.phi = 0.0, 1.0
        game.runner.psi = math.pi / 2
        game.w1, game.w2, game.R = 120.0, 100.0, 120.0
        game.reduced_canvas = Canvas()

        game.draw_reduced_state()

        self.assertIn("φ", game.reduced_canvas.texts)
        self.assertIn("ψ", game.reduced_canvas.texts)
        colors = [kwargs.get("fill") for _, kwargs in game.reduced_canvas.lines]
        self.assertIn("#d97706", colors)
        self.assertIn("#7c3aed", colors)

    def test_car_motion_vector_uses_actual_displacement_after_wall_clamp(self):
        game = Game.__new__(Game)
        game.captured = False
        game.wall_mode = False
        game.mode_3d = False
        game.arena_mode = True
        game.control_mode = "runner"
        game.keys = set()
        game.dt, game.w1, game.w2, game.R = 0.02, 120.0, 100.0, 120.0
        game.capture_radius = 1.0
        game.driver, game.runner = Driver(), Runner()
        game.driver.x, game.driver.y = 799.0, 300.0
        game.driver.theta = math.pi / 2  # faces right, into the right wall
        game.runner.x, game.runner.y = 400.0, 100.0
        game.runner.psi = 0.0
        game._record_trails = lambda: None

        game.update_absolute_motion()

        self.assertEqual(game.driver.x, Game.ARENA_WIDTH)
        self.assertAlmostEqual(game._driver_motion[0], 1.0)
        self.assertAlmostEqual(game._driver_motion[1], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
