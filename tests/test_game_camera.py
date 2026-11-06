import unittest

from pursuit_game.game import Game


class FollowCameraTests(unittest.TestCase):
    def make_game(self):
        game = Game.__new__(Game)
        game.canvas_width = 1000
        game.canvas_height = 750
        game.driver = type("Position", (), {"x": 0.0, "y": 0.0})()
        game.runner = type("Position", (), {"x": 0.0, "y": 0.0})()
        return game

    def test_camera_keeps_both_players_visible_after_long_chase(self):
        game = self.make_game()
        game.driver.x, game.driver.y = -18_000.0, 7_500.0
        game.runner.x, game.runner.y = 22_000.0, -12_500.0

        positions = (
            game.world_to_screen(game.driver.x, game.driver.y),
            game.world_to_screen(game.runner.x, game.runner.y),
        )
        for px, py in positions:
            self.assertGreaterEqual(px, 60)
            self.assertLessEqual(px, game.canvas_width - 60)
            self.assertGreaterEqual(py, 60)
            self.assertLessEqual(py, game.canvas_height - 60)

    def test_camera_recenters_on_pair_midpoint(self):
        game = self.make_game()
        game.driver.x, game.driver.y = 1_000.0, 2_000.0
        game.runner.x, game.runner.y = 1_100.0, 2_100.0

        p = game.world_to_screen(game.driver.x, game.driver.y)
        e = game.world_to_screen(game.runner.x, game.runner.y)
        self.assertAlmostEqual((p[0] + e[0]) / 2, game.canvas_width / 2)
        self.assertAlmostEqual((p[1] + e[1]) / 2, game.canvas_height / 2)


if __name__ == "__main__":
    unittest.main()
