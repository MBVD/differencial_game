import math


class ScatteringSurfaceController:
    def __init__(self, w1, w2, R, l, distance_eps=18.0):
        self.w1 = w1
        self.w2 = w2
        self.R = R
        self.l = l
        self.distance_eps = distance_eps

        self.wall_y = self.l * self.w2 / self.w1

    @staticmethod
    def sign(value):
        if value > 0:
            return 1.0

        if value < 0:
            return -1.0

        return 0.0

    @staticmethod
    def clamp(value, left, right):
        return max(left, min(right, value))

    def distance_to_wall(self, x, y):
        return abs(y - self.wall_y)

    def is_near_wall(self, x, y):
        if y < self.wall_y:
            return False

        return self.distance_to_wall(x, y) <= self.distance_eps

    def compute_driver_phi_near_wall(self, x, y):
        if abs(x) < 1e-6:
            return 0.0

        required_phi = self.R * (self.w1 - self.w2) / (self.w1 * x)

        return self.clamp(required_phi, -1.0, 1.0)

    def compute_runner_direction_near_wall(self, x, y):
        if x > 0:
            u_red_x = 1.0
        elif x < 0:
            u_red_x = -1.0
        else:
            u_red_x = 1.0

        u_red_y = 0.0

        return u_red_x, u_red_y