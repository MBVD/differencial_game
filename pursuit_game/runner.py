import math


class Runner:
    def __init__(self):
        self.x = 600.0
        self.y = 250.0
        self.psi = 0.0

    def update(self, keys):
        dx = 0
        dy = 0

        if "Up" in keys:
            dy -= 1
        if "Down" in keys:
            dy += 1
        if "Left" in keys:
            dx -= 1
        if "Right" in keys:
            dx += 1

        if dx != 0 or dy != 0:
            # The model angle is measured from the forward (screen-up) axis.
            self.psi = math.atan2(dx, -dy)
