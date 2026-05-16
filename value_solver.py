import math


class ValueFunctionSolver:
    def __init__(
        self,
        w1=120.0,
        w2=100.0,
        R=120.0,
        l=35.0,
        grid_half_x=320.0,
        grid_half_y=320.0,
        grid_step=10.0,
        iterations=60,
        tau=0.08,
        large_value=200.0
    ):
        self.w1 = w1
        self.w2 = w2
        self.R = R
        self.l = l

        self.grid_half_x = grid_half_x
        self.grid_half_y = grid_half_y
        self.grid_step = grid_step

        self.grid_nx = int(2 * self.grid_half_x / self.grid_step) + 1
        self.grid_ny = int(2 * self.grid_half_y / self.grid_step) + 1

        self.x_values = [
            -self.grid_half_x + i * self.grid_step
            for i in range(self.grid_nx)
        ]

        self.y_values = [
            -self.grid_half_y + j * self.grid_step
            for j in range(self.grid_ny)
        ]

        self.value_grid = [
            [0.0 for _ in range(self.grid_nx)]
            for _ in range(self.grid_ny)
        ]

        self.phi_candidates = [-1.0, 0.0, 1.0]

        self.psi_candidates = [
            2.0 * math.pi * k / 24.0
            for k in range(24)
        ]

        self.iterations = iterations
        self.tau = tau
        self.large_value = large_value

    def is_terminal(self, x, y):
        return math.hypot(x, y) <= self.l

    def reduced_dynamics(self, x, y, phi, psi):
        dx = -(self.w1 / self.R) * y * phi + self.w2 * math.sin(psi)
        dy = (self.w1 / self.R) * x * phi - self.w1 + self.w2 * math.cos(psi)

        return dx, dy

    def interp_value(self, x, y, table=None):
        if table is None:
            table = self.value_grid

        if x <= -self.grid_half_x:
            x = -self.grid_half_x

        if x >= self.grid_half_x:
            x = self.grid_half_x - 1e-9

        if y <= -self.grid_half_y:
            y = -self.grid_half_y

        if y >= self.grid_half_y:
            y = self.grid_half_y - 1e-9

        fx = (x + self.grid_half_x) / self.grid_step
        fy = (y + self.grid_half_y) / self.grid_step

        i = int(math.floor(fx))
        j = int(math.floor(fy))

        if i < 0:
            i = 0

        if i > self.grid_nx - 2:
            i = self.grid_nx - 2

        if j < 0:
            j = 0

        if j > self.grid_ny - 2:
            j = self.grid_ny - 2

        tx = fx - i
        ty = fy - j

        v00 = table[j][i]
        v10 = table[j][i + 1]
        v01 = table[j + 1][i]
        v11 = table[j + 1][i + 1]

        value = (
            (1.0 - tx) * (1.0 - ty) * v00
            + tx * (1.0 - ty) * v10
            + (1.0 - tx) * ty * v01
            + tx * ty * v11
        )

        return value

    def compute_value_grid(self, progress_callback=None):
        for j, y in enumerate(self.y_values):
            for i, x in enumerate(self.x_values):
                if self.is_terminal(x, y):
                    self.value_grid[j][i] = 0.0
                else:
                    self.value_grid[j][i] = self.large_value

        for it in range(self.iterations):
            new_grid = [
                row[:]
                for row in self.value_grid
            ]

            for j, y in enumerate(self.y_values):
                for i, x in enumerate(self.x_values):
                    if self.is_terminal(x, y):
                        new_grid[j][i] = 0.0
                        continue

                    best_for_p = float("inf")

                    for phi in self.phi_candidates:
                        worst_for_e = -float("inf")

                        for psi in self.psi_candidates:
                            dx, dy = self.reduced_dynamics(x, y, phi, psi)

                            xn = x + self.tau * dx
                            yn = y + self.tau * dy

                            future_cost = self.interp_value(
                                xn,
                                yn,
                                self.value_grid
                            )

                            total_cost = self.tau + future_cost

                            if total_cost > worst_for_e:
                                worst_for_e = total_cost

                        if worst_for_e < best_for_p:
                            best_for_p = worst_for_e

                    new_grid[j][i] = best_for_p

            self.value_grid = new_grid

            if progress_callback is not None:
                progress_callback(it + 1, self.iterations)

        return self.value_grid

    def estimate_gradient(self, x, y):
        h = self.grid_step

        vx_plus = self.interp_value(x + h, y)
        vx_minus = self.interp_value(x - h, y)

        vy_plus = self.interp_value(x, y + h)
        vy_minus = self.interp_value(x, y - h)

        vx = (vx_plus - vx_minus) / (2.0 * h)
        vy = (vy_plus - vy_minus) / (2.0 * h)

        return vx, vy