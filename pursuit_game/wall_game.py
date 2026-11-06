"""Exact reduced wall-pursuit game from Isaacs' half-plane example.

Coordinates are the evader-relative position of the pursuer. The wall is x=0,
the pursuer stays in x>=0, and capture occurs inside x**2+y**2<=l**2.
The evader moves along the wall at speed w2 and the pursuer has simple motion
at speed w1>w2. The closed-form time value is the positive branch of Isaacs'
retrograde-path equation; it avoids the capped-grid false plateau of the old
finite-horizon HJI table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from array import array


@dataclass
class WallPursuitGame:
    w1: float = 120.0
    w2: float = 100.0
    capture_radius: float = 35.0
    x: float = 300.0
    y: float = 250.0
    evader_wall_y: float = 0.0
    pursuer_heading: float = math.pi / 4.0
    elapsed: float = 0.0
    captured: bool = False

    def __post_init__(self) -> None:
        if self.w1 <= self.w2 or self.w2 <= 0:
            raise ValueError("For the wall solution, pursuer speed must exceed evader speed")
        if self.capture_radius <= 0 or self.x < 0:
            raise ValueError("Use a positive capture radius and a state in the half-plane x>=0")

    @property
    def separation(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def value(self) -> float:
        """Exact remaining capture time in seconds, zero on the target disk."""
        if self.separation <= self.capture_radius:
            return 0.0
        w1, w2, radius = self.w1, self.w2, self.capture_radius
        a = abs(self.y)
        denom = w1 * w1 - w2 * w2
        radicand = (w1 * a - radius * w2) ** 2 + denom * self.x * self.x
        numerator = -(radius * w1 - a * w2) + math.sqrt(max(0.0, radicand))
        return max(0.0, numerator / denom)

    def gradient(self) -> tuple[float, float]:
        """Analytic value gradient; at y=0 choose the upper dispersal branch."""
        if self.separation <= self.capture_radius:
            return 0.0, 0.0
        w1, w2, radius = self.w1, self.w2, self.capture_radius
        a = abs(self.y)
        denom = w1 * w1 - w2 * w2
        discriminant = math.sqrt((w1 * a - radius * w2) ** 2 + denom * self.x * self.x)
        if discriminant <= 1e-14:
            return 0.0, 0.0
        vx = self.x / discriminant
        vy_abs = (w2 + w1 * (w1 * a - radius * w2) / discriminant) / denom
        sign = -1.0 if self.y < 0 else 1.0
        return vx, sign * vy_abs

    def optimal_controls(self, value_grid: "WallValueGrid | None" = None) -> tuple[float, int]:
        """Return pursuer heading and evader's maximizing wall direction."""
        vx, vy = (value_grid.gradient_at(self.x, self.y)
                  if value_grid is not None else self.gradient())
        if math.hypot(vx, vy) <= 1e-14:
            return self.pursuer_heading, 1 if self.y >= 0 else -1
        heading = math.atan2(vy, vx)
        evader_direction = 1 if vy >= 0 else -1
        return heading, evader_direction

    def saddle_abscissa(self) -> float:
        """Return the dispersal point on y=0 associated with this characteristic.

        The two Isaacs branches intersect on y=0.  This is a projection onto
        the dispersal surface, not a capture target and not an attracting
        equilibrium of the game.
        """
        if self.separation <= self.capture_radius:
            return self.capture_radius
        w1, w2, radius = self.w1, self.w2, self.capture_radius
        branch = 1.0 if self.y >= 0.0 else -1.0
        # Invert x=(l+w1*t)cos(s), y=(l+w1*t)sin(s)-branch*w2*t.
        a = w1 * w1 - w2 * w2
        b = 2.0 * (radius * w1 - branch * self.y * w2)
        c = radius * radius - self.x * self.x - self.y * self.y
        tau = (-b + math.sqrt(max(0.0, b * b - 4.0 * a * c))) / (2.0 * a)
        return math.sqrt(max(0.0, (radius + w1 * tau) ** 2 - (w2 * tau) ** 2))

    def hamiltonian_residual(self) -> float:
        """Exact stationary HJI residual off the target (should be roundoff)."""
        if self.separation <= self.capture_radius:
            return 0.0
        vx, vy = self.gradient()
        return 1.0 - self.w1 * math.hypot(vx, vy) + self.w2 * abs(vy)

    def step(self, dt: float, value_grid: "WallValueGrid | None" = None) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.captured:
            return
        heading, evader_direction = self.optimal_controls(value_grid)
        self.pursuer_heading = heading

        x_next = self.x - self.w1 * math.cos(heading) * dt
        y_next = self.y + (self.w2 * evader_direction - self.w1 * math.sin(heading)) * dt
        # The wall is a state constraint, not a screen edge: P cannot cross it.
        self.x = max(0.0, x_next)
        self.evader_wall_y -= self.w2 * evader_direction * dt
        self.y = y_next
        self.elapsed += dt
        self.captured = self.separation <= self.capture_radius

    @property
    def pursuer_world_position(self) -> tuple[float, float]:
        return self.x, self.evader_wall_y + self.y

    @property
    def evader_world_position(self) -> tuple[float, float]:
        return 0.0, self.evader_wall_y


class WallValueGrid:
    """Fixed-coordinate sampled HJI value grid for the one-wall model.

    Values are initialized from the exact Isaacs solution, so this local grid
    has a non-capped reference at the chosen initial point and can be checked
    under refinement instead of falsely treating a finite-horizon plateau as
    convergence. Runtime feedback uses bilinear values and grid gradients.
    """

    def __init__(self, w1: float = 120.0, w2: float = 100.0,
                 capture_radius: float = 35.0, start: tuple[float, float] = (300, 250),
                 x_max: float = 800.0, y_extent: float = 800.0,
                 spacing: float = 10.0) -> None:
        if w1 <= w2 or w2 <= 0 or capture_radius <= 0 or spacing <= 0:
            raise ValueError("Grid requires w1>w2>0 and positive radius/spacing")
        if x_max <= start[0] or y_extent <= abs(start[1]):
            raise ValueError("Grid bounds must contain the fixed initial state")
        self.w1, self.w2, self.capture_radius = w1, w2, capture_radius
        self.start = start
        self.spacing = spacing
        self.x_max = x_max
        self.y_extent = y_extent
        self.nx = round(x_max / spacing) + 1
        self.ny = round(2 * y_extent / spacing) + 1
        self.values = array("f")
        for j in range(self.ny):
            y = -y_extent + j * spacing
            for i in range(self.nx):
                x = i * spacing
                self.values.append(self._exact_value(x, y))

    def _exact_value(self, x: float, y: float) -> float:
        radius = self.capture_radius
        if math.hypot(x, y) <= radius:
            return 0.0
        a = abs(y)
        denom = self.w1 * self.w1 - self.w2 * self.w2
        radicand = (self.w1 * a - radius * self.w2) ** 2 + denom * x * x
        numerator = -(radius * self.w1 - a * self.w2) + math.sqrt(max(0.0, radicand))
        return max(0.0, numerator / denom)

    def value_at(self, x: float, y: float) -> float:
        if x < 0 or x > self.x_max or abs(y) > self.y_extent:
            return self._exact_value(x, y)
        gx, gy = x / self.spacing, (y + self.y_extent) / self.spacing
        i, j = min(int(gx), self.nx - 2), min(int(gy), self.ny - 2)
        ax, ay = gx - i, gy - j
        v00 = self.values[j * self.nx + i]
        v10 = self.values[j * self.nx + i + 1]
        v01 = self.values[(j + 1) * self.nx + i]
        v11 = self.values[(j + 1) * self.nx + i + 1]
        return ((1 - ay) * ((1 - ax) * v00 + ax * v10)
                + ay * ((1 - ax) * v01 + ax * v11))

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        h = self.spacing
        # The value has a cusp across the dispersal surface. Select the upper
        # characteristic there, as Isaacs' construction does in the reference.
        if abs(y) < 0.5 * h:
            vx, vy = WallPursuitGame(self.w1, self.w2, self.capture_radius,
                                     x, h).gradient()
            return vx, vy
        return ((self.value_at(x + h, y) - self.value_at(x - h, y)) / (2 * h),
                (self.value_at(x, y + h) - self.value_at(x, y - h)) / (2 * h))

    def saddle_abscissa(self, x: float, y: float) -> float:
        state = WallPursuitGame(self.w1, self.w2, self.capture_radius, x, y)
        return state.saddle_abscissa()

    def refinement_error(self, x: float, y: float) -> float:
        return abs(self.value_at(x, y) - self._exact_value(x, y))
