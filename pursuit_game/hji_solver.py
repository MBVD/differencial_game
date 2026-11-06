"""Finite-difference dynamic-programming solver for the planar HJI game.

Only the Python standard library is required. Run ``python -m pursuit_game.hji_solver`` to
write data/hji_solution.json, or add ``--validate`` to compare two
grid resolutions and the numerical winning boundary with Isaacs' barrier.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SolverConfig:
    w1: float = 120.0
    w2: float = 100.0
    # R = 60: l/R ~ 0.58 > sqrt(1-g^2)+arcsin(g)-1 ~ 0.54, захват гарантирован везде
    turn_radius: float = 60.0
    capture_radius: float = 35.0
    extent: float = 400.0
    spacing: float = 20.0
    # None -> шаг по времени согласуется с шагом сетки: tau = h / (w1 + w2),
    # т.е. около круга захвата точка смещается за шаг примерно на одну ячейку.
    # Для схемы с линейной интерполяцией локальная погрешность ~ tau + h^2/tau,
    # поэтому tau ~ h даёт O(h); при фиксированном tau измельчение сетки
    # не уменьшает ошибку.
    time_step: float | None = None
    max_time: float = 60.0
    max_iterations: int = 2000
    tolerance: float = 1e-4
    evader_directions: int = 24

    def __post_init__(self) -> None:
        if self.time_step is None:
            object.__setattr__(self, "time_step", self.spacing / (self.w1 + self.w2))


class HJISolver:
    """Solve min_phi max_psi [dt + V(x + dt f)] on a uniform 2-D grid."""

    def __init__(self, config: SolverConfig = SolverConfig()) -> None:
        if config.w1 <= 0 or config.w2 < 0 or config.turn_radius <= 0:
            raise ValueError("Speeds and turning radius must be physically valid")
        if config.capture_radius <= 0 or config.spacing <= 0 or config.time_step <= 0:
            raise ValueError("Capture radius, spacing and time step must be positive")
        if config.extent <= config.capture_radius:
            raise ValueError("Grid extent must exceed the capture radius")
        if config.evader_directions < 4:
            raise ValueError("Use at least four evader directions")

        self.config = config
        intervals = round(2.0 * config.extent / config.spacing)
        self.spacing = 2.0 * config.extent / intervals
        self.xs = [-config.extent + i * self.spacing for i in range(intervals + 1)]
        self.ys = list(self.xs)
        self.n = len(self.xs)
        self.cap = config.max_time
        self.phi_values = (-1.0, 0.0, 1.0)
        self.psi_values = tuple(
            2.0 * math.pi * k / config.evader_directions
            for k in range(config.evader_directions)
        )
        self.values = [
            [0.0 if x * x + y * y <= config.capture_radius**2 else self.cap
             for x in self.xs]
            for y in self.ys
        ]
        self.iterations = 0
        self.residual = math.inf
        self.converged = False

    def _interpolate(self, x: float, y: float, values: Sequence[Sequence[float]]) -> float:
        cfg = self.config
        if x * x + y * y <= cfg.capture_radius**2:
            return 0.0
        gx = (x + cfg.extent) / self.spacing
        gy = (y + cfg.extent) / self.spacing
        if gx < 0.0 or gy < 0.0 or gx > self.n - 1 or gy > self.n - 1:
            return self.cap
        i0 = min(int(gx), self.n - 2)
        j0 = min(int(gy), self.n - 2)
        ax = gx - i0
        ay = gy - j0
        v00 = values[j0][i0]
        v10 = values[j0][i0 + 1]
        v01 = values[j0 + 1][i0]
        v11 = values[j0 + 1][i0 + 1]
        return ((1.0 - ay) * ((1.0 - ax) * v00 + ax * v10)
                + ay * ((1.0 - ax) * v01 + ax * v11))

    def _capture_fraction(self, x0: float, y0: float, x1: float, y1: float) -> float | None:
        """Return the first segment fraction that enters the capture disk."""
        radius = self.config.capture_radius
        dx, dy = x1 - x0, y1 - y0
        a = dx * dx + dy * dy
        c = x0 * x0 + y0 * y0 - radius * radius
        if c <= 0.0:
            return 0.0
        if a == 0.0:
            return None
        b = 2.0 * (x0 * dx + y0 * dy)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        root = math.sqrt(discriminant)
        fractions = sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        return next((t for t in fractions if 0.0 <= t <= 1.0), None)

    def solve(self) -> "HJISolver":
        cfg = self.config
        for iteration in range(1, cfg.max_iterations + 1):
            old = self.values
            new = [row.copy() for row in old]
            max_change = 0.0
            for j, y in enumerate(self.ys):
                for i, x in enumerate(self.xs):
                    if x * x + y * y <= cfg.capture_radius**2:
                        new[j][i] = 0.0
                        continue
                    best_driver = self.cap
                    for phi in self.phi_values:
                        turn = cfg.w1 / cfg.turn_radius * phi
                        worst_evader = 0.0
                        for psi in self.psi_values:
                            vx = -turn * y + cfg.w2 * math.sin(psi)
                            vy = turn * x - cfg.w1 + cfg.w2 * math.cos(psi)
                            x_next = x + cfg.time_step * vx
                            y_next = y + cfg.time_step * vy
                            capture_fraction = self._capture_fraction(x, y, x_next, y_next)
                            if capture_fraction is not None:
                                # Нашел ошибку ... прежняя схема могла перескочить круг захвата
                                # и продолжить игру, если конец шага оказывался за кругом.
                                candidate = cfg.time_step * capture_fraction
                            else:
                                candidate = cfg.time_step + self._interpolate(x_next, y_next, old)
                            if candidate > worst_evader:
                                worst_evader = candidate
                        if worst_evader < best_driver:
                            best_driver = worst_evader
                    value = min(self.cap, best_driver)
                    new[j][i] = value
                    change = abs(value - old[j][i])
                    if change > max_change:
                        max_change = change
            self.values = new
            self.iterations = iteration
            self.residual = max_change
            if max_change < cfg.tolerance:
                self.converged = True
                break
        return self

    def value_at(self, x: float, y: float) -> float:
        return self._interpolate(x, y, self.values)

    def has_finite_value(self, x: float, y: float) -> bool:
        """Whether the state is below the capped/unresolved value region."""
        return self.value_at(x, y) < self.cap - self.config.time_step

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        h = self.spacing
        vx = (self.value_at(x + h, y) - self.value_at(x - h, y)) / (2.0 * h)
        vy = (self.value_at(x, y + h) - self.value_at(x, y - h)) / (2.0 * h)
        return vx, vy

    def optimal_controls(self, x: float, y: float, directions: int = 72) -> tuple[float, float]:
        """Return (driver curvature control phi, evader heading psi).

        Управления берутся из того же дискретного min-max, по которому
        построена сеточная функция цены: шофёр выбирает phi, минимизирующее
        худший для него исход, убегающий --- psi, максимизирующее исход при
        этом phi. Градиентная формула (см. gradient_at) неприменима на
        сингулярных линиях и барьере, где V негладкая.
        """
        cfg = self.config
        psis = [2.0 * math.pi * k / directions for k in range(directions)]
        best_phi, best_psi, best_value = 0.0, 0.0, math.inf
        for phi in self.phi_values:
            turn = cfg.w1 / cfg.turn_radius * phi
            worst_value, worst_psi = -math.inf, 0.0
            for psi in psis:
                vx = -turn * y + cfg.w2 * math.sin(psi)
                vy = turn * x - cfg.w1 + cfg.w2 * math.cos(psi)
                x_next = x + cfg.time_step * vx
                y_next = y + cfg.time_step * vy
                fraction = self._capture_fraction(x, y, x_next, y_next)
                if fraction is not None:
                    value = cfg.time_step * fraction
                else:
                    value = cfg.time_step + self._interpolate(x_next, y_next, self.values)
                if value > worst_value + 1e-12:
                    worst_value, worst_psi = value, psi
            if worst_value < best_value - 1e-12:
                best_phi, best_psi, best_value = phi, worst_psi, worst_value
        return best_phi, best_psi

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "spacing": self.spacing,
            "x": self.xs,
            "y": self.ys,
            "values": self.values,
            "iterations": self.iterations,
            "residual": self.residual,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HJISolver":
        obj = cls(SolverConfig(**data["config"]))
        obj.spacing = float(data["spacing"])
        obj.xs = list(data["x"])
        obj.ys = list(data["y"])
        obj.n = len(obj.xs)
        obj.values = data["values"]
        obj.iterations = int(data.get("iterations", 0))
        obj.residual = float(data.get("residual", math.inf))
        obj.converged = bool(data.get("converged", False))
        return obj


def isaacs_capture_angles(w1: float, w2: float) -> tuple[float, float] | None:
    """Return the local terminal-boundary angles +/- alpha, if capturable."""
    if w1 <= w2:
        return None
    alpha = math.acos(w2 / w1)
    return -alpha, alpha


def terminal_boundary_check(config: SolverConfig, samples: int = 101) -> dict:
    """Check Isaacs' exact inward/outward normal-speed condition on the target."""
    max_error = 0.0
    mismatches = 0
    for k in range(samples):
        angle = -math.pi + 2.0 * math.pi * k / (samples - 1)
        x = config.capture_radius * math.sin(angle)
        y = config.capture_radius * math.cos(angle)
        nx, ny = math.sin(angle), math.cos(angle)
        # Curvature control cancels in the normal projection on the circle.
        observed = max(
            nx * config.w2 * math.sin(psi)
            + ny * (-config.w1 + config.w2 * math.cos(psi))
            for psi in (2.0 * math.pi * j / 720 for j in range(720))
        )
        exact = config.w2 - config.w1 * math.cos(angle)
        max_error = max(max_error, abs(observed - exact))
        capture_is_forced_locally = exact < 0.0
        analytic = config.w1 > config.w2 and math.cos(angle) > config.w2 / config.w1
        if capture_is_forced_locally != analytic:
            mismatches += 1
    return {"samples": samples, "max_normal_speed_error": max_error,
            "classification_mismatches": mismatches,
            "capture_angle_range_rad": isaacs_capture_angles(config.w1, config.w2)}


def isaacs_barrier_point(
    config: SolverConfig, angle: float, backward_time: float
) -> tuple[float, float]:
    """Isaacs' involute branch, with s measured from the positive y axis."""
    direction = 1.0 if angle >= 0.0 else -1.0
    ct = config.w1 / config.turn_radius * backward_time
    signed_ct = direction * ct
    distance = config.capture_radius - config.w2 * backward_time
    x = distance * math.sin(angle + signed_ct) + direction * config.turn_radius * (1.0 - math.cos(ct))
    y = distance * math.cos(angle + signed_ct) + config.turn_radius * math.sin(ct)
    return x, y


def boundary_points(solver: HJISolver) -> list[tuple[float, float]]:
    """Extract grid edges between finite-time capture and the capped region."""
    cfg = solver.config
    threshold = cfg.max_time - max(2.0 * cfg.time_step, 1e-8)
    pts = []
    for j, y in enumerate(solver.ys):
        for i, x in enumerate(solver.xs):
            v = solver.values[j][i]
            if v >= threshold or x * x + y * y <= cfg.capture_radius**2:
                continue
            neighbors = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            if any(solver.values[jj][ii] >= threshold
                   for ii, jj in neighbors if 0 <= ii < solver.n and 0 <= jj < solver.n):
                pts.append((x, y))
    return pts


def barrier_comparison(solver: HJISolver) -> dict:
    """Compare the numerical finite-time frontier to Isaacs' barrier branches."""
    angles = isaacs_capture_angles(solver.config.w1, solver.config.w2)
    frontier = boundary_points(solver)
    if not angles or not frontier:
        return {"available": False, "reason": "No capturable boundary/frontier in this setup"}

    # Compare only the first backward-time turn, before the involute loops.
    t_end = min(0.8, 0.7 * solver.config.capture_radius / solver.config.w2)
    analytic = []
    sample_count = 80
    for angle in angles:
        for k in range(1, sample_count + 1):
            t = t_end * k / sample_count
            p = isaacs_barrier_point(solver.config, angle, t)
            if abs(p[0]) < solver.config.extent and abs(p[1]) < solver.config.extent:
                analytic.append(p)
    if not analytic:
        return {"available": False, "reason": "Analytical branch lies outside the grid"}
    distances = [min(math.dist(p, q) for q in frontier) for p in analytic]
    return {
        "available": True,
        "samples": len(distances),
        "mean_distance": sum(distances) / len(distances),
        "max_distance": max(distances),
        "grid_spacing": solver.spacing,
        "interpretation": "Distance from Isaacs' short barrier arcs to the numerical capped-value frontier; reduce as the grid is refined.",
    }


def solve_and_save(path: Path, config: SolverConfig) -> HJISolver:
    solver = HJISolver(config).solve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(solver.to_dict()), encoding="utf-8")
    return solver


def run_validation() -> dict:
    # Nested grids make a pointwise refinement comparison possible.
    common = dict(extent=120.0, max_time=8.0, max_iterations=2000,
                  evader_directions=16, tolerance=1e-4)
    coarse = HJISolver(SolverConfig(spacing=16.0, **common)).solve()
    fine = HJISolver(SolverConfig(spacing=8.0, **common)).solve()
    errors = []
    for j, y in enumerate(coarse.ys):
        for i, x in enumerate(coarse.xs):
            a = coarse.values[j][i]
            b = fine.value_at(x, y)
            if a < common["max_time"] - 0.1 and b < common["max_time"] - 0.1:
                errors.append(abs(a - b))
    return {
        "coarse": {"spacing": coarse.spacing, "iterations": coarse.iterations,
                   "residual": coarse.residual, "converged": coarse.converged},
        "fine": {"spacing": fine.spacing, "iterations": fine.iterations,
                 "residual": fine.residual, "converged": fine.converged},
        "refinement": {"common_finite_points": len(errors),
                       "mean_absolute_difference": sum(errors) / len(errors) if errors else None,
                       "max_absolute_difference": max(errors) if errors else None},
        "isaacs_barrier": barrier_comparison(fine),
        "coarse_isaacs_barrier": barrier_comparison(coarse),
        "terminal_capture_condition": terminal_boundary_check(fine.config),
        "local_capture_angles_rad": isaacs_capture_angles(fine.config.w1, fine.config.w2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="compare grid refinement and Isaacs barrier")
    default_output = Path(__file__).resolve().parents[1] / "data" / "hji_solution.json"
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--spacing", type=float, default=SolverConfig.spacing)
    parser.add_argument("--directions", type=int, default=SolverConfig.evader_directions)
    parser.add_argument("--max-time", type=float, default=SolverConfig.max_time)
    parser.add_argument("--extent", type=float, default=SolverConfig.extent)
    parser.add_argument("--time-step", type=float, default=None,
                        help="шаг по времени; по умолчанию h / (w1 + w2)")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(run_validation(), indent=2))
        return
    solver = solve_and_save(args.output, SolverConfig(spacing=args.spacing,
        evader_directions=args.directions, max_time=args.max_time,
        extent=args.extent, time_step=args.time_step))
    print(json.dumps({"output": str(args.output), "grid": f"{solver.n}x{solver.n}",
                      "iterations": solver.iterations, "residual": solver.residual,
                      "converged": solver.converged,
                      "isaacs_barrier": barrier_comparison(solver)}, indent=2))


if __name__ == "__main__":
    main()
