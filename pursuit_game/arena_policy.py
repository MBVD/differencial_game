"""Стратегия игроков в прямоугольной арене на основе редуцированной функции цены.

Функция цены V(x, y) строится полулагранжевой схемой (hji_solver) в
редуцированном пространстве игры «шофёр-убийца», т.е. на двумерной сетке.
Стены арены в редуцированных координатах не выражаются: их положение зависит
от абсолютных координат и курса шофёра. Поэтому стены учитываются
приближённо --- коротким прогнозом движения (rollout) на горизонте H с
постоянными управлениями и упором в стены, после которого состояние
переводится в редуцированные координаты и оценивается по V(x, y):

    Q(phi, psi) = t_захв + L,            если захват наступил на горизонте,
    Q(phi, psi) = H + L + V(x(H), y(H))  иначе,

Шофёр выбирает phi по этой таблице. Убегающему горизонта H мало: за 0,3 с
он «не видит» стену, к которой бежит. Поэтому направления убегающего
оцениваются более длинным прогнозом: после H он ещё evader_tail секунд
сохраняет выбранное направление (скользя вдоль стен), а шофёр поворачивает
к нему (базовая стратегия преследования); затем --- оценка по V(x, y).

L --- время, потерянное шофёром из-за упора в стену (доля шага, на
которую стена укоротила его перемещение). V(x, y) не знает о стенах и
считает, что шофёр всегда движется со скоростью w1; без поправки L шофёр,
упёршийся в угол, может «застрять».
    phi* = argmin_phi max_psi Q,   psi* = argmax_psi Q(phi*, psi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .hji_solver import HJISolver, SolverConfig

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "reduced_hji.json"


def reduced_solver_config(w1: float = 120.0, w2: float = 100.0, turn_radius: float = 60.0,
                          capture_radius: float = 35.0) -> SolverConfig:
    """Сетка для арены 800x600: относительное расстояние не превышает 1000."""
    return SolverConfig(w1=w1, w2=w2, turn_radius=turn_radius, capture_radius=capture_radius,
                        extent=1100.0, spacing=10.0, max_time=100.0, max_iterations=6000,
                        tolerance=1e-4, evader_directions=24)


def to_reduced(px: float, py: float, theta: float, ex: float, ey: float) -> tuple[float, float]:
    """Мировые координаты (ось y экрана вниз) -> редуцированные (ось y по курсу P)."""
    dx = ex - px
    dy = py - ey
    s, c = math.sin(theta), math.cos(theta)
    return dx * c - dy * s, dx * s + dy * c


@dataclass(frozen=True)
class ArenaPolicyConfig:
    width: float = 800.0
    height: float = 600.0
    horizon: float = 0.3
    substep: float = 0.05
    directions: int = 36
    hysteresis: float = 0.05
    # продолжение прогноза для оценки направлений убегающего, с
    evader_tail: float = 2.0
    # шофёр оценивает phi против ответа убегающего (а не против худшего psi)
    driver_best_response: bool = True


class ReducedArenaPolicy:
    """Интерфейс совпадает с прежним сеточным решателем арены (value_at, optimal_controls)."""

    def __init__(self, solver: HJISolver, config: ArenaPolicyConfig = ArenaPolicyConfig()):
        self.solver = solver
        self.config = config
        self.cap = solver.cap
        self.converged = solver.converged
        self.iterations = solver.iterations
        self._psis = [2.0 * math.pi * k / config.directions for k in range(config.directions)]
        self._phis = (-1.0, 0.0, 1.0)
        self._last_phi: float | None = None

    def value_at(self, px: float, py: float, ex: float, ey: float, theta: float) -> float:
        return self.solver.value_at(*to_reduced(px, py, theta, ex, ey))

    def _rollout(self, px, py, theta, ex, ey, phi, psi, tail: float = 0.0) -> float:
        cfg, sc = self.config, self.solver.config
        w, h = cfg.width, cfg.height
        steps = max(1, round(cfg.horizon / cfg.substep))
        dt = cfg.horizon / steps
        turn = sc.w1 / sc.turn_radius * phi
        evx, evy = sc.w2 * math.sin(psi) * dt, -sc.w2 * math.cos(psi) * dt
        l2 = sc.capture_radius ** 2
        step = sc.w1 * dt
        lost = 0.0  # время, «потерянное» шофёром из-за упора в стену
        for k in range(1, steps + 1):
            theta += turn * dt
            nx = min(w, max(0.0, px + sc.w1 * math.sin(theta) * dt))
            ny = min(h, max(0.0, py - sc.w1 * math.cos(theta) * dt))
            lost += dt * (1.0 - math.hypot(nx - px, ny - py) / step)
            px, py = nx, ny
            ex = min(w, max(0.0, ex + evx))
            ey = min(h, max(0.0, ey + evy))
            if (ex - px) ** 2 + (ey - py) ** 2 <= l2:
                return k * dt + lost
        t = cfg.horizon
        # Продолжение прогноза (rollout с базовой стратегией): убегающий
        # сохраняет выбранное направление и скользит вдоль стен, шофёр
        # поворачивает к убегающему. Так оценка «видит» стены дальше H.
        for k in range(1, max(0, round(tail / cfg.substep)) + 1):
            x, y = to_reduced(px, py, theta, ex, ey)
            theta += sc.w1 / sc.turn_radius * (1.0 if x > 0 else -1.0) * dt
            nx = min(w, max(0.0, px + sc.w1 * math.sin(theta) * dt))
            ny = min(h, max(0.0, py - sc.w1 * math.cos(theta) * dt))
            lost += dt * (1.0 - math.hypot(nx - px, ny - py) / step)
            px, py = nx, ny
            ex = min(w, max(0.0, ex + evx))
            ey = min(h, max(0.0, ey + evy))
            t += dt
            if (ex - px) ** 2 + (ey - py) ** 2 <= l2:
                return t + lost
        return t + lost + self.solver.value_at(*to_reduced(px, py, theta, ex, ey))

    def _blocked(self, px, py, theta) -> bool:
        """Шофёр упирается в стену: курс направлен наружу у границы арены."""
        cfg = self.config
        dx, dy = math.sin(theta), -math.cos(theta)
        eps = 1e-6
        out_x = (px <= eps and dx < -0.5) or (px >= cfg.width - eps and dx > 0.5)
        out_y = (py <= eps and dy < -0.5) or (py >= cfg.height - eps and dy > 0.5)
        return out_x or out_y

    def evaluate(self, px, py, ex, ey, theta):
        """Таблица Q(phi, psi) прогноза для всех пар управлений."""
        return {phi: [self._rollout(px, py, theta, ex, ey, phi, psi) for psi in self._psis]
                for phi in self._phis}

    def optimal_controls(self, px: float, py: float, ex: float, ey: float,
                         theta: float) -> tuple[float, float]:
        """Вернуть (phi шофёра, мировой курс psi убегающего).

        При почти равных оценках шофёр сохраняет прежнее управление
        (гистерезис): иначе у стены или в углу он может бесконечно
        переключаться между поворотами налево и направо и «застрять».
        """
        worst = {}
        for phi, row in self.evaluate(px, py, ex, ey, theta).items():
            k = max(range(len(row)), key=row.__getitem__)
            if self.config.driver_best_response and self.config.evader_tail > 0:
                k = self._evader_choice(px, py, theta, ex, ey, phi)
            worst[phi] = (row[k], self._psis[k])
        last = self._last_phi
        if self._blocked(px, py, theta):
            # Упор в стену: прямолинейное движение бесполезно. Шофёр
            # поворачивает и продолжает поворот в ту же сторону, пока не
            # освободится (иначе в углу он колеблется между -1 и +1).
            if last in (-1.0, 1.0):
                best_phi = last
            else:
                best_phi = min((-1.0, 1.0), key=lambda phi: worst[phi][0])
        else:
            best_phi = min(worst, key=lambda phi: worst[phi][0])
            if (last is not None
                    and worst[last][0] <= worst[best_phi][0] + self.config.hysteresis):
                best_phi = last
        self._last_phi = best_phi
        psi = worst[best_phi][1]
        if self.config.evader_tail > 0 and not self.config.driver_best_response:
            psi = self._psis[self._evader_choice(px, py, theta, ex, ey, best_phi)]
        return best_phi, psi

    def _evader_choice(self, px, py, theta, ex, ey, phi) -> int:
        """Убегающий оценивает направления длинным прогнозом: стены дальше H."""
        row = [self._rollout(px, py, theta, ex, ey, phi, p, self.config.evader_tail)
               for p in self._psis]
        return max(range(len(row)), key=row.__getitem__)


def load_or_build(path: Path = DATA_FILE, config: SolverConfig | None = None,
                  build: bool = True) -> HJISolver | None:
    """Загрузить сохранённую редуцированную сетку или посчитать её (нужен NumPy)."""
    import json
    config = config or reduced_solver_config()
    try:
        solver = HJISolver.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if solver.config == config and solver.converged:
            return solver
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not build:
        return None
    from .hji_fast import solve_fast
    solver = solve_fast(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(solver.to_dict()), encoding="utf-8")
    tmp.replace(path)
    return solver


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    s = load_or_build()
    print(f"{s.n}x{s.n} узлов, итераций {s.iterations}, сходимость {s.converged}, "
          f"{time.perf_counter() - t0:.1f} с")
