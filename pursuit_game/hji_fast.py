"""Векторизованная (NumPy) версия полулагранжевой схемы из hji_solver.

Схема та же, что в HJISolver.solve (итерации по значению, вариант Якоби),
но точки z+ = z + tau*f, признаки захвата, индексы и веса билинейной
интерполяции не зависят от V, поэтому вычисляются один раз (таблица
переходов). Итерация сводится к выборке, взвешенной сумме и редукции
max по psi / min по phi. Результат совпадает с HJISolver до ошибок округления.
"""

from __future__ import annotations

import numpy as np

from .hji_solver import HJISolver, SolverConfig


def build_transitions(solver: HJISolver) -> dict:
    cfg = solver.config
    n, h, L = solver.n, solver.spacing, cfg.extent
    xs = np.asarray(solver.xs)
    X, Y = np.meshgrid(xs, xs)  # X[j, i] = x_i, Y[j, i] = y_j
    X, Y = X.ravel(), Y.ravel()
    phis = np.asarray(solver.phi_values)
    psis = np.asarray(solver.psi_values)
    turn = cfg.w1 / cfg.turn_radius * phis[:, None, None]
    vx = -turn * Y + cfg.w2 * np.sin(psis)[None, :, None]
    vy = turn * X - cfg.w1 + cfg.w2 * np.cos(psis)[None, :, None]
    xn, yn = X + cfg.time_step * vx, Y + cfg.time_step * vy  # (|Phi|, |Psi|, N)

    r = cfg.capture_radius
    dx, dy = xn - X, yn - Y
    a = dx * dx + dy * dy
    b = 2.0 * (X * dx + Y * dy)
    c = np.broadcast_to(X * X + Y * Y - r * r, a.shape)
    disc = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
    frac = np.where((t1 >= 0) & (t1 <= 1), t1, np.where((t2 >= 0) & (t2 <= 1), t2, np.nan))
    frac = np.where(c <= 0.0, 0.0, frac)
    hit = ~np.isnan(frac)

    gx, gy = (xn + L) / h, (yn + L) / h
    inside = xn * xn + yn * yn <= r * r
    outside = (gx < 0) | (gy < 0) | (gx > n - 1) | (gy > n - 1)
    gxc, gyc = np.clip(gx, 0, n - 1), np.clip(gy, 0, n - 1)
    i0 = np.minimum(np.floor(gxc).astype(np.int64), n - 2)
    j0 = np.minimum(np.floor(gyc).astype(np.int64), n - 2)
    ax, ay = gxc - i0, gyc - j0
    idx = np.stack([j0 * n + i0, j0 * n + i0 + 1, (j0 + 1) * n + i0, (j0 + 1) * n + i0 + 1])
    w = np.stack([(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay])
    return dict(idx=idx, w=w, hit=hit, cost_hit=cfg.time_step * np.nan_to_num(frac),
                inside=inside, outside=outside, target=(X * X + Y * Y <= r * r))


def solve_fast(config: SolverConfig, record_residuals: bool = False) -> HJISolver:
    """Решить сеточное уравнение; вернуть HJISolver с заполненными values."""
    solver = HJISolver(config)
    cfg, cap = solver.config, solver.cap
    T = build_transitions(solver)
    V = np.where(T["target"], 0.0, cap)
    residuals = []
    solver.converged = False
    for k in range(1, cfg.max_iterations + 1):
        interp = (T["w"] * V[T["idx"]]).sum(axis=0)
        interp = np.where(T["inside"], 0.0, np.where(T["outside"], cap, interp))
        cand = np.where(T["hit"], T["cost_hit"], cfg.time_step + interp)
        Vn = np.minimum(cap, cand.max(axis=1).min(axis=0))
        Vn[T["target"]] = 0.0
        res = float(np.abs(Vn - V).max())
        V = Vn
        residuals.append(res)
        solver.iterations, solver.residual = k, res
        if res < cfg.tolerance:
            solver.converged = True
            break
    solver.values = V.reshape(solver.n, solver.n).tolist()
    if record_residuals:
        solver.residual_history = residuals
    return solver
