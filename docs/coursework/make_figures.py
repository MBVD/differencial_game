"""Построение графиков функции цены для курсовой.

Запуск из корня репозитория:  python docs/coursework/make_figures.py
Результат: docs/coursework/figures/v_heatmap.png, v_slices.png,
v_convergence.png и сводка расчётов docs/coursework/figures/runs.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pursuit_game.hji_fast import solve_fast  # noqa: E402
from pursuit_game.hji_solver import (  # noqa: E402
    SolverConfig, isaacs_barrier_point, isaacs_capture_angles)

OUT = Path(__file__).resolve().parent / "figures"
WINDOW = (-200.0, 200.0, -150.0, 250.0)  # xmin, xmax, ymin, ymax
GRID = dict(extent=600.0, turn_radius=60.0, max_time=100.0, max_iterations=20000)


def barrier_branches(cfg: SolverConfig):
    """Начальные участки барьера Айзекса до их пересечения на оси y."""
    alpha = isaacs_capture_angles(cfg.w1, cfg.w2)[1]
    branches = []
    for sign in (1.0, -1.0):
        pts = []
        t = 0.0
        t_end = 0.95 * cfg.capture_radius / cfg.w2  # расстояние l - w2*t > 0
        while t < t_end:
            x, y = isaacs_barrier_point(cfg, sign * alpha, t)
            if t > 0 and sign * x < 0:  # ветви пересекаются на оси x = 0
                break
            pts.append((x, y))
            t += 0.002
        branches.append(np.array(pts))
    return branches


CACHE = ROOT / "build" / "hji_cache"


def cached_solve(cfg: SolverConfig):
    """Решение с кэшем в build/hji_cache (расчёт при h = 2,5 занимает минуты)."""
    from pursuit_game.hji_solver import HJISolver
    path = CACHE / f"h{cfg.spacing:g}_L{cfg.extent:g}_R{cfg.turn_radius:g}_T{cfg.max_time:g}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        s = HJISolver.from_dict(data)
        if s.config == cfg:
            s.residual_history = data["residual_history"]
            s.solve_time_s = data.get("solve_time_s")
            return s
    t0 = time.perf_counter()
    s = solve_fast(cfg, record_residuals=True)
    s.solve_time_s = round(time.perf_counter() - t0, 2)
    CACHE.mkdir(parents=True, exist_ok=True)
    data = s.to_dict()
    data["residual_history"] = s.residual_history
    data["solve_time_s"] = s.solve_time_s
    path.write_text(json.dumps(data), encoding="utf-8")
    return s


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs, solvers = [], {}
    for h in (20.0, 10.0, 5.0):
        cfg = SolverConfig(spacing=h, **GRID)
        s = cached_solve(cfg)
        # проверка: убегающий прямо по курсу, оба движутся по прямой,
        # V(0, y) = (y - l) / (w1 - w2)
        axis_err = max(abs(s.value_at(0.0, y) - (y - cfg.capture_radius) / (cfg.w1 - cfg.w2))
                       for y in np.arange(60.0, 301.0, 5.0))
        runs.append(dict(h=h, tau=round(cfg.time_step, 5), n=s.n, iterations=s.iterations,
                         converged=s.converged, residual=s.residual,
                         time_s=s.solve_time_s, axis_err=round(float(axis_err), 3)))
        solvers[h] = s
        print(runs[-1], flush=True)
    # самосходимость: разность решений на соседних сетках в общих узлах окна
    xmin, xmax, ymin, ymax = WINDOW
    hs = sorted(solvers, reverse=True)
    for coarse, fine in zip(hs, hs[1:]):
        sc, sf = solvers[coarse], solvers[fine]
        diffs = []
        for y in np.arange(ymin, ymax + 0.01, coarse):
            for x in np.arange(xmin, xmax + 0.01, coarse):
                if x * x + y * y <= sc.config.capture_radius ** 2:
                    continue
                a, b = sc.value_at(x, y), sf.value_at(x, y)
                if a < sc.cap - 1 and b < sf.cap - 1:
                    diffs.append(abs(a - b))
        for r in runs:
            if r["h"] == fine:
                r["diff_vs_coarser_mean"] = round(float(np.mean(diffs)), 3)
                r["diff_vs_coarser_max"] = round(float(np.max(diffs)), 3)
    print(runs)
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    s = solvers[5.0]
    cfg = s.config
    xmin, xmax, ymin, ymax = WINDOW
    l = cfg.capture_radius
    cut = cfg.max_time - cfg.time_step
    # билинейная интерполяция сеточного решения на мелкую сетку для показа
    gx = np.arange(xmin, xmax + 0.01, 1.0)
    gy = np.arange(ymin, ymax + 0.01, 1.0)
    Xw, Yw = np.meshgrid(gx, gy)
    Vw = np.array([[s.value_at(x, y) for x in gx] for y in gy])
    Vm = np.ma.masked_where((Vw >= cut) | (Xw ** 2 + Yw ** 2 <= l ** 2), Vw)
    vtop = 5.0 * math.ceil(float(Vm.max()) / 5.0)  # верх цветовой шкалы

    th = np.linspace(0, 2 * math.pi, 400)
    alpha = isaacs_capture_angles(cfg.w1, cfg.w2)[1]
    s_arc = np.linspace(-alpha, alpha, 100)
    branches = barrier_branches(cfg)
    y_cross = max(br[-1, 1] for br in branches)

    # 1. Тепловая карта + линии уровня
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ax.set_facecolor("#d9d9d9")
    pcm = ax.pcolormesh(Xw, Yw, Vm, cmap="viridis", shading="auto", vmin=0, vmax=vtop)
    cs = ax.contour(Xw, Yw, Vm, levels=np.arange(2, 30, 2), colors="white", linewidths=0.7)
    ax.clabel(cs, fmt="%d", fontsize=8)
    ax.fill(l * np.sin(th), l * np.cos(th), color="white", zorder=3)
    ax.plot(l * np.sin(th), l * np.cos(th), "k-", lw=1.2, zorder=4, label="окружность захвата")
    ax.plot(l * np.sin(s_arc), l * np.cos(s_arc), color="#d62728", lw=3, zorder=5,
            label=r"полезная часть: $\cos s > w_2/w_1$")
    for k, br in enumerate(branches):
        ax.plot(br[:, 0], br[:, 1], "--", color="#ff7f0e", lw=1.8, zorder=5,
                label="барьер (Айзекс)" if k == 0 else None)
    ax.arrow(0, -15, 0, 25, width=1.5, color="k", zorder=6)
    ax.text(5, -15, "P", fontsize=12, zorder=6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Функция цены V(x, y), h = {cfg.spacing:g}, τ = {cfg.time_step:.4f}")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.8)
    cb.set_label("V, время до захвата")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    if (Vw >= cut).any():
        ax.text(xmax - 2, ymin + 3, f"серым: V ≥ Tmax = {cfg.max_time:g}\n(захват за Tmax не гарантирован)",
                ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "v_heatmap.png", dpi=200)
    plt.close(fig)

    # 2. Срезы функции цены на разных сетках
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ys = np.linspace(l + 0.5, 250, 500)
    xs_ = np.linspace(-150, 150, 500)
    for h, sv in solvers.items():
        v1 = np.array([sv.value_at(0.0, y) for y in ys])
        v2 = np.array([sv.value_at(x, 45.0) for x in xs_])
        a1.plot(ys, np.where(v1 >= cut, np.nan, v1), label=f"h = {h:g}")
        a2.plot(xs_, np.where(v2 >= cut, np.nan, v2), label=f"h = {h:g}")
    a1.axvline(y_cross, color="#ff7f0e", ls="--", lw=1.2, label="барьер")
    for br in branches:
        k = np.argmin(np.abs(br[:, 1] - 45.0))
        if abs(br[k, 1] - 45.0) < 1.0:
            a2.axvline(br[k, 0], color="#ff7f0e", ls="--", lw=1.2)
    a1.set_xlabel("y  (x = 0, убегающий прямо по курсу)")
    a2.set_xlabel("x  (y = 45)")
    for a in (a1, a2):
        a.set_ylabel("V")
        a.set_ylim(0, vtop)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Срезы функции цены на трёх сетках")
    fig.tight_layout()
    fig.savefig(OUT / "v_slices.png", dpi=200)
    plt.close(fig)

    # 3. Сходимость итераций
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for h, sv in solvers.items():
        ax.semilogy(range(1, len(sv.residual_history) + 1), sv.residual_history,
                    label=f"h = {h:g}, τ = {sv.config.time_step:.4f}")
    ax.axhline(cfg.tolerance, color="k", ls=":", lw=1, label="ε = 1e-4")
    ax.set_xlabel("номер итерации k")
    ax.set_ylabel("невязка  max|V(k) − V(k−1)|")
    ax.set_title("Сходимость итераций по значению")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "v_convergence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
