"""Моделирование игры в арене 800x600 со стратегиями на основе V(x, y).

Запуск из корня репозитория:  python docs/coursework/make_arena_figures.py
Нужна сохранённая сетка data/reduced_hji.json (python -m pursuit_game.arena_policy).
Результат: figures/arena_trajectory.png, figures/arena_times.png, figures/arena_runs.json.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pursuit_game.arena_policy import (ArenaPolicyConfig, ReducedArenaPolicy,  # noqa: E402
                                       load_or_build, to_reduced)

OUT = Path(__file__).resolve().parent / "figures"
W, H = 800.0, 600.0
W1, W2, R, L = 120.0, 100.0, 60.0, 35.0
DT, TMAX = 0.02, 60.0


def pursuit_phi(px, py, th, ex, ey):
    """Простое преследование: поворот с максимальной кривизной в сторону E."""
    x, y = to_reduced(px, py, th, ex, ey)
    return 1.0 if x > 0 else -1.0


def flee_psi(px, py, ex, ey):
    """Простое бегство: двигаться прямо от P."""
    return math.atan2(ex - px, -(ey - py))


def play(policy, start, driver="hji", evader="hji", record=False):
    px, py, th, ex, ey = start
    policy._last_phi = None
    t, traj = 0.0, [(px, py, ex, ey)]
    while t < TMAX:
        phi, psi = policy.optimal_controls(px, py, ex, ey, th)
        if driver == "pursuit":
            phi = pursuit_phi(px, py, th, ex, ey)
        if evader == "flee":
            psi = flee_psi(px, py, ex, ey)
        th += W1 / R * phi * DT
        px = min(W, max(0.0, px + W1 * math.sin(th) * DT))
        py = min(H, max(0.0, py - W1 * math.cos(th) * DT))
        ex = min(W, max(0.0, ex + W2 * math.sin(psi) * DT))
        ey = min(H, max(0.0, ey - W2 * math.cos(psi) * DT))
        t += DT
        if record:
            traj.append((px, py, ex, ey))
        if math.hypot(ex - px, ey - py) <= L:
            return t, traj
    return None, traj


WALL_NEAR = 25.0  # «прижат к стене»: расстояние E до ближайшей стены в момент захвата
_POLICY = None


def _init_worker(config):
    global _POLICY
    _POLICY = ReducedArenaPolicy(load_or_build(build=False), config)


def _play_worker(args):
    start, driver, evader = args
    t, traj = play(_POLICY, start, driver, evader, record=True)
    px, py, ex, ey = traj[-1]
    return t, min(ex, ey, W - ex, H - ey)


def random_starts(n, seed=1):
    rnd = random.Random(seed)
    starts = []
    while len(starts) < n:
        px, py = rnd.uniform(50, 750), rnd.uniform(50, 550)
        ex, ey = rnd.uniform(50, 750), rnd.uniform(50, 550)
        if math.hypot(ex - px, ey - py) >= 150:
            starts.append((px, py, rnd.uniform(0, 2 * math.pi), ex, ey))
    return starts


def main(n_games: int = 200) -> None:
    solver = load_or_build(build=False)
    if solver is None:
        raise SystemExit("нет data/reduced_hji.json: python -m pursuit_game.arena_policy")
    policy = ReducedArenaPolicy(solver, ArenaPolicyConfig(width=W, height=H))
    summary_cfg = dict(horizon=policy.config.horizon, evader_tail=policy.config.evader_tail)
    starts = random_starts(n_games)
    summary = {}
    times_main = []
    for driver, evader in (("hji", "hji"), ("pursuit", "hji"), ("hji", "flee"), ("pursuit", "flee")):
        t0 = time.perf_counter()
        with Pool(initializer=_init_worker, initargs=(policy.config,)) as pool:
            results = pool.map(_play_worker, [(st, driver, evader) for st in starts])
        times = [t for t, _ in results]
        ok = [t for t in times if t is not None]
        near = sum(1 for t, d in results if t is not None and d < WALL_NEAR)
        key = f"P:{driver} / E:{evader}"
        summary[key] = dict(games=len(times), captured=len(ok),
                            mean_time=round(sum(ok) / len(ok), 2) if ok else None,
                            max_time=round(max(ok), 2) if ok else None,
                            evader_at_wall=near,
                            wall_s=round(time.perf_counter() - t0, 1))
        print(key, summary[key], flush=True)
        if (driver, evader) == ("hji", "hji"):
            times_main = times
            values = [policy.value_at(st[0], st[1], st[3], st[4], st[2]) for st in starts]
    t0 = time.perf_counter()
    for _ in range(200):
        policy.optimal_controls(300.0, 500.0, 600.0, 250.0, 0.0)
    summary["config"] = summary_cfg
    summary["decision_ms"] = round((time.perf_counter() - t0) / 200 * 1000, 2)
    fixed = (300.0, 500.0, 0.0, 600.0, 250.0)
    t_fixed, traj = play(policy, fixed, record=True)
    summary["fixed_start"] = dict(capture_time=round(t_fixed, 2) if t_fixed else None,
                                  reduced_value=round(policy.value_at(300, 500, 600, 250, 0.0), 2))
    OUT.mkdir(exist_ok=True)
    (OUT / "arena_runs.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(summary["decision_ms"], summary["fixed_start"])

    # траектории для фиксированного старта
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], "k-", lw=2)
    ax.plot([p[0] for p in traj], [p[1] for p in traj], color="tab:blue", label="шофёр P")
    ax.plot([p[2] for p in traj], [p[3] for p in traj], color="tab:red", label="убегающий E")
    ax.plot(*traj[0][:2], "o", color="tab:blue")
    ax.plot(*traj[0][2:], "o", color="tab:red")
    end = traj[-1]
    ax.add_patch(plt.Circle(end[:2], L, fill=False, ls="--", color="tab:blue"))
    ax.set_xlim(-20, W + 20)
    ax.set_ylim(H + 20, -20)
    ax.set_aspect("equal")
    result = f"захват за {t_fixed:.2f} с" if t_fixed else f"захвата нет за {TMAX:g} с"
    ax.set_title(f"Игра в арене: P(300, 500), E(600, 250); {result}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "arena_trajectory.png", dpi=200)
    plt.close(fig)

    # время захвата в арене и V(x, y) без стен
    fig, ax = plt.subplots(figsize=(7, 4.6))
    pairs = [(v, t) for v, t in zip(values, times_main) if t is not None]
    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=12)
    lim = max(max(p[0] for p in pairs), max(p[1] for p in pairs)) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="t = V")
    ax.set_xlabel("V(x, y) — цена редуцированной игры без стен, с")
    ax.set_ylabel("время захвата в арене, с")
    ax.set_title(f"{len(pairs)} из {len(times_main)} случайных стартов: захват")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "arena_times.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
