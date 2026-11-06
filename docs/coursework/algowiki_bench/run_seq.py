import sys, time, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pursuit_game.hji_solver import HJISolver, SolverConfig, barrier_comparison
sp = float(sys.argv[1])
cfg = SolverConfig(spacing=sp)
t = time.perf_counter()
s = HJISolver(cfg).solve()
dt = time.perf_counter() - t
fin = sum(v < cfg.max_time - cfg.time_step for row in s.values for v in row)
print(json.dumps(dict(spacing=sp, n=s.n, nodes=s.n**2, iters=s.iterations, residual=s.residual,
    converged=s.converged, time_s=round(dt,2), per_iter_s=round(dt/s.iterations,4), finite_nodes=fin,
    barrier=barrier_comparison(s))))
