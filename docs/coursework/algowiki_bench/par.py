"""Row-block parallel Jacobi sweep with multiprocessing (same kernel as HJISolver.solve)."""
import sys, time, math, json
from multiprocessing import Pool
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pursuit_game.hji_solver import HJISolver, SolverConfig
S = None
def init(sp):
    global S; S = HJISolver(SolverConfig(spacing=sp))
def sweep(args):
    j0, j1, old = args
    cfg = S.config; rows = []; mc = 0.0
    for j in range(j0, j1):
        y = S.ys[j]; row = []
        for i, x in enumerate(S.xs):
            if x*x + y*y <= cfg.capture_radius**2: row.append(0.0); continue
            best = S.cap
            for phi in S.phi_values:
                turn = cfg.w1/cfg.turn_radius*phi; worst = 0.0
                for psi in S.psi_values:
                    vx = -turn*y + cfg.w2*math.sin(psi); vy = turn*x - cfg.w1 + cfg.w2*math.cos(psi)
                    xn = x + cfg.time_step*vx; yn = y + cfg.time_step*vy
                    f = S._capture_fraction(x, y, xn, yn)
                    cnd = cfg.time_step*f if f is not None else cfg.time_step + S._interpolate(xn, yn, old)
                    worst = max(worst, cnd)
                best = min(best, worst)
            v = min(S.cap, best); row.append(v); mc = max(mc, abs(v-old[j][i]))
        rows.append(row)
    return rows, mc
if __name__ == '__main__':
    sp, p, iters = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    init(sp); n = S.n; V = S.values
    blocks = [(k*n//p, (k+1)*n//p) for k in range(p)]
    with Pool(p, initializer=init, initargs=(sp,)) as pool:
        t = time.perf_counter()
        for _ in range(iters):
            res = pool.map(sweep, [(a, b, V) for a, b in blocks])
            V = [r for rows, _ in res for r in rows]
        dt = time.perf_counter() - t
    print(json.dumps(dict(spacing=sp, procs=p, iters=iters, time_s=round(dt,2), per_iter_s=round(dt/iters,4))))
