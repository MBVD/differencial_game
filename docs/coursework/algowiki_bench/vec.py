"""Vectorized (numpy) version of the same semi-Lagrangian value iteration, with precomputed transitions."""
import sys, time, json, math
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from pursuit_game.hji_solver import HJISolver, SolverConfig

def build(cfg):
    s = HJISolver(cfg)
    n, h, L = s.n, s.spacing, cfg.extent
    xs = np.array(s.xs); X, Y = np.meshgrid(xs, xs)          # X[j,i]=x_i, Y[j,i]=y_j
    X = X.ravel(); Y = Y.ravel(); N = X.size
    phis = np.array([-1.0, 0.0, 1.0]); psis = np.array(s.psi_values)
    P, K = len(phis), len(psis)
    turn = cfg.w1 / cfg.turn_radius * phis[:, None, None]     # (P,1,1)
    vx = -turn * Y[None, None, :] + cfg.w2 * np.sin(psis)[None, :, None]
    vy = turn * X[None, None, :] - cfg.w1 + cfg.w2 * np.cos(psis)[None, :, None]
    xn = X + cfg.time_step * vx; yn = Y + cfg.time_step * vy   # (P,K,N)
    # capture along segment
    r = cfg.capture_radius
    dx, dy = xn - X, yn - Y
    a = dx*dx + dy*dy; b = 2*(X*dx + Y*dy); c = (X*X + Y*Y - r*r) * np.ones_like(a)
    disc = b*b - 4*a*c
    with np.errstate(invalid='ignore', divide='ignore'):
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1 = (-b - sq) / (2*a); t2 = (-b + sq) / (2*a)
    ok1 = (t1 >= 0) & (t1 <= 1); ok2 = (t2 >= 0) & (t2 <= 1)
    frac = np.where(ok1, t1, np.where(ok2, t2, np.nan))
    frac = np.where(c <= 0, 0.0, frac)
    hit = ~np.isnan(frac)
    cost_hit = cfg.time_step * np.nan_to_num(frac)
    gx = (xn + L) / h; gy = (yn + L) / h
    inside = xn*xn + yn*yn <= r*r
    out = (gx < 0) | (gy < 0) | (gx > n-1) | (gy > n-1)
    i0 = np.clip(np.floor(np.clip(gx, 0, n-1)).astype(int), 0, n-2)
    j0 = np.clip(np.floor(np.clip(gy, 0, n-1)).astype(int), 0, n-2)
    ax = np.clip(gx, 0, n-1) - i0; ay = np.clip(gy, 0, n-1) - j0
    idx = np.stack([j0*n+i0, j0*n+i0+1, (j0+1)*n+i0, (j0+1)*n+i0+1])
    w = np.stack([(1-ax)*(1-ay), ax*(1-ay), (1-ax)*ay, ax*ay])
    target = X*X + Y*Y <= r*r
    return dict(s=s, N=N, idx=idx, w=w, hit=hit, cost_hit=cost_hit, inside=inside, out=out, target=target)

def solve(cfg, T):
    cap = cfg.max_time
    V = np.where(T['target'], 0.0, cap)
    it = 0
    for it in range(1, cfg.max_iterations + 1):
        interp = (T['w'] * V[T['idx']]).sum(axis=0)              # (P,K,N) gather + weighted sum
        interp = np.where(T['inside'], 0.0, np.where(T['out'], cap, interp))
        cand = np.where(T['hit'], T['cost_hit'], cfg.time_step + interp)
        Vn = np.minimum(cap, cand.max(axis=1).min(axis=0))       # max over psi, min over phi
        Vn[T['target']] = 0.0
        res = np.abs(Vn - V).max(); V = Vn
        if res < cfg.tolerance: break
    return V, it, res

if __name__ == '__main__':
    sp = float(sys.argv[1]); cfg = SolverConfig(spacing=sp)
    t0 = time.perf_counter(); T = build(cfg); t1 = time.perf_counter()
    V, it, res = solve(cfg, T); t2 = time.perf_counter()
    out = dict(spacing=sp, nodes=T['N'], iters=it, residual=float(res), build_s=round(t1-t0,3),
               solve_s=round(t2-t1,3), per_iter_ms=round(1000*(t2-t1)/it,3),
               mem_transition_MB=round(sum(T[k].nbytes for k in ('idx','w','hit','cost_hit','inside','out'))/2**20,1))
    print(json.dumps(out))
