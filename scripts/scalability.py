# usage: PYTHONPATH=src python scripts/scalability.py <rows> <cols>   (20 simulated minutes, optimiser timing vs a fixed plan)
import sys, time, numpy as np
from dataclasses import replace
from traffic.config import SimConfig
from traffic.controllers import REGISTRY
from traffic.scenarios import SCENARIOS, build_sim
sc = replace(SCENARIOS["am_ramp"], start_hour=7.0, duration_s=1200)
r, c = int(sys.argv[1]), int(sys.argv[2]); mode = sys.argv[3] if len(sys.argv) > 3 else "persistence"
cfg = SimConfig(warmup_s=0, grid_rows=r, grid_cols=c)
res = {}
for name, params in (("fixed", ("fixed_sync", {})), ("smart", ("forecast_opt", {"mode": mode}))):
    ctrl = REGISTRY[params[0]](**params[1]); tsolve = []
    if name == "smart":
        orig = ctrl._decide
        def timed(*a, orig=orig):
            t0 = time.perf_counter(); orig(*a); tsolve.append(time.perf_counter() - t0)
        ctrl._decide = timed
    sim = build_sim(sc, cfg, ctrl, 1)
    t0 = time.perf_counter(); out = sim.run(); wall = time.perf_counter() - t0
    res[name] = dict(wall=wall, delay=out["mean_delay_s"], veh=out["n_vehicles"], solve_ms=1000*np.mean(tsolve) if tsolve else 0,
                     solve_max_ms=1000*np.max(tsolve) if tsolve else 0, fail=getattr(ctrl, "n_fail", 0))
n_i = r * c
print(f"{r}x{c} ({n_i:3d} int, {4*n_i:3d} links)  sim wall: fixed {res['fixed']['wall']:5.1f}s smart {res['smart']['wall']:5.1f}s | "
      f"solve mean {res['smart']['solve_ms']:6.0f} ms max {res['smart']['solve_max_ms']:6.0f} ms fails {res['smart']['fail']} | "
      f"delay fixed {res['fixed']['delay']:.0f} smart {res['smart']['delay']:.0f}  cars {res['smart']['veh']}")
