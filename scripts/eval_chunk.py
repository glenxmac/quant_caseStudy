"""Held-out evaluation on one scenario (seeds s0..s0+n-1), resumable. Counts-only optimiser against the fixed-time
baselines: untuned 60 s green wave (fixed_default) and the tuned-cycle green wave (fixed_tuned).
usage: PYTHONPATH=src python scripts/eval_chunk.py <scenario> <first_seed> <n_seeds> [controllers,comma,separated]
       -> outputs/final_eval_counts/<scenario>_<seed>.csv  (or <scenario>_<seed>_<subset>.csv for a subset)"""
import json, sys, os
from traffic.config import SimConfig
from traffic.experiments import scen
if __name__ == "__main__":
    sc, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tuned = json.load(open("outputs/tuned_scenarios.json"))
    base = scen.baseline_controllers(tuned)
    ctrls = {k: base[k] for k in ("fixed_default", "fixed_tuned")}
    for m in ("persistence", "patterns", "live", "perfect"):
        ctrls["opt_" + m] = ("forecast_opt", {"mode": m})
    tag = ""
    if len(sys.argv) > 4:
        keep = sys.argv[4].split(","); ctrls = {k: ctrls[k] for k in keep}; tag = "_" + "-".join(keep)
    os.makedirs("outputs/final_eval_counts", exist_ok=True)
    out = f"outputs/final_eval_counts/{sc}_{s0}{tag}.csv"
    if os.path.exists(out): print("skip", out); sys.exit()
    df = scen.run_scen(ctrls, [sc], range(s0, s0 + n), SimConfig(warmup_s=300), workers=4)
    df.to_csv(out, index=False); print("wrote", out, len(df))
