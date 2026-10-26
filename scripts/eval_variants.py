"""Held-out evaluation of optimiser variants (same seeds as eval_chunk), resumable per scenario.
  opt_nogap     the default optimiser without the model-based gap-out (how much of the gain is the QP)
  opt_noinc     the default optimiser ignoring the incident feed (what the feed is worth in the accident)
usage: PYTHONPATH=src python scripts/eval_variants.py <scenario> <first_seed> <n_seeds>  -> outputs/final_eval_counts/variants_<scenario>_<seed>.csv"""
import sys, os
from traffic.config import SimConfig
from traffic.experiments import scen
VARIANTS = {"opt_nogap": ("forecast_opt", {"mode": "live", "gap_out": False}),
            "opt_noinc": ("forecast_opt", {"mode": "live", "incidents": False})}
if __name__ == "__main__":
    sc, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("outputs/final_eval_counts", exist_ok=True)
    out = f"outputs/final_eval_counts/variants_{sc}_{s0}.csv"
    if os.path.exists(out): print("skip", out); sys.exit()
    df = scen.run_scen(VARIANTS, [sc], range(s0, s0 + n), SimConfig(warmup_s=300), workers=4)
    df.to_csv(out, index=False); print("wrote", out, len(df))
