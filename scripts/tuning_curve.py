"""Cycle sweep behind the tuned fixed-time baseline: mean delay of the green wave for each common cycle length
on the NOMINAL scenarios, tuning seeds only. The best cycle per scenario is what `main.py tune-scen` stores.
usage: PYTHONPATH=src python scripts/tuning_curve.py  -> presentation/figs/tuning_curve.csv"""
from pathlib import Path

from traffic.config import SimConfig
from traffic.experiments.scen import CONST_CYCLES, TUNE_SEEDS, run_scen

if __name__ == "__main__":
    ctrls = {str(c): ("fixed_sync", {"cycle_s": float(c)}) for c in CONST_CYCLES}
    df = run_scen(ctrls, ["am_ramp", "pm_ramp"], TUNE_SEEDS, SimConfig(), workers=4)
    out = (df.assign(cycle_s=df.controller.astype(float)).groupby(["scenario", "cycle_s"]).mean_delay_s
             .mean().round(2).reset_index())
    Path("presentation/figs").mkdir(parents=True, exist_ok=True)
    out.to_csv("presentation/figs/tuning_curve.csv", index=False)
    print(out)
