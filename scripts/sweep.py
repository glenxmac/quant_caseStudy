"""Hyper-parameter check for the counts-only optimiser, on NOMINAL days only (am_ramp, pm_ramp) and tuning seeds
100-104. One parameter at a time around the defaults (config 0). Resumable: each config is written to
outputs/sweep_counts/<id>.json.   usage: PYTHONPATH=src python scripts/sweep.py <first> <last> [mode]
                                         PYTHONPATH=src python scripts/sweep.py summary   -> outputs/sweep_summary.csv"""
import glob, json, os, sys, time
from traffic.config import SimConfig
from traffic.experiments import scen

SEEDS = (100, 101, 102, 103, 104)
GRID = [("switch_w", [0.5, 5.0]), ("quad_w", [1.0, 10.0]), ("term_w", [1.0, 6.0]), ("l_eff", [8.0, 16.0]),
        ("gap_out", [False]), ("gap_s", [2.0, 8.0])]
CONFIGS = [{}] + [{k: v} for k, vs in GRID for v in vs]

if __name__ == "__main__":
    if sys.argv[1] == "summary":
        rows = sorted((json.load(open(f)) for f in glob.glob("outputs/sweep_counts/*.json")), key=lambda r: r["id"])
        for r in rows:
            print(f"{r['id']:3d} {str(r['params']):28s} am {r['am']:6.1f}  pm {r['pm']:6.1f}  score {r['score']:6.1f}  "
                  f"p95 {r['p95']:6.1f}  max {r['max']:6.1f}")
        import pandas as pd
        pd.DataFrame([{"id": r["id"], "change": str(r["params"]) if r["params"] else "defaults", "am_delay_s": r["am"],
                       "pm_delay_s": r["pm"], "score": r["score"], "p95_s": r["p95"]} for r in rows]
                     ).round(1).to_csv("outputs/sweep_summary.csv", index=False)
        print("wrote outputs/sweep_summary.csv")
        sys.exit()
    first, last = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "persistence"
    os.makedirs("outputs/sweep_counts", exist_ok=True)
    for i in range(first, min(last, len(CONFIGS) - 1) + 1):
        out = f"outputs/sweep_counts/{i}.json"
        if os.path.exists(out): continue
        p = dict(CONFIGS[i]); p["mode"] = mode
        t = time.time()
        df = scen.run_scen({"c": ("forecast_opt", p)}, ["am_ramp", "pm_ramp"], SEEDS, SimConfig(warmup_s=300), workers=8)
        res = dict(id=i, params=CONFIGS[i], mode=mode, am=float(df[df.scenario == "am_ramp"].mean_delay_s.mean()),
                   pm=float(df[df.scenario == "pm_ramp"].mean_delay_s.mean()), p95=float(df.p95_delay_s.mean()),
                   max=float(df.mean_delay_s.max()), per_run=df[["scenario", "seed", "mean_delay_s"]].values.tolist(),
                   secs=round(time.time() - t))
        res["score"] = (res["am"] + res["pm"]) / 2
        json.dump(res, open(out, "w")); print(i, CONFIGS[i], round(res["am"], 1), round(res["pm"], 1), res["secs"], "s", flush=True)
