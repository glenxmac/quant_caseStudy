"""Entry point: count -> forecast -> model the queues -> optimise -> measure wait time.

  python main.py history [--n-days 60] [--seed-start 0] [--tag NAME]   # simulate forecaster training days -> outputs/history/
  python main.py history --n-days 10 --seed-start 5000 --tag fixed_sync_fresh   # the fresh days (never used for any decision)
  python main.py forecast [--final] [--ablation]   # fit + score the arrival forecaster (--final: refit, score the TEST days)
  python main.py forecast-plots [--stage charts]   # forecaster figures -> outputs/forecast_*.png
  python main.py tune-scen                         # tune the fixed-time cycle on nominal days -> outputs/tuned_scenarios.json
  python main.py viz-clips                         # tabbed demo, tuned fixed vs optimiser -> outputs/demo_viewer.html
  python main.py days [--n-days 300]               # true demand envelope across random days -> outputs/demand_days.png

Held-out evaluation, ablations and timing are in scripts/ (see README Setup).
"""
import argparse
import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _read_days(folder):
    import pandas as pd
    files = sorted(glob.glob(f"outputs/history/{folder}/day_*.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def forecast_ladder(models, B, tag, W):
    """Per-horizon ladder table (CSV + markdown) for the scored days; on the test stage also the fresh days + figure."""
    import pandas as pd
    from traffic.forecasting.evaluate import ladder_by_horizon
    from traffic.forecasting.features import build_features, to_windows
    sets = {"validation (10 days)" if tag == "val" else "test (10 days)": B}
    if tag == "test":
        sets["fresh (10 days)"] = build_features(to_windows(_read_days("fixed_sync_fresh"), W), W, 3)
        sets["held-out: test + fresh (20 days)"] = pd.concat(list(sets.values()), ignore_index=True)
    new = pd.concat([ladder_by_horizon(models, L, W).assign(dataset=k) for k, L in sets.items()], ignore_index=True)
    path = Path("outputs/forecast_ladder_by_horizon.csv")
    old = pd.read_csv(path) if path.exists() else new.iloc[:0]
    lad = pd.concat([old[~old.dataset.isin(new.dataset)], new], ignore_index=True)
    lad.to_csv(path, index=False)
    cols = [c for c in lad.columns if c.startswith("+")] + ["overall"]
    md = []
    for ds, g in lad.groupby("dataset", sort=False):
        md += [f"**{ds}** (error = MAE as % of mean true rate)", "", "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        md += ["| " + r.label + " | " + " | ".join(f"{r[c]:.1f}%" for c in cols) + " |" for _, r in g.iterrows()] + [""]
    Path("outputs/forecast_ladder_by_horizon.md").write_text("\n".join(md))
    print("\n".join(md))
    if tag == "test":
        from traffic.viz.forecastplots import plot_ladder
        k = "held-out: test + fresh (20 days)"
        plot_ladder(new[new.dataset == k], "outputs/forecast_ladder_by_horizon.png",
                    "Forecast error by horizon, 20 held-out days: live entrance counts do the work")


def forecast_cmd(args):
    import pandas as pd
    from traffic.forecasting.evaluate import ablation, paired_by_day, run
    hist = _read_days("fixed_sync")
    res, info, models, B, preds = run(hist, W_min=args.window_min, final=args.final)
    pd.set_option("display.width", 170)
    print(info)
    tag = "test" if args.final else "val"
    res.to_csv(f"outputs/forecast_{tag}_W{args.window_min:g}.csv", index=False)
    Path("outputs/.cache").mkdir(parents=True, exist_ok=True)       # the fitted models (online.py reads final_*.pkl)
    Path(f"outputs/.cache/{'final' if args.final else 'val'}_{args.window_min}.pkl").write_bytes(pickle.dumps((res, info, models)))
    forecast_ladder(models, B, tag, args.window_min)
    piv = lambda g: res[res.group == g].pivot_table(index="level", columns="model", values="rMAE_%").round(1)
    print(f"\nrMAE (percent) scored against the SENSOR counts (what a real city could check; includes Poisson noise of about {info['noise_floor_rel_%']:.0f} percent)")
    print(res[res.group == "all"].set_index("model")["rMAE_vs_sensor_%"].round(1).to_string())
    print("\nrMAE % (error / mean true rate), all windows"); print(piv("all").to_string())
    for g in ("horizon", "regime", "time"):
        print(f"\nrMAE % by {g}"); print(piv(g).to_string())
    sk = res[res.group == "all"].set_index("model")[["RMSE_vph", "skill_vs_profile"]].round(3)
    print("\nRMSE (veh/h per entry) and MSE skill vs hour-profile"); print(sk.to_string())
    if args.ablation:
        ab = ablation(hist, [_read_days("fixed_sync_fresh")], args.window_min)
        print("\nFeature-group ladder (GLM; held-out days: first test days + fresh days)"); print(ab.round(3).to_string(index=False))
        ab.to_csv(f"outputs/forecast_ablation_W{args.window_min:g}.csv", index=False)
    names = [n for n in preds if n != "persistence"]
    pb = paired_by_day(B, {n: preds[n] for n in names}, "hour_profile")
    print("\nPaired by day: % reduction in MSE vs hour-profile (mean over days, 95% CI)")
    print(pb.drop(columns="vs").round(1).to_string(index=False))
    pc = paired_by_day(B, {n: preds[n] for n in ("poisson_glm", "poisson_glm+live+guard", "gbm_poisson")}, other=["gbm_poisson"])
    print("\nPaired by day: model vs gradient boosting (positive = model has LOWER MSE than gbm)")
    print(pc.round(1).to_string(index=False))
    pb.to_csv(f"outputs/forecast_{tag}_paired_W{args.window_min:g}.csv", index=False)


def days_cmd(args):
    from traffic.daygen import sample_day
    from traffic.network import build_grid
    from traffic.viz.dayplots import demand_matrix, plot_demand_days
    days = [sample_day(i, 5.0, 16 * 3600) for i in range(args.n_days or 300)]
    hours, dem, rain = demand_matrix(days, build_grid(2, 2))
    plot_demand_days(hours, dem, (rain > 0.1).mean(0), "outputs/demand_days.png")
    print("wrote outputs/demand_days.png", dem.shape)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["history", "forecast", "forecast-plots", "tune-scen", "viz-clips", "days"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-days", type=int, default=None, help="history: days to simulate (default 60); days: (default 300)")
    p.add_argument("--seed-start", type=int, default=0, help="history: first day seed")
    p.add_argument("--tag", default=None, help="history: output sub-folder (default fixed_sync)")
    p.add_argument("--demand-scale", type=float, default=0.85, help="history: demand scale (1.0 = saturation knee)")
    p.add_argument("--window-min", type=float, default=5.0, help="forecast: window length W (minutes)")
    p.add_argument("--final", action="store_true", help="forecast: refit on train+val and score the TEST days (once)")
    p.add_argument("--ablation", action="store_true", help="forecast: feature-group ladder (calendar -> weather -> live entrance counts)")
    p.add_argument("--stage", default="charts", choices=["val", "final", "charts"], help="forecast-plots: fit stage (cached) or draw charts")
    p.add_argument("--out", default="outputs/demo_viewer.html", help="viz-clips: output file")
    args = p.parse_args()
    if args.command == "history":
        from traffic.forecasting.history import make_history
        df = make_history(args.n_days or 60, args.seed_start, controller=("fixed_sync", {"cycle_s": 60.0}),
                          workers=args.workers, scale=args.demand_scale, tag=args.tag)
        print("history:", df.shape, "days:", df.day.nunique())
    elif args.command == "forecast":
        forecast_cmd(args)
    elif args.command == "forecast-plots":
        from traffic.viz.forecastplots import make_plots
        day = make_plots(args.window_min, args.stage)
        print("wrote outputs/forecast_day.png, forecast_two_days.png, forecast_regimes.png; day", day)
    elif args.command == "tune-scen":
        from traffic.config import SimConfig
        from traffic.experiments.scen import load_or_tune
        print("tuned cycle:", load_or_tune(SimConfig(warmup_s=300), workers=args.workers))
    elif args.command == "viz-clips":
        from traffic.viz.clips import CLIPS, build_clip
        from traffic.viz.render import export_clips
        print("wrote", export_clips([build_clip(c) for c in CLIPS], args.out))
    elif args.command == "days":
        days_cmd(args)


if __name__ == "__main__":
    main()
