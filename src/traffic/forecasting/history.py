"""History generator: simulate full random days and log one row per (30 s bin, entry).

This is the forecaster's training data, produced by the SAME vehicle simulator the controllers
run in, so entry-road occupancy and censoring are realistic. One row = one entry, one 30 s bin.

Columns
  day, weekend, t_start_s, hour (clock hour at the END of the bin), entry
  observed      cars that passed the entry sensor in the bin (noisy, censored when the road is full)
  entry_count   cars on the entry road at the end of the bin       (censoring flag only, never a model input)
  entry_queue   of those, cars moving slower than 2 m/s            (diagnostic only)
  rain          mean rain intensity in the bin, 0..1               (known 'now'; the forecast is built from it)
  true_rate     mean true demand rate in the bin, veh/s            (TARGET: never an input)
  created, backlog, accident   diagnostics only. `created` and `backlog` are not observable by a
                controller (cars waiting outside the network); `accident` is the incident flag, kept
                out of the default feature set (it is a later rung of the value-of-information ladder).

The controller that drives the sim shapes occupancy, so a forecaster trained under one controller
may shift under another: generate under the baseline, then test under a different controller.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SimConfig
from ..daygen import sample_day
from ..scenarios import build_sim

BIN_S = 30
DEFAULT_CONTROLLER = ("fixed_sync", {"cycle_s": 60.0})     # the untuned 60 s green wave


def simulate_day(day_seed: int, controller=DEFAULT_CONTROLLER, start_hour: float = 5.0, hours: float = 16.0,
                 weekend: bool = False, scale: float = 0.85, cfg: SimConfig = None) -> pd.DataFrame:
    from ..controllers import REGISTRY      # imported here: controllers -> forecasting.online -> features -> history
    name, params = controller
    scen = sample_day(day_seed, start_hour, int(hours * 3600), weekend=weekend, scale=scale)
    sim = build_sim(scen, cfg or SimConfig(), REGISTRY[name](**params), day_seed, series_every_s=BIN_S)
    sim.run()
    sr = {k: np.array(v) for k, v in sim.arrival_series.items()}
    t = sr["t"]                                            # bin edges: bin k covers (t[k-1], t[k]]
    n_e = sr["created"].shape[1]
    rain_fn = scen.rain_fn()
    sub = np.arange(0.5, BIN_S, 1.0)                       # sample the true rate / rain inside each bin
    rows = []
    for k in range(1, len(t)):
        t0, t1 = t[k - 1], t[k]
        ts = t0 + sub
        rate = np.mean([sim.demand.rates(x) for x in ts], axis=0)
        rain = float(np.mean([rain_fn(x) for x in ts])) if rain_fn else 0.0
        acc = int(any(a.start_s < t1 and a.end_s > t0 for a in scen.accidents))
        rows.append(pd.DataFrame({
            "day": day_seed, "weekend": int(weekend), "t_start_s": t0, "hour": start_hour + t1 / 3600.0,
            "entry": np.arange(n_e),
            "observed": sr["observed"][k] - sr["observed"][k - 1],
            "created": sr["created"][k] - sr["created"][k - 1],
            "backlog": sr["backlog"][k], "entry_count": sr["entry_count"][k], "entry_queue": sr["entry_queue"][k],
            "rain": rain, "accident": acc, "true_rate": rate}))
    return pd.concat(rows, ignore_index=True)


def _job(args):
    seed, weekend, controller, hours, out, scale = args
    path = Path(out) / f"day_{seed}.csv.gz"
    if path.exists():
        return path
    df = simulate_day(seed, controller, hours=hours, weekend=weekend, scale=scale)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)                      # atomic: a killed job never leaves a half-written day
    return path


def make_history(days: int, seed: int = 0, controller=DEFAULT_CONTROLLER, hours: float = 16.0,
                 out_dir: str = "outputs/history", workers: int = 4, scale: float = 0.85,
                 tag: str = None) -> pd.DataFrame:
    """`days` random days (seeds seed..seed+days-1; every 6th and 7th is a weekend). Resumable: a day
    already on disk is not re-simulated. Day order is calendar order, so a temporal split is by day."""
    out = Path(out_dir) / (tag or controller[0])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed + i, (i % 7) in (5, 6), controller, hours, str(out), scale) for i in range(days)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(_job, jobs))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
