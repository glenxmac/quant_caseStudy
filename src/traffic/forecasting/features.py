"""Feature engineering. Everything here uses only information available at the decision epoch
(the end of a window). Targets and oracle columns are built separately and never enter FEATURES.
Counts-only sensing: the live inputs are the 8 entrance counters, the clock, the weather feed and the event calendar
(published in advance, so its value for the window being forecast is known at decision time).
`entry_count` (cars on the entry road) is kept only to flag censoring (cnt_lag*, the guard); offline it
comes from the simulator and stands in for a survey flag, online from the controller's queue model.

Row = (day, entry, decision window w, horizon h). One global model serves all horizons (h is a feature).
"""
import numpy as np
import pandas as pd

from .history import BIN_S

K_LAGS = 6                                   # 6 windows of memory = 30 min at W = 5


def to_windows(hist: pd.DataFrame, W_min: float = 5.0) -> pd.DataFrame:
    """Aggregate 30 s bins to windows of W_min minutes (W_min must be a multiple of 0.5)."""
    nb = int(round(W_min * 60 / BIN_S))
    h = hist.sort_values(["day", "entry", "t_start_s"]).copy()
    h["w"] = (h.t_start_s // (nb * BIN_S)).astype(int)
    g = h.groupby(["day", "entry", "w"], sort=True)
    out = g.agg(n=("observed", "size"), observed=("observed", "sum"), entry_count=("entry_count", "last"),
                rain=("rain", "mean"), event=("event", "mean"), accident=("accident", "max"),
                true_rate=("true_rate", "mean"), weekend=("weekend", "first"), hour=("hour", "last")).reset_index()
    out = out[out.n == nb].drop(columns="n").reset_index(drop=True)
    out["obs_rate"] = out.observed / (W_min * 60)            # veh/s, noisy and censored
    return out


def build_features(win: pd.DataFrame, W_min: float = 5.0, H: int = 3, rain_sigma: float = 0.10,
                   seed: int = 0) -> pd.DataFrame:
    """Long table with FEATURES, the target `y` (true rate of window w+h, veh/s) and bookkeeping/oracle columns."""
    df = win.sort_values(["day", "entry", "w"]).reset_index(drop=True)
    df["net_rate"] = df.groupby(["day", "w"]).obs_rate.transform("sum")      # whole-network demand signal
    g = df.groupby(["day", "entry"])
    for i in range(K_LAGS):
        df[f"lag{i}"] = g.obs_rate.shift(i)
    for i in range(K_LAGS):
        df[f"cnt_lag{i}"] = g.entry_count.shift(i)          # used by the censoring guard, not as model inputs
    for i in range(3):
        df[f"net_lag{i}"] = g.net_rate.shift(i)
    df["roll6"] = df[[f"lag{i}" for i in range(K_LAGS)]].mean(axis=1)
    df["net_roll6"] = g.net_rate.transform(lambda s: s.rolling(K_LAGS).mean())
    df["rain_now"] = df.rain
    df["ev_now"] = np.log1p(df.event)                       # event calendar: log(1 + announced extra demand)
    for i in range(K_LAGS):
        df[f"ev_lag{i}"] = g.event.shift(i)                 # used by the profile x multiplier fallback, not a model input
    df["hour_dec"] = df.hour
    for e in range(8):
        df[f"e{e}"] = (df.entry == e).astype(float)
    df["true_now"] = df.true_rate                           # ORACLE ONLY (never a feature)
    rng = np.random.default_rng(seed)
    parts = []
    for h in range(1, H + 1):
        p = df.copy()
        gh = p.groupby(["day", "entry"])
        p["h"] = h
        p["y"] = gh.true_rate.shift(-h)
        p["obs_target"] = gh.obs_rate.shift(-h)              # what the sensor will count in that window (scoring only)
        p["rain_target"] = gh.rain.shift(-h)                 # truth, used only to label the regime
        # noisy forecast of the rain intensity h windows ahead: truth + noise that grows with lead time
        p["rain_fh"] = (p.rain_target + rng.normal(0, rain_sigma * np.sqrt(h), len(p))).clip(0, 1)
        p["ev_fh"] = np.log1p(gh.event.shift(-h))            # the calendar for the window being forecast (known ahead)
        p["thr_hour"] = p.hour_dec + h * W_min / 60.0        # clock hour of the window being forecast
        parts.append(p)
    L = pd.concat(parts, ignore_index=True)
    for k in (1, 2, 3, 4):                                   # harmonics let a log-linear model draw a double-peaked day
        L[f"thr_s{k}"] = np.sin(2 * np.pi * k * L.thr_hour / 24)
        L[f"thr_c{k}"] = np.cos(2 * np.pi * k * L.thr_hour / 24)
    L = L.dropna(subset=[f"lag{K_LAGS - 1}", "net_lag2", "net_roll6", "y", "rain_fh"]).reset_index(drop=True)
    return L


CALENDAR = (["h", "weekend"] + [f"e{e}" for e in range(8)] + [f"thr_{t}{k}" for k in (1, 2, 3, 4) for t in "sc"]
            + ["ev_now", "ev_fh"])
WEATHER = ["rain_now", "rain_fh"]
LIVE_COUNTS = [f"lag{i}" for i in range(K_LAGS)] + ["roll6", "net_lag0", "net_lag1", "net_lag2", "net_roll6"]
FEATURE_SETS = {"calendar": CALENDAR, "calendar+weather": CALENDAR + WEATHER, "+live counts": CALENDAR + WEATHER + LIVE_COUNTS}
PATTERN_FEATURES = FEATURE_SETS["calendar+weather"]     # patterns-only models: the fallback, no live sensors
LIVE_FEATURES = FEATURE_SETS["+live counts"]             # the main forecaster's inputs: entrance counts, no occupancy
FEATURES = LIVE_FEATURES
GBM_EXTRA = ["thr_hour"]                                 # trees can use raw clock time directly
