"""Live forecasting inside the simulation: turns what the controller can see (entry counts, the clock, the
weather feed) and its own estimate of cars on each entry road (for the censoring guard) into the same feature
rows the models were trained on, once per window, and returns expected arrival rates for the next H windows.

Modes
  live         GLM + live counts + censoring guard (the main forecaster); patterns-only for the first 30 min
  patterns     profile x rain effect (no live sensors)
  persistence  repeat the last window's observed rate (the 'no forecast' reference)
  perfect      the true demand rate (oracle; an upper bound, only used for the value-of-information ladder)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import K_LAGS

DEFAULT_MODELS = "outputs/.cache/final_5.0.pkl"      # models fitted on the 40 training + 10 validation days
_CACHE = {}


def load_models(path: str = DEFAULT_MODELS) -> dict:
    if path not in _CACHE:
        obj = pickle.loads(Path(path).read_bytes())
        models = obj[2] if isinstance(obj, tuple) else obj
        _CACHE[path] = {m.name: m for m in models}
    return _CACHE[path]


def online_rows(w_idx, hour_dec, weekend, rates, cnts, queue, rain_now, rain_fh, W_min=5.0, H=3):
    """Feature rows for one decision epoch. rates, cnts: (K, n_entries), newest window first; cnts are cars on
    each entry road (here the controller's estimate), read by the censoring guard. Mirrors features.build_features
    column for column; the occupancy columns are filled but are not model inputs."""
    n_e = rates.shape[1]
    h = np.repeat(np.arange(1, H + 1), n_e)
    e = np.tile(np.arange(n_e), H)
    n = len(h)
    full = lambda v: np.full(n, float(v))
    d = {"day": np.zeros(n), "w": np.full(n, int(w_idx)), "h": h, "entry": e, "weekend": full(weekend),
         "hour_dec": full(hour_dec), "thr_hour": hour_dec + h * W_min / 60.0}
    for i in range(K_LAGS):
        d[f"lag{i}"] = rates[i, e]
        d[f"cnt_lag{i}"] = cnts[i, e]
    d["roll6"] = rates.mean(axis=0)[e]
    net = rates.sum(axis=1)
    for i in range(3):
        d[f"net_lag{i}"] = full(net[i])
    d["net_roll6"] = full(net.mean())
    d["entry_count"], d["entry_queue"] = cnts[0, e], np.asarray(queue)[e]
    d["rain_now"], d["rain_fh"] = full(rain_now), np.asarray(rain_fh)[h - 1]
    for k in (1, 2, 3, 4):
        d[f"thr_s{k}"] = np.sin(2 * np.pi * k * d["thr_hour"] / 24)
        d[f"thr_c{k}"] = np.cos(2 * np.pi * k * d["thr_hour"] / 24)
    for j in range(n_e):
        d[f"e{j}"] = (e == j).astype(float)
    return pd.DataFrame(d)


class OnlineForecaster:
    """Runs inside the simulation: once per window it rebuilds features from what the sensors saw so far and
    predicts entry arrival rates (veh/s) for the next H windows. Uses only information available at time t
    (clock, weather feed, entry counts, and the controller's estimated cars on each entry road, passed in as
    `entry_occ`); `perfect` mode is an oracle for the value-of-information ladder.
    """
    def __init__(self, mode: str = "live", W_s: float = 300.0, H: int = 3, rain_sigma: float = 0.10, seed: int = 0,
                 models_path: str = DEFAULT_MODELS):
        assert mode in ("live", "patterns", "persistence", "perfect")
        self.mode, self.W, self.H, self.rain_sigma = mode, float(W_s), H, rain_sigma
        self.seed, self.models_path = seed, models_path

    # ---------------------------------------------------------------- setup
    def attach(self, sim):
        """External feeds only: the clock, the weather feed, and (oracle mode) the true demand."""
        self.sim = sim
        dm = sim.demand
        self.start_hour = 12.0 if dm.start_hour is None else float(dm.start_hour)
        self.weekend = int(bool(dm.weekend))
        self.n_e = len(sim.net.entry_links)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(self.seed), 5]))
        # the learned models are fitted to the 2x2 grid's 8 entries; other grids use the model-free modes
        self.M = None if (self.mode == "perfect" or (self.mode == "persistence" and self.n_e != 8)) else load_models(self.models_path)
        self.rates, self.cnts = [], []
        self.queue = np.zeros(self.n_e)                 # entry_queue column: not measured (no cameras), not a model input
        self.prev_cum = np.zeros(self.n_e)
        self.next_boundary = self.W
        self.w_idx, self.t_w, self.rain_mean = 0, 0.0, 0.0
        self.F = None
        self._refresh()

    def rain_mean_now(self):
        """Rain intensity over the last window (the weather feed), used to scale the saturation flow."""
        return float(self.rain_mean)

    def _rain_between(self, t0, t1):
        f = self.sim.rain
        if f is None:
            return 0.0
        return float(np.mean([f(x) for x in np.arange(t0 + 15.0, t1, 30.0)]))

    # ---------------------------------------------------------------- once per window
    def update(self, t, obs, entry_occ=None):
        """obs: only entry_cum is read. entry_occ: estimated cars on each entry road (censoring guard input)."""
        while t + 1e-9 >= self.next_boundary:
            tb = self.next_boundary
            cum = np.asarray(obs["entry_cum"], dtype=float)
            self.rates.insert(0, (cum - self.prev_cum) / self.W)
            occ = np.zeros(self.n_e) if entry_occ is None else np.asarray(entry_occ, dtype=float)
            self.cnts.insert(0, occ.copy())
            self.rates, self.cnts = self.rates[:K_LAGS], self.cnts[:K_LAGS]
            self.prev_cum = cum.copy()
            self.rain_mean = self._rain_between(tb - self.W, tb)
            self.w_idx, self.t_w = self.w_idx + 1, tb
            self.next_boundary += self.W
            self._refresh()

    def _refresh(self):
        if self.mode == "perfect":
            return
        H, W_min = self.H, self.W / 60.0
        live = self.mode == "live" and len(self.rates) >= K_LAGS
        R = np.array(self.rates[:K_LAGS]) if live else np.zeros((K_LAGS, self.n_e))
        C = np.array(self.cnts[:K_LAGS]) if live else np.zeros((K_LAGS, self.n_e))
        if self.mode == "persistence" and self.rates:
            self.F = np.tile(self.rates[0], (H, 1))
            return
        if self.M is None:                                  # model-free start-up on a non-2x2 grid: flat prior
            self.F = np.full((H, self.n_e), 0.06)
            return
        # noisy weather forecast: the true mean rain in each future window plus noise that grows with lead time
        rf = [float(np.clip(self._rain_between(self.t_w + (h - 1) * self.W, self.t_w + h * self.W)
                            + self.rng.normal(0, self.rain_sigma * np.sqrt(h)), 0, 1)) for h in range(1, H + 1)]
        rows = online_rows(self.w_idx, self.start_hour + self.t_w / 3600.0, self.weekend, R, C, self.queue,
                           self.rain_mean, rf, W_min, H)
        model = self.M["poisson_glm+live+guard"] if live else self.M["profile_x_rain"]
        self.F = np.maximum(model.predict(rows), 0.0).reshape(H, self.n_e)       # veh/s per entry, per window ahead

    # ---------------------------------------------------------------- for the optimiser
    def lam_steps(self, t, step_s, N):
        """Expected arrival rate (veh/s) at each entry for N steps of step_s starting at t. Shape (n_entries, N)."""
        out = np.zeros((self.n_e, N))
        for k in range(N):
            tau = t + (k + 0.5) * step_s
            if self.mode == "perfect":
                out[:, k] = np.mean([self.sim.demand.rates(tau + dx) for dx in (-step_s / 4, 0.0, step_s / 4)], axis=0)
            else:
                h = int(np.clip(np.floor((tau - self.t_w) / self.W) + 1, 1, self.H))
                out[:, k] = self.F[h - 1]
        return out
