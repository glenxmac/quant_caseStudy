"""The forecaster ladder. Every model maps the long feature table (features.build_features) to a
predicted true arrival rate (veh/s) for one entry, h windows ahead. `y` is the TRUE rate, so models
are scored on what the optimiser needs, not on the noisy counts.

  (The main forecaster is PoissonGLM on live counts wrapped in CensorGuard, 'poisson_glm+live+guard' in online.py.
   Patterns-only models (calendar + weather) are its fallback and the 'patterns' rung; see README 'original mistake'.)
  Persistence       last window's observed rate                       (no learning)
  HourProfile       mean true rate at that clock time, by entry and weekday/weekend   (no day awareness)
  Structured        HourProfile x a day multiplier from the last 30 min of network-wide counts
  PoissonGLM        log-linear model on entrance-count lags, rain, clock harmonics
  GradientBoosting  trees with Poisson loss on the same inputs plus raw clock time
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import GBM_EXTRA, K_LAGS, PATTERN_FEATURES

SCALE = 3600.0         # fit in veh/h so Poisson deviances are well scaled
BLOCK_T = 55           # entry road >= ~90% full (capacity ~61 cars): the sensor then reports the blockage, not the demand


class Persistence:
    name = "persistence"

    def fit(self, L, W_min=5.0):
        return self

    def predict(self, L):
        return L.lag0.to_numpy()


class HourProfile:
    name = "hour_profile"

    def fit(self, L, W_min=5.0):
        self.tab = np.full((2, 8, 96), np.nan)
        b = (L.thr_hour.to_numpy() * 4).astype(int) % 96
        t = pd.DataFrame({"w": L.weekend.astype(int), "e": L.entry.astype(int), "b": b, "y": L.y})
        m = t.groupby(["w", "e", "b"]).y.mean()
        for (w, e, bb), v in m.items():
            self.tab[w, e, bb] = v
        for w in range(2):                                   # fill unseen bins: nearest seen bin, else global mean
            for e in range(8):
                s = pd.Series(self.tab[w, e]).interpolate(limit_direction="both")
                self.tab[w, e] = s.fillna(L.y.mean()).to_numpy()
        self.W_min = W_min
        return self

    def lookup(self, weekend, entry, hour):
        b = (np.asarray(hour) * 4).astype(int) % 96
        return self.tab[np.asarray(weekend).astype(int), np.asarray(entry).astype(int), b]

    def predict(self, L):
        return self.lookup(L.weekend, L.entry, L.thr_hour)


class Structured(HourProfile):
    """Profile x multiplier x event calendar. The multiplier is observed / expected summed over the last K windows
    and all 8 entries (about 2,000 cars, so it is a stable estimate of 'how busy is today vs a typical day'); the
    expected count includes any announced event, so an event does not inflate the multiplier.
    Only uncensored (entry, window) pairs enter it: an entry road that was full says nothing about demand.
    If under 25% of the profile mass is uncensored (network gridlock), the multiplier falls back to 1."""
    name = "profile_x_day_multiplier"

    def predict(self, L):
        obs = np.zeros(len(L)); prof = np.zeros(len(L)); prof_all = np.zeros(len(L))
        for i in range(K_LAGS):
            hour_i = L.hour_dec.to_numpy() - (i + 0.5) * self.W_min / 60.0
            p_i = self.lookup(L.weekend, L.entry, hour_i) * (1.0 + L[f"ev_lag{i}"].to_numpy())
            ok = (L[f"cnt_lag{i}"].to_numpy() < BLOCK_T)
            obs += L[f"lag{i}"].to_numpy() * ok
            prof += p_i * ok
            prof_all += p_i
        t = pd.DataFrame({"day": L.day.to_numpy(), "w": L.w.to_numpy(), "h": L.h.to_numpy(), "o": obs, "p": prof, "a": prof_all})
        gs = t.groupby(["day", "w", "h"])
        so, sp, sa = gs.o.transform("sum"), gs.p.transform("sum"), gs.a.transform("sum")
        m = np.where(sp >= 0.25 * sa, np.clip(so / np.maximum(sp, 1e-12), 0.4, 2.5), 1.0)
        return super().predict(L) * m * np.exp(L.ev_fh.to_numpy())


def censored_network(L):
    """True when any entry road in the network was full during the last 3 windows, so the network-wide
    count features (and this entry's lags) are contaminated."""
    b = np.zeros(len(L), dtype=bool)
    for i in range(3):
        b |= L[f"cnt_lag{i}"].to_numpy() >= BLOCK_T
    return pd.Series(b).groupby([L.day.to_numpy(), L.w.to_numpy(), L.h.to_numpy()]).transform("max").to_numpy()


class CensorGuard:
    """Use the learned model unless the sensors are censored; then use profile x clean multiplier.
    The learned model is trained only on uncensored rows (it is never asked about the others)."""

    def __init__(self, base, fallback=None):
        self.base, self.fallback = base, fallback or Structured()
        self.name = base.name + "+guard"

    def fit(self, L, W_min=5.0):
        self.base.fit(L[~censored_network(L)], W_min)
        self.fallback.fit(L, W_min)
        return self

    def predict(self, L):
        g = censored_network(L)
        return np.where(g, self.fallback.predict(L), self.base.predict(L))


class ProfileRain(HourProfile):
    """Patterns-only, deliberately tiny: the hour profile times fitted rain and event effects,
    y = profile x exp(a + b * rain_forecast + c * event_calendar). Three parameters, so it cannot overfit the
    few rainy or event days."""
    name = "profile_x_rain"

    def _X(self, L):
        return np.column_stack([np.ones(len(L)), L.rain_fh.to_numpy(), L.ev_fh.to_numpy()])

    def fit(self, L, W_min=5.0):
        super().fit(L, W_min)
        r = np.log(np.maximum(L.y.to_numpy(), 1e-9) / np.maximum(super().predict(L), 1e-9))
        self.coef = np.linalg.lstsq(self._X(L), r, rcond=None)[0]
        return self

    def predict(self, L):
        return super().predict(L) * np.exp(self._X(L) @ self.coef)


class PoissonGLM:
    """Log-linear model. A plain additive version cannot express that each entry has its own busy hour
    (inbound vs outbound tide), so it also gets entry x clock-harmonic and weekend x clock-harmonic
    interactions. These are generic interactions, not the simulator's tide formula."""

    def __init__(self, alpha=1e-3, features=None, tag=""):
        self.alpha = alpha
        self.features = list(features or PATTERN_FEATURES)
        self.name = "poisson_glm" + tag

    def design(self, L):
        X = L[self.features].copy()
        harm = [f"thr_{t}{k}" for k in (1, 2, 3, 4) for t in "sc"]
        inter = {f"{c}_x_e{e}": L[c] * L[f"e{e}"] for c in harm for e in range(8)}
        inter.update({f"{c}_x_wkend": L[c] * L["weekend"] for c in harm})
        return pd.concat([X, pd.DataFrame(inter, index=L.index)], axis=1)

    def fit(self, L, W_min=5.0):
        self.m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=2000))
        self.m.fit(self.design(L), L.y * SCALE)
        return self

    def predict(self, L):
        return self.m.predict(self.design(L)) / SCALE


class GradientBoosting:
    def __init__(self, max_iter=300, max_leaf_nodes=31, learning_rate=0.06, features=None, tag=""):
        self.kw = dict(loss="poisson", max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                       learning_rate=learning_rate, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        self.cols = list(features or PATTERN_FEATURES) + GBM_EXTRA
        self.name = "gbm_poisson" + tag

    def fit(self, L, W_min=5.0):
        self.m = HistGradientBoostingRegressor(**self.kw).fit(L[self.cols], L.y * SCALE)
        return self

    def predict(self, L):
        return self.m.predict(L[self.cols]) / SCALE
