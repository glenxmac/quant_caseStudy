"""Train / validate / test the forecaster ladder.

Split by WHOLE DAYS in calendar order (a rolling-origin style split: train on the past, evaluate on the
future), never by rows, because neighbouring windows of one day share the same day effect, storm and
accident, so a row split leaks. Hyper-parameters are chosen on validation; the test days are touched
once (`final=True`), after refitting on train + validation.
"""
import numpy as np
import pandas as pd

from .features import FEATURE_SETS, LIVE_FEATURES, PATTERN_FEATURES, build_features, to_windows
from .model import (CensorGuard, GradientBoosting, HourProfile, Persistence, PoissonGLM, ProfileRain, Structured,
                    censored_network)


def split_days(days, frac=(40 / 60, 10 / 60, 10 / 60)):
    d = np.sort(np.unique(days))
    a, b = int(round(len(d) * frac[0])), int(round(len(d) * (frac[0] + frac[1])))
    return d[:a], d[a:b], d[b:]


def regime(L):
    r = np.where(L.accident > 0, "accident", np.where(np.maximum(L.rain_now, L.rain_target) > 0.1, "rain", "normal"))
    return pd.Series(r, index=L.index)


def band(L):
    h = L.thr_hour
    return pd.Series(np.select([(h >= 6.5) & (h < 9.5), (h >= 15.5) & (h < 18.5)], ["AM peak", "PM peak"], "off-peak"),
                     index=L.index)


def metrics(y, p, ref):
    e = p - y
    return {"rMAE_%": 100 * np.abs(e).mean() / y.mean(), "RMSE_vph": 3600 * np.sqrt((e ** 2).mean()),
            "skill_vs_profile": 1 - (e ** 2).mean() / ((ref - y) ** 2).mean(), "n": len(y)}


def candidates(features, tag):
    return {"glm": [PoissonGLM(a, features, tag) for a in (1e-4, 1e-2, 1.0)],
            "gbm": [GradientBoosting(i, l, features=features, tag=tag) for i in (150, 300) for l in (15, 31)]}


def run(hist: pd.DataFrame, W_min: float = 5.0, H: int = 3, final: bool = False, verbose: bool = True):
    """Fit and score the ladder: patterns-only models (calendar + weather) and models that also read live sensor
    counts ('+live'; the main forecaster is the GLM + live counts + censoring guard). Labels = true rate (an
    accurate historical survey); scored against the true rate AND against what the sensor counts."""
    import copy
    L = build_features(to_windows(hist, W_min), W_min, H)
    tr, va, te = split_days(L.day)
    A = L[L.day.isin(np.concatenate([tr, va]) if final else tr)]
    B = L[L.day.isin(te if final else va)]
    sel = L[L.day.isin(va)]
    ftr = L[L.day.isin(tr)]
    chosen = {}
    for key, feats, tag in (("pat", PATTERN_FEATURES, ""), ("live", LIVE_FEATURES, "+live")):
        for fam, cs in candidates(feats, tag).items():           # model selection on validation only
            chosen[(key, fam)] = min(cs, key=lambda m: ((m.fit(ftr, W_min).predict(sel) - sel.y) ** 2).mean())
    models = [HourProfile(), ProfileRain(), chosen[("pat", "glm")], chosen[("pat", "gbm")],
              Persistence(), Structured(), chosen[("live", "glm")], chosen[("live", "gbm")],
              CensorGuard(copy.deepcopy(chosen[("live", "glm")])), CensorGuard(copy.deepcopy(chosen[("live", "gbm")]))]
    preds = {m.name: m.fit(A, W_min).predict(B) for m in models}
    preds["oracle_true_rate_now"] = B.true_now.to_numpy()      # unreachable benchmark: knows today's rate exactly
    y, ref, obs = B.y.to_numpy(), preds["hour_profile"], B.obs_target.to_numpy()
    out = []
    for name, p in preds.items():
        for grp, key in (("all", pd.Series("all", index=B.index)), ("horizon", B.h), ("regime", regime(B)), ("time", band(B))):
            for k in sorted(key.unique(), key=str):
                m = (key == k).to_numpy()
                out.append({"model": name, "group": grp, "level": k, **metrics(y[m], p[m], ref[m]),
                            "rMAE_vs_sensor_%": 100 * np.abs(p[m] - obs[m]).mean() / obs[m].mean()})
    res = pd.DataFrame(out)
    nf = float(np.mean(1 / np.sqrt(np.maximum(B.y.to_numpy() * W_min * 60, 1e-9))))
    info = {"share_rows_guarded_%": 100 * float(censored_network(B).mean()), "split_days": (len(tr), len(va), len(te)),
            "noise_floor_rel_%": 100 * nf, "glm_alpha": chosen[("pat", "glm")].alpha,
            "gbm": (chosen[("pat", "gbm")].kw["max_iter"], chosen[("pat", "gbm")].kw["max_leaf_nodes"]),
            "scored_on": "TEST" if final else "VALIDATION"}
    return res, info, models, B, preds


def ablation(hist: pd.DataFrame, others, W_min: float = 5.0, H: int = 3):
    """Feature-group ladder: what does each kind of information add? Same GLM, fit on train+val days, scored on
    held-out days (the test days plus any extra histories in `others`). Live-count tiers get the censoring guard."""
    L = build_features(to_windows(hist, W_min), W_min, H)
    tr, va, te = split_days(L.day)
    fit = L[L.day.isin(np.concatenate([tr, va]))]
    T = pd.concat([L[L.day.isin(te)]] + [build_features(to_windows(h, W_min), W_min, H) for h in others], ignore_index=True)
    ref = HourProfile().fit(fit, W_min).predict(T)
    y, obs = T.y.to_numpy(), T.obs_target.to_numpy()
    rows = [{"tier": "hour-of-day profile (weekday/weekend)", **metrics(y, ref, ref), "rMAE_vs_sensor_%": 100 * np.abs(ref - obs).mean() / obs.mean()}]
    pr = ProfileRain().fit(fit, W_min).predict(T)
    rows.append({"tier": "profile x rain effect (2 parameters)", **metrics(y, pr, ref), "rMAE_vs_sensor_%": 100 * np.abs(pr - obs).mean() / obs.mean()})
    for label, key in (("calendar only", "calendar"), ("+ weather", "calendar+weather"),
                       ("+ live entrance counts", "+live counts")):
        m = PoissonGLM(1e-4, FEATURE_SETS[key])
        m = CensorGuard(m) if key.startswith("+") else m
        p = m.fit(fit, W_min).predict(T)
        rows.append({"tier": label, **metrics(y, p, ref), "rMAE_vs_sensor_%": 100 * np.abs(p - obs).mean() / obs.mean()})
    return pd.DataFrame(rows)


LADDER = [("persistence", "Raw count carried forward (persistence)"),
          ("hour_profile", "Hour-of-day profile (entry x weekday/weekend x 15 min)"),
          ("profile_x_rain", "+ weather (profile x rain)"),
          ("poisson_glm+live+guard", "+ live entrance counts: Poisson GLM + guard (main)"),
          ("gbm_poisson+live+guard", "GBM + live counts + guard (check)"),
          ("oracle_true_rate_now", "Oracle: true rate now carried forward (unreachable)")]


def ladder_by_horizon(models, L, W_min=5.0):
    """Error (MAE as % of the mean true rate) by forecast horizon for the LADDER rows; `models` already fitted."""
    M = {m.name: m for m in models}
    y = L.y.to_numpy()
    rows = []
    for name, label in LADDER:
        p = L.true_now.to_numpy() if name.startswith("oracle") else M[name].predict(L)
        r = {"model": name, "label": label}
        for h in sorted(L.h.unique()):
            m = (L.h == h).to_numpy()
            r[f"+{h * W_min:g} min"] = 100 * np.abs(p[m] - y[m]).mean() / y[m].mean()
        r["overall"] = 100 * np.abs(p - y).mean() / y.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def paired_by_day(B, preds, ref="hour_profile", other=None):
    """Per-day squared error for each model, compared with a reference model on the SAME days.
    Reports the mean % reduction in MSE across days with a 95% t-interval (days are the unit, because
    windows within a day are not independent)."""
    from scipy import stats
    y = B.y.to_numpy()
    sq = {n: pd.Series((p - y) ** 2).groupby(B.day.to_numpy()).mean() for n, p in preds.items()}
    rows = []
    for n in sq:
        for r in ([ref] if other is None else other):
            if n == r:
                continue
            d = 100 * (1 - sq[n] / sq[r])
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"model": n, "vs": r, "mse_reduction_%": d.mean(), "ci95_+-": h, "days": len(d)})
    return pd.DataFrame(rows)


def robustness(models, hist_other: pd.DataFrame, W_min: float = 5.0, H: int = 3):
    """Score already-fitted models on days simulated under a DIFFERENT controller (covariate shift:
    entry-road occupancy and censoring depend on the signal plan that generated the data)."""
    L = build_features(to_windows(hist_other, W_min), W_min, H)
    ref = next(m for m in models if m.name == "hour_profile").predict(L)
    out = []
    for m in models:
        out.append({"model": m.name, **metrics(L.y.to_numpy(), m.predict(L), ref)})
    out.append({"model": "oracle_true_rate_now", **metrics(L.y.to_numpy(), L.true_now.to_numpy(), ref)})
    return pd.DataFrame(out), L
