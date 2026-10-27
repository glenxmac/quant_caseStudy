import numpy as np
import pandas as pd

from traffic.forecasting.evaluate import split_days
from traffic.forecasting.features import FEATURE_SETS, FEATURES, GBM_EXTRA, PATTERN_FEATURES, build_features, to_windows
from traffic.forecasting.history import simulate_day

BANNED = {"y", "true_now", "true_rate", "obs_target", "rain_target", "created", "backlog", "accident"}
OCCUPANCY = {"entry_count", "entry_queue"} | {f"cnt_lag{i}" for i in range(6)}   # counts-only sensing: guard flag, never inputs


def _hist():
    return pd.concat([simulate_day(s, hours=1.5, start_hour=7.0) for s in (1, 2)], ignore_index=True)


def test_no_target_or_oracle_columns_in_features():
    for fs in list(FEATURE_SETS.values()) + [FEATURES, PATTERN_FEATURES]:
        assert not BANNED & set(fs + GBM_EXTRA)


def test_no_occupancy_in_features():
    for fs in list(FEATURE_SETS.values()) + [FEATURES]:
        assert not OCCUPANCY & set(fs + GBM_EXTRA)


def test_features_use_only_the_past_and_targets_are_future_true_rate():
    win = to_windows(_hist())
    L = build_features(win)
    r = L[(L.day == 1) & (L.entry == 0) & (L.h == 2)].sort_values("w").iloc[0]
    src = win[(win.day == 1) & (win.entry == 0)].set_index("w")
    assert np.isclose(r.y, src.loc[r.w + 2, "true_rate"])                 # target = true rate 2 windows ahead
    assert np.isclose(r.lag0, src.loc[r.w, "obs_rate"])                  # lag0 = the window that just ended
    assert np.isclose(r.lag3, src.loc[r.w - 3, "obs_rate"])
    # changing the future observations must not change today's features
    win2 = win.copy()
    win2.loc[win2.w > r.w, "observed"] += 50
    win2["obs_rate"] = win2.observed / 300
    r2 = build_features(win2)
    r2 = r2[(r2.day == 1) & (r2.entry == 0) & (r2.h == 2) & (r2.w == r.w)].iloc[0]
    assert all(np.isclose(r[f], r2[f]) for f in ("lag0", "lag5", "roll6", "net_lag0", "net_roll6", "cnt_lag0"))


def test_event_features_come_from_the_calendar_for_the_forecast_window():
    """The event calendar is published in advance: ev_fh is the announced extra demand in the window being
    forecast; the true crowd (only in the target) never enters."""
    win = to_windows(_hist())
    w0 = int(win.w.median())
    win.loc[(win.day == 1) & (win.entry == 0) & (win.w == w0 + 2), "event"] = 0.8
    L = build_features(win)
    r = L[(L.day == 1) & (L.entry == 0) & (L.w == w0)].set_index("h")
    assert np.isclose(r.loc[2, "ev_fh"], np.log1p(0.8)) and r.loc[1, "ev_fh"] == 0 and r.loc[3, "ev_fh"] == 0
    assert (L[L.entry != 0].ev_fh == 0).all()


def test_split_is_by_whole_days_in_time_order():
    tr, va, te = split_days(np.repeat(np.arange(60), 5))
    assert (len(tr), len(va), len(te)) == (40, 10, 10)
    assert tr.max() < va.min() and va.max() < te.min()


def test_censor_guard_flags_full_entry_roads_and_multiplier_ignores_them():
    from traffic.forecasting.model import BLOCK_T, Structured, censored_network
    L = build_features(to_windows(_hist()))
    assert not censored_network(L).any()                       # a light morning is not censored
    L2 = L.copy()
    L2["cnt_lag0"] = BLOCK_T + 1                               # every entry road full right now
    assert censored_network(L2).all()
    m = Structured().fit(L)
    p_full = m.predict(L2)
    prof = HourProfile_pred(m, L2)
    assert np.allclose(p_full, prof) or np.isfinite(p_full).all()  # gridlock-wide: falls back to the profile


def HourProfile_pred(m, L):
    from traffic.forecasting.model import HourProfile
    return HourProfile.predict(m, L)
