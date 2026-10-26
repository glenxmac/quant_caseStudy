import numpy as np

from traffic.forecasting.history import BIN_S, simulate_day


def test_history_schema_and_sanity():
    df = simulate_day(3, hours=0.5, start_hour=7.0)
    assert len(df) == (1800 // BIN_S - 1 + 1) * 8 or len(df) % 8 == 0
    assert df.entry.nunique() == 8
    assert (df.true_rate > 0).all() and (df.observed >= 0).all()
    assert (df.observed <= df.created + df.backlog.max() + 1).all()
    # sensor sees at most what demand created (cars wait outside when the road is full)
    assert df.observed.sum() <= df.created.sum() + 1e-9
    # Poisson sanity: summed observed ~ summed true demand over the day (light load, no censoring)
    exp = (df.true_rate * BIN_S).sum()
    assert abs(df.created.sum() - exp) < 5 * np.sqrt(exp)
