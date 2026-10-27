"""Random days: storms, accidents, scheduled events and a day-level demand effect, all drawn from stated
stochastic processes (assumptions, not calibrated to data). Used to build forecaster
history and to evaluate controllers on many different days.
"""
import numpy as np

from .demand import Event
from .scenarios import Accident, Scenario

SHIFT_SD_H = 0.20         # sd (hours) of the day-specific shift in when the rushes happen
DAY_SIGMA = 0.10          # lognormal sd of the day-level demand multiplier
STORMS_PER_DAY = 0.8      # Poisson mean storms per 24 h
AFTERNOON_SHARE = 0.7     # share of storms that start 13:00-19:00 (rest uniform over the day)
ACC_RATE_PER_H = 0.10     # network-wide accidents per hour when dry
ACC_RAIN_MULT = 3.0       # accident rate multiplier at full rain intensity (linear in intensity)
EVENTS_PER_DAY = 0.4      # Poisson mean scheduled events (stadium, concert) letting out per day, 11:00-20:00


def _rain_at(spans, t):
    r = 0.0
    for s, e, peak in spans:
        r = max(r, peak * float(np.clip(min((t - s) / 300.0, (e - t) / 300.0), 0.0, 1.0)))
    return r


def sample_day(seed: int, start_hour: float, duration_s: int, weekend: bool = False,
               n_links: int = 16, scale: float = 0.85) -> Scenario:
    """One random day, restricted to the window [start_hour, start_hour + duration]."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
    day_effect = float(np.exp(rng.normal(-0.5 * DAY_SIGMA ** 2, DAY_SIGMA)))
    shift = float(rng.normal(0, SHIFT_SD_H))
    spans = []
    for _ in range(rng.poisson(STORMS_PER_DAY)):
        start = rng.uniform(13, 19) if rng.random() < AFTERNOON_SHARE else rng.uniform(0, 24)
        dur_s = float(np.clip(rng.lognormal(np.log(50), 0.5), 15, 150)) * 60
        peak = 0.3 + 0.7 * rng.beta(2, 2)
        s_rel = (start - start_hour) * 3600
        if s_rel + dur_s > 0 and s_rel < duration_s:
            spans.append((s_rel, s_rel + dur_s, peak))
    spans = tuple(spans)
    accidents, t, lam = [], 0.0, ACC_RATE_PER_H * ACC_RAIN_MULT / 3600.0
    while True:                                   # thinning: candidates at the max rate
        t += rng.exponential(1 / lam)
        if t >= duration_s:
            break
        if rng.random() < (1 + (ACC_RAIN_MULT - 1) * _rain_at(spans, t)) / ACC_RAIN_MULT:
            dur = float(np.clip(rng.lognormal(np.log(12), 0.5), 5, 40)) * 60
            accidents.append(Accident(link=int(rng.integers(n_links)), start_s=t, end_s=t + dur,
                                      pos_frac=float(rng.uniform(0.3, 0.7))))
    # events: own random stream, so the storms, accidents and demand of every day are unchanged by them
    rng_ev = np.random.default_rng(np.random.SeedSequence([int(seed), 11]))
    events = []
    for _ in range(rng_ev.poisson(EVENTS_PER_DAY)):
        start = rng_ev.uniform(11, 20)
        dur_s = float(np.clip(rng_ev.lognormal(np.log(45), 0.4), 20, 90)) * 60
        announced = float(rng_ev.uniform(0.4, 1.2))                      # published extra demand (share of normal)
        size = announced * float(rng_ev.lognormal(0.0, 0.25))           # the crowd that actually turns up
        heading = int(rng_ev.integers(4))
        s_rel = (start - start_hour) * 3600
        if s_rel + dur_s > 0 and s_rel < duration_s:
            events.append(Event(heading=heading, start_s=s_rel, end_s=s_rel + dur_s, size=size, announced=announced))
    return Scenario(name=f"day_{seed}", start_hour=start_hour, duration_s=duration_s, weekend=weekend,
                    day_effect=day_effect, time_shift_h=shift, rain=spans, accidents=tuple(accidents),
                    events=tuple(events), scale=scale)
