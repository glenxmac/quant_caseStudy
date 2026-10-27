"""Demand generation: time-of-day profile x tide x day effect x weather x noise x scheduled events.

Arrivals at each entry are Poisson with this time-varying rate. Kept separate from the
simulator so the forecaster can be trained on history from the same hidden process.

scale 1.0 = the saturation knee of the 2x2 grid under the fixed-time baseline
(measured: served throughput plateaus ~4550 veh/h; delay/spillback blow up above
~0.16 veh/s per entry). The weekday AM peak reaches ~1.0 and the PM peak ~1.1.
"""
from dataclasses import dataclass

import numpy as np

DEFAULT_BASE_RATE = 0.16
S_OR_E_HEADING = (False, True, True, False)      # N, E, S, W: entries whose cars head S or E


@dataclass(frozen=True)
class Event:
    """A scheduled event letting out (stadium, concert): extra demand on the entries whose cars head `heading`
    (the side of the grid nearest the venue). The city's event calendar publishes the time and the expected
    extra demand `announced` (share of normal, e.g. 1.0 = double); the real crowd adds `size`. The surge ramps
    up over ramp_s, holds, and tails off over 2 x ramp_s."""
    heading: int
    start_s: float
    end_s: float
    size: float
    announced: float
    ramp_s: float = 180.0

    def shape(self, t: float) -> float:
        return float(np.clip(min((t - self.start_s) / self.ramp_s, (self.end_s - t) / (2 * self.ramp_s)), 0.0, 1.0))


def hourly_profile(hours, weekend: bool = False) -> np.ndarray:
    """Relative demand multiplier by hour of day (1.0 ~ saturation knee)."""
    h = np.asarray(hours, dtype=float) % 24
    if weekend:
        return 0.15 + 0.50 * np.exp(-0.5 * ((h - 13.0) / 3.5) ** 2)
    return (0.18 + 0.30 * np.exp(-0.5 * ((h - 12.5) / 3.0) ** 2)
            + 0.75 * np.exp(-0.5 * ((h - 7.7) / 1.0) ** 2)
            + 0.82 * np.exp(-0.5 * ((h - 17.0) / 1.2) ** 2))


def tide(hours) -> np.ndarray:
    """+1 around the AM peak (inbound-heavy), -1 around the PM peak (outbound-heavy)."""
    h = np.asarray(hours, dtype=float) % 24
    return np.exp(-0.5 * ((h - 7.7) / 1.3) ** 2) - np.exp(-0.5 * ((h - 17.0) / 1.5) ** 2)


def _ar1(rng, n, sigma, phi=0.9):
    """Stationary AR(1) path with marginal std sigma (correlated over ~10 minutes at 1-min bins)."""
    e = rng.normal(0, 1, n) * sigma * np.sqrt(1 - phi ** 2)
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class DemandModel:
    """rates(t) -> arrival rate (veh/s) at each entry at sim time t (s)."""

    def __init__(self, n_entries: int, base_rate: float = DEFAULT_BASE_RATE, scale: float = 1.0,
                 profile=None, entry_heading=None, start_hour=None, weekend: bool = False,
                 day_effect: float = 1.0, noise_sigma: float = 0.0, rain=None,
                 rain_boost: float = 0.10, tidal: float = 0.25, seed: int = 0, horizon_s: float = 0.0,
                 time_shift_h: float = 0.0, events=()):
        self.n_entries, self.base_rate, self.scale = n_entries, base_rate, scale
        self.profile, self.start_hour, self.weekend = profile, start_hour, weekend
        self.day_effect, self.rain, self.rain_boost, self.tidal = day_effect, rain, rain_boost, tidal
        self.time_shift_h = time_shift_h        # today's rush arrives this many hours later than usual
        self.events = tuple(events)
        if entry_heading is None:
            self.sign = np.zeros(n_entries)
            self.heading = np.full(n_entries, -1)
        else:
            self.sign = np.where(np.array(S_OR_E_HEADING)[np.asarray(entry_heading)], 1.0, -1.0)
            self.heading = np.asarray(entry_heading)
        self.noise = None
        if noise_sigma > 0:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 99]))
            nb = int(horizon_s // 60) + 2
            shared = _ar1(rng, nb, noise_sigma)                       # whole-network wobble
            local = np.stack([_ar1(rng, nb, noise_sigma * 0.5) for _ in range(n_entries)], axis=1)
            self.noise = np.exp(shared[:, None] + local)

    def _hour(self, t):
        return None if self.start_hour is None else self.start_hour + t / 3600.0 - self.time_shift_h

    def multiplier(self, t: float) -> float:
        """Network-wide demand multiplier (before tide / noise / per-entry weights)."""
        h = self._hour(t)
        m = 1.0 if (h is None and self.profile is None) else (
            float(self.profile(t)) if self.profile is not None else float(hourly_profile(h, self.weekend)))
        r = float(self.rain(t)) if self.rain is not None else 0.0
        return m * self.day_effect * (1.0 + self.rain_boost * r)

    def rates(self, t: float) -> np.ndarray:
        h = self._hour(t)
        w = np.ones(self.n_entries) if h is None else 1.0 + self.tidal * float(tide(h)) * self.sign
        r = np.full(self.n_entries, self.base_rate * self.scale * self.multiplier(t)) * w
        if self.noise is not None:
            r = r * self.noise[min(int(t // 60), len(self.noise) - 1)]
        if self.events:
            r = r * (1.0 + self._event_extra(t, "size"))
        return r

    def _event_extra(self, t, attr):
        x = np.zeros(self.n_entries)
        for ev in self.events:
            s = ev.shape(t)
            if s > 0:
                x[self.heading == ev.heading] += getattr(ev, attr) * s
        return x

    def scheduled(self, t: float) -> np.ndarray:
        """The event calendar: expected extra demand at each entry at time t (announced size x shape), known in advance."""
        return self._event_extra(t, "announced")


def make_history(days: int, seed: int = 0, **kw):
    """Synthetic history for forecaster training (see forecasting/history.py)."""
    from .forecasting.history import make_history as _mh
    return _mh(days, seed, **kw)
