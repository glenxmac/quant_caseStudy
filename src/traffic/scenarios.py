"""Scenarios: a time window (start hour, duration) with a demand pattern, rain and accidents.

The same scenario + seed gives identical demand for every controller (common random numbers).
`nominal_of` maps a disturbed scenario to the typical-day scenario that baseline plans are
tuned on, mirroring practice: plans are set for a normal day, then the day happens.
"""
from dataclasses import dataclass, replace

import numpy as np

from .demand import DemandModel, Event
from .network import build_grid
from .sim import Simulation


@dataclass(frozen=True)
class Accident:
    """Blocks the (single) lane of `link` at pos_frac of its length from start_s to end_s."""
    link: int
    start_s: float
    end_s: float
    pos_frac: float = 0.5


@dataclass(frozen=True)
class Scenario:
    """A time window of a day (start hour, duration) with demand pattern, rain and accidents."""
    name: str
    start_hour: float
    duration_s: int
    weekend: bool = False
    day_effect: float = 1.0
    noise_sigma: float = 0.06
    rain: tuple = ()                 # ((start_s, end_s, peak_intensity), ...)
    accidents: tuple = ()
    events: tuple = ()               # scheduled events (demand.Event), published in advance by the event calendar
    time_shift_h: float = 0.0        # day-specific shift of the whole daily pattern (hours)
    scale: float = 0.85              # nominal day peaks at ~0.85 of the saturation knee: busy but manageable
    description: str = ""

    def rain_fn(self):
        if not self.rain:
            return None
        spans = self.rain

        def f(t):
            r = 0.0
            for s, e, peak in spans:
                r = max(r, peak * float(np.clip(min((t - s) / 300.0, (e - t) / 300.0), 0.0, 1.0)))
            return r
        return f

    def demand(self, net, seed: int) -> DemandModel:
        return DemandModel(len(net.entry_links), scale=self.scale, entry_heading=net.link_heading[net.entry_links],
                           start_hour=self.start_hour, weekend=self.weekend, day_effect=self.day_effect,
                           noise_sigma=self.noise_sigma, rain=self.rain_fn(), seed=seed, horizon_s=self.duration_s,
                           time_shift_h=self.time_shift_h, events=self.events)


H = 3600
SCENARIOS = {s.name: s for s in [
    Scenario("am_ramp", 6.0, 3 * H, description="Typical weekday morning: 06:00-09:00, ramp to the peak and ease off"),
    Scenario("am_busy", 6.0, 3 * H, day_effect=1.15, description="Same morning but 15% busier than the plan assumes"),
    Scenario("am_rain", 6.0, 3 * H, rain=((1.25 * H, 2.5 * H, 0.8),),
             description="Storm from 07:15 to 08:30 across the peak: slower driving, a little more demand"),
    Scenario("am_accident", 6.0, 3 * H, accidents=(Accident(link=5, start_s=1.7 * H, end_s=1.7 * H + 900),),
             description="Lane blocked on an internal road at 07:42 for 15 minutes, at peak"),
    Scenario("pm_ramp", 15.5, 3 * H, description="Typical weekday evening: 15:30-18:30, reverse tide"),
    Scenario("pm_event", 15.5, 3 * H, events=(Event(heading=1, start_s=0.75 * H, end_s=1.5 * H, size=1.0, announced=1.0),),
             description="Typical evening plus a stadium letting out at 16:15 on the west side: the two eastbound "
                         "entries double for 45 minutes, announced in advance by the event calendar"),
]}

NOMINAL = {"am_busy": "am_ramp", "am_rain": "am_ramp", "am_accident": "am_ramp", "pm_event": "pm_ramp"}


def nominal_of(name: str) -> str:
    return NOMINAL.get(name, name)


def build_sim(scenario: Scenario, cfg, controller, seed: int, **kw) -> Simulation:
    """Simulation for a scenario. Same scenario + seed => identical demand for every controller."""
    net = build_grid(cfg.grid_rows, cfg.grid_cols)
    cfg = replace(cfg, horizon_s=scenario.duration_s)
    sim = Simulation(cfg, controller, seed, demand=scenario.demand(net, seed), rain=scenario.rain_fn(),
                     events=list(scenario.accidents), **kw)
    if hasattr(controller, "attach"):
        controller.attach(sim)        # external feeds only: clock, weather (and the true demand for the oracle forecaster)
    return sim

