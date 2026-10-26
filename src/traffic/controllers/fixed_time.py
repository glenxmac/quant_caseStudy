"""Baseline: the fixed-time green wave (the thing we must beat by 20%+)."""
import numpy as np

from .base import Controller
from ..config import VehicleConfig


class FixedTimeSynchronised(Controller):
    """Fixed cycle with a green wave. cycle_s: full cycle (both phases + lost time). split: share requested
    for phase 0 (NS). The sim takes lost time out of each phase, so effective green = cycle*share - lost_time.
    The offset grows with distance from the grid's top-left corner by the free-flow link travel time, so
    platoons released heading E or S meet greens."""

    def __init__(self, cycle_s: float = 60.0, split: float = 0.5):
        self.cycle_s = cycle_s
        self.split = split

    def reset(self, cfg, net, rng):
        super().reset(cfg, net, rng)
        tt = cfg.free_flow_link_s(VehicleConfig())
        self.offsets = (net.int_rc.sum(axis=1) * tt) % self.cycle_s

    def act(self, t, obs):
        x = (t + self.offsets) % self.cycle_s
        return np.where(x < self.cycle_s * self.split, 0, 1)
