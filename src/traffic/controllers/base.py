"""Controller interface. Every controller (fixed-time or the optimiser) implements this,
so the experiment harness treats them identically.

observations (dict of numpy arrays), all a city could measure without cameras:
  entry_cum[e]      cumulative cars that entered each entry road (entry counter)
  phase[i], t_in_phase[i], switching[i], t      own signal state and the clock
  incidents         incident feed: [(link, pos_frac, start_s, end_s or None)], reported once started;
                    end_s only once cleared
  stop_cross[l]     simulator truth (stop-line counts per link), for tests and diagnosis only
"""
from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Base class: `reset` once per run, then `act(t, obs)` returns the requested phase (0=NS, 1=EW) per intersection."""
    def reset(self, cfg, net, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.net = net
        self.rng = rng

    @abstractmethod
    def act(self, t: float, obs: dict) -> np.ndarray:
        """Return the requested phase (0 or 1) for every intersection, shape (n_int,).
        The simulator enforces min green and lost (all-red) time on switches."""
