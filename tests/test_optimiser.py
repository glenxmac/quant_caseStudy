import numpy as np
import pytest

from traffic.config import SimConfig
from traffic.controllers.optimiser import link_flow_matrix, smooth_cycle, solve_splits, webster_cycle
from traffic.network import build_grid

CFG = SimConfig()


def _setup():
    net = build_grid(2, 2)
    T = link_flow_matrix(net, CFG.turn_probs)
    storage = np.full(net.n_links, 30.0)
    p = {"step_s": 60.0, "sat": 0.4, "lost_s": CFG.lost_time_s, "min_green_s": CFG.min_green_s,
         "max_green_s": CFG.max_green_s, "quad_w": 3.0, "term_w": 3.0, "spill_w": 5.0, "switch_w": 2.0}
    return net, T, storage, p


def _lam(net, per_heading):
    h = net.link_heading[net.entry_links]
    return np.array([[per_heading[int(k)]] * 5 for k in h])


def test_balanced_demand_gives_even_split():
    net, T, storage, p = _setup()
    x, _, st = solve_splits(np.zeros(net.n_links), _lam(net, [0.1] * 4), np.full(net.n_int, 0.5), 80.0, net, T, storage, p)
    assert x is not None and np.allclose(x, 0.5, atol=0.08)


def test_heavy_ns_gets_more_green_and_bounds_hold():
    net, T, storage, p = _setup()
    lam = _lam(net, [0.25, 0.05, 0.25, 0.05])        # headings N,E,S,W
    x, _, _ = solve_splits(np.zeros(net.n_links), lam, np.full(net.n_int, 0.5), 80.0, net, T, storage, p)
    assert (x > 0.5).all()
    C = 80.0
    assert (x * C - p["lost_s"] >= p["min_green_s"] - 1e-3).all()
    assert ((1 - x) * C - p["lost_s"] >= p["min_green_s"] - 1e-3).all()


def test_smoothed_cycle_reaches_its_target_and_both_caps():
    """Rounding the smoothed value itself used to stall the cycle at 115 s and 45 s (never 120 / 40)."""
    for start, target in ((60.0, 120.0), (115.0, 120.0), (60.0, 40.0), (45.0, 40.0), (100.0, 110.0)):
        s, c = start, start
        for _ in range(30):
            s, c = smooth_cycle(s, target, 40.0, 120.0)
        assert c == target


def test_webster_cycle_grows_with_demand_and_is_clamped():
    net, T, _, _ = _setup()
    lo, _ = webster_cycle(net, T, np.full(8, 0.03), 0.4, 12.0, 40.0, 120.0)
    hi, _ = webster_cycle(net, T, np.full(8, 0.12), 0.4, 12.0, 40.0, 120.0)
    assert 40.0 <= lo <= hi <= 120.0


def test_optimiser_runs_end_to_end_and_respects_signal_limits():
    """Short closed-loop run inside the simulator: no solver failures, valid phases, completes."""
    import os
    import pytest
    from dataclasses import replace
    from traffic.controllers.optimiser import ForecastOptimiser
    from traffic.scenarios import SCENARIOS, build_sim
    if not os.path.exists("outputs/.cache/final_5.0.pkl"):
        pytest.skip("needs fitted forecaster: python main.py forecast --final")
    scen = replace(SCENARIOS["am_ramp"], duration_s=900)
    ctrl = ForecastOptimiser(mode="live")
    sim = build_sim(scen, SimConfig(warmup_s=0), ctrl, seed=1)
    out = sim.run()
    assert ctrl.n_solves >= 10 and ctrl.n_fail == 0
    assert out["exited"] > 0
    assert all(((x >= 0) & (x <= 1)).all() for _, _, x in [(t, C, np.atleast_1d(xx)) for t, C, xx in ctrl.plans])


def test_blocked_link_loses_green():
    """Intersection 0's NS approaches are reported blocked and the cross street has a real load: that intersection
    moves to its lower bound on NS share; intersections 1 and 3 (not downstream of it) barely change."""
    net, T, storage, p = _setup()
    q0 = np.full(net.n_links, 10.0)
    lam = _lam(net, [0.1, 0.2, 0.1, 0.2])                # headings N,E,S,W
    x_ref, _, _ = solve_splits(q0, lam, np.full(net.n_int, 0.5), 80.0, net, T, storage, p)
    blocked = np.zeros(net.n_links, dtype=bool)
    blocked[[0 * 4 + 0, 0 * 4 + 2]] = True               # intersection 0, headings N and S
    x, _, _ = solve_splits(q0, lam, np.full(net.n_int, 0.5), 80.0, net, T, storage, p, blocked)
    x_lo = max((p["min_green_s"] + p["lost_s"]) / 80.0, 1 - (p["max_green_s"] + p["lost_s"]) / 80.0)   # EW max green binds
    assert x[0, 0] < x_ref[0, 0] - 0.1 and x[0, 0] == pytest.approx(x_lo, abs=0.01)
    assert np.allclose(x[[1, 3], 0], x_ref[[1, 3], 0], atol=0.05)


def test_estimator_conserves_cars_and_holds_them_on_red():
    """All red: every counted car stays on its entry road. Then green everywhere: cars move on and leave, and the
    estimate never goes negative or above storage."""
    from traffic.controllers.optimiser import QueueEstimator
    net, T, _, _ = _setup()
    storage = np.full(net.n_links, 30.0); storage[net.entry_links] = 61.0
    est = QueueEstimator(net, CFG, T, storage)
    no = np.zeros(net.n_links, dtype=bool)
    cum = np.zeros(len(net.entry_links))
    for t in range(1, 121):
        cum += 0.2
        est.step(float(t), cum, no, np.zeros(net.n_links), 0.4, no)
    assert np.allclose(est.n[net.entry_links], cum) and est.n.sum() == pytest.approx(cum.sum())
    for t in range(121, 721):
        est.step(float(t), cum, ~no, np.full(net.n_links, 30.0), 0.4, no)
    assert est.n.sum() < 1.0 and (est.n >= -1e-9).all() and (est.n <= storage + 1e-9).all()


def _allowed_obs(obs):
    """What the counts-only controller may see: entry counter, clock, own signal state, incident feed."""
    return {k: obs[k] for k in ("entry_cum", "phase", "t_in_phase", "switching", "t", "incidents")}


def test_optimiser_needs_only_counts_clock_signals_and_incident_feed():
    """Closed loop with an accident, the controller handed ONLY the allowed keys (any camera read would raise
    KeyError). The incident feed marks the road blocked only while it is, and the estimate tracks the truth."""
    from dataclasses import replace
    from traffic.controllers.optimiser import ForecastOptimiser
    from traffic.scenarios import SCENARIOS, Accident, build_sim

    class CountsOnly(ForecastOptimiser):
        def act(self, t, obs):
            req = super().act(t, _allowed_obs(obs))
            if int(t) % 60 == 0:
                self.trace.append((t, self.blocked[5], np.abs(self.est.n - np.bincount(self.sim.link, minlength=16)).mean()))
            return req

    scen = replace(SCENARIOS["am_ramp"], start_hour=7.0, duration_s=1800,
                   accidents=(Accident(link=5, start_s=600, end_s=900),))
    ctrl = CountsOnly(mode="perfect")
    ctrl.trace = []
    sim = build_sim(scen, SimConfig(warmup_s=0), ctrl, seed=104)
    out = sim.run()
    t, blk, err = (np.array(v, dtype=float) for v in zip(*ctrl.trace))
    assert ctrl.n_solves >= 25 and ctrl.n_fail == 0 and out["exited"] > 0
    assert not blk[t < 600].any() and blk[(t >= 600) & (t < 900)].all() and not blk[t >= 900].any()
    assert err.mean() < 3.0                                   # mean abs error, cars per link


def test_incident_feed_reports_end_only_after_clearance():
    from traffic.controllers.fixed_time import FixedTimeSynchronised
    from traffic.scenarios import SCENARIOS, Accident, build_sim
    from dataclasses import replace
    scen = replace(SCENARIOS["am_ramp"], duration_s=60, accidents=(Accident(link=5, start_s=20, end_s=40),))
    sim = build_sim(scen, SimConfig(warmup_s=0), FixedTimeSynchronised(), seed=1)
    feeds = {}
    for _ in range(120):
        feeds[sim.t] = sim.observe()["incidents"]
        sim.step()
    assert feeds[10.0] == [] and feeds[30.0] == [(5, 0.5, 20.0, None)] and feeds[50.0] == [(5, 0.5, 20.0, 40.0)]


def test_incident_fallback_runs_fixed_plan_only_in_window():
    """From the incident report until fallback_s after clearance the optimiser runs the fixed green wave at
    fallback_cycle_s (same phase requests as FixedTimeSynchronised); before and after it runs its own plan."""
    from dataclasses import replace
    from traffic.controllers.fixed_time import FixedTimeSynchronised
    from traffic.controllers.optimiser import ForecastOptimiser
    from traffic.scenarios import SCENARIOS, Accident, build_sim

    class Traced(ForecastOptimiser):
        def act(self, t, obs):
            req = super().act(t, _allowed_obs(obs))
            self.trace.append((t, self.in_fallback, req.copy()))
            return req

    scen = replace(SCENARIOS["am_ramp"], start_hour=7.0, duration_s=900,
                   accidents=(Accident(link=5, start_s=300, end_s=450),))
    ctrl = Traced(mode="perfect", fallback_s=150.0, fallback_cycle_s=120.0)
    ctrl.trace = []
    sim = build_sim(scen, SimConfig(warmup_s=0), ctrl, seed=104)
    sim.run()
    ref = FixedTimeSynchronised(cycle_s=120.0)
    ref.reset(sim.cfg, sim.net, np.random.default_rng(0))
    t, on = np.array([r[0] for r in ctrl.trace]), np.array([r[1] for r in ctrl.trace])
    assert not on[t < 300].any() and on[(t >= 300) & (t < 600)].all() and not on[t >= 600].any()
    for tt, o, req in ctrl.trace:
        if o:
            assert (req == ref.act(tt, None)).all()
    assert ForecastOptimiser().fallback_s > 0                 # on by default
