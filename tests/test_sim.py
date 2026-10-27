"""Simulator validation: conservation, safety, common random numbers, calibration."""
import numpy as np
import pytest

from traffic.config import SimConfig
from traffic.controllers import REGISTRY
from traffic.controllers.base import Controller
from traffic.demand import DemandModel
from traffic.sim import Simulation


class HoldPhase(Controller):
    """Keep one phase green forever (for calibration)."""
    def __init__(self, phase=0):
        self.p = phase

    def act(self, t, obs):
        return np.full(self.net.n_int, self.p)


CONTROLLERS = {"fixed_60": ("fixed_sync", {}), "fixed_120": ("fixed_sync", {"cycle_s": 120.0}),
               "optimiser": ("forecast_opt", {"mode": "perfect"})}   # perfect mode: no fitted forecaster needed


def _make(label):
    name, params = CONTROLLERS[label]
    return REGISTRY[name](**params)


@pytest.mark.parametrize("label", list(CONTROLLERS))
def test_conservation_and_no_overlap(label):
    from dataclasses import replace
    from traffic.scenarios import SCENARIOS, build_sim
    cfg = SimConfig(warmup_s=60)
    scen = replace(SCENARIOS["am_busy"], start_hour=7.5, duration_s=600)      # near the peak, 15% above nominal
    r = build_sim(scen, cfg, _make(label), seed=3).run()
    assert r["created"] == r["exited"] + r["on_road"] + r["buffered"]   # in = out + inside
    assert r["min_gap_m"] >= 0.0                                        # no car overlaps another


def test_common_random_numbers_same_demand_for_every_controller():
    cfg = SimConfig(horizon_s=300, warmup_s=30)
    created = {n: Simulation(cfg, _make(n), seed=7).run()["created"] for n in ("fixed_60", "fixed_120")}
    created["hold"] = Simulation(cfg, HoldPhase(0), seed=7).run()["created"]
    assert len(set(created.values())) == 1


def test_same_seed_same_result():
    cfg = SimConfig(horizon_s=300, warmup_s=30)
    a = Simulation(cfg, _make("fixed_120"), seed=5).run()
    b = Simulation(cfg, _make("fixed_120"), seed=5).run()
    assert a == b


def test_fixed_plan_is_a_green_wave():
    """Each intersection runs the same cycle, offset by the free-flow travel time per grid step."""
    from traffic.config import VehicleConfig
    cfg = SimConfig(horizon_s=60, warmup_s=0)
    sim = Simulation(cfg, _make("fixed_120"), seed=0)
    ctrl, net, tt = sim.controller, sim.net, cfg.free_flow_link_s(VehicleConfig())
    assert np.allclose(ctrl.offsets, (net.int_rc.sum(axis=1) * tt) % 120.0)
    phases = np.array([ctrl.act(t, None) for t in np.arange(0.0, 240.0, 1.0)])
    assert np.allclose(phases.mean(0), 0.5, atol=0.01)                  # split 0.5: half the cycle each phase
    for i in range(net.n_int):                                          # shifting time by the offset aligns the plans
        shift = int(round(ctrl.offsets[i]))
        assert (np.roll(phases[:, i], shift) == phases[:, 0]).mean() > 0.97


class RedThenGreen(Controller):
    """EW green until t_release, then NS green: builds a standing NS queue, then releases it."""
    def __init__(self, t_release):
        self.t_release = t_release

    def act(self, t, obs):
        return np.full(self.net.n_int, 0 if t >= self.t_release else 1)


def measure_saturation_flow(cfg, t_release=150.0, window=(10.0, 40.0), rain=None):
    sim = Simulation(cfg, RedThenGreen(t_release), seed=1, demand=DemandModel(4, base_rate=1.0), rain=rain)
    sim.run()
    t0 = t_release + cfg.lost_time_s + window[0]
    n = len([t for t in sim.exit_times if t0 <= t < t0 + (window[1] - window[0])])
    return n / (window[1] - window[0]) / 2          # two approaches (N and S) discharge


def test_saturation_flow_matches_config():
    """Release a standing queue; the emergent per-approach discharge should be close to
    SimConfig.saturation_flow, the value the optimiser's fluid model uses."""
    cfg = SimConfig(grid_rows=1, grid_cols=1, horizon_s=260, warmup_s=0)
    s = measure_saturation_flow(cfg)
    assert s == pytest.approx(cfg.saturation_flow, abs=0.06), s


def test_entry_roads_are_longer_and_distance_accounting_is_consistent():
    cfg = SimConfig(horizon_s=300, warmup_s=0)
    sim = Simulation(cfg, REGISTRY["fixed_sync"](), seed=2, demand=DemandModel(8, scale=0.3))
    assert (sim.Llink[sim.net.entry_links] == cfg.entry_link_length_m).all()
    assert (np.delete(sim.Llink, sim.net.entry_links) == cfg.link_length_m).all()
    r = sim.run()
    assert r["created"] == r["exited"] + r["on_road"] + r["buffered"]
    assert (sim.cum_observed <= sim.cum_created).all()        # sensor never sees more than demand
    assert r["mean_delay_s"] < 60                              # light load: delay is small, not a distance bug


def test_rain_cuts_saturation_flow():
    cfg = SimConfig(grid_rows=1, grid_cols=1, horizon_s=260, warmup_s=0)
    dry = measure_saturation_flow(cfg)
    wet = measure_saturation_flow(cfg, rain=lambda t: 1.0)
    assert wet < 0.92 * dry, (dry, wet)


def test_accident_blocks_then_releases():
    from traffic.scenarios import Accident
    cfg = SimConfig(horizon_s=400, warmup_s=0)
    ev = Accident(link=5, start_s=60, end_s=200, pos_frac=0.5)
    sim = Simulation(cfg, REGISTRY["fixed_sync"](), seed=4, demand=DemandModel(8, scale=1.0), events=[ev])
    p_ev = ev.pos_frac * sim.Llink[5]
    behind = None
    for _ in range(int(400 / cfg.dt)):
        sim.step()
        if behind is None and sim.t >= 120:
            behind = set(sim.vid[(sim.link == 5) & (sim.pos < p_ev)])
        if behind is not None and sim.t < 199:
            on = (sim.link == 5) & np.isin(sim.vid, list(behind))
            assert (sim.pos[on] <= p_ev + 0.5).all()          # nobody passes the blockage
    assert len(behind) > 0
    still_behind = (sim.link == 5) & np.isin(sim.vid, list(behind)) & (sim.pos < p_ev)
    assert still_behind.sum() < len(behind)                   # queue has been released after it cleared


def test_demand_profile_and_tide():
    from traffic.demand import hourly_profile
    from traffic.network import build_grid
    assert hourly_profile(7.7) > 2 * hourly_profile(3.0) and hourly_profile(17.0) > hourly_profile(12.0)
    net = build_grid(2, 2)
    head = net.link_heading[net.entry_links]
    dm = DemandModel(8, entry_heading=head, start_hour=6.0)
    am, pm = dm.rates(1.7 * 3600), dm.rates(11.0 * 3600)       # 07:42 and 17:00
    se = np.isin(head, [1, 2])                                  # E / S heading entries
    assert am[se].mean() > am[~se].mean() and pm[se].mean() < pm[~se].mean()


def test_event_raises_demand_on_one_side_and_calendar_publishes_only_the_announced_size():
    from traffic.demand import Event
    from traffic.network import build_grid
    net = build_grid(2, 2)
    head = net.link_heading[net.entry_links]
    ev = Event(heading=1, start_s=600, end_s=2400, size=1.5, announced=1.0)
    base = DemandModel(8, entry_heading=head, start_hour=16.0)
    dm = DemandModel(8, entry_heading=head, start_hour=16.0, events=(ev,))
    east = head == 1
    assert np.allclose(dm.rates(300), base.rates(300)) and np.allclose(dm.rates(3000), base.rates(3000))
    assert np.allclose(dm.rates(1500)[east], 2.5 * base.rates(1500)[east])     # the real crowd: +150%
    assert np.allclose(dm.rates(1500)[~east], base.rates(1500)[~east])
    assert np.allclose(dm.scheduled(1500), np.where(east, 1.0, 0.0))            # the calendar: +100% announced
    assert 0 < dm.scheduled(600 + 90)[east][0] < 1.0                             # ramps up over ramp_s


def test_scenarios_use_common_random_numbers_and_are_reproducible():
    from traffic.scenarios import SCENARIOS, build_sim
    cfg = SimConfig(warmup_s=0)
    sc = SCENARIOS["am_rain"]
    short = type(sc)(**{**sc.__dict__, "duration_s": 400, "start_hour": 7.0, "rain": ((100, 300, 0.8),)})
    a = build_sim(short, cfg, REGISTRY["fixed_sync"](), seed=3).run()
    b = build_sim(short, cfg, REGISTRY["fixed_sync"](cycle_s=120.0), seed=3).run()
    c = build_sim(short, cfg, REGISTRY["fixed_sync"](), seed=3).run()
    assert a["created"] == b["created"] and a == c


def test_stop_line_detector_counts_every_departure():
    """Entry road: cars through the entry sensor = cars over its stop line + cars still on it."""
    cfg = SimConfig(horizon_s=600, warmup_s=0)
    sim = Simulation(cfg, REGISTRY["fixed_sync"](), seed=2, demand=DemandModel(8, scale=1.2))
    sim.run()
    ent = sim.net.entry_links
    on_road = np.bincount(sim.link, minlength=sim.net.n_links)[ent]
    assert np.array_equal(sim.cum_observed, sim.cum_cross[ent] + on_road)
    assert sim.observe()["stop_cross"].sum() >= sim.exited
