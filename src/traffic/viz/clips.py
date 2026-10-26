"""Demo clips for the viewer: short windows of a scenario, tuned fixed-time vs our optimiser, same traffic.

Each clip runs the whole scenario (the forecaster needs its history) and records only the clip window.
Results are cached as JSON per clip so the build is resumable.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import SimConfig
from ..controllers import REGISTRY
from ..controllers.optimiser import ForecastOptimiser
from ..scenarios import SCENARIOS, build_sim

H = 3600
CLIPS = [
    dict(id="normal", name="Normal hour", scenario="am_ramp", t0=0.5 * H, t1=0.5 * H + 900,
         desc="Early weekday morning, light traffic: how much does a smarter plan help when nothing is wrong?"),
    dict(id="rush2", name="Morning rush", scenario="am_busy", t0=2.25 * H, t1=2.25 * H + 900,
         desc="08:15, the worst quarter-hour for the fixed plan (highest delay across held-out runs): demand 15% above what it was tuned for."),
    dict(id="storm2", name="Storm", scenario="am_rain", t0=2.25 * H, t1=2.25 * H + 900,
         desc="08:15, the worst quarter-hour for the fixed plan during the storm: slower driving, lower capacity, queues everywhere."),
    dict(id="accident", name="Accident", scenario="am_accident", t0=1.65 * H, t1=1.65 * H + 900,
         desc="A lane is blocked on an internal road at 07:42 for 15 minutes, at the peak."),
]
SEED = 300                       # demo seed, outside the tuning (100-104) and evaluation (200-205) seeds
CACHE_VERSION = "v3"             # v3: counts-only sensing (no cameras), controller beliefs recorded
FC_BEFORE_S, FC_AFTER_S = 3600.0, 900.0     # forecast chart span around the clip


class RecordedOptimiser(ForecastOptimiser):
    """ForecastOptimiser that also records what it believes, for the viewer (the control is unchanged).

    Inside the clip window, once per second: estimated cars per link (est.n), estimated cars at the stop line
    (est.q) and the entry-counter totals. At every forecast window: the counted network arrival rate of the window
    that just ended, the forecast for the next window, and the true demand rate (hidden from the controller).
    """
    def __init__(self, window=(0.0, float("inf")), **kw):
        super().__init__(**kw)
        self.window = window

    def reset(self, cfg, net, rng):
        super().reset(cfg, net, rng)
        self.belief, self.fc_log, self._tw = {}, [], None

    def _true_rate(self, a, b):
        return float(np.mean([self.sim.demand.rates(x).sum() for x in np.arange(a + 15.0, b, 30.0)]))

    def act(self, t, obs):
        req = super().act(t, obs)
        if self.window[0] <= t < self.window[1]:
            self.belief[round(t, 1)] = (np.round(self.est.n, 1).tolist(), np.round(self.est.q, 1).tolist(),
                                        np.asarray(obs["entry_cum"]).astype(int).tolist())
        fc = self.fc
        if fc.t_w != self._tw:                      # a new forecast window started at fc.t_w
            self._tw, W = fc.t_w, fc.W
            if fc.rates and fc.F is not None:
                self.fc_log.append([round(fc.t_w, 1), round(60 * float(fc.rates[0].sum()), 2),
                                    round(60 * self._true_rate(fc.t_w - W, fc.t_w), 2),
                                    [round(60 * float(r.sum()), 2) for r in fc.F]])
        return req


def _plan_series(ctrl):
    plans = getattr(ctrl, "plans", None)
    if not plans:
        return None
    return [[round(t, 1), float(C), round(float(np.mean(x)), 3), np.round(x, 3).tolist()] for t, C, x in plans]


def _run_panel(args):
    label, name, params, spec = args
    scen = SCENARIOS[spec["scenario"]]
    cfg = SimConfig(warmup_s=300)
    if name == "forecast_opt":
        ctrl = RecordedOptimiser(window=(spec["t0"], spec["t1"]), **params)
    else:
        ctrl = REGISTRY[name](**params)
    sim = build_sim(scen, cfg, ctrl, SEED, record=True, log_every_s=1.0,
                    record_window=(spec["t0"], spec["t1"]))
    sim.run()
    lg, frames = sim.log, []
    for k in range(len(lg["t"])):
        cars = np.column_stack([lg["vid"][k], lg["link"][k], np.round(lg["pos"][k], 1),
                                np.round(lg["vel"][k], 1)]).ravel().tolist()
        frames.append({"t": round(lg["t"][k], 1), "ex": lg["exited"][k], "bf": lg["buffered"][k],
                       "ph": lg["phase"][k].tolist(), "sw": lg["switching"][k].astype(int).tolist(),
                       "rn": lg["rain"][k], "c": cars})
    tc, mean, cnt = sim.delay_series(300.0)
    sel = (tc >= spec["t0"]) & (tc <= spec["t1"]) & ~np.isnan(mean)
    delay = float(np.average(mean[sel], weights=cnt[sel])) if sel.any() else float("nan")
    tc2, m2, _ = sim.delay_series(120.0)             # delay of cars by the time they arrived, 2-minute bins
    keep = (tc2 >= spec["t0"] - 60) & (tc2 <= spec["t1"] + 60)
    dser = [[float(a), None if np.isnan(b) else round(float(b), 1)] for a, b in zip(tc2[keep], m2[keep])]
    out = {"name": label, "frames": frames, "plan": _plan_series(sim.controller), "delay": round(delay, 1), "dser": dser}
    if isinstance(ctrl, RecordedOptimiser):
        for f in frames:                            # what the controller believed at that frame
            b = ctrl.belief.get(f["t"])
            if b is not None:
                f["en"], f["eq"], f["ec"] = b
        lo, hi = spec["t0"] - FC_BEFORE_S, spec["t1"] + FC_AFTER_S
        out["fc"] = [r for r in ctrl.fc_log if lo <= r[0] <= hi]
    return out


def build_clip(spec, cache_dir="outputs/viewer_cache", workers=2):
    cache = Path(cache_dir) / f"{spec['id']}_{CACHE_VERSION}.json"
    if cache.exists():
        clip = json.loads(cache.read_text())
        clip.setdefault("scenario", spec["scenario"])
        return clip
    tuned = json.loads(Path("outputs/tuned_scenarios.json").read_text())
    nom = {"am_busy": "am_ramp", "am_rain": "am_ramp", "am_accident": "am_ramp"}.get(spec["scenario"], spec["scenario"])
    jobs = [("Fixed-time (tuned)", "fixed_sync", {"cycle_s": tuned[nom]["const"]}, spec),
            ("Smart control (forecast + optimiser)", "forecast_opt", {"mode": "live"}, spec)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        panels = list(ex.map(_run_panel, jobs))
    scen = SCENARIOS[spec["scenario"]]
    clip = {"id": spec["id"], "scenario": spec["scenario"], "name": spec["name"], "desc": spec["desc"], "t0": spec["t0"], "t1": spec["t1"],
            "accidents": [dict(link=a.link, start=a.start_s, end=a.end_s, pos=a.pos_frac) for a in scen.accidents],
            "panels": panels}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(clip, separators=(",", ":")))
    return clip
