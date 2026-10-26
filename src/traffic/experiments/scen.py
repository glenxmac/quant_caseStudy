"""Scenario experiments: baseline tuning on the NOMINAL day, evaluation on every scenario.

The fixed-time cycle is tuned only on the nominal scenario (tuning seeds 100+, disjoint from the
evaluation seeds), then applied unchanged to the busy / rainy / accident variants: the way
real plans are set for a normal day and then meet a real one.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from ..controllers import REGISTRY
from ..scenarios import SCENARIOS, build_sim, nominal_of

TUNE_SEEDS = (100, 101, 102, 103)
CONST_CYCLES = (30, 45, 60, 90, 120, 150, 180)


def run_one(job):
    label, name, params, scen, seed, cfg = job
    res = build_sim(SCENARIOS[scen], cfg, REGISTRY[name](**params), seed).run()
    res.update(controller=label, scenario=scen, seed=seed)
    return res


def run_scen(controllers: dict, scenarios, seeds, cfg, workers=4) -> pd.DataFrame:
    """controllers: label -> (registry_name, params | f(scenario_name) -> params)."""
    jobs = [(lab, name, (p(sc) if callable(p) else p), sc, sd, cfg)
            for lab, (name, p) in controllers.items() for sc in scenarios for sd in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return pd.DataFrame(list(ex.map(run_one, jobs, chunksize=2)))
    return pd.DataFrame([run_one(j) for j in jobs])


def tune_cycle(cfg, nominal: str, workers=4) -> float:
    """Best common cycle length for the green wave on the nominal scenario (tuning seeds)."""
    ctrls = {str(c): ("fixed_sync", {"cycle_s": float(c)}) for c in CONST_CYCLES}
    df = run_scen(ctrls, [nominal], TUNE_SEEDS, cfg, workers)
    return float(df.groupby("controller").mean_delay_s.mean().idxmin())


def load_or_tune(cfg, nominals=("am_ramp", "pm_ramp"), path="outputs/tuned_scenarios.json", workers=4):
    """Resumable: the tuned cycle ("const") is saved per nominal scenario as soon as it is found (~1 min each)."""
    p = Path(path)
    t = json.loads(p.read_text()) if p.exists() else {}
    for nom in nominals:
        if "const" not in t.get(nom, {}):
            t.setdefault(nom, {})["const"] = tune_cycle(cfg, nom, workers)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(t, indent=2))
    return t


def baseline_controllers(tuned: dict) -> dict:
    """The two fixed-time baselines: untuned 60 s green wave and the green wave with the tuned cycle."""
    return {
        "fixed_default": ("fixed_sync", {}),
        "fixed_tuned": ("fixed_sync", lambda sc: {"cycle_s": tuned[nominal_of(sc)]["const"]}),
    }
