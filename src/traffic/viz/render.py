"""Render the demo clips to ONE self-contained HTML file (no server, no dependencies).

Each clip draws two runs side by side on a shared timeline. Because arrivals use common
random numbers, both panels face the same traffic, so differences are the controller's.
"""
import json
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def scenario_averages(eval_dir="outputs/final_eval_counts"):
    """scenario -> {fixed, smart, n}: mean delay over the held-out evaluation runs."""
    import pandas as pd
    files = sorted(Path(eval_dir).glob("*.csv"))
    if not files:
        return {}
    df = pd.concat(map(pd.read_csv, files))
    out = {}
    for sc, g in df.groupby("scenario"):
        a, b = g[g.controller == "fixed_tuned"], g[g.controller == "opt_live"]
        out[sc] = {"fixed": float(a.mean_delay_s.mean()), "smart": float(b.mean_delay_s.mean()), "n": int(len(b))}
    return out


def export_clips(clips: list, path: str, cfg=None, h0: float = 6.0) -> str:
    """Several short demo clips (see viz/clips.py) in one tabbed HTML page."""
    from ..config import SimConfig, VehicleConfig
    from ..network import build_grid
    cfg = cfg or SimConfig()
    net, veh = build_grid(cfg.grid_rows, cfg.grid_cols), VehicleConfig()
    lens = np.full(net.n_links, cfg.link_length_m)
    lens[net.entry_links] = cfg.entry_link_length_m
    avg = scenario_averages()
    for c in clips:
        c["avg"] = avg.get(c.get("scenario"))
    data = {"L": cfg.link_length_m, "rows": net.rows, "cols": net.cols, "len": lens.tolist(), "entries": net.entry_links.tolist(),
            "vlen": veh.length_m, "slot": veh.slot_m, "vmax": veh.v_free_mps, "logDt": 1.0, "h0": h0, "clips": clips}
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)
