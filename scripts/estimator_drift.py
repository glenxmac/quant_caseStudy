"""Drift check for the counts-only queue estimator: estimated vs true cars on every link over a run.
The truth is read from the simulator for diagnosis only; the controller never sees it.
usage: PYTHONPATH=src python scripts/estimator_drift.py <scenario> <seed> [mode]  -> outputs/drift/<scenario>_<seed>.npz
       PYTHONPATH=src python scripts/estimator_drift.py summary                  -> table + outputs/estimator_drift.png"""
import glob, os, sys
import numpy as np
from traffic.config import SimConfig
from traffic.controllers.optimiser import ForecastOptimiser
from traffic.scenarios import SCENARIOS, build_sim


class Logged(ForecastOptimiser):
    def reset(self, cfg, net, rng):
        super().reset(cfg, net, rng)
        self.log_t, self.log_est, self.log_true = [], [], []

    def act(self, t, obs):
        req = super().act(t, obs)
        if int(round(t)) % 10 == 0 and abs(t - round(t)) < 1e-6:
            s = self.sim
            self.log_t.append(t); self.log_est.append(self.est.n.copy())
            self.log_true.append(np.bincount(s.link, minlength=self.net.n_links) if len(s.link) else np.zeros(self.net.n_links))
        return req


def run(sc, seed, mode="persistence"):
    out = f"outputs/drift/{sc}_{seed}.npz"
    if os.path.exists(out):
        return
    os.makedirs("outputs/drift", exist_ok=True)
    ctrl = Logged(mode=mode)
    res = build_sim(SCENARIOS[sc], SimConfig(warmup_s=300), ctrl, seed).run()
    np.savez(out, t=np.array(ctrl.log_t), est=np.array(ctrl.log_est), true=np.array(ctrl.log_true),
             entry=ctrl.net.entry_links, delay=res["mean_delay_s"])
    print("wrote", out, round(res["mean_delay_s"], 1))


def summary():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {}
    for f in sorted(glob.glob("outputs/drift/*.npz")):
        d = np.load(f); name = os.path.basename(f)[:-4]
        err = d["est"] - d["true"]
        ent = np.zeros(err.shape[1], bool); ent[d["entry"]] = True
        for kind, m in (("entry", ent), ("internal", ~ent)):
            e = err[:, m]
            print(f"{name:18s} {kind:8s} MAE {np.abs(e).mean():5.2f}  bias {e.mean():+5.2f}  worst {np.abs(e).max():5.1f}"
                  f"  p95 {np.percentile(np.abs(e), 95):4.1f}  true mean {d['true'][:, m].mean():5.1f}")
        rows[name] = d
    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True)
    picks = [("am_ramp_100", 1, "Typical morning: entry road (link 1, eastbound into the top-left junction)"),
             ("am_ramp_100", 5, "Typical morning: internal road (link 5, top-left to top-right, eastbound)"),
             ("am_accident_100", 5, "Accident 07:42-07:57 on link 5 (incident feed; signals on the fixed fallback plan from 07:42)"),
             ("am_accident_100", 0, "Accident: link 0, which feeds link 5 (northbound, bottom-left to top-left)")]
    for ax, (name, l, title) in zip(axes, picks):
        if name not in rows:
            continue
        d = rows[name]; h = 6.0 + d["t"] / 3600
        ax.plot(h, d["true"][:, l], color="#444", lw=1.0, label="true cars on road (simulator)")
        ax.plot(h, d["est"][:, l], color="#2a7de1", lw=1.2, label="controller's estimate (entry counts only)")
        ax.set_title(title, fontsize=10, loc="left"); ax.set_ylabel("cars")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left"); axes[-1].set_xlabel("hour of day")
    fig.tight_layout(); fig.savefig("outputs/estimator_drift.png", dpi=150); print("wrote outputs/estimator_drift.png")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1], int(sys.argv[2]), *(sys.argv[3:4]))
