"""Headline chart: mean delay per car by scenario, fixed-time plans vs the optimiser (held-out seeds).
usage: PYTHONPATH=src python scripts/results_chart.py  -> outputs/results_delay.png"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCEN = [("am_ramp", "Typical\nmorning"), ("am_busy", "Busy\nmorning"), ("am_rain", "Storm"),
        ("pm_ramp", "Evening"), ("am_accident", "Accident"), ("pm_event", "Stadium\nevent")]
CTRL = [("fixed_default", "Fixed 60 s (untuned)", "#a9a8a3"),
        ("fixed_tuned", "Fixed, tuned green wave", "#eb6834"),
        ("opt_live", "Optimiser (counts only)", "#2a78d6")]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

if __name__ == "__main__":
    d = pd.concat(pd.read_csv(f) for f in glob.glob("outputs/final_eval_counts/*.csv") if "variants_" not in f)
    m = d.groupby(["scenario", "controller"]).mean_delay_s.mean()
    n = d[d.controller == "opt_live"].groupby("scenario").seed.nunique()
    fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    x, w = np.arange(len(SCEN)), 0.26
    for j, (c, label, col) in enumerate(CTRL):
        v = [m[(s, c)] for s, _ in SCEN]
        ax.bar(x + (j - 1) * w, v, w - 0.02, color=col, label=label, zorder=2)
    for i, (s, _) in enumerate(SCEN):
        opt, tuned = m[(s, "opt_live")], m[(s, "fixed_tuned")]
        ax.text(x[i] + w, opt + 6, f"{opt:.0f} s\n{100 * (opt / tuned - 1):+.0f}%", ha="center", va="bottom",
                fontsize=10, color=INK)
    ax.set_xticks(x, [f"{lab}\n(n={n[s]})" for s, lab in SCEN], fontsize=11, color=INK)
    ax.set_ylabel("Mean delay per car (s)", color=MUTED, fontsize=11)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_ylim(0, 470)
    ax.set_title("Mean delay per car on held-out days (labels: optimiser vs the tuned fixed plan)",
                 loc="left", fontsize=12.5, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig("outputs/results_delay.png", dpi=200)
    print("wrote outputs/results_delay.png")
