"""True demand envelope across many random days (python main.py days -> outputs/demand_days.png).

Style: one axis per panel (no dual axes), thin 2px lines, recessive grid, text in ink colours
(never series colours), direct labels at line ends plus a legend.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
KNEE_VPH = 8 * 0.16 * 3600           # entry demand at scale 1.0: the grid's saturation knee


def demand_matrix(days, net, step_s=300):
    """True total entry demand (veh/h) and rain intensity on a time grid, for each day."""
    dur = days[0].duration_s
    t = np.arange(step_s / 2, dur, step_s)
    D = np.zeros((len(days), len(t)))
    R = np.zeros_like(D)
    for i, d in enumerate(days):
        dm, rf = d.demand(net, seed=i), d.rain_fn()
        D[i] = [dm.rates(x).sum() * 3600 for x in t]
        if rf is not None:
            R[i] = [rf(x) for x in t]
    return days[0].start_hour + t / 3600.0, D, R


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def _clock(ax, lo, hi):
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda h, _: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"))
    ax.set_xticks(np.arange(np.ceil(lo), hi + 0.01, 1 if hi - lo <= 8 else 2))


def plot_demand_days(hours, demand, rain_frac, path, n_sample=18):
    """demand: (n_days, n_t) total entry demand in veh/h; rain_frac: (n_t,) share of days raining."""
    n_days = demand.shape[0]
    mean, sd = demand.mean(0), demand.std(0)
    cv = sd / mean
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3.2, 1.3, 1.2], "hspace": 0.14})
    ax = axes[0]
    _style(ax, "vehicles per hour (all entries)")
    for i in range(min(n_sample, n_days)):
        ax.plot(hours, demand[i], color=GREY, lw=0.7, alpha=0.35, label="individual days" if i == 0 else None)
    ax.fill_between(hours, mean - sd, mean + sd, color=BLUE, alpha=0.16, lw=0, label="mean \u00b1 1 std across days")
    ax.plot(hours, mean, color=BLUE, lw=2, label="mean")
    ax.axhline(KNEE_VPH, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(hours[-1], KNEE_VPH + 70, "grid saturation \u2248 4,600 veh/h (fixed-time baseline)", color=INK2, fontsize=8.5, ha="right")
    for lo, hi in ((6.5, 9.5), (15.5, 18.5)):
        m = (hours >= lo) & (hours <= hi)
        j = np.argmax(np.where(m, mean, -1))
        hh, mm = int(hours[j]), int((hours[j] % 1) * 60)
        ax.annotate(f"peak {hh:02d}:{mm:02d}, {mean[j]:,.0f} veh/h", (hours[j], mean[j]), xytext=(hours[j], KNEE_VPH * 1.12),
                    ha="center", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=GRID, lw=1))
    ax.set_ylim(0, KNEE_VPH * 1.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.03), frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_title(f"True demand across {n_days} simulated weekdays", loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.03, "Arrivals generated at the 8 entry roads before any queueing, so this is demand, not what sensors see.",
            transform=ax.transAxes, color=INK2, fontsize=9)

    ax = axes[1]
    _style(ax, "CV across days\n(std \u00f7 mean)")
    ax.plot(hours, cv, color=ORANGE, lw=2)
    ax.set_ylim(0, max(0.2, cv.max() * 1.25))

    ax = axes[2]
    _style(ax, "share of days\nraining")
    ax.plot(hours, rain_frac, color=AQUA, lw=2)
    ax.set_ylim(0, max(0.1, rain_frac.max() * 1.3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    _clock(ax, hours[0], hours[-1])
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
