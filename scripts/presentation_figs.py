"""Figures for the slide deck -> presentation/figs/*.png (white background, one palette).
usage: PYTHONPATH=src python scripts/presentation_figs.py   (run scripts/tuning_curve.py first for the cycle sweep)"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

OUT = Path("presentation/figs")
INK, MUTED, GRID = "#1a1a1a", "#5a5955", "#e6e5e1"
BLUE, ORANGE, GREY, LBLUE, GREEN, RED = "#2a78d6", "#eb6834", "#b4b3ae", "#a9c8f0", "#1baf7a", "#d64545"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
DPI = 220
SCEN = [("am_ramp", "Typical\nmorning"), ("am_busy", "Busy\nmorning"), ("am_rain", "Storm"),
        ("pm_ramp", "Evening"), ("am_accident", "Accident"), ("pm_event", "Stadium\nevent")]


def clean(ax, grid="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def clock(ax, lo, hi, step=1):
    ax.set_xlim(lo, hi)
    ax.set_xticks(np.arange(np.ceil(lo), hi + 0.01, step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda h, _: f"{int(h):02d}:00"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", OUT / name)


def eval_means():
    d = pd.concat(pd.read_csv(f) for f in glob.glob("outputs/final_eval_counts/*.csv") if "variants_" not in f)
    return d.groupby(["scenario", "controller"]).mean_delay_s.mean()


# ---------------------------------------------------------------- network schematic
def fig_network():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 10); ax.set_aspect("equal"); ax.axis("off")
    J = [(3.5, 6.5), (6.5, 6.5), (3.5, 3.5), (6.5, 3.5)]
    for c in (3.5, 6.5):
        ax.plot([0, 10], [c, c], color="#d9d8d3", lw=16, solid_capstyle="butt", zorder=1)
        ax.plot([c, c], [0, 10], color="#d9d8d3", lw=16, solid_capstyle="butt", zorder=1)
    rng = np.random.default_rng(4)
    # a few cars queued before stop lines (small rectangles)
    def car(x, y, horiz, col):
        w, h = (0.42, 0.22) if horiz else (0.22, 0.42)
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, color=col, zorder=3, lw=0))
    for k in range(4):
        car(2.75 - 0.55 * k, 6.35, True, ORANGE)       # eastbound queue at junction A
        car(6.65, 7.25 + 0.55 * k, False, ORANGE)      # southbound queue at junction B
    for k in range(2):
        car(3.35, 2.55 - 0.55 * k, False, "#6d6c68")
        car(9.0 - 0.9 * k, 3.65, True, "#6d6c68")
    for x in rng.uniform(0.5, 9.5, 6):
        car(x, 6.65, True, "#6d6c68")
    for (x, y) in J:
        ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color="#2b2b2b", zorder=4))
        ax.add_patch(Circle((x - 0.18, y + 0.0), 0.11, color=GREEN, zorder=5))
        ax.add_patch(Circle((x + 0.18, y + 0.0), 0.11, color=RED, zorder=5))
    # entry counters on the 8 entry roads
    ent = [(0.9, 6.5, 1, 0), (0.9, 3.5, 1, 0), (9.1, 6.5, -1, 0), (9.1, 3.5, -1, 0),
           (3.5, 9.1, 0, -1), (6.5, 9.1, 0, -1), (3.5, 0.9, 0, 1), (6.5, 0.9, 0, 1)]
    for (x, y, dx, dy) in ent:
        ax.add_patch(Circle((x, y), 0.26, color=BLUE, zorder=6))
        ax.add_patch(FancyArrowPatch((x + dx * 0.45, y + dy * 0.45), (x + dx * 1.15, y + dy * 1.15),
                                     arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=2, zorder=6))
    ax.text(5, -0.35, "\u25CF entry counter (the only sensor)     \u25A0 signal     \u25AC cars", ha="center",
            fontsize=12, color=MUTED)
    ax.text(5, 5.0, "200 m", ha="center", va="center", fontsize=11, color=MUTED, rotation=90 if False else 0,
            bbox=dict(fc="white", ec="none", pad=1.5), zorder=7)
    ax.text(1.9, 6.95, "400 m", ha="center", fontsize=11, color=MUTED)
    save(fig, "fig_network.png")


# ---------------------------------------------------------------- random days
def fig_days():
    from traffic.daygen import sample_day
    from traffic.network import build_grid
    net = build_grid(2, 2)
    t = np.arange(150, 16 * 3600, 300.0)
    hours = 5 + t / 3600
    days, D = [], []
    for i in range(40):
        d = sample_day(i, 5.0, 16 * 3600)
        dm = d.demand(net, seed=i)
        days.append(d); D.append([dm.rates(x).sum() * 3600 for x in t])
    D = np.array(D)
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    clean(ax)
    for r in D:
        ax.plot(hours, r, color=GREY, lw=0.8, alpha=0.45)
    ax.plot(hours, D.mean(0), color=BLUE, lw=3, label="average weekday", zorder=5)
    hi = int(np.argmax([d.day_effect for d in days]))
    sh = int(np.argmax([abs(d.time_shift_h) for d in days]))
    ax.plot(hours, D[hi], color=ORANGE, lw=2.2, zorder=6, label=f"a busy day ({100 * (days[hi].day_effect - 1):+.0f}% level)")
    ax.plot(hours, D[sh], color=GREEN, lw=2.2, zorder=6, label=f"a day whose rush runs {abs(days[sh].time_shift_h) * 60:.0f} min {'late' if days[sh].time_shift_h > 0 else 'early'}")
    ax.plot([], [], color=GREY, lw=1, label="other random days")
    ax.axhline(8 * 0.16 * 3600, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(10.2, 8 * 0.16 * 3600 + 90, "saturation knee of the grid", color=MUTED, fontsize=11)
    ax.set_ylabel("vehicles per hour entering (all 8 entries)")
    ax.set_ylim(0, 7000)
    clock(ax, 5, 21, 2)
    ax.legend(frameon=False, loc="upper center", fontsize=11, bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "fig_days.png")
    return days[sh].time_shift_h * 60


# ---------------------------------------------------------------- demand -> Poisson
def fig_poisson():
    from traffic.network import build_grid
    from traffic.scenarios import SCENARIOS
    net = build_grid(2, 2)
    dm = SCENARIOS["am_ramp"].demand(net, 200)
    tw = np.arange(0, 3 * 3600, 300.0)
    lam = np.array([np.mean([dm.rates(tw_i + s)[0] for s in range(0, 300, 30)]) * 300 for tw_i in tw])
    e = int(np.argmax([dm.rates(1.7 * 3600)[k] for k in range(8)]))
    lam = np.array([np.mean([dm.rates(tw_i + s)[e] for s in range(0, 300, 30)]) * 300 for tw_i in tw])
    rng = np.random.default_rng(11)
    cnt = rng.poisson(lam)
    h = 6 + (tw + 150) / 3600
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    clean(ax)
    ax.fill_between(h, lam - 2 * np.sqrt(lam), lam + 2 * np.sqrt(lam), color=LBLUE, alpha=0.5, lw=0,
                    label="\u00b12 standard deviations")
    ax.plot(h, lam, color=BLUE, lw=3, label="true expected count (the rate)")
    ax.scatter(h, cnt, s=34, color=ORANGE, zorder=5, label="what the counter reports")
    ax.set_ylabel("cars entering in a 5-minute window")
    ax.set_ylim(0, None)
    clock(ax, 6, 9, 1)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    save(fig, "fig_poisson.png")
    pk = lam.max()
    print("peak expected count", pk, "noise %", 100 / np.sqrt(pk), "entry", e)
    return pk


# ---------------------------------------------------------------- forecast ladder
def fig_ladder():
    d = pd.read_csv("outputs/forecast_ladder_by_horizon.csv")
    d = d[d.dataset.str.startswith("held-out")].set_index("model")["overall"]
    rows = [("Last count repeated", d["persistence"], GREY), ("Hour-of-day profile", d["hour_profile"], GREY),
            ("+ weather and event calendar", d["profile_x_rain"], GREY),
            ("Poisson GLM + live counts", d["poisson_glm+live+guard"], BLUE),
            ("Boosted trees + live counts", d["gbm_poisson+live+guard"], LBLUE),
            ("Oracle (true rate, held flat)", d["oracle_true_rate_now"], "#e3e2dd")]
    ev = [("Hour-of-day profile", 40.4, GREY), ("Profile \u00d7 weather \u00d7 calendar", 18.4, GREY),
          ("Poisson GLM + live counts", 13.3, BLUE), ("Boosted trees + live counts", 20.6, LBLUE)]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.3), gridspec_kw={"height_ratios": [6, 4], "hspace": 0.5})
    for ax, data, title in ((axes[0], rows, "All 20 held-out days, averaged over +5, +10, +15 min"),
                            (axes[1], ev, "Inside event windows, 5 min ahead")):
        clean(ax, None)
        y = np.arange(len(data))[::-1]
        ax.barh(y, [r[1] for r in data], color=[r[2] for r in data], height=0.68, zorder=3)
        ax.set_yticks(y, [r[0] for r in data], fontsize=12, color=INK)
        for yy, r in zip(y, data):
            ax.text(r[1] + 0.5, yy, f"{r[1]:.1f}%", va="center", fontsize=12, color=INK,
                    fontweight="bold" if r[2] == BLUE else "normal")
        ax.set_xticks([]); ax.set_xlim(0, max(r[1] for r in data) * 1.15)
        ax.set_title(title, loc="left", fontsize=12.5, color=MUTED)
    save(fig, "fig_ladder.png")


# ---------------------------------------------------------------- results bars
def grouped(ax, series, labels, colors, names, w=0.34):
    x = np.arange(len(labels))
    n = len(series)
    for j, (v, c, nm) in enumerate(zip(series, colors, names)):
        ax.bar(x + (j - (n - 1) / 2) * w, v, w - 0.03, color=c, label=nm, zorder=3)
    ax.set_xticks(x, labels, fontsize=12, color=INK)
    return x, w


def fig_untuned():
    m = eval_means()
    un = [m[(s, "fixed_default")] for s, _ in SCEN]
    op = [m[(s, "opt_live")] for s, _ in SCEN]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    clean(ax)
    x, w = grouped(ax, [un, op], [l for _, l in SCEN], [GREY, BLUE], ["Fixed 60 s plan (untuned)", "Optimiser"])
    for i, (a, b) in enumerate(zip(un, op)):
        ax.text(x[i], max(a, b) + 6, f"\u2212{100 * (1 - b / a):.0f}%", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=BLUE)
    ax.set_ylabel("mean delay per car (s)")
    ax.set_ylim(0, max(un) * 1.12)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    save(fig, "fig_untuned.png")
    return [100 * (1 - b / a) for a, b in zip(un, op)]


def fig_vs_tuned():
    m = eval_means()
    tu = [m[(s, "fixed_tuned")] for s, _ in SCEN]
    op = [m[(s, "opt_live")] for s, _ in SCEN]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    clean(ax)
    x, w = grouped(ax, [tu, op], [l for _, l in SCEN], [ORANGE, BLUE], ["Fixed, tuned green wave", "Optimiser"])
    for i, (a, b) in enumerate(zip(tu, op)):
        p = 100 * (1 - b / a)
        ax.text(x[i], max(a, b) + 4, f"\u2212{p:.0f}%", ha="center", va="bottom", fontsize=13,
                fontweight="bold" if p >= 20 else "normal", color=BLUE if p >= 20 else MUTED)
    ax.set_ylabel("mean delay per car (s)")
    ax.set_ylim(0, max(tu) * 1.2)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    save(fig, "fig_vs_tuned.png")
    return [100 * (1 - b / a) for a, b in zip(tu, op)]


# ---------------------------------------------------------------- tuning sweep
def fig_tuning():
    df = pd.read_csv(OUT / "tuning_curve.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    clean(ax)
    for sc, col, lab in (("am_ramp", BLUE, "typical morning"), ("pm_ramp", ORANGE, "typical evening")):
        d = df[df.scenario == sc].sort_values("cycle_s")
        b = d.loc[d.mean_delay_s.idxmin()]
        ax.plot(d.cycle_s, d.mean_delay_s, color=col, lw=2.8, marker="o", ms=7, zorder=4,
                label=f"{lab}: best cycle {b.cycle_s:.0f} s")
        ax.scatter([b.cycle_s], [b.mean_delay_s], s=190, facecolor="white", edgecolor=col, lw=3, zorder=6)
    ax.set_xlabel("common cycle length of the green wave (s)")
    ax.set_ylabel("mean delay per car (s)")
    ax.set_xticks([30, 60, 90, 120, 150, 180])
    ax.set_ylim(0, 800)
    ax.axvline(60, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(63, 650, "untuned plan: 60 s", color=MUTED, fontsize=11.5)
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    save(fig, "fig_tuning.png")


# ---------------------------------------------------------------- scalability
def fig_scal():
    s = pd.read_csv("outputs/scalability.csv")
    fig, ax = plt.subplots(figsize=(5.9, 4.5))
    clean(ax, "y")
    ax.plot(s.intersections, s.solve_mean_ms, color=BLUE, lw=3, marker="o", ms=9, zorder=4)
    for _, r in s.iterrows():
        lab = f"{r.solve_mean_ms:.0f} ms" if r.solve_mean_ms < 1000 else f"{r.solve_mean_ms / 1000:.1f} s"
        ax.annotate(lab, (r.intersections, r.solve_mean_ms), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=12.5, fontweight="bold", color=BLUE)
    ax.axhline(60000, color=ORANGE, lw=2.2, ls=(0, (4, 3)))
    ax.text(4.2, 60000 * 1.25, "time available: one decision every 60 s", color=ORANGE, fontsize=12,
            fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(s.intersections, [f"{g}\n({n})" for g, n in zip(s.grid, s.intersections)], fontsize=11.5)
    ax.minorticks_off()
    ax.set_yticks([10, 100, 1000, 10000, 60000], ["10 ms", "100 ms", "1 s", "10 s", "60 s"])
    ax.set_ylim(10, 300000)
    ax.set_ylabel("time to solve one QP")
    ax.set_xlabel("grid (intersections)")
    save(fig, "fig_scal.png")


# ---------------------------------------------------------------- max-pressure (earlier camera version)
def fig_maxp():
    sc = ["Typical", "Busy", "Storm", "Evening", "Accident"]
    tu = [59.9, 142.1, 199.6, 95.0, 199.8]
    mp = [58.3, 177.7, 229.7, 137.3, 243.7]
    op = [46.5, 111.3, 167.7, 91.7, 206.6]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    clean(ax)
    grouped(ax, [tu, mp, op], sc, [ORANGE, RED, BLUE],
            ["Fixed, tuned green wave", "Max-pressure", "Optimiser"], w=0.28)
    ax.set_ylabel("mean delay per car (s)")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    save(fig, "fig_maxp.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_network()
    fig_days()
    fig_poisson()
    fig_ladder()
    print("vs untuned %", np.round(fig_untuned(), 1))
    print("vs tuned %", np.round(fig_vs_tuned(), 1))
    if (OUT / "tuning_curve.csv").exists():
        fig_tuning()
    fig_scal()
    fig_maxp()
