"""Two charts for the modelling section: (1) one disturbed day, forecast vs truth; (2) the censoring story."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..forecasting.evaluate import robustness, run
from ..forecasting.features import build_features, to_windows
from ..forecasting.model import BLOCK_T

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, RED, GREY, AQUA, VIOLET = "#2a78d6", "#eb6834", "#e34948", "#a9a8a2", "#1baf7a", "#6250d6"


def _hist(folder):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"outputs/history/{folder}/day_*.csv.gz"))], ignore_index=True)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_day(models, hist, out, W=5.0, entry=0):
    """The fresh day with the worst queueing outside the network, one entry, forecasts made 5 min ahead."""
    day = int(hist.groupby("day").backlog.max().idxmax())
    win = to_windows(hist[hist.day == day], W)
    L = build_features(win, W, 3)
    D = L[(L.entry == entry) & (L.h == 1)].sort_values("w")
    w = win[win.entry == entry].sort_values("w")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _style(ax)
    full = (w.entry_count >= BLOCK_T).to_numpy()
    rain = (w.rain > 0.1).to_numpy()
    ax.fill_between(w.hour, 0, 1, where=full, transform=ax.get_xaxis_transform(), color=RED, alpha=0.10, lw=0)
    ax.fill_between(w.hour, 0, 1, where=rain, transform=ax.get_xaxis_transform(), color=BLUE, alpha=0.08, lw=0)
    ax.scatter(w.hour, w.obs_rate * 3600, s=9, color=GREY, label="Sensor count (noisy, censored)", zorder=2)
    ax.plot(D.thr_hour, D.y * 3600, color=INK, lw=1.6, label="True demand rate", zorder=3)
    for name, col, lab, ls in (("hour_profile", ORANGE, "Hour-of-day profile (patterns only)", "-"),
                               ("poisson_glm+live", RED, "GLM + live counts, no guard", "--"),
                               ("poisson_glm+live+guard", BLUE, "GLM + live counts + censoring guard", "-")):
        ax.plot(D.thr_hour, models[name].predict(D) * 3600, color=col, lw=1.6, ls=ls, label=lab, zorder=4)
    ax.set_xlabel("Hour of day", color=MUTED)
    ax.set_ylabel("Arrivals at this entry (veh/h)", color=MUTED)
    ax.set_xlim(5, 21)
    wet = "wet " if rain.any() else ""
    ax.set_title(f"A gridlocked {wet}day (fresh day {day}, entry {entry}): forecasts made 5 min ahead",
                 loc="left", fontsize=11.5, color=INK)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    h += [Patch(color=RED, alpha=0.18)] + ([Patch(color=BLUE, alpha=0.15)] if rain.any() else [])
    l += ["Entry road full (sensor blind)"] + (["Raining"] if rain.any() else [])
    ax.legend(h, l, frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return day


def plot_skill(rows, out):
    """rows: {dataset: {model: skill}} - MSE skill vs the hour-of-day profile (0 = no better, 1 = perfect)."""
    order = [("profile_x_day_multiplier", "Profile × day multiplier", VIOLET), ("poisson_glm+live", "GLM + live (no guard)", "#e9a99f"),
             ("gbm_poisson+live", "Boosting + live (no guard)", "#efc7a8"), ("poisson_glm+live+guard", "GLM + live + guard", BLUE),
             ("gbm_poisson+live+guard", "Boosting + live + guard", AQUA)]
    sets = list(rows)
    fig, axes = plt.subplots(1, len(sets), figsize=(11, 4.0), sharey=True)
    for ax, ds in zip(axes, sets):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        vals = [rows[ds][k] for k, _, _ in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, color=[c for _, _, c in order], height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + (0.05 if v >= 0 else -0.05), yi, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5,
                    color=INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(rows[ds]["oracle_true_rate_now"], color=INK, lw=1, ls=":")
        ax.set_title(ds, loc="left", fontsize=10, color=INK)
        ax.set_yticks(y)
        ax.set_xlim(-2.6, 0.95)
    axes[0].set_yticklabels([n for _, n, _ in order], fontsize=9, color=MUTED)
    axes[0].set_xlabel("")
    fig.text(0.5, 0.01, "MSE skill vs hour-of-day profile (0 = no better than the profile, 1 = perfect). Dotted line = oracle that knows today's exact rate.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_ladder(lad, out, title):
    """Forecast error vs horizon, one line per ladder rung (lad: rows of evaluate.ladder_by_horizon)."""
    style = {"persistence": (GREY, "-"), "hour_profile": (ORANGE, "-"), "profile_x_rain": ("#eda100", "-"),
             "poisson_glm+live+guard": (BLUE, "-"), "gbm_poisson+live+guard": (AQUA, "-"), "oracle_true_rate_now": (INK, "--")}
    short = {"persistence": "Raw count carried forward", "hour_profile": "Hour-of-day profile",
             "profile_x_rain": "Profile x rain", "poisson_glm+live+guard": "GLM + live counts (main)",
             "gbm_poisson+live+guard": "Boosting + live counts", "oracle_true_rate_now": "Oracle: true rate now"}
    hcols = [c for c in lad.columns if c.startswith("+")]
    x = np.array([float(c[1:].split()[0]) for c in hcols])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    _style(ax)
    for _, r in lad.iterrows():                              # legend, not end labels: several rungs end at the same value
        col, ls = style[r.model]
        v = r[hcols].to_numpy(float)
        ax.plot(x, v, color=col, ls=ls, lw=2, marker="o", ms=8, mec="white", mew=2,
                label=f"{short[r.model]}  ({' / '.join(f'{a:.1f}' for a in v)})",
                zorder=4 if r.model == "poisson_glm+live+guard" else 3)
    top = max(22, lad[hcols].to_numpy(float).max() + 2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{v:g} min" for v in x], color=MUTED)
    ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    ax.set_ylim(0, top)
    ax.set_xlabel("Forecast horizon", color=MUTED)
    ax.set_ylabel("Error, % of mean true rate (MAE)", color=MUTED)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=INK,
              title="error at +5 / +10 / +15 min, %", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _cached(name, fn):
    """Fitting takes about a minute, and shell calls here cap at 3 minutes, so each fit is cached on disk."""
    import pickle
    from pathlib import Path
    path = Path("outputs/.cache") / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = fn()
    path.write_bytes(pickle.dumps(out))
    return out


def make_plots(W=5.0, stage="charts"):
    hist = _hist("fixed_sync")
    if stage == "val":
        _cached(f"val_{W}", lambda: run(hist, W_min=W, final=False)[:3])
        return None, {}
    if stage == "final":
        _cached(f"final_{W}", lambda: run(hist, W_min=W, final=True)[:3])
        return None, {}
    res_v = _cached(f"val_{W}", lambda: run(hist, W_min=W, final=False)[:3])[0]
    res_t, _, models = _cached(f"final_{W}", lambda: run(hist, W_min=W, final=True)[:3])
    M = {m.name: m for m in models}
    fresh = _hist("fixed_sync_fresh")
    day = plot_day(M, fresh, "outputs/forecast_day.png", W)
    rb, _ = robustness(models, fresh, W)
    pick = lambda res: res[res.group == "all"].set_index("model").skill_vs_profile.to_dict()
    rows = {"Validation days (10)": pick(res_v), "Fresh days (10), never used": rb.set_index("model").skill_vs_profile.to_dict(),
            "First test days (10), seen after the failure": pick(res_t)}
    plot_skill(rows, "outputs/forecast_skill.png")
    print("typical/accident days:", plot_two_days(M, fresh, "outputs/forecast_two_days.png", W))
    print(plot_regimes(M, [(fresh, False), (hist, True)], "outputs/forecast_regimes.png", W))
    return day, rows


def _pick_days(hist):
    d = hist.groupby("day").agg(weekend=("weekend", "max"), rain=("rain", lambda x: (x > 0.1).mean()),
                                acc=("accident", lambda x: x.sum() / 8), full=("entry_count", "max"), backlog=("backlog", "max"))
    calm = d[(d.weekend == 0) & (d.acc == 0) & (d.rain == 0) & (d.full < BLOCK_T)]
    typical = int((calm if len(calm) else d[d.weekend == 0]).backlog.idxmin())
    acc = d[(d.weekend == 0) & (d.acc > 0) & (d.full < BLOCK_T)]
    accident = int((acc if len(acc) else d[d.acc > 0]).sort_values(["rain", "acc"], ascending=[True, False]).index[0])
    return typical, accident


def plot_two_days(models, hist, out, W=5.0):
    """Network total (sum of the 8 entries), forecasts made 5 min ahead, for a typical day and an accident day."""
    typ, acc = _pick_days(hist)
    lines = [("hour_profile", ORANGE, "Hour-of-day profile (patterns only)"), ("profile_x_rain", VIOLET, "Profile × rain effect (patterns only)"),
             ("poisson_glm+live+guard", BLUE, "GLM + live counts + guard"), ("gbm_poisson+live+guard", AQUA, "Boosting + live counts + guard")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)
    for ax, day, title in ((axes[0], typ, "A typical dry weekday"), (axes[1], acc, "A day with an accident")):
        _style(ax)
        win = to_windows(hist[hist.day == day], W)
        L = build_features(win, W, 3)
        D = L[L.h == 1].sort_values(["w", "entry"])
        tot = win.groupby("w").agg(hour=("hour", "last"), obs=("obs_rate", "sum"), acc=("accident", "max"), rain=("rain", "mean"))
        ax.scatter(tot.hour, tot.obs * 3600, s=9, color=GREY, label="Sensor counts (sum of 8 entries)", zorder=2)
        g = D.groupby("w")
        ax.plot(g.thr_hour.first(), g.y.sum() * 3600, color=INK, lw=1.7, label="True demand", zorder=3)
        for name, col, lab in lines:
            ax.plot(g.thr_hour.first(), pd.Series(models[name].predict(D), index=D.index).groupby(D.w).sum() * 3600,
                    color=col, lw=1.2, label=lab, zorder=4)
        a = tot.acc.to_numpy() > 0
        ax.fill_between(tot.hour, 0, 1, where=a, transform=ax.get_xaxis_transform(), color=RED, alpha=0.14, lw=0)
        r = tot.rain.to_numpy() > 0.1
        ax.fill_between(tot.hour, 0, 1, where=r, transform=ax.get_xaxis_transform(), color=BLUE, alpha=0.08, lw=0)
        ax.set_ylabel("Network arrivals (veh/h)", color=MUTED)
        ax.set_title(f"{title} (fresh day {day})" + ("  — red band: an accident is active somewhere" if a.any() else ""),
                     loc="left", fontsize=11, color=INK)
        ax.set_xlim(5, 21)
    axes[1].set_xlabel("Hour of day", color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=3, loc="lower center", labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return typ, acc


def regime_labels(L):
    r = np.where(L.weekend > 0, "Weekend",
        np.where(L.accident > 0, "Weekday + accident",
        np.where(np.maximum(L.rain_now, L.rain_target) > 0.1, "Weekday + rain", "Weekday, dry")))
    return pd.Series(r, index=L.index)


def plot_regimes(models, sets, out, W=5.0):
    """rMAE (error as % of the mean true rate) by day type, pooled over held-out days only."""
    from ..forecasting.evaluate import split_days
    parts = []
    for hist, only_test in sets:
        if only_test:
            hist = hist[hist.day.isin(split_days(hist.day.unique())[2])]
        parts.append(build_features(to_windows(hist, W), W, 3))
    L = pd.concat(parts, ignore_index=True)
    reg = regime_labels(L)
    order = ["Weekday, dry", "Weekday + rain", "Weekday + accident", "Weekend"]
    series = [("hour_profile", ORANGE, "Hour-of-day profile (patterns only)"), ("profile_x_rain", VIOLET, "Profile × rain (patterns only)"),
              ("poisson_glm+live+guard", BLUE, "GLM + live counts + guard"), ("gbm_poisson+live+guard", AQUA, "Boosting + live counts + guard")]
    y = L.y.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _style(ax)
    width = 0.2
    table = {}
    for j, (name, col, lab) in enumerate(series):
        p = models[name].predict(L)
        vals = [100 * np.abs(p[(reg == r).to_numpy()] - y[(reg == r).to_numpy()]).mean() / y[(reg == r).to_numpy()].mean() for r in order]
        table[name] = vals
        xs = np.arange(len(order)) + (j - 1.5) * width
        ax.bar(xs, vals, width * 0.9, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    n = reg.value_counts()
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{r}\n({n.get(r, 0):,} windows)" for r in order], color=MUTED, fontsize=9)
    ax.set_ylabel("Forecast error, % of mean true rate", color=MUTED)
    ax.set_title("Where each model is strong: error by day type (held-out days only, 5 min ahead)", loc="left", fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return pd.DataFrame(table, index=order).round(1)
