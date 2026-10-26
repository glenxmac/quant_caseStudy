"""Our controller: count -> forecast -> model the queues -> optimise.

Sensing assumption (counts only): a car counter on each entry road (cumulative cars in), the clock, the weather
feed, the controller's own signal state, and the city's incident feed ("link X blocked from t0", end reported when
it clears). It knows the road layout, link lengths, average turning shares and nominal saturation flow. It reads no
cameras: queues inside the network are not measured but estimated.

Four layers (see README):
  estimate  every second: a store-and-forward model of every link (QueueEstimator) driven by the entry counts and
            the controller's own signals gives the estimated cars on each link;
  forecast  once per 5-min window: expected arrival rate at each entry for the next 15 min (forecasting/online.py);
  strategy  every decision_s (60 s): choose the cycle length and the NS/EW green split for the next horizon by
            solving a convex program on the same store-and-forward model, starting from the estimate;
  reactive  every second: switch phase when the planned green has run out (and, optionally, gap out when the
            estimate says the served approaches are empty). The simulator enforces min green and the lost time.

Fluid model (per approach link a, step k of length D seconds, N steps):
    q[a,k+1] = q[a,k] + arrivals[a,k] - u[a,k]                 vehicle conservation
    u[a,k]  <= s * D * g[a,k]                                   service while green (s = nominal saturation flow)
    u[a,k]  <= q[a,k] + arrivals[a,k]                           cannot serve cars that are not there
    u[b,:]   = 0                                                links reported blocked by the incident feed
    arrivals = D * forecast (entry links) + T u[:,k-1]          turning shares T route discharge downstream
    g[a,k]   = x[i,k] - lost/C (NS)  or  1 - x[i,k] - lost/C (EW)   x = NS share of the cycle at intersection i
    x_lo <= x <= x_hi                                           min and max green
    q[a,k+1] <= storage[a] + slack[a,k]                         full roads (soft: spillback slack)
Objective: total queue (vehicle-seconds of delay) + a quadratic term (balances queues, i.e. fairness)
           + terminal queue (so the horizon does not dump work after its end) + spillback slack
           + switching penalty (|x[k] - x[k-1]|).
The cycle length C is not in the program (it would make it non-convex): it comes from Webster's formula
on the forecast link flows, with a tunable effective lost time, and is smoothed between decisions.
"""
import cvxpy as cp
import numpy as np

from ..config import VehicleConfig
from ..forecasting.online import OnlineForecaster
from .base import Controller


def link_flow_matrix(net, turn_probs):
    """T[a, u] = share of the flow discharged from link u that enters link a."""
    T = np.zeros((net.n_links, net.n_links))
    for u in range(net.n_links):
        for t in range(3):
            a = net.next_link[u, t]
            if a >= 0:
                T[a, u] += turn_probs[t]
    return T


def webster_cycle(net, T, lam_entry, sat, l_eff, c_min, c_max):
    """Network-wide cycle from the forecast: steady link flows f = (I - T)^-1 lam, critical flow ratio per
    intersection Y = y_NS + y_EW, Webster C = (1.5 L + 5) / (1 - Y) with L = l_eff."""
    lam = np.zeros(net.n_links)
    lam[net.entry_links] = lam_entry
    f = np.linalg.solve(np.eye(net.n_links) - T, lam)
    y = f / sat
    Y = 0.0
    for i in range(net.n_int):
        ids = np.arange(4 * i, 4 * i + 4)
        ys = [y[ids[net.link_phase[ids] == p]].max() for p in (0, 1)]
        Y = max(Y, ys[0] + ys[1])
    if Y >= 0.92:
        return float(c_max), Y
    return float(np.clip((1.5 * l_eff + 5.0) / (1.0 - Y), c_min, c_max)), Y


def solve_splits(q0, lam, x_prev, C, net, T, storage, p, blocked=None):
    """Solve the convex program. q0: (n_links,) cars on each link now; lam: (n_entries, N) forecast veh/s;
    x_prev: (n_int,) current NS share; blocked: (n_links,) bool, links that cannot discharge.
    Returns (x plan (n_int, N), q plan, status)."""
    n_l, n_i = net.n_links, net.n_int
    N, D = lam.shape[1], p["step_s"]
    s = np.asarray(p["sat"], dtype=float)                  # scalar, or (n_links,) per link
    s = np.broadcast_to(s[:, None] if s.ndim == 1 else s, (n_l, N))
    lh = p["lost_s"] / C                                   # lost-time share per phase
    lo = (p["min_green_s"] + p["lost_s"]) / C
    hi_g = (p["max_green_s"] + p["lost_s"]) / C
    x_lo, x_hi = max(lo, 1.0 - hi_g), min(1.0 - lo, hi_g)
    if x_lo > x_hi:
        x_lo = x_hi = 0.5
    A = np.zeros((n_l, N))
    A[net.entry_links, :] = lam * D                        # exogenous arrivals (veh per step)
    x = cp.Variable((n_i, N))
    q = cp.Variable((n_l, N + 1), nonneg=True)
    u = cp.Variable((n_l, N), nonneg=True)
    slack = cp.Variable((n_l, N), nonneg=True)
    sign = np.where(net.link_phase == 0, 1.0, -1.0)[:, None]
    off = np.where(net.link_phase == 0, -lh, 1.0 - lh)[:, None]
    g = cp.multiply(sign, x[net.link_dest, :]) + off       # green share of each approach per step
    cons = [q[:, 0] == q0, x >= x_lo, x <= x_hi, u <= cp.multiply(s * D, g), q[:, 1:] <= storage[:, None] + slack]
    for k in range(N):
        arr = A[:, k] + (T @ u[:, k - 1] if k > 0 else 0)
        cons += [q[:, k + 1] == q[:, k] + arr - u[:, k], u[:, k] <= q[:, k] + arr]
    if blocked is not None and np.any(blocked):             # reported incident: no service until it clears
        cons += [u[np.nonzero(blocked)[0], :] == 0]
    dx = cp.diff(cp.hstack([x_prev.reshape(-1, 1), x]), axis=1)
    obj = (cp.sum(q[:, 1:]) + p["quad_w"] * cp.sum_squares(q[:, 1:]) / 10.0 + p["term_w"] * cp.sum(q[:, N])
           + p["spill_w"] * cp.sum(slack) + p["switch_w"] * cp.sum(cp.abs(dx)))
    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None, None, "error"
    if x.value is None:
        return None, None, prob.status
    return np.clip(x.value, x_lo, x_hi), q.value, prob.status


class QueueEstimator:
    """Estimated cars on every link from entry counts and the controller's own signals (no cameras).

    Same store-and-forward model as the program, at 1 s resolution. Each link has a delay line (cars driving to
    the stop line, free-flow travel time) and a stop-line store q. Every second: cars due at the line join q; a
    green, unblocked link discharges min(q, s * 1 s) (none in the first startup_s of green); the discharge is
    split by the average turning shares into the downstream delay lines, or leaves the network. Entry links are fed
    by the entry-counter increments. Amounts are expected values.
    Spillback: a road cannot hold more than its storage, so queued cars above it are handed back to the roads that
    feed it (turning-share weights); they leave again on the next green. A road reported blocked holds pos_frac
    of its storage, and its feeders stop discharging once it is full (single lane: the head car waits). A hard
    room limit on every road was tried and dropped: with no measurement to correct it, the estimate locked up in
    a loop of full roads and stayed locked after the real network had cleared.
    """
    def __init__(self, net, cfg, T, storage, startup_s: float = 0.0, slot_s: float = 1.0):
        veh = VehicleConfig()
        L = np.full(net.n_links, cfg.link_length_m)
        L[net.entry_links] = cfg.entry_link_length_m
        self.net, self.T, self.storage, self.startup_s, self.dt = net, T, storage, startup_s, slot_s
        self.delay = np.maximum(1, np.round(L / veh.v_free_mps / slot_s)).astype(int)    # slots to the stop line
        self.K = int(self.delay.max()) + 2
        self.pipe = np.zeros((net.n_links, self.K))       # cars on the way, by (absolute slot % K) of arrival
        self.q = np.zeros(net.n_links)                    # cars at the stop line
        self.k = 0                                        # last slot processed
        self.prev_cum = np.zeros(len(net.entry_links))
        self._rows = np.arange(net.n_links)
        w = T.sum(axis=1, keepdims=True)
        self._back = np.divide(T, w, out=np.zeros_like(T), where=w > 0)     # [l, f] share of l's inflow from f

    @property
    def n(self):
        """Estimated cars on each link (queued + driving)."""
        return self.q + self.pipe.sum(axis=1)

    def near(self, within_s):
        """Cars at the stop line or due there within `within_s` seconds."""
        m = int(round(within_s / self.dt))
        idx = (self.k + 1 + np.arange(m)) % self.K
        return self.q + self.pipe[:, idx].sum(axis=1)

    def _add(self, links, amount, k):
        self.pipe[links, (k + self.delay[links]) % self.K] += amount

    def step(self, t, entry_cum, green, t_green, sat, blocked, cap=None):
        """Advance to time t. green, t_green: each link's signal state (and time in green) since the last call;
        blocked: reported incidents; cap: storage of each road (less the road beyond a reported block)."""
        k_now = int(np.floor(t / self.dt + 1e-9))
        cum = np.array(entry_cum, dtype=float)
        lim_all = self.storage if cap is None else cap
        for k in range(self.k + 1, k_now + 1):
            col = k % self.K
            self.q += self.pipe[:, col]
            self.pipe[:, col] = 0.0
            ok = green & ~blocked & (t_green + (k - self.k - 1) * self.dt >= self.startup_s)
            u = np.where(ok, np.minimum(self.q, sat * self.dt), 0.0)
            if blocked.any():                             # room behind a block: its feeders stop once it is full
                b = np.nonzero(blocked)[0]
                inflow = self.T[b] @ u
                frac = np.where(inflow > 1e-12, np.clip(lim_all[b] - self.n[b], 0.0, None) / np.maximum(inflow, 1e-12), 1.0)
                for j, bl in enumerate(b):
                    feeders = np.nonzero(self.T[bl] > 0)[0]
                    u[feeders] *= min(frac[j], 1.0)
            self.q -= u
            self._add(self._rows, self.T @ u, k)
            over = np.minimum(np.clip(self.n - lim_all, 0.0, None), self.q)   # spillback onto the feeders
            over[self.net.entry_links] = 0.0
            self.q += self._back.T @ over - over
        self.k = max(self.k, k_now)
        self._add(self.net.entry_links, cum - self.prev_cum, self.k)
        self.prev_cum = cum


class ForecastOptimiser(Controller):
    """Forecast-driven signal control (rolling-horizon convex QP) on counts-only sensing.

    Every second the queue estimator advances on the entry counts. Every `decision_s` seconds: forecast entry
    arrivals (`mode` sets the information level: persistence, patterns, live, perfect), pick a cycle length
    (Webster), solve for each intersection's NS green share over the horizon starting from the estimated cars on
    each link, and apply the first step. Between decisions the green is executed as durations; `gap_out` ends a
    green early when the estimate says its approaches are empty and the cross street has cars. Links reported
    blocked by the incident feed get no service, in the estimator and in the program (`incidents`).
    Incident fallback (`fallback_s`): from the moment an incident is reported until fallback_s after the feed reports
    it cleared, the signals run the fixed green-wave plan at fallback_cycle_s (the cycle tuned for the nominal
    morning) instead of the optimised plan. Reason: during and after a blockage the counts-only estimate is known to
    be wrong (it assumes the backlog drains at full saturation once the road reopens; spillback holds it far longer),
    so the robust plan is safer than optimising on it. The estimator and forecaster keep running; days with no
    reported incident are unaffected. fallback_s = 0 turns it off.
    The estimator discharges at est_sat x the nominal saturation flow after startup_s of green: calibrated on
    tuning-seed recordings (estimate vs simulator truth), not tuned for delay.
    """
    def __init__(self, mode: str = "live", decision_s: float = 60.0, horizon_windows: int = 3, l_eff: float = 12.0,
                 c_min: float = 40.0, c_max: float = 120.0, quad_w: float = 3.0, term_w: float = 3.0,
                 spill_w: float = 5.0, switch_w: float = 2.0, gap_out: bool = True, gap_s: float = 4.0,
                 gap_thr: float = 0.5, rain_cap_loss: float = 0.175, startup_s: float = 1.0, est_sat: float = 0.95,
                 incidents: bool = True, fallback_s: float = 3600.0, fallback_cycle_s: float = 120.0,
                 seed: int = 0, models_path: str = None):
        self.mode, self.decision_s, self.H = mode, decision_s, horizon_windows
        self.l_eff, self.c_min, self.c_max = l_eff, c_min, c_max
        self.quad_w, self.term_w, self.spill_w, self.switch_w = quad_w, term_w, spill_w, switch_w
        self.gap_out, self.gap_s, self.gap_thr = gap_out, gap_s, gap_thr
        self.rain_cap_loss, self.startup_s, self.est_sat, self.incidents = rain_cap_loss, startup_s, est_sat, incidents
        self.fallback_s, self.fallback_cycle_s = fallback_s, fallback_cycle_s
        kw = {"models_path": models_path} if models_path else {}
        self.fc = OnlineForecaster(mode, H=horizon_windows, seed=seed, **kw)

    def reset(self, cfg, net, rng):
        super().reset(cfg, net, rng)
        self.T = link_flow_matrix(net, cfg.turn_probs)
        veh = VehicleConfig()
        self.storage = np.full(net.n_links, float(cfg.storage_per_link(veh)))
        self.storage[net.entry_links] = float(int(cfg.entry_link_length_m // veh.slot_m))
        self.est = QueueEstimator(net, cfg, self.T, self.storage, self.startup_s)
        self.C, self.x = 60.0, np.full(net.n_int, 0.5)
        self.next_decision = 0.0
        self.n_solves, self.n_fail, self.plans = 0, 0, []
        self.blocked = np.zeros(net.n_links, dtype=bool)
        self.cap = self.storage.copy()                     # storage, less the road beyond a reported block
        self._green = np.zeros(net.n_links, dtype=bool)
        self._t_green = np.zeros(net.n_links)
        self.fallback_offsets = (net.int_rc.sum(axis=1) * cfg.free_flow_link_s(VehicleConfig())) % self.fallback_cycle_s
        self.in_fallback = False

    def attach(self, sim):
        self.sim = sim
        self.fc.attach(sim)

    def _nominal_sat(self):
        return self.cfg.saturation_flow * (1 - self.rain_cap_loss * self.fc.rain_mean_now())

    def _params(self):
        c = self.cfg
        return {"step_s": self.decision_s, "sat": self._nominal_sat(),
                "lost_s": c.lost_time_s, "min_green_s": c.min_green_s, "max_green_s": c.max_green_s,
                "quad_w": self.quad_w, "term_w": self.term_w, "spill_w": self.spill_w, "switch_w": self.switch_w}

    def _decide(self, t):
        N = int(round(self.H * self.fc.W / self.decision_s))
        lam = self.fc.lam_steps(t, self.decision_s, N)
        p = self._params()
        C_t, _ = webster_cycle(self.net, self.T, lam.mean(axis=1), p["sat"], self.l_eff, self.c_min, self.c_max)
        self.C = float(np.clip(round((0.6 * self.C + 0.4 * C_t) / 5.0) * 5.0, self.c_min, self.c_max))
        x, _, status = solve_splits(self.est.n, lam, self.x, self.C, self.net, self.T, self.cap, p, self.blocked)
        self.n_solves += 1
        if x is None:
            self.n_fail += 1
        else:
            self.x = x[:, 0]
        self.plans.append((t, self.C, self.x.copy()))

    def act(self, t, obs):
        net, cfg = self.net, self.cfg
        cap = self.storage.copy()
        if self.incidents:                         # incident feed: blocked while reported and not yet cleared
            self.blocked[:] = False
            for link, pos_frac, _, end in obs.get("incidents", ()):
                if end is None:
                    self.blocked[link] = True
                    cap[link] = pos_frac * self.storage[link]
        self.est.step(t, obs["entry_cum"], self._green, self._t_green, self.est_sat * self._nominal_sat(), self.blocked, cap)
        self.cap = cap
        phase, tin, sw = obs["phase"], obs["t_in_phase"], obs["switching"]
        self._green = (~sw[net.link_dest]) & (phase[net.link_dest] == net.link_phase)
        self._t_green = tin[net.link_dest]
        self.fc.update(t, obs, entry_occ=self.est.n[net.entry_links])
        if t + 1e-9 >= self.next_decision:
            self._decide(t)
            self.next_decision = t + self.decision_s
        share = np.where(phase == 0, self.x, 1.0 - self.x)
        target = np.clip(self.C * share - cfg.lost_time_s, cfg.min_green_s, cfg.max_green_s)
        req = phase.copy()
        done = (~sw) & (tin >= target)
        if self.gap_out:                           # reactive layer on the estimate: end an empty green early
            near = np.where(self.blocked, 0.0, self.est.near(self.gap_s)).reshape(net.n_int, 4)
            ns, ew = near[:, 0] + near[:, 2], near[:, 1] + near[:, 3]
            served, other = np.where(phase == 0, ns, ew), np.where(phase == 0, ew, ns)
            done |= (~sw) & (tin >= cfg.min_green_s) & (served < self.gap_thr) & (other >= 1.0)
        req[done] = 1 - phase[done]
        if self.incidents and self.fallback_s > 0:   # incident reported now or within fallback_s of clearing
            self.in_fallback = any(end is None or t < end + self.fallback_s for _, _, _, end in obs.get("incidents", ()))
            if self.in_fallback:                       # fixed green wave, as FixedTimeSynchronised
                x = (t + self.fallback_offsets) % self.fallback_cycle_s
                return np.where(x < 0.5 * self.fallback_cycle_s, 0, 1)
        return req
