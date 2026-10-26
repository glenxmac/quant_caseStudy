"""Core simulator: vehicle-level Intelligent Driver Model (IDM) on a generated grid.

Each step (dt):
  1. Poisson arrivals at entry links (turn sequence drawn at creation -> common random
     numbers: demand is identical whatever the controller does). Cars wait in an entry
     buffer if the link has no room; that wait counts as delay.
  2. Every control_every_s the controller sees its observations (entry counts, own signal
     state, incident feed) and requests a phase; the sim enforces min green and an all-red
     lost time on every switch.
  3. IDM acceleration for all cars (vectorised over the whole network). The head car on
     each link sees either the stop line (red, or no room downstream = box-clearing) as a
     stopped obstacle, or the tail of the next link as its leader. A car too close to stop
     before a late red ('dilemma zone') runs it.
  4. Cars past the stop line move to the next link (or exit); entry/exit are the sensors.

Cars-in = cars-out + cars-on-road + cars-waiting is checked in tests (conservation).
"""
from collections import deque

import numpy as np

from .config import SimConfig, VehicleConfig
from .demand import DemandModel
from .metrics import summarise
from .network import build_grid

MAX_TURNS = 64        # pre-drawn turn decisions per vehicle
B_MAX = 4.5           # m/s^2 hardest braking a driver will attempt (dilemma-zone rule)
A_MIN = -9.0          # physical floor on deceleration


class Simulation:
    """Vehicle-level traffic simulator (time-stepped, vectorised numpy).

    Cars follow the Intelligent Driver Model on a grid of signalised intersections. Each step: weather, arrivals,
    controller decision (every `control_every_s`), signal state, car dynamics, logging. Arrivals, routes and the controller
    use separate seeded random streams, so every controller sees identical demand for the same seed.
    """
    def __init__(self, cfg: SimConfig, controller, seed: int, demand: "DemandModel | None" = None,
                 vehicle: "VehicleConfig | None" = None, record: bool = False,
                 log_every_s: float = 1.0, control_every_s: float = 1.0,
                 rain=None, events=None, record_window=(0.0, float("inf")),
                 series_every_s: float = 60.0):
        self.cfg = cfg
        self.veh = vehicle or VehicleConfig()
        self.net = build_grid(cfg.grid_rows, cfg.grid_cols)
        self.controller = controller
        self.demand = demand or DemandModel(len(self.net.entry_links))
        # separate seeded streams; the third is unused (it was sensor noise) but kept so the controller stream is unchanged
        s_arr, s_route, _, s_ctrl = np.random.SeedSequence(seed).spawn(4)
        self.rng_arr = np.random.default_rng(s_arr)      # arrival counts
        self.rng_route = np.random.default_rng(s_route)  # turn decisions
        controller.reset(cfg, self.net, np.random.default_rng(s_ctrl))

        nl, ni = self.net.n_links, self.net.n_int
        self.t = 0.0
        self.step_idx = 0
        self.ctrl_every = max(1, int(round(control_every_s / cfg.dt)))
        self.log_every = max(1, int(round(log_every_s / cfg.dt)))
        self.record = record
        self.series_every = max(1, int(round(series_every_s / cfg.dt)))
        self.record_window = record_window
        self.rain = rain                       # callable t -> intensity in [0, 1]
        self.events = list(events or [])       # accidents
        self._weather()
        self.log = {"t": [], "link": [], "pos": [], "vel": [], "vid": [],
                    "phase": [], "switching": [], "exited": [], "buffered": [], "rain": []}

        # vehicle state (parallel arrays)
        self.link = np.zeros(0, dtype=np.int64)
        self.pos = np.zeros(0)
        self.vel = np.zeros(0)
        self.vid = np.zeros(0, dtype=np.int64)
        self.tcreate = np.zeros(0)
        self.hop = np.zeros(0, dtype=np.int64)
        self.origin = np.zeros(0, dtype=np.int64)
        self.turns = np.zeros((0, MAX_TURNS), dtype=np.int8)
        self.buffers = [deque() for _ in self.net.entry_links]
        self.next_id = 0
        self.dist = np.zeros(0)                          # distance on completed links
        self.Llink = np.full(nl, cfg.link_length_m)      # per-link length (entries are longer)
        self.Llink[self.net.entry_links] = cfg.entry_link_length_m
        # entry sensor series: what demand generated vs what the sensor actually saw
        n_e = len(self.net.entry_links)
        self.cum_created = np.zeros(n_e)
        self.cum_observed = np.zeros(n_e)
        self.cum_cross = np.zeros(nl)                    # stop-line detector: cars that left each link
        self.arrival_series = {"t": [], "created": [], "observed": [], "backlog": [],
                               "entry_count": [], "entry_queue": []}

        # signals
        self.phase = np.zeros(ni, dtype=np.int64)
        self.pending = np.zeros(ni, dtype=np.int64)
        self.switch_left = np.zeros(ni)
        self.t_in_phase = np.full(ni, cfg.min_green_s)

        # counters / stats
        self.created = 0
        self.exited = 0
        self.exit_times = []
        self.delays_done, self.origins_done, self.tc_done = [], [], []
        self.exits_post_warmup = 0
        self.sum_queue = np.zeros(nl)
        self.max_queue = 0
        self.spill_count = 0
        self.steps_post_warmup = 0
        self.min_gap_seen = np.inf
        self._last_blocked = np.zeros(nl, dtype=bool)

    # ------------------------------------------------------------------ helpers
    def _weather(self):
        r = float(self.rain(self.t)) if self.rain is not None else 0.0
        v = self.veh
        self.rain_now = r
        self._vf = v.v_free_mps * (1 - v.rain_speed_drop * r)
        self._Th = v.time_headway_s * (1 + v.rain_headway_gain * r)
        self._acc = v.accel_mps2 * (1 - v.rain_accel_drop * r)

    def _rear_info(self):
        nl = self.net.n_links
        rear_pos = np.full(nl, np.inf)
        rear_vel = np.zeros(nl)
        if len(self.pos):
            order = np.lexsort((-self.pos, self.link))
            l_s = self.link[order]
            tail = np.ones(len(order), dtype=bool)
            tail[:-1] = l_s[1:] != l_s[:-1]
            rear_pos[l_s[tail]] = self.pos[order][tail]
            rear_vel[l_s[tail]] = self.vel[order][tail]
        return rear_pos, rear_vel

    def _link_green(self) -> np.ndarray:
        d = self.net.link_dest
        return (self.switch_left[d] <= 0) & (self.phase[d] == self.net.link_phase)

    def _keep(self, mask):
        for name in ("link", "pos", "vel", "vid", "tcreate", "hop", "origin", "turns", "dist"):
            setattr(self, name, getattr(self, name)[mask])

    # ------------------------------------------------------------------ sensing
    def observe(self) -> dict:
        """What a controller may read: entry counters, its own signal state, the clock and the incident feed.
        `stop_cross` (stop-line counts per link) is simulator truth, kept for tests and diagnosis only."""
        return {"phase": self.phase.copy(), "t_in_phase": self.t_in_phase.copy(), "switching": self.switch_left > 0,
                "t": self.t, "entry_cum": self.cum_observed.copy(), "stop_cross": self.cum_cross.copy(),
                "incidents": self.incident_feed()}

    def incident_feed(self) -> list:
        """What the city's incident feed has reported by now: (link, pos_frac, start_s, end_s or None) for every
        accident that has started; the end time is reported only once the road has cleared."""
        return [(int(ev.link), float(ev.pos_frac), float(ev.start_s), float(ev.end_s) if self.t >= ev.end_s else None)
                for ev in self.events if self.t >= ev.start_s]

    # ------------------------------------------------------------------ steps
    def _arrivals(self):
        rates = self.demand.rates(self.t)
        counts = self.rng_arr.poisson(rates * self.cfg.dt)     # fixed draw count per step
        for k in np.nonzero(counts)[0]:
            for _ in range(counts[k]):
                turns = self.rng_route.choice(3, size=MAX_TURNS, p=self.cfg.turn_probs).astype(np.int8)
                self.buffers[k].append((self.next_id, self.t, turns))
                self.next_id += 1
                self.created += 1
                self.cum_created[k] += 1

    def _insert(self):
        if not any(self.buffers):
            return
        rear_pos, rear_vel = self._rear_info()
        Lv, v_free = self.veh.length_m, self._vf
        new = []
        for k, buf in enumerate(self.buffers):
            l = self.net.entry_links[k]
            if not buf:
                continue
            v = min(v_free, rear_vel[l]) if np.isfinite(rear_pos[l]) else v_free
            # enter only with the IDM equilibrium gap for the entry speed (no forced braking)
            if rear_pos[l] >= 2 * Lv + self.veh.min_gap_m + v * self._Th:
                vid, tc, trn = buf.popleft()
                new.append((l, vid, tc, trn, v, k))
                rear_pos[l] = Lv
                self.cum_observed[k] += 1                 # car passes the entry sensor
        if new:
            self.link = np.concatenate([self.link, [n[0] for n in new]]).astype(np.int64)
            self.pos = np.concatenate([self.pos, np.full(len(new), Lv)])
            self.vel = np.concatenate([self.vel, [n[4] for n in new]])
            self.vid = np.concatenate([self.vid, [n[1] for n in new]]).astype(np.int64)
            self.tcreate = np.concatenate([self.tcreate, [n[2] for n in new]])
            self.hop = np.concatenate([self.hop, np.zeros(len(new), dtype=np.int64)])
            self.origin = np.concatenate([self.origin, [n[5] for n in new]]).astype(np.int64)
            self.turns = np.vstack([self.turns, np.stack([n[3] for n in new])])
            self.dist = np.concatenate([self.dist, np.zeros(len(new))])

    def _control(self):
        req = np.asarray(self.controller.act(self.t, self.observe()))
        ok = (self.switch_left <= 0) & (req != self.phase) & (self.t_in_phase >= self.cfg.min_green_s)
        self.pending[ok] = req[ok]
        self.switch_left[ok] = self.cfg.lost_time_s

    def _advance_signals(self):
        dt = self.cfg.dt
        sw = self.switch_left > 0
        self.t_in_phase[~sw] += dt
        self.switch_left[sw] -= dt
        done = sw & (self.switch_left <= 1e-9)
        self.phase[done] = self.pending[done]
        self.t_in_phase[done] = 0.0
        self.switch_left[done] = 0.0

    def _dynamics(self):
        n = len(self.pos)
        net, cfg, v = self.net, self.cfg, self.veh
        if n == 0:
            self._last_blocked = np.zeros(net.n_links, dtype=bool)
            return
        Lv, s0 = v.length_m, v.min_gap_m
        order = np.lexsort((-self.pos, self.link))
        l_s, p_s, v_s = self.link[order], self.pos[order], self.vel[order]
        head = np.ones(n, dtype=bool)
        head[1:] = l_s[1:] != l_s[:-1]
        tail = np.ones(n, dtype=bool)
        tail[:-1] = l_s[1:] != l_s[:-1]
        gap = np.full(n, np.inf)
        vlead = np.zeros(n)
        gap[1:] = np.where(~head[1:], p_s[:-1] - Lv - p_s[1:], np.inf)
        vlead[1:] = v_s[:-1]
        rear_pos = np.full(net.n_links, np.inf)
        rear_vel = np.zeros(net.n_links)
        rear_pos[l_s[tail]] = p_s[tail]
        rear_vel[l_s[tail]] = v_s[tail]
        # link is 'spilled back' when its stopped tail has no room for another car to enter
        self._last_blocked = (rear_pos < Lv + v.slot_m) & (rear_vel < 1.0)

        # head car of every link: stop-line obstacle or next link's tail as leader
        h = np.nonzero(head)[0]
        hl, hp, hv = l_s[h], p_s[h], v_s[h]
        hidx = order[h]
        nxt = net.next_link[hl, self.turns[hidx, self.hop[hidx]]]
        has_next = nxt >= 0
        nxt_s = np.where(has_next, nxt, 0)
        dist = np.maximum(self.Llink[hl] - hp, 0.0)
        green = self._link_green()[hl]
        room = ~has_next | (rear_pos[nxt_s] >= v.slot_m)       # box-clearing
        committed = hv ** 2 / (2 * B_MAX) >= dist              # cannot stop before the line
        obstacle = (green & ~room) | (~green & ~committed)
        r_pos = rear_pos[nxt_s]
        see_next = has_next & np.isfinite(r_pos)
        gap[h] = np.where(obstacle, dist, np.where(see_next, dist + r_pos - Lv, np.inf))
        vlead[h] = np.where(obstacle, 0.0, np.where(see_next, rear_vel[nxt_s], 0.0))

        # accidents: the front-most car behind the blockage sees a stopped obstacle
        if self.events:
            inv = np.empty(n, dtype=np.int64)
            inv[order] = np.arange(n)
            for ev in self.events:
                if ev.start_s <= self.t < ev.end_s:
                    p_ev = ev.pos_frac * self.Llink[ev.link]
                    cand = np.nonzero((self.link == ev.link) & (self.pos < p_ev))[0]
                    if len(cand):
                        i = cand[np.argmax(self.pos[cand])]
                        g_ev = p_ev - self.pos[i]
                        if g_ev < gap[inv[i]]:
                            gap[inv[i]], vlead[inv[i]] = g_ev, 0.0

        gap = np.maximum(gap, 0.1)
        dv = v_s - vlead
        s_star = np.maximum(s0 + v_s * self._Th + v_s * dv / (2 * np.sqrt(self._acc * v.decel_mps2)), s0)
        acc = self._acc * (1 - (v_s / self._vf) ** 4 - (s_star / gap) ** 2)
        acc = np.clip(acc, A_MIN, self._acc)
        v_unc = v_s + acc * cfg.dt
        v_new = np.maximum(v_unc, 0.0)
        step = np.where(v_unc >= 0, 0.5 * (v_s + v_new) * cfg.dt, v_s ** 2 / (2 * np.maximum(-acc, 1e-9)))
        p_new = p_s + step
        if n > 1:
            g_new = np.where(~head[1:], p_new[:-1] - Lv - p_new[1:], np.inf)
            self.min_gap_seen = min(self.min_gap_seen, float(g_new.min()))
        self.pos[order] = p_new
        self.vel[order] = v_new

    def _transfers(self):
        if len(self.pos) == 0:
            return
        slot = self.veh.slot_m
        cross = np.nonzero(self.pos > self.Llink[self.link])[0]
        if len(cross) == 0:
            return
        cross = cross[np.argsort(-self.pos[cross])]
        rear_pos, _ = self._rear_info()
        remove = np.zeros(len(self.pos), dtype=bool)
        for i in cross:
            Lcur = self.Llink[self.link[i]]
            nx = self.net.next_link[self.link[i], self.turns[i, self.hop[i]]]
            if nx < 0:                                          # exit sensor
                self.cum_cross[self.link[i]] += 1
                remove[i] = True
                self.exited += 1
                self.exit_times.append(self.t)
                if self.t >= self.cfg.warmup_s:
                    self.exits_post_warmup += 1
                if self.tcreate[i] >= self.cfg.warmup_s:
                    dist = self.dist[i] + Lcur
                    self.delays_done.append(max(0.0, self.t - self.tcreate[i] - dist / self.veh.v_free_mps))
                    self.origins_done.append(self.origin[i])
                    self.tc_done.append(self.tcreate[i])
                continue
            over = self.pos[i] - Lcur
            if rear_pos[nx] >= over + slot:
                self.cum_cross[self.link[i]] += 1
                self.link[i] = nx
                self.pos[i] = over
                self.hop[i] += 1
                self.dist[i] += Lcur
                rear_pos[nx] = min(rear_pos[nx], over)
            else:                                               # no room: hold at the line
                self.pos[i] = Lcur - 0.01
                self.vel[i] = 0.0
        if remove.any():
            self._keep(~remove)

    def _record(self):
        nl = self.net.n_links
        if self.step_idx % self.series_every == 0:
            sr = self.arrival_series
            sr["t"].append(self.t); sr["created"].append(self.cum_created.copy())
            sr["observed"].append(self.cum_observed.copy())
            sr["backlog"].append(np.array([len(b) for b in self.buffers]))
            ent = self.net.entry_links                  # cars on each entry road (truth: the forecaster's censoring flag)
            sr["entry_count"].append(np.bincount(self.link, minlength=nl)[ent] if len(self.pos) else np.zeros(len(ent), int))
            sr["entry_queue"].append(np.bincount(self.link[self.vel < 2.0], minlength=nl)[ent] if len(self.pos) else np.zeros(len(ent), int))
        if self.t >= self.cfg.warmup_s:
            q = np.bincount(self.link[self.vel < 0.5], minlength=nl) if len(self.pos) else np.zeros(nl)
            self.sum_queue += q
            self.max_queue = max(self.max_queue, int(q.max()))
            self.spill_count += int(self._last_blocked.sum())
            self.steps_post_warmup += 1
        if self.record and self.step_idx % self.log_every == 0 and self.record_window[0] <= self.t < self.record_window[1]:
            lg = self.log
            lg["t"].append(self.t)
            lg["link"].append(self.link.copy()); lg["pos"].append(self.pos.copy())
            lg["vel"].append(self.vel.copy()); lg["vid"].append(self.vid.copy())
            lg["phase"].append(self.phase.copy()); lg["switching"].append(self.switch_left > 0)
            lg["exited"].append(self.exited); lg["buffered"].append(self.buffered)
            lg["rain"].append(round(self.rain_now, 2))

    def step(self):
        self._weather()
        self._arrivals()
        self._insert()
        if self.step_idx % self.ctrl_every == 0:
            self._control()
        self._advance_signals()
        self._dynamics()
        self._transfers()
        self._record()
        self.t += self.cfg.dt
        self.step_idx += 1

    # ------------------------------------------------------------------ results
    @property
    def buffered(self) -> int:
        return sum(len(b) for b in self.buffers)

    def run(self) -> dict:
        for _ in range(int(round(self.cfg.horizon_s / self.cfg.dt))):
            self.step()
        return self.results()

    def delay_series(self, bin_s: float = 600.0):
        """Mean delay of cars grouped by the time they ARRIVED at the network (entry), including
        cars still driving or waiting (delay so far). Returns (bin_centres_s, mean_delay, counts)."""
        cfg, v0 = self.cfg, self.veh.v_free_mps
        tc, d = list(self.tc_done), list(self.delays_done)
        for i in np.nonzero(self.tcreate >= cfg.warmup_s)[0]:
            tc.append(self.tcreate[i])
            d.append(max(0.0, self.t - self.tcreate[i] - (self.dist[i] + self.pos[i]) / v0))
        for buf in self.buffers:
            for _, t0, _ in buf:
                if t0 >= cfg.warmup_s:
                    tc.append(t0); d.append(self.t - t0)
        edges = np.arange(0.0, self.t + bin_s, bin_s)
        idx = np.digitize(tc, edges) - 1
        mean = np.full(len(edges) - 1, np.nan)
        cnt = np.zeros(len(edges) - 1, dtype=int)
        tc, d = np.asarray(tc), np.asarray(d)
        for b in range(len(edges) - 1):
            m = idx == b
            cnt[b] = int(m.sum())
            if cnt[b]:
                mean[b] = d[m].mean()
        return 0.5 * (edges[:-1] + edges[1:]), mean, cnt

    def results(self) -> dict:
        cfg, v0 = self.cfg, self.veh.v_free_mps
        delays, origins = list(self.delays_done), list(self.origins_done)
        for i in np.nonzero(self.tcreate >= cfg.warmup_s)[0]:        # still driving: delay so far
            dist = self.dist[i] + self.pos[i]
            delays.append(max(0.0, self.t - self.tcreate[i] - dist / v0)); origins.append(self.origin[i])
        for k, buf in enumerate(self.buffers):                       # still waiting to enter
            for _, tc, _ in buf:
                if tc >= cfg.warmup_s:
                    delays.append(self.t - tc); origins.append(k)
        steps = max(self.steps_post_warmup, 1)
        window = max(self.t - cfg.warmup_s, 1e-9)
        res = summarise(delays, origins, len(self.net.entry_links), self.exits_post_warmup, window,
                        self.sum_queue.sum() / steps / self.net.n_links, self.max_queue,
                        self.spill_count / steps / self.net.n_links)
        res.update(created=self.created, exited=self.exited, on_road=len(self.pos),
                   buffered=self.buffered, min_gap_m=self.min_gap_seen)
        return res
