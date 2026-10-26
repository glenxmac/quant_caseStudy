# Smart Traffic Signal Control (Quant Case Study, Problem 1)

Forecast-driven signal control on a 2x2 intersection grid that sees only a car counter at each entrance.
The pipeline is **count -> forecast -> model the queues -> optimise -> measure wait time**, compared against
fixed-time lights (an untuned 60 s plan and a tuned-cycle green wave).

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python main.py history --n-days 60 && python main.py forecast --final   # forecaster data, models and the ladder table
for sc in am_ramp am_busy am_rain pm_ramp; do PYTHONPATH=src python scripts/eval_chunk.py $sc 200 3; done   # held-out evaluation -> outputs/final_eval_counts/
PYTHONPATH=src python scripts/eval_chunk.py am_ramp 203 7                                   # typical morning: 10 seeds in all
PYTHONPATH=src python scripts/eval_chunk.py am_accident 200 5; PYTHONPATH=src python scripts/eval_chunk.py am_accident 205 5   # accident: 10 seeds
PYTHONPATH=src python scripts/eval_variants.py am_ramp 200 3; PYTHONPATH=src python scripts/eval_variants.py am_accident 200 10   # ablations: no gap-out, no incident feed
PYTHONPATH=src python scripts/estimator_drift.py am_ramp 100 && PYTHONPATH=src python scripts/estimator_drift.py summary   # queue estimate vs truth
PYTHONPATH=src python scripts/results_chart.py          # headline chart -> outputs/results_delay.png
python main.py forecast-plots                 # forecaster figures -> outputs/forecast_*.png
python main.py viz-clips                      # demo viewer: normal hour / rush / storm / accident, fixed vs smart -> outputs/demo_viewer.html
PYTHONPATH=src python scripts/scalability.py 10 10      # optimiser timing on a 10x10 grid
python main.py tune-scen                      # fixed-plan cycle tuning on nominal days (cached in outputs/tuned_scenarios.json)
PYTHONPATH=src python scripts/sweep.py 0 11 && PYTHONPATH=src python scripts/sweep.py summary   # optimiser hyper-parameter check
```

## Figures (`outputs/`)
| File | Shows |
|---|---|
| `results_delay.png` | Headline: mean delay per car by scenario, both fixed plans vs the optimiser |
| `forecast_ladder_by_horizon.png` | Forecast error at +5/+10/+15 min: raw count, profile, + weather, + live counts, oracle |
| `estimator_drift.png` | The controller's estimated cars on a road vs the truth (normal morning, accident) |
| `forecast_two_days.png` | Network arrivals vs forecasts over a typical day and an accident day |
| `forecast_day.png` | The censoring guard on a gridlocked day (counts drop to zero, the guarded forecast does not) |
| `forecast_regimes.png`, `forecast_skill.png` | Forecast error by day type; skill on validation, fresh and test days |
| `demand_days.png` | The true demand envelope over 300 random days |
| `demo_viewer.html` | The live demo (open in a browser) |

## Layout
- `src/traffic/` - simulator (`sim.py`), demand and scenarios, `controllers/` (fixed-time, `optimiser.py`), `forecasting/` (history, features, models, evaluation, online wrapper), `experiments/`, `viz/`
- `scripts/` - experiment scripts (held-out evaluation, ablations, estimator drift, optimiser sweep, scalability timing)
- `tests/` - unit and integration tests (`pytest`)
- `outputs/` - generated results, charts and the demo viewer. Submitted: the fitted forecaster, held-out results, tuned parameters, summary tables, figures and the demo viewer; raw history and caches are git-ignored (regenerate with `main.py`). `outputs/final_eval/` holds the earlier camera-based version's results, kept for comparison.
- `docs/` - the original case study brief (unchanged)
- `presentation/` - the slide deck

## Assumptions
**Network**
- Grid of `rows x cols` intersections (default 2x2). Everything is generated from the grid size, so larger networks (10+) are a config change, not new code.
- Every intersection is 4-way, one lane per approach, turning cars share the lane, 2 signal phases (NS / EW). Turning movements are assumed conflict-free within their phase (no permissive-left modelling).
- Network boundary = the 8 entry and 8 exit road ends of the 2x2. Vehicles enter at entries, take a turn at each intersection with fixed probabilities (70% straight / 15% left / 15% right), and leave at an exit.
- Internal links are 200 m (stop line to stop line); entry roads are 400 m. Exit roads are not modelled (nothing blocks them; cars are counted and removed at the last stop line).

**Vehicles**
- One standard car: 4.5 m long, 2 m standstill gap (6.5 m per queued car, so ~30 cars fit on a 200 m link).
- Car-following: Intelligent Driver Model (v0 = 13.9 m/s, T = 1.5 s, a = 1.5 m/s^2, b = 2.0 m/s^2). Saturation flow is emergent and calibrated (0.40 veh/s), not hard-coded.
- No lane changes, overtaking, pedestrians or emergency vehicles (bonus items, out of scope).

**What the controller can see (counts only)**
- A car counter on each entry road: cumulative cars that entered. Nothing inside the network: no cameras, no queue or occupancy sensors, no stop-line detectors.
- The clock, a weather feed (rain intensity now, and a noisy rain forecast), and its own signal state.
- An incident feed: the city reports "road X blocked" when an accident happens and reports when it clears.
- It knows the road layout, link lengths, the average turning shares (from a past survey) and the nominal saturation flow.
- Queues inside the network are therefore not measured but estimated, by the controller's own queue model. The simulator knows the true queues; they are used only for scoring and for the drift diagnostic.

**Demand**
- Arrivals at entries are Poisson with a time-varying rate lambda(t) = daily profile x directional tide x today's level and timing shift x rain effect x slow AR(1) noise. The Poisson part is the count noise around the rate: a 5-minute count of 30-90 cars has 10-20% pure noise.
- The entry counter sees cars that actually entered. Once an entry road is full, cars wait outside and the count falls below demand (censoring).
- The forecaster learns from true rates (stand-in for an accurate historical traffic survey), reads only live entry counts as sensor inputs, and is scored against the true rate. See 'Forecaster'.

**Time-varying scenarios** (`src/traffic/scenarios.py`)
- Demand follows a weekday profile (AM peak ~07:45, PM peak ~17:00) times a day-level effect and AR(1) noise. A directional tide makes AM entries inbound-heavy and PM outbound-heavy.
- The nominal day peaks at ~0.85 of the grid's saturation knee (busy but manageable); disturbances (a 15% busier day, rain, an accident) push it toward saturation.
- Rain adds ~10% demand and changes driving: free speed -20%, headway +30%, acceleration -15% at full intensity. Measured effect: saturation flow falls from 0.40 to 0.33 veh/s (-17%).
- An accident blocks the single lane of one road for 15 minutes at the peak. There is no rerouting: cars queue behind it and are released when it clears.
- Baselines are tuned on the NOMINAL day only (seeds 100-103), then applied unchanged to the disturbed variants.

**Random days** (`src/traffic/daygen.py`, forecaster history)
- Each day draws a day-level demand effect (lognormal, sigma 0.10), a timing shift of the whole daily pattern (sd 12 min), storms (Poisson, ~0.8 per day) and accidents (Poisson, 0.10/h, up to 3x in rain). These are assumptions, not calibrated to data.

**Randomness**
- Each vehicle's arrival time and turning decisions are drawn at spawn time from per-purpose seeded streams, so every controller sees identical demand (common random numbers).

## Forecaster
**Job.** Predict the arrival rate at each of the 8 entrances for the next 5, 10 and 15 minutes, once per 5-minute window, using only what is known at that moment. One global model with the horizon as a feature.

**Inputs.** Time of day (sin/cos harmonics of the hour being forecast), weekday/weekend, the entrance (one-hot), the horizon, rain now and the noisy rain forecast, and the live entrance counts (last 6 five-minute counts, their 30-minute mean, and the network total). **Target:** the true rate in the window ahead (labels from the simulator's survey stand-in, never an input).

**Model.** Poisson GLM (log link, entry x clock interactions; the Poisson likelihood fits count data and the log link makes the effects multiply like the demand does) with a censoring guard: when an entry road is estimated full (>= 55 cars of ~61) in the last 15 minutes, the counts describe the blockage, not the demand, so the forecast falls back to the hour profile times a multiplier built from clean data. Gradient boosting (Poisson loss) is a check. Model selection is light: the GLM's regularisation from three values and four GBM settings, picked on validation days.

**Data.** 60 simulated days (05:00-21:00) under the 60 s fixed plan, split by whole days in time order (40 train / 10 validation / 10 test), plus 10 fresh days never used for any decision.

**Ladder by horizon** (error = mean absolute error as % of the mean true rate; 20 held-out days: test + fresh; `outputs/forecast_ladder_by_horizon.{csv,md,png}`):

| Model | +5 min | +10 min | +15 min | Overall |
|---|---|---|---|---|
| Raw count carried forward (persistence) | 22.8 | 23.6 | 24.7 | 23.7 |
| Hour-of-day profile (mean over training days, entry x weekday/weekend x 15 min) | 10.0 | 9.8 | 9.7 | 9.8 |
| + weather (profile x rain) | 9.7 | 9.6 | 9.5 | 9.6 |
| **+ live entrance counts: Poisson GLM + guard (main)** | **6.4** | **6.7** | **7.3** | **6.8** |
| Gradient boosting + live counts + guard (check) | 6.5 | 6.9 | 7.3 | 6.9 |
| Oracle: true rate now carried forward (unreachable) | 4.5 | 7.4 | 9.7 | 7.2 |

How to read it:
- The profile knows the typical day; what it misses is today (level ~10% off, rush hour shifted by ~12 minutes). Weather adds 0.2-0.3 points.
- The raw count is the worst predictor because a single 5-minute count is mostly Poisson noise. The live-count model turns those noisy counts into an estimate of today's level, which cuts the profile's error by about a third.
- The oracle knows today's true rate exactly but assumes it stays flat. It wins at +5 min; from +10 min the model is better, because it also knows the daily shape (e.g. the rush still ramping up). Most of the model's skill is a good nowcast plus the daily profile; changes beyond that are random and nothing in the inputs predicts them.
- Validation, test and fresh days separately give the same picture (`outputs/forecast_ladder_by_horizon.md`).

**Original mistake and correction.**
1. *Labels vs inputs.* The first design mixed up "trained on the true rate" and "trained on observed counts". Now stated plainly: labels = true rate (an accurate historical survey, available only because this is a simulator), inputs = what a deployed system could see.
2. *Censoring.* When an entry road fills, the counter reports the blockage, not the demand. On one test day (gridlock) every learned model collapsed. Fix: the guard above (threshold chosen from the road's capacity after seeing that failure, then confirmed on 10 fresh days: guarded 6.6-6.8%, unguarded 8.4-8.5%). Offline the guard's flag uses the simulator's count of cars on the entry road; online it uses the controller's own estimate of it.
3. *Fewer sensors, same accuracy.* An earlier version also read whole-road occupancy. Removing it changed nothing (6.82% before and after), so the forecaster now uses entrance counts only.
4. *Scoring against sensors.* Scored against the noisy counts every model lands within ~1.3 points of the others; against the true rate they separate clearly. Both are reported.

Limits: the data comes from our own demand model, so the results show recovery of a process we wrote, not real-world accuracy; rain windows are about 7% of the data.

## Approach
**Pipeline.** Entrance counts -> forecaster -> queue estimator -> optimiser -> signals, on four timescales:

| Timescale | Layer | Uses |
|---|---|---|
| every second | estimate: advance the queue model on the entrance counts and own signals | counts, own signal state |
| every second | reactive: switch when the planned green has run out; gap-out when the model says the approach is empty | the estimate |
| every 60 s | strategy: cycle length and green splits for the next 15 minutes (convex QP) | the forecast and the estimate |
| every day | learning: refit the forecaster | history |

**Queue estimator** (`QueueEstimator` in `src/traffic/controllers/optimiser.py`). The same store-and-forward model the optimiser plans with, stepped every second. Each road has a travel-time delay line (cars driving to the stop line at free-flow speed) and a stop-line queue. Entry roads gain the counter increments; a green, unblocked road discharges min(queue, saturation flow x 1 s); discharge is split downstream by the average turning shares; queued cars above a road's storage are handed back to its feeders (spillback). A road reported blocked discharges nothing. The discharge rate (0.95 x nominal, rain-adjusted) and a 1 s start-up delay were calibrated by replaying tuning-seed runs against the simulator's truth, not tuned for delay.

**Optimiser.** Every 60 s, for every intersection, solve a convex quadratic program over a 15-minute horizon (15 steps of 60 s), apply the first step, and re-plan (receding horizon).

```
decide     x[i,k]   NS green share of intersection i in step k      (EW share = 1 - x)
state      q[l,k]   cars on link l                                  q[:,0] = the estimator's cars now
flow       u[l,k]   cars discharged                                 u <= s*D*g(x) ,  u <= q + arrivals ,  u = 0 on blocked links
           q[k+1] = q[k] + arrivals - u ;  arrivals = forecast on entry links, turning-weighted discharge upstream
constrain  min/max green (8 s / 60 s) and 4 s lost time per switch bound x ; q <= storage (soft: spillback slack)
minimise   sum q + w_q * sum q^2 + w_T * q_end + w_s * spillback slack + w_x * sum |change in x|
```
- `sum q` is total waiting (vehicle-seconds); the quadratic term spreads queues across approaches (fairness); the terminal term stops the plan dumping work past the horizon; the switching term stops the plan flapping.
- The cycle length C comes from Webster's formula on the forecast flows (smoothed, clamped to 40-120 s, rounded to 5 s). Keeping C outside the program keeps it convex.
- The plan is executed as green durations: switch when time-in-phase reaches `C * share - lost time`. Gap-out ends a green early when the estimate says fewer than 0.5 cars are due at its stop lines within 4 s and the cross street has at least one.
- Saturation flow is 0.40 veh/s, scaled down with the rain feed (-17% in full rain).
- **Incident fallback.** From the moment an incident is reported until 60 minutes after it clears, the signals run the tuned fixed green wave (120 s) instead of the optimised plan. During and after a blockage the counts-only estimate is known to be wrong (it assumes the backlog drains at full saturation once the road reopens; spillback holds it far longer), so the robust plan is safer than optimising on a wrong state. Days with no incident are unaffected.
- Information level is a switch (`mode`): `persistence` (repeat the last window's count), `patterns` (calendar + weather only), `live` (the full forecaster), `perfect` (oracle: the true rate). This is the value-of-information ladder.

**Why this method.** The queue model is known and linear, so the decision is a convex program: fast, globally optimal per solve and easy to explain. The same model serves as the state estimator and as the planning model. Discrete alternatives (mixed-integer phase sequencing) are not needed because the only decisions are continuous green shares. Reinforcement learning was not attempted (see limitations).

**Why not more models or solvers.** The results say where the remaining error is, and it is not in the model class:
- *Forecaster.* The GLM and gradient boosting are level (6.8% vs 6.9%), so a more flexible model finds nothing more in these inputs. What is left is today's random variation, which no input sees coming. And even a perfect forecast does not lower delay (value-of-information table), so a better forecaster cannot improve the headline.
- *Optimiser.* The QP is solved to global optimality for its model, and the hyper-parameter check is flat around the defaults. The weak spots (storm, accident, tail delay) come from the queue estimate drifting, because nothing inside the network is measured, not from the solver. The next gain would come from a sensor (a stop-line counter per road) or from robust planning over the estimate's uncertainty, not from another algorithm.

**Experiment design.**
- Scenarios: typical morning (`am_ramp`), 15% busier morning, storm across the peak, accident at the peak, typical evening. Same scenario + seed = identical demand for every controller (common random numbers).
- Baselines: untuned fixed 60 s green wave; fixed green wave with the cycle tuned on the typical day (120 s morning, 180 s evening). Both tuned only on nominal days and applied unchanged to the disturbed scenarios.
- The optimiser's hyper-parameters were checked on nominal days and tuning seeds only, one at a time around the defaults (`outputs/sweep_summary.csv`): the defaults scored best or tied, and only gap-out moves the result materially. Gap-out on/off and its threshold were chosen on tuning seeds. The incident fallback was chosen on accident tuning seeds 100-109 (`outputs/accident_tuning_counts.csv`).
- Seeds: tuning 100-109, held-out evaluation 200-202 (typical morning and accident 200-209), demo clips 300. Forecaster training days use their own seeds (0-59 plus fresh 5000+).

## Results
Held-out evaluation, mean delay per car in seconds (lower is better). Typical morning and accident: 10 seeds (200-209); the others: 3 seeds (200-202). `outputs/final_eval_counts/`.

| Controller | Typical morning | Busy morning | Storm | Evening | Accident |
|---|---|---|---|---|---|
| Fixed 60 s (untuned) | 114.1 | 265.5 | 294.7 | 245.4 | 417.1 |
| Fixed, tuned green wave | 72.2 | 142.1 | 199.6 | 95.0 | 242.1 |
| **Optimiser, live forecast** | **48.5** | **107.1** | **182.8** | **82.3** | **227.6** |

Improvement of the optimiser in mean delay:

| | Typical | Busy | Storm | Evening | Accident |
|---|---|---|---|---|---|
| vs untuned fixed 60 s | 58% | 60% | 38% | 66% | 45% |
| vs tuned green wave | **33%** | **25%** | 8% | 13% | 6% |
| seconds saved vs tuned, 95% interval | 23.7 +- 4.8 (n=10) | 35.0 +- 27.9 | 16.8 +- 60.0 | 12.7 +- 12.3 | 14.5 +- 22.1 (n=10) |

- The brief's target ("20%+ over fixed-timing lights") is met against the untuned plan in every scenario, and against the tuned green wave on the typical and busy mornings. It is not met against the tuned plan in the storm, the evening or the accident. The tuned baselines are deliberately strong; real fixed plans are rarely tuned this well.
- The typical, busy and evening savings exclude zero; storm and accident do not (3 and 10 seeds).
- **Tail delay** (95th percentile) improves on the typical (216 -> 140 s), busy (483 -> 404 s) and accident (1032 -> 1013 s) runs, and is worse in the storm (693 -> 769 s) and the evening (272 -> 289 s). The optimiser lowers the average by serving the heavy approaches; the cost lands on cars on the light ones.
- Throughput is unchanged between controllers (all clear the same demand). Spillback falls in the busy scenario (0.031 -> 0.020).
- **Compared with the earlier camera-based version** (same seeds 200-202, vs tuned): typical 22% -> 31%, busy 22% -> 25%, evening 3% -> 13%, storm 16% -> 8%. Removing the cameras did not cost accuracy on normal days; the gain there comes from the model-based gap-out. The storm is the one scenario where the counts-only model is worse.

**What part of the gain is the optimisation.** Gap-out off (`scripts/eval_variants.py`, typical morning seeds 200-202): QP alone 53.3 s (11% better than tuned), QP + gap-out 41.5 s (31%). In light traffic most of the gain is the reactive layer acting on the queue model; at the busy peak the QP does most of the work.

**Queue estimate vs truth** (`scripts/estimator_drift.py`, `outputs/estimator_drift.png`, tuning seeds). On a typical morning the estimate is off by 0.3-0.5 cars on entry roads and 1.1-1.3 on internal roads (true counts ~6 and ~4). In the storm the network total runs 7-16 cars high during the rain (tuning seeds 100-102; on the demo seed about 30 cars, ~10%). In the accident it tracks the blockage, but after clearance it assumes the backlog drains at full saturation, so it reads too low on the roads around it (entry-road error 2.7-5.8 cars on accident days). Before the incident fallback existed, the optimiser acted on that low estimate, under-served the backlog and the error lasted about 30 minutes; this is why the fallback exists. The estimate is sensitive to the assumed discharge rate (+-5% shifts the entry-road bias by several cars in congestion) and nothing measured inside the network corrects it.

**Accidents.** Ten seeds (200-209). `am_accident` is the typical morning plus the accident with identical demand per seed, so the accident's own cost is the difference between the two runs:

| Controller | Typical morning | With accident | Cost of the accident (s per car) | Worst seed | Gridlocked seeds |
|---|---|---|---|---|---|
| Fixed 60 s (untuned) | 114.1 | 417.1 | +303 | 1189 | 1 |
| Fixed, tuned green wave | 72.2 | 242.1 | +170 | 400 | 0 |
| **Optimiser (with incident fallback)** | 48.5 | **227.6** | +179 | 309 | 0 |
| Optimiser ignoring the incident feed | 41.5 (3 seeds) | 461.8 | - | 1158 | 2 |
| Optimiser without gap-out | 53.3 (3 seeds) | 241.0 | - | 376 | 0 |

- The accident adds about 170-180 s per car under any controller that keeps the network moving, about three quarters of all delay in that scenario. Cars behind a 15-minute single-lane blockage wait for it to clear and there is no rerouting, so no signal plan can remove that. The controller's job is damage limitation.
- The optimiser wins the accident day (6%) because it is ahead before the accident; the accident itself costs it slightly more than the tuned plan.
- Without the incident feed the counts-only controller cannot know a road is blocked, and it gridlocks the network on two of ten seeds: the feed is essential.
- How the fallback was chosen (tuning seeds 100-109): no fallback 336.9 s (one gridlock, four near), fallback only while blocked 392.9 s (worse: the damage is done after clearance), fallback until 15/30/45/60 min after clearance 218.0/213.6/210.4/209.8 s, tuned fixed 225.0 s. Also tried and rejected: forcing the longest cycle during the incident (270.9 s) and a reduced discharge rate near the incident in the estimator (414.8 s). In this scenario the 60-minute window covers almost all of the rest of the run, so the honest statement is "after a blockage, run the fixed plan for an hour".

**Value of information.** Mean delay by information level (same seeds as the main table):

| Information | Typical | Busy | Storm | Evening | Accident |
|---|---|---|---|---|---|
| Last window's count repeated (persistence) | 48.9 | 109.2 | 164.3 | 84.7 | 227.1 |
| Calendar + weather only (patterns) | 49.1 | 119.6 | 185.9 | 84.9 | 222.9 |
| Live forecast (the model) | 48.5 | 107.1 | 182.8 | 82.3 | 227.6 |
| Perfect forecast (oracle) | 49.3 | 110.0 | 181.7 | 86.1 | 226.9 |

The forecast does not matter much for delay: persistence, the learned forecast and the oracle are within noise of each other (the storm's spread is 3 seeds of noise). This is despite the learned forecast being a much better forecast than the profile (6.8% vs 9.8% error). The reason: at a 15-minute horizon the forecast's value is mostly "today's level now", and the controller re-plans every minute from counts that already carry it. The queue model and the optimiser are where the gain is. A forecast should matter when information arrives before the cars do (e.g. a scheduled event surge) or when plans must be committed further ahead; neither is in these scenarios.

**Demo.** `python main.py viz-clips` builds `outputs/demo_viewer.html`: the tuned fixed plan against the optimiser on identical traffic for a normal hour, the morning peak, the storm and the accident. The smart panel shows what the controller can see (entrance counters), its estimated cars per road next to the truth, its planned split and cycle, and a chart of counted arrivals, the forecast and the true demand (hidden from the controller). Clip windows were chosen by a stated rule (the worst quarter-hour of the fixed plan across held-out seeds); each clip is one random day (seed 300) and the viewer shows the held-out scenario average next to it. Open at a moment with `#clip=3&f=600`.

**Scalability** (20 simulated minutes, `scripts/scalability.py`, `outputs/scalability.csv`). One central QP per decision:

| Grid | Intersections | Mean solve | Max solve | Delay, untuned fixed vs optimiser |
|---|---|---|---|---|
| 2x2 | 4 | 38 ms | 49 ms | 45 vs 39 s |
| 4x4 | 16 | 203 ms | 237 ms | 79 vs 73 s |
| 6x6 | 36 | 493 ms | 621 ms | 109 vs 104 s |
| 10x10 | 100 | 1.7 s | 2.0 s | 147 vs 132 s |

Cost grows roughly linearly (about 17 ms per intersection) against a 60 s decision interval; the queue estimator adds little. These larger-grid runs used the model-free persistence forecast, because the learned forecaster is fitted to the 2x2 grid's 8 entry roads, and nothing was tuned for them (one seed, 20 minutes, untuned fixed plan as the reference).

## Key decisions and limitations
**Decisions**
- Counts only: the controller sees entrance counters, the clock, weather and an incident feed, nothing inside the network. Queues are estimated with the same model the optimiser plans with.
- Time-stepped (0.5 s) simulation, not event-driven: car following, queues and spillback emerge from vehicle dynamics. Accidents and storms are scheduled events inside it.
- Forecast targets are true rates; inputs are what sensors could see. The first design mixed these up and is documented as the "original mistake".
- Forecaster selection by validation on whole held-out days (never random rows), with fresh days for a last check.
- Model-based convex control over learned control: the structure is known, the solves are fast and every number can be explained.
- Headline baseline is the tuned fixed green wave, not the untuned plan, and tuning is restricted to nominal days so the disturbed scenarios are a fair test of robustness.
- During and after an incident, fall back to the robust fixed plan rather than optimise on a state the model is known to get wrong.

**Limitations (stated plainly)**
- The forecast adds little to control. The result stands against the brief's expectation that ML should drive the improvement.
- Not met against the tuned plan: the 20% target in the storm (8%), evening (13%) and accident (6%). Tail delay is worse than the tuned plan in the storm and evening.
- The queue estimate has no correction from inside the network, so its accuracy rests on the assumed discharge rate and turning shares. It drifts in the storm and is wrong after an accident; a stop-line counter per road is the obvious first sensor to add.
- Accident handling is damage limitation plus a fallback to the fixed plan; the optimiser does not itself handle a blockage well, and it needs the incident feed (without it, two of ten seeds gridlocked).
- No rerouting or diversions: cars keep their pre-drawn turns.
- Held-out evaluation has 3 seeds per scenario (10 for the typical morning and the accident), so the storm, busy and evening intervals are wide.
- All data comes from our own demand model with assumed rain and accident rates, so forecast accuracy shows recovery of a process we wrote, not real-world accuracy. The history was generated under the fixed 60 s plan.
- One central solver, two-phase intersections, no protected turns, no coordination offsets chosen by the optimiser, no lane model, no reinforcement learning.
- The forecaster does not run on grids other than 2x2 without regenerating history and retraining.
- Not done: robust optimisation over forecast uncertainty, scenarios where the forecast should matter (scheduled event surges), tuning on disturbed days, an event-queue scheduler.

## References
Methods used in this repository and where they come from. The provided starter code was read but not used and is not included; the simulator, controllers and forecaster were written for this project.

**Simulation**
- Intelligent Driver Model (car following, `sim.py`): M. Treiber, A. Hennecke, D. Helbing (2000), "Congested traffic states in empirical observations and microscopic simulations", *Physical Review E* 62(2), 1805-1824. Parameter ranges and the ballistic update: M. Treiber, A. Kesting (2013), *Traffic Flow Dynamics: Data, Models and Simulation*, Springer.
- Dilemma zone (amber running rule, `sim.py`): D. Gazis, R. Herman, A. Maradudin (1960), "The problem of the amber signal light in traffic flow", *Operations Research* 8(1), 112-132.
- Common random numbers (identical demand for every controller): A. M. Law (2015), *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill, ch. 11.

**Signal control and optimisation**
- Webster cycle length C = (1.5L + 5) / (1 - Y): F. V. Webster (1958), *Traffic Signal Settings*, Road Research Technical Paper No. 39, HMSO, London.
- Store-and-forward queue model: D. Gazis, R. Potts (1963), "The oversaturated intersection", *Proc. 2nd International Symposium on the Theory of Traffic Flow*, 221-237; K. Aboudolas, M. Papageorgiou, E. Kosmatopoulos (2009), "Store-and-forward based methods for the signal control problem in large-scale congested urban road networks", *Transportation Research Part C* 17(2), 163-174 (the quadratic-programming form used here).
- Receding-horizon (model predictive) control: D. Mayne, J. Rawlings, C. Rao, P. Scokaert (2000), "Constrained model predictive control: stability and optimality", *Automatica* 36(6), 789-814.
- Fixed-time plans, green waves, minimum green and gap-out: P. Koonce et al. (2008), *Traffic Signal Timing Manual*, FHWA-HOP-08-024, US Federal Highway Administration.

**Forecasting**
- Poisson generalised linear model: J. Nelder, R. Wedderburn (1972), "Generalized linear models", *Journal of the Royal Statistical Society A* 135(3), 370-384.
- Histogram gradient boosting (Poisson loss): G. Ke et al. (2017), "LightGBM: a highly efficient gradient boosting decision tree", *NeurIPS* 30 (scikit-learn's `HistGradientBoostingRegressor` follows this design).
- Validation by whole days in time order, naive and persistence benchmarks: R. Hyndman, G. Athanasopoulos (2021), *Forecasting: Principles and Practice*, 3rd ed., OTexts.

**Software**
- CVXPY: S. Diamond, S. Boyd (2016), "CVXPY: a Python-embedded modeling language for convex optimization", *JMLR* 17(83), 1-5.
- Clarabel solver: P. Goulart, Y. Chen (2024), "Clarabel: an interior-point solver for conic programs with quadratic objectives", arXiv:2405.12762.
- scikit-learn: F. Pedregosa et al. (2011), "Scikit-learn: machine learning in Python", *JMLR* 12, 2825-2830.
- NumPy (C. Harris et al., 2020, *Nature* 585, 357-362), SciPy (P. Virtanen et al., 2020, *Nature Methods* 17, 261-272), pandas, matplotlib.

## AI use
Anthropic's Claude (via Claude Code) was used throughout as a pair-programmer: it wrote or drafted most of the simulator, forecasting pipeline, optimiser, tests, viewer and this document, working from the brief and the author's direction. The author set the problem framing and made the decisions recorded above (forecasting true rates rather than sensor counts, the counts-only sensing assumption, keeping the live-count model with a patterns fallback, the time-stepped simulation choice, the headline baseline and how the limits are presented). Every reported number comes from code in this repository and the held-out results can be regenerated with the commands in Setup. Correctness is supported by the test suite and by running the experiments.
