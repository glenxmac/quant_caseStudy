# AGENTS.md: handover for any assistant or reviewer picking this project up

Read `README.md` first (assumptions, forecaster, approach, results, limitations), then `docs/README.md` (the original brief and rubric).

## 1. What this is
"Senior Quant" case study, **Problem 1: smart traffic-light optimisation**, for an internal Quant Analyst application. Real deadline: Thursday morning (the brief says Tuesday 09:00). Format: 15 min live demo + 15 min Q&A.
Rubric (25 each): ML modelling; mathematical optimisation; software engineering (code, demo/visualisation, scalability, tests); presentation. Success criteria in the brief: 20%+ over fixed timing, 4+ intersections, real-time visualisation, response to accidents/closures.

**The author wants it simple.** One story: count -> forecast -> model the queues -> optimise -> measure wait time. Do not add sensors, baselines or mechanisms without being asked.

## 2. Sensing assumption (counts only, since 2026-10-26)
The controller sees: a car counter on each entry road (`obs["entry_cum"]`), the clock, the weather feed, the event calendar (announced events only, `DemandModel.scheduled`, read by the forecaster), its own signal state, and the incident feed (`obs["incidents"]`). It must NOT read `count`, `queue`, `down_queue`, `stop_cross`, `spill`, `entry_count`, `entry_queue` (a test passes it only the allowed keys). Queues are estimated by `QueueEstimator` (same store-and-forward model as the QP). The earlier camera-based version's results (including max-pressure) are summarised in the README; its result files were removed, and a backup of that code was taken outside the repo.

## 3. State: all main pieces are built
| Piece | Where | Status |
|---|---|---|
| Vehicle-level simulator (IDM, 0.5 s steps), scenarios, random days, incident feed, scheduled events (`demand.Event`, `pm_event`) | `src/traffic/{sim,scenarios,demand,daygen,network}.py` | done |
| Baselines: fixed 60 s green wave (`fixed_default`), tuned-cycle green wave (`fixed_tuned`) | `controllers/fixed_time.py` | done (the only baselines; max-pressure and time-of-day were removed) |
| Forecaster (history, features, models, evaluation, online wrapper) | `forecasting/` | done; Poisson GLM + live entrance counts + event calendar + censoring guard; no occupancy features |
| Queue estimator + optimiser (rolling-horizon convex QP, Webster cycle, model-based gap-out, incident fallback) | `controllers/optimiser.py` | done |
| Demo viewer (normal / rush / storm / accident / stadium event; controller's estimate vs truth, forecast vs counts) | `viz/clips.py`, `viz/render.py`, `viz/viewer_template.html` | done |
| Scalability timing | `scripts/scalability.py`, `outputs/scalability.csv` | done (up to 10x10) |
| README writeup | `README.md` | updated for counts only; the AI-use section needs the author's own words |
| Figures | `outputs/*.png` (list in README "Figures") | done; headline chart from `scripts/results_chart.py` |
| Tests | `tests/` (run `pytest`) | 33 pass |

## 4. Headline findings (see README Results)
- Optimiser vs tuned fixed green wave (held-out): -32% typical, -14% evening, -7% accident (n.s.), -5% stadium event (n.s.; wins 9 of 10 seeds, one seed 217 vs 149 s) (10 seeds each), -22% busy, -15% storm (3 seeds). Vs the untuned 60 s plan: 43-64% better everywhere. All numbers are after the cycle-smoothing fix (`smooth_cycle`; the old rounding stalled the cycle at 45-115 s).
- **The learned forecast adds almost nothing to control**: persistence, learned and perfect forecasts give about the same delay, even in the stadium event. The forecast is a good nowcast (7.1% vs profile 10.4%; oracle wins only at +5 min) and predicts event surges from the calendar (13% vs 40% error; GLM beats boosting 13% vs 21% there), but the controller re-plans every minute from counts that already carry today's level, and in a surge its limit is the queue estimate. Say so; do not oversell the ML.
- Stadium event: the forecast sees the surge coming, but the queue estimate collapses when roads fill (assumes full entry roads keep draining; counters go quiet), so on bad days the optimiser starves the eastbound entries (seed 207). The event costs it 46 s per car against 39 for the tuned 180 s plan; it wins the day only because it is ahead before the surge. Cycle cap 180 s was checked on tuning seeds and not adopted (worse on both the normal evening and the event). The event was added after everything else was settled; no optimiser setting was changed for it.
- Gap-out (model-based) is a big part of the gain in light traffic: without it the typical-morning gain (seeds 200-202) is 12% not 30%.
- Incident fallback: from report until 60 min after clearance the signals run the tuned 120 s green wave. Without the incident feed the counts-only controller gridlocks 2 of 10 accident seeds (457 s vs 225 s mean). Chosen on accident tuning seeds 100-109.
- Tail (p95) delay is better than tuned fixed in the typical, busy, storm and accident runs, level in the evening and event.
- Queue estimate: good on normal days (entry ~0.4 cars, internal ~1.2 cars mean abs error), drifts high in the storm, wrong after an accident and in an event surge. No sensor corrects it.
- No more model or solver variants are needed (README "Why not more models or solvers"): GLM level with boosting (better on events), a perfect forecast does not lower delay, the QP is globally optimal and the sweep is flat. Remaining error is the unmeasured queue state.

## 5. Conventions (keep these)
- Common random numbers: same scenario + seed = identical demand for every controller. Seeds: tuning 100-109, held-out evaluation 200+, demo 300, forecaster history 0-59 plus fresh 5000+/6000+/1000+.
- Never tune on an evaluation seed. Baselines and optimiser hyper-parameters are tuned on nominal days only; accident mechanisms are judged on 10 tuning seeds (3 proved misleading before).
- Forecast labels = true rates; model inputs = only what the entrance counters, clock, weather feed and event calendar (announced size, never the real crowd) could show at time t. `build_features` must use the past only (tested). Oracle columns (`true_now`, `obs_target`, `rain_target`) and occupancy (`entry_count`, `entry_queue`, `cnt_lag*`) are never features (tested).
- Estimator parameters (discharge rate, start-up delay) are calibrated against simulator truth on tuning seeds, not tuned for delay.
- Long jobs are chunked and resumable (the shell here caps a command at about 3 minutes; use background jobs).
- Python: if the default interpreter fails to import scipy, use a venv with `requirements.txt` (scikit-learn 1.7.2 is pinned for the pickled forecaster).

## 6. Layout
```
main.py                CLI: history, forecast, forecast-plots, tune-scen, viz-clips, days
src/traffic/           config, network, demand, daygen, scenarios, sim, metrics
  controllers/         base, fixed_time (green wave), optimiser (QueueEstimator + QP) (+ REGISTRY in __init__)
  forecasting/         history, features, model, evaluate, online
  experiments/         scen (scenario runs, fixed-cycle tuning)
  viz/                 clips + render + viewer_template.html (demo), forecastplots, dayplots
scripts/               eval_chunk (held-out), eval_variants (ablations), estimator_drift, results_chart, sweep, scalability
tests/                 simulator, history, forecasting, optimiser, smoke
outputs/               generated; fitted forecaster in outputs/.cache/final_5.0.pkl; held-out results in final_eval_counts/
```

## 7. Commands
```
pip install -r requirements.txt && pip install -e . && pytest
python main.py history --n-days 60 && python main.py forecast --final   # forecaster data, models, ladder table
for sc in am_ramp am_busy am_rain pm_ramp; do PYTHONPATH=src python scripts/eval_chunk.py $sc 200 3; done
for sc in am_ramp pm_ramp; do PYTHONPATH=src python scripts/eval_chunk.py $sc 203 7; done
for sc in am_accident pm_event; do for s in 200 205; do PYTHONPATH=src python scripts/eval_chunk.py $sc $s 5; done; done
rm outputs/viewer_cache/*_v4.json; python main.py viz-clips             # outputs/demo_viewer.html (~36 MB)
PYTHONPATH=src python scripts/results_chart.py; python main.py forecast-plots   # figures
PYTHONPATH=src python scripts/scalability.py 10 10
```

## 8. What is not done (candidate future work, all listed in the README limitations)
A stop-line counter per road to correct the queue estimate; an estimator that flags full entry roads from the calendar and a silent counter (or a calendar-triggered fallback) so the event forecast can pay off; robust optimisation over forecast and estimate uncertainty; more evaluation seeds (3 for busy/storm); coordination offsets chosen by the optimiser; RL. The forecaster is fitted to the 2x2 grid only.
