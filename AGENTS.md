# AGENTS.md: handover for any assistant or reviewer picking this project up

Read `README.md` first (assumptions, forecaster, approach, results, limitations), then `docs/README.md` (the original brief and rubric).

## 1. What this is
"Senior Quant" case study, **Problem 1: smart traffic-light optimisation**, for an internal Quant Analyst application. Real deadline: Thursday morning (the brief says Tuesday 09:00). Format: 15 min live demo + 15 min Q&A.
Rubric (25 each): ML modelling; mathematical optimisation; software engineering (code, demo/visualisation, scalability, tests); presentation. Success criteria in the brief: 20%+ over fixed timing, 4+ intersections, real-time visualisation, response to accidents/closures.

**The author wants it simple.** One story: count -> forecast -> model the queues -> optimise -> measure wait time. Do not add sensors, baselines or mechanisms without being asked.

## 2. Sensing assumption (counts only, since 2026-10-26)
The controller sees: a car counter on each entry road (`obs["entry_cum"]`), the clock, the weather feed, its own signal state, and the incident feed (`obs["incidents"]`). It must NOT read `count`, `queue`, `down_queue`, `stop_cross`, `spill`, `entry_count`, `entry_queue` (a test passes it only the allowed keys). Queues are estimated by `QueueEstimator` (same store-and-forward model as the QP). The earlier camera-based version's results are in `outputs/final_eval/`; a backup of that code was taken outside the repo.

## 3. State: all main pieces are built
| Piece | Where | Status |
|---|---|---|
| Vehicle-level simulator (IDM, 0.5 s steps), scenarios, random days, incident feed | `src/traffic/{sim,scenarios,demand,daygen,network}.py` | done |
| Baselines: fixed 60 s green wave (`fixed_default`), tuned-cycle green wave (`fixed_tuned`) | `controllers/fixed_time.py` | done (the only baselines; max-pressure and time-of-day were removed) |
| Forecaster (history, features, models, evaluation, online wrapper) | `forecasting/` | done; Poisson GLM + live entrance counts + censoring guard; no occupancy features |
| Queue estimator + optimiser (rolling-horizon convex QP, Webster cycle, model-based gap-out, incident fallback) | `controllers/optimiser.py` | done |
| Demo viewer (normal / rush / storm / accident; controller's estimate vs truth, forecast vs counts) | `viz/clips.py`, `viz/render.py`, `viz/viewer_template.html` | done |
| Scalability timing | `scripts/scalability.py`, `outputs/scalability.csv` | done (up to 10x10) |
| README writeup | `README.md` | updated for counts only; the AI-use section needs the author's own words |
| Figures | `outputs/*.png` (list in README "Figures") | done; headline chart from `scripts/results_chart.py` |
| Tests | `tests/` (run `pytest`) | 30 pass |

## 4. Headline findings (see README Results)
- Optimiser vs tuned fixed green wave (held-out): -33% typical (10 seeds), -25% busy, -8% storm, -13% evening (3 seeds), -6% accident (10 seeds). Vs the untuned 60 s plan: 38-66% better everywhere.
- **The learned forecast adds almost nothing to control**: persistence, learned and perfect forecasts give about the same delay. The forecast is a good nowcast (6.8% vs profile 9.8%; oracle wins only at +5 min), but the controller re-plans every minute from counts that already carry today's level. Say so; do not oversell the ML.
- Gap-out (model-based) is a big part of the gain in light traffic: without it the typical-morning gain is 11% not 31%.
- Incident fallback: from report until 60 min after clearance the signals run the tuned 120 s green wave. Without the incident feed the counts-only controller gridlocks 2 of 10 accident seeds. Chosen on accident tuning seeds 100-109.
- Tail (p95) delay is worse than tuned fixed in the storm and evening.
- Queue estimate: good on normal days (entry ~0.4 cars, internal ~1.2 cars mean abs error), drifts high in the storm, wrong after an accident. No sensor corrects it.
- No more model or solver variants are needed (README "Why not more models or solvers"): GLM = boosting, a perfect forecast does not lower delay, the QP is globally optimal and the sweep is flat. Remaining error is the unmeasured queue state.

## 5. Conventions (keep these)
- Common random numbers: same scenario + seed = identical demand for every controller. Seeds: tuning 100-109, held-out evaluation 200+, demo 300, forecaster history 0-59 plus fresh 5000+/6000+/1000+.
- Never tune on an evaluation seed. Baselines and optimiser hyper-parameters are tuned on nominal days only; accident mechanisms are judged on 10 tuning seeds (3 proved misleading before).
- Forecast labels = true rates; model inputs = only what the entrance counters, clock and weather feed could show at time t. `build_features` must use the past only (tested). Oracle columns (`true_now`, `obs_target`, `rain_target`) and occupancy (`entry_count`, `entry_queue`, `cnt_lag*`) are never features (tested).
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
                       (final_eval/ = earlier camera-based version, kept for comparison only)
```

## 7. Commands
```
pip install -r requirements.txt && pip install -e . && pytest
python main.py history --n-days 60 && python main.py forecast --final   # forecaster data, models, ladder table
for sc in am_ramp am_busy am_rain pm_ramp; do PYTHONPATH=src python scripts/eval_chunk.py $sc 200 3; done
PYTHONPATH=src python scripts/eval_chunk.py am_ramp 203 7
PYTHONPATH=src python scripts/eval_chunk.py am_accident 200 5; PYTHONPATH=src python scripts/eval_chunk.py am_accident 205 5
rm outputs/viewer_cache/*_v3.json; python main.py viz-clips             # outputs/demo_viewer.html (~28 MB)
PYTHONPATH=src python scripts/results_chart.py; python main.py forecast-plots   # figures
PYTHONPATH=src python scripts/scalability.py 10 10
```

## 8. What is not done (candidate future work, all listed in the README limitations)
A stop-line counter per road to correct the queue estimate; a scenario where the forecast should matter (scheduled event surge); robust optimisation over forecast uncertainty; more evaluation seeds (3 per scenario for busy/storm/evening); coordination offsets chosen by the optimiser; RL. The forecaster is fitted to the 2x2 grid only.
