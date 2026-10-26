"""Metrics: report more than the mean (controllers rank differently across loads).

Delay = elapsed time - distance travelled / free-flow speed, measured from the moment a
vehicle is CREATED (so waiting to enter the network counts). Vehicles still in the system
at the end are included with their delay so far, so oversaturated runs are not flattered
by only counting the cars that got through.
"""
import numpy as np


def summarise(delays, origins, n_entries, throughput_veh, window_s,
              mean_queue, max_queue, spillback_frac) -> dict:
    delays = np.asarray(delays, dtype=float)
    origins = np.asarray(origins, dtype=int)
    if len(delays):
        per_entry = [delays[origins == k].mean() for k in range(n_entries) if np.any(origins == k)]
        mean_d, p95_d, worst = float(delays.mean()), float(np.percentile(delays, 95)), float(max(per_entry))
    else:
        mean_d = p95_d = worst = float("nan")
    return {
        "mean_delay_s": mean_d,
        "p95_delay_s": p95_d,
        "worst_entry_delay_s": worst,          # fairness: worst origin's mean delay
        "throughput_vph": 3600.0 * throughput_veh / window_s,
        "mean_queue": float(mean_queue),       # mean stopped cars per link
        "max_queue": int(max_queue),
        "spillback_frac": float(spillback_frac),   # share of link-time where queue reaches the upstream stop line
        "n_vehicles": int(len(delays)),
    }
