"""All tunable parameters in one place so experiments are reproducible.

Every value here is an ASSUMPTION; see docs/DETAILS.md 'Assumptions'. Nothing is hidden in the sim code.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleConfig:
    """One standard car (no vehicle mix). Intelligent Driver Model (IDM) parameters.

    IDM gives the 'elastic' stop-and-go behaviour: cars accelerate out of a queue
    gradually, brake smoothly behind a leader, and discharge at a finite headway.
    Saturation flow is therefore an EMERGENT output (~1/(T + (s0+L)/v)), which we
    calibrate and then feed to the optimiser's fluid model.
    """
    length_m: float = 4.5
    min_gap_m: float = 2.0          # s0, bumper-to-bumper gap when stopped
    v_free_mps: float = 13.9        # v0, ~50 km/h urban speed limit
    time_headway_s: float = 1.5     # T, desired time gap
    accel_mps2: float = 1.5         # a, max comfortable acceleration
    decel_mps2: float = 2.0         # b, comfortable braking
    # Rain (intensity r in [0,1]) changes driver behaviour, which cuts road capacity:
    #   free speed x (1 - rain_speed_drop*r), time headway x (1 + rain_headway_gain*r),
    #   acceleration x (1 - rain_accel_drop*r).
    rain_speed_drop: float = 0.20
    rain_headway_gain: float = 0.30
    rain_accel_drop: float = 0.15

    @property
    def slot_m(self) -> float:
        """Road length one queued car occupies (car + standstill gap)."""
        return self.length_m + self.min_gap_m


@dataclass(frozen=True)
class SimConfig:
    """Simulator settings: grid, link lengths, turning shares, signal timing, saturation flow."""
    dt: float = 0.5                 # seconds per sim step (IDM is stable at 0.5 s)
    horizon_s: int = 3600           # sim length (s)
    warmup_s: int = 300             # discarded from metrics
    # --- geometry ---
    grid_rows: int = 2              # network = rows x cols; code must work for any size
    grid_cols: int = 2
    link_length_m: float = 200.0    # stop line to stop line between neighbouring intersections
    # Entry roads are longer so a backed-up grid does not immediately silence the entry
    # sensors (which would make the observed arrival rate collapse exactly when demand is
    # highest). Exit roads are not modelled: nothing blocks them, so cars are counted and
    # removed as they cross the last stop line.
    entry_link_length_m: float = 400.0
    lanes_per_approach: int = 1     # single lane; turning vehicles share it
    # --- turning shares at every intersection (straight, left, right) ---
    turn_probs: tuple = (0.70, 0.15, 0.15)
    # --- signals ---
    lost_time_s: float = 4.0        # yellow + all-red on each phase switch
    min_green_s: float = 8.0
    max_green_s: float = 60.0
    # --- fluid model (optimiser) ---
    # Saturation flow per approach while green. NOT an input to the vehicle sim: it is
    # MEASURED by releasing a standing queue in the IDM sim (tests/test_sim.py,
    # measure_saturation_flow -> 0.40 veh/s ~ 1440 veh/h/lane) and used by the optimiser.
    saturation_flow: float = 0.40   # veh/s

    @property
    def n_intersections(self) -> int:
        return self.grid_rows * self.grid_cols

    def storage_per_link(self, vehicle: VehicleConfig) -> int:
        """Max queued cars between two intersections (spillback threshold)."""
        return int(self.link_length_m // vehicle.slot_m)

    def free_flow_link_s(self, vehicle: VehicleConfig) -> float:
        return self.link_length_m / vehicle.v_free_mps

