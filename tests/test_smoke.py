from traffic.controllers import REGISTRY
from traffic.config import SimConfig


def test_registry_has_the_baseline_and_the_optimiser():
    assert set(REGISTRY) == {"fixed_sync", "forecast_opt"}


def test_config_defaults_sane():
    c = SimConfig()
    assert c.min_green_s < c.max_green_s
    assert c.grid_rows * c.grid_cols == 4   # brief requires >= 4 intersections
