from .base import Controller
from .fixed_time import FixedTimeSynchronised
from .optimiser import ForecastOptimiser

REGISTRY = {
    "fixed_sync": FixedTimeSynchronised,
    "forecast_opt": ForecastOptimiser,
}
