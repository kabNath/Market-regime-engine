"""market-regime-engine: regime detection, risk allocation and live monitoring
for systematic trading systems.

A sanitised, generalised reference implementation of the regime + risk-control
layer of a production systematic book.
"""
from .regime import Regime, RegimeConfig, RegimeDetector, RegimeResult
from .allocation import (
    inverse_vol_weights,
    apply_vol_target,
    cap_exposure,
    select_top_n,
)
from .monitoring import (
    HealthStatus,
    HealthIssue,
    HealthReport,
    MonitorConfig,
    HealthMonitor,
)

__version__ = "0.1.0"

__all__ = [
    "Regime", "RegimeConfig", "RegimeDetector", "RegimeResult",
    "inverse_vol_weights", "apply_vol_target", "cap_exposure", "select_top_n",
    "HealthStatus", "HealthIssue", "HealthReport", "MonitorConfig", "HealthMonitor",
]
