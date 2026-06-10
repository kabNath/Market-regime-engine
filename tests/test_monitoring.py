"""Tests for the health monitor."""
import numpy as np
import pandas as pd

from regime_engine import HealthMonitor, HealthStatus
from regime_engine.data import synthetic_regime_series


def _clean_series(days: int = 300) -> pd.Series:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=days)
    rets = rng.normal(0.0003, 0.008, days)
    price = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")
    return price


def test_clean_series_is_ok():
    # Use a recent, gap-free series so the stale-feed check stays quiet by
    # construction we drop the datetime index to isolate integrity logic.
    price = _clean_series().reset_index(drop=True)
    report = HealthMonitor().run(close=price)
    assert report.status == HealthStatus.OK
    assert report.tradeable


def test_nonpositive_price_is_critical():
    price = _clean_series().reset_index(drop=True)
    price.iloc[-1] = -5.0
    report = HealthMonitor().run(close=price)
    assert report.status == HealthStatus.CRITICAL
    assert not report.tradeable


def test_drawdown_breach_is_critical():
    price = _clean_series().reset_index(drop=True)
    equity = price.copy()
    equity.iloc[-1] = equity.max() * 0.85  # -15% from peak, beyond -8% limit
    report = HealthMonitor().run(close=price, equity=equity)
    assert report.status == HealthStatus.CRITICAL
    assert report.metrics["trailing_drawdown"] <= -0.08


def test_crisis_regime_warns():
    price = _clean_series().reset_index(drop=True)
    regimes = ["BULL"] * 250 + ["CRISIS"]
    report = HealthMonitor().run(close=price, regimes=regimes)
    assert report.status >= HealthStatus.WARN
    assert any(i.code == "MODEL_CRISIS" for i in report.issues)


def test_report_serialises():
    price, _ = synthetic_regime_series()
    report = HealthMonitor().run(close=price.reset_index(drop=True))
    d = report.to_dict()
    assert "status" in d and "issues" in d and "tradeable" in d
