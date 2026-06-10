"""Tests for the regime detector."""
import numpy as np
import pandas as pd

from regime_engine import RegimeDetector, Regime
from regime_engine.data import synthetic_regime_series


def _const_growth(days: int, annual_drift: float, annual_vol: float, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    mu = annual_drift / 252.0
    sigma = annual_vol / np.sqrt(252.0)
    rets = rng.normal(mu, sigma, days)
    idx = pd.bdate_range("2018-01-01", periods=days)
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


def test_strong_uptrend_is_bull():
    price = _const_growth(400, annual_drift=0.20, annual_vol=0.10)
    result = RegimeDetector().classify_latest(price)
    assert result.regime == Regime.BULL
    assert result.score >= 5


def test_high_vol_crash_is_crisis():
    # Calm uptrend, then a violent high-vol selloff.
    calm = _const_growth(300, 0.15, 0.10, seed=2)
    rng = np.random.default_rng(3)
    crash_rets = rng.normal(-0.6 / 252, 0.6 / np.sqrt(252), 30)
    crash = calm.iloc[-1] * np.exp(np.cumsum(crash_rets))
    idx = pd.bdate_range(calm.index[-1] + pd.Timedelta(days=1), periods=30)
    price = pd.concat([calm, pd.Series(crash, index=idx, name="close")])
    result = RegimeDetector().classify_latest(price)
    assert result.regime == Regime.CRISIS


def test_fit_returns_all_regimes_on_synthetic():
    price, _ = synthetic_regime_series()
    frame = RegimeDetector().fit(price)
    seen = set(frame["regime"].dropna().unique())
    # The synthetic series is built to contain every regime.
    assert {"BULL", "BEAR", "CRISIS"}.issubset(seen)


def test_insufficient_history_raises():
    price = _const_growth(50, 0.1, 0.1)
    try:
        RegimeDetector().classify_latest(price)
        assert False, "expected ValueError"
    except ValueError:
        pass
