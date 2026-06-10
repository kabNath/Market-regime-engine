"""Tests for walk-forward evaluation, frictions, and stress testing."""
import numpy as np
import pandas as pd

from regime_engine.backtest import (
    CostModel,
    inject_crash,
    stress_test,
    walk_forward,
)
from regime_engine.data import synthetic_regime_series


def _price():
    price, _ = synthetic_regime_series(seed=7)
    return price


def test_walk_forward_runs_and_reports():
    res = walk_forward(_price())
    assert len(res.equity) > 200
    assert set(res.metrics) >= {"cagr", "vol", "sharpe", "max_drawdown", "cost_drag_annual"}
    assert res.turnover >= 0


def test_no_lookahead_exposure_lags_information():
    """The return earned at t must come from exposure decided strictly before t."""
    res = walk_forward(_price(), rebalance_every=1)
    # On the first evaluated bar the prior exposure is the warmup value (0 → first
    # decision applies next bar). Reconstruct one step manually:
    price = _price()
    rets = price.pct_change().fillna(0.0)
    idx = res.exposure.index
    # equity return at idx[1] should equal exposure at idx[0] times asset return at idx[1], minus frictions
    gross = res.exposure.loc[idx[0]] * rets.loc[idx[1]]
    realised = res.equity.loc[idx[1]] / res.equity.loc[idx[0]] - 1
    assert realised <= gross + 1e-12  # frictions can only reduce it


def test_costs_strictly_reduce_performance():
    price = _price()
    free = walk_forward(price, costs=CostModel(cost_bps=0, slippage_bps=0))
    costly = walk_forward(price, costs=CostModel(cost_bps=25, slippage_bps=25))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.metrics["cost_drag_annual"] > 0


def test_inject_crash_is_continuous_and_drops():
    price = _price()
    stressed = inject_crash(price, start_frac=0.5, crash_days=20, total_drop=-0.30)
    assert len(stressed) == len(price)
    start = int(len(price) * 0.5)
    window = stressed.iloc[start - 1 : start + 25]
    assert window.min() < stressed.iloc[start - 1] * 0.80  # a real drop happened
    # continuity: no absurd single-bar gap at the splice points
    jumps = stressed.pct_change().abs().max()
    assert jumps < 0.5


def test_stress_scenarios_engage_crisis_state():
    outcomes = stress_test(_price())
    assert len(outcomes) == 3
    # The crisis machinery must engage in at least the violent scenarios.
    assert sum(o.days_in_crisis > 0 for o in outcomes) >= 2
