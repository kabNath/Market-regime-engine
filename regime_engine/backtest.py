"""Walk-forward evaluation with transaction costs, slippage and stress tests.

Most public trading repos evaluate a strategy on the same data used to design
it, with zero frictions. This module does the opposite, on purpose:

* **Walk-forward protocol** — the regime detector only ever sees data up to
  the decision date; allocations are applied to the *next* bar. No look-ahead
  by construction.
* **Frictions** — proportional transaction costs and slippage are charged on
  every unit of turnover, and turnover itself is reported, because a strategy
  that "works" at 0 bps but dies at 10 bps is not a strategy.
* **Stress tests** — synthetic crash scenarios are injected at arbitrary
  points to verify the crisis gates and drawdown kill-switch actually fire
  when it matters.

The strategy evaluated here is deliberately simple (regime-gated exposure):
the point of this module is the *evaluation methodology*, not the alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .regime import Regime, RegimeConfig, RegimeDetector


# Exposure applied for each regime by the demo strategy (long-only book).
DEFAULT_EXPOSURE: Dict[str, float] = {
    Regime.BULL.value: 1.0,
    Regime.NEUTRAL.value: 0.6,
    Regime.BEAR.value: 0.25,
    Regime.CRISIS.value: 0.0,
}


@dataclass(frozen=True)
class CostModel:
    """Proportional frictions charged per unit of turnover.

    ``cost_bps`` models commissions/fees; ``slippage_bps`` models execution
    shortfall. Both are charged on |Δexposure| at every rebalance.
    """

    cost_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def total_rate(self) -> float:
        return (self.cost_bps + self.slippage_bps) / 1e4


@dataclass
class WalkForwardResult:
    equity: pd.Series
    exposure: pd.Series
    regimes: pd.Series
    turnover: float                  # average annualised one-way turnover
    total_costs: float               # cumulative friction drag (fraction of capital)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"CAGR {m['cagr']:+.2%} | vol {m['vol']:.2%} | Sharpe {m['sharpe']:.2f} | "
            f"maxDD {m['max_drawdown']:.2%} | turnover {self.turnover:.2f}x/yr | "
            f"cost drag {m['cost_drag_annual']:.2%}/yr"
        )


def _perf_metrics(equity: pd.Series, ann_factor: int = 252) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    n_years = len(rets) / ann_factor
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1
    vol = float(rets.std() * np.sqrt(ann_factor))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ann_factor)) if rets.std() > 0 else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {"cagr": float(cagr), "vol": vol, "sharpe": sharpe, "max_drawdown": dd}


def walk_forward(
    close: pd.Series,
    detector: Optional[RegimeDetector] = None,
    exposure_map: Optional[Dict[str, float]] = None,
    costs: Optional[CostModel] = None,
    rebalance_every: int = 5,
    warmup: Optional[int] = None,
    ann_factor: int = 252,
) -> WalkForwardResult:
    """Run a strictly out-of-sample, cost-aware walk-forward simulation.

    At each rebalance date ``t`` the detector classifies using prices up to
    and **including** ``t``; the resulting exposure is applied from ``t+1``
    onward. Frictions are charged on every change in exposure.
    """
    detector = detector or RegimeDetector()
    exposure_map = exposure_map or DEFAULT_EXPOSURE
    costs = costs or CostModel()
    warmup = warmup or detector.config.mom_long_window + 5

    if len(close) <= warmup + rebalance_every:
        raise ValueError(f"need more than {warmup + rebalance_every} observations")

    rets = close.pct_change().fillna(0.0)

    exposure = pd.Series(np.nan, index=close.index, dtype=float)
    regimes = pd.Series(index=close.index, dtype=object)

    current_expo = 0.0
    for i in range(warmup, len(close)):
        if (i - warmup) % rebalance_every == 0:
            visible = close.iloc[: i + 1]           # information set: up to t only
            result = detector.classify_latest(visible)
            target = exposure_map[result.regime.value]
            current_expo = target
            regimes.iloc[i] = result.regime.value
        # exposure decided at t earns the return of t+1 (applied via shift below)
        exposure.iloc[i] = current_expo

    exposure = exposure.ffill().fillna(0.0)
    regimes = regimes.ffill()

    # Strategy return: yesterday's exposure times today's asset return.
    strat_rets = exposure.shift(1).fillna(0.0) * rets

    # Frictions on each unit of |Δexposure|.
    d_expo = exposure.diff().abs().fillna(0.0)
    friction = d_expo * costs.total_rate
    strat_rets_net = strat_rets - friction

    equity = (1.0 + strat_rets_net).cumprod()
    eval_slice = equity.iloc[warmup:]

    n_years = max(len(eval_slice) / ann_factor, 1e-9)
    turnover = float(d_expo.iloc[warmup:].sum() / n_years)
    total_costs = float(friction.iloc[warmup:].sum())

    metrics = _perf_metrics(eval_slice, ann_factor)
    metrics["cost_drag_annual"] = total_costs / n_years
    return WalkForwardResult(
        equity=eval_slice,
        exposure=exposure.iloc[warmup:],
        regimes=regimes.iloc[warmup:],
        turnover=turnover,
        total_costs=total_costs,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Stress testing
# --------------------------------------------------------------------------- #
def inject_crash(
    close: pd.Series,
    start_frac: float = 0.7,
    crash_days: int = 25,
    total_drop: float = -0.35,
    crash_vol: float = 0.60,
    seed: int = 42,
) -> pd.Series:
    """Return a copy of ``close`` with a synthetic high-volatility crash
    injected at ``start_frac`` of the way through the series. Subsequent
    prices are rescaled so the series stays continuous.
    """
    rng = np.random.default_rng(seed)
    out = close.copy().astype(float)
    start = int(len(out) * start_frac)
    end = min(start + crash_days, len(out))
    n = end - start

    mu = np.log(1.0 + total_drop) / n
    sigma = crash_vol / np.sqrt(252.0)
    shock = np.exp(np.cumsum(rng.normal(mu, sigma, n)))

    pre = out.iloc[start - 1]
    crashed = pre * shock
    scale_after = crashed[-1] / out.iloc[end - 1]
    out.iloc[start:end] = crashed
    out.iloc[end:] = out.iloc[end:] * scale_after
    return out


@dataclass
class StressOutcome:
    name: str
    strategy_max_dd: float
    asset_max_dd: float
    days_in_crisis: int
    protected: bool                  # strategy DD meaningfully better than asset DD

    def __str__(self) -> str:
        flag = "PASS" if self.protected else "FAIL"
        return (
            f"[{flag}] {self.name}: strategy maxDD {self.strategy_max_dd:.2%} "
            f"vs asset {self.asset_max_dd:.2%} | {self.days_in_crisis} bars in CRISIS"
        )


def stress_test(
    close: pd.Series,
    scenarios: Optional[List[dict]] = None,
    protection_ratio: float = 0.65,
    **walk_kwargs,
) -> List[StressOutcome]:
    """Run the walk-forward strategy through injected crash scenarios and
    check that the crisis machinery actually reduces realised drawdown.

    A scenario "passes" when the strategy's max drawdown is at most
    ``protection_ratio`` of the underlying asset's max drawdown.
    """
    scenarios = scenarios or [
        {"name": "flash_crash", "crash_days": 10, "total_drop": -0.25, "crash_vol": 0.80},
        {"name": "bear_grind", "crash_days": 90, "total_drop": -0.35, "crash_vol": 0.30},
        {"name": "vol_explosion", "crash_days": 30, "total_drop": -0.40, "crash_vol": 0.70},
    ]

    outcomes: List[StressOutcome] = []
    for sc in scenarios:
        name = sc.pop("name")
        stressed = inject_crash(close, **sc)
        sc["name"] = name

        res = walk_forward(stressed, **walk_kwargs)
        asset_eq = stressed.iloc[-len(res.equity):]
        asset_dd = float((asset_eq / asset_eq.cummax() - 1).min())
        strat_dd = res.metrics["max_drawdown"]
        days_crisis = int((res.regimes == "CRISIS").sum())

        outcomes.append(
            StressOutcome(
                name=name,
                strategy_max_dd=strat_dd,
                asset_max_dd=asset_dd,
                days_in_crisis=days_crisis,
                protected=strat_dd >= asset_dd * protection_ratio,  # DDs are negative
            )
        )
    return outcomes
