"""Market regime detection.

A six-indicator regime classifier that maps a price history onto one of four
states: ``BULL``, ``NEUTRAL``, ``BEAR`` or ``CRISIS``.

The design mirrors the regime layer of a production systematic-trading system:
a small, transparent set of orthogonal indicators is combined into a single
score, with hard "crisis" gates that override the score when tail risk
materialises. The goal is robustness and explainability, not curve-fitting —
every state the engine emits can be traced back to the indicators that caused
it.

This is a generalised, sanitised reference implementation. Thresholds are
sensible defaults, not a proprietary configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd


class Regime(str, Enum):
    """The four market states the detector can emit."""

    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    CRISIS = "CRISIS"


@dataclass(frozen=True)
class RegimeConfig:
    """Tunable parameters for :class:`RegimeDetector`.

    All windows are expressed in trading days. Defaults assume daily data.
    """

    trend_window: int = 200          # long-term trend filter (price vs SMA)
    mom_mid_window: int = 63         # ~3-month momentum
    mom_long_window: int = 252       # ~12-month momentum
    vol_window: int = 20             # short realised-volatility window
    vol_long_window: int = 63        # baseline volatility window
    drawdown_window: int = 252       # rolling peak lookback for drawdown
    ann_factor: int = 252            # annualisation factor for volatility

    crisis_vol: float = 0.35         # annualised vol above this => CRISIS gate
    crisis_drawdown: float = -0.20   # drawdown beyond this => CRISIS gate
    bull_score: int = 5              # score >= this => BULL
    bear_score: int = 2              # score <= this => BEAR


@dataclass
class RegimeResult:
    """The classification for a single point in time."""

    regime: Regime
    score: int                       # 0..6 bullish-evidence score
    indicators: Dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        """Human-readable breakdown of why this regime was chosen."""
        parts = [f"{k}={v:+.4f}" for k, v in self.indicators.items()]
        return f"{self.regime.value} (score {self.score}/6) | " + ", ".join(parts)


class RegimeDetector:
    """Classify market regime from a series of closing prices.

    Example
    -------
    >>> detector = RegimeDetector()
    >>> frame = detector.fit(close_prices)        # full history, vectorised
    >>> latest = detector.classify_latest(close_prices)
    >>> print(latest.explain())
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ----- indicator computation -------------------------------------------------
    def _indicators(self, close: pd.Series) -> pd.DataFrame:
        c = self.config
        close = close.astype(float)
        log_ret = np.log(close).diff()

        sma = close.rolling(c.trend_window, min_periods=c.trend_window // 2).mean()
        trend = close / sma - 1.0

        mom_mid = close.pct_change(c.mom_mid_window)
        mom_long = close.pct_change(c.mom_long_window)

        vol_short = log_ret.rolling(c.vol_window, min_periods=c.vol_window // 2).std() * np.sqrt(c.ann_factor)
        vol_long = log_ret.rolling(c.vol_long_window, min_periods=c.vol_long_window // 2).std() * np.sqrt(c.ann_factor)
        vol_accel = vol_short / vol_long

        rolling_peak = close.rolling(c.drawdown_window, min_periods=1).max()
        drawdown = close / rolling_peak - 1.0

        return pd.DataFrame(
            {
                "trend": trend,
                "mom_mid": mom_mid,
                "mom_long": mom_long,
                "vol": vol_short,
                "vol_accel": vol_accel,
                "drawdown": drawdown,
            }
        )

    # ----- scoring & classification ----------------------------------------------
    def _score_row(self, row: pd.Series) -> int:
        """Count bullish evidence across the six indicators (0..6)."""
        score = 0
        score += int(row["trend"] > 0)                 # above long-term trend
        score += int(row["mom_mid"] > 0)               # positive medium momentum
        score += int(row["mom_long"] > 0)              # positive long momentum
        score += int(row["vol"] < self.config.crisis_vol * 0.6)  # contained vol
        score += int(row["vol_accel"] < 1.0)           # vol not accelerating
        score += int(row["drawdown"] > -0.05)          # shallow drawdown
        return score

    def _classify_row(self, row: pd.Series, score: int) -> Regime:
        c = self.config
        # Hard crisis gates always take precedence over the score.
        if (row["vol"] >= c.crisis_vol) or (row["drawdown"] <= c.crisis_drawdown):
            return Regime.CRISIS
        if score >= c.bull_score:
            return Regime.BULL
        if score <= c.bear_score:
            return Regime.BEAR
        return Regime.NEUTRAL

    # ----- public API ------------------------------------------------------------
    def fit(self, close: pd.Series) -> pd.DataFrame:
        """Return a frame of indicators, score and regime for the whole series."""
        ind = self._indicators(close)
        score = ind.apply(self._score_row, axis=1).astype("Int64")
        regime = [
            self._classify_row(row, int(s)) if not row.isna().any() and pd.notna(s) else None
            for (_, row), s in zip(ind.iterrows(), score)
        ]
        out = ind.copy()
        out["score"] = score
        out["regime"] = regime
        return out

    def classify_latest(self, close: pd.Series) -> RegimeResult:
        """Classify the most recent observation."""
        ind = self._indicators(close)
        row = ind.iloc[-1]
        if row.isna().any():
            raise ValueError(
                "Not enough history to classify the latest point; "
                f"need at least {self.config.mom_long_window} observations."
            )
        score = self._score_row(row)
        regime = self._classify_row(row, score)
        return RegimeResult(regime=regime, score=score, indicators=row.round(6).to_dict())
