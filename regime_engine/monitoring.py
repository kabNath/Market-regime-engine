"""Live health monitoring and anomaly detection.

This is the part of a systematic system that runs *every day in production* and
decides whether the book is safe to keep trading. It answers three questions:

1. **Is the data trustworthy?** (gaps, stale feeds, non-positive prices, spikes)
2. **Is the book inside its risk mandate?** (trailing drawdown limit)
3. **Is the model behaving sanely?** (regime stability, crisis state)

Each check emits a structured :class:`HealthIssue`; the worst severity across
all checks becomes the overall :class:`HealthStatus`. The output is designed to
be machine-readable (for alerting / kill-switches) *and* human-readable (for a
morning monitoring dashboard).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


class HealthStatus(IntEnum):
    """Ordered severity levels (higher = worse)."""

    OK = 0
    WARN = 1
    CRITICAL = 2

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass
class HealthIssue:
    level: HealthStatus
    code: str
    message: str


@dataclass
class HealthReport:
    status: HealthStatus
    issues: List[HealthIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def tradeable(self) -> bool:
        """A CRITICAL report should halt trading (kill-switch)."""
        return self.status < HealthStatus.CRITICAL

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": str(self.status),
            "tradeable": self.tradeable,
            "issues": [
                {"level": str(i.level), "code": i.code, "message": i.message}
                for i in self.issues
            ],
            "metrics": self.metrics,
        }

    def __str__(self) -> str:
        lines = [f"[{self.timestamp:%Y-%m-%d %H:%M}] HEALTH: {self.status}  (tradeable={self.tradeable})"]
        for i in self.issues:
            lines.append(f"  - {str(i.level):8s} {i.code}: {i.message}")
        if not self.issues:
            lines.append("  - all checks passed")
        return "\n".join(lines)


@dataclass
class MonitorConfig:
    trailing_dd_limit: float = -0.08      # halt/alert if trailing DD breaches this
    stale_data_days: int = 4              # last bar older than this => stale feed
    jump_sigma: float = 8.0               # single-bar move beyond N sigma => spike
    jump_window: int = 63                 # window for the sigma estimate
    regime_flip_limit: int = 4            # > this many flips in window => unstable
    regime_flip_window: int = 21


class HealthMonitor:
    """Run the full battery of production checks and return a HealthReport."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()

    # ----- individual checks -----------------------------------------------------
    def check_data_integrity(self, close: pd.Series) -> List[HealthIssue]:
        issues: List[HealthIssue] = []
        c = self.config

        if close.isna().any():
            n = int(close.isna().sum())
            issues.append(HealthIssue(HealthStatus.WARN, "DATA_NAN", f"{n} missing price(s) in series"))

        if (close.dropna() <= 0).any():
            issues.append(HealthIssue(HealthStatus.CRITICAL, "DATA_NONPOSITIVE", "non-positive price detected"))

        if close.index.has_duplicates:
            issues.append(HealthIssue(HealthStatus.WARN, "DATA_DUP_INDEX", "duplicate timestamps in index"))

        # Stale-feed check (only when the index is datetime-like).
        if isinstance(close.index, pd.DatetimeIndex) and len(close) > 0:
            last = close.index[-1].to_pydatetime()
            age = datetime.utcnow() - last
            if age > timedelta(days=c.stale_data_days):
                issues.append(
                    HealthIssue(HealthStatus.CRITICAL, "DATA_STALE",
                                f"last bar is {age.days}d old (limit {c.stale_data_days}d)")
                )

        # Spike check: any single-bar return beyond N sigma. Guard against a
        # zero/NaN volatility estimate (e.g. a flat warm-up window) which would
        # otherwise divide to +/-inf and raise a false alarm.
        ret = close.pct_change()
        sigma = ret.rolling(c.jump_window, min_periods=c.jump_window // 2).std()
        z = (ret / sigma.replace(0.0, np.nan)).abs()
        last_z = z.iloc[-1] if len(z) else np.nan
        if pd.notna(last_z) and last_z > c.jump_sigma:
            issues.append(
                HealthIssue(HealthStatus.WARN, "DATA_SPIKE",
                            f"latest move is {last_z:.1f} sigma")
            )
        return issues

    def check_drawdown(self, equity: pd.Series) -> List[HealthIssue]:
        """Trailing drawdown against the mandate limit."""
        if len(equity) == 0:
            return []
        peak = equity.cummax()
        dd = float(equity.iloc[-1] / peak.iloc[-1] - 1.0)
        if dd <= self.config.trailing_dd_limit:
            return [HealthIssue(HealthStatus.CRITICAL, "RISK_DRAWDOWN",
                                f"trailing drawdown {dd:.2%} breached limit {self.config.trailing_dd_limit:.2%}")]
        return []

    def check_regime_stability(self, regimes: Sequence[Optional[str]]) -> List[HealthIssue]:
        """Too many regime flips in a short window suggests an unstable model."""
        c = self.config
        recent = [r for r in list(regimes)[-c.regime_flip_window:] if r is not None]
        flips = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
        if flips > c.regime_flip_limit:
            return [HealthIssue(HealthStatus.WARN, "MODEL_REGIME_UNSTABLE",
                                f"{flips} regime flips in last {c.regime_flip_window} bars")]
        return []

    def check_crisis(self, current_regime: Optional[str]) -> List[HealthIssue]:
        if current_regime == "CRISIS":
            return [HealthIssue(HealthStatus.WARN, "MODEL_CRISIS",
                                "regime is CRISIS — defensive routing engaged")]
        return []

    # ----- orchestration ---------------------------------------------------------
    def run(
        self,
        close: pd.Series,
        equity: Optional[pd.Series] = None,
        regimes: Optional[Sequence[Optional[str]]] = None,
    ) -> HealthReport:
        issues: List[HealthIssue] = []
        issues += self.check_data_integrity(close)
        if equity is not None:
            issues += self.check_drawdown(equity)
        if regimes is not None:
            issues += self.check_regime_stability(regimes)
            issues += self.check_crisis(regimes[-1] if len(regimes) else None)

        status = max((i.level for i in issues), default=HealthStatus.OK)
        metrics = {"n_observations": int(len(close))}
        if equity is not None and len(equity):
            metrics["trailing_drawdown"] = round(float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0), 4)
        return HealthReport(status=status, issues=issues, metrics=metrics)
