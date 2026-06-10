"""Data loading helpers and a synthetic regime-switching price generator.

The synthetic generator lets the demo and the tests run with **no network and
no data files**, while producing a series that contains clearly-labelled bull,
bear and crisis segments so the detector has something meaningful to find.
For real use, :func:`load_ohlcv_csv` reads a standard OHLCV file; any source
(yfinance, a broker export, QuantConnect) works as long as it has a ``close``
column and a datetime index.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def load_ohlcv_csv(path: str, date_col: str = "date", close_col: str = "close") -> pd.Series:
    """Load a closing-price series from a CSV file with a datetime index."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)[close_col].sort_index()


def synthetic_regime_series(
    seed: int = 7,
    end: str | None = None,
) -> Tuple[pd.Series, List[str]]:
    """Generate a daily price series with embedded regimes.

    By default the series ends on the most recent business day so that
    freshness checks behave naturally in the demo. Returns the price series and
    the per-bar true regime label (a sanity reference, not used by the
    detector).
    """
    rng = np.random.default_rng(seed)

    # (label, n_days, annual_drift, annual_vol)
    segments = [
        ("BULL",   320,  0.18, 0.12),
        ("NEUTRAL",120,  0.02, 0.14),
        ("BEAR",   180, -0.22, 0.24),
        ("CRISIS",  40, -0.55, 0.55),
        ("BULL",   400,  0.20, 0.13),
        ("NEUTRAL",150,  0.03, 0.15),
        ("BEAR",   140, -0.18, 0.22),
    ]

    daily_rets: List[float] = []
    labels: List[str] = []
    for label, n, drift, vol in segments:
        mu = drift / 252.0
        sigma = vol / np.sqrt(252.0)
        daily_rets.extend(rng.normal(mu, sigma, n).tolist())
        labels.extend([label] * n)

    end_ts = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    idx = pd.bdate_range(end=end_ts, periods=len(daily_rets))
    price = 100.0 * np.exp(np.cumsum(daily_rets))
    return pd.Series(price, index=idx, name="close"), labels
