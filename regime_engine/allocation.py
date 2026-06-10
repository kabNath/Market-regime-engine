"""Risk-based portfolio allocation utilities.

These are the standard building blocks of a risk-managed systematic book:

* inverse-volatility (risk-parity-style) weighting,
* portfolio-level volatility targeting,
* a hard gross-exposure cap,
* top-N concentration.

None of this is proprietary alpha — it is the risk plumbing that sits *around*
a signal and keeps a live book inside its mandate. It is included here because
robust position sizing is as much a part of a production system as the model
that generates the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(returns: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Risk-parity-style weights: each asset weighted by the inverse of its
    realised volatility, normalised to sum to one.

    Parameters
    ----------
    returns:
        DataFrame of periodic returns, one column per asset.
    lookback:
        Trailing window used to estimate volatility.
    """
    vol = returns.tail(lookback).std()
    vol = vol.replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.fillna(0.0)
    total = inv.sum()
    if total == 0:
        # Degenerate case: fall back to equal weight.
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)
    return inv / total


def apply_vol_target(
    weights: pd.Series,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 63,
    ann_factor: int = 252,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale weights so the *portfolio* hits a target annualised volatility.

    The scaling factor is capped by ``max_leverage`` so the book never levers
    up beyond its mandate when realised volatility is low.
    """
    aligned = returns[weights.index].tail(lookback)
    cov = aligned.cov() * ann_factor
    port_var = float(weights.values @ cov.values @ weights.values)
    port_vol = np.sqrt(max(port_var, 1e-12))
    scale = min(target_vol / port_vol, max_leverage) if port_vol > 0 else 0.0
    return weights * scale


def cap_exposure(weights: pd.Series, cap: float = 0.95) -> pd.Series:
    """Cap gross exposure at ``cap`` (e.g. keep at least 5% in cash)."""
    gross = weights.abs().sum()
    if gross > cap:
        weights = weights * (cap / gross)
    return weights


def select_top_n(scores: pd.Series, n: int = 3) -> pd.Index:
    """Return the index of the ``n`` highest-scoring assets (concentration)."""
    return scores.sort_values(ascending=False).head(n).index
