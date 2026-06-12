#!/usr/bin/env python3
"""Generate docs/regime_spy.png — regimes detected on real SPY data.

Run from the repo root so that `regime_engine` is importable:

    pip install yfinance matplotlib
    PYTHONPATH=. python examples/regime_real_data.py     # writes docs/regime_spy.png
    # (or `pip install -e .` once, then run normally)

Falls back to a network-free synthetic series if yfinance is unavailable, so
it always produces a figure. Label-agnostic: matches whatever strings your
detector emits (BULL/Bull/bull/RegimeType.BULL ... all work).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from regime_engine import RegimeDetector

# canonical regime -> colour (matched case-insensitively, as a substring)
REGIME_COLORS = {
    "BULL": "#2ca02c",     # green
    "NEUTRAL": "#9467bd",  # purple
    "BEAR": "#ff7f0e",     # orange
    "CRISIS": "#d62fa0",   # pink
}
CANON_ORDER = ["BULL", "NEUTRAL", "BEAR", "CRISIS"]


def color_for(label) -> tuple[str, str]:
    """Map any label to (display_name, colour), case/format-insensitive."""
    L = str(label).strip().upper()
    for key, col in REGIME_COLORS.items():
        if key in L:                      # handles "Bull", "bull", "RegimeType.BULL"
            return key, col
    return str(label), "#bbbbbb"          # unknown label -> grey, keep its name


def load_prices() -> tuple[pd.Series, str]:
    try:
        import yfinance as yf
        close = (
            yf.download("SPY", period="10y", auto_adjust=True, progress=False)["Close"]
            .squeeze()
            .dropna()
        )
        if len(close) > 250:
            close.name = "SPY"
            return close, "SPY — 10y (yfinance, auto-adjusted)"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] yfinance unavailable ({exc}); using synthetic series", file=sys.stderr)

    rng = np.random.default_rng(7)
    segments = [(500, 0.0006, 0.008), (180, 0.0, 0.011), (160, -0.0015, 0.022),
                (40, -0.010, 0.055), (420, 0.0007, 0.009)]
    rets = np.concatenate([rng.normal(mu, sd, n) for n, mu, sd in segments])
    idx = pd.bdate_range("2016-01-01", periods=len(rets))
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="SYNTH"), \
        "synthetic series with planted regimes"


def shade_regimes(ax, regimes: pd.Series) -> dict[str, str]:
    """Shade contiguous regime runs; return {display_name: colour} seen."""
    vals = regimes.values
    idx = regimes.index
    seen: dict[str, str] = {}
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            name, col = color_for(vals[start])
            seen[name] = col
            right = idx[i] if i < len(idx) else idx[-1]
            ax.axvspan(idx[start], right, color=col, alpha=0.22, linewidth=0)
            start = i
    return seen


def main() -> None:
    close, label = load_prices()

    detector = RegimeDetector()
    regimes = detector.fit(close)["regime"].reindex(close.index).ffill().dropna()
    close = close.loc[regimes.index]

    # diagnostic: show exactly what labels the detector emits
    print("regime labels from fit():", list(pd.unique(regimes.astype(str).values)))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 5))

    seen = shade_regimes(ax, regimes)
    ax.plot(close.index, close.values, color="#111111", linewidth=1.1, zorder=3)
    ax.set_yscale("log")
    ax.set_title(f"Market regimes detected on {label}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (log scale)")
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.margins(x=0)

    # legend ordered canonically, then any unknown labels; guarded against empty
    ordered = [r for r in CANON_ORDER if r in seen] + \
              [r for r in seen if r not in CANON_ORDER]
    handles = [Patch(facecolor=seen[r], alpha=0.5, label=r) for r in ordered]
    if handles:
        ax.legend(handles=handles, loc="upper left", frameon=False, ncol=len(handles))

    out = Path("docs"); out.mkdir(exist_ok=True)
    path = out / "regime_spy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}  ({len(close)} bars, regimes: {ordered})")


if __name__ == "__main__":
    main()
