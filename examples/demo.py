"""End-to-end demo: detect regimes on a synthetic series and run the monitor.

Run with::

    python examples/demo.py

It prints the latest regime and a health report, and saves a regime-coloured
price chart to ``docs/regime_example.png``.
"""
from __future__ import annotations

import os
import sys

# Allow running as `python examples/demo.py` without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime_engine import RegimeDetector, HealthMonitor
from regime_engine.data import synthetic_regime_series

REGIME_COLORS = {
    "BULL": "#1b9e77",
    "NEUTRAL": "#7570b3",
    "BEAR": "#d95f02",
    "CRISIS": "#e7298a",
}


def main() -> None:
    price, _true_labels = synthetic_regime_series()

    detector = RegimeDetector()
    frame = detector.fit(price)

    latest = detector.classify_latest(price)
    print("Latest classification")
    print("  " + latest.explain())
    print()

    monitor = HealthMonitor()

    # A calm, recent uptrend: the monitor should pass cleanly.
    rng = np.random.default_rng(0)
    calm_rets = rng.normal(0.12 / 252, 0.10 / np.sqrt(252), 300)
    calm_idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
    calm = pd.Series(100.0 * np.exp(np.cumsum(calm_rets)), index=calm_idx, name="close")
    healthy_report = monitor.run(
        close=calm,
        equity=calm,
        regimes=detector.fit(calm)["regime"].tolist(),
    )
    print("Health check — calm recent book")
    print(healthy_report)
    print()

    # The full book ends in a deep drawdown: the monitor's risk kill-switch
    # should fire and mark the book non-tradeable.
    stressed_report = monitor.run(
        close=price,
        equity=price,  # treat the series as the book's equity curve for the demo
        regimes=frame["regime"].tolist(),
    )
    print("Health check — full book (stressed)")
    print(stressed_report)

    # --- plot price coloured by detected regime --------------------------------
    os.makedirs("docs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    regimes = frame["regime"].ffill()
    for regime, color in REGIME_COLORS.items():
        mask = regimes == regime
        ax.scatter(price.index[mask], price[mask], s=6, c=color, label=regime)
    ax.set_title("Detected market regimes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/regime_example.png", dpi=120)
    print("\nSaved chart -> docs/regime_example.png")


if __name__ == "__main__":
    main()
