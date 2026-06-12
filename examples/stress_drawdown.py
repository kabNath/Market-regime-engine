#!/usr/bin/env python3
"""Generate docs/stress_drawdown.png — strategy vs buy-and-hold max drawdown
under injected stress scenarios.

The numbers in STRESS_RESULTS are the measured outputs of
`examples/walkforward.py`. They are kept here as plot inputs so this script has
no dependency on the backtest internals. If you change the strategy or the
scenarios, re-run walkforward.py and update the dict below.
(Want this to recompute automatically instead of reading literals? Share
walkforward.py and it can call your backtest API directly.)

    pip install matplotlib
    python examples/stress_drawdown.py     # writes docs/stress_drawdown.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# scenario -> (strategy maxDD %, buy-and-hold asset maxDD %)  [negative numbers]
STRESS_RESULTS = {
    "Flash crash":   (-25.8, -65.5),
    "Bear grind":    (-16.2, -65.6),
    "Vol explosion": (-18.7, -68.9),
}

STRAT_COLOR = "#2c7fb8"   # blue
ASSET_COLOR = "#d62728"   # red


def main() -> None:
    scenarios = list(STRESS_RESULTS)
    strat = np.array([STRESS_RESULTS[s][0] for s in scenarios])
    asset = np.array([STRESS_RESULTS[s][1] for s in scenarios])
    x = np.arange(len(scenarios))
    w = 0.38

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2))

    b1 = ax.bar(x - w / 2, strat, w, label="Regime-gated strategy", color=STRAT_COLOR)
    b2 = ax.bar(x + w / 2, asset, w, label="Buy & hold (asset)", color=ASSET_COLOR, alpha=0.85)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 13), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    # reduction annotations above each scenario
    for xi, (sdd, add) in zip(x, zip(strat, asset)):
        red = (1 - sdd / add) * 100
        ax.annotate(f"-{red:.0f}% DD", (xi, 2), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#2ca02c")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Maximum drawdown (%)")
    ax.set_ylim(min(asset.min(), strat.min()) - 8, 8)
    ax.set_title("Crisis gates cut max drawdown by ~60–75% under stress",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    out = Path("docs"); out.mkdir(exist_ok=True)
    path = out / "stress_drawdown.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
