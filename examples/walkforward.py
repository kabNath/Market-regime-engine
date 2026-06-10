"""Walk-forward, cost-aware evaluation + stress tests.

Run with::

    python examples/walkforward.py

Prints net-of-cost performance at several friction levels, then runs the
crash-scenario battery to verify the crisis machinery engages.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from regime_engine.backtest import CostModel, stress_test, walk_forward
from regime_engine.data import synthetic_regime_series


def main() -> None:
    price, _ = synthetic_regime_series()

    print("Walk-forward (regime-gated exposure), net of frictions")
    print("-" * 72)
    for bps in (0, 5, 10, 25):
        res = walk_forward(price, costs=CostModel(cost_bps=bps, slippage_bps=bps))
        print(f"  {2*bps:>3d} bps round-trip : {res.summary()}")

    print()
    print("Stress scenarios (synthetic crashes injected into the tape)")
    print("-" * 72)
    for outcome in stress_test(price):
        print(f"  {outcome}")


if __name__ == "__main__":
    main()
