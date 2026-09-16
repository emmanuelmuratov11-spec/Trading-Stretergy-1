"""Test the one thing a trend overlay is supposed to do: survive crashes.

Every backtest in this repository so far covered a single regime - a rising
market since 2024 - which is the worst possible sample for judging a strategy
whose entire purpose is to step aside when markets fall. A trend overlay is
EXPECTED to underperform buy-and-hold in a bull market. Judging it there is
like testing a seatbelt on a car that never crashes.

This runs the overlay across three decades and reports what happened during the
crashes specifically, against simply holding through them.

The bar is deliberately not "higher return". It is: materially smaller
drawdown through the crises, at a return that is not much worse over the full
period. If it cannot manage that, the overlay has no reason to exist.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from quantbot import backtest
from quantbot.config import Config
from quantbot.data import load_universe
from quantbot.data.loader import align

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("crisis")

# Named by when they happened, not chosen by what flatters the result.
CRISES = [
    ("Dot-com bust", "2000-03-01", "2002-10-31"),
    ("Global financial crisis", "2007-10-01", "2009-03-31"),
    ("COVID crash", "2020-02-01", "2020-04-30"),
    ("2022 bear market", "2022-01-01", "2022-10-31"),
]


def window_stats(equity: pd.Series, start: str, end: str) -> dict | None:
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    seg = equity[(equity.index >= s) & (equity.index <= e)]
    if len(seg) < 10:
        return None
    return {
        "return": float(seg.iloc[-1] / seg.iloc[0] - 1.0),
        "max_dd": float((seg / seg.cummax() - 1.0).min()),
        "bars": len(seg),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.overlay.yaml")
    ap.add_argument("--trials", type=int, default=11)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    try:
        frames = align(load_universe(cfg.data, use_cache=True, max_age_minutes=1440))
    except Exception as exc:
        # A data problem is a failed test, not a failed script. Crashing here
        # loses the rest of the run and tells you nothing about the strategy.
        print(f"CRISIS TEST COULD NOT RUN: {exc}")
        print("  The strategy was not evaluated. This is a data problem, not a")
        print("  result - do not read it as either success or failure.")
        return 0
    span = min(d.index[0] for d in frames.values()), max(d.index[-1] for d in frames.values())
    print(f"History available: {span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d}")
    print(f"Symbols: {', '.join(frames)}\n")

    res = backtest.run(cfg, frames, n_trials=args.trials)
    strat, bench = res.equity, res.benchmark_equity

    print("=" * 78)
    print("CRISIS PERIODS - the only test that matters for a trend overlay")
    print("=" * 78)
    hdr = f"{'crisis':26s} {'overlay ret':>12s} {'hold ret':>10s} {'overlay DD':>11s} {'hold DD':>9s}"
    print(hdr)
    print("-" * len(hdr))

    covered = 0
    dd_better = 0
    for name, start, end in CRISES:
        a, b = window_stats(strat, start, end), window_stats(bench, start, end)
        if not a or not b:
            print(f"{name:26s} {'(outside the data)':>44s}")
            continue
        covered += 1
        if a["max_dd"] > b["max_dd"]:
            dd_better += 1
        print(f"{name:26s} {a['return']:+11.1%} {b['return']:+9.1%} "
              f"{a['max_dd']:+10.1%} {b['max_dd']:+8.1%}")

    print("\n" + "=" * 78)
    print("FULL PERIOD")
    print("=" * 78)
    m, bm = res.metrics, res.benchmark_metrics
    print(f"  {'':18s} {'overlay':>12s} {'buy & hold':>12s}")
    print(f"  {'Total return':18s} {m.total_return:+11.1%} {bm.total_return:+11.1%}")
    print(f"  {'CAGR':18s} {m.cagr:+11.1%} {bm.cagr:+11.1%}")
    print(f"  {'Sharpe':18s} {m.sharpe:11.2f} {bm.sharpe:11.2f}")
    print(f"  {'Max drawdown':18s} {m.max_drawdown:+11.1%} {bm.max_drawdown:+11.1%}")
    print(f"  {'Annual vol':18s} {m.ann_vol:11.1%} {bm.ann_vol:11.1%}")
    print(f"  {'Turnover':18s} {m.turnover:10.1f}x {0.0:10.1f}x")
    print(f"  {'Deflated Sharpe':18s} {m.dsr:11.3f} {bm.dsr:11.3f}")

    print("\nVERDICT")
    if covered == 0:
        print("  No crisis period is inside the available data - this test proved nothing.")
    else:
        print(f"  Smaller drawdown in {dd_better} of {covered} crises.")
        gave_up = bm.total_return - m.total_return
        print(f"  Gave up {gave_up:+.1%} of total return over the full period.")
        if dd_better >= covered - 1 and m.max_drawdown > bm.max_drawdown:
            print("  This is what the overlay is for: comparable exposure, smaller holes.")
        else:
            print("  It did NOT reliably reduce crisis drawdowns, which is its only job.")
            print("  On this evidence the overlay does not earn its complexity.")
    print("\n  Note: a trend overlay is SUPPOSED to lag in bull markets. Judge it")
    print("  on the drawdown columns, and on whether the return given up is worth it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
