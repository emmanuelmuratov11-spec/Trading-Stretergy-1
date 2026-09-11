"""Demonstration: how easy it is to manufacture a profitable-looking backtest.

Run this before you believe any backtest, including the ones this repo prints.

The setup is rigged so the answer is not in doubt. It generates a **driftless
random walk** -- a price series with, by construction, exactly zero predictable
edge. No strategy can genuinely make money on it. Then it searches N random
strategy configurations, keeps the one with the best Sharpe over the search
period, and re-runs that same winner on a later, untouched holdout period.

What you will see: the winner looks good on the period it was selected on, and
collapses on the period it was not. Nothing was learned. The search simply
found the configuration whose noise happened to line up, which is exactly what
happens when you "keep tuning until the backtest looks good" on real data --
except on real data there is no giveaway that the edge was never there.

This is why `--trials` exists, and why the deflated Sharpe ratio haircuts a
result by how many configurations you tried.

    python scripts/overfit_demo.py --trials 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quantbot import backtest
from quantbot.config import Config
from quantbot.data.base import validate_ohlcv
from quantbot.metrics import deflated_sharpe

logging.basicConfig(level=logging.ERROR)


def random_walk(bars: int, seed: int, start: float = 30_000.0) -> pd.DataFrame:
    """A driftless geometric random walk: provably zero predictable edge."""
    rng = np.random.default_rng(seed)
    vol = 0.60 / np.sqrt(365 * 24)
    rets = rng.normal(0.0, vol, bars)          # no drift, no autocorrelation
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    wick = np.abs(rng.normal(0, 1, bars)) * vol * close * 0.7
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"),
                        periods=bars, freq="h", tz="UTC")
    return validate_ohlcv(pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + wick,
        "low": np.clip(np.minimum(open_, close) - wick, 1e-6, None),
        "close": close, "volume": np.exp(rng.normal(10, 0.5, bars)),
    }, index=idx), "RW/USD")


def sample_config(rng: np.random.Generator) -> Config:
    """One arbitrary point in the parameter space a tuner would search."""
    c = Config()
    c.data.source, c.data.symbols = "synthetic", ["RW/USD"]
    c.data.benchmark = "RW/USD"
    c.model.train_bars = 800
    c.model.min_train_bars = 400
    c.model.retrain_every = 400
    c.model.n_seeds = 1
    c.model.max_iter = 60
    c.model.embargo_bars = c.labels.horizon_bars

    c.labels.horizon_bars = int(rng.choice([6, 12, 24, 48]))
    c.model.embargo_bars = c.labels.horizon_bars
    c.labels.upper_sigma = float(rng.uniform(0.8, 2.5))
    c.labels.lower_sigma = float(rng.uniform(0.8, 2.5))
    c.labels.vol_window = int(rng.choice([24, 48, 72, 168]))

    c.risk.kelly_fraction = float(rng.uniform(0.1, 1.0))
    c.risk.conviction_deadband = float(rng.uniform(0.0, 0.15))
    c.risk.max_position_weight = float(rng.uniform(0.2, 1.0))
    c.risk.target_annual_vol = float(rng.uniform(0.10, 0.60))
    c.risk.scalar_smoothing = int(rng.choice([1, 12, 48, 120]))
    c.costs.rebalance_band = float(rng.uniform(0.0, 0.5))
    c.features.rsi_window = int(rng.choice([7, 14, 21]))
    c.seed = int(rng.integers(0, 10_000))
    c.validate()
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--bars", type=int, default=4200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    full = random_walk(args.bars, args.seed)
    cut = int(len(full) * 0.58)
    search_data = {"RW/USD": full.iloc[:cut]}
    holdout_data = {"RW/USD": full.iloc[cut:]}

    print(__doc__.split("\n\n")[1].strip(), "\n")
    print(f"Searching {args.trials} configurations on a DRIFTLESS RANDOM WALK")
    print(f"(zero real edge by construction)   search={cut} bars, "
          f"holdout={len(full) - cut} bars\n")

    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(args.trials):
        cfg = sample_config(rng)
        try:
            res = backtest.run(cfg, search_data, n_trials=1)
        except Exception:
            continue
        results.append((res.metrics.sharpe, res.metrics.total_return, cfg))
        print(f"  trial {i + 1:3d}/{args.trials}  Sharpe {res.metrics.sharpe:+6.2f}",
              flush=True)

    if not results:
        print("no trials completed")
        return 1

    results.sort(key=lambda t: t[0], reverse=True)
    best_sharpe, best_ret, best_cfg = results[0]

    print(f"\n{'=' * 68}")
    print(f"BEST OF {len(results)} on the search period")
    print(f"{'=' * 68}")
    print(f"  Sharpe        {best_sharpe:+.2f}")
    print(f"  Total return  {best_ret:+.2%}")
    print("  ^ this is the number a tuner would show you.\n")

    hold = backtest.run(best_cfg, holdout_data, n_trials=1)
    print(f"{'=' * 68}")
    print("THE SAME CONFIGURATION, on data it was never selected on")
    print(f"{'=' * 68}")
    print(f"  Sharpe        {hold.metrics.sharpe:+.2f}")
    print(f"  Total return  {hold.metrics.total_return:+.2%}")
    print(f"  Max drawdown  {hold.metrics.max_drawdown:.2%}")

    naive = deflated_sharpe(best_sharpe, pd.Series([0.0]), 8760, 1)
    honest = deflated_sharpe(
        best_sharpe,
        pd.Series(np.random.default_rng(0).normal(0, 0.01, 2000)),
        8760, n_trials=len(results),
    )
    print(f"\n  Sharpe decayed by {best_sharpe - hold.metrics.sharpe:+.2f} "
          f"between the period it was picked on and the next one.")
    print(f"  Deflated Sharpe accounting for {len(results)} trials: {honest:.3f}")
    print("\n  Nothing was learned. There was nothing to learn: the price series")
    print("  was a coin flip. The search found the config whose noise fit best.")
    print("\n  On real data this failure looks identical -- except there is no")
    print("  giveaway, and you find out by losing money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
