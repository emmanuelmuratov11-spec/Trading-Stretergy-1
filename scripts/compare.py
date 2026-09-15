"""Compare a small, PRE-REGISTERED set of strategies on real market data.

This is deliberately not a search. The configurations below were chosen in
advance from a hypothesis -- that costs and noise dominate at hourly frequency,
and that a simple documented effect (time-series momentum) travels better than
a flexible model -- and every one of them is reported, winners and losers
alike. Reporting all of them is what separates this from tuning: a search that
only shows you its best result has told you nothing.

Each strategy is scored on two periods:

  DEV      the earlier stretch, which everything has been developed against
  HOLDOUT  the most recent stretch, untouched until the end

A strategy that looks good on DEV and falls apart on HOLDOUT has been fitted to
the past, which is the normal outcome and the thing worth detecting. One that
holds up across both has cleared a much higher bar -- still not proof, but the
first evidence worth acting on.

    python scripts/compare.py --holdout-frac 0.30
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
from quantbot.data.base import bars_per_year
from quantbot.data.loader import align
from quantbot.metrics import compute as compute_metrics

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare")

# The pre-registered set. Adding to this list after seeing results is how
# honest comparison turns into a search, so raise --trials if you do.
CANDIDATES = [
    ("ml     1h", "ml",    "1h"),
    ("trend  1h", "trend", "1h"),
    ("trend  6h", "trend", "6h"),
    ("trend  1d", "trend", "1d"),
    ("blend  1d", "blend", "1d"),
    # Added after the live record showed the ML engine losing on momentum-like
    # calls at this horizon. Derived from results, so it raises --trials.
    ("revert 1h", "revert", "1h"),
    ("revert 6h", "revert", "6h"),
]


def build(kind: str, timeframe: str, base: Config) -> Config:
    cfg = Config.from_dict(base.to_dict())
    cfg.model.kind = kind
    cfg.data.timeframe = timeframe
    if timeframe == "1d":
        cfg.labels.horizon_bars = 5
        cfg.labels.vol_window = 20
        cfg.model.embargo_bars = 5
        cfg.model.train_bars = 400
        cfg.model.min_train_bars = 150
        cfg.model.retrain_every = 20
        cfg.model.trend_lookbacks = [5, 20, 60, 120]
        cfg.risk.cov_span = 40
        cfg.risk.scalar_smoothing = 10
        cfg.costs.rebalance_every = 1
    elif timeframe == "6h":
        cfg.labels.horizon_bars = 8
        cfg.labels.vol_window = 40
        cfg.model.embargo_bars = 8
        cfg.model.train_bars = 900
        cfg.model.min_train_bars = 350
        cfg.model.retrain_every = 40
        cfg.model.trend_lookbacks = [4, 12, 28, 56]
        cfg.risk.cov_span = 60
        cfg.risk.scalar_smoothing = 20
        cfg.costs.rebalance_every = 2
    cfg.validate()
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    ap.add_argument("--trials", type=int, default=10,
                    help="total configurations tested against this data, ever")
    args = ap.parse_args()

    base = Config.from_yaml(args.config)
    rows = []

    # One shared CALENDAR cutoff for every candidate.
    #
    # Splitting each config at a fraction of its own scored bars gives every
    # timeframe a different holdout window: the ML run loses bars to warm-up,
    # so its holdout was 163 days while the daily runs got 269. Those windows
    # covered opposite markets (BTC +13% vs -11%), which made the "beat buy &
    # hold" column compare strategies against different realities. Anchoring
    # every candidate to the same dates is what makes the rows comparable.
    probe = Config.from_dict(base.to_dict())
    probe.data.timeframe = "1d"
    probe.validate()
    probe_frames = align(load_universe(probe.data, use_cache=True, max_age_minutes=600))
    # The span every symbol actually covers: latest start, earliest end.
    start = max(d.index[0] for d in probe_frames.values())
    end = min(d.index[-1] for d in probe_frames.values())
    cutoff = start + (end - start) * (1 - args.holdout_frac)
    print(f"shared holdout begins {cutoff:%Y-%m-%d} "
          f"(full span {start:%Y-%m-%d} to {end:%Y-%m-%d})\n")

    for label, kind, timeframe in CANDIDATES:
        cfg = build(kind, timeframe, base)
        try:
            frames = align(load_universe(cfg.data, use_cache=True, max_age_minutes=600))
        except Exception as exc:
            log.error("%s: could not load data: %s", label, exc)
            continue

        # Run ONE walk-forward backtest over the whole series, then score the
        # two periods separately. Re-running the backtest on the holdout alone
        # would make the model retrain from scratch inside it and spend most of
        # the window on warm-up, which measures the warm-up rather than the
        # strategy. Walk-forward already guarantees every prediction is
        # out-of-sample; the holdout is simply the stretch not looked at while
        # developing.
        try:
            res = backtest.run(cfg, frames, n_trials=args.trials)
        except Exception as exc:
            log.error("%s: backtest failed: %s", label, exc)
            continue

        bpy = bars_per_year(cfg.data.timeframe, cfg.data.annual_days)
        idx = res.returns.index
        cut = int(idx.searchsorted(cutoff))
        if cut < 30 or len(idx) - cut < 30:
            log.error("%s: too few bars either side of the shared cutoff", label)
            continue

        def score(sl: slice):
            r = res.returns.iloc[sl]
            eq = (1 + r).cumprod() * base.initial_capital
            w = res.weights.iloc[sl]
            b = res.benchmark_equity.iloc[sl]
            bret = b.pct_change().fillna(0.0)
            return (compute_metrics(r, eq, bpy, w, n_trials=args.trials),
                    compute_metrics(bret, (1 + bret).cumprod() * base.initial_capital,
                                    bpy, n_trials=1))

        dev_m, _ = score(slice(0, cut))
        hold_m, hold_b = score(slice(cut, None))
        rows.append((label, dev_m, hold_m, hold_b))
        print(f"  ran {label}  ({len(idx)} scored bars, dev={cut}, "
              f"holdout={len(idx) - cut}, holdout starts {idx[cut]:%Y-%m-%d})",
              flush=True)

    if not rows:
        print("nothing ran")
        return 1

    print("\n" + "=" * 96)
    print("PRE-REGISTERED COMPARISON ON REAL DATA   (all candidates shown, "
          f"trials={args.trials})")
    print("=" * 96)
    hdr = (f"{'strategy':11s} | {'DEV ret':>9s} {'DEV Shp':>8s} {'turn':>6s} | "
           f"{'HOLD ret':>9s} {'HOLD Shp':>9s} {'HOLD DD':>8s} | "
           f"{'its B&H':>8s} {'edge':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for label, dm, hm, hb in rows:
        # Each row is compared against the benchmark over ITS OWN window.
        # One shared benchmark row would compare different calendar spans.
        print(f"{label:11s} | {dm.total_return:+8.1%} {dm.sharpe:8.2f} "
              f"{dm.turnover:5.0f}x | "
              f"{hm.total_return:+8.1%} {hm.sharpe:9.2f} {hm.max_drawdown:8.1%} | "
              f"{hb.total_return:+8.1%} {hm.total_return - hb.total_return:+8.1%}")

    print("\nRead the HOLDOUT columns first -- DEV is where everything was")
    print("developed, so a good DEV number is the expected outcome, not news.")
    print("A strategy is only interesting if HOLDOUT return beats buy & hold")
    print("AND its deflated Sharpe clears 0.9. Anything else is noise.\n")
    for label, _, hm, hb in rows:
        beat = hm.total_return > hb.total_return
        print(f"  {label}   holdout DSR {hm.dsr:.3f}   "
              f"{'beat' if beat else 'lost to'} buy & hold"
              + ("   <- WORTH A LOOK" if hm.dsr >= 0.9 and beat else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
