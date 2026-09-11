"""Measure how many independent bets a portfolio actually holds.

Counting symbols overstates diversification badly. Three crypto majors that
correlate at 0.85 are not three bets, they are closer to one. This reports the
real number using two standard measures:

**Average pairwise correlation** - the blunt summary.

**Diversification ratio** = (weighted average volatility) / (portfolio
volatility). It is 1.0 when everything moves together and rises as holdings
become independent. Its square is roughly the effective number of independent
bets, which is the figure that actually drives risk-adjusted return: for a
given edge, N independent bets improve the Sharpe by about sqrt(N).

That last relationship is why this matters more than another forecasting
attempt. Going from 1 effective bet to 4 is worth the same as roughly doubling
your predictive skill - and unlike skill, it is close to arithmetic.

    python scripts/diversification.py -c config.diversified.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from quantbot.config import Config
from quantbot.data import load_universe
from quantbot.data.loader import align

logging.basicConfig(level=logging.WARNING)


def analyse(returns: pd.DataFrame, label: str) -> dict:
    corr = returns.corr()
    n = len(corr)
    iu = np.triu_indices(n, k=1)
    pair = corr.to_numpy()[iu]
    avg_corr = float(np.nanmean(pair))

    w = np.repeat(1.0 / n, n)                 # equal weight, for comparability
    vols = returns.std().to_numpy()
    cov = returns.cov().to_numpy()
    port_vol = float(np.sqrt(w @ cov @ w))
    weighted_vol = float(w @ vols)
    dr = weighted_vol / port_vol if port_vol > 0 else 1.0

    return {
        "label": label, "n_markets": n, "avg_corr": avg_corr,
        "max_corr": float(np.nanmax(pair)), "min_corr": float(np.nanmin(pair)),
        "div_ratio": dr, "effective_bets": dr**2,
        "sharpe_uplift": dr,     # for a fixed edge, Sharpe scales with DR
    }


def report(row: dict) -> str:
    return (
        f"  {row['label']:22s} {row['n_markets']:3d} markets  "
        f"avg corr {row['avg_corr']:+.2f}  "
        f"(min {row['min_corr']:+.2f}, max {row['max_corr']:+.2f})  "
        f"effective bets {row['effective_bets']:5.2f}  "
        f"Sharpe x{row['sharpe_uplift']:.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*",
                    default=["config.yaml", "config.stocks.yaml",
                             "config.diversified.yaml"])
    args = ap.parse_args()

    print("HOW MANY INDEPENDENT BETS IS THIS PORTFOLIO ACTUALLY MAKING?\n")
    rows = []
    for path in args.configs:
        if not os.path.exists(path):
            continue
        cfg = Config.from_yaml(path)
        try:
            frames = align(load_universe(cfg.data, use_cache=True, max_age_minutes=600))
        except Exception as exc:
            print(f"  {path}: could not load ({exc})")
            continue
        rets = pd.DataFrame({s: d["close"].pct_change() for s, d in frames.items()}).dropna()
        if len(rets) < 60 or rets.shape[1] < 2:
            print(f"  {path}: not enough overlapping history")
            continue
        row = analyse(rets, os.path.basename(path).replace("config.", "").replace(".yaml", ""))
        rows.append(row)
        print(report(row))

    if len(rows) >= 2:
        best = max(rows, key=lambda r: r["effective_bets"])
        worst = min(rows, key=lambda r: r["effective_bets"])
        gain = best["sharpe_uplift"] / max(worst["sharpe_uplift"], 1e-9)
        print(f"\n  {best['label']} holds {best['effective_bets']:.2f} effective bets "
              f"against {worst['effective_bets']:.2f} for {worst['label']}.")
        print(f"  For the SAME per-market edge that is roughly a {gain:.2f}x "
              f"better Sharpe, from correlation structure alone.")
        print("\n  This is not a prediction and does not depend on the model being")
        print("  right. It is the one improvement here that is close to arithmetic.")

    print("\n  Caveat: correlations rise in a crisis, exactly when the")
    print("  diversification is most needed. Treat these as fair-weather figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
