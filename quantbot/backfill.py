"""Grade the model's historical out-of-sample predictions, right now.

The live journal is honest but slow: with a 24-hour horizon the first call
cannot be graded for a day, and a trustworthy sample takes weeks. Meanwhile the
walk-forward backtest has already made thousands of predictions on real data,
every one produced by a model that had never seen that bar and whose training
window was purged of overlapping labels. Those are out-of-sample calls with
known outcomes, so they can be graded immediately.

What this gives you that a backtest summary does not: the win rate, calibration
and regime attribution of the individual *calls*, rather than one blended
equity curve.

What it is NOT: a live track record. These fills are assumed, not executed, so
slippage and the discipline of actually placing the trade are absent. It
answers "was the model right?", not "would I have captured it?". The live
journal still answers the second, and it still needs wall-clock time.

**Predictions are sampled one horizon apart, not every bar.** Consecutive
overlapping calls share most of their outcome window, so counting all of them
would inflate the sample size and make a coin flip look statistically
significant. One call per horizon keeps the observations independent.
"""

from __future__ import annotations

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from quantbot.config import Config
from quantbot.data.base import TIMEFRAME_MINUTES, bars_per_year
from quantbot.features import realised_vol
from quantbot.journal import Journal

log = logging.getLogger(__name__)


def _context_series(df: pd.DataFrame, cfg: Config, bpy: float) -> pd.DataFrame:
    """Causal regime features for every bar, matching signals.generate()."""
    close = df["close"]
    logret = np.log(close).diff()
    lb = min(168, max(len(close) // 4, 10))
    vol = realised_vol(close, cfg.labels.vol_window, bpy)
    mom = np.log(close).diff(lb)
    mvol = logret.ewm(span=lb, adjust=False, min_periods=5).std() * np.sqrt(lb)
    roll_max = close.rolling(min(336, len(close)), min_periods=1).max()
    return pd.DataFrame({
        "vol": vol,
        "trend_z": (mom / mvol.replace(0.0, np.nan)),
        "drawdown": close / roll_max - 1.0,
    }, index=df.index)


def run(cfg: Config, frames: dict[str, pd.DataFrame], result,
        stride: int | None = None) -> Journal:
    """Replay a backtest's out-of-sample predictions into a graded journal."""
    bpy = bars_per_year(cfg.data.timeframe, cfg.data.annual_days)
    bar_minutes = TIMEFRAME_MINUTES[cfg.data.timeframe]
    horizon = cfg.labels.horizon_bars
    step = stride if stride and stride > 0 else horizon

    journal = Journal(max_entries=200_000)
    probs = result.probabilities
    weights = result.weights

    for sym in probs.columns:
        df = frames.get(sym)
        if df is None:
            continue
        ctx = _context_series(df, cfg, bpy)
        series = probs[sym].dropna()
        if series.empty:
            continue
        # Independent samples: one call per horizon, not one per bar.
        for ts in series.index[::step]:
            p = float(series.loc[ts])
            if not np.isfinite(p):
                continue
            w = float(weights[sym].get(ts, 0.0)) if sym in weights.columns else 0.0
            row = ctx.loc[ts] if ts in ctx.index else None
            context = {}
            if row is not None:
                context = {k: (float(v) if np.isfinite(v) else 0.0)
                           for k, v in row.items()}
            journal.record(
                ts=ts.to_pydatetime().astimezone(timezone.utc),
                symbol=sym, engine=cfg.model.kind, prob=p, target_weight=w,
                price=float(df.loc[ts, "close"]), horizon_bars=horizon,
                bar_minutes=bar_minutes,
                sigma=float(context.get("vol", 0.0)) / max(np.sqrt(bpy), 1.0),
                context=context,
            )

    # Grade everything whose horizon closed inside the data we hold.
    graded = journal.resolve(frames, now=max(d.index[-1] for d in frames.values())
                             .to_pydatetime().astimezone(timezone.utc))
    log.info("backfill: %d predictions recorded, %d graded (stride %d bars)",
             len(journal.predictions), graded, step)
    return journal
