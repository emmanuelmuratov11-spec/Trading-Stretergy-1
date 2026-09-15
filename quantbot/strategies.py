"""Signal engines.

Two families, deliberately very different in complexity:

**ml** - the gradient-boosted ensemble. ~46 features, hundreds of effective
parameters, refit weekly. Maximum flexibility, maximum capacity to fit noise.

**trend** - time-series momentum. Two parameters. Position is long when price
is above its own recent average, scaled by how strong that move is relative to
volatility.

The trend engine exists because it has the better *prior*. Time-series momentum
is the most robustly documented effect in finance: it holds across equities,
bonds, currencies and commodities, over more than a century of data, and it is
what the managed-futures industry has run on for decades. It is not a pattern
somebody found by searching this dataset -- which is exactly what makes it more
believable than a backtest fitted here.

Fewer parameters means less to overfit. When a 2-parameter rule and a
300-parameter model score similarly in a backtest, the 2-parameter rule is far
more likely to still work next year.

Neither engine is expected to beat buy-and-hold. Both are measured against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.config import Config

EPS = 1e-12


def trend_probabilities(df: pd.DataFrame, cfg: Config, bars_per_year: float) -> pd.Series:
    """Multi-horizon time-series momentum, expressed as a pseudo-probability.

    Returning a probability in [0, 1] rather than a raw position lets the trend
    engine flow through exactly the same sizing, risk and cost machinery as the
    ML model, so the two are compared on equal terms.

    Averaging several lookbacks matters more than picking the best one. A single
    horizon is a parameter begging to be tuned; the average of four is far more
    stable and is what practitioners actually run.
    """
    close = df["close"]
    logret = np.log(close).diff()

    scores = []
    for lb in cfg.model.trend_lookbacks:
        # Momentum over the lookback, normalised by the volatility of that same
        # horizon, so the score means the same thing in calm and wild regimes.
        mom = np.log(close).diff(lb)
        vol = logret.ewm(span=max(lb, 10), adjust=False, min_periods=5).std() * np.sqrt(lb)
        scores.append(mom / (vol + EPS))

    z = pd.concat(scores, axis=1).mean(axis=1)
    # tanh squashes outliers instead of letting one violent move dominate.
    return 0.5 + 0.5 * np.tanh(z / max(cfg.model.trend_scale, EPS))


def revert_probabilities(df: pd.DataFrame, cfg: Config, bars_per_year: float) -> pd.Series:
    """Short-horizon mean reversion: buy weakness, fade strength.

    This engine was derived from the live record's FAILURES rather than from a
    backtest search. Over 213 graded calls the ML model won 64% buying off the
    highs and in downtrends, and only 31% buying near the highs and 38% in
    quiet markets. Inverted, those losing buckets are 69% and 62%. A signal
    that is reliably wrong carries as much information as one that is reliably
    right - you flip it.

    Read together those buckets say one thing: at a 24-hour horizon this market
    mean-reverts, and the model was behaving like a momentum signal. That also
    matches a documented prior rather than being pure data-mining - short-
    horizon reversal and long-horizon momentum are separate, well-established
    effects, which is exactly why the trend engine earned money at 6h and 1d
    while losing at 1h.

    IMPORTANT: being derived from observed results makes this in-sample by
    construction. The dev/holdout comparison is what decides whether it is real,
    and finding it by scanning regime buckets raises the multiple-testing count.
    """
    close = df["close"]
    logret = np.log(close).diff()

    scores = []
    for lb in cfg.model.revert_lookbacks:
        ma = close.rolling(lb, min_periods=max(2, lb // 2)).mean()
        vol = logret.ewm(span=max(lb, 5), adjust=False, min_periods=3).std() * np.sqrt(lb)
        # Negative of the trend score: stretched BELOW the mean is bullish.
        scores.append(-(np.log(close / ma)) / (vol + EPS))

    z = pd.concat(scores, axis=1).mean(axis=1)
    return 0.5 + 0.5 * np.tanh(z / max(cfg.model.revert_scale, EPS))


def probabilities(df: pd.DataFrame, cfg: Config, bars_per_year: float,
                  ml_fn=None) -> tuple[pd.Series, int]:
    """Dispatch to the configured engine. Returns (probabilities, n_windows)."""
    kind = cfg.model.kind
    if kind == "trend":
        return trend_probabilities(df, cfg, bars_per_year), 0
    if kind == "revert":
        return revert_probabilities(df, cfg, bars_per_year), 0
    if kind == "ml":
        if ml_fn is None:
            raise ValueError("the ml engine needs a walk-forward fitter")
        return ml_fn()
    if kind == "blend":
        # Average the two views. Agreement concentrates the position; when they
        # disagree the result sits near 0.5, which the deadband then ignores --
        # a sensible default when neither has demonstrated an edge.
        if ml_fn is None:
            raise ValueError("the blend engine needs a walk-forward fitter")
        ml_p, n = ml_fn()
        tr_p = trend_probabilities(df, cfg, bars_per_year)
        return (ml_p.fillna(0.5) * 0.5 + tr_p.reindex(ml_p.index).fillna(0.5) * 0.5), n
    raise ValueError(
        f"unknown model.kind {kind!r}; use ml, trend, revert or blend")
