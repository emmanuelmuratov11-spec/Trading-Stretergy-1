"""Triple-barrier labelling (López de Prado, *Advances in Financial ML*).

Fixed-horizon labels ("will price be higher in 24 bars?") ignore the path taken
to get there. A trade that dives 8% before recovering is not a winner in any
sense that matters, because a real stop would have closed it. The triple barrier
labels by which of three barriers the price touches first:

    upper (profit target)  -> +1
    lower (stop loss)      -> -1
    vertical (time out)    ->  sign of the return at expiry

Barriers are scaled by trailing realised volatility, so the label means the same
thing across calm and violent regimes.

Each label also carries `t_end`, the bar at which it resolved. Two labels whose
windows overlap share information, so training on one and testing on the other
leaks. `validation.py` uses `t_end` to purge exactly those overlaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.config import LabelConfig


def triple_barrier(close: pd.Series, high: pd.Series, low: pd.Series,
                   cfg: LabelConfig) -> pd.DataFrame:
    """Return a frame with columns [label, ret, t_end, sigma, touch].

    `label` is in {-1, 0, +1}; 0 only occurs when the horizon expires exactly flat.
    `t_end` is the integer index position at which the label resolved.
    """
    n = len(close)
    logret = np.log(close).diff()
    sigma = logret.ewm(span=cfg.vol_window, adjust=False,
                       min_periods=cfg.vol_window // 2).std()
    sigma = sigma.clip(lower=cfg.min_sigma)

    c = close.to_numpy(dtype="float64")
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    s = sigma.to_numpy(dtype="float64")

    labels = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    t_end = np.full(n, -1, dtype="int64")
    touch = np.empty(n, dtype=object)

    H = int(cfg.horizon_bars)

    for i in range(n):
        if not np.isfinite(s[i]) or i + 1 >= n:
            continue
        entry = c[i]
        up = entry * np.exp(cfg.upper_sigma * s[i] * np.sqrt(H))
        dn = entry * np.exp(-cfg.lower_sigma * s[i] * np.sqrt(H))
        end = min(i + H, n - 1)

        hit = 0
        # Walk forward bar by bar. Within a bar we cannot know whether the high
        # or the low came first, so when both barriers are touched in the same
        # bar we resolve pessimistically: assume the stop hit first.
        for j in range(i + 1, end + 1):
            up_hit = h[j] >= up
            dn_hit = lo[j] <= dn
            if up_hit and dn_hit:
                hit = -1
                break
            if dn_hit:
                hit = -1
                break
            if up_hit:
                hit = 1
                break
        else:
            j = end

        if hit == 1:
            labels[i], rets[i], touch[i] = 1.0, float(np.log(up / entry)), "upper"
        elif hit == -1:
            labels[i], rets[i], touch[i] = -1.0, float(np.log(dn / entry)), "lower"
        else:
            r = float(np.log(c[j] / entry))
            labels[i] = float(np.sign(r))
            rets[i], touch[i] = r, "vertical"
        t_end[i] = j

    return pd.DataFrame(
        {"label": labels, "ret": rets, "t_end": t_end,
         "sigma": s, "touch": touch},
        index=close.index,
    )


def binary_target(labels: pd.DataFrame) -> pd.Series:
    """Map {-1, 0, +1} to {0, 1} for a classifier.

    Unresolved labels (NaN, from the volatility warm-up or the final horizon
    where no future bars exist) and exactly-flat labels are returned as NaN so
    callers drop them. Note that `NaN != 0` is True, so the mask must test
    `notna()` explicitly -- otherwise unresolved bars silently become "down"
    and the model trains on fabricated targets.
    """
    y = labels["label"]
    valid = y.notna() & (y != 0)
    return (y > 0).astype("float64").where(valid)
