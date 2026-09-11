"""Offline market simulator.

Used for tests and for any environment without exchange connectivity. It is a
regime-switching stochastic-volatility process with fat tails and realistic
intraday structure -- close enough to real markets to exercise every code path,
and nowhere near real enough that results on it mean anything about real edge.

Backtest numbers produced on synthetic data are plumbing checks, not evidence.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from quantbot.data.base import TIMEFRAME_MINUTES, validate_ohlcv


def generate(
    symbol: str = "BTC/USDT",
    bars: int = 6000,
    timeframe: str = "1h",
    seed: int = 0,
    start_price: float = 30_000.0,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # Python's builtin hash() is salted per process (PYTHONHASHSEED), which
    # would make "synthetic" data differ between runs and silently destroy
    # test reproducibility. Derive the seed from a stable digest instead.
    digest = hashlib.sha256(f"{symbol}:{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    minutes = TIMEFRAME_MINUTES[timeframe]
    per_year = (365 * 24 * 60) / minutes

    # Two-state regime: calm/trending vs turbulent/mean-reverting.
    p_switch = 1.5 / per_year * 24
    regime = np.zeros(bars, dtype=int)
    for i in range(1, bars):
        regime[i] = 1 - regime[i - 1] if rng.random() < p_switch else regime[i - 1]

    base_vol = np.where(regime == 0, 0.45, 0.95) / np.sqrt(per_year)
    drift = np.where(regime == 0, 0.22, -0.18) / per_year

    # Stochastic volatility with clustering (GARCH-like persistence).
    vol = np.zeros(bars)
    vol[0] = base_vol[0]
    for i in range(1, bars):
        vol[i] = 0.94 * vol[i - 1] + 0.06 * base_vol[i] * np.exp(rng.normal(0, 0.22))

    # Student-t shocks give the fat tails that Gaussian models miss.
    shocks = rng.standard_t(df=4, size=bars) / np.sqrt(4 / 2)
    rets = drift + vol * shocks

    # A touch of short-horizon autocorrelation so momentum features are not pure noise.
    for i in range(1, bars):
        rets[i] += 0.035 * rets[i - 1]

    close = start_price * np.exp(np.cumsum(rets))

    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 1, bars)) * vol * close * 0.8
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    low = np.clip(low, 1e-6, None)
    volume = np.exp(rng.normal(10, 0.6, bars)) * (1 + 4 * np.abs(rets) / (vol + 1e-9) * 0.1)

    end = end or pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end=end, periods=bars, freq=f"{minutes}min", tz="UTC")

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    return validate_ohlcv(df, symbol)


class SyntheticSource:
    name = "synthetic"

    def __init__(self, seed: int = 0, **_: object) -> None:
        self.seed = seed

    def fetch(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        minutes = TIMEFRAME_MINUTES[timeframe]
        bars = max(600, int(lookback_days * 24 * 60 / minutes))
        start = {"BTC/USDT": 30_000.0, "ETH/USDT": 1_900.0, "SOL/USDT": 95.0}.get(symbol, 100.0)
        return generate(symbol, bars=bars, timeframe=timeframe, seed=self.seed, start_price=start)
