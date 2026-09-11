"""Shared OHLCV contract.

Every data source returns the same thing: a UTC-indexed, ascending, duplicate-free
DataFrame with float columns [open, high, low, close, volume]. Enforcing that here
means the rest of the system never has to care where the bars came from.
"""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440,
}


class DataError(RuntimeError):
    """Raised when market data is missing, malformed, or untrustworthy."""


def bars_per_year(timeframe: str) -> float:
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise DataError(f"unsupported timeframe {timeframe!r}")
    # Crypto trades continuously, so the year is a real calendar year.
    return (365.0 * 24.0 * 60.0) / minutes


def validate_ohlcv(df: pd.DataFrame, symbol: str = "?") -> pd.DataFrame:
    """Normalise and sanity-check a bar frame, or raise DataError."""
    if df is None or len(df) == 0:
        raise DataError(f"{symbol}: no bars returned")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")

    out = df[OHLCV_COLUMNS].astype("float64").copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise DataError(f"{symbol}: index must be a DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"

    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Bars with non-positive prices are corrupt, not merely unusual.
    bad = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bool(bad.any()):
        out = out[~bad]
    if len(out) == 0:
        raise DataError(f"{symbol}: every bar was invalid")

    # A high below the low means the feed is broken; repair rather than trust.
    hi = out[["open", "high", "low", "close"]].max(axis=1)
    lo = out[["open", "high", "low", "close"]].min(axis=1)
    out["high"] = hi
    out["low"] = lo
    out["volume"] = out["volume"].clip(lower=0.0)
    return out
