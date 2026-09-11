"""On-disk bar cache.

Refetching two years of hourly bars on every CI run is slow and rude to the
exchange. Cached bars older than `max_age_minutes` are refetched.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from quantbot.data.base import validate_ohlcv


def _path(cache_dir: str, source: str, symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_").replace("-", "_").upper()
    return os.path.join(cache_dir, f"{source}_{safe}_{timeframe}.csv")


def read(cache_dir: str, source: str, symbol: str, timeframe: str,
         max_age_minutes: float) -> pd.DataFrame | None:
    path = _path(cache_dir, source, symbol, timeframe)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 60.0 > max_age_minutes:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return validate_ohlcv(df, symbol)
    except Exception:
        return None


def write(cache_dir: str, source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    tmp = _path(cache_dir, source, symbol, timeframe) + ".tmp"
    df.to_csv(tmp)
    os.replace(tmp, _path(cache_dir, source, symbol, timeframe))
