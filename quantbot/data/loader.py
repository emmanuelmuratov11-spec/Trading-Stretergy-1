"""Universe loading: fetch, cache, align, and cross-check symbols."""

from __future__ import annotations

import logging

import pandas as pd

from quantbot.config import DataConfig
from quantbot.data.base import DataError
from quantbot.data.cache import read as cache_read, write as cache_write
from quantbot.data.exchanges import BinanceSource, CoinbaseSource
from quantbot.data.synthetic import SyntheticSource

log = logging.getLogger(__name__)

_SOURCES = {
    "binance": BinanceSource,
    "coinbase": CoinbaseSource,
    "synthetic": SyntheticSource,
}


def get_source(name: str, **kwargs: object):
    try:
        return _SOURCES[name](**kwargs)
    except KeyError:
        raise DataError(
            f"unknown data source {name!r}; choose from {sorted(_SOURCES)}"
        ) from None


def load_universe(
    cfg: DataConfig,
    use_cache: bool = True,
    max_age_minutes: float = 30.0,
    fallback: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load bars for every configured symbol.

    Falls back to the alternate exchange when the primary is unreachable, so a
    single blocked host does not take the whole run down. A symbol that cannot
    be loaded anywhere is dropped with a warning rather than killing the run --
    but if *nothing* loads we raise, because trading on an empty universe is
    worse than failing loudly.
    """
    order = [cfg.source]
    if fallback and cfg.source in ("binance", "coinbase"):
        order.append("coinbase" if cfg.source == "binance" else "binance")

    out: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for symbol in cfg.symbols:
        df = None
        if use_cache:
            df = cache_read(cfg.cache_dir, cfg.source, symbol, cfg.timeframe, max_age_minutes)
            if df is not None:
                log.info("%s: %d bars from cache", symbol, len(df))

        if df is None:
            last_err = "no source attempted"
            for source_name in order:
                try:
                    src = get_source(source_name)
                    df = src.fetch(symbol, cfg.timeframe, cfg.lookback_days)
                    log.info("%s: %d bars from %s", symbol, len(df), source_name)
                    if use_cache:
                        cache_write(cfg.cache_dir, cfg.source, symbol, cfg.timeframe, df)
                    break
                except Exception as exc:
                    last_err = f"{source_name}: {exc}"
                    log.warning("%s: %s", symbol, last_err)
                    df = None
            if df is None:
                failures[symbol] = last_err
                continue

        out[symbol] = df

    if not out:
        raise DataError(
            "no symbols could be loaded. Details: "
            + "; ".join(f"{k} -> {v}" for k, v in failures.items())
        )
    if failures:
        log.warning("dropped %d symbol(s): %s", len(failures), sorted(failures))
    return out


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every symbol to the shared timestamp index.

    Cross-asset features compare symbols bar for bar, so they must sit on one
    clock. Symbols are reindexed onto the intersection, never forward-filled
    across gaps -- a stale bar presented as fresh is a silent lie.
    """
    if not frames:
        return {}
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise DataError("symbols share no overlapping timestamps")
    return {sym: df.loc[common].copy() for sym, df in frames.items()}
