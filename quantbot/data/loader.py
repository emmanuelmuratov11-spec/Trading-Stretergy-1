"""Universe loading: fetch, cache, align, and cross-check symbols."""

from __future__ import annotations

import logging

import pandas as pd

from quantbot.config import DataConfig
from quantbot.data.base import DataError
from quantbot.data.cache import read as cache_read, write as cache_write
from quantbot.data.equities import StooqSource, YahooSource
from quantbot.data.exchanges import BinanceSource, CoinbaseSource
from quantbot.data.synthetic import SyntheticSource

log = logging.getLogger(__name__)

_SOURCES = {
    "binance": BinanceSource,
    "coinbase": CoinbaseSource,
    "stooq": StooqSource,
    "yahoo": YahooSource,
    "synthetic": SyntheticSource,
}

# Sources that cover the same instruments, so one can stand in for the other.
_FALLBACKS = {
    "binance": "coinbase",
    "coinbase": "binance",
    "stooq": "yahoo",
    "yahoo": "stooq",
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
    if fallback and cfg.source in _FALLBACKS:
        order.append(_FALLBACKS[cfg.source])

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

    warn_if_truncated(out, cfg.lookback_days)
    return out


def warn_if_truncated(frames: dict[str, pd.DataFrame], requested_days: int,
                      tolerance: float = 0.6) -> list[str]:
    """Flag any symbol whose returned history is far shorter than requested.

    A source that quietly serves a shorter window than asked for invalidates
    every conclusion that depends on the period, and the truncation is
    invisible unless it is checked. This is not hypothetical: stooq returned
    four years against a thirty-year request, so a crisis test ran over a
    window containing no crisis and would have looked perfectly healthy.
    """
    messages: list[str] = []
    for sym, df in frames.items():
        if len(df) < 2:
            continue
        span_days = (df.index[-1] - df.index[0]).days
        if span_days < requested_days * tolerance:
            msg = (f"{sym}: asked for {requested_days} days of history but got "
                   f"{span_days} ({df.index[0].date()} to {df.index[-1].date()}). "
                   "Any conclusion that depends on the period is unreliable.")
            log.warning("%s", msg)
            messages.append(msg)
    return messages


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
