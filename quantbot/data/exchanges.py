"""Public REST adapters for spot exchanges.

Only public market-data endpoints are used, so no API key is required and
nothing here can move money. Both adapters page backwards until the requested
lookback is covered, because every exchange caps bars per request.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from quantbot.data.base import TIMEFRAME_MINUTES, DataError, validate_ohlcv

_TIMEOUT = 25
_RETRIES = 4


def _is_policy_denial(exc: Exception) -> bool:
    """True for an egress-policy block rather than a transient network fault.

    A blocked host answers CONNECT with 403 and will answer 403 every time, so
    retrying wastes ~16s per symbol and tells the user nothing new. requests
    surfaces this as a ProxyError, not an HTTP status, so it has to be sniffed
    from the exception rather than from a response code.
    """
    text = str(exc).lower()
    return isinstance(exc, requests.exceptions.ProxyError) or (
        "tunnel connection failed" in text
        and ("403" in text or "407" in text)
    )


def _get(url: str, params: dict) -> object:
    """GET with backoff. Rate limits are retried; policy denials are not."""
    last: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                time.sleep(2**attempt)
                last = DataError(f"{resp.status_code} from {url}")
                continue
            if resp.status_code in (403, 407):
                raise DataError(
                    f"{resp.status_code} from {url} - blocked by network egress "
                    "policy. This host is unreachable from this environment."
                )
            resp.raise_for_status()
            return resp.json()
        except DataError:
            raise
        except Exception as exc:
            if _is_policy_denial(exc):
                raise DataError(
                    f"{url} is blocked by this network's egress policy "
                    f"(proxy refused the tunnel). Not retrying. If you are "
                    f"running in a restricted sandbox, use data.source: "
                    f"synthetic, or run this where the exchange is reachable."
                ) from exc
            last = exc  # transient network flake -- worth another try
            time.sleep(2**attempt)
    raise DataError(f"failed to fetch {url}: {last}")


class BinanceSource:
    """Binance spot klines. Symbols look like 'BTC/USDT'."""

    name = "binance"
    BASE = "https://api.binance.com/api/v3/klines"
    LIMIT = 1000

    def fetch(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        if timeframe not in TIMEFRAME_MINUTES:
            raise DataError(f"unsupported timeframe {timeframe!r}")
        market = symbol.replace("/", "").replace("-", "").upper()
        minutes = TIMEFRAME_MINUTES[timeframe]
        needed = int(lookback_days * 24 * 60 / minutes) + 5

        rows: list[list] = []
        end_ms = int(time.time() * 1000)
        while len(rows) < needed:
            batch = _get(self.BASE, {
                "symbol": market, "interval": timeframe,
                "limit": min(self.LIMIT, needed - len(rows)), "endTime": end_ms,
            })
            if not batch:
                break
            rows = list(batch) + rows
            end_ms = int(batch[0][0]) - 1
            if len(batch) < 2:
                break
            time.sleep(0.12)

        if not rows:
            raise DataError(f"{symbol}: binance returned no klines")

        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore",
        ])
        df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        return validate_ohlcv(df, symbol)


class CoinbaseSource:
    """Coinbase Exchange candles. Used as a fallback when Binance is unreachable."""

    name = "coinbase"
    BASE = "https://api.exchange.coinbase.com/products/{product}/candles"
    LIMIT = 300
    GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}

    def fetch(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        gran = self.GRANULARITY.get(timeframe)
        if gran is None:
            raise DataError(f"coinbase does not offer timeframe {timeframe!r}")
        product = symbol.replace("/", "-").upper().replace("USDT", "USD")

        needed = int(lookback_days * 86400 / gran) + 5
        end = pd.Timestamp.now(tz="UTC")
        frames: list[pd.DataFrame] = []
        collected = 0
        while collected < needed:
            start = end - pd.Timedelta(seconds=gran * self.LIMIT)
            batch = _get(self.BASE.format(product=product), {
                "granularity": gran,
                "start": start.isoformat(),
                "end": end.isoformat(),
            })
            if not batch:
                break
            part = pd.DataFrame(batch, columns=["time", "low", "high", "open", "close", "volume"])
            frames.append(part)
            collected += len(part)
            end = start
            if len(part) < 2:
                break
            time.sleep(0.25)

        if not frames:
            raise DataError(f"{symbol}: coinbase returned no candles")

        df = pd.concat(frames, ignore_index=True)
        df.index = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
        return validate_ohlcv(df, symbol)
