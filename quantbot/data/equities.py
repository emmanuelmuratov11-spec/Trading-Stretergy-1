"""Stock market data from free, key-free sources.

Two adapters, tried in order:

**Stooq** serves plain daily CSV with no API key and no rate limit worth
worrying about. Symbols are suffixed by exchange: AAPL -> aapl.us.

**Yahoo Finance** is the fallback, via the chart endpoint used by its own
front end. No key either, but it is an undocumented endpoint and can change.

Both are daily bars only. That is not a real limitation here: the trend engine
works better at daily frequency anyway, and intraday equity data behind a free
key is not worth the fragility.

Equities differ from crypto in ways that matter downstream:

* ~252 trading days a year, not 365. Annualising a daily Sharpe by sqrt(365)
  on stock data overstates it by about 20%, so `DataConfig.annual_days`
  carries the right figure per asset class.
* Markets close. Gaps over nights, weekends and holidays are real price moves
  that no stop can protect you from, which is why a gap-risk warning belongs
  in the alert rather than a pretence that a stop is a guarantee.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from quantbot.data.base import DataError, validate_ohlcv

_TIMEOUT = 30


def _normalise(symbol: str) -> str:
    """'AAPL', 'aapl.us' and 'AAPL/USD' all mean the same instrument."""
    s = symbol.split("/")[0].strip().lower()
    return s if "." in s else f"{s}.us"


class StooqSource:
    """Daily equity/ETF bars from stooq.com. No API key."""

    name = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def fetch(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        if timeframe != "1d":
            raise DataError(
                f"stooq serves daily bars only (asked for {timeframe!r}). "
                "Set data.timeframe: 1d for equities."
            )
        ticker = _normalise(symbol)
        try:
            resp = requests.get(self.BASE, params={"s": ticker, "i": "d"},
                                timeout=_TIMEOUT)
        except Exception as exc:
            raise DataError(f"{symbol}: stooq request failed: {exc}") from exc
        if resp.status_code in (403, 407):
            raise DataError(f"{symbol}: stooq blocked by egress policy "
                            f"({resp.status_code}); not retrying")
        if resp.status_code != 200 or not resp.text.strip():
            raise DataError(f"{symbol}: stooq returned {resp.status_code}")

        text = resp.text.strip()
        # Stooq answers an unknown ticker with a body, not an error status.
        if text.lower().startswith("no data") or "\n" not in text:
            raise DataError(f"{symbol}: stooq has no data for {ticker!r}")

        df = pd.read_csv(io.StringIO(text))
        if "Date" not in df.columns:
            raise DataError(f"{symbol}: unexpected stooq columns {list(df.columns)}")
        df.columns = [c.strip().lower() for c in df.columns]
        df.index = pd.to_datetime(df["date"], utc=True)
        if "volume" not in df.columns:
            df["volume"] = 0.0      # some indices carry no volume
        out = validate_ohlcv(df, symbol)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        return out[out.index >= cutoff]


class YahooSource:
    """Daily bars from Yahoo Finance's chart endpoint. Fallback for stooq."""

    name = "yahoo"
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    def fetch(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        if timeframe != "1d":
            raise DataError(f"yahoo adapter serves daily bars only (got {timeframe!r})")
        ticker = symbol.split("/")[0].strip().upper()
        rng = "5y" if lookback_days > 730 else "2y" if lookback_days > 365 else "1y"
        try:
            resp = requests.get(
                self.BASE.format(sym=ticker),
                params={"range": rng, "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=_TIMEOUT,
            )
        except Exception as exc:
            raise DataError(f"{symbol}: yahoo request failed: {exc}") from exc
        if resp.status_code in (403, 407):
            raise DataError(f"{symbol}: yahoo blocked by egress policy "
                            f"({resp.status_code}); not retrying")
        if resp.status_code != 200:
            raise DataError(f"{symbol}: yahoo returned {resp.status_code}")

        try:
            result = resp.json()["chart"]["result"][0]
            stamps = result["timestamp"]
            q = result["indicators"]["quote"][0]
        except Exception as exc:
            raise DataError(f"{symbol}: could not parse yahoo response: {exc}") from exc

        df = pd.DataFrame({
            "open": q["open"], "high": q["high"], "low": q["low"],
            "close": q["close"], "volume": q.get("volume") or [0] * len(stamps),
        }, index=pd.to_datetime(stamps, unit="s", utc=True))
        # Yahoo emits nulls for halted sessions; they are gaps, not prices.
        df = df.dropna(subset=["open", "high", "low", "close"])
        time.sleep(0.2)
        out = validate_ohlcv(df, symbol)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        return out[out.index >= cutoff]
