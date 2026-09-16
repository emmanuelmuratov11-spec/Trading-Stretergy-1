"""Equity data adapter tests."""

import pytest

from quantbot.config import Config
from quantbot.data.base import DataError, bars_per_year
from quantbot.data.equities import StooqSource, YahooSource, _normalise
from quantbot.data.loader import get_source


def test_ticker_normalisation():
    assert _normalise("AAPL") == "aapl.us"
    assert _normalise("aapl.us") == "aapl.us"
    assert _normalise("SPY/USD") == "spy.us"


def test_equity_sources_are_registered():
    assert isinstance(get_source("stooq"), StooqSource)
    assert isinstance(get_source("yahoo"), YahooSource)


def test_intraday_is_rejected_rather_than_silently_wrong():
    """Free equity feeds are daily. Accepting '1h' and returning daily bars
    would corrupt every horizon calculation downstream."""
    for src in (StooqSource(), YahooSource()):
        with pytest.raises(DataError):
            src.fetch("AAPL", "1h", 100)


def test_equity_year_is_252_sessions_not_365_days():
    """Annualising a daily equity Sharpe by sqrt(365) overstates it ~20%."""
    assert bars_per_year("1d", 252) == 252
    assert bars_per_year("1d", 365) == 365
    assert bars_per_year("1h", 365) == 8760


def test_stocks_config_is_valid_and_uses_the_trend_engine():
    cfg = Config.from_yaml("config.stocks.yaml")
    cfg.validate()
    assert cfg.data.annual_days == 252
    assert cfg.data.asset_class == "equity"
    assert cfg.data.timeframe == "1d"
    assert cfg.model.kind == "trend"
    assert cfg.state_dir != Config.from_yaml("config.yaml").state_dir, (
        "stocks must keep a separate book from crypto"
    )


def test_crypto_and_stock_configs_do_not_share_state():
    crypto = Config.from_yaml("config.yaml")
    stocks = Config.from_yaml("config.stocks.yaml")
    assert crypto.state_dir != stocks.state_dir


def test_diversification_measure_distinguishes_real_from_fake_spread():
    """Counting symbols overstates diversification: correlated holdings are one
    bet wearing several hats, and the measure must say so."""
    import numpy as np
    import pandas as pd

    from scripts.diversification import analyse

    rng = np.random.default_rng(0)
    n = 1200
    common = rng.normal(0, 0.02, n)
    together = pd.DataFrame({f"C{i}": 0.95 * common + 0.05 * rng.normal(0, 0.02, n)
                             for i in range(4)})
    apart = pd.DataFrame({f"D{i}": rng.normal(0, 0.02, n) for i in range(4)})

    a, b = analyse(together, "together"), analyse(apart, "apart")
    assert a["avg_corr"] > 0.8 and b["avg_corr"] < 0.2
    assert a["effective_bets"] < 1.5, "4 correlated holdings are ~1 real bet"
    assert b["effective_bets"] > 3.0, "4 independent holdings are ~4 real bets"
    assert b["sharpe_uplift"] > a["sharpe_uplift"]


def test_diversified_config_is_valid_and_spans_asset_classes():
    cfg = Config.from_yaml("config.diversified.yaml")
    cfg.validate()
    syms = set(cfg.data.symbols)
    assert {"SPY", "TLT", "GLD"} <= syms, "needs equities, bonds and gold at minimum"
    assert len(syms) >= 10
    assert cfg.risk.max_position_weight <= 0.25, "no single market may dominate"


def test_long_lookbacks_request_full_history():
    """Capping the yahoo request at 5 years silently truncates any test that
    needs a crisis in the window, which is what invalidated the first crisis
    run. Stooq is deliberately NOT given a date range - see the test below."""
    import inspect

    src = inspect.getsource(YahooSource.fetch)
    assert '"max"' in src, "a multi-decade lookback must ask for full history"


def test_truncated_history_is_flagged():
    """The truncation must be loud. A quiet short window produced a crisis test
    that proved nothing while looking like it had run correctly - stooq served
    four years against a thirty-year request."""
    import pandas as pd

    from quantbot.data.loader import warn_if_truncated

    idx = pd.date_range("2022-09-19", "2026-09-15", freq="D", tz="UTC")
    short = {"SPY": pd.DataFrame({"close": range(len(idx))}, index=idx)}

    flagged = warn_if_truncated(short, requested_days=11000)
    assert flagged, "a 4-year answer to a 30-year request must be flagged"
    assert "SPY" in flagged[0] and "11000" in flagged[0]

    assert warn_if_truncated(short, requested_days=1000) == [], (
        "history that covers the request must not be flagged"
    )


def test_stooq_does_not_send_a_date_range():
    """Adding d1/d2 looked like the fix and was strictly worse: 73 bars per
    symbol against ~1457 for the plain request, leaving the universe with no
    overlapping timestamps."""
    import inspect
    assert '"d1"' not in inspect.getsource(StooqSource.fetch)


def test_a_short_cache_entry_is_not_reused(tmp_path):
    """One bad fetch poisoned the cache for every later run."""
    import pandas as pd

    from quantbot.data.cache import read as cache_read, write as cache_write

    idx = pd.date_range("2026-06-01", periods=73, freq="D", tz="UTC")
    short = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 1.0}, index=idx)
    cache_write(str(tmp_path), "stooq", "SPY", "1d", short)

    assert cache_read(str(tmp_path), "stooq", "SPY", "1d", 10_000,
                      min_span_days=1000) is None, "a 73-day cache entry must be a miss"
    assert cache_read(str(tmp_path), "stooq", "SPY", "1d", 10_000,
                      min_span_days=10) is not None
