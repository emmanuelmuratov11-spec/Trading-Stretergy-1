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
