import numpy as np
import pandas as pd
import pytest

from quantbot.config import DataConfig
from quantbot.data import load_universe
from quantbot.data.base import DataError, bars_per_year, validate_ohlcv
from quantbot.data.loader import align
from quantbot.data.synthetic import generate


def test_synthetic_data_is_reproducible():
    """hash() is salted per process; a naive seed would break determinism."""
    a = generate("BTC/USDT", bars=200, seed=3)
    b = generate("BTC/USDT", bars=200, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_different_symbols_differ():
    a = generate("BTC/USDT", bars=200, seed=3)
    b = generate("ETH/USDT", bars=200, seed=3)
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_validate_rejects_empty():
    with pytest.raises(DataError):
        validate_ohlcv(pd.DataFrame(), "X")


def test_validate_repairs_inverted_bars():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [10, 10, 10], "high": [5, 12, 11],
                       "low": [20, 8, 9], "close": [10, 11, 10],
                       "volume": [1, 1, -5]}, index=idx)
    out = validate_ohlcv(df, "X")
    assert (out["high"] >= out["low"]).all()
    assert (out["volume"] >= 0).all()


def test_validate_drops_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"], tz="UTC")
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                       "close": [1, 2, 3], "volume": [1, 1, 1]}, index=idx)
    assert len(validate_ohlcv(df, "X")) == 2


def test_index_is_normalised_to_utc():
    idx = pd.date_range("2024-01-01", periods=3, freq="h")   # naive
    df = pd.DataFrame({"open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
                       "close": [1.0] * 3, "volume": [1.0] * 3}, index=idx)
    assert str(validate_ohlcv(df, "X").index.tz) == "UTC"


def test_bars_per_year():
    assert bars_per_year("1h") == 365 * 24
    assert bars_per_year("1d") == 365
    with pytest.raises(DataError):
        bars_per_year("3s")


def test_align_restricts_to_shared_timestamps():
    a = generate("BTC/USDT", bars=300, seed=1)
    b = generate("ETH/USDT", bars=200, seed=1)
    out = align({"a": a, "b": b})
    assert len(out["a"]) == len(out["b"]) == 200


def test_load_universe_uses_synthetic_offline():
    cfg = DataConfig(source="synthetic", symbols=["BTC/USDT"], lookback_days=30)
    frames = load_universe(cfg, use_cache=False)
    assert "BTC/USDT" in frames and len(frames["BTC/USDT"]) > 100


def test_unknown_source_raises():
    with pytest.raises(DataError):
        load_universe(DataConfig(source="not-a-real-exchange"), use_cache=False)


def test_policy_denials_are_not_retried():
    """A blocked host answers 403 every time; retrying just wastes 16 seconds."""
    import requests

    from quantbot.data.exchanges import _is_policy_denial

    assert _is_policy_denial(requests.exceptions.ProxyError("Tunnel connection failed: 403 Forbidden"))
    assert not _is_policy_denial(requests.exceptions.ConnectionError("connection reset by peer"))
    assert not _is_policy_denial(ValueError("something else"))
