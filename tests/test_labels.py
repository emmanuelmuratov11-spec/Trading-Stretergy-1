import numpy as np
import pandas as pd

from quantbot.config import LabelConfig
from quantbot.labels import binary_target, triple_barrier


def _frame(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="h", tz="UTC")
    s = pd.Series(prices, index=idx, dtype="float64")
    return s, s * 1.0005, s * 0.9995


def test_upper_barrier_is_detected():
    close, high, low = _frame([100.0] * 60 + [130.0] * 20)
    lab = triple_barrier(close, high, low, LabelConfig(horizon_bars=10, vol_window=20))
    touched = lab.iloc[55:60]["touch"]
    assert (touched == "upper").any(), "a sharp rally did not trigger the upper barrier"


def test_lower_barrier_is_detected():
    close, high, low = _frame([100.0] * 60 + [70.0] * 20)
    lab = triple_barrier(close, high, low, LabelConfig(horizon_bars=10, vol_window=20))
    assert (lab.iloc[55:60]["touch"] == "lower").any()


def test_binary_target_drops_unresolved_labels():
    """NaN != 0 is True, so a naive mask silently turns unresolved bars into
    'down' labels and trains the model on fabricated targets."""
    lab = pd.DataFrame({
        "label": [1.0, -1.0, np.nan, 0.0, 1.0],
        "ret": [0.1, -0.1, np.nan, 0.0, 0.1],
        "t_end": [1, 2, -1, 4, 5],
        "sigma": [0.01] * 5,
        "touch": ["upper", "lower", None, "vertical", "upper"],
    })
    y = binary_target(lab)
    assert y.isna().sum() == 2, "unresolved and flat labels must both be dropped"
    assert y.dropna().tolist() == [1.0, 0.0, 1.0]


def test_ambiguous_bars_resolve_pessimistically():
    """When one bar touches both barriers we cannot know which came first, so
    the label must assume the stop hit -- optimism here inflates backtests."""
    idx = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    close = pd.Series([100.0] * 40, index=idx)
    high = close.copy()
    low = close.copy()
    high.iloc[25:] = 100.0
    low.iloc[25:] = 100.0
    high.iloc[30] = 200.0    # both barriers inside one bar
    low.iloc[30] = 50.0
    lab = triple_barrier(close, high, low, LabelConfig(horizon_bars=10, vol_window=10,
                                                      min_sigma=0.01))
    assert lab.iloc[29]["touch"] == "lower"
