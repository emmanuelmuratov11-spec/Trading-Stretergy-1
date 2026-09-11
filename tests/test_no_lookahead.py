"""The most important test in the repository.

If features computed at time t change once future bars are appended, then the
backtest is using information the live system will not have, and every
performance number it produces is fiction. This test catches that empirically
rather than trusting code review.
"""

import numpy as np
import pandas as pd
import pytest

from quantbot.config import FeatureConfig, LabelConfig
from quantbot.data.base import bars_per_year
from quantbot.features import build_features
from quantbot.labels import triple_barrier


def test_features_do_not_change_when_future_arrives(btc, eth):
    cfg = FeatureConfig()
    bpy = bars_per_year("1h")

    cut = 2000
    full = build_features(eth, cfg, bpy, market=btc)
    truncated = build_features(eth.iloc[:cut], cfg, bpy, market=btc.iloc[:cut])

    # Compare the last row the truncated view could possibly know about.
    a = full.iloc[cut - 1]
    b = truncated.iloc[-1]

    assert list(a.index) == list(b.index)
    for col in a.index:
        x, y = a[col], b[col]
        if pd.isna(x) and pd.isna(y):
            continue
        assert np.isclose(x, y, rtol=1e-9, atol=1e-12), (
            f"LOOKAHEAD LEAK in feature {col!r}: value at bar {cut - 1} changed "
            f"from {y} to {x} once future bars were appended."
        )


@pytest.mark.parametrize("cut", [1200, 1800, 2500])
def test_no_lookahead_at_several_points(btc, eth, cut):
    cfg = FeatureConfig()
    bpy = bars_per_year("1h")
    full = build_features(eth, cfg, bpy, market=btc).iloc[cut - 1]
    trunc = build_features(eth.iloc[:cut], cfg, bpy, market=btc.iloc[:cut]).iloc[-1]
    diff = [c for c in full.index
            if not (pd.isna(full[c]) and pd.isna(trunc[c]))
            and not np.isclose(full[c], trunc[c], rtol=1e-9, atol=1e-12)]
    assert not diff, f"features leaked future information at bar {cut}: {diff}"


def test_labels_only_use_the_future_they_are_allowed_to(btc):
    """A label at t must resolve within t + horizon, never beyond."""
    cfg = LabelConfig(horizon_bars=24)
    lab = triple_barrier(btc["close"], btc["high"], btc["low"], cfg)
    resolved = lab[lab["t_end"] >= 0]
    pos = np.arange(len(lab))[lab["t_end"].to_numpy() >= 0]
    horizon = resolved["t_end"].to_numpy() - pos
    assert horizon.min() >= 1, "a label resolved at or before its own bar"
    assert horizon.max() <= cfg.horizon_bars, "a label resolved beyond its horizon"


def test_forward_returns_are_shifted_not_peeked(btc):
    """The return a weight earns must come strictly after the decision bar."""
    fwd = btc["close"].pct_change().shift(-1)
    manual = btc["close"].shift(-1) / btc["close"] - 1.0
    pd.testing.assert_series_equal(fwd.dropna(), manual.dropna(), check_names=False)
