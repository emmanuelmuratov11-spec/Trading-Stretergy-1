"""Chart rendering tests.

These assert that charts are produced and are non-trivial, not that they look
right -- that needs human eyes. What they do protect against is the common
failure of a reporting path silently breaking and nobody noticing for weeks,
because nobody reads a chart that was never generated.
"""

import os

import pytest

from quantbot import backtest
from quantbot.data import load_universe
from quantbot.data.loader import align
from quantbot.portfolio import Portfolio

pytest.importorskip("matplotlib")


def _frames(cfg):
    cfg.data.lookback_days = 120
    return align(load_universe(cfg.data, use_cache=False))


def test_backtest_chart_is_written(cfg, tmp_path):
    from quantbot import plotting
    res = backtest.run(cfg, _frames(cfg))
    out = str(tmp_path / "nested" / "report.png")
    path = plotting.render_backtest(res, cfg, out, title="test")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 20_000, "chart looks empty"


def test_portfolio_chart_needs_history(cfg, tmp_path):
    from quantbot import plotting
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    assert plotting.render_portfolio(pf, cfg, str(tmp_path / "a.png")) is None

    for _ in range(6):
        pf.mark({"BTC/USDT": 50_000.0})
    assert plotting.render_portfolio(pf, cfg, str(tmp_path / "b.png")) is not None


def test_series_colours_are_distinct_and_stable():
    """Strategy is blue and benchmark is orange in every panel: colour follows
    the entity, so a reader who learned 'blue is the strategy' stays right."""
    from quantbot import plotting
    assert plotting.STRATEGY != plotting.BENCHMARK
    assert plotting.STRATEGY == "#2a78d6" and plotting.BENCHMARK == "#eb6834"


def test_chart_surface_is_opaque(cfg, tmp_path):
    """Transparent PNGs render dark-on-dark in chat apps with dark themes."""
    from PIL import Image  # ships with matplotlib's test deps
    from quantbot import plotting
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    for _ in range(6):
        pf.mark({"BTC/USDT": 50_000.0})
    path = plotting.render_portfolio(pf, cfg, str(tmp_path / "c.png"))
    img = Image.open(path)
    if img.mode == "RGBA":
        assert img.getpixel((2, 2))[3] == 255, "chart background is transparent"
