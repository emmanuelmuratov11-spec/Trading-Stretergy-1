"""End-to-end tests. Slow-ish, but they are the ones that catch wiring bugs."""

import numpy as np
import pandas as pd

from quantbot import backtest, signals
from quantbot.data import load_universe
from quantbot.data.loader import align
from quantbot.portfolio import Portfolio


def _frames(cfg):
    cfg.data.lookback_days = 120
    return align(load_universe(cfg.data, use_cache=False))


def test_backtest_runs_and_is_self_consistent(cfg):
    res = backtest.run(cfg, _frames(cfg))
    assert len(res.equity) > 0
    assert len(res.equity) == len(res.returns) == len(res.weights)
    # Equity must be reproducible from the return stream it reports.
    rebuilt = (1 + res.returns).cumprod() * cfg.initial_capital
    assert np.allclose(rebuilt.to_numpy(), res.equity.to_numpy(), rtol=1e-9)


def test_backtest_respects_leverage_limits(cfg):
    cfg.risk.max_gross_leverage = 0.5
    res = backtest.run(cfg, _frames(cfg))
    assert res.weights.abs().sum(axis=1).max() <= 0.5 + 1e-9


def test_no_shorts_when_disabled(cfg):
    cfg.risk.allow_shorts = False
    res = backtest.run(cfg, _frames(cfg))
    assert (res.weights >= -1e-12).all().all()


def test_higher_costs_never_improve_returns(cfg):
    """A sanity check on the cost model: charging more cannot earn more."""
    frames = _frames(cfg)
    cheap = backtest.run(cfg, frames)
    cfg.costs.taker_fee_bps = 60.0
    cfg.costs.half_spread_bps = 30.0
    dear = backtest.run(cfg, frames)
    assert dear.metrics.total_return <= cheap.metrics.total_return + 1e-12


def test_backtest_is_deterministic(cfg):
    frames = _frames(cfg)
    a = backtest.run(cfg, frames).equity.to_numpy()
    b = backtest.run(cfg, frames).equity.to_numpy()
    assert np.allclose(a, b), "same inputs must give the same result"


def test_live_signals_match_backtest_machinery(cfg):
    frames = _frames(cfg)
    sig = signals.generate(cfg, frames)
    assert set(sig.targets) == set(frames)
    for sym, p in sig.probabilities.items():
        assert 0.0 <= p <= 1.0
    assert sum(abs(w) for w in sig.targets.values()) <= cfg.risk.max_gross_leverage + 1e-9
    for sym, stop in sig.stops.items():
        assert stop < sig.prices[sym], "a long stop must sit below the entry price"


def test_alert_renders_without_crashing(cfg):
    frames = _frames(cfg)
    sig = signals.generate(cfg, frames)
    pf = Portfolio(cash=cfg.initial_capital, peak_equity=cfg.initial_capital)
    orders = pf.plan_rebalance(sig.targets, sig.prices)
    alert = signals.format_alert(sig, orders, pf, cfg, mode="paper")
    assert alert.title and "PORTFOLIO" in alert.body
    assert "Not investment advice" in alert.body


def test_drawdown_guard_flattens_in_a_crash(cfg):
    """Force a collapse and confirm the guard actually stops trading."""
    frames = _frames(cfg)
    crashed = {}
    for sym, df in frames.items():
        d = df.copy()
        n = len(d)
        factor = pd.Series(np.linspace(1.0, 0.25, n), index=d.index)
        for col in ("open", "high", "low", "close"):
            d[col] = d[col] * factor
        crashed[sym] = d
    cfg.risk.max_drawdown_stop = 0.05
    res = backtest.run(cfg, crashed)
    if res.metrics.max_drawdown <= -0.05:
        assert res.halted_bars > 0, "guard never tripped despite a large drawdown"
