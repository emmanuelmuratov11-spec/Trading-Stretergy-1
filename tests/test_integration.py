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


def test_rebalance_interval_cuts_turnover(cfg):
    """Turnover is a guaranteed cost against an uncertain edge, so the
    rebalance schedule must measurably reduce it."""
    frames = _frames(cfg)
    cfg.costs.rebalance_every = 1
    every_bar = backtest.run(cfg, frames)
    cfg.costs.rebalance_every = 12
    throttled = backtest.run(cfg, frames)
    assert throttled.metrics.turnover < every_bar.metrics.turnover
    assert throttled.costs.sum() < every_bar.costs.sum()


def test_risk_reducing_moves_are_never_delayed(cfg):
    """A schedule that postpones an exit would turn a cost control into a
    risk control failure."""
    frames = _frames(cfg)
    cfg.costs.rebalance_every = 50
    cfg.risk.max_drawdown_stop = 0.03
    crashed = {}
    for sym, df in frames.items():
        d = df.copy()
        factor = pd.Series(np.linspace(1.0, 0.3, len(d)), index=d.index)
        for col in ("open", "high", "low", "close"):
            d[col] = d[col] * factor
        crashed[sym] = d
    res = backtest.run(cfg, crashed)
    if res.halted_bars > 0:
        flat = res.weights.abs().sum(axis=1).iloc[-1]
        assert flat < 1e-9, "book was not flattened while the guard was halted"


def test_trend_engine_runs_end_to_end(cfg):
    cfg.model.kind = "trend"
    res = backtest.run(cfg, _frames(cfg))
    assert len(res.equity) > 0
    assert res.weights.abs().sum(axis=1).max() <= cfg.risk.max_gross_leverage + 1e-9


def test_blend_sits_between_its_components(cfg):
    """A blend that agrees with neither parent would mean the wiring is wrong."""
    frames = _frames(cfg)
    cfg.model.kind = "trend"
    trend = backtest.run(cfg, frames).probabilities.stack().mean()
    cfg.model.kind = "blend"
    blend = backtest.run(cfg, frames).probabilities.stack().mean()
    cfg.model.kind = "ml"
    ml = backtest.run(cfg, frames).probabilities.stack().mean()
    assert min(trend, ml) - 0.05 <= blend <= max(trend, ml) + 0.05


def test_unknown_engine_is_rejected(cfg):
    import pytest
    cfg.model.kind = "definitely-not-a-strategy"
    with pytest.raises(ValueError):
        cfg.validate()


def test_throttled_schedule_never_raises_turnover(cfg):
    """A weaker but robust claim than a fixed percentage: whichever no-trade
    band happens to be binding, scheduling cannot INCREASE turnover."""
    frames = _frames(cfg)
    cfg.costs.rebalance_every = 24
    throttled = backtest.run(cfg, frames)
    cfg.costs.rebalance_every = 1
    every_bar = backtest.run(cfg, frames)
    assert throttled.metrics.turnover <= every_bar.metrics.turnover + 1e-9


def test_summary_reports_actual_trade_statistics(cfg):
    """The printed report must carry the trade win rate, not only the bar-level
    hit rate. This regressed once silently: the stats were computed and stored
    but never rendered, so the run logs showed no trades section at all."""
    res = backtest.run(cfg, _frames(cfg))
    text = res.summary()
    assert "ACTUAL ROUND-TRIP TRADES" in text
    assert "WIN RATE" in text or "no completed trades" in text
    assert "Expectancy" in text or "no completed trades" in text


def test_trade_stats_are_populated_on_the_result(cfg):
    res = backtest.run(cfg, _frames(cfg))
    assert res.trades is not None
    if res.trades.n_trades > 0:
        assert 0.0 <= res.trades.win_rate <= 1.0
        assert res.trades.ci_low <= res.trades.win_rate <= res.trades.ci_high
