"""Round-trip trade statistics tests.

The distinction these pin down matters: a bar-level hit rate is not the success
rate of a trade, and win rate without payoff is not a verdict on a strategy.
"""

import numpy as np
import pandas as pd
import pytest

from quantbot.trades import extract, format_stats, summarise


def _frames(weights, prices):
    idx = pd.date_range("2026-01-01", periods=len(weights), freq="D", tz="UTC")
    return (pd.DataFrame({"X": weights}, index=idx),
            pd.DataFrame({"X": prices}, index=idx))


def test_a_flat_to_position_to_flat_cycle_is_one_trade():
    w, p = _frames([0, 0, 0.5, 0.5, 0.5, 0, 0], [100] * 7)
    trades = extract(w, p, cost_per_turn=0.0)
    assert len(trades) == 1
    assert trades[0].bars_held == 3


def test_scaling_a_position_does_not_open_a_second_trade():
    """Adding to a winner is the same trade, not a new one."""
    w, p = _frames([0, 0.2, 0.4, 0.6, 0.3, 0, 0], [100] * 7)
    assert len(extract(w, p, cost_per_turn=0.0)) == 1


def test_two_separate_cycles_are_two_trades():
    w, p = _frames([0, 0.5, 0, 0, 0.5, 0.5, 0], [100] * 7)
    assert len(extract(w, p, cost_per_turn=0.0)) == 2


def test_a_position_still_open_at_the_end_is_counted():
    """Dropping open trades hides current losers and flatters the result."""
    w, p = _frames([0, 0.5, 0.5, 0.5], [100, 100, 100, 80])
    trades = extract(w, p, cost_per_turn=0.0)
    assert len(trades) == 1
    assert trades[0].net_contribution < 0, "an open loser must still be counted"


def test_bar_hit_rate_and_trade_win_rate_are_not_the_same_number():
    """A trade that drifts up for many bars then collapses has a high bar hit
    rate and loses money. Reporting only the former is misleading."""
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 90]
    w, p = _frames([0.5] * 8 + [0.0], prices)
    trades = extract(w, p, cost_per_turn=0.0)
    assert len(trades) == 1
    assert trades[0].net_contribution < 0, "the trade lost"

    bar_rets = pd.Series(prices).pct_change().dropna()
    bar_hit = (bar_rets > 0).mean()
    assert bar_hit > 0.8, "bar-level hit rate looks excellent"
    stats = summarise(trades)
    assert stats.win_rate == 0.0, "trade-level win rate says it lost"


def test_low_win_rate_with_a_big_payoff_is_profitable():
    """Trend following is characteristically low win rate. Judging it on win
    rate alone would reject the property that makes it work."""
    from quantbot.trades import Trade
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    trades = [Trade("X", ts, ts, 1, 1, 0.3, 10, 0.0, +0.10, 0.0) for _ in range(3)]
    trades += [Trade("X", ts, ts, 1, 1, 0.3, 10, 0.0, -0.01, 0.0) for _ in range(7)]
    s = summarise(trades)
    assert s.win_rate == pytest.approx(0.3)
    assert s.payoff_ratio == pytest.approx(10.0)
    assert s.expectancy > 0, "30% win rate at 10:1 payoff makes money"
    assert s.profit_factor > 1
    assert "Positive expectancy" in format_stats(s)


def test_high_win_rate_with_a_bad_payoff_loses():
    from quantbot.trades import Trade
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    trades = [Trade("X", ts, ts, 1, 1, 0.3, 5, 0.0, +0.01, 0.0) for _ in range(7)]
    trades += [Trade("X", ts, ts, 1, 1, 0.3, 5, 0.0, -0.05, 0.0) for _ in range(3)]
    s = summarise(trades)
    assert s.win_rate == pytest.approx(0.7)
    assert s.expectancy < 0, "70% win rate can still lose money"
    assert "NEGATIVE expectancy" in format_stats(s)


def test_costs_are_charged_on_both_legs():
    w, p = _frames([0, 0.5, 0.5, 0], [100] * 4)
    free = extract(w, p, cost_per_turn=0.0)[0]
    dear = extract(w, p, cost_per_turn=0.01)[0]
    assert dear.net_contribution < free.net_contribution
    assert dear.cost == pytest.approx(2 * 0.5 * 0.01)


def test_win_rate_carries_a_confidence_interval():
    from quantbot.trades import Trade
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    few = [Trade("X", ts, ts, 1, 1, 0.3, 5, 0.0, 0.01, 0.0) for _ in range(3)]
    s = summarise(few)
    assert s.ci_high - s.ci_low > 0.4, "3 trades cannot pin down a win rate"


def test_empty_input_is_safe():
    s = summarise([])
    assert s.n_trades == 0 and "no completed trades" in format_stats(s)
