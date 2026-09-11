import json

from quantbot.portfolio import Portfolio, Position

PRICES = {"BTC/USDT": 50_000.0, "ETH/USDT": 2_000.0}


def test_rebalance_produces_correct_notionals():
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    orders = pf.plan_rebalance({"BTC/USDT": 0.5, "ETH/USDT": 0.0}, PRICES)
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert abs(orders[0].notional - 5_000.0) < 1e-6


def test_small_moves_are_skipped_but_exits_are_not():
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    pf.apply(pf.plan_rebalance({"BTC/USDT": 0.50}, PRICES), fee_rate=0.0)

    tiny = pf.plan_rebalance({"BTC/USDT": 0.505}, PRICES, band=0.02)
    assert tiny == [], "a 0.5% drift should not trigger a trade"

    exit_orders = pf.plan_rebalance({"BTC/USDT": 0.0}, PRICES, band=0.02)
    assert len(exit_orders) == 1 and exit_orders[0].side == "SELL"


def test_realised_pnl_is_correct():
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    pf.apply(pf.plan_rebalance({"BTC/USDT": 0.5}, PRICES), fee_rate=0.0)
    higher = {"BTC/USDT": 60_000.0, "ETH/USDT": 2_000.0}
    pf.apply(pf.plan_rebalance({"BTC/USDT": 0.0}, higher), fee_rate=0.0)
    # 0.1 BTC bought at 50k, sold at 60k -> 1000
    assert abs(pf.realised_pnl - 1_000.0) < 1e-6
    assert pf.positions["BTC/USDT"].qty == 0.0


def test_fees_reduce_equity():
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    pf.apply(pf.plan_rebalance({"BTC/USDT": 1.0}, PRICES), fee_rate=0.001)
    assert pf.fees_paid > 0
    assert pf.equity(PRICES) < 10_000.0


def test_state_survives_a_round_trip(tmp_path):
    pf = Portfolio(cash=10_000.0, peak_equity=10_000.0)
    pf.apply(pf.plan_rebalance({"BTC/USDT": 0.4}, PRICES))
    pf.mark(PRICES)
    path = str(tmp_path / "state" / "portfolio.json")
    pf.save(path)
    back = Portfolio.load(path, 10_000.0)
    assert abs(back.equity(PRICES) - pf.equity(PRICES)) < 1e-9
    assert back.positions["BTC/USDT"].qty == pf.positions["BTC/USDT"].qty
    assert back.trade_count == pf.trade_count


def test_missing_state_file_starts_fresh(tmp_path):
    pf = Portfolio.load(str(tmp_path / "nope.json"), 5_000.0)
    assert pf.cash == 5_000.0 and pf.trade_count == 0


def test_history_does_not_grow_without_bound():
    pf = Portfolio(cash=100.0, peak_equity=100.0)
    pf.history = [{"ts": "x", "equity": 1.0, "cash": 1.0}] * 5200
    pf.mark({})
    assert len(pf.history) <= 5000
