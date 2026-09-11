import numpy as np
import pandas as pd

from quantbot import metrics


def _series(rets):
    idx = pd.date_range("2024-01-01", periods=len(rets), freq="h", tz="UTC")
    r = pd.Series(rets, index=idx)
    return r, (1 + r).cumprod() * 10_000


def test_sharpe_of_a_constant_series_is_zero():
    r, eq = _series([0.001] * 500)
    assert metrics.sharpe_ratio(r, 8760) == 0.0   # zero variance -> undefined, not infinite


def test_max_drawdown_is_measured_from_the_peak():
    eq = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert np.isclose(metrics.max_drawdown(eq), -0.5)


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    r, eq = _series(rng.normal(0.0005, 0.01, 3000))
    one = metrics.compute(r, eq, 8760, n_trials=1).dsr
    many = metrics.compute(r, eq, 8760, n_trials=500).dsr
    assert many < one, "searching more configurations must lower confidence"


def test_norm_inv_matches_known_quantiles():
    assert abs(metrics.norm_inv(0.975) - 1.959964) < 1e-4
    assert abs(metrics.norm_inv(0.5)) < 1e-9


def test_empty_input_does_not_explode():
    m = metrics.compute(pd.Series(dtype=float), pd.Series(dtype=float), 8760)
    assert m.sharpe == 0.0 and m.n_bars == 0


def test_turnover_is_reported_from_weights():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    r = pd.Series([0.0, 0.01, -0.01, 0.0], index=idx)
    eq = (1 + r).cumprod() * 100
    w = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0]}, index=idx)
    assert metrics.compute(r, eq, 8760, weights=w).turnover > 0


def _m(total_return, max_dd, sharpe):
    from quantbot.metrics import zero_metrics
    m = zero_metrics()
    return type(m)(**{**m.to_dict(), "total_return": total_return,
                      "max_drawdown": max_dd, "sharpe": sharpe})


def test_losing_less_is_not_reported_as_losing_to_the_benchmark():
    """Sharpe ordering inverts when both streams are negative: dividing a
    negative mean by a smaller std makes it MORE negative. Ranking on Sharpe
    alone would advise holding an asset that lost far more."""
    strategy = _m(-0.04, -0.07, -0.55)     # lost 4%, shallow drawdown
    benchmark = _m(-0.27, -0.59, -0.29)    # lost 27%, brutal drawdown
    better, text = metrics.compare_to_benchmark(strategy, benchmark)
    assert better, "preserving capital must not be scored as a loss to the benchmark"
    assert "both lost money" in text, "it must still say that both lost money"


def test_positive_returns_are_ranked_on_sharpe():
    better, _ = metrics.compare_to_benchmark(_m(0.5, -0.2, 1.8), _m(0.6, -0.5, 1.1))
    assert better, "higher risk-adjusted return should win when both are positive"


def test_making_money_beats_a_losing_benchmark():
    better, text = metrics.compare_to_benchmark(_m(0.10, -0.05, 0.9), _m(-0.30, -0.6, -0.4))
    assert better and "while buy & hold lost" in text


def test_losing_money_never_beats_a_winning_benchmark():
    better, text = metrics.compare_to_benchmark(_m(-0.10, -0.2, -0.5), _m(0.40, -0.3, 1.2))
    assert not better and "Lost money" in text
