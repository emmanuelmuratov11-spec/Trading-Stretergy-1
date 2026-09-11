import numpy as np
import pandas as pd

from quantbot.config import RiskConfig
from quantbot.risk import (
    DrawdownGuard, apply_portfolio_limits, covariance_path, edge_to_weight,
    portfolio_vol_scalar, smooth_scalar, volatility_scalar,
)


def test_no_edge_means_no_position():
    cfg = RiskConfig()
    assert edge_to_weight(np.array([0.5]), cfg)[0] == 0.0


def test_deadband_suppresses_noise_signals():
    cfg = RiskConfig(conviction_deadband=0.10)
    assert edge_to_weight(np.array([0.52]), cfg)[0] == 0.0   # edge 0.04 < 0.10
    assert edge_to_weight(np.array([0.60]), cfg)[0] > 0.0    # edge 0.20 > 0.10


def test_weights_never_exceed_the_per_symbol_cap():
    cfg = RiskConfig(max_position_weight=0.25, kelly_fraction=1.0)
    w = edge_to_weight(np.array([0.999, 0.001]), cfg)
    assert np.all(np.abs(w) <= 0.25 + 1e-12)


def test_shorts_are_blocked_unless_enabled():
    assert edge_to_weight(np.array([0.1]), RiskConfig(allow_shorts=False))[0] == 0.0
    assert edge_to_weight(np.array([0.1]), RiskConfig(allow_shorts=True))[0] < 0.0


def test_gross_leverage_cap_is_enforced():
    cfg = RiskConfig(max_gross_leverage=1.0)
    w = pd.DataFrame({"a": [0.6], "b": [0.6], "c": [0.6]})
    out = apply_portfolio_limits(w, cfg)
    assert out.abs().sum(axis=1).iloc[0] <= 1.0 + 1e-9


def test_higher_volatility_means_smaller_positions():
    cfg = RiskConfig()
    calm = volatility_scalar(np.array([0.2]), cfg)[0]
    wild = volatility_scalar(np.array([1.5]), cfg)[0]
    assert calm > wild


def test_portfolio_vol_scalar_hits_the_target():
    cfg = RiskConfig(target_annual_vol=0.20, max_vol_scale=100.0)
    cov = np.array([[[0.04, 0.0], [0.0, 0.04]]])   # 20% vol each, uncorrelated
    w = np.array([[1.0, 0.0]])
    scale = portfolio_vol_scalar(w, cov, cfg)[0]
    achieved = np.sqrt((w * scale) @ cov[0] @ (w * scale).T)[0][0]
    assert np.isclose(achieved, 0.20, atol=1e-9)


def test_correlation_is_priced_in():
    """Two correlated positions are one bet, and must be sized down as such."""
    cfg = RiskConfig(target_annual_vol=0.20, max_vol_scale=100.0)
    w = np.array([[0.5, 0.5]])
    uncorr = np.array([[[0.04, 0.00], [0.00, 0.04]]])
    corr = np.array([[[0.04, 0.039], [0.039, 0.04]]])
    assert portfolio_vol_scalar(w, corr, cfg)[0] < portfolio_vol_scalar(w, uncorr, cfg)[0]


def test_zero_weights_do_not_divide_by_zero():
    cfg = RiskConfig()
    cov = np.array([[[0.04, 0.0], [0.0, 0.04]]])
    with np.errstate(divide="raise", invalid="raise"):
        out = portfolio_vol_scalar(np.array([[0.0, 0.0]]), cov, cfg)
    assert out[0] == 0.0


def test_drawdown_guard_halts_then_recovers():
    cfg = RiskConfig(max_drawdown_stop=0.20, drawdown_recovery=0.10)
    g = DrawdownGuard(cfg)
    assert g.update(100.0)
    assert g.update(85.0)            # -15%, still trading
    assert not g.update(79.0)        # -21%, halt
    assert not g.update(88.0)        # -12%, still halted
    assert g.update(92.0)            # -8%, back on


def test_smoothing_reduces_turnover():
    raw = np.tile([1.0, 2.0], 100)   # maximally jittery
    smoothed = smooth_scalar(raw, span=20)
    assert np.abs(np.diff(smoothed)).sum() < np.abs(np.diff(raw)).sum()


def test_covariance_path_is_causal():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (500, 2)), columns=["a", "b"])
    full = covariance_path(r, 100, 8760)
    trunc = covariance_path(r.iloc[:300], 100, 8760)
    assert np.allclose(full[299], trunc[-1], rtol=1e-9), "covariance saw the future"
