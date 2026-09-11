"""Position sizing and capital preservation.

Prediction accuracy is the glamorous part and the smaller part. What actually
separates an account that survives from one that does not is sizing: how much
you bet on each signal, how fast you cut losers, and when you stop trading
altogether. This module is deliberately conservative at every fork.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.config import RiskConfig

EPS = 1e-12


def edge_to_weight(prob: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    """Convert calibrated probabilities into a raw directional weight in [-1, 1].

    Uses a *fraction* of the Kelly criterion. Full Kelly maximises long-run
    growth only if your probabilities are exactly right; since they are not,
    full Kelly is a reliable route to ruin. A third of Kelly gives up little
    growth for a large reduction in drawdown.
    """
    p = np.clip(np.asarray(prob, dtype="float64"), 0.001, 0.999)
    edge = 2.0 * p - 1.0                      # in [-1, 1]

    # Ignore signals too close to a coin flip -- they are noise and their
    # transaction costs are real.
    edge = np.where(np.abs(edge) < cfg.conviction_deadband, 0.0, edge)

    w = edge * cfg.kelly_fraction
    if not cfg.allow_shorts:
        w = np.clip(w, 0.0, None)
    return np.clip(w, -cfg.max_position_weight, cfg.max_position_weight)


def volatility_scalar(realised_vol: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    """Per-symbol inverse-volatility tilt (risk parity across the book).

    Equal *notional* across symbols is not equal *risk*: a 50% weight in a
    120%-vol altcoin carries several times the risk of a 50% weight in BTC.
    This tilts allocation toward the calmer symbols. It sets relative sizing
    only -- absolute portfolio risk is set by `portfolio_vol_scalar`.
    """
    vol = np.asarray(realised_vol, dtype="float64")
    vol = np.where(np.isfinite(vol) & (vol > EPS), vol, cfg.vol_ceiling)
    vol = np.clip(vol, cfg.vol_floor, cfg.vol_ceiling)
    ref = max(cfg.target_annual_vol, EPS)
    return np.clip(ref / vol, 0.0, 1.0)


def covariance_path(returns: pd.DataFrame, span: int, bars_per_year: float) -> np.ndarray:
    """EWMA covariance matrix per bar, annualised. Shape (T, n, n).

    Causal by construction: `ewm` only ever looks backwards.
    """
    n = returns.shape[1]
    cov = returns.ewm(span=span, min_periods=max(10, span // 4)).cov()
    arr = cov.to_numpy(dtype="float64").reshape(len(returns), n, n) * bars_per_year
    # Warm-up bars are NaN; fall back to a diagonal of the target vol so that
    # early bars size conservatively instead of crashing.
    bad = ~np.isfinite(arr)
    if bad.any():
        arr[bad] = 0.0
        flat = ~np.isfinite(arr).any(axis=(1, 2))
        for t in range(len(arr)):
            if not flat[t] and np.allclose(arr[t], 0.0):
                arr[t] = np.eye(n)
    return arr


def portfolio_vol_scalar(weights: np.ndarray, cov: np.ndarray,
                         cfg: RiskConfig) -> np.ndarray:
    """Scale the whole book so its *ex-ante* volatility matches the target.

    This is the single highest-value risk control in the file. Without it a
    strategy quietly takes far more risk in turbulent markets, and correlated
    positions across symbols compound that: two 40% weights in assets
    correlated at 0.9 is not diversification, it is one 80% bet. Using the full
    covariance matrix rather than per-symbol vols is what catches that.
    """
    w = np.asarray(weights, dtype="float64")
    # var[t] = w[t] . Sigma[t] . w[t]
    var = np.einsum("ti,tij,tj->t", w, cov, w)
    vol = np.sqrt(np.clip(var, 0.0, None))
    # np.where evaluates both branches, so guard the denominator itself rather
    # than relying on the condition to suppress the division.
    safe_vol = np.where(vol > EPS, vol, 1.0)
    scale = np.where(vol > EPS, cfg.target_annual_vol / safe_vol, 0.0)
    return np.clip(scale, 0.0, cfg.max_vol_scale)


def smooth_scalar(scale: np.ndarray, span: int) -> np.ndarray:
    """Damp bar-to-bar jitter in the sizing scalar.

    An unsmoothed volatility scalar moves a little every single bar, and since
    every move is a trade, the book churns continuously. Turnover is pure cost:
    smoothing the scalar cuts that drag dramatically while barely changing the
    risk profile, because volatility itself is slow-moving.
    """
    if span <= 1 or len(scale) == 0:
        return scale
    return pd.Series(scale).ewm(span=span, adjust=False).mean().to_numpy()


def apply_portfolio_limits(weights: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    """Cap per-symbol weight, then scale the whole book down to the gross limit."""
    w = weights.clip(-cfg.max_position_weight, cfg.max_position_weight)
    gross = w.abs().sum(axis=1)
    over = gross > cfg.max_gross_leverage
    if bool(over.any()):
        scale = pd.Series(1.0, index=w.index)
        scale[over] = cfg.max_gross_leverage / gross[over]
        w = w.mul(scale, axis=0)
    return w


class DrawdownGuard:
    """A kill switch: flatten everything after a severe drawdown.

    Models fail. Regimes break. This is the backstop that turns a catastrophic
    loss into a merely bad one. Once tripped it stays flat until equity has
    recovered a good way back toward the high-water mark, so it does not
    whipsaw in and out at the bottom.
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.peak = -np.inf
        self.halted = False

    def update(self, equity: float) -> bool:
        """Record new equity; return True if trading is permitted."""
        self.peak = max(self.peak, equity)
        dd = equity / self.peak - 1.0 if self.peak > 0 else 0.0

        if self.halted:
            if dd > -self.cfg.drawdown_recovery:
                self.halted = False
        elif dd <= -self.cfg.max_drawdown_stop:
            self.halted = True
        return not self.halted

    @property
    def state(self) -> str:
        return "HALTED" if self.halted else "ACTIVE"


def apply_rebalance_schedule(target: pd.Series, prev_w: pd.Series,
                             bar_index: int, every: int,
                             trading_allowed: bool) -> pd.Series:
    """Hold positions between scheduled rebalances, per symbol.

    Trading costs are paid every time a weight moves, and the model only
    refits weekly, so re-trading each bar pays a spread to chase a signal that
    has barely moved. Between scheduled bars each position is held.

    Two exemptions, because a cost control must never become a risk control
    failure: a move that REDUCES a position happens immediately, and when
    trading is disallowed (the drawdown guard has fired) the flattening is not
    delayed either.

    The reduction test is per symbol. Asking whether *any* symbol is reducing
    lets one noisy name drag the whole book into a rebalance, which fires on
    nearly every bar once a few symbols are held and cancels the schedule
    outright.
    """
    if every <= 1 or not trading_allowed or (bar_index % every) == 0:
        return target
    reducing = target.abs() < prev_w.abs() - 1e-12
    return target.where(reducing, prev_w)


def stop_loss_price(entry: float, sigma: float, cfg: RiskConfig, long: bool = True) -> float:
    """Hard stop, placed at a volatility-scaled distance from entry."""
    move = cfg.stop_loss_sigma * max(sigma, 1e-4)
    return entry * float(np.exp(-move)) if long else entry * float(np.exp(move))
