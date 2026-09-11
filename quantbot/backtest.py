"""Walk-forward backtest engine.

Timing convention, which is where backtests most often cheat:

    features[t]  -> probability[t]  -> target weight[t]
    weight[t] earns the return from bar t to bar t+1

So a decision made from bar t's close is only ever paid the return that comes
*after* bar t. Nothing in the loop can see a price it would not have had.

Costs are charged on the traded notional whenever weights change, and are
deliberately pessimistic. Halving the cost assumption is the easiest way to
turn a losing strategy into a winning-looking one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbot.config import Config
from quantbot.data.base import bars_per_year
from quantbot.features import build_features, realised_vol
from quantbot.labels import binary_target, triple_barrier
from quantbot.metrics import Metrics, compute as compute_metrics
from quantbot.model import WalkForwardModel
from quantbot.risk import (
    DrawdownGuard, apply_portfolio_limits, covariance_path, edge_to_weight,
    portfolio_vol_scalar, smooth_scalar, volatility_scalar,
)
from quantbot.validation import walk_forward_splits

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    probabilities: pd.DataFrame
    costs: pd.Series
    metrics: Metrics
    benchmark_equity: pd.Series
    benchmark_metrics: Metrics
    halted_bars: int = 0
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        from quantbot.metrics import format_report
        lines = [format_report(self.metrics, "STRATEGY (out-of-sample)"),
                 "", format_report(self.benchmark_metrics, "BENCHMARK (buy & hold)")]
        edge = self.metrics.sharpe - self.benchmark_metrics.sharpe
        lines += ["", f"  Sharpe vs benchmark: {edge:+.2f}",
                  f"  Bars halted by drawdown guard: {self.halted_bars:,}"]
        return "\n".join(lines)


def _oos_probabilities(X: pd.DataFrame, y: pd.Series, t_end: np.ndarray,
                       cfg: Config) -> tuple[pd.Series, int]:
    """Produce strictly out-of-sample probabilities via walk-forward refitting.

    Every value returned was predicted by a model that had never seen that bar,
    nor any bar whose label window overlapped it.
    """
    n = len(X)
    probs = pd.Series(np.nan, index=X.index, dtype="float64")
    splits = walk_forward_splits(
        n=n,
        train_bars=cfg.model.train_bars,
        test_bars=cfg.model.retrain_every,
        t_end=t_end,
        embargo=cfg.model.embargo_bars,
        min_train_bars=cfg.model.min_train_bars,
    )
    if not splits:
        log.warning("not enough history for a single walk-forward split "
                    "(have %d bars, need > %d)", n, cfg.model.min_train_bars)
        return probs, 0

    for split in splits:
        model = WalkForwardModel(cfg.model, seed=cfg.seed)
        model.fit(X.iloc[split.train_idx], y.iloc[split.train_idx])
        if model.fitted:
            probs.iloc[split.test_idx] = model.predict_proba(X.iloc[split.test_idx])
    return probs, len(splits)


def run(cfg: Config, frames: dict[str, pd.DataFrame],
        n_trials: int = 1) -> BacktestResult:
    """Run the full walk-forward backtest over an aligned universe."""
    cfg.validate()
    bpy = bars_per_year(cfg.data.timeframe)
    symbols = list(frames)
    market = frames.get(cfg.data.benchmark)

    prob_cols: dict[str, pd.Series] = {}
    vol_cols: dict[str, pd.Series] = {}
    ret_cols: dict[str, pd.Series] = {}
    diagnostics: dict = {}

    for sym in symbols:
        df = frames[sym]
        mkt = None if sym == cfg.data.benchmark else market
        X = build_features(df, cfg.features, bpy, market=mkt)
        labels = triple_barrier(df["close"], df["high"], df["low"], cfg.labels)
        y = binary_target(labels)

        # Drop the indicator warm-up, where most features are NaN.
        valid = X.notna().mean(axis=1) > 0.75
        X, y = X[valid], y[valid]
        t_end = labels.loc[valid, "t_end"].to_numpy()
        # t_end is a position in the *original* frame; re-map onto the filtered one.
        pos = pd.Series(np.arange(len(X)), index=X.index)
        orig_pos = pd.Series(np.arange(len(df)), index=df.index)
        mapped = np.searchsorted(orig_pos[valid].to_numpy(), t_end, side="left")
        t_end_local = np.clip(mapped, 0, len(X) - 1)

        probs, n_splits = _oos_probabilities(X, y, t_end_local, cfg)
        prob_cols[sym] = probs.reindex(df.index)
        vol_cols[sym] = realised_vol(df["close"], cfg.labels.vol_window, bpy)
        ret_cols[sym] = df["close"].pct_change().shift(-1)  # return earned *after* bar t
        diagnostics[sym] = {"splits": n_splits, "bars": len(X),
                            "predicted": int(probs.notna().sum())}
        log.info("%s: %d walk-forward windows, %d OOS predictions",
                 sym, n_splits, int(probs.notna().sum()))

    probabilities = pd.DataFrame(prob_cols)
    vols = pd.DataFrame(vol_cols)
    fwd_returns = pd.DataFrame(ret_cols)

    # Only simulate bars where at least one symbol has a real prediction.
    live = probabilities.notna().any(axis=1)
    index = probabilities.index[live]
    if len(index) == 0:
        raise RuntimeError(
            "no out-of-sample predictions were produced. The history is shorter "
            "than model.train_bars + model.min_train_bars; fetch more bars or "
            "lower those settings."
        )

    probabilities = probabilities.loc[index]
    vols = vols.loc[index]
    fwd_returns = fwd_returns.loc[index]

    # --- Target weights, built in three stages.
    # 1. Conviction: calibrated edge -> fractional-Kelly directional weight.
    raw = pd.DataFrame(
        {s: edge_to_weight(probabilities[s].fillna(0.5).to_numpy(), cfg.risk)
         for s in symbols}, index=index)
    # 2. Risk parity: tilt toward the calmer symbols.
    tilt = pd.DataFrame(
        {s: volatility_scalar(vols[s].to_numpy(), cfg.risk) for s in symbols},
        index=index)
    tilted = raw * tilt
    # 3. Portfolio volatility target, using the full covariance matrix so that
    #    correlation between symbols is priced in rather than ignored.
    bar_returns = pd.DataFrame(
        {s: frames[s]["close"].pct_change() for s in symbols}
    ).reindex(probabilities.index.union(index)).sort_index()
    cov_full = covariance_path(bar_returns.fillna(0.0), cfg.risk.cov_span, bpy)
    rows = bar_returns.index.get_indexer(index)
    cov = cov_full[rows]
    pvs = smooth_scalar(
        portfolio_vol_scalar(tilted.to_numpy(), cov, cfg.risk),
        cfg.risk.scalar_smoothing,
    )
    targets = apply_portfolio_limits(tilted.mul(pvs, axis=0), cfg.risk)

    # --- Sequential simulation: the drawdown guard depends on equity, which
    # --- depends on past weights, so this cannot be vectorised.
    guard = DrawdownGuard(cfg.risk)
    equity = cfg.initial_capital
    cost_rate = (cfg.costs.taker_fee_bps + cfg.costs.half_spread_bps) / 10_000.0
    typical_vol = float(np.nanmedian(vols.to_numpy())) or 0.5

    eq_curve, ret_curve, cost_curve = [], [], []
    applied = pd.DataFrame(0.0, index=index, columns=symbols)
    prev_w = pd.Series(0.0, index=symbols)
    halted_bars = 0

    for t, ts in enumerate(index):
        allowed = guard.update(equity)
        if not allowed:
            halted_bars += 1
        target = targets.loc[ts] if allowed else pd.Series(0.0, index=symbols)

        delta = target - prev_w
        # No-trade band. Rebalancing costs real money every time, so a move is
        # only worth making if it is material in absolute terms *and* relative
        # to the position already held. Without this the book churns on noise.
        band = np.maximum(
            cfg.costs.min_trade_weight,
            cfg.costs.rebalance_band * prev_w.abs().to_numpy(),
        )
        small = delta.abs() < pd.Series(band, index=symbols)
        # Always allow a move that fully exits a position.
        small &= ~((target.abs() < 1e-9) & (prev_w.abs() > 1e-9))
        new_w = target.copy()
        new_w[small] = prev_w[small]
        traded = (new_w - prev_w).abs()

        # Slippage grows with volatility -- the market is thinner when it moves.
        bar_vol = vols.loc[ts].fillna(typical_vol)
        slip = cfg.costs.slippage_coef * (bar_vol / max(typical_vol, 1e-9)) / 10_000.0 * 10.0
        cost = float((traded * (cost_rate + slip)).sum())

        gross_ret = float((new_w * fwd_returns.loc[ts].fillna(0.0)).sum())
        net_ret = gross_ret - cost

        equity *= (1.0 + net_ret)
        eq_curve.append(equity)
        ret_curve.append(net_ret)
        cost_curve.append(cost)
        applied.loc[ts] = new_w
        prev_w = new_w

    equity_s = pd.Series(eq_curve, index=index, name="equity")
    returns_s = pd.Series(ret_curve, index=index, name="returns")
    costs_s = pd.Series(cost_curve, index=index, name="costs")

    # --- Benchmark: buy and hold the benchmark symbol over the same window.
    bench_sym = cfg.data.benchmark if cfg.data.benchmark in frames else symbols[0]
    bench_ret = frames[bench_sym]["close"].pct_change().reindex(index).fillna(0.0)
    bench_eq = (1.0 + bench_ret).cumprod() * cfg.initial_capital

    return BacktestResult(
        equity=equity_s,
        returns=returns_s,
        weights=applied,
        probabilities=probabilities,
        costs=costs_s,
        metrics=compute_metrics(returns_s, equity_s, bpy, applied, n_trials=n_trials),
        benchmark_equity=bench_eq,
        benchmark_metrics=compute_metrics(bench_ret, bench_eq, bpy, n_trials=1),
        halted_bars=halted_bars,
        diagnostics=diagnostics,
    )
