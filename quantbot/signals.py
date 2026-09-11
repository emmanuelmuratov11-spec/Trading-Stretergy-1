"""Live signal generation and alert formatting.

The backtest answers "would this have worked?". This module answers "what
should I do right now?", using exactly the same feature, model and risk code so
the two cannot drift apart. If live and backtest logic diverge, the backtest
stops being evidence about the live system -- so they share one implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantbot.config import Config
from quantbot.data.base import bars_per_year
from quantbot.features import build_features, realised_vol
from quantbot.labels import binary_target, triple_barrier
from quantbot.model import WalkForwardModel
from quantbot.notify.base import Alert
from quantbot.portfolio import Order, Portfolio
from quantbot.strategies import trend_probabilities
from quantbot.risk import (
    apply_portfolio_limits, covariance_path, edge_to_weight,
    portfolio_vol_scalar, stop_loss_price, volatility_scalar,
)

log = logging.getLogger(__name__)


@dataclass
class SignalSet:
    asof: pd.Timestamp
    prices: dict[str, float]
    probabilities: dict[str, float]
    targets: dict[str, float]
    sigmas: dict[str, float]
    stops: dict[str, float]
    vols: dict[str, float]
    staleness_minutes: float = 0.0
    warnings: list[str] = field(default_factory=list)


def generate(cfg: Config, frames: dict[str, pd.DataFrame]) -> SignalSet:
    """Fit on all available history and predict the most recent bar."""
    cfg.validate()
    bpy = bars_per_year(cfg.data.timeframe)
    symbols = list(frames)
    market = frames.get(cfg.data.benchmark)
    warnings: list[str] = []

    probs: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    vols: dict[str, float] = {}
    prices: dict[str, float] = {}

    asof = min(df.index[-1] for df in frames.values())

    for sym in symbols:
        df = frames[sym]
        prices[sym] = float(df["close"].iloc[-1])

        if cfg.model.kind == "trend":
            # No fitting required: the rule is fixed, so the live path is the
            # backtest path with no room to diverge.
            tp = trend_probabilities(df, cfg, bpy).iloc[-1]
            probs[sym] = float(tp) if np.isfinite(tp) else 0.5
            continue

        mkt = None if sym == cfg.data.benchmark else market
        X = build_features(df, cfg.features, bpy, market=mkt)
        labels = triple_barrier(df["close"], df["high"], df["low"], cfg.labels)
        y = binary_target(labels)

        valid = X.notna().mean(axis=1) > 0.75
        Xv, yv = X[valid], y[valid]

        sig = float(labels["sigma"].iloc[-1]) if np.isfinite(labels["sigma"].iloc[-1]) else 0.01
        sigmas[sym] = sig
        v = realised_vol(df["close"], cfg.labels.vol_window, bpy).iloc[-1]
        vols[sym] = float(v) if np.isfinite(v) else cfg.risk.vol_ceiling

        # The final `horizon_bars` rows have labels that cannot have resolved
        # yet, so they must not be trained on.
        trainable = Xv.iloc[:-cfg.labels.horizon_bars] if len(Xv) > cfg.labels.horizon_bars else Xv
        ytrain = yv.iloc[:len(trainable)]
        window = trainable.iloc[-cfg.model.train_bars:]
        ywindow = ytrain.iloc[-cfg.model.train_bars:]

        if len(window) < cfg.model.min_train_bars:
            warnings.append(
                f"{sym}: only {len(window)} trainable bars "
                f"(need {cfg.model.min_train_bars}); holding flat"
            )
            probs[sym] = 0.5
            continue

        model = WalkForwardModel(cfg.model, seed=cfg.seed).fit(window, ywindow)
        if not model.fitted:
            warnings.append(f"{sym}: model failed to fit; holding flat")
            probs[sym] = 0.5
            continue
        probs[sym] = float(model.predict_proba(Xv.iloc[[-1]])[0])

    # --- Same three-stage sizing as the backtest.
    raw = np.array([edge_to_weight(np.array([probs[s]]), cfg.risk)[0] for s in symbols])
    tilt = np.array([volatility_scalar(np.array([vols[s]]), cfg.risk)[0] for s in symbols])
    tilted = raw * tilt

    bar_returns = pd.DataFrame({s: frames[s]["close"].pct_change() for s in symbols}).fillna(0.0)
    cov = covariance_path(bar_returns, cfg.risk.cov_span, bpy)[-1]
    pvs = portfolio_vol_scalar(tilted.reshape(1, -1), cov.reshape(1, len(symbols), len(symbols)),
                               cfg.risk)[0]
    scaled = pd.DataFrame([tilted * pvs], columns=symbols)
    targets = apply_portfolio_limits(scaled, cfg.risk).iloc[0].to_dict()

    stops = {s: stop_loss_price(prices[s], sigmas[s], cfg.risk, long=True) for s in symbols}

    staleness = (datetime.now(timezone.utc) - asof.to_pydatetime()).total_seconds() / 60.0
    if staleness > 180:
        warnings.append(
            f"market data is {staleness:.0f} minutes old -- the feed may be stale; "
            "treat these signals with suspicion"
        )

    return SignalSet(asof=asof, prices=prices, probabilities=probs, targets=targets,
                     sigmas=sigmas, stops=stops, vols=vols,
                     staleness_minutes=staleness, warnings=warnings)


def format_alert(sig: SignalSet, orders: list[Order], portfolio: Portfolio,
                 cfg: Config, mode: str = "paper", halted: bool = False) -> Alert:
    """Render a signal set and its orders as a phone-readable message."""
    equity = portfolio.equity(sig.prices)
    dd = portfolio.drawdown(sig.prices)
    pnl = equity - cfg.initial_capital
    pnl_pct = pnl / cfg.initial_capital if cfg.initial_capital else 0.0

    lines: list[str] = []
    lines.append(f"As of {sig.asof:%Y-%m-%d %H:%M} UTC  ({cfg.data.timeframe} bars)")
    lines.append("")

    if halted:
        lines.append("!! DRAWDOWN GUARD TRIPPED -- standing down, all positions flat.")
        lines.append(f"   Drawdown {dd:.1%} breached the {cfg.risk.max_drawdown_stop:.0%} limit.")
        lines.append("")

    if orders:
        lines.append("ACTION REQUIRED" if mode == "alert" else "ORDERS EXECUTED (paper)")
        for o in orders:
            lines.append(f"  {o.side:4s} {o.qty:.6g} {o.symbol}  @ ~{o.price:,.2f}")
            lines.append(f"       ${o.notional:,.2f}   ({o.reason})")
            if o.side == "BUY":
                lines.append(f"       suggested stop: {sig.stops[o.symbol]:,.2f}")
        lines.append("")
    else:
        lines.append("No action -- current positions already match the target.")
        lines.append("")

    lines.append("SIGNALS")
    for sym in sorted(sig.probabilities):
        p = sig.probabilities[sym]
        conf = "strong" if abs(p - 0.5) > 0.10 else "weak" if abs(p - 0.5) > 0.03 else "none"
        arrow = "UP" if p > 0.5 else "DOWN"
        lines.append(f"  {sym:10s} p(up)={p:.3f} [{arrow} {conf}]  "
                     f"target={sig.targets[sym]:+.1%}  vol={sig.vols[sym]:.0%}")
    lines.append("")

    lines.append("PORTFOLIO")
    lines.append(f"  Equity      ${equity:,.2f}")
    lines.append(f"  P&L         ${pnl:+,.2f} ({pnl_pct:+.2%})")
    lines.append(f"  Cash        ${portfolio.cash:,.2f}")
    lines.append(f"  Drawdown    {dd:.2%}")
    lines.append(f"  Fees paid   ${portfolio.fees_paid:,.2f}  over {portfolio.trade_count} trades")
    held = {s: p for s, p in portfolio.positions.items() if abs(p.qty) > 1e-10}
    if held:
        for s, p in sorted(held.items()):
            px = sig.prices.get(s, p.avg_price)
            lines.append(f"    {s:10s} {p.qty:.6g} @ {p.avg_price:,.2f} "
                         f"-> {px:,.2f} ({p.unrealised(px):+,.2f})")

    if sig.warnings:
        lines.append("")
        lines.append("WARNINGS")
        for w in sig.warnings:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append("Paper trading. Not investment advice." if mode == "paper"
                 else "Signal only -- you place the trade. Not investment advice.")

    title = "Trading signal"
    if halted:
        title = "RISK HALT - positions flat"
    elif orders:
        title = f"{len(orders)} order(s) - " + ", ".join(
            f"{o.side} {o.symbol.split('/')[0]}" for o in orders[:3])

    return Alert(title=title, body="\n".join(lines),
                 urgent=bool(orders) or halted,
                 meta={"equity": equity, "orders": len(orders)})
