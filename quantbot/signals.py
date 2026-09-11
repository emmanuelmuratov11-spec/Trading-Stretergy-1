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
    take_profits: dict[str, float] = field(default_factory=dict)
    # Market conditions per symbol at call time, carried into the journal so a
    # later post-mortem can ask which regimes the model is actually good in.
    context: dict[str, dict] = field(default_factory=dict)
    staleness_minutes: float = 0.0
    warnings: list[str] = field(default_factory=list)


def generate(cfg: Config, frames: dict[str, pd.DataFrame]) -> SignalSet:
    """Fit on all available history and predict the most recent bar."""
    cfg.validate()
    bpy = bars_per_year(cfg.data.timeframe, cfg.data.annual_days)
    symbols = list(frames)
    market = frames.get(cfg.data.benchmark)
    warnings: list[str] = []

    probs: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    vols: dict[str, float] = {}
    prices: dict[str, float] = {}
    context: dict[str, dict] = {}

    asof = min(df.index[-1] for df in frames.values())

    for sym in symbols:
        df = frames[sym]
        prices[sym] = float(df["close"].iloc[-1])

        # Volatility, stop distance and regime are needed by EVERY engine --
        # they size the position and price the stop. Computing them before the
        # engine branch keeps a stop from going missing just because the
        # strategy needs no fitting.
        labels = triple_barrier(df["close"], df["high"], df["low"], cfg.labels)
        sig = float(labels["sigma"].iloc[-1]) if np.isfinite(labels["sigma"].iloc[-1]) else 0.01
        sigmas[sym] = sig
        v = realised_vol(df["close"], cfg.labels.vol_window, bpy).iloc[-1]
        vols[sym] = float(v) if np.isfinite(v) else cfg.risk.vol_ceiling

        # Regime snapshot: how volatile, how trending, how far off the highs.
        close = df["close"]
        logret = np.log(close).diff()
        lb = min(168, max(len(close) // 4, 10))
        mom = float(np.log(close).diff(lb).iloc[-1])
        mvol = float(logret.ewm(span=lb, adjust=False, min_periods=5).std().iloc[-1] * np.sqrt(lb))
        roll_max = float(close.rolling(min(336, len(close)), min_periods=1).max().iloc[-1])
        context[sym] = {
            "vol": vols[sym],
            "trend_z": (mom / mvol) if np.isfinite(mvol) and mvol > 1e-12 else 0.0,
            "drawdown": float(close.iloc[-1] / roll_max - 1.0) if roll_max > 0 else 0.0,
        }

        if cfg.model.kind == "trend":
            # No fitting required: the rule is fixed, so the live path is the
            # backtest path with no room to diverge.
            tp = trend_probabilities(df, cfg, bpy).iloc[-1]
            probs[sym] = float(tp) if np.isfinite(tp) else 0.5
            continue

        mkt = None if sym == cfg.data.benchmark else market
        X = build_features(df, cfg.features, bpy, market=mkt)
        y = binary_target(labels)

        valid = X.notna().mean(axis=1) > 0.75
        Xv, yv = X[valid], y[valid]

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
    # The profit target is the triple barrier the model was actually trained
    # against, not a number invented for the alert -- so the reward side of the
    # ratio means the same thing the label meant.
    h = np.sqrt(cfg.labels.horizon_bars)
    take_profits = {
        s: float(prices[s] * np.exp(cfg.labels.upper_sigma * max(sigmas[s], 1e-4) * h))
        for s in symbols
    }

    staleness = (datetime.now(timezone.utc) - asof.to_pydatetime()).total_seconds() / 60.0
    if staleness > 180:
        warnings.append(
            f"market data is {staleness:.0f} minutes old -- the feed may be stale; "
            "treat these signals with suspicion"
        )

    return SignalSet(asof=asof, prices=prices, probabilities=probs, targets=targets,
                     sigmas=sigmas, stops=stops, take_profits=take_profits,
                     vols=vols, context=context,
                     staleness_minutes=staleness, warnings=warnings)


def format_alert(sig: SignalSet, orders: list[Order], portfolio: Portfolio,
                 cfg: Config, mode: str = "paper", halted: bool = False,
                 journal=None) -> Alert:
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
        lines.append("ACTION REQUIRED - place these yourself" if mode == "alert"
                     else "ORDERS EXECUTED (paper money)")
        unit = "shares" if cfg.data.asset_class == "equity" else "units"
        for o in orders:
            base = o.symbol.split("/")[0]
            stop = sig.stops.get(o.symbol, 0.0)
            tp = sig.take_profits.get(o.symbol, 0.0)
            p = sig.probabilities.get(o.symbol, 0.5)
            lines.append("")
            lines.append(f"  ---- {o.side} {base} ----")
            lines.append(f"    {unit.capitalize():12s} {o.qty:,.6g}")
            if o.side == "BUY":
                # A marketable limit: crosses the spread enough to fill, while
                # refusing a price far worse than the one the signal assumed.
                limit = o.price * 1.0015
                lines.append(f"    {'Entry':12s} ~{o.price:,.2f}   "
                             f"(limit {limit:,.2f} or better)")
                risk = (o.price - stop) * o.qty
                reward = (tp - o.price) * o.qty
                rr = reward / risk if risk > 0 else 0.0
                lines.append(f"    {'Stop loss':12s} {stop:,.2f}   "
                             f"({stop / o.price - 1:+.1%}, risk ${risk:,.0f})")
                lines.append(f"    {'Target':12s} {tp:,.2f}   "
                             f"({tp / o.price - 1:+.1%}, reward ${reward:,.0f})")
                lines.append(f"    {'Reward:risk':12s} {rr:.2f} : 1")
            else:
                limit = o.price * 0.9985
                lines.append(f"    {'Exit':12s} ~{o.price:,.2f}   "
                             f"(limit {limit:,.2f} or better)")
            pct = o.notional / equity if equity > 0 else 0.0
            lines.append(f"    {'Size':12s} ${o.notional:,.2f}  = {pct:.1%} of book")
            strength = ("strong" if abs(p - 0.5) > 0.10
                        else "moderate" if abs(p - 0.5) > 0.03 else "weak")
            lines.append(f"    {'Confidence':12s} p(up) {p:.3f}  ({strength})")
            ctx = sig.context.get(o.symbol, {})
            if ctx:
                tz = ctx.get("trend_z", 0.0)
                regime = "uptrend" if tz > 0.5 else "downtrend" if tz < -0.5 else "flat"
                lines.append(f"    {'Conditions':12s} {regime}, "
                             f"vol {ctx.get('vol', 0):.0%}, "
                             f"{ctx.get('drawdown', 0):+.1%} off highs")
            lines.append(f"    {'Reason':12s} {o.reason}")
        lines.append("")
        if cfg.data.asset_class == "equity":
            lines.append("  NOTE: stocks gap overnight and at weekends. A stop is")
            lines.append("  an instruction, not a guarantee - a gap can open past it.")
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

    if journal is not None:
        from quantbot.journal import format_scorecard
        lines.append("")
        lines.append(format_scorecard(journal))
        recent = [p for p in journal.resolved()][-6:]
        if recent:
            lines.append("")
            lines.append("LAST FEW CALLS, GRADED")
            for p in recent:
                mark = "HIT " if p.correct else "MISS"
                lines.append(f"  {mark} {p.symbol:9s} p={p.prob:.2f} "
                             f"{p.entry_price:,.2f} -> {p.exit_price:,.2f}  "
                             f"{p.realised_return:+.2%}")

        findings = journal.diagnosis()
        if findings:
            lines.append("")
            lines.append("WHY (conditions the evidence supports)")
            for f in findings:
                lines.append(f"  - {f}")

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
