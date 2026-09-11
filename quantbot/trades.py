"""Round-trip trade reconstruction and win-rate statistics.

The `hit_rate` reported by `metrics.py` is the fraction of *bars* with a
positive return. That is a useful number but it is NOT the success rate of the
trades you would actually place, and confusing the two badly misleads:

* One trade held for 30 bars contributes 30 bar observations, so a single big
  winner can push the bar hit rate around while the trade count stays at one.
* A trade that drifts up for 25 bars and collapses on the last 5 has a bar hit
  rate of 83% and loses money.

This module reconstructs the actual trades - open when a position appears,
close when it goes flat - and reports the statistics a trader cares about.

**Win rate alone is close to meaningless.** A 40% win rate with a 3:1 payoff
prints money; a 65% win rate with a 1:3 payoff bleeds. Trend following is
characteristically low win rate with large winners, and judging it on win rate
alone would reject the strategy for the exact property that makes it work. So
expectancy per trade is the headline here, and the win rate is context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    avg_weight: float
    bars_held: int
    gross_return: float       # price move over the holding period
    net_contribution: float   # weight-scaled, after estimated costs
    cost: float

    @property
    def won(self) -> bool:
        return self.net_contribution > 0


@dataclass
class TradeStats:
    n_trades: int = 0
    win_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 1.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0     # avg win / avg loss
    expectancy: float = 0.0       # average net contribution per trade
    profit_factor: float = 0.0    # gross wins / gross losses
    avg_bars_held: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    total: float = 0.0
    by_symbol: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("by_symbol", None)
        return d


def extract(weights: pd.DataFrame, prices: pd.DataFrame,
            cost_per_turn: float = 0.0011) -> list[Trade]:
    """Rebuild round-trip trades from a weight path.

    A trade runs from the bar a position first becomes non-zero to the bar it
    returns to zero. Scaling up or down mid-trade does not open a new trade; it
    changes the average weight, which is what the P&L is scaled by.
    """
    trades: list[Trade] = []
    for sym in weights.columns:
        if sym not in prices.columns:
            continue
        w = weights[sym].fillna(0.0).to_numpy()
        px = prices[sym].reindex(weights.index).ffill().to_numpy()
        idx = weights.index

        open_at = None
        acc_w: list[float] = []
        for t in range(len(w)):
            live = abs(w[t]) > EPS
            if live and open_at is None:
                open_at, acc_w = t, [w[t]]
            elif live:
                acc_w.append(w[t])
            elif open_at is not None:
                trades.append(_close(sym, idx, px, open_at, t, acc_w, cost_per_turn))
                open_at, acc_w = None, []
        if open_at is not None:
            # Still open at the end of the sample: mark it out at the last bar
            # rather than dropping it, since ignoring open losers flatters the
            # result.
            trades.append(_close(sym, idx, px, open_at, len(w) - 1, acc_w,
                                 cost_per_turn))
    return trades


def _close(sym, idx, px, start, end, acc_w, cost_per_turn) -> Trade:
    entry_px, exit_px = float(px[start]), float(px[end])
    gross = exit_px / entry_px - 1.0 if entry_px > 0 else 0.0
    avg_w = float(np.mean(acc_w)) if acc_w else 0.0
    # Round trip: in and out, each paying the per-turn cost.
    cost = 2.0 * abs(avg_w) * cost_per_turn
    return Trade(
        symbol=sym, entry_time=idx[start], exit_time=idx[end],
        entry_price=entry_px, exit_price=exit_px, avg_weight=avg_w,
        bars_held=int(end - start), gross_return=gross,
        net_contribution=gross * avg_w - cost, cost=cost,
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    import math
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarise(trades: list[Trade]) -> TradeStats:
    if not trades:
        return TradeStats()
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    n = len(trades)
    lo, hi = wilson(len(wins), n)

    avg_win = float(np.mean([t.net_contribution for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.net_contribution for t in losses])) if losses else 0.0
    gross_win = sum(t.net_contribution for t in wins)
    gross_loss = -sum(t.net_contribution for t in losses)

    by_symbol = {}
    for sym in {t.symbol for t in trades}:
        g = [t for t in trades if t.symbol == sym]
        gw = sum(1 for t in g if t.won)
        slo, shi = wilson(gw, len(g))
        by_symbol[sym] = {
            "n": len(g), "win_rate": gw / len(g),
            "ci_low": slo, "ci_high": shi,
            "total": sum(t.net_contribution for t in g),
        }

    return TradeStats(
        n_trades=n,
        win_rate=len(wins) / n,
        ci_low=lo, ci_high=hi,
        avg_win=avg_win, avg_loss=avg_loss,
        payoff_ratio=abs(avg_win / avg_loss) if avg_loss < -EPS else 0.0,
        expectancy=float(np.mean([t.net_contribution for t in trades])),
        profit_factor=(gross_win / gross_loss) if gross_loss > EPS else 0.0,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])),
        best=max(t.net_contribution for t in trades),
        worst=min(t.net_contribution for t in trades),
        total=sum(t.net_contribution for t in trades),
        by_symbol=by_symbol,
    )


def format_stats(s: TradeStats, title: str = "TRADES") -> str:
    if s.n_trades == 0:
        return f"--- {title} ---\n  no completed trades"
    breakeven = 1.0 / (1.0 + s.payoff_ratio) if s.payoff_ratio > 0 else float("nan")
    lines = [
        f"--- {title} ---",
        f"  Trades taken      {s.n_trades:,}",
        f"  WIN RATE          {s.win_rate:.1%}   (95% CI {s.ci_low:.1%}-{s.ci_high:.1%})",
        f"  Avg win           {s.avg_win:+.3%} of book per trade",
        f"  Avg loss          {s.avg_loss:+.3%} of book per trade",
        f"  Payoff ratio      {s.payoff_ratio:.2f}  (avg win / avg loss)",
        f"  Break-even WR     {breakeven:.1%}  <- win rate needed at this payoff",
        f"  Expectancy        {s.expectancy:+.4%} of book PER TRADE",
        f"  Profit factor     {s.profit_factor:.2f}  (>1 makes money)",
        f"  Avg hold          {s.avg_bars_held:.1f} bars",
        f"  Best / worst      {s.best:+.2%} / {s.worst:+.2%}",
        f"  Sum of trades     {s.total:+.2%}",
    ]
    verdict = ("Positive expectancy." if s.expectancy > 0
               else "NEGATIVE expectancy - this loses money per trade.")
    lines.append(f"  {verdict}")
    if s.win_rate > breakeven and s.expectancy < 0:
        lines.append("  Win rate clears break-even but expectancy is negative:")
        lines.append("  the losses are landing in the bigger positions.")
    if s.by_symbol:
        lines.append("  By symbol:")
        for sym, d in sorted(s.by_symbol.items(), key=lambda kv: -kv[1]["total"]):
            lines.append(f"    {sym:10s} n={d['n']:4d}  win {d['win_rate']:5.1%}  "
                         f"[{d['ci_low']:.0%}-{d['ci_high']:.0%}]  "
                         f"total {d['total']:+.2%}")
    return "\n".join(lines)
