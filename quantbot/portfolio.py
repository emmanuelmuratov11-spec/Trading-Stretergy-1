"""Paper-trading portfolio with durable state.

The portfolio is the system's memory between runs. On GitHub Actions every run
starts in a fresh container, so state must round-trip through a JSON file that
is committed back to the repo -- otherwise the bot forgets its positions every
hour and re-buys everything it already owns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

EPS = 1e-12


@dataclass
class Order:
    symbol: str
    side: str            # BUY | SELL
    qty: float
    price: float
    notional: float
    reason: str = ""

    def describe(self, quote: str = "USD") -> str:
        return (f"{self.side} {self.qty:.6g} {self.symbol.split('/')[0]} "
                f"@ ~{self.price:,.2f} ({self.notional:,.2f} {quote})")


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

    def value(self, price: float) -> float:
        return self.qty * price

    def unrealised(self, price: float) -> float:
        return (price - self.avg_price) * self.qty


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realised_pnl: float = 0.0
    fees_paid: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    trade_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    history: list[dict] = field(default_factory=list)

    # ---------- valuation ----------

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None and pos.qty != 0.0:
                total += pos.value(px)
        return total

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= EPS:
            return {s: 0.0 for s in prices}
        return {s: self.positions.get(s, Position()).value(prices[s]) / eq
                for s in prices}

    def drawdown(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        peak = max(self.peak_equity, eq)
        return eq / peak - 1.0 if peak > 0 else 0.0

    # ---------- trading ----------

    def plan_rebalance(self, targets: dict[str, float], prices: dict[str, float],
                       min_notional: float = 10.0,
                       band: float = 0.02) -> list[Order]:
        """Compute the orders needed to move to `targets` (weights of equity)."""
        eq = self.equity(prices)
        orders: list[Order] = []
        for sym, target_w in targets.items():
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            current_qty = self.positions.get(sym, Position()).qty
            current_w = current_qty * price / eq if eq > EPS else 0.0
            delta_w = target_w - current_w

            # Skip immaterial moves, but always allow a full exit.
            exiting = abs(target_w) < EPS and abs(current_qty) > EPS
            if abs(delta_w) < band and not exiting:
                continue

            delta_notional = delta_w * eq
            if abs(delta_notional) < min_notional and not exiting:
                continue

            qty = delta_notional / price
            if exiting:
                qty = -current_qty
            if abs(qty) < EPS:
                continue
            orders.append(Order(
                symbol=sym,
                side="BUY" if qty > 0 else "SELL",
                qty=abs(qty),
                price=price,
                notional=abs(qty) * price,
                reason=f"target {target_w:+.1%} vs current {current_w:+.1%}",
            ))
        return orders

    def apply(self, orders: list[Order], fee_rate: float = 0.0006) -> None:
        """Execute orders against the paper book."""
        for o in orders:
            pos = self.positions.setdefault(o.symbol, Position())
            fee = o.notional * fee_rate
            self.fees_paid += fee
            self.cash -= fee
            self.trade_count += 1

            if o.side == "BUY":
                new_qty = pos.qty + o.qty
                if new_qty > EPS:
                    pos.avg_price = (pos.avg_price * pos.qty + o.price * o.qty) / new_qty
                pos.qty = new_qty
                self.cash -= o.notional
            else:
                sold = min(o.qty, max(pos.qty, 0.0))
                self.realised_pnl += (o.price - pos.avg_price) * sold
                pos.qty -= o.qty
                self.cash += o.notional
                if abs(pos.qty) < 1e-10:
                    pos.qty = 0.0
                    pos.avg_price = 0.0

    def mark(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.peak_equity = max(self.peak_equity, eq)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.history.append({"ts": self.updated_at, "equity": round(eq, 2),
                             "cash": round(self.cash, 2)})
        # Keep the file from growing without bound across years of hourly runs.
        if len(self.history) > 5000:
            self.history = self.history[-5000:]
        return eq

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = {s: asdict(p) for s, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        positions = {s: Position(**p) for s, p in (d.get("positions") or {}).items()}
        known = {"cash", "realised_pnl", "fees_paid", "peak_equity", "halted",
                 "trade_count", "created_at", "updated_at", "history"}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(positions=positions, **kwargs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, initial_capital: float) -> "Portfolio":
        if not os.path.exists(path):
            now = datetime.now(timezone.utc).isoformat()
            return cls(cash=initial_capital, peak_equity=initial_capital,
                       created_at=now, updated_at=now)
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
