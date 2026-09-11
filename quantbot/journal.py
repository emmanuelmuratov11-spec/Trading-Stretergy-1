"""Prediction journal: record what the model expected, then grade it.

A backtest tells you what *would* have happened. This tells you what actually
did. Every run writes down each prediction before the outcome is known --
symbol, probability, intended position, entry price, and the bar by which it
should have resolved. Later runs look back at the ones whose horizon has
elapsed, fetch the realised price, and score them.

That produces the one thing a backtest can never give you: a live, honest track
record with no hindsight in it, because the prediction was committed to disk
before the market moved.

Two design rules keep this from becoming another way to fool yourself:

**Small samples say nothing.** 3 wins out of 5 is not a 60% hit rate, it is
noise. Every rate here is reported with a Wilson confidence interval, and no
adjustment is allowed to act until a minimum number of outcomes exist.

**Adjustments only ever reduce risk.** If live results look bad, the system
trades smaller or stands down. If they look good, it does *not* size up. That
asymmetry is deliberate: a winning streak over 30 trades is indistinguishable
from luck, while a losing streak is at least consistent with having no edge.
Sizing up on noise is how accounts die.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

EPS = 1e-12


@dataclass
class Prediction:
    id: str
    ts: str                  # when the call was made
    symbol: str
    engine: str
    prob: float              # model's probability that price rises
    target_weight: float
    entry_price: float
    horizon_bars: int
    resolve_at: str          # when it should be graded
    sigma: float = 0.0
    # Filled in at resolution time:
    resolved: bool = False
    exit_price: float | None = None
    realised_return: float | None = None
    correct: bool | None = None
    pnl_weight: float | None = None   # return x weight, i.e. contribution


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a hit rate.

    The naive rate lies badly on small samples: 3/5 looks like 60% but its
    interval runs from 23% to 88%, which is the honest summary.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class Journal:
    predictions: list[Prediction] = field(default_factory=list)
    max_entries: int = 20_000

    # ---------- recording ----------

    def record(self, ts: datetime, symbol: str, engine: str, prob: float,
               target_weight: float, price: float, horizon_bars: int,
               bar_minutes: int, sigma: float = 0.0) -> Prediction:
        resolve_at = ts + timedelta(minutes=bar_minutes * horizon_bars)
        p = Prediction(
            id=f"{symbol.replace('/', '')}-{int(ts.timestamp())}",
            ts=ts.isoformat(), symbol=symbol, engine=engine,
            prob=float(prob), target_weight=float(target_weight),
            entry_price=float(price), horizon_bars=int(horizon_bars),
            resolve_at=resolve_at.isoformat(), sigma=float(sigma),
        )
        # Re-running the same bar must not double-count a call.
        if not any(x.id == p.id for x in self.predictions):
            self.predictions.append(p)
        if len(self.predictions) > self.max_entries:
            self.predictions = self.predictions[-self.max_entries:]
        return p

    # ---------- grading ----------

    def resolve(self, frames: dict[str, pd.DataFrame],
                now: datetime | None = None) -> int:
        """Grade every matured prediction against the realised price.

        Uses the fetched bar history rather than the latest tick, so a
        prediction is scored at the price that actually prevailed at its
        horizon even if no run happened at that moment.
        """
        now = now or datetime.now(timezone.utc)
        graded = 0
        for p in self.predictions:
            if p.resolved:
                continue
            resolve_at = pd.Timestamp(p.resolve_at)
            if resolve_at > pd.Timestamp(now):
                continue
            df = frames.get(p.symbol)
            if df is None or len(df) == 0:
                continue
            future = df.index[df.index >= resolve_at]
            if len(future) == 0:
                continue          # horizon passed but bars not fetched yet
            exit_price = float(df.loc[future[0], "close"])
            r = exit_price / max(p.entry_price, EPS) - 1.0
            p.exit_price = exit_price
            p.realised_return = r
            # "Correct" means the direction it leaned actually happened.
            p.correct = bool((r > 0) == (p.prob > 0.5)) if abs(p.prob - 0.5) > 1e-9 else None
            p.pnl_weight = r * p.target_weight
            p.resolved = True
            graded += 1
        return graded

    # ---------- scoring ----------

    def resolved(self) -> list[Prediction]:
        return [p for p in self.predictions if p.resolved and p.correct is not None]

    def scorecard(self, min_confidence: float = 0.0) -> dict:
        rs = [p for p in self.resolved() if abs(p.prob - 0.5) >= min_confidence]
        n = len(rs)
        if n == 0:
            return {"n": 0}
        wins = sum(1 for p in rs if p.correct)
        lo, hi = wilson_interval(wins, n)
        pnl = sum(p.pnl_weight or 0.0 for p in rs)
        # Brier score: mean squared error of the probabilities themselves.
        # 0.25 is what you get by always saying "50/50"; lower is better,
        # higher means the confidence is actively misleading.
        brier = sum(((p.prob) - (1.0 if (p.realised_return or 0) > 0 else 0.0)) ** 2
                    for p in rs) / n
        return {
            "n": n, "wins": wins, "hit_rate": wins / n,
            "ci_low": lo, "ci_high": hi,
            "sum_pnl_weight": pnl, "brier": brier,
            "beats_coinflip": lo > 0.5,
            "worse_than_coinflip": hi < 0.5,
        }

    def calibration(self, bins: int = 4) -> list[dict]:
        """Reliability by confidence band: does 'more confident' mean 'more right'?"""
        rs = self.resolved()
        out = []
        edges = [0.0, 0.05, 0.10, 0.20, 1.0][: bins + 1]
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            band = [p for p in rs if lo_e <= abs(p.prob - 0.5) < hi_e]
            if not band:
                continue
            wins = sum(1 for p in band if p.correct)
            lo, hi = wilson_interval(wins, len(band))
            out.append({
                "band": f"{lo_e:.2f}-{hi_e:.2f}", "n": len(band),
                "hit_rate": wins / len(band), "ci_low": lo, "ci_high": hi,
                "pnl": sum(p.pnl_weight or 0.0 for p in band),
            })
        return out

    # ---------- adaptation ----------

    def risk_multiplier(self, min_samples: int = 60) -> tuple[float, str]:
        """A live-evidence size multiplier in (0, 1]. Never above 1.

        Returns (multiplier, reason). Until `min_samples` outcomes exist it
        returns 1.0 with a note that there is not enough evidence -- acting on
        20 trades is how you convert noise into a decision.
        """
        s = self.scorecard()
        n = s.get("n", 0)
        if n < min_samples:
            return 1.0, f"only {n}/{min_samples} graded calls - not adapting yet"

        # The upper bound of the hit-rate interval is below a coin flip: the
        # evidence is consistent with having no edge at all.
        if s["worse_than_coinflip"]:
            return 0.0, (f"live hit rate {s['hit_rate']:.1%} "
                         f"(95% CI {s['ci_low']:.1%}-{s['ci_high']:.1%}) is below "
                         "a coin flip - standing down")
        if s["sum_pnl_weight"] < 0 and s["hit_rate"] < 0.5:
            return 0.5, (f"live P&L negative and hit rate {s['hit_rate']:.1%} "
                         "below 50% - halving size")
        if s["brier"] > 0.25:
            return 0.5, (f"Brier {s['brier']:.3f} is worse than always saying "
                         "50/50 - confidence is misleading, halving size")
        return 1.0, f"live evidence acceptable over {n} calls"

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"predictions": [asdict(p) for p in self.predictions]},
                      fh, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Journal":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as fh:
                raw = json.load(fh)
            known = {f for f in Prediction.__dataclass_fields__}
            return cls(predictions=[
                Prediction(**{k: v for k, v in p.items() if k in known})
                for p in raw.get("predictions", [])
            ])
        except Exception:
            return cls()


def format_scorecard(journal: Journal) -> str:
    """A short live-performance block for the phone alert."""
    s = journal.scorecard()
    if s["n"] == 0:
        pending = len([p for p in journal.predictions if not p.resolved])
        return f"LIVE TRACK RECORD\n  no calls graded yet ({pending} pending)"

    lines = ["LIVE TRACK RECORD (graded, no hindsight)"]
    lines.append(f"  Calls graded    {s['n']}")
    lines.append(f"  Hit rate        {s['hit_rate']:.1%}  "
                 f"(95% CI {s['ci_low']:.1%}-{s['ci_high']:.1%})")
    lines.append(f"  Sum P&L/weight  {s['sum_pnl_weight']:+.4f}")
    lines.append(f"  Brier           {s['brier']:.3f}  (0.25 = no better than a coin flip)")
    if s["beats_coinflip"]:
        lines.append("  Statistically better than a coin flip.")
    elif s["worse_than_coinflip"]:
        lines.append("  Statistically WORSE than a coin flip.")
    else:
        lines.append("  Not yet distinguishable from a coin flip.")

    cal = journal.calibration()
    if cal:
        lines.append("  By confidence band:")
        for c in cal:
            lines.append(f"    |p-0.5| {c['band']}  n={c['n']:4d}  "
                         f"hit {c['hit_rate']:.0%}  pnl {c['pnl']:+.4f}")

    mult, reason = journal.risk_multiplier()
    lines.append(f"  Size multiplier {mult:.2f}x - {reason}")
    return "\n".join(lines)
