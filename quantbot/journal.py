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
    # Market conditions at the moment of the call, so failures can later be
    # attributed to a regime rather than left as "it just did not work".
    context: dict = field(default_factory=dict)
    # Filled in at resolution time:
    resolved: bool = False
    exit_price: float | None = None
    realised_return: float | None = None
    correct: bool | None = None
    pnl_weight: float | None = None   # return x weight, i.e. contribution


def norm_quantile(p: float) -> float:
    """Inverse normal CDF, for turning a corrected confidence level into z."""
    from quantbot.metrics import norm_inv
    return norm_inv(p)


def _bucket_key(p: "Prediction", dimension: str):
    """Which bucket a prediction falls in, for a named attribution dimension."""
    if dimension == "symbol":
        return p.symbol
    if dimension == "engine":
        return p.engine
    if dimension == "direction":
        return "long" if p.target_weight > 1e-9 else "flat"
    if dimension == "volatility regime":
        v = p.context.get("vol")
        if v is None:
            return None
        return "calm <50%" if v < 0.5 else "normal 50-90%" if v < 0.9 else "wild >90%"
    if dimension == "trend regime":
        t = p.context.get("trend_z")
        if t is None:
            return None
        return "downtrend" if t < -0.5 else "flat" if t < 0.5 else "uptrend"
    if dimension == "drawdown regime":
        d = p.context.get("drawdown")
        if d is None:
            return None
        return "near highs" if d > -0.05 else "off highs" if d > -0.20 else "deep drawdown"
    return None


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
               bar_minutes: int, sigma: float = 0.0,
               context: dict | None = None) -> Prediction:
        resolve_at = ts + timedelta(minutes=bar_minutes * horizon_bars)
        p = Prediction(
            id=f"{symbol.replace('/', '')}-{int(ts.timestamp())}",
            ts=ts.isoformat(), symbol=symbol, engine=engine,
            prob=float(prob), target_weight=float(target_weight),
            entry_price=float(price), horizon_bars=int(horizon_bars),
            resolve_at=resolve_at.isoformat(), sigma=float(sigma),
            context=dict(context or {}),
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

    # ---------- attribution: WHY it works or does not ----------

    def _bucket_stats(self, group: list) -> dict:
        n = len(group)
        wins = sum(1 for p in group if p.correct)
        lo, hi = wilson_interval(wins, n)
        return {
            "n": n, "hit_rate": wins / n if n else 0.0,
            "ci_low": lo, "ci_high": hi,
            "pnl": sum(p.pnl_weight or 0.0 for p in group),
            "avg_move": sum(abs(p.realised_return or 0.0) for p in group) / n if n else 0.0,
            # Only call a bucket good or bad when the interval excludes chance.
            "verdict": ("works" if lo > 0.5 else "fails" if hi < 0.5 else "unproven"),
        }

    def attribution(self, min_n: int = 20) -> dict[str, list[dict]]:
        """Break the record down by the conditions each call was made under.

        A single overall hit rate hides everything useful. A model can be
        genuinely predictive in calm markets and actively harmful in volatile
        ones, and the blended average will show neither. Each bucket carries
        its own interval, and buckets thinner than `min_n` are reported but
        flagged unproven rather than acted on.
        """
        rs = self.resolved()
        out: dict[str, list[dict]] = {}

        def group_by(name: str, keyfn):
            buckets: dict[str, list] = {}
            for p in rs:
                try:
                    k = keyfn(p)
                except Exception:
                    continue
                if k is None:
                    continue
                buckets.setdefault(str(k), []).append(p)
            rows = []
            for k, g in sorted(buckets.items()):
                st = self._bucket_stats(g)
                st["key"] = k
                st["thin"] = st["n"] < min_n
                rows.append(st)
            if rows:
                out[name] = rows

        group_by("symbol", lambda p: p.symbol)
        group_by("engine", lambda p: p.engine)
        group_by("direction", lambda p: "long" if p.target_weight > 1e-9 else "flat")

        def vol_band(p):
            v = p.context.get("vol")
            if v is None:
                return None
            return "calm <50%" if v < 0.5 else "normal 50-90%" if v < 0.9 else "wild >90%"
        group_by("volatility regime", vol_band)

        def trend_band(p):
            t = p.context.get("trend_z")
            if t is None:
                return None
            return "downtrend" if t < -0.5 else "flat" if t < 0.5 else "uptrend"
        group_by("trend regime", trend_band)

        def dd_band(p):
            d = p.context.get("drawdown")
            if d is None:
                return None
            return "near highs" if d > -0.05 else "off highs" if d > -0.20 else "deep drawdown"
        group_by("drawdown regime", dd_band)

        return out

    def diagnosis(self, min_n: int = 20) -> list[str]:
        """Plain-language findings: only conditions the evidence actually supports."""
        findings: list[str] = []
        attr = self.attribution(min_n=min_n)
        for dimension, rows in attr.items():
            for r in rows:
                if r["thin"] or r["verdict"] == "unproven":
                    continue
                verb = "predicts well" if r["verdict"] == "works" else "is worse than a coin flip"
                findings.append(
                    f"{dimension}={r['key']}: {verb} "
                    f"({r['hit_rate']:.0%} over {r['n']} calls, "
                    f"95% CI {r['ci_low']:.0%}-{r['ci_high']:.0%}, pnl {r['pnl']:+.4f})"
                )
        if not findings:
            graded = len(self.resolved())
            findings.append(
                f"no condition has enough evidence yet ({graded} calls graded; "
                f"each bucket needs {min_n}+ before it is worth believing)"
            )
        return findings

    def regime_gates(self, min_n: int = 40, alpha: float = 0.05) -> list[dict]:
        """Conditions where live evidence says the model is worse than chance.

        This is the adaptation that can actually lift the win rate: stop taking
        the trades that lose. It is also the easiest place in the whole system
        to fool yourself, because the condition is chosen AFTER looking at the
        results.

        Scanning ~10 regime buckets and keeping whichever looks worst is a
        search, and a 95% interval is expected to exclude chance in 1 bucket in
        20 by luck alone. So the threshold is Bonferroni-corrected by the number
        of buckets actually examined: with 10 buckets each must clear 99.5%
        confidence, not 95%. That is deliberately hard to trigger. A gate that
        fires on the first suggestive pattern is a gate that fits noise.

        Only conditions that are worse than a coin flip are returned. A bucket
        that looks unusually GOOD is never used to size up, for the same reason
        the risk multiplier never exceeds 1.0.
        """
        rs = self.resolved()
        if not rs:
            return []

        attr = self.attribution(min_n=min_n)
        buckets = [(dim, r) for dim, rows in attr.items() for r in rows]
        n_tested = max(len(buckets), 1)
        # Bonferroni: split the error budget across every bucket examined.
        z = norm_quantile(1.0 - (alpha / n_tested) / 2.0)

        gates = []
        for dim, r in buckets:
            if r["n"] < min_n:
                continue
            group = [p for p in rs if _bucket_key(p, dim) == r["key"]]
            wins = sum(1 for p in group if p.correct)
            lo, hi = wilson_interval(wins, len(group), z=z)
            if hi < 0.5:
                gates.append({
                    "dimension": dim, "key": r["key"], "n": len(group),
                    "hit_rate": wins / len(group), "ci_low": lo, "ci_high": hi,
                    "pnl": r["pnl"], "z": z, "buckets_tested": n_tested,
                })
        return gates

    def gate_report(self, min_n: int = 40, alpha: float = 0.05) -> str:
        gates = self.regime_gates(min_n=min_n, alpha=alpha)
        attr = self.attribution(min_n=min_n)
        tested = sum(len(rows) for rows in attr.values())
        if not gates:
            return ("REGIME GATES\n  none active - no condition is worse than "
                    f"chance once corrected for the {tested} buckets examined")
        lines = ["REGIME GATES (live evidence says skip these)"]
        for g in gates:
            lines.append(f"  SKIP {g['dimension']}={g['key']}  "
                         f"hit {g['hit_rate']:.0%} over {g['n']} calls, "
                         f"corrected CI {g['ci_low']:.0%}-{g['ci_high']:.0%}")
        return "\n".join(lines)

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


def format_attribution(journal: Journal, min_n: int = 20) -> str:
    """The 'why' block: where the calls land well and where they do not."""
    attr = journal.attribution(min_n=min_n)
    if not attr:
        return "WHY IT WORKS / FAILS\n  nothing graded yet"
    lines = ["WHY IT WORKS / FAILS (graded calls, by condition)"]
    for dimension, rows in attr.items():
        lines.append(f"  {dimension}:")
        for r in rows:
            flag = "  (thin)" if r["thin"] else ""
            lines.append(
                f"    {r['key']:16s} n={r['n']:4d}  hit {r['hit_rate']:5.0%}  "
                f"[{r['ci_low']:.0%}-{r['ci_high']:.0%}]  pnl {r['pnl']:+.4f}  "
                f"{r['verdict']}{flag}"
            )
    lines.append("  Findings:")
    for f in journal.diagnosis(min_n=min_n):
        lines.append(f"    - {f}")
    return "\n".join(lines)


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
