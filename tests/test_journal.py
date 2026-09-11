"""Prediction-journal tests.

The journal is what turns "the backtest says so" into "here is what actually
happened". These tests pin the two properties that stop it becoming another
self-deception: small samples must not drive decisions, and live evidence must
never be allowed to increase risk.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quantbot.journal import Journal, format_scorecard, wilson_interval


def _frame(prices, start, freq_min=60):
    idx = pd.date_range(start, periods=len(prices), freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame({"open": prices, "high": prices, "low": prices,
                         "close": prices, "volume": [1.0] * len(prices)}, index=idx)


def test_wilson_interval_is_wide_on_small_samples():
    lo, hi = wilson_interval(3, 5)
    assert lo < 0.30 and hi > 0.85, "3/5 must not read as a confident 60%"
    lo2, hi2 = wilson_interval(300, 500)
    assert hi2 - lo2 < 0.15, "500 samples should give a tight interval"


def test_a_correct_upward_call_is_graded_correct():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    j = Journal()
    j.record(ts=t0, symbol="BTC/USD", engine="ml", prob=0.7, target_weight=0.3,
             price=100.0, horizon_bars=2, bar_minutes=60)
    frames = {"BTC/USD": _frame([100.0, 105.0, 110.0, 112.0], t0)}
    assert j.resolve(frames, now=t0 + timedelta(hours=5)) == 1
    p = j.predictions[0]
    assert p.correct is True
    assert p.realised_return == pytest.approx(0.10)
    assert p.pnl_weight == pytest.approx(0.03)


def test_a_wrong_call_is_graded_wrong():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    j = Journal()
    j.record(ts=t0, symbol="BTC/USD", engine="ml", prob=0.8, target_weight=0.5,
             price=100.0, horizon_bars=2, bar_minutes=60)
    j.resolve({"BTC/USD": _frame([100.0, 95.0, 90.0, 88.0], t0)},
              now=t0 + timedelta(hours=5))
    assert j.predictions[0].correct is False
    assert j.predictions[0].pnl_weight < 0


def test_immature_predictions_are_not_graded():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    j = Journal()
    j.record(ts=t0, symbol="BTC/USD", engine="ml", prob=0.7, target_weight=0.3,
             price=100.0, horizon_bars=24, bar_minutes=60)
    assert j.resolve({"BTC/USD": _frame([100.0] * 5, t0)},
                     now=t0 + timedelta(hours=1)) == 0
    assert not j.predictions[0].resolved


def test_the_same_bar_is_not_double_counted():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    j = Journal()
    for _ in range(3):
        j.record(ts=t0, symbol="BTC/USD", engine="ml", prob=0.6,
                 target_weight=0.2, price=100.0, horizon_bars=2, bar_minutes=60)
    assert len(j.predictions) == 1, "a rerun of the same bar must not re-log"


def _stuff(j, n, correct, prob=0.7, ret=0.01):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        p = j.record(ts=t0 + timedelta(hours=i), symbol="BTC/USD", engine="ml",
                     prob=prob, target_weight=0.3, price=100.0,
                     horizon_bars=1, bar_minutes=60)
        p.resolved = True
        p.realised_return = ret if correct else -ret
        p.correct = correct
        p.pnl_weight = (ret if correct else -ret) * 0.3
    return j


def test_no_adaptation_before_enough_evidence():
    j = _stuff(Journal(), 10, correct=False)
    mult, reason = j.risk_multiplier(min_samples=60)
    assert mult == 1.0 and "not adapting yet" in reason


def test_sustained_losing_stands_the_system_down():
    j = _stuff(Journal(), 120, correct=False)
    mult, reason = j.risk_multiplier(min_samples=60)
    assert mult == 0.0, "evidence of no edge must cut size to zero"
    assert "coin flip" in reason


def test_winning_never_increases_size():
    """The asymmetry that matters: a hot streak is indistinguishable from luck,
    so live evidence may reduce risk and must never raise it."""
    j = _stuff(Journal(), 300, correct=True)
    mult, _ = j.risk_multiplier(min_samples=60)
    assert mult <= 1.0, "live results must never size the book up"


def test_scorecard_reports_uncertainty_not_just_a_rate():
    j = _stuff(Journal(), 80, correct=True)
    s = j.scorecard()
    assert s["n"] == 80 and s["ci_low"] <= s["hit_rate"] <= s["ci_high"]
    assert s["brier"] < 0.25, "confident and right should beat a coin flip"


def test_calibration_splits_by_confidence_band():
    j = Journal()
    _stuff(j, 40, correct=True, prob=0.9)
    t0 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    for i in range(40):
        p = j.record(ts=t0 + timedelta(hours=i), symbol="ETH/USD", engine="ml",
                     prob=0.52, target_weight=0.1, price=50.0,
                     horizon_bars=1, bar_minutes=60)
        p.resolved, p.realised_return, p.correct, p.pnl_weight = True, -0.01, False, -0.001
    bands = j.calibration()
    assert len(bands) >= 2, "distinct confidence levels must land in distinct bands"


def test_journal_survives_a_round_trip(tmp_path):
    j = _stuff(Journal(), 5, correct=True)
    path = str(tmp_path / "state" / "journal.json")
    j.save(path)
    back = Journal.load(path)
    assert len(back.predictions) == 5
    assert back.scorecard()["n"] == 5


def test_corrupt_journal_does_not_crash_the_run(tmp_path):
    path = tmp_path / "journal.json"
    path.write_text("{not valid json")
    assert Journal.load(str(path)).predictions == []


def test_scorecard_renders_before_anything_is_graded():
    j = Journal()
    j.record(ts=datetime(2026, 1, 1, tzinfo=timezone.utc), symbol="BTC/USD",
             engine="ml", prob=0.6, target_weight=0.2, price=100.0,
             horizon_bars=24, bar_minutes=60)
    assert "no calls graded yet" in format_scorecard(j)


def _call(j, i, *, correct, vol, trend_z=0.0, dd=-0.01, symbol="BTC/USD",
          prob=0.65, weight=0.3, ret=0.01):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    p = j.record(ts=t0 + timedelta(hours=i), symbol=symbol, engine="ml",
                 prob=prob, target_weight=weight, price=100.0, horizon_bars=1,
                 bar_minutes=60,
                 context={"vol": vol, "trend_z": trend_z, "drawdown": dd})
    p.resolved = True
    p.realised_return = ret if correct else -ret
    p.correct = correct
    p.pnl_weight = (ret if correct else -ret) * weight
    return p


def test_attribution_finds_a_regime_specific_failure():
    """The headline hit rate can hide everything that matters: a model can be
    predictive in calm markets and harmful in violent ones, and the blended
    average shows neither."""
    j = Journal()
    for i in range(60):
        _call(j, i, correct=True, vol=0.3)             # calm: right
    for i in range(60, 120):
        _call(j, i, correct=False, vol=1.2)            # wild: wrong

    overall = j.scorecard()
    assert 0.45 < overall["hit_rate"] < 0.55, "blended rate should look like noise"

    bands = {r["key"]: r for r in j.attribution()["volatility regime"]}
    assert bands["calm <50%"]["verdict"] == "works"
    assert bands["wild >90%"]["verdict"] == "fails"

    findings = " | ".join(j.diagnosis())
    assert "calm" in findings and "wild" in findings


def test_thin_buckets_are_flagged_not_trusted():
    j = Journal()
    for i in range(5):
        _call(j, i, correct=True, vol=0.3)
    rows = j.attribution(min_n=20)["volatility regime"]
    assert rows[0]["thin"] is True
    assert "no condition has enough evidence yet" in " ".join(j.diagnosis(min_n=20))


def test_attribution_splits_by_symbol_and_trend():
    j = Journal()
    for i in range(40):
        _call(j, i, correct=True, vol=0.4, trend_z=1.5, symbol="BTC/USD")
    for i in range(40, 80):
        _call(j, i, correct=False, vol=0.4, trend_z=-1.5, symbol="ETH/USD")
    attr = j.attribution()
    assert {r["key"] for r in attr["symbol"]} == {"BTC/USD", "ETH/USD"}
    assert {r["key"] for r in attr["trend regime"]} == {"uptrend", "downtrend"}


def test_context_survives_persistence(tmp_path):
    j = Journal()
    _call(j, 0, correct=True, vol=0.42, trend_z=1.1)
    path = str(tmp_path / "j.json")
    j.save(path)
    back = Journal.load(path)
    assert back.predictions[0].context["vol"] == pytest.approx(0.42)
    assert back.predictions[0].context["trend_z"] == pytest.approx(1.1)


def test_attribution_renders_without_crashing():
    from quantbot.journal import format_attribution
    j = Journal()
    for i in range(30):
        _call(j, i, correct=i % 2 == 0, vol=0.6)
    text = format_attribution(j)
    assert "WHY IT WORKS / FAILS" in text and "volatility regime" in text
