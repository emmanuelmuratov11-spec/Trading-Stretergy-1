"""Backfill tests: grading historical out-of-sample calls."""

import numpy as np

from quantbot import backfill, backtest
from quantbot.data import load_universe
from quantbot.data.loader import align


def _frames(cfg):
    cfg.data.lookback_days = 200
    return align(load_universe(cfg.data, use_cache=False))


def test_backfill_grades_historical_calls(cfg):
    frames = _frames(cfg)
    res = backtest.run(cfg, frames)
    j = backfill.run(cfg, frames, res)
    assert len(j.predictions) > 20
    assert len(j.resolved()) > 10, "most matured calls should be gradable"
    s = j.scorecard()
    assert 0.0 <= s["hit_rate"] <= 1.0
    assert s["ci_low"] <= s["hit_rate"] <= s["ci_high"]


def test_samples_are_independent_not_overlapping(cfg):
    """Consecutive bars share almost all of their outcome window. Counting
    every bar would inflate the sample and make noise look significant."""
    frames = _frames(cfg)
    res = backtest.run(cfg, frames)
    j = backfill.run(cfg, frames, res)

    import pandas as pd
    for sym in {p.symbol for p in j.predictions}:
        ts = sorted(pd.Timestamp(p.ts) for p in j.predictions if p.symbol == sym)
        if len(ts) < 3:
            continue
        gaps = [(b - a).total_seconds() / 3600 for a, b in zip(ts[:-1], ts[1:])]
        assert min(gaps) >= cfg.labels.horizon_bars - 1e-6, (
            "predictions overlap; samples are not independent"
        )


def test_stride_controls_sample_density(cfg):
    frames = _frames(cfg)
    res = backtest.run(cfg, frames)
    sparse = backfill.run(cfg, frames, res, stride=cfg.labels.horizon_bars * 4)
    dense = backfill.run(cfg, frames, res, stride=cfg.labels.horizon_bars)
    assert len(sparse.predictions) < len(dense.predictions)


def test_context_is_recorded_for_attribution(cfg):
    frames = _frames(cfg)
    res = backtest.run(cfg, frames)
    j = backfill.run(cfg, frames, res)
    graded = j.resolved()
    assert graded
    assert all(np.isfinite(p.context.get("vol", np.nan)) for p in graded[:20])
    attr = j.attribution(min_n=5)
    assert "volatility regime" in attr or "symbol" in attr


def test_backfill_only_uses_out_of_sample_predictions(cfg):
    """Probabilities come from the walk-forward result, whose every value was
    produced by a model that had not seen that bar."""
    frames = _frames(cfg)
    res = backtest.run(cfg, frames)
    j = backfill.run(cfg, frames, res)
    import pandas as pd
    for p in j.predictions[:30]:
        ts = pd.Timestamp(p.ts)
        assert ts in res.probabilities.index
        assert np.isfinite(res.probabilities.loc[ts, p.symbol])
