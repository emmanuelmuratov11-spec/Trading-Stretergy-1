"""Leak-free cross-validation and walk-forward splitting.

Standard k-fold is invalid on financial data for two reasons:

1. **Temporal order.** Training on 2025 to predict 2023 is not a test, it is a
   time machine. Splits must always be past -> future.

2. **Overlapping labels.** A triple-barrier label at bar t resolves at bar
   t_end > t. If t is in the training set and t+1 is in the test set, both
   labels are driven by largely the same future price path, so the model has
   effectively seen the answer. The fix is to *purge* training samples whose
   label windows overlap the test set, and then *embargo* a further gap after
   it to kill residual serial correlation.

Skipping this is the single most common reason a backtest shows a Sharpe of 4
and the live account bleeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __len__(self) -> int:
        return len(self.test_idx)


def purged_train_indices(
    train_candidates: np.ndarray,
    test_idx: np.ndarray,
    t_end: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """Drop training samples that overlap the test window, plus an embargo."""
    if len(test_idx) == 0 or len(train_candidates) == 0:
        return train_candidates
    lo, hi = int(test_idx.min()), int(test_idx.max())
    starts = train_candidates
    ends = t_end[train_candidates]
    # A sample is contaminated when its [start, t_end] window intersects the
    # embargoed test window [lo - embargo, hi + embargo].
    overlaps = (ends >= lo - embargo) & (starts <= hi + embargo)
    return train_candidates[~overlaps]


def walk_forward_splits(
    n: int,
    train_bars: int,
    test_bars: int,
    t_end: np.ndarray,
    embargo: int,
    min_train_bars: int = 500,
    expanding: bool = False,
) -> list[Split]:
    """Yield successive out-of-sample windows, each trained only on its past.

    This is the "self-learning" loop: the model is refit from scratch on each
    window, so it continuously adapts to the most recent regime, and every
    prediction it is judged on was made with no knowledge of that window.
    """
    if test_bars < 1:
        raise ValueError("test_bars must be >= 1")
    splits: list[Split] = []
    start = max(train_bars, min_train_bars)
    if start >= n:
        return splits

    for test_start in range(start, n, test_bars):
        test_stop = min(test_start + test_bars, n)
        test_idx = np.arange(test_start, test_stop)
        train_lo = 0 if expanding else max(0, test_start - train_bars)
        candidates = np.arange(train_lo, test_start)
        train_idx = purged_train_indices(candidates, test_idx, t_end, embargo)
        if len(train_idx) < min_train_bars:
            continue
        splits.append(Split(train_idx=train_idx, test_idx=test_idx))
    return splits
