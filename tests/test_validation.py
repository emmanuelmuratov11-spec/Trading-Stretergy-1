import numpy as np

from quantbot.validation import purged_train_indices, walk_forward_splits


def test_purging_removes_overlapping_labels():
    n = 1000
    t_end = np.minimum(np.arange(n) + 24, n - 1)
    test_idx = np.arange(500, 600)
    train = purged_train_indices(np.arange(0, 500), test_idx, t_end, embargo=24)
    assert t_end[train].max() < test_idx.min(), "a training label reaches into the test window"


def test_embargo_creates_a_real_gap():
    n = 1000
    t_end = np.arange(n)          # labels resolve instantly
    test_idx = np.arange(500, 600)
    train = purged_train_indices(np.arange(0, 500), test_idx, t_end, embargo=50)
    assert train.max() < 450, "embargo did not push the training set back"


def test_walk_forward_is_always_past_to_future():
    n = 3000
    t_end = np.minimum(np.arange(n) + 24, n - 1)
    splits = walk_forward_splits(n, 1000, 200, t_end, embargo=24, min_train_bars=400)
    assert splits
    for s in splits:
        assert s.train_idx.max() < s.test_idx.min(), "training data came after test data"
        assert t_end[s.train_idx].max() < s.test_idx.min()


def test_test_windows_do_not_overlap():
    n = 3000
    t_end = np.minimum(np.arange(n) + 24, n - 1)
    splits = walk_forward_splits(n, 1000, 200, t_end, embargo=24, min_train_bars=400)
    seen: set[int] = set()
    for s in splits:
        assert not (seen & set(s.test_idx.tolist())), "test windows overlap"
        seen |= set(s.test_idx.tolist())
