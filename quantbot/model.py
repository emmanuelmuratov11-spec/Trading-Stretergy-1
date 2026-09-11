"""The learning component: a walk-forward, self-retraining classifier ensemble.

Design choices and why:

* **Gradient-boosted trees**, not a deep net. With a few thousand noisy samples
  and ~45 features, boosted trees are far harder to overfit, need no scaling,
  and handle the NaNs from indicator warm-up natively.

* **Recency via weighted bootstrap, not sample_weight.** Markets are
  non-stationary, so recent bars deserve more influence. Passing `sample_weight`
  to HistGradientBoosting is the obvious way to do that and it is ~20x slower,
  because it disables sklearn's fast unweighted histogram path. Instead each
  ensemble member draws a bootstrap sample with probability proportional to an
  exponential recency decay. That buys the same recency emphasis at full speed,
  and because every seed draws a different resample it also decorrelates the
  ensemble, which is exactly what bagging is for.

* **Chronological calibration.** Position size is driven by how confident the
  model is, so a score of 0.8 must mean roughly an 80% hit rate; raw boosting
  output is notoriously overconfident. Calibration is fitted on the most recent
  slice of the *training* window, held out chronologically. sklearn's
  CalibratedClassifierCV would instead use random folds interleaved in time --
  temporally sloppy, and 3x the fits.

* **Refit on every window.** The model never sees the data it is scored on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from quantbot.config import ModelConfig

log = logging.getLogger(__name__)


def time_decay_weights(n: int, half_life: float | None = None) -> np.ndarray:
    """Exponential recency weights, newest bar weighted 1.0.

    A flat weighting treats a bar from two years ago as being as informative
    about tomorrow as last week's. A half-life of a third of the window is a
    reasonable trade between adaptivity and effective sample size.
    """
    if n <= 0:
        return np.array([])
    hl = half_life if half_life and half_life > 0 else max(n / 3.0, 1.0)
    age = np.arange(n)[::-1]
    return np.power(0.5, age / hl)


class _Calibrator:
    """Maps raw scores to calibrated probabilities.

    Platt/sigmoid scaling by default: two parameters, so it stays stable on the
    few hundred samples a calibration slice holds. Isotonic is more flexible but
    happily overfits small slices, so it is only used when the slice is large.
    """

    def __init__(self, method: str = "auto") -> None:
        self.method = method
        self.impl = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_Calibrator":
        raw = np.asarray(raw, dtype="float64").reshape(-1, 1)
        y = np.asarray(y, dtype="int")
        if len(y) < 40 or len(np.unique(y)) < 2:
            return self
        use_isotonic = self.method == "isotonic" or (self.method == "auto" and len(y) >= 800)
        if use_isotonic:
            self.impl = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
            self.impl.fit(raw.ravel(), y)
        else:
            self.impl = LogisticRegression(C=1.0, solver="lbfgs")
            self.impl.fit(raw, y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype="float64")
        if self.impl is None:
            return raw
        if isinstance(self.impl, IsotonicRegression):
            return np.clip(self.impl.predict(raw), 0.01, 0.99)
        return np.clip(self.impl.predict_proba(raw.reshape(-1, 1))[:, 1], 0.01, 0.99)


def _make_estimator(cfg: ModelConfig, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        max_features=cfg.max_features,
        early_stopping=False,
        random_state=seed,
    )


class WalkForwardModel:
    """Fit an ensemble on one training window and predict the next window."""

    def __init__(self, cfg: ModelConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed
        self.members: list[tuple[HistGradientBoostingClassifier, _Calibrator]] = []
        self.columns: list[str] = []
        self.train_rate: float = 0.5
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WalkForwardModel":
        mask = y.notna()
        Xf, yf = X.loc[mask], y.loc[mask].astype(int)

        if len(yf) < 100 or yf.nunique() < 2:
            # Degenerate window: refuse to fit rather than emit a constant
            # signal dressed up as a prediction.
            self.fitted = False
            return self

        self.columns = list(Xf.columns)
        self.train_rate = float(yf.mean())
        self.members = []

        Xv, yv = Xf.to_numpy(dtype="float64"), yf.to_numpy(dtype="int")
        n = len(yv)

        # Hold out the most recent slice for calibration. It is the closest
        # thing available to the regime the model is about to be asked about.
        cal_n = int(np.clip(round(n * 0.15), 40, 600)) if self.cfg.calibrate else 0
        fit_n = n - cal_n
        if cal_n and (fit_n < 100 or len(np.unique(yv[fit_n:])) < 2):
            cal_n, fit_n = 0, n

        weights = time_decay_weights(fit_n)
        probs = weights / weights.sum()

        for k in range(self.cfg.n_seeds):
            seed = self.seed + k * 101
            rng = np.random.default_rng(seed)
            # Recency-weighted bootstrap (see module docstring).
            idx = rng.choice(fit_n, size=fit_n, replace=True, p=probs)
            Xb, yb = Xv[idx], yv[idx]
            if len(np.unique(yb)) < 2:
                idx = np.arange(fit_n)
                Xb, yb = Xv[idx], yv[idx]

            est = _make_estimator(self.cfg, seed)
            est.fit(Xb, yb)

            cal = _Calibrator(method="auto")
            if cal_n:
                cal.fit(est.predict_proba(Xv[fit_n:])[:, 1], yv[fit_n:])
            self.members.append((est, cal))

        self.fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability the upper barrier is hit first. 0.5 == no opinion."""
        if not self.fitted or not self.members:
            return np.full(len(X), 0.5)
        Xa = X[self.columns].to_numpy(dtype="float64")
        preds = np.column_stack([
            cal.transform(est.predict_proba(Xa)[:, 1]) for est, cal in self.members
        ])
        return preds.mean(axis=1)

    def feature_importance(self, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 3) -> pd.Series:
        """Permutation importance on the supplied (held-out) data."""
        from sklearn.inspection import permutation_importance
        if not self.fitted:
            return pd.Series(dtype="float64")
        mask = y.notna()
        est, _ = self.members[0]
        r = permutation_importance(
            est, X.loc[mask][self.columns].to_numpy(dtype="float64"),
            y.loc[mask].astype(int).to_numpy(),
            n_repeats=n_repeats, random_state=self.seed, scoring="roc_auc",
        )
        return pd.Series(r.importances_mean, index=self.columns).sort_values(ascending=False)
