"""Feature engineering.

THE ONE RULE: every feature at timestamp t may use information from bars at or
before t, and nothing else. Any rolling window, any z-score, any normalisation
constant must be computed causally. A single accidental `.shift(-1)` or a
full-sample `.mean()` turns a worthless strategy into a spectacular backtest,
which is the most expensive mistake in this codebase.

`tests/test_no_lookahead.py` enforces this empirically: it verifies that
features at time t are bit-identical whether or not future bars exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbot.config import FeatureConfig

EPS = 1e-12


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score. Uses only the trailing window -- never the full sample."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + EPS)


def realised_vol(close: pd.Series, window: int, bars_per_year: float) -> pd.Series:
    """Annualised trailing volatility of log returns."""
    logret = np.log(close).diff()
    return logret.ewm(span=window, adjust=False, min_periods=window // 2).std() * np.sqrt(bars_per_year)


def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    bars_per_year: float,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the causal feature matrix for one symbol.

    `market` is the benchmark symbol (typically BTC) used for cross-asset
    features; pass None for the benchmark itself.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    logret = np.log(close).diff()
    feats: dict[str, pd.Series] = {}

    # --- Momentum across horizons, volatility-normalised so that a "big move"
    # --- means the same thing in calm and violent regimes.
    ref_vol = logret.ewm(span=cfg.vol_windows[-1], adjust=False,
                         min_periods=10).std()
    for w in cfg.return_windows:
        r = np.log(close).diff(w)
        feats[f"ret_{w}"] = r
        feats[f"ret_{w}_norm"] = r / (ref_vol * np.sqrt(w) + EPS)

    # --- Volatility level, and whether volatility is itself rising.
    for w in cfg.vol_windows:
        vol = realised_vol(close, w, bars_per_year)
        feats[f"vol_{w}"] = vol
        feats[f"vol_{w}_z"] = _zscore(vol, max(w * 2, 48))
    short_v, long_v = cfg.vol_windows[0], cfg.vol_windows[-1]
    feats["vol_ratio"] = (realised_vol(close, short_v, bars_per_year)
                          / (realised_vol(close, long_v, bars_per_year) + EPS))

    # --- Trend: distance from moving averages, scaled by volatility.
    for w in cfg.ma_windows:
        ma = close.rolling(w, min_periods=w // 2).mean()
        feats[f"ma_dist_{w}"] = (close - ma) / (close * ref_vol * np.sqrt(w) + EPS)
    if len(cfg.ma_windows) >= 2:
        fast = close.rolling(cfg.ma_windows[0], min_periods=2).mean()
        slow = close.rolling(cfg.ma_windows[-1], min_periods=2).mean()
        feats["ma_cross"] = (fast - slow) / (close + EPS)

    # --- Classic oscillators.
    feats["rsi"] = (_rsi(close, cfg.rsi_window) - 50.0) / 50.0
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    feats["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / (close + EPS)

    bb_w = cfg.ma_windows[1] if len(cfg.ma_windows) > 1 else 20
    bb_mid = close.rolling(bb_w, min_periods=bb_w // 2).mean()
    bb_std = close.rolling(bb_w, min_periods=bb_w // 2).std()
    feats["bb_pctb"] = (close - bb_mid) / (2.0 * bb_std + EPS)

    # --- Range and liquidity structure.
    atr = _atr(df, cfg.atr_window)
    feats["atr_norm"] = atr / (close + EPS)
    feats["hl_range"] = (high - low) / (close + EPS)
    feats["close_loc"] = (close - low) / (high - low + EPS)   # where in the bar we closed
    feats["vol_z"] = _zscore(np.log1p(volume), 168)
    feats["dollar_vol_z"] = _zscore(np.log1p(volume * close), 168)

    # --- Drawdown from the trailing high: risk-on/risk-off context.
    roll_max = close.rolling(336, min_periods=24).max()
    feats["drawdown"] = close / (roll_max + EPS) - 1.0

    # --- Skew and kurtosis of recent returns: tail-risk regime.
    feats["ret_skew"] = logret.rolling(168, min_periods=48).skew()
    feats["ret_kurt"] = logret.rolling(168, min_periods=48).kurt()

    # --- Autocorrelation proxy: is this market trending or mean-reverting now?
    feats["autocorr"] = logret.rolling(168, min_periods=48).corr(logret.shift(1))

    if cfg.include_calendar:
        # Crypto has real intraday and weekday seasonality. Encode cyclically so
        # that hour 23 sits next to hour 0 rather than 23 units away.
        hour = df.index.hour.to_numpy(dtype="float64")
        dow = df.index.dayofweek.to_numpy(dtype="float64")
        feats["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
        feats["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
        feats["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7), index=df.index)
        feats["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7), index=df.index)

    if cfg.include_cross_asset and market is not None:
        mret = np.log(market["close"]).diff()
        feats["mkt_ret_24"] = np.log(market["close"]).diff(24)
        feats["mkt_vol"] = realised_vol(market["close"], 72, bars_per_year)
        # Relative strength and rolling beta to the market factor.
        feats["rel_strength"] = np.log(close).diff(24) - np.log(market["close"]).diff(24)
        cov = logret.rolling(168, min_periods=48).cov(mret)
        var = mret.rolling(168, min_periods=48).var()
        feats["beta"] = cov / (var + EPS)
        feats["corr_mkt"] = logret.rolling(168, min_periods=48).corr(mret)

    out = pd.DataFrame(feats, index=df.index)
    # Infinities come from division by a vanishing volatility in dead markets.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_")]
