"""Configuration objects, loaded from YAML with conservative defaults."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "binance"          # binance | coinbase | synthetic
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"
    lookback_days: int = 720
    cache_dir: str = "data_cache"
    benchmark: str = "BTC/USDT"
    # 365 for crypto (always open), ~252 for equities (trading sessions).
    annual_days: float = 365.0
    asset_class: str = "crypto"   # crypto | equity, for alert wording


@dataclass
class FeatureConfig:
    return_windows: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 24, 72, 168])
    vol_windows: list[int] = field(default_factory=lambda: [24, 72, 168])
    ma_windows: list[int] = field(default_factory=lambda: [12, 48, 168])
    rsi_window: int = 14
    atr_window: int = 14
    include_calendar: bool = True
    include_cross_asset: bool = True


@dataclass
class LabelConfig:
    """Triple-barrier labelling parameters.

    Barriers are scaled by recent realised volatility so that a 'big move'
    means the same thing in calm and turbulent regimes.
    """
    horizon_bars: int = 24           # vertical barrier
    upper_sigma: float = 1.5         # profit-taking barrier, in sigmas
    lower_sigma: float = 1.5         # stop-loss barrier, in sigmas
    vol_window: int = 72
    min_sigma: float = 1e-4


@dataclass
class ModelConfig:
    # ml | trend | blend. See quantbot/strategies.py for why trend exists.
    kind: str = "ml"
    trend_lookbacks: list[int] = field(default_factory=lambda: [24, 72, 168, 336])
    trend_scale: float = 1.5
    train_bars: int = 4320           # ~6 months of hourly bars
    retrain_every: int = 168         # retrain weekly on hourly data
    min_train_bars: int = 1500
    n_seeds: int = 3                 # ensemble size, averaged to cut variance
    max_iter: int = 220
    learning_rate: float = 0.045
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 60
    l2_regularization: float = 1.5
    max_features: float = 0.7
    embargo_bars: int = 24           # must be >= label horizon
    calibrate: bool = True


@dataclass
class RiskConfig:
    target_annual_vol: float = 0.28  # portfolio vol target
    max_gross_leverage: float = 1.0  # 1.0 == never borrow
    max_position_weight: float = 0.5 # per-symbol cap
    kelly_fraction: float = 0.30     # fraction of "full" Kelly; < 1 always
    conviction_deadband: float = 0.04  # ignore p within this of 0.5
    allow_shorts: bool = False
    stop_loss_sigma: float = 3.0
    max_drawdown_stop: float = 0.25  # flatten and stand down past this DD
    drawdown_recovery: float = 0.10  # re-enter once DD recovers below this
    vol_floor: float = 0.05
    vol_ceiling: float = 2.50
    max_vol_scale: float = 3.0    # cap on leveraging *up* a quiet market
    cov_span: int = 168           # EWMA span for the covariance estimate
    scalar_smoothing: int = 120   # damp sizing jitter; turnover is pure cost


@dataclass
class CostConfig:
    """Costs are deliberately pessimistic. Optimistic costs are the single
    most common way a backtest lies to you."""
    taker_fee_bps: float = 6.0       # 0.06% - typical retail taker fee
    half_spread_bps: float = 2.0
    slippage_coef: float = 0.35      # multiplies (vol / typical vol)
    min_trade_weight: float = 0.04   # absolute no-trade band
    rebalance_band: float = 0.40     # also ignore moves under 40% of the position
    rebalance_every: int = 6         # only consider rebalancing every N bars


@dataclass
class AlertConfig:
    channels: list[str] = field(default_factory=lambda: ["console"])
    min_weight_change: float = 0.05  # only alert on meaningful changes
    send_heartbeat: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    initial_capital: float = 10_000.0
    state_dir: str = "state"
    seed: int = 7

    def validate(self) -> None:
        if self.model.embargo_bars < self.labels.horizon_bars:
            raise ValueError(
                f"embargo_bars ({self.model.embargo_bars}) must be >= "
                f"labels.horizon_bars ({self.labels.horizon_bars}); otherwise "
                "overlapping labels leak future information into training."
            )
        if not 0 < self.risk.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.risk.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if self.risk.max_position_weight <= 0:
            raise ValueError("max_position_weight must be positive")
        if self.labels.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.costs.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        if self.model.kind not in ("ml", "trend", "blend"):
            raise ValueError(
                f"model.kind must be ml, trend or blend (got {self.model.kind!r})")
        if self.model.kind in ("trend", "blend") and not self.model.trend_lookbacks:
            raise ValueError("trend_lookbacks must not be empty")

    @classmethod
    def from_yaml(cls, path: str | None) -> "Config":
        if not path or not os.path.exists(path):
            cfg = cls()
            cfg.validate()
            return cfg
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        sections = {
            "data": DataConfig, "features": FeatureConfig, "labels": LabelConfig,
            "model": ModelConfig, "risk": RiskConfig, "costs": CostConfig,
            "alerts": AlertConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sections.items():
            payload = raw.get(key) or {}
            known = {f.name for f in dataclasses.fields(klass)}
            unknown = set(payload) - known
            if unknown:
                raise ValueError(f"unknown keys in config section '{key}': {sorted(unknown)}")
            kwargs[key] = klass(**payload)
        for scalar in ("initial_capital", "state_dir", "seed"):
            if scalar in raw:
                kwargs[scalar] = raw[scalar]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
