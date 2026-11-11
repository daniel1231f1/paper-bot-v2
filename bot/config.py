"""Typed, validated, hashable configuration.

Everything except the `ops` section is *locked* for the duration of an experiment: the
config snapshot is stored in the state file at experiment start and its SHA-256 is shown
on the dashboard. Editing config.yaml mid-experiment has no effect (and is flagged).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or out-of-range values."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class ExperimentCfg:
    name: str = "ensemble-v2"
    days: int = 10
    starting_capital: float = 100.0
    min_first_session_hours: float = 8.0  # day 1 needs at least this much session left, else start next open

    def __post_init__(self) -> None:
        _require(self.days >= 1, "experiment.days must be >= 1")
        _require(0 <= self.min_first_session_hours <= 24, "experiment.min_first_session_hours must be 0..24")
        _require(self.starting_capital > 0, "experiment.starting_capital must be > 0")


@dataclass(frozen=True)
class ScheduleCfg:
    timezone: str = "Asia/Kolkata"
    active_start: str = "07:00"
    active_end: str = "23:00"
    entry_cutoff_minutes: int = 60  # no new entries this close to the session end

    def __post_init__(self) -> None:
        for name in ("active_start", "active_end"):
            hh, _, mm = getattr(self, name).partition(":")
            _require(hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60,
                     f"schedule.{name} must be HH:MM")
        _require(self.active_start < self.active_end, "schedule window must not cross midnight")
        _require(self.entry_cutoff_minutes >= 0, "schedule.entry_cutoff_minutes must be >= 0")


@dataclass(frozen=True)
class MarketCfg:
    exchange: str = "coinbaseexchange"  # ccxt id; public candle endpoints only, no keys
    universe: tuple[str, ...] = ("BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "LINK/USD", "AVAX/USD")
    timeframe_minutes: int = 5
    min_notional: float = 5.0  # exchange minimum order value, quote currency
    default_qty_step: float = 1e-6
    qty_step: dict[str, float] = field(default_factory=dict)  # per-symbol lot size

    def __post_init__(self) -> None:
        _require(len(self.universe) >= 1, "market.universe must not be empty")
        _require(len(set(self.universe)) == len(self.universe), "market.universe has duplicates")
        _require(self.timeframe_minutes in (1, 5, 15), "market.timeframe_minutes must be 1, 5 or 15")
        _require(self.min_notional > 0 and self.default_qty_step > 0, "market minimums must be > 0")

    def step(self, symbol: str) -> float:
        return float(self.qty_step.get(symbol, self.default_qty_step))


@dataclass(frozen=True)
class CostCfg:
    """All rates are fractions of notional, charged per side."""
    taker_fee: float = 0.0010
    half_spread: float = 0.0002
    slippage: float = 0.0003
    impact_coef: float = 0.10  # extra slippage = coef * order_notional / bar_dollar_volume
    max_participation: float = 0.02  # max share of the fill bar's volume we may take

    def __post_init__(self) -> None:
        for name in ("taker_fee", "half_spread", "slippage", "impact_coef"):
            _require(0 <= getattr(self, name) < 0.05 or name == "impact_coef", f"costs.{name} out of range")
        _require(0 < self.max_participation <= 1, "costs.max_participation must be in (0, 1]")

    @property
    def side_cost(self) -> float:
        """Fee + half-spread + base slippage for one market fill."""
        return self.taker_fee + self.half_spread + self.slippage

    @property
    def round_trip(self) -> float:
        return 2 * self.side_cost


@dataclass(frozen=True)
class StrategyCfg:
    # indicator lookbacks (bars of the base timeframe unless suffixed _h = hourly bars)
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    bb_period: int = 20
    adx_period: int = 14
    atr_period: int = 14
    vwap_bars: int = 288
    donchian_bars: int = 48
    vol_avg_bars: int = 288
    atr_pct_bars: int = 2016  # window for the ATR percentile (volatility regime)
    ema_fast_h: int = 20
    ema_slow_h: int = 50
    adx_period_h: int = 14
    atr_period_h: int = 14
    # regime classifier
    adx_trend_h: float = 22.0  # hourly ADX above this + EMA alignment => trend
    adx_range_h: float = 18.0  # hourly ADX below this => range
    atr_pct_dead: float = 0.15  # below this percentile the market is too quiet to cover costs
    atr_pct_shock: float = 0.97  # above this percentile we stand aside
    # signals
    allow_short: bool = False
    enabled_signals: tuple[str, ...] = ("trend_pullback", "breakout", "mean_reversion")
    breakout_vol_mult: float = 1.5
    pullback_rsi_max: float = 45.0
    mr_bb_z: float = -2.0
    mr_rsi_max: float = 30.0
    # exits, in multiples of the hourly ATR
    stop_atr: float = 1.5
    target_atr: float = 3.0
    mr_stop_atr: float = 1.5
    mr_target_atr: float = 2.25
    max_hold_bars: int = 96
    min_stop_cost_mult: float = 2.5  # skip if stop distance < this x round-trip cost

    def __post_init__(self) -> None:
        _require(self.ema_fast < self.ema_mid < self.ema_slow, "strategy EMAs must be ordered")
        _require(self.adx_range_h <= self.adx_trend_h, "adx_range_h must be <= adx_trend_h")
        _require(0 <= self.atr_pct_dead < self.atr_pct_shock <= 1, "ATR percentile bounds invalid")
        _require(self.stop_atr > 0 and self.target_atr > 0 and self.max_hold_bars > 0, "exit params must be > 0")
        known = {"trend_pullback", "breakout", "mean_reversion"}
        _require(set(self.enabled_signals) <= known, f"strategy.enabled_signals must be within {sorted(known)}")


@dataclass(frozen=True)
class ModelCfg:
    kind: str = "gbt"  # "gbt" (HistGradientBoosting) or "logreg"
    train_days: int = 60
    retrain_hours: int = 24
    min_samples: int = 300
    calib_fraction: float = 0.25
    min_prob: float = 0.45
    min_ev_r: float = 0.10  # required expected value per trade, in R, after costs
    seed: int = 7

    def __post_init__(self) -> None:
        _require(self.kind in ("gbt", "logreg"), "model.kind must be gbt or logreg")
        _require(self.train_days >= 7 and self.min_samples >= 50, "model window too small")
        _require(0.05 <= self.calib_fraction <= 0.5, "model.calib_fraction must be in [0.05, 0.5]")
        _require(0 < self.min_prob < 1, "model.min_prob must be in (0, 1)")


@dataclass(frozen=True)
class RiskCfg:
    risk_per_trade: float = 0.01
    max_positions: int = 3
    max_position_pct: float = 0.33  # of equity, per position
    daily_loss_limit: float = 0.03  # of day-start equity => no new entries until next session
    kill_switch_drawdown: float = 0.15  # peak-to-trough => flatten and stop for good
    loss_streak: int = 3
    loss_streak_cooldown_bars: int = 24
    symbol_cooldown_bars: int = 6

    def __post_init__(self) -> None:
        _require(0 < self.risk_per_trade <= 0.02, "risk.risk_per_trade must be in (0, 0.02]")
        _require(1 <= self.max_positions <= 3, "risk.max_positions must be 1..3")
        _require(0 < self.max_position_pct <= 1, "risk.max_position_pct must be in (0, 1]")
        _require(0 < self.daily_loss_limit < self.kill_switch_drawdown < 1, "risk limits inconsistent")


@dataclass(frozen=True)
class VerdictCfg:
    """Pre-registered pass criteria for the final report."""
    min_profit_factor: float = 1.3
    max_drawdown: float = 0.15
    min_trades: int = 30
    bootstrap_samples: int = 5000
    confidence: float = 0.90


@dataclass(frozen=True)
class OpsCfg:
    """Operational knobs. NOT part of the locked hash: they don't change trading decisions on fresh data."""
    poll_seconds: int = 20
    max_signal_age_seconds: int = 150  # a signal bar older than this may not open a trade
    stale_after_seconds: int = 900  # newest closed bar older than this => "data stale, not trading"
    warmup_bars: int = 4200  # history needed for converged indicators (~14.6 days of 5m bars)
    max_retries: int = 4
    retry_base_seconds: float = 1.5
    publish_seconds: int = 300
    idle_heartbeat_seconds: int = 1800  # while asleep: refresh the dashboard heartbeat this often
    equity_curve_points: int = 3000


@dataclass(frozen=True)
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    market: MarketCfg = field(default_factory=MarketCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    verdict: VerdictCfg = field(default_factory=VerdictCfg)
    ops: OpsCfg = field(default_factory=OpsCfg)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return _build(cls, data, "config")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _require(isinstance(raw, dict), "config file must be a mapping")
        return cls.from_dict(raw)

    # ---- locking ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))  # tuples -> lists, plain JSON types

    def locked_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        d.pop("ops")
        return d

    @property
    def hash(self) -> str:
        canon = json.dumps(self.locked_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    @property
    def bar_ms(self) -> int:
        return self.market.timeframe_minutes * 60_000


def _build(cls: type, data: Mapping[str, Any], path: str) -> Any:
    hints = get_type_hints(cls)
    unknown = set(data) - set(hints)
    _require(not unknown, f"unknown key(s) in {path}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, tp in hints.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(tp, data[name], f"{path}.{name}")
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    try:
        if is_dataclass(tp):
            _require(isinstance(value, Mapping), f"{path} must be a mapping")
            return _build(tp, value, path)
        origin = get_origin(tp)
        if origin is tuple:
            (inner, *_rest) = get_args(tp)
            return tuple(_coerce(inner, v, path) for v in value)
        if origin is dict:
            _k, v_tp = get_args(tp)
            return {str(k): _coerce(v_tp, v, path) for k, v in dict(value).items()}
        if tp is bool:
            _require(isinstance(value, bool), f"{path} must be true/false")
            return value
        if tp is int:
            _require(isinstance(value, int) and not isinstance(value, bool), f"{path} must be an integer")
            return value
        if tp is float:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{path} must be a number")
            return float(value)
        if tp is str:
            _require(isinstance(value, str), f"{path} must be a string")
            return value
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    raise ConfigError(f"{path}: unsupported type {tp!r}")
