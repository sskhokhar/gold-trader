from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class EntryType(str, Enum):
    MARKETABLE_LIMIT = "MARKETABLE_LIMIT"


class Regime(str, Enum):
    TREND = "trend"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_LIQUIDITY = "low_liquidity"
    NEUTRAL = "neutral"


class DataSource(str, Enum):
    STREAM = "stream"
    REST_FALLBACK = "rest_fallback"
    MOCK = "mock"


class IntentState(str, Enum):
    EVENT_DETECTED = "event_detected"
    ANALYSIS_COMPLETE = "analysis_complete"
    INTENT_GENERATED = "intent_generated"
    RISK_APPROVED = "risk_approved"
    ENTRY_SUBMITTED = "entry_submitted"
    ENTRY_FILLED = "entry_filled"
    OCO_SUBMITTED = "oco_submitted"
    CLOSED = "closed"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    HALTED = "halted"
    ERROR = "error"


TERMINAL_INTENT_STATES = {
    IntentState.CLOSED,
    IntentState.DENIED,
    IntentState.EXPIRED,
    IntentState.CANCELLED,
    IntentState.HALTED,
    IntentState.ERROR,
}


class FeatureSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str = Field(default="GLD")
    timestamp: datetime
    data_source: DataSource = Field(default=DataSource.REST_FALLBACK)
    data_age_sec: float = Field(default=0.0, ge=0.0)
    last_price: float = Field(gt=0)
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    spread: float = Field(ge=0)
    spread_bps: float = Field(ge=0)
    volume: float = Field(ge=0)
    avg_volume: float = Field(gt=0)
    volume_spike_ratio: float = Field(ge=0)
    vwap: float = Field(gt=0)
    vwap_displacement_pct: float = Field(ge=0)
    bar_range_pct: float = Field(ge=0)
    rolling_median_range_pct: float = Field(gt=0)
    bar_range_expansion_ratio: float = Field(ge=0)
    macro_proxies: dict[str, float] = Field(default_factory=dict)
    greed_score: float = Field(ge=0, le=100)
    regime: Regime
    is_rth: bool
    data_fresh: bool

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timestamp_tz(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def validate_quote(self) -> "FeatureSnapshot":
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        return self


class MarketSentimentReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str = Field(default="GLD")
    generated_at: datetime
    regime: Regime
    greed_score: float = Field(ge=0, le=100)
    sentiment_score: float = Field(ge=-1, le=1)
    rationale: list[str] = Field(min_length=1)

    @field_validator("generated_at", mode="before")
    @classmethod
    def ensure_generated_at_tz(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class StrategyIntent(BaseModel):
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str = Field(default="GLD")
    side: Side
    entry_type: EntryType = Field(default=EntryType.MARKETABLE_LIMIT)
    entry_price: float = Field(gt=0)
    sl: float = Field(gt=0)
    tp: float = Field(gt=0)
    qty_hint: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    regime: Regime
    generated_at: datetime
    expires_at: datetime
    invalidation_reason: str = Field(min_length=3)
    cancel_after_sec: int = Field(default=30, ge=5, le=120)

    @field_validator("generated_at", "expires_at", mode="before")
    @classmethod
    def ensure_intent_times_tz(cls, value):
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def validate_levels(self) -> "StrategyIntent":
        if self.expires_at <= self.generated_at:
            raise ValueError("expires_at must be after generated_at")
        if self.side == Side.BUY:
            if not (self.sl < self.entry_price < self.tp):
                raise ValueError("BUY intent requires sl < entry_price < tp")
        else:
            if not (self.tp < self.entry_price < self.sl):
                raise ValueError("SELL intent requires tp < entry_price < sl")
        return self


class RiskDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    intent_id: str
    approved: bool
    reason_code: str
    reason_detail: str = ""
    size_multiplier: float = Field(default=1.0, gt=0)
    final_qty: float = Field(default=0, ge=0)
    risk_per_share: float = Field(default=0, ge=0)
    risk_amount: float = Field(default=0, ge=0)
    soft_lock: bool = False
    hard_lock: bool = False
    generated_at: datetime


class ExecutionCommand(BaseModel):
    intent_id: str
    client_order_id: str = Field(min_length=8, max_length=64)
    cancel_after_sec: int = Field(default=30, ge=5, le=120)
    max_slippage_bps: float = Field(default=20, ge=1, le=200)
    risk_signature: str = Field(min_length=8)
    symbol: str = Field(default="GLD")
    side: Side
    qty: float = Field(gt=0)
    entry_limit_price: float = Field(gt=0)
    sl: float = Field(gt=0)
    tp: float = Field(gt=0)
    time_in_force: str = Field(default="day")


class ExecutionReport(BaseModel):
    intent_id: str
    broker_order_id: str | None
    status: str
    filled_qty: float = Field(default=0, ge=0)
    avg_fill_price: float | None = Field(default=None, gt=0)
    timestamps: dict[str, str] = Field(default_factory=dict)
    reject_reason: str | None = None
    entry_order_id: str | None = None
    oco_order_id: str | None = None


class DailyState(BaseModel):
    day: str
    day_start_equity: float = Field(gt=0)
    equity_hwm: float = Field(gt=0)
    current_equity: float = Field(gt=0)
    equity_change_pct: float = 0.0
    soft_lock: bool = False
    hard_lock: bool = False
    max_entries_per_day: int = Field(default=8, ge=1)
    entries_taken: int = Field(default=0, ge=0)
    consecutive_losses: int = Field(default=0, ge=0)
    last_trade_closed_at: datetime | None = None
    last_lock_reason: str = ""


class TransitionEvent(BaseModel):
    intent_id: str
    from_state: IntentState
    to_state: IntentState
    at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
