from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from math import floor
from pathlib import Path
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.schemas import (
    DailyState,
    ExecutionCommand,
    FeatureSnapshot,
    RiskDecision,
    StrategyIntent,
)

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

NY_TZ = ZoneInfo("America/New_York")


@dataclass(slots=True)
class RiskConfig:
    per_trade_risk_pct: float = 0.0025
    daily_soft_lock_pct: float = 0.0075
    daily_hard_lock_pct: float = 0.015
    daily_loss_hard_lock_pct: float = -0.015
    max_entries_per_day: int = 8
    cooldown_after_close_sec: int = 180
    max_consecutive_losses: int = 3
    max_spread: float = 0.50
    intent_ttl_seconds: int = 45
    no_new_entries_after: time = time(15, 30)
    daily_profit_target_usd: float | None = None
    daily_loss_limit_usd: float | None = None


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    h, m = value.split(":")[:2]
    return time(int(h), int(m))


def _to_float(value, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _to_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _load_profile(path: Path, defaults: RiskConfig) -> dict:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def load_risk_config(
    profile_path: str | None = None,
) -> tuple[RiskConfig, str]:
    defaults = RiskConfig()
    source = "defaults"

    final = asdict(defaults)
    profile_file = (
        Path(profile_path)
        if profile_path
        else Path(__file__).resolve().parent / "config" / "risk_profile.yaml"
    )

    profile_data = _load_profile(profile_file, defaults)
    if profile_data:
        source = "profile"
        for key in final:
            if key not in profile_data:
                continue
            if key == "no_new_entries_after":
                final[key] = _parse_time(profile_data[key])
            elif key in {"daily_profit_target_usd", "daily_loss_limit_usd"}:
                val = profile_data[key]
                final[key] = _to_float(val, 0.0) if val is not None else None
            elif isinstance(final[key], int):
                final[key] = _to_int(profile_data[key], final[key])
            elif isinstance(final[key], float):
                final[key] = _to_float(profile_data[key], final[key])

    env_map = {
        "per_trade_risk_pct": "RISK_PER_TRADE_PCT",
        "daily_soft_lock_pct": "RISK_DAILY_SOFT_LOCK_PCT",
        "daily_hard_lock_pct": "RISK_DAILY_HARD_LOCK_PCT",
        "daily_loss_hard_lock_pct": "RISK_DAILY_LOSS_LOCK_PCT",
        "max_entries_per_day": "RISK_MAX_ENTRIES_PER_DAY",
        "cooldown_after_close_sec": "RISK_COOLDOWN_AFTER_CLOSE_SEC",
        "max_consecutive_losses": "RISK_MAX_CONSECUTIVE_LOSSES",
        "max_spread": "RISK_MAX_SPREAD",
        "intent_ttl_seconds": "RISK_INTENT_TTL_SECONDS",
        "no_new_entries_after": "RISK_NO_NEW_ENTRIES_AFTER",
        "daily_profit_target_usd": "DAILY_PROFIT_TARGET_USD",
        "daily_loss_limit_usd": "DAILY_LOSS_LIMIT_USD",
    }

    env_used = False
    for key, env_key in env_map.items():
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        env_used = True
        if key == "no_new_entries_after":
            final[key] = _parse_time(raw)
        elif key in {"daily_profit_target_usd", "daily_loss_limit_usd"}:
            final[key] = _to_float(raw, final[key])
        elif isinstance(final[key], int):
            final[key] = _to_int(raw, final[key])
        elif isinstance(final[key], float):
            final[key] = _to_float(raw, final[key])

    if env_used:
        source = "env"

    cfg = RiskConfig(**final)
    return cfg, source


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None, config_source: str | None = None) -> None:
        if config is not None:
            self.config = config
            self.config_source = config_source or "injected"
        else:
            self.config, self.config_source = load_risk_config()

    def update_locks(self, day_state: DailyState, current_equity: float) -> DailyState:
        day_state.current_equity = current_equity
        day_state.equity_hwm = max(day_state.equity_hwm, current_equity)
        day_state.equity_change_pct = (
            (current_equity - day_state.day_start_equity) / day_state.day_start_equity
        )
        day_state.dollar_pnl = current_equity - day_state.day_start_equity

        if day_state.equity_change_pct >= self.config.daily_hard_lock_pct:
            day_state.hard_lock = True
            day_state.soft_lock = True
            day_state.last_lock_reason = "hard_profit_lock"
        elif day_state.equity_change_pct <= self.config.daily_loss_hard_lock_pct:
            day_state.hard_lock = True
            day_state.soft_lock = False
            day_state.last_lock_reason = "hard_loss_lock"
        elif day_state.equity_change_pct >= self.config.daily_soft_lock_pct:
            day_state.soft_lock = True
            if not day_state.last_lock_reason:
                day_state.last_lock_reason = "soft_profit_lock"

        if self.config.daily_profit_target_usd is not None and day_state.dollar_pnl >= self.config.daily_profit_target_usd:
            day_state.hard_lock = True
            day_state.soft_lock = True
            day_state.last_lock_reason = "dollar_profit_target_reached"
        if self.config.daily_loss_limit_usd is not None and day_state.dollar_pnl <= -self.config.daily_loss_limit_usd:
            day_state.hard_lock = True
            day_state.last_lock_reason = "dollar_loss_limit_reached"

        return day_state

    def evaluate(
        self,
        intent: StrategyIntent,
        snapshot: FeatureSnapshot,
        day_state: DailyState,
        now: datetime | None = None,
        max_spread_override: float | None = None,
    ) -> RiskDecision:
        ts = now or datetime.now(tz=NY_TZ)
        local_ts = ts.astimezone(NY_TZ)
        decision = RiskDecision(
            intent_id=intent.intent_id,
            approved=False,
            reason_code="UNKNOWN",
            generated_at=ts,
            soft_lock=day_state.soft_lock,
            hard_lock=day_state.hard_lock,
        )

        if day_state.hard_lock:
            decision.reason_code = "HARD_LOCK"
            decision.reason_detail = day_state.last_lock_reason or "hard lock active"
            return decision

        if local_ts.time() > self.config.no_new_entries_after:
            decision.reason_code = "ENTRY_CUTOFF"
            decision.reason_detail = "new entries disabled after configured cutoff"
            return decision

        if intent.expires_at < ts:
            decision.reason_code = "INTENT_EXPIRED"
            decision.reason_detail = "strategy intent is stale"
            return decision

        max_spread_threshold = (
            float(max_spread_override)
            if max_spread_override is not None
            else float(self.config.max_spread)
        )
        if snapshot.spread > max_spread_threshold:
            decision.reason_code = "SPREAD_TOO_WIDE"
            decision.reason_detail = (
                f"spread {snapshot.spread:.4f} exceeds {max_spread_threshold:.4f}"
            )
            return decision

        if day_state.entries_taken >= min(
            day_state.max_entries_per_day, self.config.max_entries_per_day
        ):
            decision.reason_code = "MAX_ENTRIES_REACHED"
            decision.reason_detail = "daily entry limit reached"
            return decision

        if day_state.consecutive_losses >= self.config.max_consecutive_losses:
            decision.reason_code = "CONSECUTIVE_LOSS_HALT"
            decision.reason_detail = "consecutive loss threshold reached"
            return decision

        if day_state.last_trade_closed_at:
            elapsed = ts - day_state.last_trade_closed_at
            if elapsed < timedelta(seconds=self.config.cooldown_after_close_sec):
                decision.reason_code = "COOLDOWN_ACTIVE"
                decision.reason_detail = "cooldown after previous trade still active"
                return decision

        risk_per_share = abs(intent.entry_price - intent.sl)
        if risk_per_share <= 0:
            decision.reason_code = "INVALID_RISK_PER_SHARE"
            decision.reason_detail = "entry and stop produce zero risk"
            return decision

        size_multiplier = 0.5 if day_state.soft_lock else 1.0
        max_risk = day_state.current_equity * self.config.per_trade_risk_pct * size_multiplier
        qty = floor(max_risk / risk_per_share)
        if qty <= 0:
            decision.reason_code = "SIZE_TOO_SMALL"
            decision.reason_detail = "calculated position size is zero"
            return decision

        decision.approved = True
        decision.reason_code = "APPROVED"
        decision.reason_detail = "risk gates passed"
        decision.size_multiplier = size_multiplier
        decision.final_qty = float(qty)
        decision.risk_per_share = risk_per_share
        decision.risk_amount = risk_per_share * qty
        return decision

    @staticmethod
    def build_execution_command(
        intent: StrategyIntent,
        decision: RiskDecision,
        max_slippage_bps: float = 20.0,
    ) -> ExecutionCommand:
        if not decision.approved:
            raise ValueError("cannot build execution command from denied decision")

        risk_signature = hashlib.sha256(
            f"{intent.intent_id}:{decision.decision_id}:{decision.final_qty}".encode("utf-8")
        ).hexdigest()[:24]

        return ExecutionCommand(
            intent_id=intent.intent_id,
            client_order_id=f"xau-{intent.intent_id[:8]}",
            cancel_after_sec=intent.cancel_after_sec,
            max_slippage_bps=max_slippage_bps,
            risk_signature=risk_signature,
            symbol=intent.symbol,
            side=intent.side,
            qty=decision.final_qty,
            entry_limit_price=intent.entry_price,
            sl=intent.sl,
            tp=intent.tp,
            time_in_force="day",
        )
