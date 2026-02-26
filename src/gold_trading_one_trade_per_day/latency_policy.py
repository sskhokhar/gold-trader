from __future__ import annotations

import os
from datetime import datetime

from gold_trading_one_trade_per_day.schemas import LatencyPolicyDecision
from gold_trading_one_trade_per_day.state_store import StateStore

DEGRADED_FLAG_KEY = "latency_degraded_mode"
RECOVERY_STREAK_KEY = "latency_recovery_streak"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def evaluate_latency_policy(
    state_store: StateStore,
    now: datetime,
) -> LatencyPolicyDecision:
    window_size = max(int(os.getenv("LATENCY_SLA_WINDOW", "50")), 1)
    degrade_p95_ms = max(float(os.getenv("LATENCY_DEGRADED_P95_MS", "8000")), 1.0)
    recover_p95_ms = max(float(os.getenv("LATENCY_RECOVERY_P95_MS", "6000")), 1.0)
    required_recovery_windows = max(int(os.getenv("LATENCY_RECOVERY_WINDOWS", "2")), 1)
    min_samples = max(int(os.getenv("LATENCY_MIN_SAMPLES", "5")), 1)
    normal_slippage = max(float(os.getenv("LATENCY_NORMAL_MAX_SLIPPAGE_BPS", "20")), 1.0)
    degraded_slippage = max(float(os.getenv("LATENCY_DEGRADED_MAX_SLIPPAGE_BPS", "8")), 1.0)

    samples = state_store.get_recent_signal_to_fill_ms(limit=window_size)
    p95 = _percentile(samples, 95.0)
    sample_size = len(samples)

    previous_degraded = bool(state_store.get_system_flag(DEGRADED_FLAG_KEY, False))
    previous_recovery_streak = int(state_store.get_system_flag(RECOVERY_STREAK_KEY, 0) or 0)

    degraded = previous_degraded
    recovery_streak = previous_recovery_streak
    reason = "normal"

    if sample_size < min_samples or p95 is None:
        if previous_degraded:
            reason = "degraded_hold_insufficient_samples"
        else:
            reason = "insufficient_samples"
    elif previous_degraded:
        if p95 <= recover_p95_ms:
            recovery_streak += 1
            reason = "degraded_recovery_window"
            if recovery_streak >= required_recovery_windows:
                degraded = False
                recovery_streak = 0
                reason = "degraded_recovered"
        else:
            recovery_streak = 0
            reason = "degraded_still_above_recovery"
    else:
        if p95 > degrade_p95_ms:
            degraded = True
            recovery_streak = 0
            reason = "p95_signal_to_fill_breach"
        else:
            reason = "normal"

    state_store.set_system_flag(DEGRADED_FLAG_KEY, degraded)
    state_store.set_system_flag(RECOVERY_STREAK_KEY, recovery_streak)

    decision = LatencyPolicyDecision(
        degraded_mode=degraded,
        reason_code=reason,
        effective_slippage_bps=min(normal_slippage, degraded_slippage)
        if degraded
        else normal_slippage,
        p95_signal_to_fill_ms=float(p95) if p95 is not None else None,
        sample_size=sample_size,
        recovery_streak=recovery_streak,
        evaluated_at=now,
    )
    state_store.set_system_flag("latency_policy_last", decision.model_dump(mode="json"))
    return decision
