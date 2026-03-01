#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
import uuid
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.analysis_cache import AnalysisCache
from gold_trading_one_trade_per_day.benchmark_models import (
    parse_models_arg,
    run_model_benchmark,
)
from gold_trading_one_trade_per_day.calendar_service import CalendarService, DailyCalendar
from gold_trading_one_trade_per_day.crew import GoldTradingOneTradePerDayCrew
from gold_trading_one_trade_per_day.event_calendar import EventCalendar
from gold_trading_one_trade_per_day.event_trigger import (
    build_feature_snapshot,
    should_wake_ai,
)
from gold_trading_one_trade_per_day.execution_service import ExecutionContext, ExecutionService
from gold_trading_one_trade_per_day.historian import Historian
from gold_trading_one_trade_per_day.latency_policy import evaluate_latency_policy
from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor
from gold_trading_one_trade_per_day.quota_guard import (
    QuotaGuard,
    call_with_rate_limit_backoff,
)
from gold_trading_one_trade_per_day.recovery import reconcile_startup
from gold_trading_one_trade_per_day.risk_engine import RiskEngine
from gold_trading_one_trade_per_day.schemas import (
    DataSource,
    EventBriefingReport,
    IntentState,
    MarketSentimentReport,
    StrategyIntent,
    TransitionEvent,
)
from gold_trading_one_trade_per_day.state_store import StateStore
from gold_trading_one_trade_per_day.tools.alpaca_tools import (
    fetch_account_equity,
    fetch_latest_quote,
    fetch_macro_proxy_returns,
    fetch_recent_bars,
    get_trading_client,
    has_real_credentials,
)
from gold_trading_one_trade_per_day.warmup import run_warmup, warmup_is_recent_and_passed
from gold_trading_one_trade_per_day.watchdog import Watchdog

NY_TZ = ZoneInfo("America/New_York")
_STREAM_SENSOR: MarketStreamSensor | None = None
_LAST_REST_FALLBACK_AT: datetime | None = None


def _now_ny() -> datetime:
    return datetime.now(tz=NY_TZ)


def determine_trading_mode(calendar: DailyCalendar, now: datetime) -> str:
    """Return "spike" if within a news event window, "daily_scalp" otherwise.

    Spike mode activates SPIKE_PRE_EVENT_MINUTES before a high-impact event and
    stays active for SPIKE_POST_EVENT_MINUTES after the release time.
    """
    if not _env_true("SPIKE_MODE_ENABLED", "true"):
        return "daily_scalp"
    if not calendar.has_high_impact:
        return "daily_scalp"

    pre_minutes = int(os.getenv("SPIKE_PRE_EVENT_MINUTES", "5"))
    post_minutes = int(os.getenv("SPIKE_POST_EVENT_MINUTES", "30"))

    now_utc = now.astimezone(timezone.utc)
    for event in calendar.events:
        if event.impact != "high":
            continue
        release = event.release_time.astimezone(timezone.utc)
        window_start = release - timedelta(minutes=pre_minutes)
        window_end = release + timedelta(minutes=post_minutes)
        if window_start <= now_utc <= window_end:
            return "spike"
    return "daily_scalp"


def _extract_task_model(task_output, model_type):
    if task_output.pydantic:
        return model_type.model_validate(task_output.pydantic.model_dump())
    if task_output.json_dict:
        return model_type.model_validate(task_output.json_dict)
    return model_type.model_validate_json(task_output.raw)


def _extract_successful_requests(crew_output, default_estimate: int) -> int:
    try:
        usage = getattr(crew_output, "token_usage", None)
        if usage is not None:
            val = int(getattr(usage, "successful_requests", 0))
            if val > 0:
                return val
    except Exception:
        pass
    return max(default_estimate, 0)


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _effective_max_spread(mode: str, configured_max_spread: float) -> float:
    if mode != "paper":
        return float(configured_max_spread)
    raw = os.getenv("PAPER_MAX_SPREAD")
    if raw is None or raw.strip() == "":
        return 0.08
    try:
        return max(float(raw), 0.0)
    except Exception:
        return 0.08


def _utc_iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_strategy_intent_timestamps(
    intent: StrategyIntent,
    now: datetime,
    ttl_seconds: int,
) -> StrategyIntent:
    now_utc = now.astimezone(timezone.utc)
    ttl = max(int(ttl_seconds), 1)
    return intent.model_copy(
        update={
            "generated_at": now_utc,
            "expires_at": now_utc + timedelta(seconds=ttl),
        }
    )


def _is_mock_allowed(mode: str) -> bool:
    if mode == "shadow":
        return True
    return not has_real_credentials()


def _get_stream_sensor() -> MarketStreamSensor:
    global _STREAM_SENSOR
    if _STREAM_SENSOR is None:
        _STREAM_SENSOR = MarketStreamSensor(symbol="XAU_USD")
        _STREAM_SENSOR.start()
    else:
        _STREAM_SENSOR.start()
    return _STREAM_SENSOR


def _build_snapshot(
    mode: str,
    calendar: EventCalendar,
) -> tuple[object | None, dict]:
    global _LAST_REST_FALLBACK_AT

    now = _now_ny()
    allow_mock = _is_mock_allowed(mode)
    fallback_interval_sec = int(os.getenv("REST_FALLBACK_INTERVAL_SECONDS", "15"))

    sensor = _get_stream_sensor()
    stream_inputs = sensor.latest_inputs()
    stream_health = sensor.health()

    data_source = DataSource.REST_FALLBACK
    fallback_used = False

    if stream_inputs and not stream_health.stale and stream_health.has_bar and stream_health.has_quote:
        bars, bid, ask, health = stream_inputs
        data_source = DataSource.STREAM
        data_age_sec = health.data_age_sec
    else:
        if _LAST_REST_FALLBACK_AT and (now - _LAST_REST_FALLBACK_AT) < timedelta(seconds=fallback_interval_sec):
            return None, {
                "status": "skipped",
                "reason": "fallback_throttled",
                "context": {
                    "stream_connected": stream_health.connected,
                    "stream_stale": stream_health.stale,
                    "stream_age_sec": stream_health.data_age_sec,
                },
            }

        fallback_used = True
        _LAST_REST_FALLBACK_AT = now
        try:
            bars = fetch_recent_bars(symbol="XAU_USD", lookback_minutes=180, allow_mock=allow_mock)
            bid, ask = fetch_latest_quote(symbol="XAU_USD", allow_mock=allow_mock)
            data_source = DataSource.MOCK if allow_mock and not has_real_credentials() else DataSource.REST_FALLBACK
            data_age_sec = stream_health.data_age_sec if stream_health.last_msg_at else 0.0
        except Exception:
            return None, {
                "status": "skipped",
                "reason": "stale_data",
                "context": {
                    "stream_connected": stream_health.connected,
                    "stream_stale": stream_health.stale,
                    "stream_age_sec": stream_health.data_age_sec,
                    "fallback_failed": True,
                },
            }

    macro = fetch_macro_proxy_returns(allow_mock=allow_mock)
    snapshot = build_feature_snapshot(
        bars=bars,
        bid=bid,
        ask=ask,
        macro_proxies=macro,
        timestamp=now,
        symbol="XAU_USD",
        data_source=data_source,
        data_age_sec=data_age_sec,
    )

    blocked, macro_label = calendar.is_blocked(at=now)
    return snapshot, {
        "fallback_used": fallback_used,
        "stream_health": {
            "connected": stream_health.connected,
            "thread_alive": stream_health.thread_alive,
            "stale": stream_health.stale,
            "data_age_sec": stream_health.data_age_sec,
            "has_bar": stream_health.has_bar,
            "has_quote": stream_health.has_quote,
        },
        "macro_event_active": blocked,
        "macro_event_label": macro_label,
    }


def _kickoff_with_backoff(
    kickoff_fn,
    deadline: datetime,
) -> object:
    base_wait = float(os.getenv("LLM_BACKOFF_BASE_SEC", "2"))
    max_wait = float(os.getenv("LLM_BACKOFF_MAX_SEC", "20"))
    simulate_429_attempts = max(int(os.getenv("SIMULATE_429_ATTEMPTS", "0")), 0)
    attempt_counter = {"value": 0}

    def _wrapped_call():
        if attempt_counter["value"] < simulate_429_attempts:
            attempt_counter["value"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED (simulated)")
        return kickoff_fn()

    return call_with_rate_limit_backoff(
        func=_wrapped_call,
        deadline=deadline,
        base_wait_sec=base_wait,
        max_wait_sec=max_wait,
        now_fn=_now_ny,
    )


def _run_cycle(
    mode: str,
    force_trigger_once: bool = False,
    force_trigger_source: str = "env",
) -> dict:
    env_force_trigger_once = _env_true("FORCE_TRIGGER_ONCE", "false")
    force_requested_any = bool(force_trigger_once or env_force_trigger_once)
    if mode == "live" and os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
        return {
            "status": "blocked",
            "reason": "live_mode_disabled",
            "detail": "Set ENABLE_LIVE_TRADING=true to allow live mode.",
            "force_trigger_ignored_live": force_requested_any,
        }

    force_requested = force_requested_any
    force_source = force_trigger_source if force_trigger_once else ("env" if env_force_trigger_once else force_trigger_source)
    force_ignored_live = mode == "live" and force_requested
    if force_ignored_live:
        force_requested = False

    state_store = StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    risk_engine = RiskEngine()
    effective_max_spread = _effective_max_spread(mode, risk_engine.config.max_spread)
    llm_runner = GoldTradingOneTradePerDayCrew(mode=mode)
    llm_settings = llm_runner.runtime_settings()
    quota_guard = QuotaGuard(state_store=state_store)
    analysis_cache = AnalysisCache(state_store=state_store)
    latency_policy = evaluate_latency_policy(state_store=state_store, now=_now_ny())

    print(
        "[RISK_CONFIG] "
        f"source={risk_engine.config_source} "
        f"soft={risk_engine.config.daily_soft_lock_pct:.4f} "
        f"hard={risk_engine.config.daily_hard_lock_pct:.4f} "
        f"loss={risk_engine.config.daily_loss_hard_lock_pct:.4f} "
        f"max_spread_effective={effective_max_spread:.4f}"
    )
    print(
        "[LLM_CONFIG] "
        f"source={llm_settings['source']} mode={llm_settings['mode']} "
        f"analyst={llm_settings['analyst_model']} strategy={llm_settings['strategy_model']} "
        f"strategy_auto_overridden={llm_settings['strategy_auto_overridden']} "
        f"crew_max_rpm={llm_settings['crew_max_rpm']} agent_max_rpm={llm_settings['agent_max_rpm']} "
        f"timeout={llm_settings['timeout_sec']} retries={llm_settings['max_retries']}"
    )

    trading_client = get_trading_client() if mode in {"paper", "live"} else None
    execution_service = ExecutionService(
        state_store=state_store,
        trading_client=trading_client,
    )
    watchdog = Watchdog(
        state_store=state_store,
        execution_service=execution_service,
    )

    warmup_enforced = os.getenv("REQUIRE_WARMUP_PASS", "false").lower() == "true"
    if mode in {"paper", "live"} and warmup_enforced:
        warmup_ok, warmup_context = warmup_is_recent_and_passed(
            state_store=state_store,
            now=_now_ny(),
        )
        if not warmup_ok:
            return {
                "status": "blocked",
                "reason": "warmup_gate_failed",
                "context": warmup_context,
            }

    today = _now_ny().date().isoformat()
    allow_mock = _is_mock_allowed(mode)
    try:
        equity = fetch_account_equity(allow_mock=allow_mock)
    except Exception:
        return {
            "status": "skipped",
            "reason": "account_equity_unavailable",
        }

    day_state = state_store.get_or_create_day_state(today, starting_equity=equity)
    day_state = risk_engine.update_locks(day_state, current_equity=equity)
    state_store.upsert_day_state(day_state)

    recovery_summary = reconcile_startup(state_store, trading_client, day_state)

    watchdog_result = watchdog.evaluate_and_act(
        mode=mode,
        latency_policy=latency_policy,
        now=_now_ny(),
    )
    if watchdog_result.halted:
        return {
            "status": "halted",
            "reason": watchdog_result.reason_code,
            "watchdog": watchdog_result.details,
            "recovery": recovery_summary,
        }

    if day_state.hard_lock:
        if mode in {"paper", "live"}:
            execution_service.flatten_all_positions()
        return {
            "status": "halted",
            "reason": day_state.last_lock_reason,
            "recovery": recovery_summary,
            "equity_change_pct": day_state.equity_change_pct,
        }

    calendar = EventCalendar()
    snapshot, snapshot_meta = _build_snapshot(mode=mode, calendar=calendar)
    if snapshot is None:
        return snapshot_meta

    # Determine trading mode (spike vs daily_scalp) using economic calendar service
    calendar_service = CalendarService()
    try:
        daily_calendar = calendar_service.get_calendar(now=_now_ny().astimezone(timezone.utc))
    except Exception:
        daily_calendar = None

    trading_mode = "daily_scalp"
    active_event = None
    if daily_calendar is not None:
        trading_mode = determine_trading_mode(daily_calendar, _now_ny())
        if trading_mode == "spike":
            active_event = daily_calendar.next_event

    if not _env_true("DAILY_SCALP_ENABLED", "true") and trading_mode == "daily_scalp":
        return {
            "status": "skipped",
            "reason": "daily_scalp_disabled",
            "trading_mode": trading_mode,
        }

    # Apply mode-specific risk parameters
    if trading_mode == "spike":
        spike_risk_pct = float(os.getenv("SPIKE_RISK_PER_TRADE_PCT", "0.03"))
        risk_engine.config.per_trade_risk_pct = spike_risk_pct
    else:
        scalp_risk_pct_raw = os.getenv("DAILY_SCALP_RISK_PER_TRADE_PCT")
        if scalp_risk_pct_raw:
            risk_engine.config.per_trade_risk_pct = float(scalp_risk_pct_raw)

    event_id = str(uuid.uuid4())
    state_store.record_event(event_id=event_id, symbol="XAU_USD", snapshot=snapshot.model_dump(mode="json"))
    state_store.upsert_latency_metric(
        event_id=event_id,
        event_detected_at=_now_ny(),
        degraded_mode=latency_policy.degraded_mode,
        metadata_patch={
            "mode": mode,
            "trading_mode": trading_mode,
            "latency_policy_reason": latency_policy.reason_code,
            "latency_policy_p95_ms": latency_policy.p95_signal_to_fill_ms,
            "latency_policy_window_size": latency_policy.sample_size,
            "force_trigger_requested": force_requested or force_ignored_live,
            "force_trigger_source": force_source if (force_requested or force_ignored_live) else None,
            "force_trigger_ignored_live": force_ignored_live,
        },
    )

    # Feature 7: post-spike confirmation delay — wait before running agent pipeline
    if trading_mode == "spike":
        spike_confirmation_delay = int(os.getenv("SPIKE_CONFIRMATION_DELAY_SEC", "30"))
        if spike_confirmation_delay > 0:
            time_module.sleep(spike_confirmation_delay)
            # Re-fetch snapshot for fresh prices after the delay
            snapshot, snapshot_meta = _build_snapshot(mode=mode, calendar=calendar)
            if snapshot is None:
                return snapshot_meta

    triggered, reason, trigger_context = should_wake_ai(
        snapshot,
        day_state,
        max_spread=effective_max_spread,
        open_warmup_minutes=int(os.getenv("OPEN_WARMUP_MINUTES", "5")),
        macro_event_active=bool(snapshot_meta.get("macro_event_active")),
        macro_event_label=snapshot_meta.get("macro_event_label"),
        trading_mode=trading_mode,
    )
    force_trigger_applied = False
    force_trigger_original_reason: str | None = None
    if not triggered and force_requested and mode in {"shadow", "paper"}:
        force_trigger_applied = True
        force_trigger_original_reason = reason
        reason = "force_trigger_once"
        triggered = True

    state_store.update_event(
        event_id,
        status="triggered" if triggered else f"skipped:{reason}",
        snapshot_patch={
            "skip_reason_code": None if triggered else reason,
            "skip_context": trigger_context,
            "fallback_used": snapshot_meta.get("fallback_used", False),
            "stream_health": snapshot_meta.get("stream_health", {}),
            "macro_event_active": snapshot_meta.get("macro_event_active", False),
            "macro_event_label": snapshot_meta.get("macro_event_label"),
            "trading_mode": trading_mode,
            "llm_route": {
                "analyst_model": llm_settings["analyst_model"],
                "strategy_model": llm_settings["strategy_model"],
                "analyst_route": llm_settings["analyst_route"],
                "strategy_route": llm_settings["strategy_route"],
                "analyst_fallback_used": llm_settings["analyst_fallback_used"],
            },
            "latency_degraded_mode": latency_policy.degraded_mode,
            "latency_policy_reason": latency_policy.reason_code,
            "latency_policy_p95_ms": latency_policy.p95_signal_to_fill_ms,
            "latency_policy_slippage_bps": latency_policy.effective_slippage_bps,
            "effective_max_spread": effective_max_spread,
            "force_trigger_applied": force_trigger_applied,
            "force_trigger_source": force_source if (force_trigger_applied or force_ignored_live) else None,
            "force_trigger_original_reason": force_trigger_original_reason,
            "force_trigger_mode": mode if force_trigger_applied else None,
            "force_trigger_ignored_live": force_ignored_live,
        },
    )
    state_store.upsert_latency_metric(
        event_id=event_id,
        metadata_patch={
            "force_trigger_applied": force_trigger_applied,
            "force_trigger_original_reason": force_trigger_original_reason,
            "force_trigger_ignored_live": force_ignored_live,
        },
    )

    if not triggered:
        return {
            "status": "skipped",
            "reason": reason,
            "event_id": event_id,
            "context": trigger_context,
            "fallback_used": snapshot_meta.get("fallback_used", False),
            "recovery": recovery_summary,
            "force_trigger_applied": False,
            "force_trigger_ignored_live": force_ignored_live,
        }

    cache_key, cached_report = analysis_cache.get(
        snapshot=snapshot,
        model_name=llm_settings["analyst_model"],
        now=_now_ny(),
    )
    analysis_cache_hit = cached_report is not None

    if (
        mode in {"paper", "live"}
        and not analysis_cache_hit
        and llm_settings["analyst_route"] == "local"
        and not llm_settings["analyst_local_available"]
        and not llm_settings["analyst_fallback_used"]
    ):
        state_store.update_event(
            event_id,
            status="skipped:analyst_model_unavailable",
            snapshot_patch={
                "skip_reason_code": "analyst_model_unavailable",
                "analysis_cache_hit": False,
            },
        )
        return {
            "status": "skipped",
            "reason": "analyst_model_unavailable",
            "event_id": event_id,
            "force_trigger_applied": force_trigger_applied,
        }

    estimated_requests = 1 if analysis_cache_hit else (3 if trading_mode == "spike" else 2)
    reservation = quota_guard.reserve(
        event_id=event_id,
        estimated_requests=estimated_requests,
        source="llm_preflight",
        now=_now_ny(),
    )
    if not reservation.allowed:
        state_store.update_event(
            event_id,
            status="skipped:quota_budget_exceeded",
            snapshot_patch={
                "skip_reason_code": "quota_budget_exceeded",
                "analysis_cache_hit": analysis_cache_hit,
                "quota_rpm_remaining": reservation.snapshot.rpm_remaining,
                "quota_rpd_remaining": reservation.snapshot.rpd_remaining,
                "retry_count": 0,
                "retry_total_wait_sec": 0.0,
            },
        )
        return {
            "status": "skipped",
            "reason": "quota_budget_exceeded",
            "event_id": event_id,
            "analysis_cache_hit": analysis_cache_hit,
            "quota": {
                "rpm_remaining": reservation.snapshot.rpm_remaining,
                "rpd_remaining": reservation.snapshot.rpd_remaining,
            },
            "force_trigger_applied": force_trigger_applied,
        }

    intent_ttl_seconds = int(os.getenv("INTENT_TTL_SECONDS", "45"))
    now_for_prompt = _now_ny()
    # Build event data JSON for spike mode
    event_data_json = "{}"
    event_briefing_json = "{}"
    if trading_mode == "spike" and active_event is not None:
        event_data_json = active_event.model_dump_json()
    inputs = {
        "event_id": event_id,
        "feature_snapshot_json": snapshot.model_dump_json(),
        "symbol": "XAU_USD",
        "current_utc_iso": _utc_iso(now_for_prompt),
        "intent_ttl_seconds": intent_ttl_seconds,
        "event_data_json": event_data_json,
        "event_briefing_json": event_briefing_json,
        "trading_mode": trading_mode,
    }
    if analysis_cache_hit and cached_report is not None:
        inputs["cached_market_report_json"] = cached_report.model_dump_json()

    deadline = _now_ny() + timedelta(seconds=intent_ttl_seconds)
    llm_start_at = _now_ny()
    state_store.upsert_latency_metric(
        event_id=event_id,
        llm_start_at=llm_start_at,
        metadata_patch={
            "analysis_cache_hit": analysis_cache_hit,
            "trading_mode": trading_mode,
        },
    )
    try:
        if analysis_cache_hit:
            backoff_result = _kickoff_with_backoff(
                kickoff_fn=lambda: llm_runner.kickoff_strategy_only(inputs),
                deadline=deadline,
            )
        elif trading_mode == "spike":
            backoff_result = _kickoff_with_backoff(
                kickoff_fn=lambda: llm_runner.kickoff_spike_mode(inputs),
                deadline=deadline,
            )
        else:
            backoff_result = _kickoff_with_backoff(
                kickoff_fn=lambda: llm_runner.crew().kickoff(inputs=inputs),
                deadline=deadline,
            )
    except Exception as exc:
        state_store.upsert_latency_metric(
            event_id=event_id,
            llm_end_at=_now_ny(),
            metadata_patch={
                "llm_error": str(exc),
                "retry_count": 0,
                "retry_total_wait_sec": 0.0,
            },
        )
        quota_snapshot = quota_guard.commit(
            event_id=event_id,
            used_requests=0,
            source="llm_error",
            now=_now_ny(),
        )
        state_store.update_event(
            event_id,
            status="error:llm_inference_failed",
            snapshot_patch={
                "analysis_cache_hit": analysis_cache_hit,
                "retry_count": 0,
                "retry_total_wait_sec": 0.0,
                "quota_rpm_remaining": quota_snapshot.rpm_remaining,
                "quota_rpd_remaining": quota_snapshot.rpd_remaining,
                "llm_error": str(exc),
            },
        )
        return {
            "status": "error",
            "reason": "llm_inference_failed",
            "event_id": event_id,
            "force_trigger_applied": force_trigger_applied,
        }

    if backoff_result.exhausted or backoff_result.value is None:
        state_store.upsert_latency_metric(
            event_id=event_id,
            llm_end_at=_now_ny(),
            metadata_patch={
                "retry_count": backoff_result.retry_count,
                "retry_total_wait_sec": backoff_result.total_wait_sec,
                "llm_last_error": backoff_result.last_error,
            },
        )
        quota_snapshot = quota_guard.commit(
            event_id=event_id,
            used_requests=0,
            source="llm_retry_exhausted",
            now=_now_ny(),
        )
        state_store.update_event(
            event_id,
            status="skipped:rate_limit_retry_exhausted",
            snapshot_patch={
                "skip_reason_code": "rate_limit_retry_exhausted",
                "analysis_cache_hit": analysis_cache_hit,
                "retry_count": backoff_result.retry_count,
                "retry_total_wait_sec": backoff_result.total_wait_sec,
                "quota_rpm_remaining": quota_snapshot.rpm_remaining,
                "quota_rpd_remaining": quota_snapshot.rpd_remaining,
                "llm_last_error": backoff_result.last_error,
            },
        )
        return {
            "status": "skipped",
            "reason": "rate_limit_retry_exhausted",
            "event_id": event_id,
            "retry_count": backoff_result.retry_count,
            "retry_total_wait_sec": backoff_result.total_wait_sec,
            "force_trigger_applied": force_trigger_applied,
        }

    crew_output = backoff_result.value
    state_store.upsert_latency_metric(
        event_id=event_id,
        llm_end_at=_now_ny(),
        metadata_patch={
            "retry_count": backoff_result.retry_count,
            "retry_total_wait_sec": backoff_result.total_wait_sec,
        },
    )
    successful_requests = _extract_successful_requests(
        crew_output,
        default_estimate=estimated_requests,
    )
    quota_snapshot = quota_guard.commit(
        event_id=event_id,
        used_requests=successful_requests,
        source="llm_success",
        now=_now_ny(),
    )

    if analysis_cache_hit:
        if len(crew_output.tasks_output) < 1:
            state_store.update_event(event_id, status="error:crew_output_missing_tasks")
            return {
                "status": "error",
                "reason": "crew_output_missing_tasks",
                "event_id": event_id,
            }
        strategy_intent = _extract_task_model(
            crew_output.tasks_output[0], StrategyIntent
        )
        assert cached_report is not None
        market_report = cached_report.model_copy(
            update={
                "report_id": str(uuid.uuid4()),
                "generated_at": _now_ny(),
            }
        )
    elif trading_mode == "spike":
        # Spike mode: 3-task pipeline — briefing(0), sentiment(1), intent(2)
        if len(crew_output.tasks_output) < 3:
            state_store.update_event(event_id, status="error:crew_output_missing_tasks")
            return {
                "status": "error",
                "reason": "crew_output_missing_tasks",
                "event_id": event_id,
            }
        _extract_task_model(crew_output.tasks_output[0], EventBriefingReport)
        market_report = _extract_task_model(
            crew_output.tasks_output[1], MarketSentimentReport
        )
        strategy_intent = _extract_task_model(
            crew_output.tasks_output[2], StrategyIntent
        )
        analysis_cache.put(
            cache_key=cache_key,
            model_name=llm_settings["analyst_model"],
            report=market_report,
            now=_now_ny(),
        )
    else:
        if len(crew_output.tasks_output) < 2:
            state_store.update_event(event_id, status="error:crew_output_missing_tasks")
            return {
                "status": "error",
                "reason": "crew_output_missing_tasks",
                "event_id": event_id,
            }

        market_report = _extract_task_model(
            crew_output.tasks_output[0], MarketSentimentReport
        )
        strategy_intent = _extract_task_model(
            crew_output.tasks_output[1], StrategyIntent
        )
        analysis_cache.put(
            cache_key=cache_key,
            model_name=llm_settings["analyst_model"],
            report=market_report,
            now=_now_ny(),
        )

    original_generated_at = strategy_intent.generated_at
    original_expires_at = strategy_intent.expires_at
    strategy_intent = _normalize_strategy_intent_timestamps(
        intent=strategy_intent,
        now=_now_ny(),
        ttl_seconds=intent_ttl_seconds,
    )

    state_store.update_event(
        event_id,
        status="analysis_complete",
        snapshot_patch={
            "analysis_cache_hit": analysis_cache_hit,
            "analysis_cache_key": cache_key,
            "retry_count": backoff_result.retry_count,
            "retry_total_wait_sec": backoff_result.total_wait_sec,
            "quota_rpm_remaining": quota_snapshot.rpm_remaining,
            "quota_rpd_remaining": quota_snapshot.rpd_remaining,
            "llm_successful_requests": successful_requests,
            "strategy_intent_time_overridden": True,
            "strategy_intent_original_generated_at": original_generated_at.isoformat(),
            "strategy_intent_original_expires_at": original_expires_at.isoformat(),
            "strategy_intent_normalized_generated_at": strategy_intent.generated_at.isoformat(),
            "strategy_intent_normalized_expires_at": strategy_intent.expires_at.isoformat(),
        },
    )

    state_store.record_analysis(market_report, event_id=event_id)
    state_store.record_intent(
        strategy_intent,
        event_id=event_id,
        state=IntentState.INTENT_GENERATED,
    )
    state_store.upsert_latency_metric(
        event_id=event_id,
        intent_id=strategy_intent.intent_id,
        metadata_patch={
            "strategy_intent_time_overridden": True,
        },
    )

    decision = risk_engine.evaluate(
        strategy_intent,
        snapshot,
        day_state,
        now=_now_ny(),
        max_spread_override=effective_max_spread,
    )
    state_store.record_risk_decision(decision)

    if not decision.approved:
        state_store.mark_intent_terminal(
            intent_id=strategy_intent.intent_id,
            terminal_state=IntentState.DENIED,
            reason=decision.reason_code,
        )
        state_store.update_event(event_id, status=f"denied:{decision.reason_code}")
        state_store.upsert_latency_metric(
            event_id=event_id,
            metadata_patch={
                "risk_denied_reason": decision.reason_code,
            },
        )
        return {
            "status": "denied",
            "reason": decision.reason_code,
            "intent_id": strategy_intent.intent_id,
            "event_id": event_id,
        }

    state_store.transition(
        TransitionEvent(
            intent_id=strategy_intent.intent_id,
            from_state=IntentState.INTENT_GENERATED,
            to_state=IntentState.RISK_APPROVED,
            at=_now_ny(),
            metadata={"reason": decision.reason_code},
        )
    )
    state_store.upsert_latency_metric(
        event_id=event_id,
        risk_approved_at=_now_ny(),
    )

    command = risk_engine.build_execution_command(strategy_intent, decision)
    if latency_policy.degraded_mode:
        command = command.model_copy(
            update={
                "max_slippage_bps": min(
                    float(command.max_slippage_bps),
                    float(latency_policy.effective_slippage_bps),
                )
            }
        )

    if mode == "shadow":
        state_store.mark_intent_terminal(
            intent_id=strategy_intent.intent_id,
            terminal_state=IntentState.HALTED,
            reason="shadow_mode_no_execution",
        )
        state_store.update_event(event_id, status="shadow_only")
        return {
            "status": "shadow_only",
            "event_id": event_id,
            "intent_id": strategy_intent.intent_id,
            "command_preview": command.model_dump(mode="json"),
            "fallback_used": snapshot_meta.get("fallback_used", False),
            "recovery": recovery_summary,
            "analysis_cache_hit": analysis_cache_hit,
            "latency_policy": latency_policy.model_dump(mode="json"),
            "force_trigger_applied": force_trigger_applied,
            "force_trigger_original_reason": force_trigger_original_reason,
        }

    report = execution_service.execute(
        ExecutionContext(
            intent=strategy_intent,
            command=command,
            latency_policy=latency_policy,
        )
    )
    state_store.upsert_latency_metric(
        event_id=event_id,
        entry_submitted_at=_parse_iso_ts(report.timestamps.get("submitted_at")),
        entry_filled_at=_parse_iso_ts(report.timestamps.get("filled_at")),
        metadata_patch={
            "execution_status": report.status,
            "execution_reject_reason": report.reject_reason,
        },
    )

    if report.status == "Executed":
        day_state.entries_taken += 1
        state_store.upsert_day_state(day_state)

        state_store.transition(
            TransitionEvent(
                intent_id=strategy_intent.intent_id,
                from_state=IntentState.RISK_APPROVED,
                to_state=IntentState.ENTRY_SUBMITTED,
                at=_now_ny(),
                metadata={"entry_order_id": report.entry_order_id},
            )
        )
        state_store.transition(
            TransitionEvent(
                intent_id=strategy_intent.intent_id,
                from_state=IntentState.ENTRY_SUBMITTED,
                to_state=IntentState.ENTRY_FILLED,
                at=_now_ny(),
                metadata={"avg_fill_price": report.avg_fill_price},
            )
        )
        state_store.transition(
            TransitionEvent(
                intent_id=strategy_intent.intent_id,
                from_state=IntentState.ENTRY_FILLED,
                to_state=IntentState.OCO_SUBMITTED,
                at=_now_ny(),
                metadata={"oco_order_id": report.oco_order_id},
            )
        )
        state_store.update_event(event_id, status="executed")
    else:
        terminal = IntentState.CANCELLED if report.status in {"Cancelled", "Not_Executed"} else IntentState.ERROR
        state_store.mark_intent_terminal(
            intent_id=strategy_intent.intent_id,
            terminal_state=terminal,
            reason=report.reject_reason or report.status,
        )
        state_store.update_event(event_id, status=f"{report.status}:{report.reject_reason or ''}")

    if _now_ny().time() >= time(15, 55):
        execution_service.flatten_all_positions()

    return {
        "status": report.status,
        "event_id": event_id,
        "intent_id": strategy_intent.intent_id,
        "trading_mode": trading_mode,
        "fallback_used": snapshot_meta.get("fallback_used", False),
        "execution": report.model_dump(mode="json"),
        "recovery": recovery_summary,
        "analysis_cache_hit": analysis_cache_hit,
        "latency_policy": latency_policy.model_dump(mode="json"),
        "force_trigger_applied": force_trigger_applied,
        "force_trigger_original_reason": force_trigger_original_reason,
    }


def run_shadow() -> None:
    print(json.dumps(_run_cycle("shadow"), indent=2))


def run_auto() -> None:
    from gold_trading_one_trade_per_day.scheduler import run_auto_scheduler
    args = _commandless_args()
    mode = args[0] if args else os.getenv("AUTO_MODE", "shadow")
    run_auto_scheduler(mode=mode)


def run_shadow_loop() -> None:
    args = _commandless_args()
    iterations = int(args[0]) if args else int(os.getenv("SHADOW_LOOP_ITERATIONS", "5"))
    sleep_sec = float(os.getenv("SHADOW_LOOP_SLEEP_SEC", "1"))
    force_env_requested = _env_true("FORCE_TRIGGER_ONCE", "false")
    force_consumed = False

    for idx in range(iterations):
        force_this_cycle = force_env_requested and not force_consumed
        payload = _run_cycle(
            "shadow",
            force_trigger_once=force_this_cycle,
            force_trigger_source="env",
        )
        if bool(payload.get("force_trigger_applied")):
            force_consumed = True
        payload["iteration"] = idx + 1
        print(json.dumps(payload, indent=2))
        if idx < (iterations - 1) and sleep_sec > 0:
            time_module.sleep(sleep_sec)


def run_paper() -> None:
    print(json.dumps(_run_cycle("paper"), indent=2))


def run_live() -> None:
    print(json.dumps(_run_cycle("live"), indent=2))


def run_paper_smoke() -> None:
    payload = _run_cycle(
        "paper",
        force_trigger_once=True,
        force_trigger_source="cli",
    )
    payload["smoke_mode"] = True
    print(json.dumps(payload, indent=2))


def warmup() -> None:
    args = _commandless_args()
    mode = args[0] if args else os.getenv("WARMUP_MODE", "paper")
    report = run_warmup(mode=mode)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    if not report.passed:
        raise SystemExit(1)


def benchmark_models() -> None:
    args = _commandless_args()
    models = parse_models_arg(args[0]) if len(args) >= 1 else None
    repeats = int(args[1]) if len(args) >= 2 else None
    payload = run_model_benchmark(models=models, repeats=repeats)
    print(json.dumps(payload, indent=2))


def resume() -> None:
    state_store = StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    execution_service = ExecutionService(
        state_store=state_store,
        trading_client=get_trading_client(),
    )
    watchdog = Watchdog(state_store=state_store, execution_service=execution_service)
    payload = watchdog.resume(now=_now_ny())
    print(json.dumps(payload, indent=2))


def reconcile() -> None:
    state_store = StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    day = _now_ny().date().isoformat()
    historian = Historian(state_store)
    metrics = historian.generate_daily_metrics(day)
    print(json.dumps(metrics, indent=2))


def run() -> None:
    run_shadow()


def run_with_trigger() -> None:
    run_paper_smoke()


def _commandless_args() -> list[str]:
    args = sys.argv[1:]
    known_commands = {
        "run",
        "run_shadow",
        "run_shadow_loop",
        "run_paper",
        "run_paper_smoke",
        "run_live",
        "run_auto",
        "warmup",
        "benchmark_models",
        "resume",
        "reconcile",
        "train",
        "replay",
        "test",
        "run_with_trigger",
    }
    if args and args[0] in known_commands:
        return args[1:]
    return args


def train() -> None:
    args = _commandless_args()
    if len(args) < 2:
        raise ValueError("Usage: train <iterations> <filename>")
    inputs = {
        "event_id": "sample-event",
        "feature_snapshot_json": "{}",
        "symbol": "XAU_USD",
    }
    GoldTradingOneTradePerDayCrew().crew().train(
        n_iterations=int(args[0]),
        filename=args[1],
        inputs=inputs,
    )


def replay() -> None:
    args = _commandless_args()
    if len(args) < 1:
        raise ValueError("Usage: replay <task_id>")
    GoldTradingOneTradePerDayCrew().crew().replay(task_id=args[0])


def test() -> None:
    args = _commandless_args()
    if len(args) < 2:
        raise ValueError("Usage: test <iterations> <model>")
    inputs = {
        "event_id": "sample-event",
        "feature_snapshot_json": "{}",
        "symbol": "XAU_USD",
    }
    crew = GoldTradingOneTradePerDayCrew().crew()
    # CrewAI changed this kwarg from openai_model_name -> eval_llm across releases.
    # Try current signature first, then fall back for older versions.
    try:
        crew.test(
            n_iterations=int(args[0]),
            eval_llm=args[1],
            inputs=inputs,
        )
    except TypeError:
        crew.test(
            n_iterations=int(args[0]),
            openai_model_name=args[1],
            inputs=inputs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="XAUUSD event-driven scalping controller")
    parser.add_argument(
        "command",
        nargs="?",
        default="run_shadow",
        choices=[
            "run",
            "run_shadow",
            "run_shadow_loop",
            "run_paper",
            "run_paper_smoke",
            "run_live",
            "run_auto",
            "warmup",
            "benchmark_models",
            "resume",
            "reconcile",
            "train",
            "replay",
            "test",
            "run_with_trigger",
        ],
    )
    # Use parse_known_args so command-specific positional args (e.g. run_shadow_loop 10)
    # pass through to _commandless_args() handlers without argparse rejecting them.
    args, _ = parser.parse_known_args()

    command = args.command
    if command == "run":
        run()
    elif command == "run_shadow":
        run_shadow()
    elif command == "run_shadow_loop":
        run_shadow_loop()
    elif command == "run_paper":
        run_paper()
    elif command == "run_paper_smoke":
        run_paper_smoke()
    elif command == "run_live":
        run_live()
    elif command == "run_auto":
        run_auto()
    elif command == "warmup":
        warmup()
    elif command == "benchmark_models":
        benchmark_models()
    elif command == "resume":
        resume()
    elif command == "reconcile":
        reconcile()
    elif command == "run_with_trigger":
        run_with_trigger()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()


if __name__ == "__main__":
    main()
