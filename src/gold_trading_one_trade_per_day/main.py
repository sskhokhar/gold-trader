#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.crew import GoldTradingOneTradePerDayCrew
from gold_trading_one_trade_per_day.event_calendar import EventCalendar
from gold_trading_one_trade_per_day.event_trigger import (
    build_feature_snapshot,
    should_wake_ai,
)
from gold_trading_one_trade_per_day.execution_service import ExecutionContext, ExecutionService
from gold_trading_one_trade_per_day.historian import Historian
from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor
from gold_trading_one_trade_per_day.recovery import reconcile_startup
from gold_trading_one_trade_per_day.risk_engine import RiskEngine
from gold_trading_one_trade_per_day.schemas import (
    DataSource,
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

NY_TZ = ZoneInfo("America/New_York")
_STREAM_SENSOR: MarketStreamSensor | None = None
_LAST_REST_FALLBACK_AT: datetime | None = None


def _now_ny() -> datetime:
    return datetime.now(tz=NY_TZ)


def _extract_task_model(task_output, model_type):
    if task_output.pydantic:
        return model_type.model_validate(task_output.pydantic.model_dump())
    if task_output.json_dict:
        return model_type.model_validate(task_output.json_dict)
    return model_type.model_validate_json(task_output.raw)


def _is_mock_allowed(mode: str) -> bool:
    if mode == "shadow":
        return True
    return not has_real_credentials()


def _get_stream_sensor() -> MarketStreamSensor:
    global _STREAM_SENSOR
    if _STREAM_SENSOR is None:
        _STREAM_SENSOR = MarketStreamSensor(symbol="GLD")
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
            bars = fetch_recent_bars(symbol="GLD", lookback_minutes=180, allow_mock=allow_mock)
            bid, ask = fetch_latest_quote(symbol="GLD", allow_mock=allow_mock)
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
        symbol="GLD",
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


def _run_cycle(mode: str) -> dict:
    if mode == "live" and os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
        return {
            "status": "blocked",
            "reason": "live_mode_disabled",
            "detail": "Set ENABLE_LIVE_TRADING=true to allow live mode.",
        }

    state_store = StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    risk_engine = RiskEngine()

    print(
        "[RISK_CONFIG] "
        f"source={risk_engine.config_source} "
        f"soft={risk_engine.config.daily_soft_lock_pct:.4f} "
        f"hard={risk_engine.config.daily_hard_lock_pct:.4f} "
        f"loss={risk_engine.config.daily_loss_hard_lock_pct:.4f}"
    )

    trading_client = get_trading_client() if mode in {"paper", "live"} else None
    execution_service = ExecutionService(
        state_store=state_store,
        trading_client=trading_client,
    )

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

    event_id = str(uuid.uuid4())
    state_store.record_event(event_id=event_id, symbol="GLD", snapshot=snapshot.model_dump(mode="json"))

    triggered, reason, trigger_context = should_wake_ai(
        snapshot,
        day_state,
        max_spread=risk_engine.config.max_spread,
        open_warmup_minutes=int(os.getenv("OPEN_WARMUP_MINUTES", "5")),
        macro_event_active=bool(snapshot_meta.get("macro_event_active")),
        macro_event_label=snapshot_meta.get("macro_event_label"),
    )

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
        }

    inputs = {
        "event_id": event_id,
        "feature_snapshot_json": snapshot.model_dump_json(),
        "symbol": "GLD",
    }
    crew_output = GoldTradingOneTradePerDayCrew().crew().kickoff(inputs=inputs)

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

    state_store.record_analysis(market_report, event_id=event_id)
    state_store.record_intent(
        strategy_intent,
        event_id=event_id,
        state=IntentState.INTENT_GENERATED,
    )

    decision = risk_engine.evaluate(strategy_intent, snapshot, day_state, now=_now_ny())
    state_store.record_risk_decision(decision)

    if not decision.approved:
        state_store.mark_intent_terminal(
            intent_id=strategy_intent.intent_id,
            terminal_state=IntentState.DENIED,
            reason=decision.reason_code,
        )
        state_store.update_event(event_id, status=f"denied:{decision.reason_code}")
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

    command = risk_engine.build_execution_command(strategy_intent, decision)

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
        }

    report = execution_service.execute(
        ExecutionContext(intent=strategy_intent, command=command)
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
        "fallback_used": snapshot_meta.get("fallback_used", False),
        "execution": report.model_dump(mode="json"),
        "recovery": recovery_summary,
    }


def run_shadow() -> None:
    print(json.dumps(_run_cycle("shadow"), indent=2))


def run_paper() -> None:
    print(json.dumps(_run_cycle("paper"), indent=2))


def run_live() -> None:
    print(json.dumps(_run_cycle("live"), indent=2))


def reconcile() -> None:
    state_store = StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    day = _now_ny().date().isoformat()
    historian = Historian(state_store)
    metrics = historian.generate_daily_metrics(day)
    print(json.dumps(metrics, indent=2))


def run() -> None:
    run_shadow()


def run_with_trigger() -> None:
    run_paper()


def _commandless_args() -> list[str]:
    args = sys.argv[1:]
    known_commands = {
        "run",
        "run_shadow",
        "run_paper",
        "run_live",
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
        "symbol": "GLD",
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
        "symbol": "GLD",
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
    parser = argparse.ArgumentParser(description="GLD event-driven scalping controller")
    parser.add_argument(
        "command",
        nargs="?",
        default="run_shadow",
        choices=[
            "run",
            "run_shadow",
            "run_paper",
            "run_live",
            "reconcile",
            "train",
            "replay",
            "test",
            "run_with_trigger",
        ],
    )
    args = parser.parse_args()

    command = args.command
    if command == "run":
        run()
    elif command == "run_shadow":
        run_shadow()
    elif command == "run_paper":
        run_paper()
    elif command == "run_live":
        run_live()
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
