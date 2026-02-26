from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from gold_trading_one_trade_per_day.schemas import LatencyPolicyDecision, StrategyIntent
from gold_trading_one_trade_per_day.state_store import StateStore


@dataclass(slots=True)
class WatchdogResult:
    halted: bool
    reason_code: str
    details: dict[str, Any]


class Watchdog:
    def __init__(self, state_store: StateStore, execution_service) -> None:
        self.state_store = state_store
        self.execution_service = execution_service
        self.loop_threshold = max(int(os.getenv("WATCHDOG_LOOP_THRESHOLD", "3")), 2)
        self.flip_threshold = max(int(os.getenv("WATCHDOG_FLIP_THRESHOLD", "3")), 2)
        self.stuck_grace_sec = max(int(os.getenv("WATCHDOG_STUCK_GRACE_SEC", "8")), 0)
        self.halt_key = "halted_by_watchdog"
        self.flatten_on_halt = os.getenv("WATCHDOG_FLATTEN_ON_HALT", "false").lower() == "true"

    def evaluate_and_act(
        self,
        mode: str,
        latency_policy: LatencyPolicyDecision,
        now: datetime,
    ) -> WatchdogResult:
        existing = self.state_store.get_system_flag(self.halt_key, None)
        if isinstance(existing, dict) and existing.get("halted"):
            return WatchdogResult(
                halted=True,
                reason_code=str(existing.get("reason_code", "halted_by_watchdog")),
                details=existing,
            )

        reason_code, details = self._detect(now=now, latency_policy=latency_policy)
        if reason_code is None:
            return WatchdogResult(halted=False, reason_code="ok", details={})

        action = {
            "cancel_all_open_orders": False,
            "flatten_all_positions": False,
        }

        if mode in {"paper", "live"}:
            cancel_result = self.execution_service.cancel_all_open_orders()
            action["cancel_all_open_orders"] = True
            action["cancel_result"] = cancel_result

            if self.flatten_on_halt:
                flatten_result = self.execution_service.flatten_all_positions()
                action["flatten_all_positions"] = True
                action["flatten_result"] = flatten_result

        payload = {
            "halted": True,
            "reason_code": reason_code,
            "detected_at": now.isoformat(),
            "details": details,
            "action": action,
        }
        self.state_store.set_system_flag(self.halt_key, payload)
        self.state_store.record_watchdog_event(
            reason_code=reason_code,
            action=action,
            metadata=details,
            at=now,
        )
        return WatchdogResult(halted=True, reason_code=reason_code, details=payload)

    def resume(self, now: datetime, note: str = "manual_resume") -> dict[str, Any]:
        payload = {
            "halted": False,
            "resumed_at": now.isoformat(),
            "note": note,
        }
        self.state_store.clear_system_flag(self.halt_key)
        self.state_store.record_watchdog_event(
            reason_code="watchdog_resumed",
            action={"resume": True},
            metadata=payload,
            at=now,
        )
        return payload

    def _detect(
        self,
        now: datetime,
        latency_policy: LatencyPolicyDecision,
    ) -> tuple[str | None, dict[str, Any]]:
        non_terminal = self._detect_non_terminal_loop()
        if non_terminal is not None:
            return "non_terminal_loop", non_terminal

        deny_flip = self._detect_deny_side_flips(now=now)
        if deny_flip is not None:
            return "deny_side_flip_loop", deny_flip

        open_entries = self.state_store.list_open_entry_orders()
        stuck = self._detect_stuck_entry(now=now, open_entries=open_entries)
        if stuck is not None:
            return "stuck_open_entry", stuck

        if latency_policy.degraded_mode and len(open_entries) > 0:
            return "latency_degraded_order_instability", {
                "open_entry_count": len(open_entries),
                "latency_reason": latency_policy.reason_code,
                "p95_signal_to_fill_ms": latency_policy.p95_signal_to_fill_ms,
            }

        return None, {}

    def _detect_non_terminal_loop(self) -> dict[str, Any] | None:
        rows = self.state_store.list_non_terminal_intents()
        if len(rows) < self.loop_threshold:
            return None

        recent = rows[-self.loop_threshold :]
        intents: list[StrategyIntent] = []
        for row in recent:
            try:
                intents.append(StrategyIntent.model_validate_json(row["intent_json"]))
            except Exception:
                return None

        symbol_set = {intent.symbol for intent in intents}
        regime_set = {intent.regime.value for intent in intents}
        if len(symbol_set) == 1 and len(regime_set) == 1:
            return {
                "count": len(intents),
                "symbol": next(iter(symbol_set)),
                "regime": next(iter(regime_set)),
                "states": [row["state"] for row in recent],
            }
        return None

    def _detect_deny_side_flips(self, now: datetime) -> dict[str, Any] | None:
        since = now - timedelta(minutes=max(int(os.getenv("WATCHDOG_DENY_LOOKBACK_MIN", "10")), 1))
        denied = self.state_store.get_denied_intents_since(since=since, limit=max(self.flip_threshold * 3, 6))
        if len(denied) < self.flip_threshold:
            return None

        recent = list(reversed(denied[: self.flip_threshold]))
        sides = [item["intent"].side.value for item in recent]
        regimes = [item["intent"].regime.value for item in recent]
        alternating = all(sides[idx] != sides[idx - 1] for idx in range(1, len(sides)))
        same_regime = len(set(regimes)) == 1
        if alternating and same_regime:
            return {
                "count": len(recent),
                "sides": sides,
                "regime": regimes[0],
                "intent_ids": [item["intent_id"] for item in recent],
            }
        return None

    def _detect_stuck_entry(
        self,
        now: datetime,
        open_entries: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for row in open_entries:
            status_text = str(row.get("status", "")).lower().replace("orderstatus.", "")
            if status_text in {
                "filled",
                "canceled",
                "cancelled",
                "rejected",
                "expired",
                "done_for_day",
            }:
                continue
            submitted = row.get("submitted_at")
            if not submitted:
                continue
            try:
                submitted_at = datetime.fromisoformat(str(submitted))
            except ValueError:
                continue

            intent: StrategyIntent | None = row.get("intent")
            cancel_after = int(intent.cancel_after_sec) if intent is not None else 30
            age_sec = max((now - submitted_at).total_seconds(), 0.0)
            if age_sec > (cancel_after + self.stuck_grace_sec):
                return {
                    "intent_id": row.get("intent_id"),
                    "broker_order_id": row.get("broker_order_id"),
                    "status": row.get("status"),
                    "age_sec": round(age_sec, 3),
                    "cancel_after_sec": cancel_after,
                    "grace_sec": self.stuck_grace_sec,
                }
        return None
