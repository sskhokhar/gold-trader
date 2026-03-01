from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from gold_trading_one_trade_per_day.schemas import IntentState
from gold_trading_one_trade_per_day.state_store import StateStore

try:
    import oandapyV20.endpoints.orders as oanda_orders
    _OANDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OANDA_AVAILABLE = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def reconcile_startup(
    state_store: StateStore,
    trading_client: Any,
    day_state: Any,
) -> dict[str, int]:
    summary = {
        "expired_intents": 0,
        "halted_intents": 0,
        "open_orders_seen": 0,
    }

    now = utc_now()
    for row in state_store.list_non_terminal_intents():
        intent_id = row["intent_id"]
        expires_at = datetime.fromisoformat(row["expires_at"])
        state = IntentState(row["state"])

        if expires_at < now and state not in {
            IntentState.CLOSED,
            IntentState.DENIED,
            IntentState.EXPIRED,
            IntentState.CANCELLED,
            IntentState.HALTED,
            IntentState.ERROR,
        }:
            state_store.mark_intent_terminal(intent_id, IntentState.EXPIRED, "startup_expired")
            summary["expired_intents"] += 1

    if getattr(day_state, "hard_lock", False):
        for row in state_store.list_non_terminal_intents():
            state_store.mark_intent_terminal(
                row["intent_id"],
                IntentState.HALTED,
                "hard_lock_active_on_startup",
            )
            summary["halted_intents"] += 1

    if trading_client is not None:
        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        try:
            if _OANDA_AVAILABLE:
                r = oanda_orders.OrdersPending(account_id)
                trading_client.request(r)
                orders = r.response.get("orders", [])
                summary["open_orders_seen"] = len(orders)
        except Exception:
            summary["open_orders_seen"] = -1

    return summary

