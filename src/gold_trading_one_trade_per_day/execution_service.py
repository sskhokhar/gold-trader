from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest

from gold_trading_one_trade_per_day.schemas import (
    EntryType,
    ExecutionCommand,
    ExecutionReport,
    LatencyPolicyDecision,
    Side,
    StrategyIntent,
)
from gold_trading_one_trade_per_day.state_store import StateStore


@dataclass(slots=True)
class ExecutionContext:
    intent: StrategyIntent
    command: ExecutionCommand
    latency_policy: LatencyPolicyDecision | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionService:
    def __init__(self, state_store: StateStore, trading_client: TradingClient | None = None):
        self.state_store = state_store
        self.trading_client = trading_client

    @staticmethod
    def _to_order_side(side: Side) -> OrderSide:
        if side == Side.BUY:
            return OrderSide.BUY
        if side == Side.SELL:
            return OrderSide.SELL
        raise ValueError(f"unsupported side: {side}")

    @staticmethod
    def _exit_side(side: Side) -> OrderSide:
        return OrderSide.SELL if side == Side.BUY else OrderSide.BUY

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return 0.0
        return float(value)

    @staticmethod
    def _order_to_payload(order: Any) -> dict[str, Any]:
        if hasattr(order, "model_dump"):
            return order.model_dump(mode="json")
        if isinstance(order, dict):
            return order
        return {"raw": str(order)}

    @staticmethod
    def _tick_size(price: float) -> float:
        return 0.01 if price >= 1.0 else 0.0001

    @classmethod
    def _quantize_price(cls, price: float, mode: str = "nearest") -> float:
        value = max(float(price), 0.0001)
        tick = cls._tick_size(value)
        units = value / tick
        if mode == "ceil":
            units = math.ceil(units)
        elif mode == "floor":
            units = math.floor(units)
        else:
            units = round(units)
        quantized = units * tick
        decimals = 2 if tick >= 0.01 else 4
        return max(round(quantized, decimals), tick)

    def execute(self, context: ExecutionContext) -> ExecutionReport:
        intent = context.intent
        cmd = context.command
        now = utc_now()

        if now > intent.expires_at:
            return ExecutionReport(
                intent_id=intent.intent_id,
                broker_order_id=None,
                status="Rejected",
                reject_reason="INTENT_EXPIRED",
                timestamps={"rejected_at": now.isoformat()},
            )

        if context.latency_policy and context.latency_policy.degraded_mode:
            if intent.entry_type != EntryType.MARKETABLE_LIMIT:
                return ExecutionReport(
                    intent_id=intent.intent_id,
                    broker_order_id=None,
                    status="Rejected",
                    reject_reason="LATENCY_DEGRADED_REQUIRES_MARKETABLE_LIMIT",
                    timestamps={"rejected_at": now.isoformat()},
                )
            if cmd.max_slippage_bps > context.latency_policy.effective_slippage_bps:
                cmd = cmd.model_copy(
                    update={
                        "max_slippage_bps": context.latency_policy.effective_slippage_bps,
                    }
                )
                context = ExecutionContext(
                    intent=intent,
                    command=cmd,
                    latency_policy=context.latency_policy,
                )

        if self.trading_client is None:
            return self._execute_mock(context)

        return self._execute_live(context)

    def _execute_mock(self, context: ExecutionContext) -> ExecutionReport:
        now = utc_now()
        entry_id = str(uuid.uuid4())
        oco_id = str(uuid.uuid4())
        cmd = context.command

        entry_payload = {
            "id": entry_id,
            "status": "filled",
            "filled_qty": cmd.qty,
            "filled_avg_price": cmd.entry_limit_price,
            "symbol": cmd.symbol,
            "side": cmd.side.value.lower(),
            "limit_price": cmd.entry_limit_price,
            "submitted_at": now.isoformat(),
            "filled_at": now.isoformat(),
        }

        self.state_store.record_order(
            order_id=entry_id,
            intent_id=cmd.intent_id,
            client_order_id=cmd.client_order_id,
            order_role="entry",
            status="filled",
            payload=entry_payload,
            broker_order_id=entry_id,
            submitted_at=now,
        )

        oco_payload = {
            "id": oco_id,
            "status": "new",
            "symbol": cmd.symbol,
            "side": ("sell" if cmd.side == Side.BUY else "buy"),
            "qty": cmd.qty,
            "order_class": "oco",
            "take_profit": {"limit_price": cmd.tp},
            "stop_loss": {"stop_price": cmd.sl},
            "submitted_at": now.isoformat(),
        }

        self.state_store.record_order(
            order_id=oco_id,
            intent_id=cmd.intent_id,
            client_order_id=f"{cmd.client_order_id}-oco",
            order_role="exit_oco",
            status="new",
            payload=oco_payload,
            broker_order_id=oco_id,
            submitted_at=now,
        )

        return ExecutionReport(
            intent_id=cmd.intent_id,
            broker_order_id=entry_id,
            entry_order_id=entry_id,
            oco_order_id=oco_id,
            status="Executed",
            filled_qty=cmd.qty,
            avg_fill_price=cmd.entry_limit_price,
            timestamps={
                "submitted_at": now.isoformat(),
                "filled_at": now.isoformat(),
                "oco_submitted_at": now.isoformat(),
            },
        )

    def _execute_live(self, context: ExecutionContext) -> ExecutionReport:
        intent = context.intent
        cmd = context.command
        now = utc_now()
        side = self._to_order_side(cmd.side)

        price_buffer = cmd.entry_limit_price * (cmd.max_slippage_bps / 10_000)
        if cmd.side == Side.BUY:
            protected_limit = self._quantize_price(
                cmd.entry_limit_price + price_buffer,
                mode="ceil",
            )
        else:
            protected_limit = self._quantize_price(
                max(cmd.entry_limit_price - price_buffer, 0.01),
                mode="floor",
            )

        entry_request = LimitOrderRequest(
            symbol=cmd.symbol,
            qty=cmd.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=protected_limit,
            client_order_id=cmd.client_order_id,
        )

        try:
            entry_order = self.trading_client.submit_order(entry_request)
        except Exception as exc:
            return ExecutionReport(
                intent_id=cmd.intent_id,
                broker_order_id=None,
                entry_order_id=None,
                status="Not_Executed",
                reject_reason=f"API_ERROR_ENTRY_SUBMIT:{exc}",
                timestamps={
                    "submitted_at": now.isoformat(),
                    "completed_at": utc_now().isoformat(),
                },
            )
        entry_payload = self._order_to_payload(entry_order)

        entry_id = str(entry_order.id)
        self.state_store.record_order(
            order_id=entry_id,
            intent_id=cmd.intent_id,
            client_order_id=cmd.client_order_id,
            order_role="entry",
            status=str(entry_order.status),
            payload=entry_payload,
            broker_order_id=entry_id,
            submitted_at=now,
        )

        deadline = time.monotonic() + cmd.cancel_after_sec
        current = entry_order
        while time.monotonic() < deadline:
            current = self.trading_client.get_order_by_client_id(cmd.client_order_id)
            status = current.status
            self.state_store.update_order_status(
                broker_order_id=str(current.id),
                status=str(status),
                payload=self._order_to_payload(current),
                filled_at=current.filled_at,
                cancelled_at=current.canceled_at,
            )
            if status == OrderStatus.FILLED:
                break
            if status in {
                OrderStatus.CANCELED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
                OrderStatus.DONE_FOR_DAY,
            }:
                return ExecutionReport(
                    intent_id=cmd.intent_id,
                    broker_order_id=str(current.id),
                    entry_order_id=str(current.id),
                    status="Not_Executed",
                    reject_reason=f"ENTRY_{status.value.upper()}",
                    timestamps={
                        "submitted_at": now.isoformat(),
                        "completed_at": utc_now().isoformat(),
                    },
                )
            time.sleep(1.0)

        current = self.trading_client.get_order_by_client_id(cmd.client_order_id)
        if current.status != OrderStatus.FILLED:
            self.trading_client.cancel_order_by_id(current.id)
            cancelled = self.trading_client.get_order_by_id(current.id)
            self.state_store.update_order_status(
                broker_order_id=str(current.id),
                status=str(cancelled.status),
                payload=self._order_to_payload(cancelled),
                cancelled_at=cancelled.canceled_at,
            )
            return ExecutionReport(
                intent_id=cmd.intent_id,
                broker_order_id=str(current.id),
                entry_order_id=str(current.id),
                status="Cancelled",
                reject_reason="cancelled_stale_entry",
                timestamps={
                    "submitted_at": now.isoformat(),
                    "cancelled_at": utc_now().isoformat(),
                },
            )

        filled_qty = self._as_float(current.filled_qty)
        fill_price = self._as_float(current.filled_avg_price)

        tp_price = self._quantize_price(cmd.tp, mode="nearest")
        sl_price = self._quantize_price(cmd.sl, mode="nearest")

        oco_request = LimitOrderRequest(
            symbol=cmd.symbol,
            qty=filled_qty,
            side=self._exit_side(cmd.side),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
            client_order_id=f"{cmd.client_order_id}-oco",
        )

        try:
            oco_order = self.trading_client.submit_order(oco_request)
        except Exception as exc:
            return ExecutionReport(
                intent_id=cmd.intent_id,
                broker_order_id=str(current.id),
                entry_order_id=str(current.id),
                status="Not_Executed",
                reject_reason=f"API_ERROR_OCO_SUBMIT:{exc}",
                filled_qty=filled_qty,
                avg_fill_price=fill_price if fill_price > 0 else None,
                timestamps={
                    "submitted_at": now.isoformat(),
                    "filled_at": current.filled_at.isoformat() if current.filled_at else utc_now().isoformat(),
                    "completed_at": utc_now().isoformat(),
                },
            )
        self.state_store.record_order(
            order_id=str(oco_order.id),
            intent_id=cmd.intent_id,
            client_order_id=f"{cmd.client_order_id}-oco",
            order_role="exit_oco",
            status=str(oco_order.status),
            payload=self._order_to_payload(oco_order),
            broker_order_id=str(oco_order.id),
            submitted_at=utc_now(),
        )

        return ExecutionReport(
            intent_id=intent.intent_id,
            broker_order_id=str(current.id),
            entry_order_id=str(current.id),
            oco_order_id=str(oco_order.id),
            status="Executed",
            filled_qty=filled_qty,
            avg_fill_price=fill_price if fill_price > 0 else None,
            timestamps={
                "submitted_at": now.isoformat(),
                "filled_at": current.filled_at.isoformat() if current.filled_at else utc_now().isoformat(),
                "oco_submitted_at": utc_now().isoformat(),
            },
        )

    def flatten_all_positions(self) -> dict[str, Any]:
        if self.trading_client is None:
            return {"status": "mock_flatten", "closed": 0}

        response = self.trading_client.close_all_positions(cancel_orders=True)
        return {
            "status": "flatten_requested",
            "closed": len(response) if isinstance(response, list) else 0,
        }

    def cancel_all_open_orders(self) -> dict[str, Any]:
        if self.trading_client is None:
            return {"status": "mock_cancel_orders", "cancelled": 0}
        try:
            response = self.trading_client.cancel_orders()
            return {
                "status": "cancel_requested",
                "cancelled": len(response) if isinstance(response, list) else 0,
            }
        except Exception as exc:
            return {
                "status": "cancel_failed",
                "error": str(exc),
            }
