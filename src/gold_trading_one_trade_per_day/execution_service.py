from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gold_trading_one_trade_per_day.schemas import (
    EntryType,
    ExecutionCommand,
    ExecutionReport,
    LatencyPolicyDecision,
    Side,
    StrategyIntent,
)
from gold_trading_one_trade_per_day.state_store import StateStore

try:
    import oandapyV20
    import oandapyV20.endpoints.orders as oanda_orders
    import oandapyV20.endpoints.positions as oanda_positions
    import oandapyV20.endpoints.trades as oanda_trades
    _OANDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OANDA_AVAILABLE = False


@dataclass(slots=True)
class ExecutionContext:
    intent: StrategyIntent
    command: ExecutionCommand
    latency_policy: LatencyPolicyDecision | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionService:
    def __init__(self, state_store: StateStore, trading_client: "oandapyV20.API | None" = None):
        self.state_store = state_store
        self.trading_client = trading_client

    @staticmethod
    def _side_units(side: Side, qty: float) -> str:
        """OANDA uses signed units: positive = buy, negative = sell."""
        return str(int(qty)) if side == Side.BUY else str(-int(qty))

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
        # XAUUSD uses 2 decimal places (e.g. 2900.50)
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
        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")

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

        tp_price = self._quantize_price(cmd.tp, mode="nearest")
        sl_price = self._quantize_price(cmd.sl, mode="nearest")

        order_body = {
            "order": {
                "type": "LIMIT",
                "instrument": cmd.symbol,
                "units": self._side_units(cmd.side, cmd.qty),
                "price": f"{protected_limit:.2f}",
                "timeInForce": "GTD",
                "gtdTime": intent.expires_at.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "takeProfitOnFill": {"price": f"{tp_price:.2f}"},
                "stopLossOnFill": {"price": f"{sl_price:.2f}"},
                "clientExtensions": {"id": cmd.client_order_id[:128]},
            }
        }

        try:
            r = oanda_orders.OrderCreate(account_id, data=order_body)
            self.trading_client.request(r)
            response = r.response
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

        order_create_txn = response.get("orderCreateTransaction", {})
        entry_id = str(order_create_txn.get("id", str(uuid.uuid4())))
        entry_payload = response

        self.state_store.record_order(
            order_id=entry_id,
            intent_id=cmd.intent_id,
            client_order_id=cmd.client_order_id,
            order_role="entry",
            status="pending",
            payload=entry_payload,
            broker_order_id=entry_id,
            submitted_at=now,
        )

        # Poll for fill
        deadline = time.monotonic() + cmd.cancel_after_sec
        fill_txn = response.get("orderFillTransaction")
        filled_qty = 0.0
        fill_price = 0.0
        trade_id: str | None = None

        if fill_txn:
            filled_qty = abs(float(fill_txn.get("units", 0)))
            fill_price = float(fill_txn.get("price", 0))
            trade_id = str(fill_txn.get("tradeOpened", {}).get("tradeID", ""))

        while not fill_txn and time.monotonic() < deadline:
            time.sleep(1.0)
            try:
                r2 = oanda_orders.OrderDetails(account_id, entry_id)
                self.trading_client.request(r2)
                order_detail = r2.response.get("order", {})
                state = order_detail.get("state", "")
                self.state_store.update_order_status(
                    broker_order_id=entry_id,
                    status=state,
                    payload=r2.response,
                )
                if state == "FILLED":
                    filled_units = abs(float(order_detail.get("units", 0)))
                    avg_price = float(order_detail.get("averageFilledPrice", fill_price or 0))
                    filled_qty = filled_units
                    fill_price = avg_price
                    trade_ids = order_detail.get("tradeOpenedIDs", [])
                    trade_id = str(trade_ids[0]) if trade_ids else None
                    fill_txn = order_detail
                    break
                if state in {"CANCELLED", "EXPIRED"}:
                    return ExecutionReport(
                        intent_id=cmd.intent_id,
                        broker_order_id=entry_id,
                        entry_order_id=entry_id,
                        status="Not_Executed",
                        reject_reason=f"ENTRY_{state}",
                        timestamps={
                            "submitted_at": now.isoformat(),
                            "completed_at": utc_now().isoformat(),
                        },
                    )
            except Exception:
                continue

        if not fill_txn:
            # Cancel the unfilled order
            try:
                r_cancel = oanda_orders.OrderCancel(account_id, entry_id)
                self.trading_client.request(r_cancel)
                self.state_store.update_order_status(
                    broker_order_id=entry_id,
                    status="CANCELLED",
                    payload=r_cancel.response,
                )
            except Exception:
                pass
            return ExecutionReport(
                intent_id=cmd.intent_id,
                broker_order_id=entry_id,
                entry_order_id=entry_id,
                status="Cancelled",
                reject_reason="cancelled_stale_entry",
                timestamps={
                    "submitted_at": now.isoformat(),
                    "cancelled_at": utc_now().isoformat(),
                },
            )

        # Entry filled — TP/SL were submitted on fill via takeProfitOnFill/stopLossOnFill
        oco_id = trade_id or str(uuid.uuid4())
        oco_payload: dict[str, Any] = {
            "trade_id": trade_id,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "submitted_at": utc_now().isoformat(),
        }
        self.state_store.record_order(
            order_id=oco_id,
            intent_id=cmd.intent_id,
            client_order_id=f"{cmd.client_order_id}-oco",
            order_role="exit_oco",
            status="new",
            payload=oco_payload,
            broker_order_id=oco_id,
            submitted_at=utc_now(),
        )

        return ExecutionReport(
            intent_id=intent.intent_id,
            broker_order_id=entry_id,
            entry_order_id=entry_id,
            oco_order_id=oco_id,
            status="Executed",
            filled_qty=filled_qty,
            avg_fill_price=fill_price if fill_price > 0 else None,
            timestamps={
                "submitted_at": now.isoformat(),
                "filled_at": utc_now().isoformat(),
                "oco_submitted_at": utc_now().isoformat(),
            },
        )

    def flatten_all_positions(self) -> dict[str, Any]:
        if self.trading_client is None:
            return {"status": "mock_flatten", "closed": 0}

        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        instrument = "XAU_USD"
        closed = 0
        try:
            # Try closing long units
            try:
                r_long = oanda_positions.PositionClose(
                    account_id,
                    instrument,
                    data={"longUnits": "ALL"},
                )
                self.trading_client.request(r_long)
                closed += 1
            except Exception:
                pass
            # Try closing short units
            try:
                r_short = oanda_positions.PositionClose(
                    account_id,
                    instrument,
                    data={"shortUnits": "ALL"},
                )
                self.trading_client.request(r_short)
                closed += 1
            except Exception:
                pass
        except Exception as exc:
            return {"status": "flatten_error", "error": str(exc), "closed": closed}
        return {"status": "flatten_requested", "closed": closed}

    def cancel_all_open_orders(self) -> dict[str, Any]:
        if self.trading_client is None:
            return {"status": "mock_cancel_orders", "cancelled": 0}
        account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        try:
            r = oanda_orders.OrdersPending(account_id)
            self.trading_client.request(r)
            orders = r.response.get("orders", [])
            cancelled = 0
            for order in orders:
                order_id = order.get("id")
                if not order_id:
                    continue
                try:
                    r_cancel = oanda_orders.OrderCancel(account_id, order_id)
                    self.trading_client.request(r_cancel)
                    cancelled += 1
                except Exception:
                    pass
            return {"status": "cancel_requested", "cancelled": cancelled}
        except Exception as exc:
            return {"status": "cancel_failed", "error": str(exc)}

