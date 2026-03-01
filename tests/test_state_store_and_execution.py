from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.execution_service import ExecutionContext, ExecutionService
from gold_trading_one_trade_per_day.schemas import (
    ExecutionCommand,
    IntentState,
    Regime,
    Side,
    StrategyIntent,
    TransitionEvent,
)
from gold_trading_one_trade_per_day.state_store import StateStore


class TestStateStoreAndExecution(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmpdir.name}/state.db"
        self.store = StateStore(self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_intent_transition_and_mock_execution(self):
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        intent = StrategyIntent(
            symbol="XAU_USD",
            side=Side.BUY,
            entry_price=2900,
            sl=2890,
            tp=2915,
            qty_hint=10,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=now,
            expires_at=now + timedelta(seconds=45),
            invalidation_reason="loss of momentum",
            cancel_after_sec=30,
        )
        self.store.record_intent(intent=intent, event_id=None)

        self.store.transition(
            TransitionEvent(
                intent_id=intent.intent_id,
                from_state=IntentState.INTENT_GENERATED,
                to_state=IntentState.RISK_APPROVED,
                at=now,
                metadata={"test": True},
            )
        )

        result = self.store.get_intent(intent.intent_id)
        self.assertIsNotNone(result)
        _, state = result
        self.assertEqual(state, IntentState.RISK_APPROVED)

        cmd = ExecutionCommand(
            intent_id=intent.intent_id,
            client_order_id="xau-test-001",
            cancel_after_sec=30,
            max_slippage_bps=20,
            risk_signature="abcdef123456",
            symbol="XAU_USD",
            side=Side.BUY,
            qty=1,
            entry_limit_price=2900,
            sl=2890,
            tp=2915,
        )
        service = ExecutionService(state_store=self.store, trading_client=None)
        report = service.execute(ExecutionContext(intent=intent, command=cmd))
        self.assertEqual(report.status, "Executed")

    def test_event_updates_and_daily_query(self):
        day = datetime.now(tz=ZoneInfo("UTC")).date().isoformat()
        self.store.record_event(
            event_id="evt-1",
            symbol="XAU_USD",
            snapshot={"skip_reason_code": None},
        )
        self.store.update_event(
            event_id="evt-1",
            status="skipped:open_warmup",
            snapshot_patch={"skip_reason_code": "open_warmup"},
        )

        events = self.store.get_events_for_day(day)
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[0]["status"], "skipped:open_warmup")

    def test_live_submit_error_returns_not_executed(self):
        class DummyClient:
            def submit_order(self, _req):
                raise RuntimeError("sub-penny increment")

        now = datetime.now(tz=ZoneInfo("America/New_York"))
        intent = StrategyIntent(
            symbol="XAU_USD",
            side=Side.BUY,
            entry_price=2900.00,
            sl=2890.50,
            tp=2915.00,
            qty_hint=10,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=now,
            expires_at=now + timedelta(seconds=45),
            invalidation_reason="smoke test",
            cancel_after_sec=30,
        )
        cmd = ExecutionCommand(
            intent_id=intent.intent_id,
            client_order_id="xau-test-live-001",
            cancel_after_sec=30,
            max_slippage_bps=20,
            risk_signature="abcdef123456",
            symbol="XAU_USD",
            side=Side.BUY,
            qty=1,
            entry_limit_price=2900.00,
            sl=2890.50,
            tp=2915.00,
        )
        service = ExecutionService(state_store=self.store, trading_client=DummyClient())
        report = service.execute(ExecutionContext(intent=intent, command=cmd))
        self.assertEqual(report.status, "Not_Executed")
        self.assertIn("API_ERROR_ENTRY_SUBMIT", report.reject_reason or "")

    def test_quantize_price_for_equity_tick(self):
        self.assertEqual(ExecutionService._quantize_price(123.6969, "ceil"), 123.70)
        self.assertEqual(ExecutionService._quantize_price(123.6969, "floor"), 123.69)

    def test_list_open_entry_orders_excludes_enum_canceled_status(self):
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        intent = StrategyIntent(
            symbol="XAU_USD",
            side=Side.BUY,
            entry_price=2900,
            sl=2890,
            tp=2915,
            qty_hint=10,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=now,
            expires_at=now + timedelta(seconds=45),
            invalidation_reason="test canceled enum status",
            cancel_after_sec=30,
        )
        self.store.record_intent(intent=intent, event_id=None)
        self.store.record_order(
            order_id="ord-canceled-enum",
            intent_id=intent.intent_id,
            client_order_id="gld-cancel-enum-1",
            order_role="entry",
            status="OrderStatus.CANCELED",
            payload={"status": "canceled"},
            broker_order_id="broker-canceled-enum",
            submitted_at=now,
        )
        open_orders = self.store.list_open_entry_orders()
        self.assertEqual(open_orders, [])


if __name__ == "__main__":
    unittest.main()
