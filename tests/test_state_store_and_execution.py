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
            symbol="GLD",
            side=Side.BUY,
            entry_price=200,
            sl=199,
            tp=202,
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
            client_order_id="gld-test-001",
            cancel_after_sec=30,
            max_slippage_bps=20,
            risk_signature="abcdef123456",
            symbol="GLD",
            side=Side.BUY,
            qty=1,
            entry_limit_price=200,
            sl=199,
            tp=202,
        )
        service = ExecutionService(state_store=self.store, trading_client=None)
        report = service.execute(ExecutionContext(intent=intent, command=cmd))
        self.assertEqual(report.status, "Executed")

    def test_event_updates_and_daily_query(self):
        day = datetime.now(tz=ZoneInfo("UTC")).date().isoformat()
        self.store.record_event(
            event_id="evt-1",
            symbol="GLD",
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


if __name__ == "__main__":
    unittest.main()
