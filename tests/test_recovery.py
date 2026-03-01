from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.recovery import reconcile_startup
from gold_trading_one_trade_per_day.schemas import DailyState, Regime, Side, StrategyIntent
from gold_trading_one_trade_per_day.state_store import StateStore


class TestRecovery(unittest.TestCase):
    def test_expires_stale_intents(self):
        with tempfile.TemporaryDirectory() as d:
            store = StateStore(f"{d}/state.db")
            now = datetime.now(tz=ZoneInfo("America/New_York"))
            stale = StrategyIntent(
                symbol="XAU_USD",
                side=Side.BUY,
                entry_price=2900,
                sl=2890,
                tp=2915,
                qty_hint=10,
                confidence=0.7,
                regime=Regime.TREND,
                generated_at=now - timedelta(minutes=5),
                expires_at=now - timedelta(minutes=4),
                invalidation_reason="stale",
                cancel_after_sec=30,
            )
            store.record_intent(stale, event_id=None)
            day_state = DailyState(
                day=now.date().isoformat(),
                day_start_equity=100000,
                equity_hwm=100000,
                current_equity=100000,
            )
            summary = reconcile_startup(store, None, day_state)
            self.assertGreaterEqual(summary["expired_intents"], 1)


if __name__ == "__main__":
    unittest.main()
