from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.risk_engine import RiskEngine, load_risk_config
from gold_trading_one_trade_per_day.schemas import DailyState, Regime, Side, StrategyIntent


class TestSchemasAndRisk(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.now(tz=ZoneInfo("America/New_York"))

    def test_strategy_intent_buy_validation(self):
        intent = StrategyIntent(
            symbol="GLD",
            side=Side.BUY,
            entry_price=200.0,
            sl=199.0,
            tp=202.0,
            qty_hint=10,
            confidence=0.7,
            regime=Regime.TREND,
            generated_at=self.now,
            expires_at=self.now + timedelta(seconds=45),
            invalidation_reason="volume collapse",
            cancel_after_sec=30,
        )
        self.assertEqual(intent.side, Side.BUY)

    def test_strategy_intent_buy_invalid_levels(self):
        with self.assertRaises(ValueError):
            StrategyIntent(
                symbol="GLD",
                side=Side.BUY,
                entry_price=200.0,
                sl=201.0,
                tp=202.0,
                qty_hint=10,
                confidence=0.7,
                regime=Regime.TREND,
                generated_at=self.now,
                expires_at=self.now + timedelta(seconds=45),
                invalidation_reason="bad setup",
                cancel_after_sec=30,
            )

    def test_risk_soft_and_hard_locks(self):
        engine = RiskEngine()
        state = DailyState(
            day="2026-02-25",
            day_start_equity=100000,
            equity_hwm=100000,
            current_equity=100000,
        )

        engine.update_locks(state, 100750)
        self.assertTrue(state.soft_lock)
        self.assertFalse(state.hard_lock)

        engine.update_locks(state, 101500)
        self.assertTrue(state.hard_lock)

    def test_risk_config_precedence_env_over_profile(self):
        with tempfile.TemporaryDirectory() as d:
            profile = f"{d}/risk_profile.yaml"
            with open(profile, "w", encoding="utf-8") as f:
                f.write("daily_soft_lock_pct: 0.02\n")

            old_env = os.environ.get("RISK_DAILY_SOFT_LOCK_PCT")
            os.environ["RISK_DAILY_SOFT_LOCK_PCT"] = "0.01"
            try:
                cfg, source = load_risk_config(profile_path=profile)
            finally:
                if old_env is None:
                    del os.environ["RISK_DAILY_SOFT_LOCK_PCT"]
                else:
                    os.environ["RISK_DAILY_SOFT_LOCK_PCT"] = old_env

            self.assertEqual(source, "env")
            self.assertAlmostEqual(cfg.daily_soft_lock_pct, 0.01)


if __name__ == "__main__":
    unittest.main()
