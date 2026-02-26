from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

from gold_trading_one_trade_per_day.crew import GoldTradingOneTradePerDayCrew
from gold_trading_one_trade_per_day.main import _normalize_strategy_intent_timestamps
from gold_trading_one_trade_per_day.schemas import EntryType, Regime, Side, StrategyIntent


class _TaskOutput:
    def __init__(self, raw: str):
        self.raw = raw
        self.pydantic = None


class TestCrewRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_agent_limits_and_crew_rpm(self):
        os.environ["CREW_MAX_RPM"] = "8"
        os.environ["AGENT_MAX_RPM"] = "6"
        crew_runner = GoldTradingOneTradePerDayCrew(mode="shadow")

        analyst = crew_runner.market_sentiment_analyst()
        strategy = crew_runner.strategy_composer()
        self.assertEqual(analyst.max_iter, 3)
        self.assertEqual(strategy.max_iter, 2)
        self.assertEqual(analyst.max_rpm, 6)
        self.assertEqual(strategy.max_rpm, 6)

        crew = crew_runner.crew()
        self.assertEqual(crew.max_rpm, 8)
        self.assertTrue(crew.cache)

    def test_strategy_guardrail_rejects_stale_intent_timestamp(self):
        stale = datetime(2023, 2, 20, 14, 30, tzinfo=timezone.utc)
        payload = StrategyIntent(
            intent_id="intent-stale-001",
            symbol="GLD",
            side=Side.BUY,
            entry_type=EntryType.MARKETABLE_LIMIT,
            entry_price=200.0,
            sl=199.5,
            tp=201.0,
            qty_hint=1.0,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=stale,
            expires_at=stale + timedelta(seconds=45),
            invalidation_reason="stale test",
            cancel_after_sec=30,
        )
        ok, msg = GoldTradingOneTradePerDayCrew._validate_strategy_intent(
            _TaskOutput(payload.model_dump_json())
        )
        self.assertFalse(ok)
        self.assertIn("generated_at too old", msg)

    def test_strategy_timestamp_normalization(self):
        stale = datetime(2023, 2, 20, 14, 30, tzinfo=timezone.utc)
        intent = StrategyIntent(
            intent_id="intent-normalize-001",
            symbol="GLD",
            side=Side.SELL,
            entry_type=EntryType.MARKETABLE_LIMIT,
            entry_price=200.0,
            sl=201.0,
            tp=199.0,
            qty_hint=1.0,
            confidence=0.8,
            regime=Regime.HIGH_VOL,
            generated_at=stale,
            expires_at=stale + timedelta(seconds=45),
            invalidation_reason="normalize test",
            cancel_after_sec=30,
        )
        now = datetime.now(timezone.utc)
        normalized = _normalize_strategy_intent_timestamps(intent, now=now, ttl_seconds=45)
        self.assertGreater(normalized.generated_at, stale)
        self.assertEqual(
            int((normalized.expires_at - normalized.generated_at).total_seconds()),
            45,
        )

    def test_strategy_guardrail_allows_past_absolute_expiry_if_ttl_valid(self):
        now = datetime.now(timezone.utc)
        generated = now - timedelta(seconds=70)
        payload = StrategyIntent(
            intent_id="intent-past-expiry-001",
            symbol="GLD",
            side=Side.BUY,
            entry_type=EntryType.MARKETABLE_LIMIT,
            entry_price=200.0,
            sl=199.5,
            tp=201.0,
            qty_hint=1.0,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=generated,
            expires_at=generated + timedelta(seconds=45),
            invalidation_reason="past expiry guardrail test",
            cancel_after_sec=30,
        )
        ok, msg = GoldTradingOneTradePerDayCrew._validate_strategy_intent(
            _TaskOutput(payload.model_dump_json())
        )
        self.assertTrue(ok, msg)

    def test_strategy_guardrail_allows_ttl_above_runtime_target(self):
        now = datetime.now(timezone.utc)
        payload = StrategyIntent(
            intent_id="intent-long-ttl-001",
            symbol="GLD",
            side=Side.BUY,
            entry_type=EntryType.MARKETABLE_LIMIT,
            entry_price=200.0,
            sl=199.5,
            tp=201.0,
            qty_hint=1.0,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=now,
            expires_at=now + timedelta(seconds=65),
            invalidation_reason="ttl guardrail tolerance test",
            cancel_after_sec=30,
        )
        ok, msg = GoldTradingOneTradePerDayCrew._validate_strategy_intent(
            _TaskOutput(payload.model_dump_json())
        )
        self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
