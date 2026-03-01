from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from gold_trading_one_trade_per_day.benchmark_models import score_model_summary
from gold_trading_one_trade_per_day.latency_policy import evaluate_latency_policy
from gold_trading_one_trade_per_day.schemas import (
    LatencyPolicyDecision,
    Regime,
    Side,
    StrategyIntent,
)
from gold_trading_one_trade_per_day.state_store import StateStore
from gold_trading_one_trade_per_day.watchdog import Watchdog
from gold_trading_one_trade_per_day.warmup import run_warmup, warmup_is_recent_and_passed


class _DummyExecutionService:
    def __init__(self) -> None:
        self.cancel_called = 0
        self.flatten_called = 0

    def cancel_all_open_orders(self):
        self.cancel_called += 1
        return {"status": "ok"}

    def flatten_all_positions(self):
        self.flatten_called += 1
        return {"status": "ok"}


class _FakeHealth:
    def __init__(self):
        self.connected = True
        self.thread_alive = True
        self.stale = False
        self.data_age_sec = 0.1
        self.has_bar = True
        self.has_quote = True


class _FakeSensor:
    def __init__(self, symbol: str = "XAU_USD"):
        self.symbol = symbol
        self.enabled = True

    def start(self) -> bool:
        return True

    def health(self):
        return _FakeHealth()

    def stop(self) -> None:
        return None


class TestLatencyWatchdogWarmupBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmpdir.name}/state.db"
        self.store = StateStore(self.db_path)
        self._old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)
        self.tmpdir.cleanup()

    def test_latency_policy_hysteresis(self):
        now = datetime(2026, 2, 25, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        for idx in range(50):
            base = now + timedelta(seconds=idx)
            self.store.upsert_latency_metric(
                event_id=f"slow-{idx}",
                event_detected_at=base,
                entry_submitted_at=base + timedelta(milliseconds=1000),
                entry_filled_at=base + timedelta(milliseconds=9000),
            )

        first = evaluate_latency_policy(self.store, now=now + timedelta(minutes=1))
        self.assertTrue(first.degraded_mode)
        self.assertEqual(first.reason_code, "p95_signal_to_fill_breach")

        for idx in range(50):
            base = now + timedelta(minutes=2, seconds=idx)
            self.store.upsert_latency_metric(
                event_id=f"fast-{idx}",
                event_detected_at=base,
                entry_submitted_at=base + timedelta(milliseconds=1000),
                entry_filled_at=base + timedelta(milliseconds=4500),
            )

        second = evaluate_latency_policy(self.store, now=now + timedelta(minutes=4))
        self.assertTrue(second.degraded_mode)
        self.assertEqual(second.reason_code, "degraded_recovery_window")

        third = evaluate_latency_policy(self.store, now=now + timedelta(minutes=5))
        self.assertFalse(third.degraded_mode)
        self.assertEqual(third.reason_code, "degraded_recovered")

    def test_watchdog_stuck_entry_halts_and_resume(self):
        now = datetime(2026, 2, 25, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        intent = StrategyIntent(
            symbol="XAU_USD",
            side=Side.BUY,
            entry_price=2900.0,
            sl=2890.0,
            tp=2915.0,
            qty_hint=10,
            confidence=0.8,
            regime=Regime.TREND,
            generated_at=now - timedelta(minutes=3),
            expires_at=now + timedelta(minutes=1),
            invalidation_reason="stale momentum",
            cancel_after_sec=30,
        )
        self.store.record_intent(intent=intent, event_id=None)
        self.store.record_order(
            order_id="order-1",
            intent_id=intent.intent_id,
            client_order_id="client-1",
            order_role="entry",
            status="new",
            payload={"status": "new"},
            broker_order_id="broker-1",
            submitted_at=now - timedelta(seconds=60),
        )

        dummy = _DummyExecutionService()
        watchdog = Watchdog(state_store=self.store, execution_service=dummy)
        policy = LatencyPolicyDecision(
            degraded_mode=False,
            reason_code="normal",
            effective_slippage_bps=20,
            evaluated_at=now,
        )
        result = watchdog.evaluate_and_act(mode="paper", latency_policy=policy, now=now)
        self.assertTrue(result.halted)
        self.assertEqual(result.reason_code, "stuck_open_entry")
        self.assertEqual(dummy.cancel_called, 1)

        resumed = watchdog.resume(now=now + timedelta(seconds=5))
        self.assertFalse(bool(self.store.get_system_flag("halted_by_watchdog", False)))
        self.assertEqual(resumed["note"], "manual_resume")

    def test_warmup_report_and_recency_gate(self):
        now = datetime(2026, 2, 25, 9, 15, tzinfo=ZoneInfo("America/New_York"))
        idx = pd.date_range(end=now, periods=20, freq="1min")
        bars = pd.DataFrame(
            {
                "open": [2900.0] * 20,
                "high": [2903.0] * 20,
                "low": [2899.0] * 20,
                "close": [2901.0] * 20,
                "volume": [10000] * 20,
            },
            index=idx,
        )

        with patch("gold_trading_one_trade_per_day.warmup.is_local_model_available", return_value=True), patch(
            "gold_trading_one_trade_per_day.warmup.fetch_recent_bars", return_value=bars
        ), patch(
            "gold_trading_one_trade_per_day.warmup.fetch_latest_quote", return_value=(2900.0, 2900.50)
        ), patch(
            "gold_trading_one_trade_per_day.warmup.fetch_macro_proxy_returns",
            return_value={"SPY": 0.0, "VXX": 0.0, "UUP": 0.0, "TLT": 0.0},
        ), patch(
            "gold_trading_one_trade_per_day.warmup.MarketStreamSensor", _FakeSensor
        ):
            report = run_warmup(mode="paper", state_store=self.store, now=now)
        self.assertTrue(report.passed)

        ok, _ = warmup_is_recent_and_passed(state_store=self.store, now=now + timedelta(minutes=20))
        self.assertTrue(ok)

        os.environ["WARMUP_MAX_AGE_MINUTES"] = "10"
        stale_ok, stale_ctx = warmup_is_recent_and_passed(
            state_store=self.store,
            now=now + timedelta(minutes=40),
        )
        self.assertFalse(stale_ok)
        self.assertEqual(stale_ctx["reason"], "warmup_report_stale")

    def test_benchmark_scoring_deterministic(self):
        strong = {
            "schema_pass_rate": 1.0,
            "hallucination_rate": 0.0,
            "consistency_score": 0.9,
            "avg_latency_ms": 1200,
        }
        weak = {
            "schema_pass_rate": 0.5,
            "hallucination_rate": 0.4,
            "consistency_score": 0.5,
            "avg_latency_ms": 4200,
        }
        self.assertEqual(score_model_summary(strong), score_model_summary(strong))
        self.assertGreater(score_model_summary(strong), score_model_summary(weak))


if __name__ == "__main__":
    unittest.main()
