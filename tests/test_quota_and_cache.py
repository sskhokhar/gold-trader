from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from gold_trading_one_trade_per_day.analysis_cache import AnalysisCache
from gold_trading_one_trade_per_day.event_trigger import build_feature_snapshot
from gold_trading_one_trade_per_day.quota_guard import QuotaGuard, call_with_rate_limit_backoff
from gold_trading_one_trade_per_day.schemas import DataSource, MarketSentimentReport, Regime
from gold_trading_one_trade_per_day.state_store import StateStore


class TestQuotaAndCache(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmpdir.name}/state.db"
        self.store = StateStore(self.db_path)
        self._old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)
        self.tmpdir.cleanup()

    def _bars(self) -> pd.DataFrame:
        idx = pd.date_range(end=datetime.now(tz=ZoneInfo("UTC")), periods=60, freq="1min")
        base = 200.0
        values = [base + i * 0.01 for i in range(59)] + [base + 1.0]
        close = pd.Series(values, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = close + 0.2
        low = close - 0.2
        volume = pd.Series([10000] * 59 + [45000], index=idx)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_quota_reservation_and_deny(self):
        os.environ["QUOTA_RPM_CAP"] = "1"
        os.environ["QUOTA_RPD_CAP"] = "2"
        guard = QuotaGuard(self.store)
        now = datetime.now(tz=ZoneInfo("America/New_York"))

        first = guard.reserve("evt-1", estimated_requests=1, source="test", now=now)
        self.assertTrue(first.allowed)
        second = guard.reserve("evt-2", estimated_requests=1, source="test", now=now)
        self.assertFalse(second.allowed)

        snap = guard.commit("evt-1", used_requests=1, source="test", now=now)
        self.assertEqual(snap.rpm_remaining, 0)

    def test_analysis_cache_key_and_ttl(self):
        snapshot = build_feature_snapshot(
            bars=self._bars(),
            bid=200.0,
            ask=200.01,
            timestamp=datetime(2026, 2, 25, 10, 15, tzinfo=ZoneInfo("America/New_York")),
            data_source=DataSource.STREAM,
            data_age_sec=1.0,
        )
        report = MarketSentimentReport(
            symbol="XAU_USD",
            generated_at=datetime.now(tz=ZoneInfo("America/New_York")),
            regime=Regime.TREND,
            greed_score=60,
            sentiment_score=0.2,
            rationale=["volume expansion"],
        )
        os.environ["ANALYSIS_CACHE_TTL_SEC"] = "60"
        cache = AnalysisCache(self.store)
        key1 = cache.make_key(snapshot, "ollama/llama4:8b")
        key2 = cache.make_key(snapshot, "ollama/llama4:8b")
        self.assertEqual(key1, key2)

        now = datetime.now(tz=ZoneInfo("America/New_York"))
        cache.put(key1, "ollama/llama4:8b", report, now)
        _, hit = cache.get(snapshot, "ollama/llama4:8b", now=now + timedelta(seconds=1))
        self.assertIsNotNone(hit)
        _, miss = cache.get(snapshot, "ollama/llama4:8b", now=now + timedelta(seconds=61))
        self.assertIsNone(miss)

    def test_backoff_exhausts_before_deadline(self):
        start = datetime(2026, 2, 25, 10, 15, tzinfo=ZoneInfo("America/New_York"))
        clock = {"now": start}

        def now_fn():
            return clock["now"]

        def sleep_fn(sec: float):
            clock["now"] = clock["now"] + timedelta(seconds=sec)

        def always_429():
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        result = call_with_rate_limit_backoff(
            func=always_429,
            deadline=start + timedelta(seconds=5),
            base_wait_sec=2,
            max_wait_sec=2,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            jitter_ratio=0.0,
        )
        self.assertTrue(result.exhausted)
        self.assertGreaterEqual(result.retry_count, 2)


if __name__ == "__main__":
    unittest.main()
