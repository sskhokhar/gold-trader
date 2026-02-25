from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from gold_trading_one_trade_per_day.event_trigger import (
    build_feature_snapshot,
    should_wake_ai,
)
from gold_trading_one_trade_per_day.schemas import DataSource, DailyState


class TestEventTrigger(unittest.TestCase):
    def _bars(self) -> pd.DataFrame:
        idx = pd.date_range(end=datetime.now(tz=ZoneInfo("UTC")), periods=60, freq="1min")
        base = 200.0
        values = [base + i * 0.01 for i in range(59)] + [base + 2.0]
        close = pd.Series(values, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high_values = [v + 0.1 for v in values[:-1]] + [values[-1] + 1.0]
        low_values = [v - 0.1 for v in values[:-1]] + [values[-1] - 1.0]
        high = pd.Series(high_values, index=idx)
        low = pd.Series(low_values, index=idx)
        volume = pd.Series([10000] * 59 + [50000], index=idx)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_snapshot_and_trigger(self):
        bars = self._bars()
        ts = datetime(2026, 2, 25, 10, 15, tzinfo=ZoneInfo("America/New_York"))
        snapshot = build_feature_snapshot(
            bars,
            bid=200.0,
            ask=200.01,
            macro_proxies={"SPY": 0.01, "VXX": -0.01},
            timestamp=ts,
            data_source=DataSource.STREAM,
            data_age_sec=2.5,
        )
        state = DailyState(
            day="2026-02-25",
            day_start_equity=100000,
            equity_hwm=100000,
            current_equity=100000,
        )
        ok, reason, context = should_wake_ai(snapshot, state)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "triggered")
        self.assertEqual(context["data_source"], "stream")

    def test_open_warmup_blocks(self):
        bars = self._bars()
        ts = datetime(2026, 2, 25, 9, 32, tzinfo=ZoneInfo("America/New_York"))
        snapshot = build_feature_snapshot(
            bars,
            bid=200.0,
            ask=200.01,
            timestamp=ts,
        )
        state = DailyState(
            day="2026-02-25",
            day_start_equity=100000,
            equity_hwm=100000,
            current_equity=100000,
        )
        ok, reason, _ = should_wake_ai(snapshot, state, open_warmup_minutes=5)
        self.assertFalse(ok)
        self.assertEqual(reason, "open_warmup")

    def test_macro_window_blocks(self):
        bars = self._bars()
        ts = datetime(2026, 2, 25, 10, 15, tzinfo=ZoneInfo("America/New_York"))
        snapshot = build_feature_snapshot(
            bars,
            bid=200.0,
            ask=200.01,
            timestamp=ts,
        )
        state = DailyState(
            day="2026-02-25",
            day_start_equity=100000,
            equity_hwm=100000,
            current_equity=100000,
        )
        ok, reason, _ = should_wake_ai(
            snapshot,
            state,
            macro_event_active=True,
            macro_event_label="CPI",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "macro_event_window")


if __name__ == "__main__":
    unittest.main()
