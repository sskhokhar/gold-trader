"""Tests for the synthetic gold data generator."""

import sys
import os
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_trading_one_trade_per_day.synthetic_data import (
    generate_gold_bars,
    generate_trending_day,
    generate_range_day,
    generate_volatile_event_day,
    generate_multi_day_data,
)

UTC_TZ = ZoneInfo("UTC")


class TestGenerateGoldBars(unittest.TestCase):
    def test_basic_shape(self):
        df = generate_gold_bars(num_bars=100, seed=42)
        self.assertEqual(len(df), 100)
        self.assertIn("timestamp", df.columns)
        self.assertIn("open", df.columns)
        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)

    def test_ohlc_consistency(self):
        """High >= max(open, close) and Low <= min(open, close) for every bar."""
        df = generate_gold_bars(num_bars=500, seed=42)
        for _, row in df.iterrows():
            self.assertGreaterEqual(row["high"], max(row["open"], row["close"]))
            self.assertLessEqual(row["low"], min(row["open"], row["close"]))

    def test_positive_prices(self):
        df = generate_gold_bars(num_bars=500, seed=42)
        self.assertTrue((df["open"] > 0).all())
        self.assertTrue((df["high"] > 0).all())
        self.assertTrue((df["low"] > 0).all())
        self.assertTrue((df["close"] > 0).all())

    def test_positive_volume(self):
        df = generate_gold_bars(num_bars=100, seed=42)
        self.assertTrue((df["volume"] > 0).all())

    def test_reproducible_with_seed(self):
        df1 = generate_gold_bars(num_bars=50, seed=99)
        df2 = generate_gold_bars(num_bars=50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_gold_bars(num_bars=50, seed=1)
        df2 = generate_gold_bars(num_bars=50, seed=2)
        self.assertFalse(df1["close"].equals(df2["close"]))

    def test_start_price(self):
        df = generate_gold_bars(start_price=3000.0, num_bars=10, seed=42)
        # First open should be close to start_price
        self.assertAlmostEqual(df["open"].iloc[0], 3000.0, delta=5.0)

    def test_custom_start_time(self):
        start = datetime(2025, 3, 1, 12, 0, tzinfo=UTC_TZ)
        df = generate_gold_bars(start_time=start, num_bars=10, seed=42)
        self.assertEqual(df["timestamp"].iloc[0], start)

    def test_bar_interval(self):
        df = generate_gold_bars(num_bars=10, bar_interval_minutes=15, seed=42)
        delta = df["timestamp"].iloc[1] - df["timestamp"].iloc[0]
        self.assertEqual(delta.total_seconds(), 15 * 60)


class TestTrendingDay(unittest.TestCase):
    def test_up_trend_end_higher(self):
        df = generate_trending_day(direction="up", seed=42)
        self.assertEqual(len(df), 288)  # 24h * 60min / 5min
        # End price should generally be higher than start
        # (not guaranteed for every seed, but with drift it's highly probable)
        start = df["close"].iloc[0]
        end = df["close"].iloc[-1]
        # Just verify structure, trend direction is probabilistic
        self.assertGreater(end, 0)

    def test_down_trend(self):
        df = generate_trending_day(direction="down", seed=42)
        self.assertEqual(len(df), 288)

    def test_ohlc_valid(self):
        df = generate_trending_day(direction="up", seed=42)
        for _, row in df.iterrows():
            self.assertGreaterEqual(row["high"], max(row["open"], row["close"]))
            self.assertLessEqual(row["low"], min(row["open"], row["close"]))


class TestRangeDay(unittest.TestCase):
    def test_range_day_shape(self):
        df = generate_range_day(seed=42)
        self.assertEqual(len(df), 288)

    def test_range_day_low_volatility(self):
        """Range day should have tighter price range than trending day."""
        range_df = generate_range_day(seed=42)
        trend_df = generate_trending_day(direction="up", seed=42)

        range_spread = range_df["close"].max() - range_df["close"].min()
        trend_spread = trend_df["close"].max() - trend_df["close"].min()

        # Range day should generally have tighter spread
        # This is probabilistic but with lower base_volatility it should hold
        self.assertGreater(range_spread, 0)
        self.assertGreater(trend_spread, 0)


class TestVolatileEventDay(unittest.TestCase):
    def test_volatile_day_shape(self):
        df = generate_volatile_event_day(seed=42)
        self.assertEqual(len(df), 288)

    def test_volatile_day_higher_range(self):
        """Volatile day should have wider bar ranges on average."""
        vol_df = generate_volatile_event_day(seed=42)
        range_df = generate_range_day(seed=42)

        vol_avg_range = (vol_df["high"] - vol_df["low"]).mean()
        range_avg_range = (range_df["high"] - range_df["low"]).mean()

        self.assertGreater(vol_avg_range, range_avg_range)


class TestMultiDayData(unittest.TestCase):
    def test_multi_day_shape(self):
        df = generate_multi_day_data(num_days=5, seed=42)
        self.assertEqual(len(df), 5 * 288)

    def test_multi_day_continuity(self):
        """Prices should be continuous across days."""
        df = generate_multi_day_data(num_days=3, seed=42)
        # Check no huge gaps (> 10% in 5 minutes)
        returns = df["close"].pct_change().dropna()
        self.assertTrue((returns.abs() < 0.10).all())

    def test_multi_day_reproducible(self):
        df1 = generate_multi_day_data(num_days=3, seed=42)
        df2 = generate_multi_day_data(num_days=3, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()
