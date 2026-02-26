"""Tests for the technical analysis indicators module."""

import sys
import os
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_trading_one_trade_per_day.indicators import (
    IndicatorSnapshot,
    MACDResult,
    BollingerBands,
    StochasticResult,
    ADXResult,
    PivotLevels,
    ema,
    ema_cross,
    rsi,
    macd,
    true_range,
    atr,
    bollinger_bands,
    vwap,
    vwap_bands,
    stochastic,
    adx,
    camarilla_pivots,
    compute_indicator_snapshot,
)
from gold_trading_one_trade_per_day.synthetic_data import generate_gold_bars


def _make_bars(n=100, seed=42):
    """Generate test bars."""
    return generate_gold_bars(num_bars=n, seed=seed)


class TestEMA(unittest.TestCase):
    def test_ema_length(self):
        s = pd.Series(range(50), dtype=float)
        result = ema(s, 9)
        self.assertEqual(len(result), 50)

    def test_ema_last_value_reasonable(self):
        s = pd.Series([100.0] * 20)
        result = ema(s, 9)
        self.assertAlmostEqual(result.iloc[-1], 100.0, places=5)

    def test_ema_trending_up(self):
        s = pd.Series(range(50), dtype=float)
        result = ema(s, 9)
        # EMA of a linearly increasing series should be below the latest value (lag)
        self.assertLess(result.iloc[-1], 49.0)
        self.assertGreater(result.iloc[-1], 40.0)


class TestEMACross(unittest.TestCase):
    def test_golden_cross(self):
        fast = pd.Series([1.0, 1.0, 2.0, 3.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0])
        result = ema_cross(fast, slow)
        # At index 2: fast went from 1 (below slow 2) to 2 (equal), not a cross
        # At index 3: fast went from 2 (equal to slow 2) to 3 (above slow 2)
        self.assertEqual(result.iloc[3], 1)

    def test_death_cross(self):
        fast = pd.Series([3.0, 3.0, 2.0, 1.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0])
        result = ema_cross(fast, slow)
        self.assertEqual(result.iloc[3], -1)

    def test_no_cross(self):
        fast = pd.Series([3.0, 3.0, 3.0, 3.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0])
        result = ema_cross(fast, slow)
        self.assertEqual(result.iloc[-1], 0)


class TestRSI(unittest.TestCase):
    def test_rsi_range(self):
        bars = _make_bars(100)
        result = rsi(bars["close"], 14)
        valid = result.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_rsi_flat_price(self):
        s = pd.Series([100.0] * 50)
        result = rsi(s, 14)
        # Flat price should give NaN or 50-ish (no gains or losses)
        # The result depends on implementation but should not be extreme
        valid = result.dropna()
        if len(valid) > 0:
            for v in valid:
                self.assertTrue(0 <= v <= 100)


class TestMACD(unittest.TestCase):
    def test_macd_structure(self):
        bars = _make_bars(100)
        result = macd(bars["close"])
        self.assertIsInstance(result, MACDResult)
        self.assertEqual(len(result.macd_line), 100)
        self.assertEqual(len(result.signal_line), 100)
        self.assertEqual(len(result.histogram), 100)

    def test_histogram_is_diff(self):
        bars = _make_bars(100)
        result = macd(bars["close"])
        diff = result.macd_line - result.signal_line
        np.testing.assert_array_almost_equal(
            result.histogram.values, diff.values, decimal=10
        )


class TestATR(unittest.TestCase):
    def test_atr_positive(self):
        bars = _make_bars(100)
        result = atr(bars["high"], bars["low"], bars["close"], 14)
        valid = result.dropna()
        self.assertTrue((valid > 0).all())

    def test_true_range_basic(self):
        result = true_range(
            pd.Series([105.0, 110.0]),
            pd.Series([95.0, 100.0]),
            pd.Series([100.0, 108.0]),
        )
        # First bar: high - low = 10
        self.assertAlmostEqual(result.iloc[0], 10.0)


class TestBollingerBands(unittest.TestCase):
    def test_bb_structure(self):
        bars = _make_bars(100)
        result = bollinger_bands(bars["close"], 20, 2.0)
        self.assertIsInstance(result, BollingerBands)
        self.assertEqual(len(result.upper), 100)
        self.assertEqual(len(result.lower), 100)

    def test_upper_above_lower(self):
        bars = _make_bars(100)
        result = bollinger_bands(bars["close"], 20, 2.0)
        valid_mask = ~(result.upper.isna() | result.lower.isna())
        self.assertTrue((result.upper[valid_mask] >= result.lower[valid_mask]).all())

    def test_pct_b_range(self):
        bars = _make_bars(100)
        result = bollinger_bands(bars["close"], 20, 2.0)
        valid = result.pct_b.dropna()
        # Most values should be between -0.5 and 1.5 for normal data
        self.assertTrue((valid > -2.0).all())
        self.assertTrue((valid < 3.0).all())


class TestVWAP(unittest.TestCase):
    def test_vwap_reasonable(self):
        bars = _make_bars(100)
        result = vwap(bars["high"], bars["low"], bars["close"], bars["volume"])
        valid = result.dropna()
        self.assertTrue((valid > 0).all())
        # VWAP should be between min and max of prices
        min_price = bars["low"].min()
        max_price = bars["high"].max()
        self.assertTrue((valid >= min_price * 0.99).all())
        self.assertTrue((valid <= max_price * 1.01).all())

    def test_vwap_bands(self):
        bars = _make_bars(100)
        vwap_line, upper, lower = vwap_bands(
            bars["high"], bars["low"], bars["close"], bars["volume"]
        )
        valid_mask = ~(upper.isna() | lower.isna())
        self.assertTrue((upper[valid_mask] >= lower[valid_mask]).all())


class TestStochastic(unittest.TestCase):
    def test_stochastic_range(self):
        bars = _make_bars(100)
        result = stochastic(bars["high"], bars["low"], bars["close"])
        self.assertIsInstance(result, StochasticResult)
        valid_k = result.k.dropna()
        self.assertTrue((valid_k >= 0).all())
        self.assertTrue((valid_k <= 100).all())


class TestADX(unittest.TestCase):
    def test_adx_positive(self):
        bars = _make_bars(100)
        result = adx(bars["high"], bars["low"], bars["close"])
        self.assertIsInstance(result, ADXResult)
        valid = result.adx.dropna()
        self.assertTrue((valid >= 0).all())


class TestCamarillaPivots(unittest.TestCase):
    def test_pivot_levels_order(self):
        result = camarilla_pivots(2660.0, 2640.0, 2650.0)
        self.assertIsInstance(result, PivotLevels)
        # S3 < S2 < S1 < Pivot < R1 < R2 < R3
        self.assertLess(result.s3, result.s2)
        self.assertLess(result.s2, result.s1)
        self.assertLess(result.s1, result.pivot)
        self.assertLess(result.pivot, result.r1)
        self.assertLess(result.r1, result.r2)
        self.assertLess(result.r2, result.r3)


class TestComputeIndicatorSnapshot(unittest.TestCase):
    def test_snapshot_complete(self):
        bars = _make_bars(100)
        snapshot = compute_indicator_snapshot(bars)
        self.assertIsInstance(snapshot, IndicatorSnapshot)
        # Verify key fields are populated
        self.assertGreater(snapshot.ema_9, 0)
        self.assertGreater(snapshot.ema_21, 0)
        self.assertGreater(snapshot.ema_50, 0)
        self.assertIn(snapshot.ema_trend, ("bullish", "bearish", "neutral"))
        self.assertTrue(0 <= snapshot.rsi_14 <= 100)
        self.assertTrue(0 <= snapshot.rsi_7 <= 100)
        self.assertIn(snapshot.rsi_zone, ("overbought", "oversold", "neutral"))
        self.assertGreater(snapshot.atr_14, 0)
        self.assertIn(snapshot.price_vs_vwap, ("above", "below", "at"))
        self.assertIn(snapshot.trend_strength, ("strong", "moderate", "weak", "no_trend"))
        self.assertTrue(-1.0 <= snapshot.confluence_score <= 1.0)

    def test_snapshot_to_dict(self):
        bars = _make_bars(100)
        snapshot = compute_indicator_snapshot(bars)
        d = snapshot.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("ema_9", d)
        self.assertIn("confluence_score", d)

    def test_snapshot_min_bars(self):
        """Should work with as few as 50 bars."""
        bars = _make_bars(55)
        snapshot = compute_indicator_snapshot(bars)
        self.assertIsInstance(snapshot, IndicatorSnapshot)


if __name__ == "__main__":
    unittest.main()
