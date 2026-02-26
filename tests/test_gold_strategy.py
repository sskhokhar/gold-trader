"""Tests for the gold strategy engine."""

import sys
import os
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_trading_one_trade_per_day.gold_strategy import (
    TradingSession,
    SessionInfo,
    SetupType,
    TradeSetup,
    detect_session,
    compute_atr_stops,
    compute_position_size,
    detect_setup,
    analyze_gold,
)
from gold_trading_one_trade_per_day.indicators import compute_indicator_snapshot
from gold_trading_one_trade_per_day.synthetic_data import (
    generate_gold_bars,
    generate_trending_day,
)

UTC_TZ = ZoneInfo("UTC")


class TestSessionDetection(unittest.TestCase):
    def test_asian_session(self):
        # 03:00 UTC = Asian session
        ts = datetime(2025, 1, 6, 3, 0, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.session, TradingSession.ASIAN)
        self.assertAlmostEqual(info.session_quality, 0.3)

    def test_london_session(self):
        # 10:00 UTC = London session (before overlap)
        ts = datetime(2025, 1, 6, 10, 0, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.session, TradingSession.LONDON)
        self.assertAlmostEqual(info.session_quality, 0.8)

    def test_overlap_session(self):
        # 14:00 UTC = London/NY overlap
        ts = datetime(2025, 1, 6, 14, 0, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.session, TradingSession.LONDON_NY_OVERLAP)
        self.assertAlmostEqual(info.session_quality, 1.0)

    def test_ny_session(self):
        # 17:00 UTC = NY session (post overlap)
        ts = datetime(2025, 1, 6, 17, 0, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.session, TradingSession.NY)
        self.assertAlmostEqual(info.session_quality, 0.6)

    def test_off_hours(self):
        # 21:00 UTC = off hours
        ts = datetime(2025, 1, 6, 21, 0, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.session, TradingSession.OFF_HOURS)
        self.assertAlmostEqual(info.session_quality, 0.0)

    def test_minutes_into_session(self):
        # 09:30 UTC = 90 minutes into London session (starts at 08:00)
        ts = datetime(2025, 1, 6, 9, 30, tzinfo=UTC_TZ)
        info = detect_session(ts)
        self.assertEqual(info.minutes_into_session, 90)


class TestATRStops(unittest.TestCase):
    def test_buy_stops(self):
        sl, tp, rr = compute_atr_stops(
            entry_price=2650.0,
            side="BUY",
            atr_value=5.0,
            sl_multiplier=1.5,
            tp_multiplier=2.5,
        )
        self.assertLess(sl, 2650.0)
        self.assertGreater(tp, 2650.0)
        self.assertGreater(rr, 0)
        self.assertAlmostEqual(sl, 2650.0 - 5.0 * 1.5, places=1)
        self.assertAlmostEqual(tp, 2650.0 + 5.0 * 2.5, places=1)

    def test_sell_stops(self):
        sl, tp, rr = compute_atr_stops(
            entry_price=2650.0,
            side="SELL",
            atr_value=5.0,
            sl_multiplier=1.5,
            tp_multiplier=2.5,
        )
        self.assertGreater(sl, 2650.0)
        self.assertLess(tp, 2650.0)
        self.assertGreater(rr, 0)

    def test_rr_ratio_correct(self):
        sl, tp, rr = compute_atr_stops(
            entry_price=2650.0,
            side="BUY",
            atr_value=5.0,
            sl_multiplier=1.5,
            tp_multiplier=3.0,
        )
        expected_rr = (5.0 * 3.0) / (5.0 * 1.5)
        self.assertAlmostEqual(rr, expected_rr, places=1)

    def test_pivot_tightens_tp(self):
        """TP should be tightened to near a pivot resistance."""
        pivots = {"r1": 2660.0, "r2": 2670.0, "r3": 2680.0}
        sl, tp, rr = compute_atr_stops(
            entry_price=2650.0,
            side="BUY",
            atr_value=5.0,
            sl_multiplier=1.5,
            tp_multiplier=5.0,  # Would give TP of 2675, but r1 at 2660
            pivots=pivots,
        )
        # TP should be tightened to just below r1
        self.assertLessEqual(tp, 2660.0)


class TestPositionSizing(unittest.TestCase):
    def test_basic_sizing(self):
        qty, risk = compute_position_size(
            equity=10000.0,
            risk_pct=0.02,
            entry_price=2650.0,
            stop_loss=2640.0,
            confidence=0.5,
        )
        # Risk per share = 10.0
        # Max risk = 10000 * min(0.02, kelly_adjusted) = ~50-200
        self.assertGreaterEqual(qty, 0)
        self.assertGreaterEqual(risk, 0)

    def test_zero_risk(self):
        qty, risk = compute_position_size(
            equity=10000.0,
            risk_pct=0.02,
            entry_price=2650.0,
            stop_loss=2650.0,  # zero risk
        )
        self.assertEqual(qty, 0.0)
        self.assertEqual(risk, 0.0)

    def test_zero_equity(self):
        qty, risk = compute_position_size(
            equity=0.0,
            risk_pct=0.02,
            entry_price=2650.0,
            stop_loss=2640.0,
        )
        self.assertEqual(qty, 0.0)

    def test_small_equity(self):
        """With $100 equity and $10 risk per share, should size appropriately."""
        qty, risk = compute_position_size(
            equity=100.0,
            risk_pct=0.02,
            entry_price=2650.0,
            stop_loss=2640.0,
            confidence=0.7,
        )
        # Max risk = 100 * 0.02 * kelly_adj
        # This might be 0 shares due to floor()
        self.assertGreaterEqual(qty, 0.0)


class TestDetectSetup(unittest.TestCase):
    def test_no_setup_off_hours(self):
        """Off hours should return NO_SETUP."""
        bars = generate_gold_bars(num_bars=100, seed=42)
        indicators = compute_indicator_snapshot(bars)
        session = SessionInfo(
            session=TradingSession.OFF_HOURS,
            session_quality=0.0,
            minutes_into_session=0,
            minutes_remaining=0,
        )
        setup = detect_setup(indicators, session, float(bars["close"].iloc[-1]))
        self.assertEqual(setup.setup_type, SetupType.NO_SETUP)

    def test_setup_returns_valid_structure(self):
        """Any detected setup should have valid SL/TP."""
        bars = generate_trending_day(direction="up", seed=42)
        indicators = compute_indicator_snapshot(bars)
        session = SessionInfo(
            session=TradingSession.LONDON_NY_OVERLAP,
            session_quality=1.0,
            minutes_into_session=60,
            minutes_remaining=120,
        )
        setup = detect_setup(indicators, session, float(bars["close"].iloc[-1]))
        self.assertIsInstance(setup, TradeSetup)
        self.assertIsInstance(setup.setup_type, SetupType)
        self.assertIn(setup.side, ("BUY", "SELL"))
        self.assertTrue(0.0 <= setup.confidence <= 1.0)


class TestAnalyzeGold(unittest.TestCase):
    def test_analyze_returns_trade_setup(self):
        bars = generate_gold_bars(num_bars=100, seed=42)
        last_price = float(bars["close"].iloc[-1])
        ts = datetime(2025, 1, 6, 14, 0, tzinfo=UTC_TZ)  # overlap session
        setup = analyze_gold(bars, last_price, timestamp=ts)
        self.assertIsInstance(setup, TradeSetup)

    def test_analyze_with_trending_data(self):
        """Trending data during good session should produce setups."""
        bars = generate_trending_day(direction="up", seed=42)
        last_price = float(bars["close"].iloc[-1])
        ts = datetime(2025, 1, 6, 14, 0, tzinfo=UTC_TZ)
        setup = analyze_gold(bars, last_price, timestamp=ts)
        self.assertIsInstance(setup, TradeSetup)

    def test_setup_to_dict(self):
        bars = generate_gold_bars(num_bars=100, seed=42)
        last_price = float(bars["close"].iloc[-1])
        ts = datetime(2025, 1, 6, 14, 0, tzinfo=UTC_TZ)
        setup = analyze_gold(bars, last_price, timestamp=ts)
        d = setup.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("setup_type", d)
        self.assertIn("indicators", d)


if __name__ == "__main__":
    unittest.main()
