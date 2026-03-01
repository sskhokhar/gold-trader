"""Tests for dual-mode trading: mode detection, dollar targets, event briefing schema."""
from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone

from gold_trading_one_trade_per_day.calendar_service import DailyCalendar, EconomicEvent
from gold_trading_one_trade_per_day.main import determine_trading_mode
from gold_trading_one_trade_per_day.risk_engine import RiskConfig, RiskEngine, load_risk_config
from gold_trading_one_trade_per_day.schemas import (
    DailyState,
    EventBriefingReport,
)


class TestDetermineTradingMode(unittest.TestCase):
    def _make_calendar(
        self,
        minutes_until_event: int,
        impact: str = "high",
    ) -> DailyCalendar:
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        release = now + timedelta(minutes=minutes_until_event)
        ev = EconomicEvent(
            event_id="test-ev",
            name="Non-Farm Payrolls",
            release_time=release,
            impact=impact,
        )
        has_high = impact == "high"
        return DailyCalendar(
            date=today,
            events=[ev],
            has_high_impact=has_high,
            next_event=ev,
            fetched_at=now,
        )

    def _empty_calendar(self) -> DailyCalendar:
        now = datetime.now(timezone.utc)
        return DailyCalendar(
            date=now.date().isoformat(),
            events=[],
            has_high_impact=False,
            next_event=None,
            fetched_at=now,
        )

    def test_daily_scalp_when_no_high_impact(self):
        cal = self._empty_calendar()
        mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        self.assertEqual(mode, "daily_scalp")

    def test_daily_scalp_well_before_event(self):
        # 120 minutes before event — outside default 5-min pre-window
        cal = self._make_calendar(minutes_until_event=120)
        mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        self.assertEqual(mode, "daily_scalp")

    def test_spike_within_pre_event_window(self):
        # 3 minutes before event — inside default 5-min pre-window
        cal = self._make_calendar(minutes_until_event=3)
        old = os.environ.get("SPIKE_MODE_ENABLED")
        os.environ["SPIKE_MODE_ENABLED"] = "true"
        try:
            mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        finally:
            if old is None:
                os.environ.pop("SPIKE_MODE_ENABLED", None)
            else:
                os.environ["SPIKE_MODE_ENABLED"] = old
        self.assertEqual(mode, "spike")

    def test_spike_during_post_event_window(self):
        # 10 minutes AFTER event — inside default 30-min post-window
        cal = self._make_calendar(minutes_until_event=-10)
        old = os.environ.get("SPIKE_MODE_ENABLED")
        os.environ["SPIKE_MODE_ENABLED"] = "true"
        try:
            mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        finally:
            if old is None:
                os.environ.pop("SPIKE_MODE_ENABLED", None)
            else:
                os.environ["SPIKE_MODE_ENABLED"] = old
        self.assertEqual(mode, "spike")

    def test_daily_scalp_after_post_event_window(self):
        # 45 minutes after event — outside default 30-min post-window
        cal = self._make_calendar(minutes_until_event=-45)
        mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        self.assertEqual(mode, "daily_scalp")

    def test_daily_scalp_medium_impact_not_spike(self):
        # Medium impact events don't trigger spike mode
        cal = self._make_calendar(minutes_until_event=3, impact="medium")
        mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        self.assertEqual(mode, "daily_scalp")

    def test_spike_mode_disabled_env(self):
        cal = self._make_calendar(minutes_until_event=3)
        old = os.environ.get("SPIKE_MODE_ENABLED")
        os.environ["SPIKE_MODE_ENABLED"] = "false"
        try:
            mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        finally:
            if old is None:
                os.environ.pop("SPIKE_MODE_ENABLED", None)
            else:
                os.environ["SPIKE_MODE_ENABLED"] = old
        self.assertEqual(mode, "daily_scalp")

    def test_custom_pre_event_minutes(self):
        """Custom SPIKE_PRE_EVENT_MINUTES=15 should activate spike at -10 min."""
        cal = self._make_calendar(minutes_until_event=10)
        old_spike = os.environ.get("SPIKE_MODE_ENABLED")
        old_pre = os.environ.get("SPIKE_PRE_EVENT_MINUTES")
        os.environ["SPIKE_MODE_ENABLED"] = "true"
        os.environ["SPIKE_PRE_EVENT_MINUTES"] = "15"
        try:
            mode = determine_trading_mode(cal, datetime.now(timezone.utc))
        finally:
            if old_spike is None:
                os.environ.pop("SPIKE_MODE_ENABLED", None)
            else:
                os.environ["SPIKE_MODE_ENABLED"] = old_spike
            if old_pre is None:
                os.environ.pop("SPIKE_PRE_EVENT_MINUTES", None)
            else:
                os.environ["SPIKE_PRE_EVENT_MINUTES"] = old_pre
        self.assertEqual(mode, "spike")


class TestDollarPnlTargets(unittest.TestCase):
    def _make_state(self, start_equity: float = 1000.0) -> DailyState:
        return DailyState(
            day="2026-03-07",
            day_start_equity=start_equity,
            equity_hwm=start_equity,
            current_equity=start_equity,
        )

    def test_dollar_pnl_tracked(self):
        engine = RiskEngine(config=RiskConfig())
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=1010.0)
        self.assertAlmostEqual(state.dollar_pnl, 10.0, places=4)

    def test_dollar_profit_target_hard_lock(self):
        cfg = RiskConfig(daily_profit_target_usd=10.0)
        engine = RiskEngine(config=cfg)
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=1010.0)
        self.assertTrue(state.hard_lock)
        self.assertEqual(state.last_lock_reason, "dollar_profit_target_reached")

    def test_dollar_profit_target_not_triggered_below(self):
        cfg = RiskConfig(daily_profit_target_usd=10.0)
        engine = RiskEngine(config=cfg)
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=1005.0)
        self.assertFalse(state.hard_lock)

    def test_dollar_loss_limit_hard_lock(self):
        cfg = RiskConfig(daily_loss_limit_usd=5.0)
        engine = RiskEngine(config=cfg)
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=994.0)
        self.assertTrue(state.hard_lock)
        self.assertEqual(state.last_lock_reason, "dollar_loss_limit_reached")

    def test_dollar_loss_limit_not_triggered_above(self):
        cfg = RiskConfig(daily_loss_limit_usd=5.0)
        engine = RiskEngine(config=cfg)
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=996.0)
        self.assertFalse(state.hard_lock)

    def test_dollar_targets_none_does_not_lock(self):
        """No dollar targets: pct-based locks still work, no dollar lock."""
        cfg = RiskConfig(
            daily_profit_target_usd=None,
            daily_loss_limit_usd=None,
            daily_hard_lock_pct=0.015,
        )
        engine = RiskEngine(config=cfg)
        state = self._make_state(start_equity=1000.0)
        engine.update_locks(state, current_equity=1005.0)
        self.assertFalse(state.hard_lock)

    def test_dollar_profit_target_env(self):
        """DAILY_PROFIT_TARGET_USD env var sets the target."""
        old = os.environ.get("DAILY_PROFIT_TARGET_USD")
        os.environ["DAILY_PROFIT_TARGET_USD"] = "50.0"
        try:
            cfg, source = load_risk_config()
            self.assertEqual(source, "env")
            self.assertAlmostEqual(cfg.daily_profit_target_usd, 50.0)
        finally:
            if old is None:
                os.environ.pop("DAILY_PROFIT_TARGET_USD", None)
            else:
                os.environ["DAILY_PROFIT_TARGET_USD"] = old

    def test_dollar_loss_limit_env(self):
        """DAILY_LOSS_LIMIT_USD env var sets the limit."""
        old = os.environ.get("DAILY_LOSS_LIMIT_USD")
        os.environ["DAILY_LOSS_LIMIT_USD"] = "25.0"
        try:
            cfg, source = load_risk_config()
            self.assertEqual(source, "env")
            self.assertAlmostEqual(cfg.daily_loss_limit_usd, 25.0)
        finally:
            if old is None:
                os.environ.pop("DAILY_LOSS_LIMIT_USD", None)
            else:
                os.environ["DAILY_LOSS_LIMIT_USD"] = old

    def test_daily_state_has_dollar_pnl_field(self):
        state = DailyState(
            day="2026-03-07",
            day_start_equity=1000.0,
            equity_hwm=1000.0,
            current_equity=1000.0,
        )
        self.assertEqual(state.dollar_pnl, 0.0)


class TestEventBriefingReportSchema(unittest.TestCase):
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def test_event_briefing_report_valid(self):
        report = EventBriefingReport(
            event_name="Non-Farm Payrolls",
            release_time=self._now(),
            impact="high",
            consensus="+180k",
            previous="+200k",
            actual="+120k",
            surprise_direction="dovish_miss",
            gold_bias="BUY",
            rationale=["weak jobs → rate cuts more likely → gold up"],
            confidence=0.8,
            generated_at=self._now(),
        )
        self.assertEqual(report.event_name, "Non-Farm Payrolls")
        self.assertEqual(report.surprise_direction, "dovish_miss")
        self.assertEqual(report.gold_bias, "BUY")
        self.assertAlmostEqual(report.confidence, 0.8)

    def test_event_briefing_report_optional_fields_none(self):
        report = EventBriefingReport(
            event_name="FOMC Rate Decision",
            release_time=self._now(),
            impact="high",
            confidence=0.5,
            generated_at=self._now(),
        )
        self.assertIsNone(report.consensus)
        self.assertIsNone(report.actual)
        self.assertIsNone(report.surprise_direction)
        self.assertIsNone(report.gold_bias)

    def test_event_briefing_confidence_bounds(self):
        with self.assertRaises(Exception):
            EventBriefingReport(
                event_name="Test",
                release_time=self._now(),
                impact="high",
                confidence=1.5,  # > 1.0 — invalid
                generated_at=self._now(),
            )
        with self.assertRaises(Exception):
            EventBriefingReport(
                event_name="Test",
                release_time=self._now(),
                impact="high",
                confidence=-0.1,  # < 0 — invalid
                generated_at=self._now(),
            )

    def test_event_briefing_report_timezone_coercion(self):
        naive_dt = datetime(2026, 3, 7, 13, 30, 0)  # no tzinfo
        report = EventBriefingReport(
            event_name="CPI",
            release_time=naive_dt,
            impact="high",
            confidence=0.7,
            generated_at=naive_dt,
        )
        self.assertIsNotNone(report.release_time.tzinfo)
        self.assertIsNotNone(report.generated_at.tzinfo)

    def test_event_briefing_report_serialization(self):
        report = EventBriefingReport(
            event_name="PPI",
            release_time=self._now(),
            impact="medium",
            confidence=0.6,
            generated_at=self._now(),
            rationale=["inflation data inline with expectations"],
        )
        json_str = report.model_dump_json()
        self.assertIn("PPI", json_str)
        self.assertIn("briefing_id", json_str)


class TestShouldWakeAiTradingMode(unittest.TestCase):
    """Tests for the trading_mode parameter in should_wake_ai."""

    def _make_snapshot_and_state(self):
        import pandas as pd
        from gold_trading_one_trade_per_day.event_trigger import build_feature_snapshot
        from gold_trading_one_trade_per_day.schemas import DataSource

        ts = datetime(2026, 2, 25, 10, 15, tzinfo=timezone.utc)
        idx = pd.date_range(end=ts, periods=60, freq="1min")
        base = 2900.0
        values = [base + i * 0.01 for i in range(59)] + [base + 2.0]
        close = pd.Series(values, index=idx)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = pd.Series([v + 0.1 for v in values[:-1]] + [values[-1] + 1.0], index=idx)
        low = pd.Series([v - 0.1 for v in values[:-1]] + [values[-1] - 1.0], index=idx)
        volume = pd.Series([10000] * 59 + [50000], index=idx)
        bars = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )
        snapshot = build_feature_snapshot(
            bars, bid=2900.0, ask=2900.10, timestamp=ts, data_source=DataSource.STREAM
        )
        state = DailyState(
            day="2026-02-25",
            day_start_equity=100000,
            equity_hwm=100000,
            current_equity=100000,
        )
        return snapshot, state

    def test_daily_scalp_blocked_by_macro_event(self):
        from gold_trading_one_trade_per_day.event_trigger import should_wake_ai

        snap, state = self._make_snapshot_and_state()
        ok, reason, _ = should_wake_ai(
            snap,
            state,
            macro_event_active=True,
            macro_event_label="NFP",
            trading_mode="daily_scalp",
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "macro_event_window")

    def test_spike_mode_not_blocked_by_macro_event(self):
        from gold_trading_one_trade_per_day.event_trigger import should_wake_ai

        snap, state = self._make_snapshot_and_state()
        # In spike mode, macro_event_active should NOT block trading
        ok, reason, ctx = should_wake_ai(
            snap,
            state,
            macro_event_active=True,
            macro_event_label="NFP",
            trading_mode="spike",
        )
        # May still fail on volume/VWAP thresholds but not on macro_event_window
        self.assertNotEqual(reason, "macro_event_window")

    def test_trading_mode_in_context(self):
        from gold_trading_one_trade_per_day.event_trigger import should_wake_ai

        snap, state = self._make_snapshot_and_state()
        _, _, ctx = should_wake_ai(snap, state, trading_mode="spike")
        self.assertEqual(ctx["trading_mode"], "spike")


if __name__ == "__main__":
    unittest.main()
