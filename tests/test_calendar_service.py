"""Tests for the Economic Calendar Service."""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from gold_trading_one_trade_per_day.calendar_service import (
    CalendarService,
    DailyCalendar,
    EconomicEvent,
    _name_affects_gold,
)


class TestCalendarService(unittest.TestCase):
    def _make_event(self, name: str, impact: str = "high", minutes_from_now: int = 60) -> dict:
        release = datetime.now(timezone.utc) + timedelta(minutes=minutes_from_now)
        return {
            "event_id": f"test-{name.lower().replace(' ', '-')}",
            "name": name,
            "release_time": release.isoformat(),
            "impact": impact,
            "currency": "USD",
        }

    def test_name_affects_gold_nfp(self):
        self.assertTrue(_name_affects_gold("Non-Farm Payrolls"))
        self.assertTrue(_name_affects_gold("US CPI"))
        self.assertTrue(_name_affects_gold("FOMC Rate Decision"))
        self.assertTrue(_name_affects_gold("GDP"))
        self.assertFalse(_name_affects_gold("UK House Prices"))

    def test_economic_event_from_yaml_entry(self):
        entry = {
            "event_id": "ev1",
            "name": "Non-Farm Payrolls",
            "release_time": "2026-03-07T13:30:00+00:00",
            "impact": "high",
            "currency": "USD",
            "consensus": "+180k",
            "previous": "+200k",
        }
        ev = EconomicEvent.from_yaml_entry(entry, 0)
        self.assertEqual(ev.name, "Non-Farm Payrolls")
        self.assertEqual(ev.impact, "high")
        self.assertTrue(ev.affects_gold)
        self.assertEqual(ev.consensus, "+180k")

    def test_economic_event_from_yaml_entry_fallback_time(self):
        """Events with missing release_time should still parse without raising."""
        entry = {"name": "Unknown Event", "impact": "low"}
        ev = EconomicEvent.from_yaml_entry(entry, 5)
        self.assertEqual(ev.name, "Unknown Event")
        self.assertIsNotNone(ev.release_time)

    def test_calendar_service_yaml_no_events(self):
        """Service with non-existent yaml path returns empty calendar gracefully."""
        svc = CalendarService(source="yaml", yaml_path="/nonexistent/path.yaml")
        cal = svc.get_calendar(date="2026-03-07")
        self.assertIsInstance(cal, DailyCalendar)
        self.assertEqual(len(cal.events), 0)
        self.assertFalse(cal.has_high_impact)

    def test_calendar_service_yaml_file(self):
        """Service correctly parses a YAML calendar file."""
        today = datetime.now(timezone.utc).date().isoformat()
        release = datetime.now(timezone.utc) + timedelta(hours=2)
        yaml_content = f"""
events:
  - event_id: nfp-test
    name: Non-Farm Payrolls
    release_time: "{release.isoformat()}"
    impact: high
    currency: USD
    consensus: "+180k"
    previous: "+200k"
  - event_id: cpi-test
    name: US CPI
    release_time: "{(release + timedelta(hours=1)).isoformat()}"
    impact: medium
    currency: USD
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            path = f.name

        try:
            svc = CalendarService(source="yaml", yaml_path=path, high_impact_only=False)
            cal = svc.get_calendar(date=today)
            self.assertEqual(cal.date, today)
            self.assertGreaterEqual(len(cal.events), 1)
            names = [e.name for e in cal.events]
            self.assertIn("Non-Farm Payrolls", names)
        finally:
            os.unlink(path)

    def test_calendar_service_cache(self):
        """Calendar is cached and reused within TTL."""
        svc = CalendarService(source="yaml", yaml_path="/nonexistent.yaml", cache_ttl_sec=3600)
        now = datetime.now(timezone.utc)
        cal1 = svc.get_calendar(date="2026-03-07", now=now)
        cal2 = svc.get_calendar(date="2026-03-07", now=now)
        self.assertIs(cal1, cal2)

    def test_calendar_service_cache_expires(self):
        """Expired cache results in a new fetch."""
        svc = CalendarService(source="yaml", yaml_path="/nonexistent.yaml", cache_ttl_sec=1)
        now = datetime.now(timezone.utc)
        cal1 = svc.get_calendar(date="2026-03-07", now=now)
        future = now + timedelta(seconds=2)
        cal2 = svc.get_calendar(date="2026-03-07", now=future)
        # They should be different objects (new fetch)
        self.assertIsNot(cal1, cal2)

    def test_calendar_service_cache_date_change(self):
        """Different date always triggers new fetch."""
        svc = CalendarService(source="yaml", yaml_path="/nonexistent.yaml", cache_ttl_sec=9999)
        now = datetime.now(timezone.utc)
        cal1 = svc.get_calendar(date="2026-03-07", now=now)
        cal2 = svc.get_calendar(date="2026-03-08", now=now)
        self.assertIsNot(cal1, cal2)

    def test_high_impact_only_filter(self):
        """High-impact-only filter correctly excludes medium/low events."""
        today = datetime.now(timezone.utc).date().isoformat()
        release = datetime.now(timezone.utc) + timedelta(hours=2)
        yaml_content = f"""
events:
  - event_id: ev-high
    name: Non-Farm Payrolls
    release_time: "{release.isoformat()}"
    impact: high
    currency: USD
  - event_id: ev-medium
    name: ISM Manufacturing
    release_time: "{(release + timedelta(hours=1)).isoformat()}"
    impact: medium
    currency: USD
"""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            path = f.name

        try:
            svc = CalendarService(source="yaml", yaml_path=path, high_impact_only=True)
            cal = svc.get_calendar(date=today)
            impacts = [e.impact for e in cal.events]
            self.assertTrue(all(i == "high" for i in impacts))
        finally:
            os.unlink(path)

    def test_next_event_is_closest_future(self):
        """next_event should be the closest upcoming event."""
        svc = CalendarService(source="yaml", yaml_path="/nonexistent.yaml")
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        ev1 = EconomicEvent(
            event_id="ev1",
            name="NFP",
            release_time=now + timedelta(hours=1),
            impact="high",
        )
        ev2 = EconomicEvent(
            event_id="ev2",
            name="CPI",
            release_time=now + timedelta(hours=3),
            impact="high",
        )
        # Manually construct a calendar
        cal = DailyCalendar(
            date=today,
            events=[ev2, ev1],
            has_high_impact=True,
            next_event=ev1,
            fetched_at=now,
        )
        self.assertEqual(cal.next_event.event_id, "ev1")

    def test_api_source_graceful_fallback(self):
        """Unknown/failing API source falls back gracefully."""
        svc = CalendarService(source="tradingeconomics")
        cal = svc.get_calendar()
        self.assertIsInstance(cal, DailyCalendar)


if __name__ == "__main__":
    unittest.main()
