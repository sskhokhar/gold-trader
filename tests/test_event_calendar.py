from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

from gold_trading_one_trade_per_day.event_calendar import EventCalendar


class TestEventCalendar(unittest.TestCase):
    def test_blocks_inside_window(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/events.yaml"
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "events:\n"
                    '  - label: CPI\n'
                    '    timestamp_utc: "2026-02-25T14:00:00+00:00"\n'
                )

            cal = EventCalendar(config_path=path, block_minutes=15)
            inside = datetime.fromisoformat("2026-02-25T14:10:00+00:00")
            outside = datetime.fromisoformat("2026-02-25T14:30:00+00:00")

            blocked, label = cal.is_blocked(inside)
            self.assertTrue(blocked)
            self.assertEqual(label, "CPI")

            blocked2, _ = cal.is_blocked(outside)
            self.assertFalse(blocked2)


if __name__ == "__main__":
    unittest.main()
