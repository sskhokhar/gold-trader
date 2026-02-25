from __future__ import annotations

import unittest

from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor


class TestMarketStream(unittest.TestCase):
    def test_health_without_credentials(self):
        sensor = MarketStreamSensor(symbol="GLD", stale_seconds=1)
        health = sensor.health()
        self.assertFalse(sensor.enabled)  # no creds in unit test env
        self.assertFalse(health.connected)
        self.assertTrue(health.stale)
        self.assertIsNone(sensor.latest_inputs())


if __name__ == "__main__":
    unittest.main()
