from __future__ import annotations

import os
import unittest

from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor


class TestMarketStream(unittest.TestCase):
    def test_health_without_credentials(self):
        old_key = os.environ.get("ALPACA_API_KEY")
        old_secret = os.environ.get("ALPACA_SECRET_KEY")
        os.environ["ALPACA_API_KEY"] = ""
        os.environ["ALPACA_SECRET_KEY"] = ""
        try:
            sensor = MarketStreamSensor(symbol="GLD", stale_seconds=1)
            health = sensor.health()
            self.assertFalse(sensor.enabled)
            self.assertFalse(health.connected)
            self.assertTrue(health.stale)
            self.assertIsNone(sensor.latest_inputs())
        finally:
            if old_key is None:
                os.environ.pop("ALPACA_API_KEY", None)
            else:
                os.environ["ALPACA_API_KEY"] = old_key
            if old_secret is None:
                os.environ.pop("ALPACA_SECRET_KEY", None)
            else:
                os.environ["ALPACA_SECRET_KEY"] = old_secret


if __name__ == "__main__":
    unittest.main()
