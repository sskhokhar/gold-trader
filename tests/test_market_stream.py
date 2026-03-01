from __future__ import annotations

import os
import unittest

from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor


class TestMarketStream(unittest.TestCase):
    def test_health_without_credentials(self):
        old_token = os.environ.get("OANDA_API_TOKEN")
        old_account = os.environ.get("OANDA_ACCOUNT_ID")
        os.environ["OANDA_API_TOKEN"] = ""
        os.environ["OANDA_ACCOUNT_ID"] = ""
        try:
            sensor = MarketStreamSensor(symbol="XAU_USD", stale_seconds=1)
            health = sensor.health()
            self.assertFalse(sensor.enabled)
            self.assertFalse(health.connected)
            self.assertTrue(health.stale)
            self.assertIsNone(sensor.latest_inputs())
        finally:
            if old_token is None:
                os.environ.pop("OANDA_API_TOKEN", None)
            else:
                os.environ["OANDA_API_TOKEN"] = old_token
            if old_account is None:
                os.environ.pop("OANDA_ACCOUNT_ID", None)
            else:
                os.environ["OANDA_ACCOUNT_ID"] = old_account


if __name__ == "__main__":
    unittest.main()
