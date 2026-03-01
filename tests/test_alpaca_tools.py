from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from gold_trading_one_trade_per_day.tools import alpaca_tools


class TestAlpacaTools(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(alpaca_tools.os.environ)

    def tearDown(self) -> None:
        alpaca_tools.os.environ.clear()
        alpaca_tools.os.environ.update(self._old_env)

    def test_macro_proxy_returns_tolerates_empty_symbol(self):
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        )

        def fake_fetch(symbol: str, lookback_minutes: int = 30, allow_mock: bool = True):
            if symbol == "VXX":
                raise RuntimeError("oanda_bars_empty")
            return bars

        with patch.object(alpaca_tools, "fetch_recent_bars", side_effect=fake_fetch):
            out = alpaca_tools.fetch_macro_proxy_returns(allow_mock=False)

        self.assertIn("SPY", out)
        self.assertIn("VXX", out)
        self.assertEqual(out["VXX"], 0.0)
        self.assertAlmostEqual(out["SPY"], 0.01)

    def test_fetch_latest_quote_falls_back_to_mock_without_credentials(self):
        alpaca_tools.os.environ.pop("OANDA_API_TOKEN", None)
        alpaca_tools.os.environ.pop("OANDA_ACCOUNT_ID", None)
        bid, ask = alpaca_tools.fetch_latest_quote(symbol="XAU_USD", allow_mock=True)
        self.assertGreater(bid, 0)
        self.assertGreaterEqual(ask, bid)

    def test_fetch_recent_bars_returns_mock_without_credentials(self):
        alpaca_tools.os.environ.pop("OANDA_API_TOKEN", None)
        alpaca_tools.os.environ.pop("OANDA_ACCOUNT_ID", None)
        bars = alpaca_tools.fetch_recent_bars(symbol="XAU_USD", lookback_minutes=30, allow_mock=True)
        self.assertEqual(list(bars.columns), ["open", "high", "low", "close", "volume"])
        self.assertGreater(len(bars), 0)


if __name__ == "__main__":
    unittest.main()
