from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd
from alpaca.data.enums import DataFeed

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
                raise RuntimeError("alpaca_bars_empty")
            return bars

        with patch.object(alpaca_tools, "fetch_recent_bars", side_effect=fake_fetch):
            out = alpaca_tools.fetch_macro_proxy_returns(allow_mock=False)

        self.assertIn("SPY", out)
        self.assertIn("VXX", out)
        self.assertEqual(out["VXX"], 0.0)
        self.assertAlmostEqual(out["SPY"], 0.01)

    def test_fetch_latest_quote_falls_back_across_feeds(self):
        class _Quote:
            def __init__(self, bid: float, ask: float):
                self.bid_price = bid
                self.ask_price = ask

        class _Client:
            def get_stock_latest_quote(self, req):
                if req.feed == DataFeed.IEX:
                    return {"GLD": _Quote(-1.0, 0.0)}
                return {"GLD": _Quote(201.01, 201.03)}

        alpaca_tools.os.environ["ALPACA_DATA_FEEDS"] = "IEX,SIP"
        with patch.object(alpaca_tools, "get_data_client", return_value=_Client()):
            bid, ask = alpaca_tools.fetch_latest_quote(symbol="GLD", allow_mock=False)
        self.assertEqual((bid, ask), (201.01, 201.03))

    def test_fetch_recent_bars_falls_back_across_feeds(self):
        class _Bars:
            def __init__(self, df: pd.DataFrame):
                self.df = df

        class _Client:
            def get_stock_bars(self, req):
                if req.feed == DataFeed.IEX:
                    return _Bars(pd.DataFrame())
                idx = pd.date_range("2026-02-26T12:00:00Z", periods=2, freq="1min")
                df = pd.DataFrame(
                    {
                        "open": [100.0, 101.0],
                        "high": [101.0, 102.0],
                        "low": [99.0, 100.0],
                        "close": [100.5, 101.5],
                        "volume": [1000.0, 1200.0],
                    },
                    index=idx,
                )
                return _Bars(df)

        alpaca_tools.os.environ["ALPACA_DATA_FEEDS"] = "IEX,SIP"
        with patch.object(alpaca_tools, "get_data_client", return_value=_Client()):
            bars = alpaca_tools.fetch_recent_bars(symbol="GLD", lookback_minutes=30, allow_mock=False)
        self.assertEqual(list(bars.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(bars), 2)


if __name__ == "__main__":
    unittest.main()
