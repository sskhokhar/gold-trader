from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def has_real_credentials() -> bool:
    key = _env("ALPACA_API_KEY")
    secret = _env("ALPACA_SECRET_KEY")
    return bool(key and secret and key != "your_alpaca_api_key_here")


def is_paper_mode() -> bool:
    return _env("ALPACA_PAPER", "true").lower() == "true"


def get_data_client() -> StockHistoricalDataClient | None:
    if not has_real_credentials():
        return None
    return StockHistoricalDataClient(_env("ALPACA_API_KEY"), _env("ALPACA_SECRET_KEY"))


def get_trading_client() -> TradingClient | None:
    if not has_real_credentials():
        return None
    return TradingClient(_env("ALPACA_API_KEY"), _env("ALPACA_SECRET_KEY"), paper=is_paper_mode())


def mock_bars(symbol: str = "GLD", periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(tz=timezone.utc), periods=periods, freq="1min")
    base = 200.0
    prices = [base]
    for i in range(1, periods):
        drift = 0.01 if i % 15 else 0.08
        prices.append(prices[-1] + drift)
    close = pd.Series(prices, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.05
    volume = pd.Series([10000 + (25000 if i % 20 == 0 else 0) for i in range(periods)], index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def fetch_recent_bars(
    symbol: str = "GLD",
    timeframe: TimeFrame | None = None,
    lookback_minutes: int = 180,
    allow_mock: bool = True,
) -> pd.DataFrame:
    client = get_data_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("alpaca_data_client_unavailable")
        return mock_bars(symbol=symbol)

    tf = timeframe or TimeFrame(1, TimeFrameUnit.Minute)
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=16)
    start_dt = end_dt - timedelta(minutes=lookback_minutes)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index().set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]].copy()
    except Exception:
        if not allow_mock:
            raise
        return mock_bars(symbol=symbol)


def fetch_latest_quote(symbol: str = "GLD", allow_mock: bool = True) -> tuple[float, float]:
    client = get_data_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("alpaca_data_client_unavailable")
        return 200.00, 200.01

    req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
    try:
        quotes = client.get_stock_latest_quote(req)
        quote = quotes[symbol]
        return float(quote.bid_price), float(quote.ask_price)
    except Exception:
        if not allow_mock:
            raise
        return 200.00, 200.01


def fetch_macro_proxy_returns(allow_mock: bool = True) -> dict[str, float]:
    proxies = ["SPY", "VXX", "UUP", "TLT"]
    out: dict[str, float] = {}
    for symbol in proxies:
        bars = fetch_recent_bars(symbol=symbol, lookback_minutes=30, allow_mock=allow_mock)
        if len(bars) < 2:
            out[symbol] = 0.0
            continue
        prev = float(bars.iloc[-2]["close"])
        last = float(bars.iloc[-1]["close"])
        out[symbol] = ((last - prev) / prev) if prev else 0.0
    return out


def fetch_account_equity(allow_mock: bool = True) -> float:
    client = get_trading_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("alpaca_trading_client_unavailable")
        return 100_000.0

    try:
        account = client.get_account()
        equity = account.equity or account.portfolio_value or "0"
        return float(equity)
    except Exception:
        if not allow_mock:
            raise
        return 100_000.0


class AlpacaDataToolInput(BaseModel):
    symbol: str = Field(default="GLD", description="Ticker symbol for analysis.")


class AlpacaDataTool(BaseTool):
    name: str = "Alpaca Market Snapshot Tool"
    description: str = (
        "Returns a compact market snapshot with latest price, quote spread, volume spike, "
        "and macro proxies for event-triggered GLD analysis."
    )
    args_schema = AlpacaDataToolInput

    def _run(self, symbol: str = "GLD") -> str:
        bars = fetch_recent_bars(symbol=symbol, lookback_minutes=60)
        bid, ask = fetch_latest_quote(symbol=symbol)
        macro = fetch_macro_proxy_returns()

        last = bars.iloc[-1]
        avg_volume = float(bars["volume"].tail(20).mean())
        volume_spike = float(last["volume"]) / max(avg_volume, 1.0)
        payload: dict[str, Any] = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_price": float(last["close"]),
            "bid": bid,
            "ask": ask,
            "spread": max(ask - bid, 0.0),
            "volume": float(last["volume"]),
            "avg_volume_20": avg_volume,
            "volume_spike_ratio": volume_spike,
            "macro_proxies": macro,
        }
        return json.dumps(payload)


class AlpacaExecutionTool(BaseTool):
    name: str = "Deterministic Execution Guard"
    description: str = (
        "Execution is blocked from direct LLM tool-calls. "
        "Orders must be routed through deterministic execution_service."
    )

    def _run(self, *args, **kwargs) -> str:
        return (
            "Direct execution denied. Use deterministic execution_service.py with "
            "validated ExecutionCommand and risk approval."
        )
