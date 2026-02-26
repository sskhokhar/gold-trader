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


def _parse_data_feed(value: str) -> DataFeed | None:
    normalized = (value or "").strip().upper()
    if normalized == "IEX":
        return DataFeed.IEX
    if normalized == "SIP":
        return DataFeed.SIP
    return None


def _configured_data_feeds() -> list[DataFeed]:
    raw = _env("ALPACA_DATA_FEEDS", "IEX")
    feeds: list[DataFeed] = []
    seen: set[DataFeed] = set()
    for token in raw.split(","):
        feed = _parse_data_feed(token)
        if feed is None or feed in seen:
            continue
        feeds.append(feed)
        seen.add(feed)
    if not feeds:
        feeds.append(DataFeed.IEX)
    return feeds


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
    # Keep a small default lag to avoid partially formed current-minute bars,
    # but make it configurable and retry without lag if the first query is empty.
    delay_min = max(int(os.getenv("ALPACA_BARS_DELAY_MINUTES", "1")), 0)

    def _make_request(end_dt: datetime, feed: DataFeed) -> StockBarsRequest:
        start_dt = end_dt - timedelta(minutes=lookback_minutes)
        return StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            feed=feed,
        )

    def _normalize_bars_df(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        df.columns = [str(col).lower() for col in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"missing bar columns: {missing}")
        return df[required].copy()

    try:
        delayed_end = datetime.now(timezone.utc) - timedelta(minutes=delay_min)
        last_error: Exception | None = None
        for feed in _configured_data_feeds():
            try:
                bars = client.get_stock_bars(_make_request(delayed_end, feed=feed))
                df = bars.df
                if df is None or len(df) == 0:
                    # Retry with no artificial delay. This helps around session open.
                    bars = client.get_stock_bars(
                        _make_request(datetime.now(timezone.utc), feed=feed)
                    )
                    df = bars.df
                if df is None or len(df) == 0:
                    raise RuntimeError("alpaca_bars_empty")
                return _normalize_bars_df(df)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("alpaca_bars_empty")
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

    try:
        last_error: Exception | None = None
        for feed in _configured_data_feeds():
            try:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed)
                quotes = client.get_stock_latest_quote(req)
                quote = quotes[symbol]
                bid = float(quote.bid_price)
                ask = float(quote.ask_price)
                if bid <= 0 or ask <= 0 or ask < bid:
                    raise RuntimeError("alpaca_quote_invalid")
                return bid, ask
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("alpaca_quote_unavailable")
    except Exception:
        if not allow_mock:
            raise
        return 200.00, 200.01


def fetch_macro_proxy_returns(allow_mock: bool = True) -> dict[str, float]:
    proxies = ["SPY", "VXX", "UUP", "TLT"]
    out: dict[str, float] = {}
    for symbol in proxies:
        try:
            bars = fetch_recent_bars(symbol=symbol, lookback_minutes=30, allow_mock=allow_mock)
        except Exception:
            out[symbol] = 0.0
            continue
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
