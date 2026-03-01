from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    import oandapyV20
    import oandapyV20.endpoints.accounts as oanda_accounts
    import oandapyV20.endpoints.instruments as oanda_instruments
    import oandapyV20.endpoints.pricing as oanda_pricing
    _OANDA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OANDA_AVAILABLE = False


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def has_real_credentials() -> bool:
    token = _env("OANDA_API_TOKEN")
    account_id = _env("OANDA_ACCOUNT_ID")
    return bool(token and account_id and token != "your_oanda_api_token_here")


def is_paper_mode() -> bool:
    return _env("OANDA_ENVIRONMENT", "practice").lower() != "live"


def _base_url() -> str:
    if is_paper_mode():
        return "https://api-fxpractice.oanda.com"
    return "https://api-fxtrade.oanda.com"


def get_oanda_client() -> "oandapyV20.API | None":
    if not _OANDA_AVAILABLE or not has_real_credentials():
        return None
    environment = "practice" if is_paper_mode() else "live"
    return oandapyV20.API(access_token=_env("OANDA_API_TOKEN"), environment=environment)


# Keep backward-compatible alias used in main.py / warmup.py
def get_trading_client() -> "oandapyV20.API | None":
    return get_oanda_client()


def mock_bars(symbol: str = "XAU_USD", periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(tz=timezone.utc), periods=periods, freq="1min")
    base = 2900.0
    prices = [base]
    for i in range(1, periods):
        drift = 0.10 if i % 15 else 0.80
        prices.append(prices[-1] + drift)
    close = pd.Series(prices, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.50
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.50
    volume = pd.Series([10000 + (25000 if i % 20 == 0 else 0) for i in range(periods)], index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def fetch_recent_bars(
    symbol: str = "XAU_USD",
    lookback_minutes: int = 180,
    allow_mock: bool = True,
    **_kwargs: Any,
) -> pd.DataFrame:
    client = get_oanda_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("oanda_client_unavailable")
        return mock_bars(symbol=symbol)

    try:
        params = {
            "count": min(lookback_minutes, 500),
            "granularity": "M1",
            "price": "M",
        }
        r = oanda_instruments.InstrumentsCandles(symbol, params=params)
        client.request(r)
        candles = r.response.get("candles", [])
        if not candles:
            raise RuntimeError("oanda_bars_empty")

        rows = []
        for c in candles:
            mid = c.get("mid", {})
            rows.append(
                {
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": float(c.get("volume", 0)),
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp")
        if df.empty:
            raise RuntimeError("oanda_bars_empty")
        return df[["open", "high", "low", "close", "volume"]].copy()
    except Exception:
        if not allow_mock:
            raise
        return mock_bars(symbol=symbol)


def fetch_latest_quote(symbol: str = "XAU_USD", allow_mock: bool = True) -> tuple[float, float]:
    client = get_oanda_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("oanda_client_unavailable")
        return 2900.00, 2900.50

    account_id = _env("OANDA_ACCOUNT_ID")
    try:
        params = {"instruments": symbol}
        r = oanda_pricing.PricingInfo(account_id, params=params)
        client.request(r)
        prices = r.response.get("prices", [])
        if not prices:
            raise RuntimeError("oanda_quote_unavailable")
        price = prices[0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        if bid <= 0 or ask <= 0 or ask < bid:
            raise RuntimeError("oanda_quote_invalid")
        return bid, ask
    except Exception:
        if not allow_mock:
            raise
        return 2900.00, 2900.50


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
    client = get_oanda_client()
    if client is None:
        if not allow_mock:
            raise RuntimeError("oanda_trading_client_unavailable")
        return 100_000.0

    account_id = _env("OANDA_ACCOUNT_ID")
    try:
        r = oanda_accounts.AccountSummary(account_id)
        client.request(r)
        account = r.response.get("account", {})
        nav = account.get("NAV") or account.get("balance") or "0"
        return float(nav)
    except Exception:
        if not allow_mock:
            raise
        return 100_000.0


class AlpacaDataToolInput(BaseModel):
    symbol: str = Field(default="XAU_USD", description="Instrument symbol for analysis.")


class AlpacaDataTool(BaseTool):
    name: str = "OANDA Market Snapshot Tool"
    description: str = (
        "Returns a compact market snapshot with latest price, quote spread, volume spike, "
        "and macro proxies for event-triggered XAU_USD analysis."
    )
    args_schema = AlpacaDataToolInput

    def _run(self, symbol: str = "XAU_USD") -> str:
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

