from __future__ import annotations

import os
import socket
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class StreamHealth:
    connected: bool
    thread_alive: bool
    has_bar: bool
    has_quote: bool
    last_msg_at: datetime | None
    data_age_sec: float
    stale: bool


class MarketStreamSensor:
    def __init__(
        self,
        symbol: str = "GLD",
        stale_seconds: int | None = None,
        max_bars: int = 500,
    ) -> None:
        self.symbol = symbol
        self.stale_seconds = stale_seconds or int(os.getenv("STREAM_STALE_SECONDS", "10"))
        self.max_bars = max_bars

        self._api_key = _env("ALPACA_API_KEY")
        self._secret_key = _env("ALPACA_SECRET_KEY")
        self._enabled = bool(self._api_key and self._secret_key and self._api_key != "your_alpaca_api_key_here")

        self._stream: StockDataStream | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bars: deque[dict[str, Any]] = deque(maxlen=max_bars)
        self._quote: dict[str, Any] | None = None
        self._last_msg_at: datetime | None = None
        self._connected = False
        self._start_attempted = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> bool:
        if not self._enabled:
            return False
        if self._thread and self._thread.is_alive():
            return True
        if self._start_attempted and (self._thread is None or not self._thread.is_alive()):
            return False

        # Avoid noisy websocket retry loops when DNS/network is unavailable.
        try:
            socket.getaddrinfo("stream.data.alpaca.markets", 443)
        except Exception:
            self._start_attempted = True
            return False

        self._stream = StockDataStream(
            self._api_key,
            self._secret_key,
            feed=DataFeed.IEX,
        )
        self._start_attempted = True

        async def on_bar(bar):
            with self._lock:
                self._bars.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    }
                )
                self._last_msg_at = _utc_now()
                self._connected = True

        async def on_quote(quote):
            with self._lock:
                self._quote = {
                    "timestamp": quote.timestamp,
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                }
                self._last_msg_at = _utc_now()
                self._connected = True

        self._stream.subscribe_bars(on_bar, self.symbol)
        self._stream.subscribe_quotes(on_quote, self.symbol)

        def _runner() -> None:
            try:
                assert self._stream is not None
                self._stream.run()
            except Exception:
                with self._lock:
                    self._connected = False

        self._thread = threading.Thread(target=_runner, daemon=True, name="market-stream")
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def health(self) -> StreamHealth:
        with self._lock:
            last_msg = self._last_msg_at
            has_bar = len(self._bars) > 0
            has_quote = self._quote is not None
            connected = self._connected

        now = _utc_now()
        age = (now - last_msg).total_seconds() if last_msg else 10_000.0
        stale = age > float(self.stale_seconds)
        thread_alive = self._thread.is_alive() if self._thread else False

        return StreamHealth(
            connected=connected,
            thread_alive=thread_alive,
            has_bar=has_bar,
            has_quote=has_quote,
            last_msg_at=last_msg,
            data_age_sec=age,
            stale=stale,
        )

    def latest_inputs(self) -> tuple[pd.DataFrame, float, float, StreamHealth] | None:
        health = self.health()
        with self._lock:
            quote = dict(self._quote) if self._quote else None
            bars = list(self._bars)

        if quote is None or not bars:
            return None

        bars_df = pd.DataFrame(bars)
        bars_df = bars_df.set_index("timestamp").sort_index()
        return bars_df, float(quote["bid_price"]), float(quote["ask_price"]), health
