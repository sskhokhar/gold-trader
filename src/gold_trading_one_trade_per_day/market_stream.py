from __future__ import annotations

import json
import os
import socket
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stream_base_url() -> str:
    env = _env("OANDA_ENVIRONMENT", "practice").lower()
    if env == "live":
        return "https://stream-fxtrade.oanda.com"
    return "https://stream-fxpractice.oanda.com"


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
        symbol: str = "XAU_USD",
        stale_seconds: int | None = None,
        max_bars: int = 500,
    ) -> None:
        self.symbol = symbol
        self.stale_seconds = stale_seconds or int(os.getenv("STREAM_STALE_SECONDS", "10"))
        self.max_bars = max_bars

        self._api_token = _env("OANDA_API_TOKEN")
        self._account_id = _env("OANDA_ACCOUNT_ID")
        self._enabled = bool(
            self._api_token
            and self._account_id
            and self._api_token != "your_oanda_api_token_here"
        )

        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bars: deque[dict[str, Any]] = deque(maxlen=max_bars)
        self._quote: dict[str, Any] | None = None
        self._last_msg_at: datetime | None = None
        self._connected = False
        self._last_start_attempt_at: datetime | None = None
        self._restart_cooldown_sec = max(int(os.getenv("STREAM_RESTART_COOLDOWN_SEC", "5")), 0)
        # Accumulate tick data to form 1-minute bars
        self._tick_open: float | None = None
        self._tick_high: float | None = None
        self._tick_low: float | None = None
        self._tick_minute: int | None = None
        self._tick_volume: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _stream_url(self) -> str:
        base = _stream_base_url()
        return f"{base}/v3/accounts/{self._account_id}/pricing/stream?instruments={self.symbol}"

    def _process_tick(self, mid: float) -> None:
        now = _utc_now()
        minute = now.minute
        with self._lock:
            if self._tick_minute is None or self._tick_minute != minute:
                if self._tick_open is not None and self._tick_minute is not None:
                    ts = now.replace(second=0, microsecond=0)
                    self._bars.append(
                        {
                            "timestamp": ts,
                            "open": self._tick_open,
                            "high": self._tick_high,
                            "low": self._tick_low,
                            "close": mid,
                            "volume": float(self._tick_volume),
                        }
                    )
                self._tick_open = mid
                self._tick_high = mid
                self._tick_low = mid
                self._tick_minute = minute
                self._tick_volume = 1
            else:
                if self._tick_high is None or mid > self._tick_high:
                    self._tick_high = mid
                if self._tick_low is None or mid < self._tick_low:
                    self._tick_low = mid
                self._tick_volume += 1
            self._last_msg_at = now
            self._connected = True

    def start(self) -> bool:
        if not self._enabled:
            return False
        if self._thread and self._thread.is_alive():
            return True
        now = _utc_now()
        if (
            self._last_start_attempt_at
            and (now - self._last_start_attempt_at).total_seconds() < float(self._restart_cooldown_sec)
        ):
            return False
        self._last_start_attempt_at = now

        stream_host = _stream_base_url().replace("https://", "").split("/")[0]
        try:
            socket.getaddrinfo(stream_host, 443)
        except Exception:
            return False

        def _runner() -> None:
            url = self._stream_url()
            headers = {"Authorization": f"Bearer {self._api_token}"}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            msg = json.loads(raw_line)
                        except Exception:
                            continue
                        msg_type = msg.get("type", "")
                        if msg_type == "HEARTBEAT":
                            with self._lock:
                                self._last_msg_at = _utc_now()
                                self._connected = True
                        elif msg_type == "PRICE":
                            bids = msg.get("bids", [])
                            asks = msg.get("asks", [])
                            if bids and asks:
                                bid = float(bids[0]["price"])
                                ask = float(asks[0]["price"])
                                mid = (bid + ask) / 2.0
                                with self._lock:
                                    self._quote = {
                                        "timestamp": _utc_now(),
                                        "bid_price": bid,
                                        "ask_price": ask,
                                    }
                                self._process_tick(mid)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._connected = False

        self._thread = threading.Thread(target=_runner, daemon=True, name="market-stream")
        self._thread.start()
        return True

    def stop(self) -> None:
        # Streaming thread will exit when the requests session closes or an exception occurs.
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

