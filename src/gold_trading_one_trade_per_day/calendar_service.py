"""Economic Calendar Service for XAUUSD news-aware trading.

Fetches and caches high-impact economic events that affect gold prices.
Supports YAML-based static calendar (default) with optional HTTP API sources.

Environment Variables:
    CALENDAR_SOURCE: "yaml" (default) | "tradingeconomics" | "fcsapi"
    CALENDAR_CACHE_TTL_SEC: How long to cache calendar data (default: 3600)
    CALENDAR_HIGH_IMPACT_ONLY: Only return high-impact events (default: true)
    CALENDAR_YAML_FILE: Path to a YAML calendar file (optional override)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

try:
    import urllib.request as _urllib_request
    import json as _json
except Exception:  # pragma: no cover
    _urllib_request = None  # type: ignore[assignment]
    _json = None  # type: ignore[assignment]

# Events that strongly affect gold (XAUUSD) — USD-denominated macro data
GOLD_AFFECTING_EVENTS = {
    "non-farm payrolls",
    "nfp",
    "cpi",
    "consumer price index",
    "fomc",
    "federal open market committee",
    "fed rate decision",
    "interest rate decision",
    "fed chair",
    "powell",
    "ppi",
    "producer price index",
    "initial jobless claims",
    "jobless claims",
    "ism manufacturing",
    "ism services",
    "ism non-manufacturing",
    "retail sales",
    "gdp",
    "gross domestic product",
    "core cpi",
    "core pce",
    "pce",
    "personal consumption expenditures",
    "unemployment rate",
    "average hourly earnings",
    "durable goods",
    "trade balance",
    "treasury",
}


class EconomicEvent(BaseModel):
    event_id: str
    name: str
    release_time: datetime
    impact: str  # "high" / "medium" / "low"
    currency: str = "USD"
    consensus: str | None = None
    previous: str | None = None
    actual: str | None = None
    affects_gold: bool = True

    @classmethod
    def from_yaml_entry(cls, entry: dict[str, Any], idx: int) -> "EconomicEvent":
        name = str(entry.get("name", ""))
        ts_str = entry.get("release_time") or entry.get("timestamp_utc") or ""
        try:
            release_time = datetime.fromisoformat(str(ts_str))
        except (ValueError, TypeError):
            release_time = datetime.now(timezone.utc)
        if release_time.tzinfo is None:
            release_time = release_time.replace(tzinfo=timezone.utc)

        impact = str(entry.get("impact", "high")).lower()
        currency = str(entry.get("currency", "USD"))
        affects = _name_affects_gold(name) or currency == "USD"

        return cls(
            event_id=str(entry.get("event_id", f"yaml-{idx}")),
            name=name,
            release_time=release_time,
            impact=impact,
            currency=currency,
            consensus=entry.get("consensus"),
            previous=entry.get("previous"),
            actual=entry.get("actual"),
            affects_gold=affects,
        )


class DailyCalendar(BaseModel):
    date: str
    events: list[EconomicEvent] = Field(default_factory=list)
    has_high_impact: bool = False
    next_event: EconomicEvent | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _name_affects_gold(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in GOLD_AFFECTING_EVENTS)


def _load_yaml_calendar(path: Path, target_date: str | None = None) -> list[EconomicEvent]:
    if _yaml is None or not path.exists():
        return []
    try:
        data = _yaml.safe_load(path.read_text()) or {}
    except Exception:
        return []

    entries = data.get("events", []) if isinstance(data, dict) else []
    events: list[EconomicEvent] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        try:
            ev = EconomicEvent.from_yaml_entry(item, idx)
        except Exception:
            continue
        if target_date and ev.release_time.date().isoformat() != target_date:
            continue
        events.append(ev)
    return events


def _find_next_event(events: list[EconomicEvent], now: datetime) -> EconomicEvent | None:
    future = [e for e in events if e.release_time > now]
    if not future:
        return None
    return min(future, key=lambda e: e.release_time)


class CalendarService:
    """Fetches and caches the daily economic calendar.

    Primary source: YAML file (reliable, configurable).
    Falls back to an empty calendar on any error so trading is never blocked.
    """

    def __init__(
        self,
        source: str | None = None,
        cache_ttl_sec: int | None = None,
        high_impact_only: bool | None = None,
        yaml_path: str | None = None,
    ) -> None:
        self.source = source or os.getenv("CALENDAR_SOURCE", "yaml")
        self.cache_ttl_sec = cache_ttl_sec if cache_ttl_sec is not None else int(
            os.getenv("CALENDAR_CACHE_TTL_SEC", "3600")
        )
        raw_hio = os.getenv("CALENDAR_HIGH_IMPACT_ONLY", "true")
        self.high_impact_only = (
            high_impact_only
            if high_impact_only is not None
            else raw_hio.strip().lower() != "false"
        )

        default_yaml = Path(__file__).resolve().parent / "config" / "event_windows.yaml"
        self.yaml_path = Path(
            yaml_path or os.getenv("CALENDAR_YAML_FILE") or str(default_yaml)
        )

        self._cache: DailyCalendar | None = None
        self._cache_date: str | None = None
        self._cache_fetched_at: datetime | None = None

    def get_calendar(self, date: str | None = None, now: datetime | None = None) -> DailyCalendar:
        """Return the daily calendar, using cached value if still fresh."""
        now = now or datetime.now(timezone.utc)
        target_date = date or now.date().isoformat()

        if self._is_cache_valid(target_date, now):
            assert self._cache is not None
            return self._cache

        calendar = self._fetch_calendar(target_date, now)
        self._cache = calendar
        self._cache_date = target_date
        self._cache_fetched_at = now
        return calendar

    def _is_cache_valid(self, target_date: str, now: datetime) -> bool:
        if self._cache is None or self._cache_date != target_date:
            return False
        if self._cache_fetched_at is None:
            return False
        age = (now - self._cache_fetched_at).total_seconds()
        return age < self.cache_ttl_sec

    def _fetch_calendar(self, target_date: str, now: datetime) -> DailyCalendar:
        try:
            if self.source == "yaml":
                events = _load_yaml_calendar(self.yaml_path, target_date=None)
            elif self.source == "tradingeconomics":
                events = self._fetch_tradingeconomics(target_date)
            elif self.source == "fcsapi":
                events = self._fetch_fcsapi(target_date)
            else:
                events = _load_yaml_calendar(self.yaml_path, target_date=None)
        except Exception:
            events = []

        if self.high_impact_only:
            events = [e for e in events if e.impact == "high"]

        events_today = [e for e in events if e.release_time.date().isoformat() == target_date]

        has_high = any(e.impact == "high" for e in events_today)
        next_ev = _find_next_event(events_today, now)

        return DailyCalendar(
            date=target_date,
            events=events_today,
            has_high_impact=has_high,
            next_event=next_ev,
            fetched_at=now,
        )

    def _fetch_tradingeconomics(self, target_date: str) -> list[EconomicEvent]:
        """Attempt to fetch from Trading Economics free API."""
        api_key = os.getenv("TRADINGECONOMICS_API_KEY", "")
        if not api_key:
            return []
        try:
            url = f"https://api.tradingeconomics.com/calendar/country/united states/{target_date}/{target_date}?c={api_key}&f=json"
            with _urllib_request.urlopen(url, timeout=10) as resp:  # type: ignore[union-attr]
                data = _json.loads(resp.read().decode())  # type: ignore[union-attr]
            if not isinstance(data, list):
                return []
            events: list[EconomicEvent] = []
            for idx, item in enumerate(data):
                name = str(item.get("Event", ""))
                impact_raw = str(item.get("Importance", "1"))
                impact_map = {"1": "low", "2": "medium", "3": "high"}
                impact = impact_map.get(impact_raw, "low")
                ts_str = item.get("Date", "")
                try:
                    release_time = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if release_time.tzinfo is None:
                    release_time = release_time.replace(tzinfo=timezone.utc)
                ev = EconomicEvent(
                    event_id=f"te-{idx}",
                    name=name,
                    release_time=release_time,
                    impact=impact,
                    currency=str(item.get("Currency", "USD")),
                    consensus=item.get("Forecast"),
                    previous=item.get("Previous"),
                    actual=item.get("Actual"),
                    affects_gold=_name_affects_gold(name),
                )
                events.append(ev)
            return events
        except Exception:
            return []

    def _fetch_fcsapi(self, target_date: str) -> list[EconomicEvent]:
        """Attempt to fetch from FCS API."""
        api_key = os.getenv("FCSAPI_KEY", "")
        if not api_key:
            return []
        try:
            url = f"https://fcsapi.com/api-v3/forex/economic_calendar?country=US&date={target_date}&access_key={api_key}"
            with _urllib_request.urlopen(url, timeout=10) as resp:  # type: ignore[union-attr]
                data = _json.loads(resp.read().decode())  # type: ignore[union-attr]
            items = data.get("response", []) if isinstance(data, dict) else []
            events: list[EconomicEvent] = []
            for idx, item in enumerate(items):
                name = str(item.get("event", ""))
                impact_raw = str(item.get("impact", "low")).lower()
                ts_str = item.get("date_released", "")
                try:
                    release_time = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if release_time.tzinfo is None:
                    release_time = release_time.replace(tzinfo=timezone.utc)
                ev = EconomicEvent(
                    event_id=f"fcs-{idx}",
                    name=name,
                    release_time=release_time,
                    impact=impact_raw,
                    currency="USD",
                    consensus=item.get("forecast"),
                    previous=item.get("previous"),
                    actual=item.get("actual"),
                    affects_gold=_name_affects_gold(name),
                )
                events.append(ev)
            return events
        except Exception:
            return []
