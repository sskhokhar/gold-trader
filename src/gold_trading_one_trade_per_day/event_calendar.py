from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class MacroWindow:
    label: str
    start_utc: datetime
    end_utc: datetime


class EventCalendar:
    def __init__(
        self,
        config_path: str | None = None,
        block_minutes: int | None = None,
    ) -> None:
        self.config_path = Path(
            config_path
            or os.getenv("EVENT_WINDOWS_FILE")
            or Path(__file__).resolve().parent / "config" / "event_windows.yaml"
        )
        self.block_minutes = block_minutes or int(os.getenv("MACRO_EVENT_BLOCK_MINUTES", "15"))
        self.windows = self._load_windows()

    def _load_windows(self) -> list[MacroWindow]:
        if yaml is None or not self.config_path.exists():
            return []

        try:
            data = yaml.safe_load(self.config_path.read_text()) or {}
        except Exception:
            return []

        events = data.get("events", []) if isinstance(data, dict) else []
        windows: list[MacroWindow] = []
        for item in events:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp_utc")
            label = item.get("label", "macro_event")
            if not ts:
                continue
            try:
                event_ts = datetime.fromisoformat(str(ts))
            except ValueError:
                continue
            if event_ts.tzinfo is None:
                event_ts = event_ts.replace(tzinfo=timezone.utc)

            delta = timedelta(minutes=self.block_minutes)
            windows.append(
                MacroWindow(
                    label=str(label),
                    start_utc=event_ts - delta,
                    end_utc=event_ts + delta,
                )
            )

        return sorted(windows, key=lambda w: w.start_utc)

    def get_active_window(self, at: datetime | None = None) -> MacroWindow | None:
        now = at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        for window in self.windows:
            if window.start_utc <= now <= window.end_utc:
                return window
        return None

    def is_blocked(self, at: datetime | None = None) -> tuple[bool, str | None]:
        active = self.get_active_window(at=at)
        if active is None:
            return False, None
        return True, active.label
