"""Event-Aware Scheduler for dual-mode XAUUSD trading.

A daemon-style runner that automatically manages trading cycles throughout the day,
switching between spike mode (news events) and daily scalp mode.

Environment Variables:
    AUTO_CYCLE_INTERVAL_SEC: Seconds between cycles in daily scalp mode (default: 45)
    AUTO_SPIKE_CYCLE_INTERVAL_SEC: Seconds between cycles in spike mode (default: 10)
    AUTO_IDLE_INTERVAL_SEC: Seconds between checks when no trading conditions (default: 120)
    SPIKE_MODE_ENABLED: Enable spike mode (default: true)
    DAILY_SCALP_ENABLED: Enable daily scalp mode (default: true)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.calendar_service import CalendarService
from gold_trading_one_trade_per_day.event_trigger import is_rth
from gold_trading_one_trade_per_day.main import _run_cycle, determine_trading_mode

NY_TZ = ZoneInfo("America/New_York")


def _env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_ny() -> datetime:
    return datetime.now(tz=NY_TZ)


def run_auto_scheduler(mode: str = "shadow") -> None:
    """Main scheduler loop: fetches calendar, manages cycle timing, and runs trades.

    Behavior:
    1. On startup: fetch today's economic calendar and log the day's plan.
    2. Loop continuously:
       a. Check if market is open (XAU_USD 24/5 hours).
       b. If in spike event window: tight polling (spike mode).
       c. Otherwise: normal polling (daily scalp mode).
       d. Stop when daily profit/loss target is hit (hard lock).
       e. Stop when daily entry limit is reached.
    3. Log daily summary on shutdown.
    """
    cycle_interval = int(os.getenv("AUTO_CYCLE_INTERVAL_SEC", "45"))
    spike_interval = int(os.getenv("AUTO_SPIKE_CYCLE_INTERVAL_SEC", "10"))
    idle_interval = int(os.getenv("AUTO_IDLE_INTERVAL_SEC", "120"))

    calendar_service = CalendarService()
    calendar_refresh_interval = int(os.getenv("CALENDAR_CACHE_TTL_SEC", "3600"))
    last_calendar_fetch = None

    print(f"[SCHEDULER] Starting auto scheduler in {mode} mode")
    daily_calendar = None
    try:
        now_utc = _now_utc()
        daily_calendar = calendar_service.get_calendar(now=now_utc)
        last_calendar_fetch = now_utc
        print(
            f"[SCHEDULER] Calendar fetched: date={daily_calendar.date} "
            f"events={len(daily_calendar.events)} "
            f"has_high_impact={daily_calendar.has_high_impact}"
        )
        if daily_calendar.events:
            for ev in daily_calendar.events:
                print(
                    f"[SCHEDULER]   Event: {ev.name} @ {ev.release_time.isoformat()} "
                    f"impact={ev.impact}"
                )
    except Exception as exc:
        print(f"[SCHEDULER] Calendar fetch failed (continuing without): {exc}")

    cycle_count = 0
    try:
        while True:
            now = _now_ny()
            now_utc = _now_utc()

            # Refresh calendar if needed
            if (
                last_calendar_fetch is None
                or (now_utc - last_calendar_fetch).total_seconds() >= calendar_refresh_interval
            ):
                try:
                    daily_calendar = calendar_service.get_calendar(now=now_utc)
                    last_calendar_fetch = now_utc
                except Exception as exc:
                    print(f"[SCHEDULER] Calendar refresh failed: {exc}")

            # Check market hours
            if not is_rth(now):
                print(f"[SCHEDULER] Market closed at {now.isoformat()}, sleeping {idle_interval}s")
                time.sleep(idle_interval)
                continue

            # Determine trading mode
            trading_mode = "daily_scalp"
            if daily_calendar is not None:
                trading_mode = determine_trading_mode(daily_calendar, now)

            # Skip if mode is disabled
            if trading_mode == "spike" and not _env_true("SPIKE_MODE_ENABLED", "true"):
                trading_mode = "daily_scalp"
            if trading_mode == "daily_scalp" and not _env_true("DAILY_SCALP_ENABLED", "true"):
                print(f"[SCHEDULER] Both modes disabled, sleeping {idle_interval}s")
                time.sleep(idle_interval)
                continue

            sleep_sec = spike_interval if trading_mode == "spike" else cycle_interval
            print(
                f"[SCHEDULER] Cycle {cycle_count + 1} | mode={trading_mode} | "
                f"next_sleep={sleep_sec}s"
            )

            try:
                result = _run_cycle(mode=mode)
                cycle_count += 1

                status = result.get("status", "unknown")
                print(f"[SCHEDULER] Cycle {cycle_count} result: status={status} mode={trading_mode}")

                # Check for hard lock (daily target reached or loss limit)
                if status == "halted":
                    reason = result.get("reason", "")
                    print(f"[SCHEDULER] Hard lock triggered: {reason}. Stopping scheduler.")
                    break

            except Exception as exc:
                print(f"[SCHEDULER] Cycle error: {exc}")

            time.sleep(sleep_sec)

    except KeyboardInterrupt:
        print("[SCHEDULER] Interrupted by user")
    finally:
        print(f"[SCHEDULER] Session complete. Total cycles: {cycle_count}")
