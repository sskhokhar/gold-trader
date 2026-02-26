from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, TypeVar

from gold_trading_one_trade_per_day.state_store import StateStore

T = TypeVar("T")


@dataclass(frozen=True)
class QuotaSnapshot:
    rpm_used: int
    rpd_used: int
    rpm_remaining: int
    rpd_remaining: int


@dataclass(frozen=True)
class QuotaReservationResult:
    allowed: bool
    snapshot: QuotaSnapshot


@dataclass(frozen=True)
class BackoffResult:
    value: object | None
    retry_count: int
    total_wait_sec: float
    exhausted: bool
    last_error: str | None


class QuotaGuard:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.rpm_cap = max(int(_env_or_default("QUOTA_RPM_CAP", "8")), 1)
        self.rpd_cap = max(int(_env_or_default("QUOTA_RPD_CAP", "600")), 1)
        self.reservation_ttl_sec = max(
            int(_env_or_default("QUOTA_RESERVATION_TTL_SEC", "120")),
            10,
        )

    def reserve(
        self,
        event_id: str,
        estimated_requests: int,
        source: str,
        now: datetime,
    ) -> QuotaReservationResult:
        est = max(int(estimated_requests), 0)
        snapshot = self._snapshot(now=now)
        allowed = snapshot.rpm_remaining >= est and snapshot.rpd_remaining >= est
        if allowed:
            self.state_store.record_quota_ledger(
                event_id=event_id,
                source=f"reserve:{source}",
                reserved=est,
                used=0,
                at=now,
            )
            snapshot = self._snapshot(now=now)
        return QuotaReservationResult(allowed=allowed, snapshot=snapshot)

    def commit(
        self,
        event_id: str,
        used_requests: int,
        source: str,
        now: datetime,
    ) -> QuotaSnapshot:
        used = max(int(used_requests), 0)
        self.state_store.record_quota_ledger(
            event_id=event_id,
            source=f"commit:{source}",
            reserved=0,
            used=used,
            at=now,
        )
        return self._snapshot(now=now)

    def snapshot(self, now: datetime) -> QuotaSnapshot:
        return self._snapshot(now=now)

    def _snapshot(self, now: datetime) -> QuotaSnapshot:
        usage = self.state_store.get_quota_usage_snapshot(
            now=now,
            rpm_cap=self.rpm_cap,
            rpd_cap=self.rpd_cap,
            reservation_ttl_sec=self.reservation_ttl_sec,
        )
        return QuotaSnapshot(
            rpm_used=usage["rpm_used"],
            rpd_used=usage["rpd_used"],
            rpm_remaining=usage["rpm_remaining"],
            rpd_remaining=usage["rpd_remaining"],
        )


def _env_or_default(name: str, default: str) -> str:
    return os.getenv(name, default)


def is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "ratelimit" in name
        or "resource_exhausted" in text
        or "rate limit" in text
        or "429" in text
        or "too many requests" in text
    )


def call_with_rate_limit_backoff(
    func: Callable[[], T],
    deadline: datetime,
    base_wait_sec: float,
    max_wait_sec: float,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[float], None] = time.sleep,
    jitter_ratio: float = 0.2,
) -> BackoffResult:
    retry_count = 0
    total_wait = 0.0
    wait = max(base_wait_sec, 0.1)
    max_wait = max(max_wait_sec, wait)
    last_error: str | None = None

    while True:
        try:
            return BackoffResult(
                value=func(),
                retry_count=retry_count,
                total_wait_sec=round(total_wait, 4),
                exhausted=False,
                last_error=last_error,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

            retry_count += 1
            last_error = str(exc)
            now = now_fn()
            if now >= deadline:
                return BackoffResult(
                    value=None,
                    retry_count=retry_count,
                    total_wait_sec=round(total_wait, 4),
                    exhausted=True,
                    last_error=last_error,
                )

            jitter = wait * jitter_ratio
            sleep_sec = min(max_wait, max(wait + random.uniform(-jitter, jitter), 0.05))
            if now + timedelta(seconds=sleep_sec) >= deadline:
                return BackoffResult(
                    value=None,
                    retry_count=retry_count,
                    total_wait_sec=round(total_wait, 4),
                    exhausted=True,
                    last_error=last_error,
                )
            sleep_fn(sleep_sec)
            total_wait += sleep_sec
            wait = min(wait * 2.0, max_wait)
