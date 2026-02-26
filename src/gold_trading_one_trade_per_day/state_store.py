from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gold_trading_one_trade_per_day.schemas import (
    DailyState,
    IntentState,
    MarketSentimentReport,
    RiskDecision,
    StrategyIntent,
    TERMINAL_INTENT_STATES,
    TransitionEvent,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StateStore:
    def __init__(self, db_path: str = "state.db") -> None:
        self.db_path = Path(db_path)
        self._initialize()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trading_day_state (
                    day TEXT PRIMARY KEY,
                    day_start_equity REAL NOT NULL,
                    equity_hwm REAL NOT NULL,
                    current_equity REAL NOT NULL,
                    equity_change_pct REAL NOT NULL,
                    soft_lock INTEGER NOT NULL,
                    hard_lock INTEGER NOT NULL,
                    last_lock_reason TEXT NOT NULL,
                    max_entries_per_day INTEGER NOT NULL,
                    entries_taken INTEGER NOT NULL,
                    consecutive_losses INTEGER NOT NULL,
                    last_trade_closed_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_reports (
                    report_id TEXT PRIMARY KEY,
                    event_id TEXT,
                    symbol TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE TABLE IF NOT EXISTS strategy_intents (
                    intent_id TEXT PRIMARY KEY,
                    event_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    intent_json TEXT NOT NULL,
                    reason TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES events(event_id)
                );

                CREATE TABLE IF NOT EXISTS intent_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES strategy_intents(intent_id)
                );

                CREATE TABLE IF NOT EXISTS risk_decisions (
                    decision_id TEXT PRIMARY KEY,
                    intent_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    reason_code TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES strategy_intents(intent_id)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    intent_id TEXT,
                    client_order_id TEXT UNIQUE,
                    broker_order_id TEXT,
                    order_role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    filled_at TEXT,
                    cancelled_at TEXT,
                    order_json TEXT NOT NULL,
                    FOREIGN KEY(intent_id) REFERENCES strategy_intents(intent_id)
                );

                CREATE TABLE IF NOT EXISTS order_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id TEXT,
                    status TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS daily_metrics (
                    day TEXT PRIMARY KEY,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS llm_quota_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day TEXT NOT NULL,
                    minute_bucket TEXT NOT NULL,
                    reserved INTEGER NOT NULL,
                    used INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    event_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_llm_quota_day
                ON llm_quota_ledger(day);

                CREATE INDEX IF NOT EXISTS idx_llm_quota_minute
                ON llm_quota_ledger(minute_bucket);

                CREATE INDEX IF NOT EXISTS idx_llm_quota_event
                ON llm_quota_ledger(event_id);

                CREATE TABLE IF NOT EXISTS analysis_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS latency_metrics (
                    event_id TEXT PRIMARY KEY,
                    intent_id TEXT,
                    day TEXT NOT NULL,
                    event_detected_at TEXT,
                    llm_start_at TEXT,
                    llm_end_at TEXT,
                    risk_approved_at TEXT,
                    entry_submitted_at TEXT,
                    entry_filled_at TEXT,
                    analysis_latency_ms REAL,
                    signal_to_submit_ms REAL,
                    signal_to_fill_ms REAL,
                    submit_to_fill_ms REAL,
                    degraded_mode INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_latency_day
                ON latency_metrics(day);

                CREATE INDEX IF NOT EXISTS idx_latency_fill
                ON latency_metrics(entry_filled_at);

                CREATE TABLE IF NOT EXISTS watchdog_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_at TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_watchdog_event_at
                ON watchdog_events(event_at);

                CREATE TABLE IF NOT EXISTS system_flags (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def get_or_create_day_state(
        self, day: str, starting_equity: float, max_entries_per_day: int = 8
    ) -> DailyState:
        existing = self.get_day_state(day)
        if existing:
            return existing

        state = DailyState(
            day=day,
            day_start_equity=starting_equity,
            equity_hwm=starting_equity,
            current_equity=starting_equity,
            equity_change_pct=0.0,
            max_entries_per_day=max_entries_per_day,
        )
        self.upsert_day_state(state)
        return state

    def get_day_state(self, day: str) -> DailyState | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trading_day_state WHERE day = ?", (day,)
            ).fetchone()

        if not row:
            return None

        closed = row["last_trade_closed_at"]
        return DailyState(
            day=row["day"],
            day_start_equity=row["day_start_equity"],
            equity_hwm=row["equity_hwm"],
            current_equity=row["current_equity"],
            equity_change_pct=row["equity_change_pct"],
            soft_lock=bool(row["soft_lock"]),
            hard_lock=bool(row["hard_lock"]),
            max_entries_per_day=row["max_entries_per_day"],
            entries_taken=row["entries_taken"],
            consecutive_losses=row["consecutive_losses"],
            last_trade_closed_at=datetime.fromisoformat(closed) if closed else None,
            last_lock_reason=row["last_lock_reason"] or "",
        )

    def upsert_day_state(self, state: DailyState) -> None:
        now = utc_now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trading_day_state (
                    day, day_start_equity, equity_hwm, current_equity, equity_change_pct,
                    soft_lock, hard_lock, last_lock_reason, max_entries_per_day,
                    entries_taken, consecutive_losses, last_trade_closed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    day_start_equity=excluded.day_start_equity,
                    equity_hwm=excluded.equity_hwm,
                    current_equity=excluded.current_equity,
                    equity_change_pct=excluded.equity_change_pct,
                    soft_lock=excluded.soft_lock,
                    hard_lock=excluded.hard_lock,
                    last_lock_reason=excluded.last_lock_reason,
                    max_entries_per_day=excluded.max_entries_per_day,
                    entries_taken=excluded.entries_taken,
                    consecutive_losses=excluded.consecutive_losses,
                    last_trade_closed_at=excluded.last_trade_closed_at,
                    updated_at=excluded.updated_at
                """,
                (
                    state.day,
                    state.day_start_equity,
                    state.equity_hwm,
                    state.current_equity,
                    state.equity_change_pct,
                    int(state.soft_lock),
                    int(state.hard_lock),
                    state.last_lock_reason,
                    state.max_entries_per_day,
                    state.entries_taken,
                    state.consecutive_losses,
                    state.last_trade_closed_at.isoformat()
                    if state.last_trade_closed_at
                    else None,
                    now,
                ),
            )

    def record_event(
        self,
        event_id: str,
        symbol: str,
        snapshot: dict[str, Any],
        status: str = IntentState.EVENT_DETECTED.value,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO events (event_id, symbol, detected_at, snapshot_json, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    symbol,
                    utc_now().isoformat(),
                    json.dumps(snapshot),
                    status,
                ),
            )

    def update_event(
        self,
        event_id: str,
        status: str,
        snapshot_patch: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT snapshot_json FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if not row:
                return
            payload = json.loads(row["snapshot_json"])
            if snapshot_patch:
                payload.update(snapshot_patch)
            conn.execute(
                """
                UPDATE events
                SET status = ?, snapshot_json = ?
                WHERE event_id = ?
                """,
                (status, json.dumps(payload), event_id),
            )

    def record_analysis(
        self,
        report: MarketSentimentReport,
        event_id: str | None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_reports (report_id, event_id, symbol, created_at, report_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    event_id,
                    report.symbol,
                    report.generated_at.isoformat(),
                    report.model_dump_json(),
                ),
            )

    def record_intent(
        self,
        intent: StrategyIntent,
        event_id: str | None,
        state: IntentState = IntentState.INTENT_GENERATED,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_intents
                (intent_id, event_id, created_at, expires_at, state, intent_json, reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.intent_id,
                    event_id,
                    intent.generated_at.isoformat(),
                    intent.expires_at.isoformat(),
                    state.value,
                    intent.model_dump_json(),
                    "",
                    utc_now().isoformat(),
                ),
            )

    def get_intent(self, intent_id: str) -> tuple[StrategyIntent, IntentState] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT intent_json, state FROM strategy_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()

        if not row:
            return None

        intent = StrategyIntent.model_validate_json(row["intent_json"])
        return intent, IntentState(row["state"])

    def list_non_terminal_intents(self) -> list[dict[str, Any]]:
        terminal_values = tuple(state.value for state in TERMINAL_INTENT_STATES)
        placeholders = ",".join(["?"] * len(terminal_values))
        query = f"""
            SELECT intent_id, state, created_at, expires_at, intent_json
            FROM strategy_intents
            WHERE state NOT IN ({placeholders})
            ORDER BY created_at ASC
        """
        with self._conn() as conn:
            rows = conn.execute(query, terminal_values).fetchall()

        return [dict(row) for row in rows]

    def transition(
        self,
        transition: TransitionEvent,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE strategy_intents
                SET state = ?, updated_at = ?
                WHERE intent_id = ? AND state = ?
                """,
                (
                    transition.to_state.value,
                    transition.at.isoformat(),
                    transition.intent_id,
                    transition.from_state.value,
                ),
            )
            conn.execute(
                """
                INSERT INTO intent_transitions
                (intent_id, from_state, to_state, at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    transition.intent_id,
                    transition.from_state.value,
                    transition.to_state.value,
                    transition.at.isoformat(),
                    json.dumps(transition.metadata),
                ),
            )

    def mark_intent_terminal(
        self,
        intent_id: str,
        terminal_state: IntentState,
        reason: str,
    ) -> None:
        if terminal_state not in TERMINAL_INTENT_STATES:
            raise ValueError("terminal_state must be terminal")

        with self._conn() as conn:
            conn.execute(
                """
                UPDATE strategy_intents
                SET state = ?, reason = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (terminal_state.value, reason, utc_now().isoformat(), intent_id),
            )

    def record_risk_decision(self, decision: RiskDecision) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO risk_decisions
                (decision_id, intent_id, created_at, approved, reason_code, decision_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.intent_id,
                    decision.generated_at.isoformat(),
                    int(decision.approved),
                    decision.reason_code,
                    decision.model_dump_json(),
                ),
            )

    def record_order(
        self,
        order_id: str,
        intent_id: str,
        client_order_id: str,
        order_role: str,
        status: str,
        payload: dict[str, Any],
        broker_order_id: str | None = None,
        submitted_at: datetime | None = None,
    ) -> None:
        submitted = (submitted_at or utc_now()).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders
                (order_id, intent_id, client_order_id, broker_order_id, order_role, status,
                submitted_at, filled_at, cancelled_at, order_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    intent_id,
                    client_order_id,
                    broker_order_id,
                    order_role,
                    status,
                    submitted,
                    payload.get("filled_at"),
                    payload.get("cancelled_at"),
                    json.dumps(payload),
                ),
            )

    def update_order_status(
        self,
        broker_order_id: str,
        status: str,
        payload: dict[str, Any],
        filled_at: datetime | None = None,
        cancelled_at: datetime | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = ?, filled_at = COALESCE(?, filled_at),
                    cancelled_at = COALESCE(?, cancelled_at),
                    order_json = ?, broker_order_id = COALESCE(broker_order_id, ?)
                WHERE broker_order_id = ?
                """,
                (
                    status,
                    filled_at.isoformat() if filled_at else None,
                    cancelled_at.isoformat() if cancelled_at else None,
                    json.dumps(payload),
                    broker_order_id,
                    broker_order_id,
                ),
            )

            conn.execute(
                """
                INSERT INTO order_updates (broker_order_id, status, event_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    broker_order_id,
                    status,
                    utc_now().isoformat(),
                    json.dumps(payload),
                ),
            )

    def record_position_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        avg_entry_price: float,
        opened_at: datetime,
        closed_at: datetime,
        pnl: float,
        pnl_pct: float,
        metadata: dict[str, Any],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO positions
                (symbol, side, qty, avg_entry_price, opened_at, closed_at, pnl, pnl_pct, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    qty,
                    avg_entry_price,
                    opened_at.isoformat(),
                    closed_at.isoformat(),
                    pnl,
                    pnl_pct,
                    json.dumps(metadata),
                ),
            )

    def save_daily_metrics(self, day: str, metrics: dict[str, Any]) -> None:
        now = utc_now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_metrics (day, metrics_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    metrics_json = excluded.metrics_json,
                    updated_at = excluded.updated_at
                """,
                (day, json.dumps(metrics), now, now),
            )

    def get_daily_metrics(self, day: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT metrics_json FROM daily_metrics WHERE day = ?", (day,)
            ).fetchone()
        if not row:
            return None
        return json.loads(row["metrics_json"])

    def get_order_updates_for_day(self, day: str) -> list[dict[str, Any]]:
        prefix = f"{day}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT broker_order_id, status, event_at, payload_json
                FROM order_updates
                WHERE event_at LIKE ?
                ORDER BY event_at ASC
                """,
                (prefix,),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.pop("payload_json"))
            result.append(data)
        return result

    def get_closed_positions_for_day(self, day: str) -> list[dict[str, Any]]:
        prefix = f"{day}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT symbol, side, qty, avg_entry_price, opened_at, closed_at, pnl, pnl_pct, metadata_json
                FROM positions
                WHERE closed_at LIKE ?
                ORDER BY closed_at ASC
                """,
                (prefix,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            out.append(data)
        return out

    def get_events_for_day(self, day: str) -> list[dict[str, Any]]:
        prefix = f"{day}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT event_id, symbol, detected_at, snapshot_json, status
                FROM events
                WHERE detected_at LIKE ?
                ORDER BY detected_at ASC
                """,
                (prefix,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["snapshot"] = json.loads(data.pop("snapshot_json"))
            out.append(data)
        return out

    def record_quota_ledger(
        self,
        event_id: str,
        source: str,
        reserved: int,
        used: int,
        at: datetime | None = None,
    ) -> None:
        ts = at or utc_now()
        day = ts.date().isoformat()
        minute_bucket = ts.strftime("%Y-%m-%dT%H:%M")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO llm_quota_ledger
                (day, minute_bucket, reserved, used, source, event_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    day,
                    minute_bucket,
                    max(int(reserved), 0),
                    max(int(used), 0),
                    source,
                    event_id,
                    ts.isoformat(),
                ),
            )

    def get_quota_usage_snapshot(
        self,
        now: datetime,
        rpm_cap: int,
        rpd_cap: int,
        reservation_ttl_sec: int,
    ) -> dict[str, int]:
        day = now.date().isoformat()
        minute_bucket = now.strftime("%Y-%m-%dT%H:%M")
        active_cutoff = (now - timedelta(seconds=reservation_ttl_sec)).isoformat()

        with self._conn() as conn:
            rpm_used = conn.execute(
                """
                SELECT COALESCE(SUM(used), 0) AS value
                FROM llm_quota_ledger
                WHERE source LIKE 'commit:%' AND minute_bucket = ?
                """,
                (minute_bucket,),
            ).fetchone()["value"]

            rpd_used = conn.execute(
                """
                SELECT COALESCE(SUM(used), 0) AS value
                FROM llm_quota_ledger
                WHERE source LIKE 'commit:%' AND day = ?
                """,
                (day,),
            ).fetchone()["value"]

            rpm_reserved = conn.execute(
                """
                SELECT COALESCE(SUM(r.reserved), 0) AS value
                FROM llm_quota_ledger r
                WHERE r.source LIKE 'reserve:%'
                  AND r.minute_bucket = ?
                  AND r.created_at >= ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM llm_quota_ledger c
                      WHERE c.event_id = r.event_id
                        AND c.source LIKE 'commit:%'
                  )
                """,
                (minute_bucket, active_cutoff),
            ).fetchone()["value"]

            rpd_reserved = conn.execute(
                """
                SELECT COALESCE(SUM(r.reserved), 0) AS value
                FROM llm_quota_ledger r
                WHERE r.source LIKE 'reserve:%'
                  AND r.day = ?
                  AND r.created_at >= ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM llm_quota_ledger c
                      WHERE c.event_id = r.event_id
                        AND c.source LIKE 'commit:%'
                  )
                """,
                (day, active_cutoff),
            ).fetchone()["value"]

        rpm_total = int(rpm_used) + int(rpm_reserved)
        rpd_total = int(rpd_used) + int(rpd_reserved)
        return {
            "rpm_used": rpm_total,
            "rpd_used": rpd_total,
            "rpm_remaining": max(int(rpm_cap) - rpm_total, 0),
            "rpd_remaining": max(int(rpd_cap) - rpd_total, 0),
        }

    def get_quota_ledger_for_day(self, day: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT day, minute_bucket, reserved, used, source, event_id, created_at
                FROM llm_quota_ledger
                WHERE day = ?
                ORDER BY created_at ASC
                """,
                (day,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_cached_analysis(
        self,
        cache_key: str,
        model_name: str,
        prompt_version: str,
        report: MarketSentimentReport,
        ttl_sec: int,
        now: datetime,
    ) -> None:
        created_at = now.isoformat()
        expires_at = (now + timedelta(seconds=max(ttl_sec, 1))).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO analysis_cache
                (cache_key, model, prompt_version, report_json, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    prompt_version = excluded.prompt_version,
                    report_json = excluded.report_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    model_name,
                    prompt_version,
                    report.model_dump_json(),
                    created_at,
                    expires_at,
                ),
            )

    def get_cached_analysis(
        self,
        cache_key: str,
        now: datetime,
    ) -> MarketSentimentReport | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT report_json
                FROM analysis_cache
                WHERE cache_key = ? AND expires_at > ?
                """,
                (cache_key, now.isoformat()),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE analysis_cache
                SET hit_count = hit_count + 1
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
        return MarketSentimentReport.model_validate_json(row["report_json"])

    @staticmethod
    def _as_iso(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _parse_iso(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _duration_ms(start: datetime | None, end: datetime | None) -> float | None:
        if start is None or end is None:
            return None
        return max((end - start).total_seconds() * 1000.0, 0.0)

    def upsert_latency_metric(
        self,
        event_id: str,
        intent_id: str | None = None,
        event_detected_at: datetime | str | None = None,
        llm_start_at: datetime | str | None = None,
        llm_end_at: datetime | str | None = None,
        risk_approved_at: datetime | str | None = None,
        entry_submitted_at: datetime | str | None = None,
        entry_filled_at: datetime | str | None = None,
        degraded_mode: bool | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM latency_metrics
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()

            payload = dict(row) if row else {}
            metadata = json.loads(payload.get("metadata_json", "{}")) if payload else {}
            if metadata_patch:
                metadata.update(metadata_patch)

            current_intent = payload.get("intent_id")
            current_degraded = bool(payload.get("degraded_mode", 0))

            iso_detected = self._as_iso(event_detected_at) or payload.get("event_detected_at")
            iso_llm_start = self._as_iso(llm_start_at) or payload.get("llm_start_at")
            iso_llm_end = self._as_iso(llm_end_at) or payload.get("llm_end_at")
            iso_risk = self._as_iso(risk_approved_at) or payload.get("risk_approved_at")
            iso_submit = self._as_iso(entry_submitted_at) or payload.get("entry_submitted_at")
            iso_fill = self._as_iso(entry_filled_at) or payload.get("entry_filled_at")

            dt_detected = self._parse_iso(iso_detected)
            dt_llm_start = self._parse_iso(iso_llm_start)
            dt_llm_end = self._parse_iso(iso_llm_end)
            dt_submit = self._parse_iso(iso_submit)
            dt_fill = self._parse_iso(iso_fill)

            day = (
                payload.get("day")
                or (dt_detected.date().isoformat() if dt_detected else utc_now().date().isoformat())
            )
            final_degraded = int(current_degraded if degraded_mode is None else bool(degraded_mode))

            analysis_latency_ms = self._duration_ms(dt_llm_start, dt_llm_end)
            signal_to_submit_ms = self._duration_ms(dt_detected, dt_submit)
            signal_to_fill_ms = self._duration_ms(dt_detected, dt_fill)
            submit_to_fill_ms = self._duration_ms(dt_submit, dt_fill)

            conn.execute(
                """
                INSERT INTO latency_metrics (
                    event_id, intent_id, day, event_detected_at, llm_start_at, llm_end_at,
                    risk_approved_at, entry_submitted_at, entry_filled_at, analysis_latency_ms,
                    signal_to_submit_ms, signal_to_fill_ms, submit_to_fill_ms, degraded_mode,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    intent_id = excluded.intent_id,
                    day = excluded.day,
                    event_detected_at = excluded.event_detected_at,
                    llm_start_at = excluded.llm_start_at,
                    llm_end_at = excluded.llm_end_at,
                    risk_approved_at = excluded.risk_approved_at,
                    entry_submitted_at = excluded.entry_submitted_at,
                    entry_filled_at = excluded.entry_filled_at,
                    analysis_latency_ms = excluded.analysis_latency_ms,
                    signal_to_submit_ms = excluded.signal_to_submit_ms,
                    signal_to_fill_ms = excluded.signal_to_fill_ms,
                    submit_to_fill_ms = excluded.submit_to_fill_ms,
                    degraded_mode = excluded.degraded_mode,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    event_id,
                    intent_id or current_intent,
                    day,
                    iso_detected,
                    iso_llm_start,
                    iso_llm_end,
                    iso_risk,
                    iso_submit,
                    iso_fill,
                    analysis_latency_ms,
                    signal_to_submit_ms,
                    signal_to_fill_ms,
                    submit_to_fill_ms,
                    final_degraded,
                    json.dumps(metadata),
                    utc_now().isoformat(),
                ),
            )

    def get_recent_signal_to_fill_ms(self, limit: int = 50) -> list[float]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT signal_to_fill_ms
                FROM latency_metrics
                WHERE signal_to_fill_ms IS NOT NULL
                ORDER BY entry_filled_at DESC
                LIMIT ?
                """,
                (max(int(limit), 1),),
            ).fetchall()
        return [float(row["signal_to_fill_ms"]) for row in rows]

    def get_latency_metrics_for_day(self, day: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM latency_metrics
                WHERE day = ?
                ORDER BY updated_at ASC
                """,
                (day,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["degraded_mode"] = bool(data.get("degraded_mode", 0))
            data["metadata"] = json.loads(data.pop("metadata_json", "{}"))
            out.append(data)
        return out

    def record_watchdog_event(
        self,
        reason_code: str,
        action: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        at: datetime | None = None,
    ) -> None:
        ts = (at or utc_now()).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO watchdog_events (event_at, reason_code, action_json, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ts,
                    reason_code,
                    json.dumps(action),
                    json.dumps(metadata or {}),
                ),
            )

    def get_watchdog_events_for_day(self, day: str) -> list[dict[str, Any]]:
        prefix = f"{day}%"
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, event_at, reason_code, action_json, metadata_json
                FROM watchdog_events
                WHERE event_at LIKE ?
                ORDER BY event_at ASC
                """,
                (prefix,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["action"] = json.loads(data.pop("action_json"))
            data["metadata"] = json.loads(data.pop("metadata_json"))
            out.append(data)
        return out

    def set_system_flag(self, key: str, value: Any) -> None:
        now = utc_now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO system_flags (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), now),
            )

    def get_system_flag(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT value_json
                FROM system_flags
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    def clear_system_flag(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM system_flags WHERE key = ?",
                (key,),
            )

    def list_open_entry_orders(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    o.intent_id,
                    o.client_order_id,
                    o.broker_order_id,
                    o.status,
                    o.submitted_at,
                    si.intent_json
                FROM orders o
                LEFT JOIN strategy_intents si
                    ON si.intent_id = o.intent_id
                WHERE o.order_role = 'entry'
                  AND replace(lower(o.status), 'orderstatus.', '') NOT IN (
                    'filled', 'canceled', 'cancelled', 'rejected', 'expired', 'done_for_day'
                  )
                ORDER BY o.submitted_at ASC
                """
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            raw_intent = data.pop("intent_json", None)
            data["intent"] = (
                StrategyIntent.model_validate_json(raw_intent) if raw_intent else None
            )
            out.append(data)
        return out

    def get_denied_intents_since(
        self,
        since: datetime,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT intent_id, intent_json, state, reason, updated_at
                FROM strategy_intents
                WHERE state = ? AND updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (IntentState.DENIED.value, since.isoformat(), max(int(limit), 1)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["intent"] = StrategyIntent.model_validate_json(data.pop("intent_json"))
            out.append(data)
        return out

    def db_health_snapshot(self) -> dict[str, Any]:
        probe_key = "__db_health_probe__"
        probe_value = {"ok": True, "at": utc_now().isoformat()}
        with self._conn() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            conn.execute(
                """
                INSERT INTO system_flags (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (probe_key, json.dumps(probe_value), utc_now().isoformat()),
            )
            row = conn.execute(
                "SELECT value_json FROM system_flags WHERE key = ?",
                (probe_key,),
            ).fetchone()
            conn.execute("DELETE FROM system_flags WHERE key = ?", (probe_key,))
        return {
            "journal_mode": str(journal_mode).lower(),
            "read_write_ok": bool(row),
        }
