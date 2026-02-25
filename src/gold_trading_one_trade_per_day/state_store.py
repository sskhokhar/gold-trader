from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
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
