from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from gold_trading_one_trade_per_day.state_store import StateStore


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


class Historian:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    def generate_daily_metrics(self, day: str) -> dict[str, Any]:
        positions = self.state_store.get_closed_positions_for_day(day)
        updates = self.state_store.get_order_updates_for_day(day)
        events = self.state_store.get_events_for_day(day)
        quota_rows = self.state_store.get_quota_ledger_for_day(day)
        latency_rows = self.state_store.get_latency_metrics_for_day(day)
        watchdog_rows = self.state_store.get_watchdog_events_for_day(day)

        pnl_values = [p["pnl"] for p in positions if p["pnl"] is not None]
        pnl_pct_values = [p["pnl_pct"] for p in positions if p["pnl_pct"] is not None]

        wins = sum(1 for p in positions if (p.get("pnl") or 0) > 0)
        losses = sum(1 for p in positions if (p.get("pnl") or 0) < 0)

        reason_counter: Counter[str] = Counter()
        spread_samples: list[float] = []
        slippage_samples: list[float] = []
        expectancy_by_regime: dict[str, list[float]] = defaultdict(list)
        expectancy_ex_macro: list[float] = []

        for p in positions:
            md = p.get("metadata") or {}
            reason = md.get("close_reason")
            if reason:
                reason_counter[reason] += 1
            if "spread" in md:
                spread_samples.append(float(md["spread"]))
            if "slippage_bps" in md:
                slippage_samples.append(float(md["slippage_bps"]))
            regime = md.get("regime", "unknown")
            if p.get("pnl") is not None:
                pnl = float(p["pnl"])
                expectancy_by_regime[regime].append(pnl)
                if not md.get("macro_event_window", False):
                    expectancy_ex_macro.append(pnl)

        status_counter = Counter(update["status"] for update in updates)

        skip_reason_counts: Counter[str] = Counter()
        fallback_count = 0
        stream_count = 0
        stale_count = 0
        retry_histogram: Counter[str] = Counter()
        cache_hits = 0
        quota_denied = 0
        retry_exhausted = 0

        for evt in events:
            snap = evt.get("snapshot") or {}
            skip_reason = snap.get("skip_reason_code")
            if skip_reason:
                skip_reason_counts[skip_reason] += 1
            if skip_reason == "quota_budget_exceeded":
                quota_denied += 1
            if skip_reason == "rate_limit_retry_exhausted":
                retry_exhausted += 1
            ds = snap.get("data_source")
            if ds == "rest_fallback":
                fallback_count += 1
            elif ds == "stream":
                stream_count += 1
            if snap.get("data_fresh") is False:
                stale_count += 1
            retries = int(snap.get("retry_count", 0))
            retry_histogram[str(retries)] += 1
            if bool(snap.get("analysis_cache_hit")):
                cache_hits += 1

        total_source_samples = stream_count + fallback_count
        stream_uptime_pct = (
            (stream_count / total_source_samples) if total_source_samples else 0.0
        )
        total_events = len(events)
        cache_hit_rate = (cache_hits / total_events) if total_events else 0.0
        gemini_requests_used = sum(
            int(row.get("used", 0))
            for row in quota_rows
            if str(row.get("source", "")).startswith("commit:")
        )
        estimated_gemini_requests_saved = cache_hits

        analysis_latency_vals = [
            float(row["analysis_latency_ms"])
            for row in latency_rows
            if row.get("analysis_latency_ms") is not None
        ]
        signal_to_submit_vals = [
            float(row["signal_to_submit_ms"])
            for row in latency_rows
            if row.get("signal_to_submit_ms") is not None
        ]
        signal_to_fill_vals = [
            float(row["signal_to_fill_ms"])
            for row in latency_rows
            if row.get("signal_to_fill_ms") is not None
        ]
        submit_to_fill_vals = [
            float(row["submit_to_fill_ms"])
            for row in latency_rows
            if row.get("submit_to_fill_ms") is not None
        ]
        degraded_mode_incidents = sum(
            1 for row in latency_rows if bool(row.get("degraded_mode", False))
        )
        watchdog_halts = sum(
            1 for row in watchdog_rows if row.get("reason_code") != "watchdog_resumed"
        )

        metrics = {
            "day": day,
            "trades_closed": len(positions),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(positions)) if positions else 0.0,
            "net_pnl": sum(pnl_values) if pnl_values else 0.0,
            "avg_pnl": mean(pnl_values) if pnl_values else 0.0,
            "avg_pnl_pct": mean(pnl_pct_values) if pnl_pct_values else 0.0,
            "avg_spread": mean(spread_samples) if spread_samples else 0.0,
            "avg_slippage_bps": mean(slippage_samples) if slippage_samples else 0.0,
            "order_status_counts": dict(status_counter),
            "close_reason_counts": dict(reason_counter),
            "skip_reason_counts": dict(skip_reason_counts),
            "stream_uptime_pct": stream_uptime_pct,
            "fallback_count": fallback_count,
            "stale_event_count": stale_count,
            "quota_denied_count": quota_denied,
            "rate_limit_retry_exhausted_count": retry_exhausted,
            "retry_count_distribution": dict(retry_histogram),
            "analysis_cache_hit_rate": cache_hit_rate,
            "estimated_gemini_requests_saved": estimated_gemini_requests_saved,
            "gemini_requests_used": gemini_requests_used,
            "latency_samples": len(latency_rows),
            "analysis_latency_ms_p50": _percentile(analysis_latency_vals, 50),
            "analysis_latency_ms_p95": _percentile(analysis_latency_vals, 95),
            "signal_to_submit_ms_p50": _percentile(signal_to_submit_vals, 50),
            "signal_to_submit_ms_p95": _percentile(signal_to_submit_vals, 95),
            "signal_to_fill_ms_p50": _percentile(signal_to_fill_vals, 50),
            "signal_to_fill_ms_p95": _percentile(signal_to_fill_vals, 95),
            "submit_to_fill_ms_p50": _percentile(submit_to_fill_vals, 50),
            "submit_to_fill_ms_p95": _percentile(submit_to_fill_vals, 95),
            "latency_degraded_mode_incidents": degraded_mode_incidents,
            "watchdog_halt_count": watchdog_halts,
            "expectancy_by_regime": {
                regime: mean(values) for regime, values in expectancy_by_regime.items()
            },
            "expectancy_ex_macro": mean(expectancy_ex_macro) if expectancy_ex_macro else 0.0,
        }

        metrics["recommendations"] = self._build_recommendations(metrics)
        self.state_store.save_daily_metrics(day, metrics)
        return metrics

    @staticmethod
    def _build_recommendations(metrics: dict[str, Any]) -> list[str]:
        recs: list[str] = []
        if metrics["avg_spread"] > 0.02:
            recs.append("Tighten spread gate or avoid low-liquidity intervals.")
        if metrics["avg_slippage_bps"] > 15:
            recs.append("Reduce entry aggressiveness or lower max_slippage_bps.")
        if metrics["win_rate"] < 0.45 and metrics["trades_closed"] >= 10:
            recs.append("Narrow trigger conditions; current edge may be weak.")
        if metrics["fallback_count"] > 0 and metrics["stream_uptime_pct"] < 0.8:
            recs.append("Stream stability is weak; investigate websocket/data connectivity.")
        if metrics["quota_denied_count"] > 0:
            recs.append("LLM quota denials occurred; reduce trigger frequency or increase cache TTL.")
        if metrics.get("signal_to_fill_ms_p95", 0.0) > 8000:
            recs.append("Latency p95 breached SLA; stay in degraded mode until recovery windows pass.")
        if metrics.get("watchdog_halt_count", 0) > 0:
            recs.append("Watchdog halts occurred; inspect watchdog_events and order lifecycle stability.")
        if metrics["trades_closed"] == 0:
            recs.append("Review trigger sensitivity; no executions recorded.")
        if not recs:
            recs.append("Metrics stable; continue paper validation before scaling.")
        return recs
