from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gold_trading_one_trade_per_day.analysis_cache import AnalysisCache
from gold_trading_one_trade_per_day.event_trigger import build_feature_snapshot
from gold_trading_one_trade_per_day.llm_factory import is_local_model_available, load_llm_runtime_config
from gold_trading_one_trade_per_day.quota_guard import QuotaGuard
from gold_trading_one_trade_per_day.schemas import MarketSentimentReport, WarmupReport
from gold_trading_one_trade_per_day.state_store import StateStore
from gold_trading_one_trade_per_day.tools.alpaca_tools import (
    fetch_latest_quote,
    fetch_macro_proxy_returns,
    fetch_recent_bars,
    has_real_credentials,
)
from gold_trading_one_trade_per_day.market_stream import MarketStreamSensor

NY_TZ = ZoneInfo("America/New_York")


def run_warmup(
    mode: str = "paper",
    state_store: StateStore | None = None,
    now: datetime | None = None,
) -> WarmupReport:
    ts = now or datetime.now(tz=NY_TZ)
    store = state_store or StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    checks: dict[str, dict] = {}
    reasons: list[str] = []

    llm_cfg = load_llm_runtime_config()
    analyst_model = llm_cfg.analyst_model
    model_ok = True
    model_detail = {"model": analyst_model, "route": "local" if analyst_model.startswith("ollama/") else "remote"}
    if analyst_model.startswith("ollama/"):
        model_ok = is_local_model_available(analyst_model)
        model_detail["local_available"] = model_ok
    checks["model"] = {"passed": model_ok, **model_detail}
    if not model_ok:
        reasons.append("model_unreachable")

    allow_mock = os.getenv("WARMUP_ALLOW_MOCK", "false").lower() == "true"
    data_ok = False
    bars = None
    bid = ask = None
    try:
        bars = fetch_recent_bars(symbol="GLD", lookback_minutes=30, allow_mock=allow_mock)
        bid, ask = fetch_latest_quote(symbol="GLD", allow_mock=allow_mock)
        data_ok = bars is not None and len(bars) >= 5 and bid is not None and ask is not None
    except Exception as exc:
        checks["alpaca_rest"] = {
            "passed": False,
            "error": str(exc),
            "has_credentials": has_real_credentials(),
        }
        reasons.append("alpaca_rest_unavailable")
    if "alpaca_rest" not in checks:
        checks["alpaca_rest"] = {
            "passed": data_ok,
            "bars_count": int(len(bars)) if bars is not None else 0,
            "spread": float(max((ask or 0) - (bid or 0), 0.0)),
            "has_credentials": has_real_credentials(),
        }
        if not data_ok:
            reasons.append("alpaca_rest_unavailable")

    sensor = MarketStreamSensor(symbol="GLD")
    stream_started = sensor.start()
    health = sensor.health()
    checks["alpaca_stream"] = {
        "passed": bool(stream_started or health.thread_alive),
        "enabled": sensor.enabled,
        "started": stream_started,
        "connected": health.connected,
        "thread_alive": health.thread_alive,
    }
    if not checks["alpaca_stream"]["passed"]:
        reasons.append("alpaca_stream_unavailable")
    sensor.stop()

    quota_guard = QuotaGuard(store)
    quota_snapshot = quota_guard.snapshot(now=ts)
    min_ratio = max(float(os.getenv("WARMUP_MIN_QUOTA_RATIO", "0.1")), 0.0)
    rpm_ratio = quota_snapshot.rpm_remaining / max(quota_guard.rpm_cap, 1)
    rpd_ratio = quota_snapshot.rpd_remaining / max(quota_guard.rpd_cap, 1)
    quota_ok = rpm_ratio >= min_ratio and rpd_ratio >= min_ratio
    checks["quota"] = {
        "passed": quota_ok,
        "rpm_remaining": quota_snapshot.rpm_remaining,
        "rpd_remaining": quota_snapshot.rpd_remaining,
        "rpm_cap": quota_guard.rpm_cap,
        "rpd_cap": quota_guard.rpd_cap,
        "min_ratio": min_ratio,
    }
    if not quota_ok:
        reasons.append("quota_low")

    db_status = store.db_health_snapshot()
    db_ok = bool(db_status.get("read_write_ok")) and db_status.get("journal_mode") == "wal"
    checks["db"] = {
        "passed": db_ok,
        **db_status,
    }
    if not db_ok:
        reasons.append("db_health_failed")

    cache_primed = False
    if data_ok and bars is not None and bid is not None and ask is not None:
        try:
            macro = fetch_macro_proxy_returns(allow_mock=allow_mock)
            snapshot = build_feature_snapshot(
                bars=bars,
                bid=float(bid),
                ask=float(ask),
                macro_proxies=macro,
                timestamp=ts,
                symbol="GLD",
            )
            report = MarketSentimentReport(
                symbol="GLD",
                generated_at=ts,
                regime=snapshot.regime,
                greed_score=snapshot.greed_score,
                sentiment_score=0.0,
                rationale=["warmup cache seed"],
            )
            cache = AnalysisCache(store)
            key = cache.make_key(snapshot=snapshot, model_name=llm_cfg.analyst_model)
            cache.put(cache_key=key, model_name=llm_cfg.analyst_model, report=report, now=ts)
            cache_primed = True
        except Exception:
            cache_primed = False

    checks["analysis_cache_prime"] = {
        "passed": cache_primed,
        "ttl_sec": int(os.getenv("ANALYSIS_CACHE_TTL_SEC", "60")),
    }

    passed = len(reasons) == 0
    report = WarmupReport(
        generated_at=ts,
        mode=mode,
        passed=passed,
        checks=checks,
        reason_codes=reasons,
    )
    store.set_system_flag("last_warmup_report", report.model_dump(mode="json"))
    return report


def warmup_is_recent_and_passed(
    state_store: StateStore,
    now: datetime,
) -> tuple[bool, dict]:
    raw = state_store.get_system_flag("last_warmup_report", None)
    if not isinstance(raw, dict):
        return False, {"reason": "missing_warmup_report"}
    try:
        report = WarmupReport.model_validate(raw)
    except Exception:
        return False, {"reason": "invalid_warmup_report"}
    max_age_minutes = max(int(os.getenv("WARMUP_MAX_AGE_MINUTES", "180")), 1)
    age = now - report.generated_at
    if age > timedelta(minutes=max_age_minutes):
        return False, {
            "reason": "warmup_report_stale",
            "age_minutes": round(age.total_seconds() / 60.0, 2),
            "max_age_minutes": max_age_minutes,
        }
    if not report.passed:
        return False, {
            "reason": "warmup_report_failed",
            "reason_codes": report.reason_codes,
        }
    return True, {"generated_at": report.generated_at.isoformat()}
