from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from crewai import Agent, Crew, Process, Task

from gold_trading_one_trade_per_day.llm_factory import build_llm, load_llm_runtime_config
from gold_trading_one_trade_per_day.schemas import FeatureSnapshot, MarketSentimentReport, Regime
from gold_trading_one_trade_per_day.state_store import StateStore

NY_TZ = ZoneInfo("America/New_York")


@dataclass(slots=True)
class BenchmarkFixture:
    name: str
    snapshot: FeatureSnapshot
    expected_regime: Regime


def _build_fixture_bars(base: float, jump: float, volume_base: int, volume_last: int) -> pd.DataFrame:
    idx = pd.date_range(end=datetime.now(tz=NY_TZ), periods=60, freq="1min")
    values = [base + i * 0.01 for i in range(59)] + [base + jump]
    close = pd.Series(values, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 0.2
    low = close - 0.2
    volume = pd.Series([volume_base] * 59 + [volume_last], index=idx)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def default_benchmark_fixtures() -> list[BenchmarkFixture]:
    from gold_trading_one_trade_per_day.event_trigger import build_feature_snapshot

    now = datetime.now(tz=NY_TZ).replace(hour=10, minute=15, second=0, microsecond=0)
    bars_trend = _build_fixture_bars(base=200.0, jump=1.4, volume_base=10_000, volume_last=65_000)
    bars_range = _build_fixture_bars(base=200.0, jump=0.08, volume_base=10_000, volume_last=9_500)
    bars_high_vol = _build_fixture_bars(base=200.0, jump=2.1, volume_base=10_000, volume_last=40_000)

    fixtures = [
        BenchmarkFixture(
            name="trend_breakout",
            snapshot=build_feature_snapshot(
                bars=bars_trend,
                bid=200.35,
                ask=200.36,
                timestamp=now,
                symbol="GLD",
            ),
            expected_regime=Regime.TREND,
        ),
        BenchmarkFixture(
            name="range_flat",
            snapshot=build_feature_snapshot(
                bars=bars_range,
                bid=200.01,
                ask=200.02,
                timestamp=now,
                symbol="GLD",
            ),
            expected_regime=Regime.RANGE,
        ),
        BenchmarkFixture(
            name="high_volatility",
            snapshot=build_feature_snapshot(
                bars=bars_high_vol,
                bid=201.4,
                ask=201.42,
                timestamp=now,
                symbol="GLD",
            ),
            expected_regime=Regime.HIGH_VOL,
        ),
    ]
    return fixtures


def _extract_report(task_output) -> MarketSentimentReport:
    if task_output.pydantic:
        return MarketSentimentReport.model_validate(task_output.pydantic.model_dump())
    if task_output.json_dict:
        return MarketSentimentReport.model_validate(task_output.json_dict)
    return MarketSentimentReport.model_validate_json(task_output.raw)


def _run_analyst_once(model_name: str, snapshot_json: str) -> tuple[MarketSentimentReport | None, float, str | None]:
    cfg = load_llm_runtime_config()
    llm = build_llm(model_name=model_name, cfg=cfg)
    analyst = Agent(
        role="Market Sentiment Analyst",
        goal="Produce only valid MarketSentimentReport JSON from a FeatureSnapshot.",
        backstory="You are a deterministic analyst for GLD intraday setups.",
        allow_delegation=False,
        reasoning=False,
        max_iter=2,
        llm=llm,
    )
    task = Task(
        description=(
            "Read the provided FeatureSnapshot JSON and output a MarketSentimentReport only.\n"
            "FeatureSnapshot JSON:\n{feature_snapshot_json}\n"
            "Keep rationale concise with factual signals only."
        ),
        expected_output="A valid MarketSentimentReport JSON object.",
        output_pydantic=MarketSentimentReport,
        markdown=False,
        agent=analyst,
    )
    scoped = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        cache=False,
        max_rpm=max(cfg.crew_max_rpm, 1),
    )

    start = time.perf_counter()
    try:
        output = scoped.kickoff(inputs={"feature_snapshot_json": snapshot_json})
        latency_ms = (time.perf_counter() - start) * 1000.0
        report = _extract_report(output.tasks_output[0])
        return report, latency_ms, None
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return None, latency_ms, str(exc)


def score_model_summary(summary: dict[str, Any]) -> float:
    schema_pass_rate = float(summary.get("schema_pass_rate", 0.0))
    hallucination_rate = float(summary.get("hallucination_rate", 1.0))
    consistency = float(summary.get("consistency_score", 0.0))
    latency_ms = max(float(summary.get("avg_latency_ms", 10000.0)), 1.0)
    latency_score = min(2000.0 / latency_ms, 1.0)
    return round(
        (0.45 * schema_pass_rate)
        + (0.25 * consistency)
        + (0.20 * (1.0 - hallucination_rate))
        + (0.10 * latency_score),
        6,
    )


def run_model_benchmark(
    models: list[str] | None = None,
    repeats: int | None = None,
    state_store: StateStore | None = None,
) -> dict[str, Any]:
    cfg = load_llm_runtime_config()
    model_list = models or [
        cfg.analyst_model,
        os.getenv("BENCHMARK_CANDIDATE_MODEL", "ollama/llama3.1:8b"),
    ]
    model_list = [m for m in model_list if m]
    run_repeats = max(repeats or int(os.getenv("BENCHMARK_REPEATS", "2")), 1)
    fixtures = default_benchmark_fixtures()

    results: dict[str, dict[str, Any]] = {}
    for model_name in model_list:
        total = 0
        schema_passes = 0
        invalid_claims = 0
        latencies: list[float] = []
        errors: list[str] = []
        sentiment_by_fixture: dict[str, list[float]] = {}

        for fixture in fixtures:
            sentiment_by_fixture[fixture.name] = []
            snapshot_json = fixture.snapshot.model_dump_json()
            for _ in range(run_repeats):
                total += 1
                report, latency_ms, error = _run_analyst_once(
                    model_name=model_name,
                    snapshot_json=snapshot_json,
                )
                latencies.append(latency_ms)
                if report is None:
                    if error:
                        errors.append(error)
                    continue
                schema_passes += 1
                sentiment_by_fixture[fixture.name].append(float(report.sentiment_score))
                if report.symbol != "GLD" or report.regime != fixture.expected_regime:
                    invalid_claims += 1

        consistency_scores = []
        for scores in sentiment_by_fixture.values():
            if len(scores) <= 1:
                consistency_scores.append(1.0 if scores else 0.0)
            else:
                consistency_scores.append(max(1.0 - pstdev(scores), 0.0))

        summary = {
            "model": model_name,
            "runs": total,
            "schema_pass_rate": (schema_passes / total) if total else 0.0,
            "hallucination_rate": (invalid_claims / schema_passes) if schema_passes else 1.0,
            "consistency_score": mean(consistency_scores) if consistency_scores else 0.0,
            "avg_latency_ms": mean(latencies) if latencies else 0.0,
            "error_count": len(errors),
            "errors": errors[:5],
        }
        summary["score"] = score_model_summary(summary)
        results[model_name] = summary

    ranked = sorted(results.values(), key=lambda item: item["score"], reverse=True)
    payload = {
        "generated_at": datetime.now(tz=NY_TZ).isoformat(),
        "fixtures": [fixture.name for fixture in fixtures],
        "repeats": run_repeats,
        "models": ranked,
        "recommended_model": ranked[0]["model"] if ranked else None,
    }

    store = state_store or StateStore(db_path=os.getenv("STATE_DB_PATH", "state.db"))
    store.set_system_flag("latest_model_benchmark", payload)
    return payload


def parse_models_arg(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    out = []
    for value in raw.split(","):
        m = value.strip()
        if m:
            out.append(m)
    return out if out else None


def run_benchmark_from_env() -> dict[str, Any]:
    models = parse_models_arg(os.getenv("BENCHMARK_MODELS"))
    repeats = int(os.getenv("BENCHMARK_REPEATS", "2"))
    return run_model_benchmark(models=models, repeats=repeats)
