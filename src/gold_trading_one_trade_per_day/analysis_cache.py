from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any

from gold_trading_one_trade_per_day.schemas import FeatureSnapshot, MarketSentimentReport
from gold_trading_one_trade_per_day.state_store import StateStore


class AnalysisCache:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.enabled = os.getenv("ANALYSIS_CACHE_ENABLED", "true").lower() == "true"
        self.ttl_sec = max(int(os.getenv("ANALYSIS_CACHE_TTL_SEC", "60")), 1)
        self.prompt_version = os.getenv("ANALYSIS_PROMPT_VERSION", "v1")

    def make_key(self, snapshot: FeatureSnapshot, model_name: str) -> str:
        payload = {
            "snapshot": _normalize_snapshot(snapshot.model_dump(mode="json")),
            "model": model_name,
            "prompt_version": self.prompt_version,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        snapshot: FeatureSnapshot,
        model_name: str,
        now: datetime,
    ) -> tuple[str, MarketSentimentReport | None]:
        if not self.enabled:
            return "", None
        key = self.make_key(snapshot, model_name)
        report = self.state_store.get_cached_analysis(cache_key=key, now=now)
        return key, report

    def put(
        self,
        cache_key: str,
        model_name: str,
        report: MarketSentimentReport,
        now: datetime,
    ) -> None:
        if not self.enabled or not cache_key:
            return
        self.state_store.upsert_cached_analysis(
            cache_key=cache_key,
            model_name=model_name,
            prompt_version=self.prompt_version,
            report=report,
            ttl_sec=self.ttl_sec,
            now=now,
        )


def _normalize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(snapshot)
    for field in ("snapshot_id", "timestamp", "data_age_sec"):
        normalized.pop(field, None)

    rounded: dict[str, Any] = {}
    for key, value in normalized.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        elif isinstance(value, dict):
            inner = {}
            for k, v in value.items():
                inner[k] = round(v, 6) if isinstance(v, float) else v
            rounded[key] = inner
        else:
            rounded[key] = value
    return rounded
