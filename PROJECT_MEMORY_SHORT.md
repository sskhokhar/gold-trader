# Project Memory (Short)

Last updated: 2026-02-26

## Goal
- Event-driven GLD scalping system.
- Deterministic safety kernel for risk + execution.
- CrewAI/Gemini used only when trigger conditions pass.

## Current Stack
- Data/sensing: Alpaca WebSocket first, REST fallback.
- Agents:
  - Analyst (local Ollama by default).
  - Strategy Composer (Gemini by default).
- Deterministic services:
  - `risk_engine.py`
  - `execution_service.py`
  - `state_store.py` (SQLite WAL)
- Quota protection:
  - `quota_guard.py` (RPM/RPD caps + reservations)
  - 429 retry/backoff until TTL expiry.
- Reuse optimization:
  - `analysis_cache.py` (cache Analyst output only, TTL).

## Locked Trading Defaults
- Symbol: `GLD`
- Risk per trade: `0.25%`
- Soft lock: `+0.75%`
- Hard profit lock: `+1.5%`
- Hard loss lock: `-1.5%`
- Max entries/day: `8`
- Cooldown: `180s`
- Max spread: `$0.02`
- Intent TTL: `45s`
- Open warmup block: `9:30-9:35 ET`

## Most Important Env Vars
- Alpaca: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER`
- LLM routing: `AGENT_MODEL_ANALYST`, `AGENT_MODEL_STRATEGY`, `OLLAMA_BASE_URL`
- LLM runtime: `AGENT_LLM_TIMEOUT_SEC`, `AGENT_LLM_MAX_RETRIES`, `CREW_MAX_RPM`, `AGENT_MAX_RPM`
- Quota: `QUOTA_RPM_CAP`, `QUOTA_RPD_CAP`
- Cache: `ANALYSIS_CACHE_ENABLED`, `ANALYSIS_CACHE_TTL_SEC`
- Trigger/data: `OPEN_WARMUP_MINUTES`, `STREAM_STALE_SECONDS`, `REST_FALLBACK_INTERVAL_SECONDS`

## Typical Commands
```bash
ollama list
./.venv/bin/python -m unittest discover -s tests -v
SHADOW_LOOP_SLEEP_SEC=16 ./.venv/bin/python -m gold_trading_one_trade_per_day.main run_shadow_loop 10
./.venv/bin/python -m gold_trading_one_trade_per_day.main run_paper
```

## What Status Messages Mean
- `outside_rth`: normal if not market hours.
- `open_warmup`: normal between 9:30-9:35 ET.
- `fallback_throttled`: loop is faster than REST fallback interval.
- `stale_data`: stream stale and REST failed; fail-closed (expected safety behavior).
- `quota_budget_exceeded`: pre-LLM quota gate denied.
- `rate_limit_retry_exhausted`: repeated 429 until TTL expired.

## Runbook (Morning)
1. Confirm Ollama model exists (`ollama list`).
2. Run tests.
3. Start with `run_shadow_loop` (not `crewai test`).
4. After shadow is stable, run `run_paper` during RTH (after 9:35 ET).
5. Use `reconcile` to inspect daily metrics.

## Current Health Snapshot
- Tests passing: `18/18`.
- Hybrid routing + quota guard + analysis cache are implemented.
- CLI supports `run_shadow_loop <iterations>`.

## Canonical Full Memory
- See `PROJECT_MEMORY.md` for complete technical history and details.
