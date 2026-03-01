# Project Memory (Short)

Last updated: 2026-03-01 (Phase F)

## Goal
- Event-driven XAU/USD scalping system.
- Deterministic safety kernel for risk + execution.
- **Math engine computes trade levels (entry/SL/TP). CrewAI/Gemini validates only.**

## Current Pipeline (Phase F)
```
Market Data → Trigger → analyze_gold() → TradeSetup (entry/SL/TP)
                                ↓
                    LLM A: Sentiment + sees math setup
                                ↓
                    LLM B: Validate/Approve/Veto math trade
                                ↓
                         Risk → Execute
```
- No valid math setup → skip LLM entirely (save quota), record `skipped:no_math_setup`
- Hard guardrail: code overwrites LLM-returned levels with math values when sides match

## Current Stack
- Data/sensing: Alpaca WebSocket first, REST fallback.
- Math engine: `gold_strategy.py` → `analyze_gold()` → `TradeSetup`.
- Agents:
  - Analyst (local Ollama by default) — also sees `trade_setup_json`.
  - Strategy Validator (Gemini by default) — approves or vetoes math trade.
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
- Symbol: `XAU_USD`
- Risk per trade: `0.25%`
- Soft lock: `+0.75%`
- Hard profit lock: `+1.5%`
- Hard loss lock: `-1.5%`
- Max entries/day: `8`
- Cooldown: `180s`
- Max spread: `$0.50`
- Intent TTL: `45s`
- Min R:R ratio: `1.5` (math engine gate; tunable via `MIN_RISK_REWARD_RATIO`)
- Min session quality: `0.3` (math engine gate; tunable via `MIN_SESSION_QUALITY`)

## Most Important Env Vars
- Alpaca: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER`
- LLM routing: `AGENT_MODEL_ANALYST`, `AGENT_MODEL_STRATEGY`, `OLLAMA_BASE_URL`
- LLM runtime: `AGENT_LLM_TIMEOUT_SEC`, `AGENT_LLM_MAX_RETRIES`, `CREW_MAX_RPM`, `AGENT_MAX_RPM`
- Quota: `QUOTA_RPM_CAP`, `QUOTA_RPD_CAP`
- Cache: `ANALYSIS_CACHE_ENABLED`, `ANALYSIS_CACHE_TTL_SEC`
- Trigger/data: `OPEN_WARMUP_MINUTES`, `STREAM_STALE_SECONDS`, `REST_FALLBACK_INTERVAL_SECONDS`
- Math engine: `MIN_SESSION_QUALITY` (default: 0.3), `MIN_RISK_REWARD_RATIO` (default: 1.5)

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
- `no_math_setup`: math engine found no high-probability setup; LLM call skipped to save quota.

## Runbook (Morning)
1. Confirm Ollama model exists (`ollama list`).
2. Run tests.
3. Start with `run_shadow_loop` (not `crewai test`).
4. After shadow is stable, run `run_paper` during RTH (after 9:35 ET).
5. Use `reconcile` to inspect daily metrics.
6. Check `trade_setup_type`, `trade_setup_confluence`, `llm_decision` in telemetry.

## Current Health Snapshot
- Tests passing: `25` in gold_strategy alone; full suite ~155.
- Math engine wired as source of truth for entry/SL/TP.
- Hard guardrail prevents LLM from hallucinating price levels.
- CLI supports all previous commands unchanged.

## Canonical Full Memory
- See `PROJECT_MEMORY.md` for complete technical history and details.
