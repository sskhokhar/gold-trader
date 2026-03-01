# Project Memory: XAUUSD Multi-Agent Scalping System

Last updated: 2026-03-01

## 1) Mission and Scope
- Build a deterministic, event-driven XAUUSD scalping system with **dual trading modes**.
- Use CrewAI agents only for high-value decision moments.
- Keep risk checks and execution deterministic and non-bypassable.
- Operate with rollout path: shadow -> paper -> live.
- "Invest $100, make $10, call it a day" — absolute dollar profit/loss targets supported.

## 2) Plan Evolution (What Was Decided and Implemented)

### Phase A: Safety-first architecture
- Deterministic event trigger before any LLM call.
- Typed contracts for all agent/execution data.
- SQLite WAL state machine for crash recovery.
- Deterministic risk gate and deterministic execution service.
- Daily anti-greed and hard-loss locks.

### Phase B: Reality-checked market constraints
- Lock thresholds recalibrated for XAU_USD:
  - soft lock: +0.75%
  - hard profit lock: +1.5%
  - hard loss lock: -1.5%
- Sensor changed to WebSocket-first + REST fallback.
- Open warm-up no-trade window (Sun 17:00-17:05 ET).
- Macro event no-trade windows (config-driven).
- Fail-closed behavior when stream + fallback are unavailable.

### Phase C: Gemini free-tier quota resilience
- Hybrid model routing:
  - Analyst: local Ollama (default)
  - Strategy: Gemini Flash
- Crew and agent RPM caps.
- Pre-LLM deterministic quota gate with RPM/RPD budgets.
- 429 backoff-until-expiry policy with persisted telemetry.
- Analysis-level cache for reusing `MarketSentimentReport` only.
- No strategy intent cache reuse across events.

### Phase D: Go-live execution optimization (implemented)
- Deterministic latency telemetry + SLA/hysteresis policy.
- Degraded mode activation when latency p95 breaches threshold.
- Deterministic watchdog supervisor (halt/cancel with manual resume).
- Pre-market warmup/health gate with persisted warmup report.
- Analyst model A/B benchmark harness with scored recommendation output.

### Phase E: News-aware dual-mode trading (implemented)
- **Spike Mode**: Activated around high-impact economic events (NFP, CPI, FOMC, etc.).
  - 3-task pipeline: EventBriefing → Sentiment → StrategyIntent
  - `event_briefing_analyst` agent with `EventBriefingReport` schema
  - Post-spike confirmation delay before analysis
  - Macro event windows ENCOURAGED (not blocked) in spike mode
  - Configurable spike-specific risk percentage
- **Daily Scalp Mode**: Existing technical trigger-based scalping.
  - 2-task pipeline: Sentiment → StrategyIntent
  - Macro event windows still blocked in daily scalp mode
- **Absolute dollar P&L targets** (the "$10 target" feature):
  - `DAILY_PROFIT_TARGET_USD` — hard lock when daily P&L reaches this amount
  - `DAILY_LOSS_LIMIT_USD` — hard lock when daily loss reaches this amount
  - `dollar_pnl` field tracked in `DailyState`
- **Economic Calendar Service** (`calendar_service.py`):
  - YAML-based calendar (default, reliable)
  - Optional HTTP API sources: Trading Economics, FCS API
  - `EconomicEvent` and `DailyCalendar` Pydantic models
  - Configurable cache TTL, high-impact-only filter
- **Event-Aware Scheduler** (`scheduler.py`):
  - Daemon-style runner that manages cycle timing automatically
  - Spike mode: tight polling (10s default)
  - Daily scalp: normal polling (45s default)
  - Stops on hard lock (daily profit/loss target reached)
  - `run_auto` CLI command

## 3) Current Architecture (Implemented)
- `MarketStreamSensor` (WebSocket-first, health/staleness).
- `EventTrigger` (`build_feature_snapshot` + `should_wake_ai` with trading_mode param).
- `CalendarService` (economic calendar, YAML + HTTP, caching, `EconomicEvent`/`DailyCalendar`).
- `CrewAI`:
  - Daily scalp pipeline (2-task): sentiment analysis → strategy intent
  - Spike mode pipeline (3-task): event briefing → sentiment → strategy intent
  - Agent: event_briefing_analyst (macro event analysis, `EventBriefingReport`)
  - Agent: market_sentiment_analyst (microstructure analysis)
  - Agent: strategy_composer (trade intent)
- `RiskEngine` deterministic pre-trade checks + dollar P&L targets.
- `ExecutionService` deterministic order workflow.
- `StateStore` SQLite WAL system of record.
- `Historian` daily analytics/recommendations.
- `QuotaGuard` + rate-limit backoff.
- `AnalysisCache` for analysis-report reuse.
- `LatencyPolicy` for p95 latency governance and degraded-mode control.
- `Watchdog` deterministic operational guardian.
- `Warmup` deterministic preflight health checks.
- `BenchmarkModels` harness for analyst model selection.
- `Scheduler` event-aware continuous runner with mode-switching.

## 4) Core Modules and Responsibilities
- `src/gold_trading_one_trade_per_day/main.py`
  - Runtime orchestration, dual-mode detection, spike confirmation delay.
  - `determine_trading_mode()`: spike vs daily_scalp based on calendar events.
  - Mode-specific risk parameters and pipeline selection.
  - CLI commands: `run_auto` (event-aware scheduler).
- `src/gold_trading_one_trade_per_day/calendar_service.py`
  - Economic calendar fetching and caching.
  - `CalendarService`, `EconomicEvent`, `DailyCalendar`.
  - YAML source (default) + HTTP API fallbacks.
- `src/gold_trading_one_trade_per_day/scheduler.py`
  - Daemon-style trading loop with automatic mode switching.
  - `run_auto_scheduler()` for continuous operation.
- `src/gold_trading_one_trade_per_day/crew.py`
  - Agent definitions and task guardrails.
  - Hybrid LLM routing integration.
  - `kickoff_spike_mode()` for 3-task spike pipeline.
  - `kickoff_strategy_only` for cached analysis path.
- `src/gold_trading_one_trade_per_day/event_trigger.py`
  - `should_wake_ai()` with `trading_mode` parameter.
  - Spike mode: macro events are encouraged (not blocked).
  - Daily scalp mode: macro events block trading (unchanged behavior).
- `src/gold_trading_one_trade_per_day/risk_engine.py`
  - Config precedence: env > profile yaml > defaults.
  - Dollar P&L target checks in `update_locks()`.
  - `daily_profit_target_usd` and `daily_loss_limit_usd` in `RiskConfig`.

## 5) Risk/Trading Constraints (Current Defaults)
- Symbol: `XAU_USD` (OANDA gold spot forex).
- Per-trade risk: `0.25%` (daily scalp), `3%` (spike mode via `SPIKE_RISK_PER_TRADE_PCT`).
- Soft lock: `+0.75%`.
- Hard profit lock: `+1.5%` (or `DAILY_PROFIT_TARGET_USD` if set).
- Hard loss lock: `-1.5%` (or `DAILY_LOSS_LIMIT_USD` if set).
- Max entries/day: `8`.
- Cooldown after close: `180s`.
- Max consecutive losses: `3`.
- Max spread: `$0.50` (XAU_USD typical spread is $0.30–$2.00).
- Intent TTL: `45s`.
- No new entries after: `23:00 ET` (XAU_USD trades 24/5).

Risk profile file:
- `src/gold_trading_one_trade_per_day/config/risk_profile.yaml`

## 6) Triggering and Session Rules
- Trigger requirements include:
  - volume spike
  - VWAP displacement
  - range expansion
  - spread gate
  - Trading hours checks (XAU_USD 24/5)
- Weekly market open warmup skip:
  - Sun 17:00–17:05 ET → `open_warmup`
- XAU_USD trading hours (is_rth):
  - Mon–Thu: always open
  - Fri: open before 17:00 ET
  - Sun: open after 17:00 ET
  - Sat: always closed
- **Macro event handling by mode**:
  - `daily_scalp`: macro events BLOCK trading (high volatility without event context).
  - `spike`: macro events DO NOT block trading (they are the trigger context).
  - Config: `event_windows.yaml` + `MACRO_EVENT_BLOCK_MINUTES` (for daily_scalp blocking).
- Spike mode windows:
  - Activates `SPIKE_PRE_EVENT_MINUTES` (default: 5) before high-impact event.
  - Stays active for `SPIKE_POST_EVENT_MINUTES` (default: 30) after release.
  - Post-spike confirmation delay: `SPIKE_CONFIRMATION_DELAY_SEC` (default: 30s).

Macro config file:
- `src/gold_trading_one_trade_per_day/config/event_windows.yaml`
Calendar YAML:
- Any YAML file supported by `CalendarService` (set `CALENDAR_YAML_FILE`).

## 7) LLM/Quota Controls
- Hybrid routing defaults:
  - `AGENT_MODEL_ANALYST=ollama/llama4:8b` (code default; recommended override: `ollama/llama3.1:8b`)
  - `AGENT_MODEL_STRATEGY=gemini/gemini-2.0-flash`
- LLM runtime defaults:
  - timeout `120s`
  - retries `5`
  - crew max RPM `8`
  - agent max RPM `6`
- Quota defaults:
  - `QUOTA_RPM_CAP=8`
  - `QUOTA_RPD_CAP=600`
  - reservation TTL `120s`
- Backoff defaults:
  - base wait `2s`
  - max wait `20s`
  - exponential with jitter

## 8) SQLite WAL Schema (Current Tables)
- `trading_day_state`
- `events`
- `analysis_reports`
- `strategy_intents`
- `intent_transitions`
- `risk_decisions`
- `orders`
- `order_updates`
- `positions`
- `daily_metrics`
- `llm_quota_ledger`
- `analysis_cache`
- `latency_metrics`
- `watchdog_events`
- `system_flags`

DB path default:
- `state.db` in project root (override with `STATE_DB_PATH`)

## 9) Intent Lifecycle
- Normal path:
  - `event_detected -> analysis_complete -> intent_generated -> risk_approved -> entry_submitted -> entry_filled -> oco_submitted -> closed`
- Terminal states:
  - `denied`, `expired`, `cancelled`, `halted`, `error`

## 10) CLI/Run Commands
- `run_shadow`
- `run_shadow_loop` (supports iteration arg)
- `run_paper`
- `run_live` (requires `ENABLE_LIVE_TRADING=true`)
- `run_auto` (event-aware scheduler; supports optional mode arg)
- `warmup`
- `benchmark_models`
- `resume` (manual watchdog clear)
- `reconcile`
- `train`, `replay`, `test`
- `run_with_trigger`

Useful examples:
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main run_shadow_loop 10`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main run_paper`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main run_auto shadow`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main warmup`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main benchmark_models "ollama/llama3.1:8b,ollama/gemma3:4b" 2`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main resume`
- `./.venv/bin/python -m unittest discover -s tests -v`

## 11) Environment Variables (Operational)
- OANDA:
  - `OANDA_API_TOKEN`
  - `OANDA_ACCOUNT_ID`
  - `OANDA_ENVIRONMENT` (`practice` [default] or `live`)
- LLM routing/runtime:
  - `AGENT_MODEL_ANALYST`
  - `AGENT_MODEL_STRATEGY`
  - `AGENT_LLM_TIMEOUT_SEC`
  - `AGENT_LLM_MAX_RETRIES`
  - `AGENT_LLM_TEMPERATURE`
  - `CREW_MAX_RPM`
  - `AGENT_MAX_RPM`
  - `OLLAMA_BASE_URL`
  - `ALLOW_ANALYST_GEMINI_FALLBACK_SHADOW`
  - `ALLOW_ANALYST_GEMINI_FALLBACK_PAPER_LIVE`
- Quota/backoff/cache:
  - `QUOTA_RPM_CAP`
  - `QUOTA_RPD_CAP`
  - `QUOTA_RESERVATION_TTL_SEC`
  - `LLM_BACKOFF_BASE_SEC`
  - `LLM_BACKOFF_MAX_SEC`
  - `ANALYSIS_CACHE_ENABLED`
  - `ANALYSIS_CACHE_TTL_SEC`
  - `ANALYSIS_PROMPT_VERSION`
  - `SIMULATE_429_ATTEMPTS`
- Session/trigger/data:
  - `OPEN_WARMUP_MINUTES`
  - `MACRO_EVENT_BLOCK_MINUTES`
  - `STREAM_STALE_SECONDS`
  - `REST_FALLBACK_INTERVAL_SECONDS`
  - `INTENT_TTL_SECONDS`
  - `EVENT_WINDOWS_FILE`
- Latency policy:
  - `LATENCY_SLA_WINDOW`
  - `LATENCY_DEGRADED_P95_MS`
  - `LATENCY_RECOVERY_P95_MS`
  - `LATENCY_RECOVERY_WINDOWS`
  - `LATENCY_MIN_SAMPLES`
  - `LATENCY_NORMAL_MAX_SLIPPAGE_BPS`
  - `LATENCY_DEGRADED_MAX_SLIPPAGE_BPS`
- Watchdog controls:
  - `WATCHDOG_LOOP_THRESHOLD`
  - `WATCHDOG_FLIP_THRESHOLD`
  - `WATCHDOG_STUCK_GRACE_SEC`
  - `WATCHDOG_DENY_LOOKBACK_MIN`
  - `WATCHDOG_FLATTEN_ON_HALT`
- Warmup controls:
  - `REQUIRE_WARMUP_PASS`
  - `WARMUP_MODE`
  - `WARMUP_MAX_AGE_MINUTES`
  - `WARMUP_MIN_QUOTA_RATIO`
  - `WARMUP_ALLOW_MOCK`
- Benchmark controls:
  - `BENCHMARK_MODELS`
  - `BENCHMARK_REPEATS`
  - `BENCHMARK_CANDIDATE_MODEL`
- Risk overrides:
  - `RISK_PER_TRADE_PCT`
  - `RISK_DAILY_SOFT_LOCK_PCT`
  - `RISK_DAILY_HARD_LOCK_PCT`
  - `RISK_DAILY_LOSS_LOCK_PCT`
  - `RISK_MAX_ENTRIES_PER_DAY`
  - `RISK_COOLDOWN_AFTER_CLOSE_SEC`
  - `RISK_MAX_CONSECUTIVE_LOSSES`
  - `RISK_MAX_SPREAD`
  - `RISK_INTENT_TTL_SECONDS`
  - `RISK_NO_NEW_ENTRIES_AFTER`
  - `DAILY_PROFIT_TARGET_USD` (absolute dollar profit target — e.g., 10.0 for "$10 target")
  - `DAILY_LOSS_LIMIT_USD` (absolute dollar loss limit — e.g., 5.0 for "$5 max loss")
- Dual-mode trading:
  - `SPIKE_MODE_ENABLED` (default: true)
  - `DAILY_SCALP_ENABLED` (default: true)
  - `SPIKE_PRE_EVENT_MINUTES` (default: 5)
  - `SPIKE_POST_EVENT_MINUTES` (default: 30)
  - `SPIKE_CONFIRMATION_DELAY_SEC` (default: 30)
  - `SPIKE_RISK_PER_TRADE_PCT` (default: 0.03)
  - `DAILY_SCALP_RISK_PER_TRADE_PCT` (override daily scalp risk)
- Economic calendar:
  - `CALENDAR_SOURCE` (default: yaml)
  - `CALENDAR_CACHE_TTL_SEC` (default: 3600)
  - `CALENDAR_HIGH_IMPACT_ONLY` (default: true)
  - `CALENDAR_YAML_FILE` (override default event_windows.yaml)
  - `TRADINGECONOMICS_API_KEY` (optional, for Trading Economics source)
  - `FCSAPI_KEY` (optional, for FCS API source)
- Scheduler:
  - `AUTO_CYCLE_INTERVAL_SEC` (default: 45)
  - `AUTO_SPIKE_CYCLE_INTERVAL_SEC` (default: 10)
  - `AUTO_IDLE_INTERVAL_SEC` (default: 120)
  - `AUTO_MODE` (default: shadow)

## 12) Tests and Current Status
- Test suite: `130` tests (as of Phase E).
- Latest result: all passing.
- New test files:
  - `tests/test_calendar_service.py` — calendar parsing, caching, filtering
  - `tests/test_trading_modes.py` — mode detection, dollar targets, event briefing schema
- Existing coverage maintained for:
  - trigger logic (including new trading_mode param)
  - risk locks/config precedence (including dollar targets)
  - state transitions/recovery
  - stream health
  - LLM routing
  - quota guard and backoff
  - analysis cache behavior
  - latency SLA hysteresis
  - watchdog halt/resume
  - warmup report pass/stale gating
  - benchmark scoring determinism

## 13) Observed Runtime Behavior and Interpretation
- `outside_rth` at night: expected, session gate working.
- `fallback_throttled` in rapid loops: expected with short sleep vs fallback interval.
- `stale_data` in paper mode: expected fail-closed when stream and REST are unavailable.
- `spread_too_wide` skips: expected protection gate.
- `macro_event_window` skips: only in `daily_scalp` mode; spike mode continues trading.

## 14) Known Issues and Practical Notes
- Ollama `0.17.0` was installed successfully.
- Some shell contexts can show MLX/Metal crashes for `ollama` CLI; normal terminal/service can still work.
- Model tag validity matters:
  - `llama4:8b` does not exist in Ollama library.
  - Working local example used: `llama3.1:8b`.
- Running outside RTH will correctly skip events.
- Spike confirmation delay (`SPIKE_CONFIRMATION_DELAY_SEC=30`) can be set to 0 in testing.
- Calendar service gracefully degrades to empty calendar on API failure (falls back to daily scalp mode).

## 15) Security and Ops Note
- `.env` currently contains active broker/model secrets in this environment.
- This document intentionally does not include secret values.
- Recommended operational hygiene: rotate keys if shared accidentally; keep `.env` untracked.

## 16) Immediate Next Actions
- Run `warmup` before paper/live sessions and enable `REQUIRE_WARMUP_PASS=true`.
- Monitor `signal_to_fill_ms_p95` and watchdog incidents in `reconcile` output.
- Use `benchmark_models` to compare analyst candidates before changing defaults.
- Run in RTH after 9:35 ET for meaningful trigger behavior.
- Keep `SHADOW_LOOP_SLEEP_SEC` >= fallback interval for cleaner loop diagnostics.
- Use `run_shadow`/`run_shadow_loop` for development; use `crewai test` sparingly to protect Gemini quota.
- Promote to paper/live only after stable skip/execute patterns and historian metrics remain healthy.
- To use `run_auto` for continuous trading: `python -m gold_trading_one_trade_per_day.main run_auto shadow`
- Set `DAILY_PROFIT_TARGET_USD=10` and `DAILY_LOSS_LIMIT_USD=5` for the "$10 profit, $5 loss" daily rule.

## 1) Mission and Scope
- Build a deterministic, event-driven GLD scalping system.
- Use CrewAI agents only for high-value decision moments.
- Keep risk checks and execution deterministic and non-bypassable.
- Operate with rollout path: shadow -> paper -> live.

## 2) Plan Evolution (What Was Decided and Implemented)

### Phase A: Safety-first architecture
- Deterministic event trigger before any LLM call.
- Typed contracts for all agent/execution data.
- SQLite WAL state machine for crash recovery.
- Deterministic risk gate and deterministic execution service.
- Daily anti-greed and hard-loss locks.

### Phase B: Reality-checked market constraints
- Lock thresholds recalibrated for GLD:
  - soft lock: +0.75%
  - hard profit lock: +1.5%
  - hard loss lock: -1.5%
- Sensor changed to WebSocket-first + REST fallback.
- Open warm-up no-trade window (9:30-9:35 ET).
- Macro event no-trade windows (config-driven).
- Fail-closed behavior when stream + fallback are unavailable.

### Phase C: Gemini free-tier quota resilience
- Hybrid model routing:
  - Analyst: local Ollama (default)
  - Strategy: Gemini Flash
- Crew and agent RPM caps.
- Pre-LLM deterministic quota gate with RPM/RPD budgets.
- 429 backoff-until-expiry policy with persisted telemetry.
- Analysis-level cache for reusing `MarketSentimentReport` only.
- No strategy intent cache reuse across events.

### Phase D: Go-live execution optimization (implemented)
- Deterministic latency telemetry + SLA/hysteresis policy.
- Degraded mode activation when latency p95 breaches threshold.
- Deterministic watchdog supervisor (halt/cancel with manual resume).
- Pre-market warmup/health gate with persisted warmup report.
- Analyst model A/B benchmark harness with scored recommendation output.
- Explicitly deferred MCP migration (phase-2 after paper stability gates).

## 3) Current Architecture (Implemented)
- `MarketStreamSensor` (WebSocket-first, health/staleness).
- `Event Trigger` (`build_feature_snapshot` + `should_wake_ai`).
- `CrewAI`:
  - Agent A: market/sentiment analysis.
  - Agent B: strategy intent composition.
- `RiskEngine` deterministic pre-trade checks.
- `ExecutionService` deterministic order workflow.
- `StateStore` SQLite WAL system of record.
- `Historian` daily analytics/recommendations.
- `QuotaGuard` + rate-limit backoff.
- `AnalysisCache` for analysis-report reuse.
- `LatencyPolicy` for p95 latency governance and degraded-mode control.
- `Watchdog` deterministic operational guardian.
- `Warmup` deterministic preflight health checks.
- `BenchmarkModels` harness for analyst model selection.

## 4) Core Modules and Responsibilities
- `src/gold_trading_one_trade_per_day/main.py`
  - Runtime orchestration.
  - Stream/fallback snapshot build.
  - Trigger checks and skip reason recording.
  - Quota reservation/commit.
  - 429 backoff orchestration.
  - Analysis cache hit path (strategy-only kickoff).
  - Risk/execution deterministic path.
  - Latency policy evaluation and telemetry writes.
  - Watchdog halt checks before event admission.
  - Warmup gate enforcement for paper/live when enabled.
  - CLI commands: `warmup`, `benchmark_models`, `resume`.
- `src/gold_trading_one_trade_per_day/crew.py`
  - Agent definitions and task guardrails.
  - Hybrid LLM routing integration.
  - `kickoff_strategy_only` for cached analysis path.
- `src/gold_trading_one_trade_per_day/llm_factory.py`
  - LLM runtime config load.
  - Local availability checks for Ollama endpoint.
  - Shadow fallback policy to Gemini when configured.
- `src/gold_trading_one_trade_per_day/quota_guard.py`
  - RPM/RPD caps.
  - Reservation/commit accounting.
  - 429 detection and TTL-aware exponential backoff.
- `src/gold_trading_one_trade_per_day/analysis_cache.py`
  - Snapshot normalization/hash keying.
  - TTL-based `MarketSentimentReport` reuse.
- `src/gold_trading_one_trade_per_day/state_store.py`
  - WAL DB schema and persistence methods.
  - Intent lifecycle, transitions, events, orders, positions, metrics.
  - Quota ledger and analysis cache storage.
- `src/gold_trading_one_trade_per_day/risk_engine.py`
  - Config precedence: env > profile yaml > defaults.
  - Spread, TTL, lock, cooldown, entries/day, sizing checks.
- `src/gold_trading_one_trade_per_day/historian.py`
  - Daily stats including skip reasons, fallback usage, quota/caching metrics.
- `src/gold_trading_one_trade_per_day/latency_policy.py`
  - Rolling-window p95 latency evaluation.
  - Degraded-mode hysteresis and effective slippage policy.
- `src/gold_trading_one_trade_per_day/watchdog.py`
  - Deterministic halt triggers (loops, side-flips, stuck entries, instability).
  - Emergency actions: cancel all open orders, optional flatten, halt flag.
- `src/gold_trading_one_trade_per_day/warmup.py`
  - 9:15 ET style preflight checks and cache priming.
  - `WarmupReport` persistence + recency/pass validation.
- `src/gold_trading_one_trade_per_day/benchmark_models.py`
  - Fixture-based analyst A/B model benchmarking with deterministic scoring.

## 5) Risk/Trading Constraints (Current Defaults)
- Symbol: `XAU_USD` (OANDA gold spot forex).
- Per-trade risk: `0.25%`.
- Soft lock: `+0.75%`.
- Hard profit lock: `+1.5%`.
- Hard loss lock: `-1.5%`.
- Max entries/day: `8`.
- Cooldown after close: `180s`.
- Max consecutive losses: `3`.
- Max spread: `$0.50` (XAU_USD typical spread is $0.30–$2.00).
- Intent TTL: `45s`.
- No new entries after: `23:00 ET` (XAU_USD trades 24/5).

Risk profile file:
- `src/gold_trading_one_trade_per_day/config/risk_profile.yaml`

## 6) Triggering and Session Rules
- Trigger requirements include:
  - volume spike
  - VWAP displacement
  - range expansion
  - spread gate
  - Trading hours checks (XAU_USD 24/5)
- Weekly market open warmup skip:
  - Sun 17:00–17:05 ET → `open_warmup`
- XAU_USD trading hours (is_rth):
  - Mon–Thu: always open
  - Fri: open before 17:00 ET
  - Sun: open after 17:00 ET
  - Sat: always closed
- Macro event skip:
  - `event_windows.yaml` + `MACRO_EVENT_BLOCK_MINUTES`
- If no valid data path:
  - skip with `stale_data`

Macro config file:
- `src/gold_trading_one_trade_per_day/config/event_windows.yaml`

## 7) LLM/Quota Controls
- Hybrid routing defaults:
  - `AGENT_MODEL_ANALYST=ollama/llama4:8b` (code default; recommended override: `ollama/llama3.1:8b`)
  - `AGENT_MODEL_STRATEGY=gemini/gemini-2.0-flash`
- LLM runtime defaults:
  - timeout `120s`
  - retries `5`
  - crew max RPM `8`
  - agent max RPM `6`
- Quota defaults:
  - `QUOTA_RPM_CAP=8`
  - `QUOTA_RPD_CAP=600`
  - reservation TTL `120s`
- Backoff defaults:
  - base wait `2s`
  - max wait `20s`
  - exponential with jitter

## 8) SQLite WAL Schema (Current Tables)
- `trading_day_state`
- `events`
- `analysis_reports`
- `strategy_intents`
- `intent_transitions`
- `risk_decisions`
- `orders`
- `order_updates`
- `positions`
- `daily_metrics`
- `llm_quota_ledger`
- `analysis_cache`
- `latency_metrics`
- `watchdog_events`
- `system_flags`

DB path default:
- `state.db` in project root (override with `STATE_DB_PATH`)

## 9) Intent Lifecycle
- Normal path:
  - `event_detected -> analysis_complete -> intent_generated -> risk_approved -> entry_submitted -> entry_filled -> oco_submitted -> closed`
- Terminal states:
  - `denied`, `expired`, `cancelled`, `halted`, `error`

## 10) CLI/Run Commands
- `run_shadow`
- `run_shadow_loop` (supports iteration arg)
- `run_paper`
- `run_live` (requires `ENABLE_LIVE_TRADING=true`)
- `warmup`
- `benchmark_models`
- `resume` (manual watchdog clear)
- `reconcile`
- `train`, `replay`, `test`
- `run_with_trigger`

Useful examples:
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main run_shadow_loop 10`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main run_paper`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main warmup`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main benchmark_models "ollama/llama3.1:8b,ollama/gemma3:4b" 2`
- `./.venv/bin/python -m gold_trading_one_trade_per_day.main resume`
- `./.venv/bin/python -m unittest discover -s tests -v`

## 11) Environment Variables (Operational)
- OANDA:
  - `OANDA_API_TOKEN`
  - `OANDA_ACCOUNT_ID`
  - `OANDA_ENVIRONMENT` (`practice` [default] or `live`)
- LLM routing/runtime:
  - `AGENT_MODEL_ANALYST`
  - `AGENT_MODEL_STRATEGY`
  - `AGENT_LLM_TIMEOUT_SEC`
  - `AGENT_LLM_MAX_RETRIES`
  - `AGENT_LLM_TEMPERATURE`
  - `CREW_MAX_RPM`
  - `AGENT_MAX_RPM`
  - `OLLAMA_BASE_URL`
  - `ALLOW_ANALYST_GEMINI_FALLBACK_SHADOW`
  - `ALLOW_ANALYST_GEMINI_FALLBACK_PAPER_LIVE`
- Quota/backoff/cache:
  - `QUOTA_RPM_CAP`
  - `QUOTA_RPD_CAP`
  - `QUOTA_RESERVATION_TTL_SEC`
  - `LLM_BACKOFF_BASE_SEC`
  - `LLM_BACKOFF_MAX_SEC`
  - `ANALYSIS_CACHE_ENABLED`
  - `ANALYSIS_CACHE_TTL_SEC`
  - `ANALYSIS_PROMPT_VERSION`
  - `SIMULATE_429_ATTEMPTS`
- Session/trigger/data:
  - `OPEN_WARMUP_MINUTES`
  - `MACRO_EVENT_BLOCK_MINUTES`
  - `STREAM_STALE_SECONDS`
  - `REST_FALLBACK_INTERVAL_SECONDS`
  - `INTENT_TTL_SECONDS`
  - `EVENT_WINDOWS_FILE`
- Latency policy:
  - `LATENCY_SLA_WINDOW`
  - `LATENCY_DEGRADED_P95_MS`
  - `LATENCY_RECOVERY_P95_MS`
  - `LATENCY_RECOVERY_WINDOWS`
  - `LATENCY_MIN_SAMPLES`
  - `LATENCY_NORMAL_MAX_SLIPPAGE_BPS`
  - `LATENCY_DEGRADED_MAX_SLIPPAGE_BPS`
- Watchdog controls:
  - `WATCHDOG_LOOP_THRESHOLD`
  - `WATCHDOG_FLIP_THRESHOLD`
  - `WATCHDOG_STUCK_GRACE_SEC`
  - `WATCHDOG_DENY_LOOKBACK_MIN`
  - `WATCHDOG_FLATTEN_ON_HALT`
- Warmup controls:
  - `REQUIRE_WARMUP_PASS`
  - `WARMUP_MODE`
  - `WARMUP_MAX_AGE_MINUTES`
  - `WARMUP_MIN_QUOTA_RATIO`
  - `WARMUP_ALLOW_MOCK`
- Benchmark controls:
  - `BENCHMARK_MODELS`
  - `BENCHMARK_REPEATS`
  - `BENCHMARK_CANDIDATE_MODEL`
- Risk overrides:
  - `RISK_PER_TRADE_PCT`
  - `RISK_DAILY_SOFT_LOCK_PCT`
  - `RISK_DAILY_HARD_LOCK_PCT`
  - `RISK_DAILY_LOSS_LOCK_PCT`
  - `RISK_MAX_ENTRIES_PER_DAY`
  - `RISK_COOLDOWN_AFTER_CLOSE_SEC`
  - `RISK_MAX_CONSECUTIVE_LOSSES`
  - `RISK_MAX_SPREAD`
  - `RISK_INTENT_TTL_SECONDS`
  - `RISK_NO_NEW_ENTRIES_AFTER`

## 12) Tests and Current Status
- Test suite: `22` tests.
- Latest result: all passing.
- Includes coverage for:
  - trigger logic
  - risk locks/config precedence
  - state transitions/recovery
  - stream health
  - LLM routing
  - quota guard and backoff
  - analysis cache behavior
  - latency SLA hysteresis
  - watchdog halt/resume
  - warmup report pass/stale gating
  - benchmark scoring determinism

## 13) Observed Runtime Behavior and Interpretation
- `outside_rth` at night: expected, session gate working.
- `fallback_throttled` in rapid loops: expected with short sleep vs fallback interval.
- `stale_data` in paper mode: expected fail-closed when stream and REST are unavailable.
- `spread_too_wide` skips: expected protection gate.

## 14) Known Issues and Practical Notes
- Ollama `0.17.0` was installed successfully.
- Some shell contexts can show MLX/Metal crashes for `ollama` CLI; normal terminal/service can still work.
- Model tag validity matters:
  - `llama4:8b` does not exist in Ollama library.
  - Working local example used: `llama3.1:8b`.
- Running outside RTH will correctly skip events.

## 15) Security and Ops Note
- `.env` currently contains active broker/model secrets in this environment.
- This document intentionally does not include secret values.
- Recommended operational hygiene: rotate keys if shared accidentally; keep `.env` untracked.

## 16) Immediate Next Actions
- Run `warmup` before paper/live sessions and enable `REQUIRE_WARMUP_PASS=true`.
- Monitor `signal_to_fill_ms_p95` and watchdog incidents in `reconcile` output.
- Use `benchmark_models` to compare analyst candidates before changing defaults.
- Run in RTH after 9:35 ET for meaningful trigger behavior.
- Keep `SHADOW_LOOP_SLEEP_SEC` >= fallback interval for cleaner loop diagnostics.
- Use `run_shadow`/`run_shadow_loop` for development; use `crewai test` sparingly to protect Gemini quota.
- Promote to paper/live only after stable skip/execute patterns and historian metrics remain healthy.
