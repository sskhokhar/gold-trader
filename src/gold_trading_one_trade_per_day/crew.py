from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from gold_trading_one_trade_per_day.llm_factory import (
    LLMRoute,
    LLMRuntimeConfig,
    build_llm,
    load_llm_runtime_config,
    resolve_llm_route,
)
from gold_trading_one_trade_per_day.schemas import EventBriefingReport, MarketSentimentReport, StrategyIntent
from gold_trading_one_trade_per_day.tools.alpaca_tools import AlpacaDataTool


@CrewBase
class GoldTradingOneTradePerDayCrew:
    """Event-driven XAU_USD scalping crew for analysis and strategy intent generation."""

    def __init__(self, mode: str = "shadow"):
        self.mode = mode
        self.runtime_cfg: LLMRuntimeConfig = load_llm_runtime_config()
        self.route: LLMRoute = resolve_llm_route(mode=mode, cfg=self.runtime_cfg)

    def runtime_settings(self) -> dict:
        return {
            "source": self.runtime_cfg.source,
            "mode": self.mode,
            "analyst_model": self.route.analyst_model,
            "strategy_model": self.route.strategy_model,
            "analyst_route": self.route.analyst_route,
            "strategy_route": self.route.strategy_route,
            "analyst_fallback_used": self.route.analyst_fallback_used,
            "analyst_local_available": self.route.analyst_local_available,
            "strategy_auto_overridden": self.route.strategy_auto_overridden,
            "timeout_sec": self.runtime_cfg.timeout_sec,
            "max_retries": self.runtime_cfg.max_retries,
            "temperature": self.runtime_cfg.temperature,
            "crew_max_rpm": self.runtime_cfg.crew_max_rpm,
            "agent_max_rpm": self.runtime_cfg.agent_max_rpm,
        }

    def _analyst_llm(self) -> LLM:
        return build_llm(self.route.analyst_model, self.runtime_cfg)

    def _strategy_llm(self) -> LLM:
        return build_llm(self.route.strategy_model, self.runtime_cfg)

    @staticmethod
    def _validate_market_report(task_output):
        try:
            if task_output.pydantic:
                model = MarketSentimentReport.model_validate(task_output.pydantic.model_dump())
            else:
                model = MarketSentimentReport.model_validate_json(task_output.raw)
            return True, model.model_dump_json()
        except Exception as exc:
            return False, f"invalid MarketSentimentReport: {exc}"

    @staticmethod
    def _validate_strategy_intent(task_output):
        try:
            if task_output.pydantic:
                model = StrategyIntent.model_validate(task_output.pydantic.model_dump())
            else:
                model = StrategyIntent.model_validate_json(task_output.raw)

            now = datetime.now(timezone.utc)
            max_past_sec = max(int(os.getenv("STRATEGY_INTENT_MAX_PAST_SEC", "300")), 1)
            max_future_sec = max(int(os.getenv("STRATEGY_INTENT_MAX_FUTURE_SEC", "30")), 1)

            if model.generated_at < (now - timedelta(seconds=max_past_sec)):
                return False, "invalid StrategyIntent: generated_at too old"
            if model.generated_at > (now + timedelta(seconds=max_future_sec)):
                return False, "invalid StrategyIntent: generated_at too far in future"

            ttl_sec = (model.expires_at - model.generated_at).total_seconds()
            if ttl_sec <= 0:
                return False, f"invalid StrategyIntent: ttl_sec={ttl_sec} must be positive"
            return True, model.model_dump_json()
        except Exception as exc:
            return False, f"invalid StrategyIntent: {exc}"

    @staticmethod
    def _validate_event_briefing(task_output):
        try:
            if task_output.pydantic:
                model = EventBriefingReport.model_validate(task_output.pydantic.model_dump())
            else:
                model = EventBriefingReport.model_validate_json(task_output.raw)
            return True, model.model_dump_json()
        except Exception as exc:
            return False, f"invalid EventBriefingReport: {exc}"

    @agent
    def event_briefing_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["event_briefing_analyst"],
            tools=[],
            allow_delegation=False,
            reasoning=False,
            max_iter=2,
            max_rpm=self.runtime_cfg.agent_max_rpm,
            llm=self._analyst_llm(),
        )

    @task
    def produce_event_briefing(self) -> Task:
        return Task(
            config=self.tasks_config["produce_event_briefing"],
            output_pydantic=EventBriefingReport,
            guardrail=self._validate_event_briefing,
            markdown=False,
        )

    @agent
    def market_sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_sentiment_analyst"],
            tools=[AlpacaDataTool()],
            allow_delegation=False,
            reasoning=False,
            max_iter=3,
            max_rpm=self.runtime_cfg.agent_max_rpm,
            llm=self._analyst_llm(),
        )

    @agent
    def strategy_composer(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_composer"],
            tools=[],
            allow_delegation=False,
            reasoning=False,
            max_iter=2,
            max_rpm=self.runtime_cfg.agent_max_rpm,
            llm=self._strategy_llm(),
        )

    @task
    def analyze_market_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_market_sentiment"],
            output_pydantic=MarketSentimentReport,
            guardrail=self._validate_market_report,
            markdown=False,
        )

    @task
    def compose_strategy_intent(self) -> Task:
        return Task(
            config=self.tasks_config["compose_strategy_intent"],
            output_pydantic=StrategyIntent,
            guardrail=self._validate_strategy_intent,
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.market_sentiment_analyst(), self.strategy_composer()],
            tasks=[self.analyze_market_sentiment(), self.compose_strategy_intent()],
            process=Process.sequential,
            verbose=True,
            chat_llm=self._strategy_llm(),
            cache=True,
            max_rpm=self.runtime_cfg.crew_max_rpm,
        )

    def spike_crew(self) -> Crew:
        """3-task pipeline: event briefing → sentiment → strategy intent."""
        return Crew(
            agents=[
                self.event_briefing_analyst(),
                self.market_sentiment_analyst(),
                self.strategy_composer(),
            ],
            tasks=[
                self.produce_event_briefing(),
                self.analyze_market_sentiment(),
                self.compose_strategy_intent(),
            ],
            process=Process.sequential,
            verbose=True,
            chat_llm=self._strategy_llm(),
            cache=True,
            max_rpm=self.runtime_cfg.crew_max_rpm,
        )

    def kickoff_spike_mode(self, inputs: dict) -> object:
        """Run the 3-task spike mode pipeline with event briefing pre-analysis."""
        return self.spike_crew().kickoff(inputs=inputs)

    def kickoff_strategy_only(self, inputs: dict) -> object:
        strategy_agent = self.strategy_composer()
        strategy_task_cfg = self.tasks_config["compose_strategy_intent"]
        task = Task(
            description=(
                f"{strategy_task_cfg['description']}\n\n"
                "Use the provided cached report instead of generating a new analyst report.\n"
                "Current UTC time (must anchor generated_at/expires_at): {current_utc_iso}\n"
                "Intent TTL seconds: {intent_ttl_seconds}\n\n"
                "Feature Snapshot JSON:\n{feature_snapshot_json}\n\n"
                "Cached MarketSentimentReport JSON:\n{cached_market_report_json}\n"
            ),
            expected_output=strategy_task_cfg["expected_output"],
            output_pydantic=StrategyIntent,
            guardrail=self._validate_strategy_intent,
            markdown=False,
            agent=strategy_agent,
        )
        scoped = Crew(
            agents=[strategy_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            chat_llm=self._strategy_llm(),
            cache=True,
            max_rpm=self.runtime_cfg.crew_max_rpm,
        )
        return scoped.kickoff(inputs=inputs)
