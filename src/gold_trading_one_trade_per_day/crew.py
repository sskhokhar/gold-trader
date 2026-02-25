from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from gold_trading_one_trade_per_day.schemas import MarketSentimentReport, StrategyIntent
from gold_trading_one_trade_per_day.tools.alpaca_tools import AlpacaDataTool


@CrewBase
class GoldTradingOneTradePerDayCrew:
    """Event-driven GLD scalping crew for analysis and strategy intent generation."""

    @staticmethod
    def _llm() -> LLM:
        model_name = os.getenv("AGENT_MODEL", "gemini/gemini-2.5-flash")
        return LLM(model=model_name, temperature=0.2)

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
            return True, model.model_dump_json()
        except Exception as exc:
            return False, f"invalid StrategyIntent: {exc}"

    @agent
    def market_sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_sentiment_analyst"],
            tools=[AlpacaDataTool()],
            allow_delegation=False,
            reasoning=False,
            max_iter=8,
            llm=self._llm(),
        )

    @agent
    def strategy_composer(self) -> Agent:
        return Agent(
            config=self.agents_config["strategy_composer"],
            tools=[],
            allow_delegation=False,
            reasoning=False,
            max_iter=8,
            llm=self._llm(),
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
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=self._llm(),
        )
