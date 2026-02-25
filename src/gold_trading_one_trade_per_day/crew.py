import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	ScrapeWebsiteTool
)
from gold_trading_one_trade_per_day.tools.alpaca_tools import AlpacaDataTool, AlpacaExecutionTool





@CrewBase
class GoldTradingOneTradePerDayCrew:
    """GoldTradingOneTradePerDay crew"""

    
    @agent
    def gold_market_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["gold_market_analyst"],
            tools=[AlpacaDataTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def risk_management_strategist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["risk_management_strategist"],
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    
    @agent
    def trade_executioner(self) -> Agent:
        
        return Agent(
            config=self.agents_config["trade_executioner"],
            tools=[AlpacaExecutionTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.5-flash",
                temperature=0.7,
            ),
        )
    

    
    @task
    def analyze_xau_usd_15m_liquidity_grabs(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_xau_usd_15m_liquidity_grabs"],
            markdown=False,
            
            
        )
    
    @task
    def validate_setup_and_calculate_risk_parameters(self) -> Task:
        return Task(
            config=self.tasks_config["validate_setup_and_calculate_risk_parameters"],
            markdown=False,
            
            
        )
    
    @task
    def execute_single_daily_trade(self) -> Task:
        return Task(
            config=self.tasks_config["execute_single_daily_trade"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the GoldTradingOneTradePerDay crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="gemini/gemini-2.5-flash"),
        )


