"""Paper trading simulator.

Wraps the backtester to provide a user-friendly interface for running
paper trading simulations with detailed reporting.

Features:
- Multiple scenario simulation (trending, range-bound, volatile)
- Aggregated performance across scenarios
- Trade log with per-trade details
- Equity curve data for plotting
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gold_trading_one_trade_per_day.backtester import (
    BacktestConfig,
    BacktestResult,
    Trade,
    run_backtest,
)
from gold_trading_one_trade_per_day.synthetic_data import (
    generate_multi_day_data,
    generate_range_day,
    generate_trending_day,
    generate_volatile_event_day,
)


@dataclass
class ScenarioResult:
    """Result from a single scenario."""
    name: str
    backtest: BacktestResult
    description: str


@dataclass
class PaperTradingReport:
    """Aggregated paper trading report across scenarios."""
    scenarios: List[ScenarioResult]
    config: BacktestConfig

    # Aggregated metrics
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    overall_win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_profit_factor: float = 0.0
    worst_drawdown_pct: float = 0.0
    avg_sharpe: float = 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics across all scenarios."""
        if not self.scenarios:
            return

        self.total_trades = sum(s.backtest.total_trades for s in self.scenarios)
        self.total_wins = sum(s.backtest.winning_trades for s in self.scenarios)
        self.total_losses = sum(s.backtest.losing_trades for s in self.scenarios)
        self.overall_win_rate = (
            (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
        )
        self.total_pnl = sum(s.backtest.total_pnl for s in self.scenarios)
        self.avg_pnl_per_trade = (
            (self.total_pnl / self.total_trades) if self.total_trades > 0 else 0.0
        )

        pfs = [s.backtest.profit_factor for s in self.scenarios if s.backtest.total_trades > 0]
        self.avg_profit_factor = (sum(pfs) / len(pfs)) if pfs else 0.0

        self.worst_drawdown_pct = max(
            (s.backtest.max_drawdown_pct for s in self.scenarios),
            default=0.0,
        )

        sharpes = [s.backtest.sharpe_ratio for s in self.scenarios if s.backtest.total_trades > 0]
        self.avg_sharpe = (sum(sharpes) / len(sharpes)) if sharpes else 0.0

    def summary(self) -> str:
        """Human-readable summary report."""
        lines = [
            "=" * 70,
            "PAPER TRADING SIMULATION REPORT",
            "=" * 70,
            f"Starting Capital: ${self.config.initial_equity:.2f}",
            f"Risk Per Trade:   {self.config.risk_per_trade_pct * 100:.1f}%",
            f"Min Confidence:   {self.config.min_confidence:.2f}",
            f"Min R:R:          {self.config.min_risk_reward:.1f}",
            "",
            "-" * 70,
            "SCENARIO BREAKDOWN",
            "-" * 70,
        ]

        for s in self.scenarios:
            b = s.backtest
            lines.extend([
                f"\n  {s.name}: {s.description}",
                f"    Trades: {b.total_trades}  |  "
                f"Win Rate: {b.win_rate:.1f}%  |  "
                f"P&L: ${b.total_pnl:.2f}  |  "
                f"Return: {b.return_pct:.2f}%",
                f"    Max DD: {b.max_drawdown_pct:.2f}%  |  "
                f"Sharpe: {b.sharpe_ratio:.2f}  |  "
                f"PF: {b.profit_factor:.2f}",
                f"    Final Equity: ${b.final_equity:.2f}",
            ])

        lines.extend([
            "",
            "-" * 70,
            "AGGREGATE PERFORMANCE",
            "-" * 70,
            f"Total Trades:        {self.total_trades}",
            f"Total Wins:          {self.total_wins}",
            f"Total Losses:        {self.total_losses}",
            f"Overall Win Rate:    {self.overall_win_rate:.1f}%",
            f"Total P&L:           ${self.total_pnl:.2f}",
            f"Avg P&L Per Trade:   ${self.avg_pnl_per_trade:.2f}",
            f"Avg Profit Factor:   {self.avg_profit_factor:.2f}",
            f"Worst Drawdown:      {self.worst_drawdown_pct:.2f}%",
            f"Avg Sharpe Ratio:    {self.avg_sharpe:.2f}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def trade_log(self) -> str:
        """Detailed trade log across all scenarios."""
        lines = [
            "=" * 100,
            "DETAILED TRADE LOG",
            "=" * 100,
            f"{'#':>3} {'Scenario':<15} {'Side':<5} {'Setup':<22} "
            f"{'Entry':>10} {'Exit':>10} {'SL':>10} {'TP':>10} "
            f"{'P&L':>8} {'Exit Reason':<12}",
            "-" * 100,
        ]

        trade_num = 0
        for s in self.scenarios:
            for t in s.backtest.trades:
                trade_num += 1
                lines.append(
                    f"{trade_num:>3} {s.name:<15} {t.side:<5} {t.setup_type:<22} "
                    f"${t.entry_price:>9.2f} ${t.exit_price:>9.2f} "
                    f"${t.stop_loss:>9.2f} ${t.take_profit:>9.2f} "
                    f"${t.pnl:>7.2f} {t.exit_reason:<12}"
                )

        lines.append("=" * 100)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "config": {
                "initial_equity": self.config.initial_equity,
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
                "min_confidence": self.config.min_confidence,
                "min_risk_reward": self.config.min_risk_reward,
                "lookback_bars": self.config.lookback_bars,
                "max_holding_bars": self.config.max_holding_bars,
            },
            "aggregate": {
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "overall_win_rate": round(self.overall_win_rate, 2),
                "total_pnl": round(self.total_pnl, 2),
                "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
                "avg_profit_factor": round(self.avg_profit_factor, 2),
                "worst_drawdown_pct": round(self.worst_drawdown_pct, 2),
                "avg_sharpe": round(self.avg_sharpe, 2),
            },
            "scenarios": [
                {
                    "name": s.name,
                    "description": s.description,
                    "total_trades": s.backtest.total_trades,
                    "win_rate": round(s.backtest.win_rate, 2),
                    "total_pnl": round(s.backtest.total_pnl, 2),
                    "return_pct": round(s.backtest.return_pct, 2),
                    "max_drawdown_pct": round(s.backtest.max_drawdown_pct, 2),
                    "sharpe_ratio": round(s.backtest.sharpe_ratio, 2),
                    "profit_factor": round(s.backtest.profit_factor, 2),
                    "final_equity": round(s.backtest.final_equity, 2),
                    "trades": [
                        {
                            "id": t.trade_id,
                            "side": t.side,
                            "setup_type": t.setup_type,
                            "confidence": t.confidence,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "stop_loss": t.stop_loss,
                            "take_profit": t.take_profit,
                            "pnl": t.pnl,
                            "pnl_pct": t.pnl_pct,
                            "exit_reason": t.exit_reason,
                            "holding_bars": t.holding_bars,
                            "rr_target": t.risk_reward_target,
                            "rr_actual": t.risk_reward_actual,
                        }
                        for t in s.backtest.trades
                    ],
                }
                for s in self.scenarios
            ],
        }


def run_paper_trading(
    initial_equity: float = 100.0,
    risk_pct: float = 0.02,
    seed: int = 42,
    num_multi_day: int = 30,
) -> PaperTradingReport:
    """Run a comprehensive paper trading simulation.

    Runs the strategy across multiple market scenarios:
    1. Trending up day
    2. Trending down day
    3. Range-bound day
    4. Volatile event day
    5. Multi-day mixed conditions

    Args:
        initial_equity: Starting capital (e.g., $100)
        risk_pct: Risk per trade as decimal (e.g., 0.02 = 2%)
        seed: Random seed for reproducibility
        num_multi_day: Number of days for multi-day scenario

    Returns:
        PaperTradingReport with all scenarios and aggregate metrics
    """
    config = BacktestConfig(
        initial_equity=initial_equity,
        risk_per_trade_pct=risk_pct,
        min_confidence=0.35,
        min_risk_reward=1.3,
        lookback_bars=60,
        max_holding_bars=50,
        slippage_pct=0.0001,
        allow_multiple_trades_per_day=True,
        max_trades_per_day=3,
    )

    scenarios: List[ScenarioResult] = []

    # Scenario 1: Trending up
    up_data = generate_trending_day(direction="up", start_price=2650.0, seed=seed)
    up_result = run_backtest(up_data, config)
    scenarios.append(ScenarioResult(
        name="Trend Up",
        backtest=up_result,
        description="Strong bullish trend day (~$20-40 move up)",
    ))

    # Scenario 2: Trending down
    down_data = generate_trending_day(direction="down", start_price=2650.0, seed=seed + 1)
    down_result = run_backtest(down_data, config)
    scenarios.append(ScenarioResult(
        name="Trend Down",
        backtest=down_result,
        description="Strong bearish trend day (~$20-40 move down)",
    ))

    # Scenario 3: Range-bound
    range_data = generate_range_day(start_price=2650.0, seed=seed + 2)
    range_result = run_backtest(range_data, config)
    scenarios.append(ScenarioResult(
        name="Range Day",
        backtest=range_result,
        description="Low-volatility range-bound day",
    ))

    # Scenario 4: Volatile event day
    vol_data = generate_volatile_event_day(start_price=2650.0, seed=seed + 3)
    vol_result = run_backtest(vol_data, config)
    scenarios.append(ScenarioResult(
        name="Volatile Event",
        backtest=vol_result,
        description="High-vol event day (NFP/FOMC-like)",
    ))

    # Scenario 5: Multi-day mixed
    multi_data = generate_multi_day_data(
        num_days=num_multi_day,
        start_price=2650.0,
        seed=seed + 4,
    )
    multi_config = BacktestConfig(
        initial_equity=initial_equity,
        risk_per_trade_pct=risk_pct,
        min_confidence=0.35,
        min_risk_reward=1.3,
        lookback_bars=60,
        max_holding_bars=50,
        slippage_pct=0.0001,
        allow_multiple_trades_per_day=True,
        max_trades_per_day=3,
    )
    multi_result = run_backtest(multi_data, multi_config)
    scenarios.append(ScenarioResult(
        name="Multi-Day Mix",
        backtest=multi_result,
        description=f"{num_multi_day}-day mixed conditions (trend + range + events)",
    ))

    report = PaperTradingReport(
        scenarios=scenarios,
        config=config,
    )
    report.compute_aggregates()
    return report
