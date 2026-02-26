"""Backtester engine for the gold strategy.

Simulates executing the strategy bar-by-bar on historical data.
Tracks all trades, equity curve, and produces performance metrics.

Key features:
- Walk-forward: computes indicators from trailing window at each bar
- Respects session timing, confluence scoring, ATR-based SL/TP
- Simulates fills at the close of the signal bar (conservative)
- SL/TP checked on subsequent bars using high/low
- One trade at a time (no overlapping positions)
- Computes Sharpe, win rate, max drawdown, profit factor, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold_trading_one_trade_per_day.gold_strategy import (
    SetupType,
    TradeSetup,
    analyze_gold,
    compute_position_size,
)


@dataclass
class Trade:
    """Record of a single completed trade."""
    trade_id: int
    entry_bar: int
    exit_bar: int
    entry_time: datetime
    exit_time: datetime
    side: str
    setup_type: str
    confidence: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    qty: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "tp_hit", "sl_hit", "timeout", "session_end"
    holding_bars: int
    risk_reward_target: float
    risk_reward_actual: float


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_equity: float = 100.0
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    min_confidence: float = 0.4
    min_risk_reward: float = 1.5
    lookback_bars: int = 60  # bars for indicator calculation
    max_holding_bars: int = 50  # force exit after N bars
    commission_per_trade: float = 0.0  # per-share commission
    slippage_pct: float = 0.0001  # 0.01% slippage
    allow_multiple_trades_per_day: bool = False
    max_trades_per_day: int = 1


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[float]
    equity_timestamps: List[datetime]

    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollar: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    consecutive_wins_max: int = 0
    consecutive_losses_max: int = 0
    final_equity: float = 0.0
    return_pct: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "GOLD STRATEGY BACKTEST RESULTS",
            "=" * 60,
            f"Initial Equity:        ${self.config.initial_equity:.2f}",
            f"Final Equity:          ${self.final_equity:.2f}",
            f"Total Return:          {self.return_pct:.2f}%",
            f"Total P&L:             ${self.total_pnl:.2f}",
            "",
            f"Total Trades:          {self.total_trades}",
            f"Winning Trades:        {self.winning_trades}",
            f"Losing Trades:         {self.losing_trades}",
            f"Win Rate:              {self.win_rate:.1f}%",
            "",
            f"Avg Win:               ${self.avg_win:.2f}",
            f"Avg Loss:              ${self.avg_loss:.2f}",
            f"Profit Factor:         {self.profit_factor:.2f}",
            f"Best Trade:            ${self.best_trade_pnl:.2f}",
            f"Worst Trade:           ${self.worst_trade_pnl:.2f}",
            "",
            f"Max Drawdown:          {self.max_drawdown_pct:.2f}%",
            f"Max Drawdown ($):      ${self.max_drawdown_dollar:.2f}",
            f"Sharpe Ratio:          {self.sharpe_ratio:.2f}",
            "",
            f"Avg Holding Period:    {self.avg_holding_bars:.1f} bars",
            f"Max Consec. Wins:      {self.consecutive_wins_max}",
            f"Max Consec. Losses:    {self.consecutive_losses_max}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _compute_metrics(result: BacktestResult) -> BacktestResult:
    """Compute summary metrics from trades and equity curve."""
    trades = result.trades
    result.total_trades = len(trades)

    if not trades:
        result.final_equity = result.config.initial_equity
        return result

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

    result.total_pnl = sum(t.pnl for t in trades)
    result.avg_win = (sum(t.pnl for t in wins) / len(wins)) if wins else 0.0
    result.avg_loss = (sum(t.pnl for t in losses) / len(losses)) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    result.best_trade_pnl = max(t.pnl for t in trades) if trades else 0.0
    result.worst_trade_pnl = min(t.pnl for t in trades) if trades else 0.0
    result.avg_holding_bars = (sum(t.holding_bars for t in trades) / len(trades)) if trades else 0.0

    # Max consecutive wins/losses
    max_cw, max_cl, cw, cl = 0, 0, 0, 0
    for t in trades:
        if t.pnl > 0:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)
    result.consecutive_wins_max = max_cw
    result.consecutive_losses_max = max_cl

    # Equity curve metrics
    eq = result.equity_curve
    result.final_equity = eq[-1] if eq else result.config.initial_equity
    result.total_pnl_pct = ((result.final_equity - result.config.initial_equity) / result.config.initial_equity * 100)
    result.return_pct = result.total_pnl_pct

    # Max drawdown
    peak = eq[0]
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        if dd > max_dd_dollar:
            max_dd_dollar = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    result.max_drawdown_dollar = max_dd_dollar
    result.max_drawdown_pct = max_dd_pct

    # Sharpe ratio (annualized, assuming ~288 bars/day, ~252 trading days)
    if len(eq) > 1:
        returns = []
        for i in range(1, len(eq)):
            r = (eq[i] - eq[i - 1]) / eq[i - 1] if eq[i - 1] > 0 else 0.0
            returns.append(r)
        returns_arr = np.array(returns)
        mean_r = np.mean(returns_arr)
        std_r = np.std(returns_arr)
        if std_r > 0:
            bars_per_year = 288 * 252
            result.sharpe_ratio = round(mean_r / std_r * np.sqrt(bars_per_year), 2)
        else:
            result.sharpe_ratio = 0.0
    else:
        result.sharpe_ratio = 0.0

    return result


def run_backtest(
    bars: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run the gold strategy backtester on historical/synthetic bars.

    Args:
        bars: DataFrame with timestamp, open, high, low, close, volume columns.
        config: Backtest configuration. Uses defaults if None.

    Returns:
        BacktestResult with trades, equity curve, and metrics.
    """
    if config is None:
        config = BacktestConfig()

    equity = config.initial_equity
    equity_curve = [equity]
    equity_timestamps = []
    trades: List[Trade] = []
    trade_counter = 0

    # Position tracking
    in_position = False
    position_side: Optional[str] = None
    position_entry_price = 0.0
    position_sl = 0.0
    position_tp = 0.0
    position_qty = 0.0
    position_entry_bar = 0
    position_entry_time: Optional[datetime] = None
    position_setup_type = ""
    position_confidence = 0.0
    position_rr_target = 0.0
    daily_trade_count: Dict[str, int] = {}

    total_bars = len(bars)
    lookback = config.lookback_bars

    for i in range(lookback, total_bars):
        current_bar = bars.iloc[i]
        current_time = current_bar.get("timestamp", datetime(2025, 1, 1))

        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)

        if i == lookback and equity_timestamps == []:
            equity_timestamps.append(current_time)

        # --- Check open position exit ---
        if in_position:
            bar_high = float(current_bar["high"])
            bar_low = float(current_bar["low"])
            bar_close = float(current_bar["close"])
            holding_bars = i - position_entry_bar

            exit_price = None
            exit_reason = None

            if position_side == "BUY":
                # Check SL first (worst case)
                if bar_low <= position_sl:
                    exit_price = position_sl
                    exit_reason = "sl_hit"
                # Check TP
                elif bar_high >= position_tp:
                    exit_price = position_tp
                    exit_reason = "tp_hit"
            else:  # SELL
                # Check SL first
                if bar_high >= position_sl:
                    exit_price = position_sl
                    exit_reason = "sl_hit"
                # Check TP
                elif bar_low <= position_tp:
                    exit_price = position_tp
                    exit_reason = "tp_hit"

            # Timeout exit
            if exit_price is None and holding_bars >= config.max_holding_bars:
                exit_price = bar_close
                exit_reason = "timeout"

            if exit_price is not None:
                # Apply slippage on exit
                if position_side == "BUY":
                    exit_price *= (1.0 - config.slippage_pct)
                else:
                    exit_price *= (1.0 + config.slippage_pct)

                # Calculate P&L
                if position_side == "BUY":
                    pnl_per_share = exit_price - position_entry_price
                else:
                    pnl_per_share = position_entry_price - exit_price

                pnl = pnl_per_share * position_qty
                pnl -= config.commission_per_trade * 2  # entry + exit commission

                pnl_pct = (pnl / equity * 100) if equity > 0 else 0.0
                risk_per_share = abs(position_entry_price - position_sl)
                actual_rr = (pnl_per_share / risk_per_share) if risk_per_share > 0 else 0.0

                trade = Trade(
                    trade_id=trade_counter,
                    entry_bar=position_entry_bar,
                    exit_bar=i,
                    entry_time=position_entry_time,
                    exit_time=current_time,
                    side=position_side,
                    setup_type=position_setup_type,
                    confidence=position_confidence,
                    entry_price=round(position_entry_price, 2),
                    exit_price=round(exit_price, 2),
                    stop_loss=round(position_sl, 2),
                    take_profit=round(position_tp, 2),
                    qty=position_qty,
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2),
                    exit_reason=exit_reason,
                    holding_bars=holding_bars,
                    risk_reward_target=round(position_rr_target, 2),
                    risk_reward_actual=round(actual_rr, 2),
                )
                trades.append(trade)
                trade_counter += 1

                equity += pnl
                in_position = False
                position_side = None

        # --- Look for new entry ---
        if not in_position:
            # Daily trade limit check
            day_key = str(current_time.date()) if hasattr(current_time, 'date') else str(current_time)[:10]
            today_trades = daily_trade_count.get(day_key, 0)
            if not config.allow_multiple_trades_per_day and today_trades >= config.max_trades_per_day:
                equity_curve.append(equity)
                equity_timestamps.append(current_time)
                continue

            # Get trailing window for indicators
            window = bars.iloc[max(0, i - lookback):i + 1].copy()
            if len(window) < 30:
                equity_curve.append(equity)
                equity_timestamps.append(current_time)
                continue

            last_price = float(current_bar["close"])

            try:
                setup = analyze_gold(
                    bars=window,
                    last_price=last_price,
                    timestamp=current_time,
                )
            except Exception:
                equity_curve.append(equity)
                equity_timestamps.append(current_time)
                continue

            if (
                setup.setup_type != SetupType.NO_SETUP
                and setup.confidence >= config.min_confidence
                and setup.risk_reward_ratio >= config.min_risk_reward
            ):
                # Apply slippage on entry
                entry_price = last_price
                if setup.side == "BUY":
                    entry_price *= (1.0 + config.slippage_pct)
                else:
                    entry_price *= (1.0 - config.slippage_pct)

                # Position sizing
                risk_per_share = abs(entry_price - setup.stop_loss)
                if risk_per_share <= 0:
                    equity_curve.append(equity)
                    equity_timestamps.append(current_time)
                    continue

                max_risk = equity * config.risk_per_trade_pct
                qty = max_risk / risk_per_share

                # For gold, fractional shares are common
                if qty < 0.001:
                    equity_curve.append(equity)
                    equity_timestamps.append(current_time)
                    continue

                # Enter position
                in_position = True
                position_side = setup.side
                position_entry_price = entry_price
                position_sl = setup.stop_loss
                position_tp = setup.take_profit
                position_qty = round(qty, 4)
                position_entry_bar = i
                position_entry_time = current_time
                position_setup_type = setup.setup_type.value
                position_confidence = setup.confidence
                position_rr_target = setup.risk_reward_ratio
                daily_trade_count[day_key] = today_trades + 1

        equity_curve.append(equity)
        equity_timestamps.append(current_time)

    # Close any open position at the end
    if in_position and total_bars > 0:
        final_bar = bars.iloc[-1]
        exit_price = float(final_bar["close"])
        current_time = final_bar.get("timestamp", datetime(2025, 1, 1))

        if position_side == "BUY":
            pnl_per_share = exit_price - position_entry_price
        else:
            pnl_per_share = position_entry_price - exit_price

        pnl = pnl_per_share * position_qty
        risk_per_share = abs(position_entry_price - position_sl)
        actual_rr = (pnl_per_share / risk_per_share) if risk_per_share > 0 else 0.0

        trade = Trade(
            trade_id=trade_counter,
            entry_bar=position_entry_bar,
            exit_bar=total_bars - 1,
            entry_time=position_entry_time,
            exit_time=current_time,
            side=position_side,
            setup_type=position_setup_type,
            confidence=position_confidence,
            entry_price=round(position_entry_price, 2),
            exit_price=round(exit_price, 2),
            stop_loss=round(position_sl, 2),
            take_profit=round(position_tp, 2),
            qty=position_qty,
            pnl=round(pnl, 2),
            pnl_pct=round((pnl / equity * 100) if equity > 0 else 0.0, 2),
            exit_reason="end_of_data",
            holding_bars=total_bars - 1 - position_entry_bar,
            risk_reward_target=round(position_rr_target, 2),
            risk_reward_actual=round(actual_rr, 2),
        )
        trades.append(trade)
        equity += pnl

    result = BacktestResult(
        config=config,
        trades=trades,
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
    )
    return _compute_metrics(result)
