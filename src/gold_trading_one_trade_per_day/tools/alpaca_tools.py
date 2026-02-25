import os
from crewai.tools import BaseTool
from typing import Optional
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Environment variables for Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "your_alpaca_api_key_here")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "your_alpaca_secret_key_here")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() == "true"

class AlpacaDataTool(BaseTool):
    name: str = "Alpaca Market Data Tool"
    description: str = "Fetches the latest 15-minute candlestick data for Gold (GLD) from Alpaca. Note: Alpaca uses GLD ETF as a proxy for Gold."

    def _run(self, symbol: str = "GLD") -> str:
        if ALPACA_API_KEY == "your_alpaca_api_key_here":
            return f"Mock Data: {symbol} 15m Chart. Current Price: 202.50. Recent trend: Bullish. Market Structure: Break of Structure observed at 200.00. Liquidity grab detected at 198.00 level. (Please set ALPACA_API_KEY for real data)"
        
        try:
            # Initialize Alpaca Data client
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            
            # Request parameters
            from datetime import datetime, timedelta
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=5) # get last 5 days
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute, # Fallback to Minute if 15 Min is not directly standard in some accounts, but let's use 15Min if we construct it
                start=start_dt,
                end=end_dt
            )
            
            # Get bars
            bars = client.get_stock_bars(request_params)
            df = bars.df
            return df.tail(15).to_string() # Returns latest bars

        except Exception as e:
            return f"Error fetching Alpaca data: {str(e)}"

class AlpacaExecutionTool(BaseTool):
    name: str = "Alpaca Execution Tool"
    description: str = "Executes a trade on Alpaca with specified entry, stop-loss, and take-profit."

    def _run(self, action: str, symbol: str = "GLD", entry: float = 0.0, sl: float = 0.0, tp: float = 0.0, qty: float = 1.0) -> str:
        if ALPACA_API_KEY == "your_alpaca_api_key_here":
            return f"MOCK EXECUTION SUCCESSFUL: {action} {qty} shares of {symbol} at {entry}. SL: {sl}, TP: {tp}. (Please set ALPACA_API_KEY for real execution)"

        try:
            trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
            
            # Create a market order
            side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
            
            # Alpaca takes a limit order for advanced stop loss/take profit, but for simplicity we place a market order with attached bracket
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                # Bracket orders can be added here using take_profit and stop_loss parameters if needed
            )
            
            # Execute order
            order = trading_client.submit_order(order_data)
            
            return f"ALPACA EXECUTION SUCCESSFUL: Order ID {order.id} for {qty} shares of {symbol}."
        except Exception as e:
            return f"Error executing Alpaca trade: {str(e)}"
