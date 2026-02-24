import os
from crewai.tools import BaseTool
from pydantic import Field
from typing import Optional, Type, Any
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts

# Environment variables for OANDA
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "your_api_key_here")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "your_account_id_here")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")  # 'practice' or 'live'

class OANDADataTool(BaseTool):
    name: str = "OANDA Market Data Tool"
    description: str = "Fetches the latest 15-minute candlestick data for XAU/USD from OANDA."

    def _run(self, symbol: str = "XAU_USD") -> str:
        if OANDA_API_KEY == "your_api_key_here":
            return "Mock Data: XAU/USD 15m Chart. Current Price: 2025.50. Recent trend: Bullish. Market Structure: Break of Structure observed at 2020.00. Liquidity grab detected at 2015.00 level. (Please set OANDA_API_KEY for real data)"
        
        try:
            client = API(access_token=OANDA_API_KEY, environment=OANDA_ENVIRONMENT)
            params = {
                "count": 50,
                "granularity": "M15"
            }
            r = instruments.InstrumentsCandles(instrument=symbol, params=params)
            client.request(r)
            
            candles = r.response.get('candles', [])
            data = []
            for candle in candles:
                data.append({
                    "time": candle['time'],
                    "open": float(candle['mid']['o']),
                    "high": float(candle['mid']['h']),
                    "low": float(candle['mid']['l']),
                    "close": float(candle['mid']['c']),
                    "volume": candle['volume']
                })
            
            df = pd.DataFrame(data)
            return df.tail(10).to_string()
        except Exception as e:
            return f"Error fetching OANDA data: {str(e)}"

class OANDAExecutionTool(BaseTool):
    name: str = "OANDA Execution Tool"
    description: str = "Executes a trade on OANDA with specified entry, stop-loss, and take-profit."

    def _run(self, action: str, symbol: str, entry: float, sl: float, tp: float, units: int = 100) -> str:
        if OANDA_API_KEY == "your_api_key_here":
            return f"MOCK EXECUTION SUCCESSFUL: {action} {units} units of {symbol} at {entry}. SL: {sl}, TP: {tp}. (Please set OANDA_API_KEY for real execution)"

        try:
            client = API(access_token=OANDA_API_KEY, environment=OANDA_ENVIRONMENT)
            
            # Simple market order for demonstration
            # In a real setup, you might use Limit orders if 'entry' is far from current price
            order_data = {
                "order": {
                    "units": str(units if action.upper() == "BUY" else -units),
                    "instrument": symbol,
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {"price": str(round(sl, 3))},
                    "takeProfitOnFill": {"price": str(round(tp, 3))}
                }
            }
            
            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=order_data)
            client.request(r)
            
            return f"OANDA EXECUTION SUCCESSFUL: Transaction ID {r.response.get('orderFillTransaction', {}).get('id', 'Unknown')}"
        except Exception as e:
            return f"Error executing OANDA trade: {str(e)}"
