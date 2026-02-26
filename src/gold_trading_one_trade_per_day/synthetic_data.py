"""Synthetic XAUUSD (gold) price data generator for backtesting.

Generates realistic intraday gold data with:
- Geometric Brownian Motion (GBM) for price evolution
- Session-aware volatility (Asian quiet, London/NY volatile)
- OHLCV bars at configurable intervals (1min, 5min, 15min)
- Trend injection: trending days, range-bound days, volatile event days
- Volume profile tied to session liquidity
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UTC_TZ = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# Session volatility multipliers (relative to base vol)
# ---------------------------------------------------------------------------
SESSION_VOL_MULT = {
    "asian":    0.6,   # 00:00-08:00 UTC  - quiet
    "london":   1.2,   # 08:00-13:30 UTC  - active
    "overlap":  1.5,   # 13:30-16:30 UTC  - most volatile
    "new_york": 1.0,   # 16:30-20:00 UTC  - moderate
    "off":      0.3,   # 20:00-00:00 UTC  - dead
}

SESSION_VOLUME_BASE = {
    "asian":    5000,
    "london":   15000,
    "overlap":  25000,
    "new_york": 12000,
    "off":      2000,
}


def _get_session(hour: int, minute: int) -> str:
    """Classify UTC hour:minute into gold trading session."""
    t = hour * 60 + minute
    if t < 8 * 60:
        return "asian"
    if t < 13 * 60 + 30:
        return "london"
    if t < 16 * 60 + 30:
        return "overlap"
    if t < 20 * 60:
        return "new_york"
    return "off"


def generate_gold_bars(
    start_price: float = 2650.0,
    num_bars: int = 500,
    bar_interval_minutes: int = 5,
    start_time: Optional[datetime] = None,
    base_volatility: float = 0.0003,
    trend_drift: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic gold OHLCV bars.

    Args:
        start_price: Starting price (typical XAUUSD ~2600-2800 in 2024-2025)
        num_bars: Number of bars to generate
        bar_interval_minutes: Bar interval (1, 5, 15 minutes)
        start_time: Start timestamp (UTC). Defaults to a Monday 00:00 UTC.
        base_volatility: Per-bar volatility (sigma for GBM). 0.0003 ~ realistic for 5min gold.
        trend_drift: Positive = uptrend, negative = downtrend, 0 = random.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    if start_time is None:
        # Default to a Monday 00:00 UTC
        start_time = datetime(2025, 1, 6, 0, 0, tzinfo=UTC_TZ)

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    price = start_price

    for i in range(num_bars):
        ts = start_time + timedelta(minutes=i * bar_interval_minutes)
        session = _get_session(ts.hour, ts.minute)

        vol_mult = SESSION_VOL_MULT.get(session, 1.0)
        effective_vol = base_volatility * vol_mult

        # GBM step with drift
        returns = trend_drift + effective_vol * rng.randn()
        close_price = price * np.exp(returns)

        # Generate intra-bar OHLC from the close
        # Use sub-steps to simulate intra-bar movement
        n_sub = max(bar_interval_minutes, 4)
        sub_prices = [price]
        sub_vol = effective_vol / np.sqrt(n_sub)
        p = price
        for _ in range(n_sub):
            p *= np.exp(trend_drift / n_sub + sub_vol * rng.randn())
            sub_prices.append(p)
        # Force last sub-price to match our close
        sub_prices[-1] = close_price

        bar_open = sub_prices[0]
        bar_high = max(sub_prices)
        bar_low = min(sub_prices)
        bar_close = close_price

        # Ensure high >= max(open, close) and low <= min(open, close)
        bar_high = max(bar_high, bar_open, bar_close)
        bar_low = min(bar_low, bar_open, bar_close)

        # Volume based on session
        base_vol = SESSION_VOLUME_BASE.get(session, 5000)
        volume = int(base_vol * (0.7 + 0.6 * rng.rand()))

        timestamps.append(ts)
        opens.append(round(bar_open, 2))
        highs.append(round(bar_high, 2))
        lows.append(round(bar_low, 2))
        closes.append(round(bar_close, 2))
        volumes.append(volume)

        price = close_price

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def generate_trending_day(
    direction: str = "up",
    start_price: float = 2650.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a single trending day (288 bars x 5min = 24 hours).

    A trending day in gold typically moves $20-40 in one direction.
    """
    drift = 0.00003 if direction == "up" else -0.00003
    return generate_gold_bars(
        start_price=start_price,
        num_bars=288,
        bar_interval_minutes=5,
        base_volatility=0.00035,
        trend_drift=drift,
        seed=seed,
    )


def generate_range_day(
    start_price: float = 2650.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a range-bound day with mean reversion."""
    return generate_gold_bars(
        start_price=start_price,
        num_bars=288,
        bar_interval_minutes=5,
        base_volatility=0.00020,
        trend_drift=0.0,
        seed=seed,
    )


def generate_volatile_event_day(
    start_price: float = 2650.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a high-volatility event day (e.g., NFP, FOMC)."""
    return generate_gold_bars(
        start_price=start_price,
        num_bars=288,
        bar_interval_minutes=5,
        base_volatility=0.0006,
        trend_drift=0.00001,
        seed=seed,
    )


def generate_multi_day_data(
    num_days: int = 30,
    start_price: float = 2650.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate multiple days of data with mixed conditions.

    Roughly: 40% trending, 40% range, 20% volatile events.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_frames = []
    price = start_price
    base_ts = datetime(2025, 1, 6, 0, 0, tzinfo=UTC_TZ)

    for day_idx in range(num_days):
        day_seed = (seed + day_idx) if seed is not None else None
        roll = random.random()
        day_start = base_ts + timedelta(days=day_idx)

        if roll < 0.2:
            # Trending up
            df = generate_gold_bars(
                start_price=price,
                num_bars=288,
                bar_interval_minutes=5,
                start_time=day_start,
                base_volatility=0.00035,
                trend_drift=0.00003,
                seed=day_seed,
            )
        elif roll < 0.4:
            # Trending down
            df = generate_gold_bars(
                start_price=price,
                num_bars=288,
                bar_interval_minutes=5,
                start_time=day_start,
                base_volatility=0.00035,
                trend_drift=-0.00003,
                seed=day_seed,
            )
        elif roll < 0.8:
            # Range
            df = generate_gold_bars(
                start_price=price,
                num_bars=288,
                bar_interval_minutes=5,
                start_time=day_start,
                base_volatility=0.00020,
                trend_drift=0.0,
                seed=day_seed,
            )
        else:
            # Volatile event
            df = generate_gold_bars(
                start_price=price,
                num_bars=288,
                bar_interval_minutes=5,
                start_time=day_start,
                base_volatility=0.0006,
                trend_drift=0.00001,
                seed=day_seed,
            )

        all_frames.append(df)
        price = float(df["close"].iloc[-1])

    result = pd.concat(all_frames, ignore_index=True)
    return result
