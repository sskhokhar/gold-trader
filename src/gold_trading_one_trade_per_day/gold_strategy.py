"""Gold (XAUUSD / GLD) strategy engine.

Combines technical indicators, session timing, and confluence scoring
to produce high-probability trade setups with precise entry/SL/TP levels.

Design philosophy:
- Gold moves in *sessions* (Asian, London, NY).  The best scalp windows
  are London open, NY open, and the London/NY overlap.
- Gold respects VWAP, round numbers, and pivot levels more than most assets.
- ATR-based SL/TP adapts to current volatility rather than fixed-dollar stops.
- Confluence of 3+ indicators produces the highest win-rate setups.
- Position sizing uses a fractional-Kelly approach capped by ATR risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from math import floor
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from gold_trading_one_trade_per_day.indicators import (
    IndicatorSnapshot,
    atr,
    compute_indicator_snapshot,
)

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
LONDON_TZ = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------

class TradingSession(str, Enum):
    ASIAN = "asian"
    LONDON = "london"
    NY = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    OFF_HOURS = "off_hours"


@dataclass(slots=True)
class SessionInfo:
    session: TradingSession
    session_quality: float  # 0.0 - 1.0, how good this session is for gold
    minutes_into_session: int
    minutes_remaining: int


def detect_session(ts: datetime) -> SessionInfo:
    """Detect the current gold trading session.

    Gold sessions (UTC):
    - Asian:    00:00 - 08:00 UTC
    - London:   08:00 - 16:30 UTC
    - NY:       13:30 - 20:00 UTC
    - Overlap:  13:30 - 16:30 UTC (London + NY both open)
    """
    utc = ts.astimezone(UTC_TZ)
    h, m = utc.hour, utc.minute
    total_min = h * 60 + m

    # London/NY overlap: 13:30 - 16:30 UTC  (best for gold)
    if 13 * 60 + 30 <= total_min < 16 * 60 + 30:
        start = 13 * 60 + 30
        end = 16 * 60 + 30
        return SessionInfo(
            session=TradingSession.LONDON_NY_OVERLAP,
            session_quality=1.0,
            minutes_into_session=total_min - start,
            minutes_remaining=end - total_min,
        )
    # London session: 08:00 - 16:30 UTC
    if 8 * 60 <= total_min < 16 * 60 + 30:
        start = 8 * 60
        end = 16 * 60 + 30
        return SessionInfo(
            session=TradingSession.LONDON,
            session_quality=0.8,
            minutes_into_session=total_min - start,
            minutes_remaining=end - total_min,
        )
    # NY session (post overlap): 16:30 - 20:00 UTC
    if 16 * 60 + 30 <= total_min < 20 * 60:
        start = 16 * 60 + 30
        end = 20 * 60
        return SessionInfo(
            session=TradingSession.NY,
            session_quality=0.6,
            minutes_into_session=total_min - start,
            minutes_remaining=end - total_min,
        )
    # Asian session: 00:00 - 08:00 UTC
    if total_min < 8 * 60:
        start = 0
        end = 8 * 60
        return SessionInfo(
            session=TradingSession.ASIAN,
            session_quality=0.3,
            minutes_into_session=total_min - start,
            minutes_remaining=end - total_min,
        )
    # Off hours
    return SessionInfo(
        session=TradingSession.OFF_HOURS,
        session_quality=0.0,
        minutes_into_session=0,
        minutes_remaining=0,
    )


# ---------------------------------------------------------------------------
# Trade setup classification
# ---------------------------------------------------------------------------

class SetupType(str, Enum):
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    BREAKOUT = "breakout"
    VWAP_BOUNCE = "vwap_bounce"
    BOLLINGER_BOUNCE = "bollinger_bounce"
    PIVOT_BOUNCE = "pivot_bounce"
    NO_SETUP = "no_setup"


@dataclass(slots=True)
class TradeSetup:
    setup_type: SetupType
    side: str  # "BUY" or "SELL"
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    reasoning: list[str]
    indicators: IndicatorSnapshot
    session: SessionInfo

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        d = dataclasses.asdict(self)
        d["indicators"] = self.indicators.to_dict()
        return d


# ---------------------------------------------------------------------------
# SL/TP calculation using ATR
# ---------------------------------------------------------------------------

def compute_atr_stops(
    entry_price: float,
    side: str,
    atr_value: float,
    sl_multiplier: float = 1.5,
    tp_multiplier: float = 2.5,
    pivots: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """Compute stop-loss and take-profit using ATR.

    For gold intraday:
    - SL: 1.0 - 2.0 ATR (tighter in range, wider in trend)
    - TP: 2.0 - 3.0 ATR for 1.5:1 to 2:1 R:R minimum

    Returns (sl, tp, rr_ratio).
    """
    sl_distance = atr_value * sl_multiplier
    tp_distance = atr_value * tp_multiplier

    if side == "BUY":
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance

        # Optionally tighten TP to next resistance pivot
        if pivots:
            for level_key in ("r1", "r2", "r3"):
                level = pivots.get(level_key, 0)
                if level and entry_price < level < tp:
                    # Place TP slightly below resistance
                    tp = level - 0.01
                    break
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

        # Optionally tighten TP to next support pivot
        if pivots:
            for level_key in ("s1", "s2", "s3"):
                level = pivots.get(level_key, 0)
                if level and tp < level < entry_price:
                    tp = level + 0.01
                    break

    risk = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    rr = (reward / risk) if risk > 0 else 0.0

    return round(sl, 2), round(tp, 2), round(rr, 2)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    confidence: float = 0.5,
    kelly_fraction: float = 0.25,
    win_rate: float = 0.55,
    avg_rr: float = 1.5,
) -> tuple[float, float]:
    """Compute position size using fractional Kelly criterion.

    Kelly formula: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg win/avg loss ratio

    We use a fraction (default 25%) of full Kelly for safety.

    Returns (qty, risk_amount).
    """
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0 or equity <= 0:
        return 0.0, 0.0

    # Full Kelly fraction
    p = min(max(win_rate, 0.01), 0.99)
    q = 1.0 - p
    b = max(avg_rr, 0.01)
    full_kelly = max((p * b - q) / b, 0.0)

    # Apply fractional Kelly with confidence adjustment
    effective_kelly = full_kelly * kelly_fraction * min(max(confidence, 0.1), 1.0)

    # Cap at risk_pct of equity
    max_risk = equity * min(risk_pct, effective_kelly)
    qty = floor(max_risk / risk_per_share)

    return max(float(qty), 0.0), max_risk


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

def detect_setup(
    indicators: IndicatorSnapshot,
    session: SessionInfo,
    last_price: float,
) -> TradeSetup:
    """Analyze indicators and session to detect the best trade setup.

    Priority order:
    1. Trend continuation (strongest in trending markets)
    2. Breakout (Bollinger squeeze + volume)
    3. VWAP bounce (institutional level)
    4. Pivot bounce (classic gold levels)
    5. Trend reversal (counter-trend, needs more confirmation)
    """
    reasons: list[str] = []
    best_setup = SetupType.NO_SETUP
    best_side = "BUY"
    best_confidence = 0.0

    # Minimum session quality gate
    min_session_quality = float(os.getenv("MIN_SESSION_QUALITY", "0.3"))
    if session.session_quality < min_session_quality:
        return TradeSetup(
            setup_type=SetupType.NO_SETUP,
            side="BUY",
            confidence=0.0,
            entry_price=last_price,
            stop_loss=last_price,
            take_profit=last_price,
            risk_reward_ratio=0.0,
            atr_multiplier_sl=0.0,
            atr_multiplier_tp=0.0,
            reasoning=["Session quality too low for gold trading"],
            indicators=indicators,
            session=session,
        )

    # --- Setup 1: Trend continuation ---
    if indicators.trend_strength in ("strong", "moderate") and indicators.adx_value >= 25:
        if indicators.ema_trend == "bullish" and indicators.plus_di > indicators.minus_di:
            # Bullish trend continuation
            if indicators.rsi_14 < 65 and indicators.price_vs_vwap == "above":
                confidence = 0.3
                if indicators.macd_histogram > 0:
                    confidence += 0.15
                if indicators.stoch_zone != "overbought":
                    confidence += 0.1
                if session.session_quality >= 0.8:
                    confidence += 0.1
                confidence += indicators.adx_value / 200  # up to +0.25 for very strong trend
                confidence = min(confidence, 0.95)
                if confidence > best_confidence:
                    best_setup = SetupType.TREND_CONTINUATION
                    best_side = "BUY"
                    best_confidence = confidence
                    reasons = [
                        f"ADX={indicators.adx_value:.1f} confirms strong trend",
                        f"EMA alignment bullish (9>{indicators.ema_9:.2f} > 21>{indicators.ema_21:.2f} > 50>{indicators.ema_50:.2f})",
                        f"+DI ({indicators.plus_di:.1f}) > -DI ({indicators.minus_di:.1f})",
                        f"Price above VWAP ({indicators.vwap_value:.2f})",
                        f"RSI={indicators.rsi_14:.1f} not overbought",
                    ]

        elif indicators.ema_trend == "bearish" and indicators.minus_di > indicators.plus_di:
            if indicators.rsi_14 > 35 and indicators.price_vs_vwap == "below":
                confidence = 0.3
                if indicators.macd_histogram < 0:
                    confidence += 0.15
                if indicators.stoch_zone != "oversold":
                    confidence += 0.1
                if session.session_quality >= 0.8:
                    confidence += 0.1
                confidence += indicators.adx_value / 200
                confidence = min(confidence, 0.95)
                if confidence > best_confidence:
                    best_setup = SetupType.TREND_CONTINUATION
                    best_side = "SELL"
                    best_confidence = confidence
                    reasons = [
                        f"ADX={indicators.adx_value:.1f} confirms strong trend",
                        f"EMA alignment bearish",
                        f"-DI ({indicators.minus_di:.1f}) > +DI ({indicators.plus_di:.1f})",
                        f"Price below VWAP ({indicators.vwap_value:.2f})",
                        f"RSI={indicators.rsi_14:.1f} not oversold",
                    ]

    # --- Setup 2: Bollinger breakout ---
    if indicators.bb_squeeze and indicators.trend_strength in ("weak", "no_trend"):
        if indicators.bb_pct_b > 0.95 and indicators.confluence_score > 0.3:
            confidence = 0.5
            if indicators.macd_cross == "bullish":
                confidence += 0.15
            if session.session_quality >= 0.8:
                confidence += 0.1
            confidence = min(confidence, 0.85)
            if confidence > best_confidence:
                best_setup = SetupType.BREAKOUT
                best_side = "BUY"
                best_confidence = confidence
                reasons = [
                    "Bollinger squeeze detected (low volatility compression)",
                    f"Price breaking above upper band (%B={indicators.bb_pct_b:.2f})",
                    f"Confluence score positive ({indicators.confluence_score:.2f})",
                ]
        elif indicators.bb_pct_b < 0.05 and indicators.confluence_score < -0.3:
            confidence = 0.5
            if indicators.macd_cross == "bearish":
                confidence += 0.15
            if session.session_quality >= 0.8:
                confidence += 0.1
            confidence = min(confidence, 0.85)
            if confidence > best_confidence:
                best_setup = SetupType.BREAKOUT
                best_side = "SELL"
                best_confidence = confidence
                reasons = [
                    "Bollinger squeeze detected (low volatility compression)",
                    f"Price breaking below lower band (%B={indicators.bb_pct_b:.2f})",
                    f"Confluence score negative ({indicators.confluence_score:.2f})",
                ]

    # --- Setup 3: VWAP bounce ---
    vwap_proximity = abs(last_price - indicators.vwap_value) / indicators.atr_14 if indicators.atr_14 > 0 else 999
    if vwap_proximity < 0.5:  # Price within 0.5 ATR of VWAP
        if indicators.confluence_score > 0.2 and indicators.rsi_zone != "overbought":
            confidence = 0.4
            if indicators.stoch_zone == "oversold":
                confidence += 0.15
            if indicators.ema_trend == "bullish":
                confidence += 0.1
            if session.session_quality >= 0.8:
                confidence += 0.1
            confidence = min(confidence, 0.8)
            if confidence > best_confidence:
                best_setup = SetupType.VWAP_BOUNCE
                best_side = "BUY"
                best_confidence = confidence
                reasons = [
                    f"Price near VWAP (distance: {vwap_proximity:.2f} ATR)",
                    f"Confluence score bullish ({indicators.confluence_score:.2f})",
                    "VWAP acts as institutional support for gold",
                ]
        elif indicators.confluence_score < -0.2 and indicators.rsi_zone != "oversold":
            confidence = 0.4
            if indicators.stoch_zone == "overbought":
                confidence += 0.15
            if indicators.ema_trend == "bearish":
                confidence += 0.1
            if session.session_quality >= 0.8:
                confidence += 0.1
            confidence = min(confidence, 0.8)
            if confidence > best_confidence:
                best_setup = SetupType.VWAP_BOUNCE
                best_side = "SELL"
                best_confidence = confidence
                reasons = [
                    f"Price near VWAP (distance: {vwap_proximity:.2f} ATR)",
                    f"Confluence score bearish ({indicators.confluence_score:.2f})",
                    "VWAP acts as institutional resistance for gold",
                ]

    # --- Setup 4: Pivot bounce ---
    pivot_threshold = indicators.atr_14 * 0.3 if indicators.atr_14 > 0 else 0.1
    for label, level in [("s1", indicators.s1), ("s2", indicators.s2), ("s3", indicators.s3)]:
        if abs(last_price - level) < pivot_threshold and last_price > level:
            if indicators.confluence_score > 0 and indicators.rsi_14 < 60:
                confidence = 0.35
                if indicators.stoch_zone == "oversold":
                    confidence += 0.15
                if session.session_quality >= 0.6:
                    confidence += 0.1
                confidence = min(confidence, 0.75)
                if confidence > best_confidence:
                    best_setup = SetupType.PIVOT_BOUNCE
                    best_side = "BUY"
                    best_confidence = confidence
                    reasons = [
                        f"Price bouncing off Camarilla {label.upper()} ({level:.2f})",
                        f"Confluence score positive ({indicators.confluence_score:.2f})",
                        "Camarilla pivots are key gold intraday levels",
                    ]
                    break

    for label, level in [("r1", indicators.r1), ("r2", indicators.r2), ("r3", indicators.r3)]:
        if abs(last_price - level) < pivot_threshold and last_price < level:
            if indicators.confluence_score < 0 and indicators.rsi_14 > 40:
                confidence = 0.35
                if indicators.stoch_zone == "overbought":
                    confidence += 0.15
                if session.session_quality >= 0.6:
                    confidence += 0.1
                confidence = min(confidence, 0.75)
                if confidence > best_confidence:
                    best_setup = SetupType.PIVOT_BOUNCE
                    best_side = "SELL"
                    best_confidence = confidence
                    reasons = [
                        f"Price bouncing off Camarilla {label.upper()} ({level:.2f})",
                        f"Confluence score negative ({indicators.confluence_score:.2f})",
                        "Camarilla pivots are key gold intraday levels",
                    ]
                    break

    # --- Finalize ---
    if best_setup == SetupType.NO_SETUP:
        return TradeSetup(
            setup_type=SetupType.NO_SETUP,
            side="BUY",
            confidence=0.0,
            entry_price=last_price,
            stop_loss=last_price,
            take_profit=last_price,
            risk_reward_ratio=0.0,
            atr_multiplier_sl=0.0,
            atr_multiplier_tp=0.0,
            reasoning=["No high-probability setup detected"],
            indicators=indicators,
            session=session,
        )

    # Dynamic ATR multipliers based on setup type and session
    if best_setup == SetupType.TREND_CONTINUATION:
        sl_mult = 1.2
        tp_mult = 2.5
    elif best_setup == SetupType.BREAKOUT:
        sl_mult = 1.5
        tp_mult = 3.0  # wider targets for breakouts
    elif best_setup in (SetupType.VWAP_BOUNCE, SetupType.PIVOT_BOUNCE):
        sl_mult = 1.0
        tp_mult = 2.0  # tighter for mean-reversion
    else:
        sl_mult = 1.5
        tp_mult = 2.5

    # Adjust for session quality (tighter in better sessions = less noise)
    if session.session_quality >= 0.8:
        sl_mult *= 0.9  # tighter stop in high-quality sessions
        tp_mult *= 1.1  # wider target

    pivots_dict = {
        "r1": indicators.r1, "r2": indicators.r2, "r3": indicators.r3,
        "s1": indicators.s1, "s2": indicators.s2, "s3": indicators.s3,
    }

    sl, tp, rr = compute_atr_stops(
        entry_price=last_price,
        side=best_side,
        atr_value=indicators.atr_14,
        sl_multiplier=sl_mult,
        tp_multiplier=tp_mult,
        pivots=pivots_dict,
    )

    # Minimum R:R filter
    min_rr = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))
    if rr < min_rr:
        return TradeSetup(
            setup_type=SetupType.NO_SETUP,
            side=best_side,
            confidence=0.0,
            entry_price=last_price,
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=rr,
            atr_multiplier_sl=sl_mult,
            atr_multiplier_tp=tp_mult,
            reasoning=[f"Risk:reward ratio {rr:.2f} below minimum {min_rr}"],
            indicators=indicators,
            session=session,
        )

    return TradeSetup(
        setup_type=best_setup,
        side=best_side,
        confidence=round(best_confidence, 3),
        entry_price=round(last_price, 2),
        stop_loss=sl,
        take_profit=tp,
        risk_reward_ratio=rr,
        atr_multiplier_sl=round(sl_mult, 2),
        atr_multiplier_tp=round(tp_mult, 2),
        reasoning=reasons,
        indicators=indicators,
        session=session,
    )


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_gold(
    bars: pd.DataFrame,
    last_price: float,
    timestamp: datetime | None = None,
) -> TradeSetup:
    """Run the full gold analysis pipeline.

    1. Compute all technical indicators
    2. Detect current trading session
    3. Find the best setup with confluence scoring
    4. Compute ATR-based SL/TP
    5. Return a complete TradeSetup

    Args:
        bars: DataFrame with open/high/low/close/volume columns (50+ bars ideal)
        last_price: Current market price
        timestamp: Current timestamp (defaults to now)
    """
    ts = timestamp or datetime.now(tz=NY_TZ)
    indicators = compute_indicator_snapshot(bars)
    session = detect_session(ts)
    return detect_setup(indicators, session, last_price)
