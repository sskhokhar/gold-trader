"""Technical analysis indicators optimized for XAUUSD (Gold) intraday trading.

All functions accept a pandas DataFrame with at minimum 'close' column,
and optionally 'high', 'low', 'volume', 'open' columns.  Returns are always
pandas Series or scalar floats so they compose cleanly with the existing
FeatureSnapshot pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA.  Gold key periods: 9, 21, 50, 200."""
    return series.ewm(span=period, adjust=False).mean()


def ema_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Returns +1 on golden cross bar, -1 on death cross bar, else 0."""
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    cross = pd.Series(0, index=fast.index, dtype=int)
    cross[(prev_fast <= prev_slow) & (fast > slow)] = 1
    cross[(prev_fast >= prev_slow) & (fast < slow)] = -1
    return cross


# ---------------------------------------------------------------------------
# Relative Strength Index (RSI)
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.  Gold sweet-spot: 14 for trend, 7 for scalping."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------------

@dataclass
class MACDResult:
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """Standard MACD.  Gold-optimized: 12/26/9 for swing, 5/13/4 for scalp."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return MACDResult(
        macd_line=macd_line,
        signal_line=signal_line,
        histogram=histogram,
    )


# ---------------------------------------------------------------------------
# Average True Range (ATR)
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Single-bar true range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.  Gold standard: 14 for daily, 10 for intraday."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

@dataclass
class BollingerBands:
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    width: pd.Series
    pct_b: pd.Series


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> BollingerBands:
    """Bollinger Bands.  Gold: 20/2.0 standard, 20/1.5 for tighter scalps."""
    middle = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    width = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return BollingerBands(
        upper=upper,
        middle=middle,
        lower=lower,
        width=width,
        pct_b=pct_b,
    )


# ---------------------------------------------------------------------------
# VWAP (Volume Weighted Average Price)
# ---------------------------------------------------------------------------

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Cumulative VWAP from start of the bar series."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def vwap_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    std_multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with standard deviation bands.  Returns (vwap, upper, lower)."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap_line = cum_tp_vol / cum_vol.replace(0, np.nan)

    cum_tp2_vol = (typical ** 2 * volume).cumsum()
    variance = (cum_tp2_vol / cum_vol.replace(0, np.nan)) - vwap_line ** 2
    std = np.sqrt(variance.clip(lower=0))

    return vwap_line, vwap_line + std_multiplier * std, vwap_line - std_multiplier * std


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

@dataclass
class StochasticResult:
    k: pd.Series
    d: pd.Series


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> StochasticResult:
    """Full Stochastic.  Gold: 14/3/3 for trend, 5/3/3 for scalp."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k_line = raw_k.rolling(window=smooth_k).mean()
    d_line = k_line.rolling(window=d_period).mean()
    return StochasticResult(k=k_line, d=d_line)


# ---------------------------------------------------------------------------
# Average Directional Index (ADX)
# ---------------------------------------------------------------------------

@dataclass
class ADXResult:
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> ADXResult:
    """ADX with +DI / -DI.  Gold: 14 standard.  ADX > 25 = trending."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    tr = true_range(high, low, close)

    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return ADXResult(adx=adx_line, plus_di=plus_di, minus_di=minus_di)


# ---------------------------------------------------------------------------
# Support / Resistance via Pivot Points (Camarilla - popular for gold)
# ---------------------------------------------------------------------------

@dataclass
class PivotLevels:
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def camarilla_pivots(
    high: float,
    low: float,
    close: float,
) -> PivotLevels:
    """Camarilla pivot points.  Widely used by gold intraday traders."""
    diff = high - low
    return PivotLevels(
        pivot=(high + low + close) / 3,
        r1=close + diff * 1.1 / 12,
        r2=close + diff * 1.1 / 6,
        r3=close + diff * 1.1 / 4,
        s1=close - diff * 1.1 / 12,
        s2=close - diff * 1.1 / 6,
        s3=close - diff * 1.1 / 4,
    )


# ---------------------------------------------------------------------------
# Consolidated indicator snapshot for the strategy engine
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSnapshot:
    """All indicators computed at once for a given bar series."""
    # EMAs
    ema_9: float
    ema_21: float
    ema_50: float
    ema_trend: str  # "bullish", "bearish", "neutral"

    # RSI
    rsi_14: float
    rsi_7: float
    rsi_zone: str  # "overbought", "oversold", "neutral"

    # MACD
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_cross: str  # "bullish", "bearish", "none"

    # ATR
    atr_14: float
    atr_pct: float  # ATR as percentage of price

    # Bollinger
    bb_upper: float
    bb_lower: float
    bb_width: float
    bb_pct_b: float
    bb_squeeze: bool

    # VWAP
    vwap_value: float
    price_vs_vwap: str  # "above", "below", "at"

    # Stochastic
    stoch_k: float
    stoch_d: float
    stoch_zone: str  # "overbought", "oversold", "neutral"

    # ADX
    adx_value: float
    plus_di: float
    minus_di: float
    trend_strength: str  # "strong", "moderate", "weak", "no_trend"

    # Pivots
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    # Derived
    confluence_score: float  # -1.0 (max bearish) to +1.0 (max bullish)
    signal_count_bull: int
    signal_count_bear: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON embedding in LLM prompts."""
        import dataclasses
        return dataclasses.asdict(self)


def compute_indicator_snapshot(bars: pd.DataFrame) -> IndicatorSnapshot:
    """Compute all indicators from a bars DataFrame.

    Expects columns: open, high, low, close, volume.
    Minimum 50 bars recommended for reliable output.
    """
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    last_close = float(close.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])

    # --- EMAs ---
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    ema9_val = float(ema9.iloc[-1])
    ema21_val = float(ema21.iloc[-1])
    ema50_val = float(ema50.iloc[-1])

    if ema9_val > ema21_val > ema50_val:
        ema_trend = "bullish"
    elif ema9_val < ema21_val < ema50_val:
        ema_trend = "bearish"
    else:
        ema_trend = "neutral"

    # --- RSI ---
    rsi14 = rsi(close, 14)
    rsi7 = rsi(close, 7)
    rsi14_val = float(rsi14.iloc[-1]) if not rsi14.isna().iloc[-1] else 50.0
    rsi7_val = float(rsi7.iloc[-1]) if not rsi7.isna().iloc[-1] else 50.0

    if rsi14_val >= 70:
        rsi_zone = "overbought"
    elif rsi14_val <= 30:
        rsi_zone = "oversold"
    else:
        rsi_zone = "neutral"

    # --- MACD ---
    macd_res = macd(close, 12, 26, 9)
    macd_line_val = float(macd_res.macd_line.iloc[-1]) if not macd_res.macd_line.isna().iloc[-1] else 0.0
    macd_signal_val = float(macd_res.signal_line.iloc[-1]) if not macd_res.signal_line.isna().iloc[-1] else 0.0
    macd_hist_val = float(macd_res.histogram.iloc[-1]) if not macd_res.histogram.isna().iloc[-1] else 0.0

    prev_hist = float(macd_res.histogram.iloc[-2]) if len(macd_res.histogram) >= 2 and not macd_res.histogram.isna().iloc[-2] else 0.0
    if prev_hist <= 0 < macd_hist_val:
        macd_cross_str = "bullish"
    elif prev_hist >= 0 > macd_hist_val:
        macd_cross_str = "bearish"
    else:
        macd_cross_str = "none"

    # --- ATR ---
    atr14 = atr(high, low, close, 14)
    atr14_val = float(atr14.iloc[-1]) if not atr14.isna().iloc[-1] else 0.0
    atr_pct_val = (atr14_val / last_close * 100) if last_close > 0 else 0.0

    # --- Bollinger ---
    bb = bollinger_bands(close, 20, 2.0)
    bb_upper_val = float(bb.upper.iloc[-1]) if not bb.upper.isna().iloc[-1] else last_close
    bb_lower_val = float(bb.lower.iloc[-1]) if not bb.lower.isna().iloc[-1] else last_close
    bb_width_val = float(bb.width.iloc[-1]) if not bb.width.isna().iloc[-1] else 0.0
    bb_pct_b_val = float(bb.pct_b.iloc[-1]) if not bb.pct_b.isna().iloc[-1] else 0.5
    # Squeeze: width below 20-period average of width
    avg_bb_width = float(bb.width.rolling(20).mean().iloc[-1]) if len(bb.width) >= 20 and not bb.width.rolling(20).mean().isna().iloc[-1] else bb_width_val
    bb_squeeze = bb_width_val < avg_bb_width * 0.8

    # --- VWAP ---
    vwap_val = float(vwap(high, low, close, volume).iloc[-1])
    if last_close > vwap_val * 1.001:
        price_vs_vwap = "above"
    elif last_close < vwap_val * 0.999:
        price_vs_vwap = "below"
    else:
        price_vs_vwap = "at"

    # --- Stochastic ---
    stoch = stochastic(high, low, close, 14, 3, 3)
    stoch_k_val = float(stoch.k.iloc[-1]) if not stoch.k.isna().iloc[-1] else 50.0
    stoch_d_val = float(stoch.d.iloc[-1]) if not stoch.d.isna().iloc[-1] else 50.0

    if stoch_k_val >= 80:
        stoch_zone = "overbought"
    elif stoch_k_val <= 20:
        stoch_zone = "oversold"
    else:
        stoch_zone = "neutral"

    # --- ADX ---
    adx_res = adx(high, low, close, 14)
    adx_val = float(adx_res.adx.iloc[-1]) if not adx_res.adx.isna().iloc[-1] else 0.0
    plus_di_val = float(adx_res.plus_di.iloc[-1]) if not adx_res.plus_di.isna().iloc[-1] else 0.0
    minus_di_val = float(adx_res.minus_di.iloc[-1]) if not adx_res.minus_di.isna().iloc[-1] else 0.0

    if adx_val >= 40:
        trend_strength = "strong"
    elif adx_val >= 25:
        trend_strength = "moderate"
    elif adx_val >= 15:
        trend_strength = "weak"
    else:
        trend_strength = "no_trend"

    # --- Pivots ---
    # Use the previous bar's high/low/close for pivots
    if len(bars) >= 2:
        prev_bar = bars.iloc[-2]
        pivots = camarilla_pivots(
            float(prev_bar["high"]),
            float(prev_bar["low"]),
            float(prev_bar["close"]),
        )
    else:
        pivots = camarilla_pivots(last_high, last_low, last_close)

    # --- Confluence scoring ---
    bull_signals = 0
    bear_signals = 0

    # EMA alignment
    if ema_trend == "bullish":
        bull_signals += 2
    elif ema_trend == "bearish":
        bear_signals += 2

    # RSI
    if rsi14_val < 30:
        bull_signals += 1  # oversold = potential bounce
    elif rsi14_val > 70:
        bear_signals += 1  # overbought = potential drop

    # MACD cross
    if macd_cross_str == "bullish":
        bull_signals += 2
    elif macd_cross_str == "bearish":
        bear_signals += 2

    # MACD histogram direction
    if macd_hist_val > 0 and macd_hist_val > prev_hist:
        bull_signals += 1
    elif macd_hist_val < 0 and macd_hist_val < prev_hist:
        bear_signals += 1

    # VWAP
    if price_vs_vwap == "above":
        bull_signals += 1
    elif price_vs_vwap == "below":
        bear_signals += 1

    # Stochastic
    if stoch_zone == "oversold" and stoch_k_val > stoch_d_val:
        bull_signals += 1
    elif stoch_zone == "overbought" and stoch_k_val < stoch_d_val:
        bear_signals += 1

    # ADX + DI
    if adx_val >= 25:
        if plus_di_val > minus_di_val:
            bull_signals += 1
        elif minus_di_val > plus_di_val:
            bear_signals += 1

    # Bollinger
    if bb_pct_b_val < 0.05:
        bull_signals += 1  # touching lower band = potential bounce
    elif bb_pct_b_val > 0.95:
        bear_signals += 1  # touching upper band = potential drop

    total = bull_signals + bear_signals
    if total == 0:
        confluence = 0.0
    else:
        confluence = round((bull_signals - bear_signals) / total, 4)

    return IndicatorSnapshot(
        ema_9=round(ema9_val, 4),
        ema_21=round(ema21_val, 4),
        ema_50=round(ema50_val, 4),
        ema_trend=ema_trend,
        rsi_14=round(rsi14_val, 2),
        rsi_7=round(rsi7_val, 2),
        rsi_zone=rsi_zone,
        macd_line=round(macd_line_val, 6),
        macd_signal=round(macd_signal_val, 6),
        macd_histogram=round(macd_hist_val, 6),
        macd_cross=macd_cross_str,
        atr_14=round(atr14_val, 4),
        atr_pct=round(atr_pct_val, 4),
        bb_upper=round(bb_upper_val, 4),
        bb_lower=round(bb_lower_val, 4),
        bb_width=round(bb_width_val, 6),
        bb_pct_b=round(bb_pct_b_val, 4),
        bb_squeeze=bb_squeeze,
        vwap_value=round(vwap_val, 4),
        price_vs_vwap=price_vs_vwap,
        stoch_k=round(stoch_k_val, 2),
        stoch_d=round(stoch_d_val, 2),
        stoch_zone=stoch_zone,
        adx_value=round(adx_val, 2),
        plus_di=round(plus_di_val, 2),
        minus_di=round(minus_di_val, 2),
        trend_strength=trend_strength,
        pivot=round(pivots.pivot, 4),
        r1=round(pivots.r1, 4),
        r2=round(pivots.r2, 4),
        r3=round(pivots.r3, 4),
        s1=round(pivots.s1, 4),
        s2=round(pivots.s2, 4),
        s3=round(pivots.s3, 4),
        confluence_score=confluence,
        signal_count_bull=bull_signals,
        signal_count_bear=bear_signals,
    )
