from __future__ import annotations

import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from gold_trading_one_trade_per_day.schemas import DataSource, DailyState, FeatureSnapshot, Regime

NY_TZ = ZoneInfo("America/New_York")


def is_rth(ts: datetime) -> bool:
    """Return True if the timestamp is within XAU_USD trading hours (24/5, Mon–Fri).

    XAU_USD is closed from Friday 17:00 ET to Sunday 17:00 ET.
    """
    local = ts.astimezone(NY_TZ)
    weekday = local.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    t = local.time()
    close_time = time(17, 0)
    if weekday == 5:  # Saturday — always closed
        return False
    if weekday == 6:  # Sunday — open after 17:00 ET
        return t >= close_time
    if weekday == 4:  # Friday — closed after 17:00 ET
        return t < close_time
    return True  # Monday–Thursday always open


def in_open_warmup(ts: datetime, warmup_minutes: int) -> bool:
    """Warmup window after the weekly Sunday market open (17:00 ET)."""
    local = ts.astimezone(NY_TZ)
    if local.weekday() != 6:
        return False
    start = local.replace(hour=17, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=warmup_minutes)
    return start <= local < end


def _calc_regime(
    spread: float,
    bar_range_pct: float,
    volume_spike_ratio: float,
    vwap_displacement_pct: float,
) -> Regime:
    if spread > 0.50:
        return Regime.LOW_LIQUIDITY
    if bar_range_pct >= 0.006:
        return Regime.HIGH_VOL
    if volume_spike_ratio >= 2.0 and vwap_displacement_pct >= 0.002:
        return Regime.TREND
    if vwap_displacement_pct <= 0.0008:
        return Regime.RANGE
    return Regime.NEUTRAL


def build_feature_snapshot(
    bars: pd.DataFrame,
    bid: float,
    ask: float,
    macro_proxies: dict[str, float] | None = None,
    timestamp: datetime | None = None,
    symbol: str = "XAU_USD",
    data_source: DataSource = DataSource.REST_FALLBACK,
    data_age_sec: float = 0.0,
) -> FeatureSnapshot:
    if bars.empty:
        raise ValueError("bars dataframe cannot be empty")

    if not {"open", "high", "low", "close", "volume"}.issubset(set(bars.columns)):
        raise ValueError("bars must include open/high/low/close/volume")

    now = timestamp or datetime.now(tz=NY_TZ)
    bars = bars.copy()

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    rolling_vwap = (typical_price * bars["volume"]).cumsum() / bars["volume"].cumsum()

    last = bars.iloc[-1]
    recent = bars.iloc[-20:] if len(bars) >= 20 else bars

    avg_volume = max(float(recent["volume"].mean()), 1.0)
    last_close = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_volume = float(last["volume"])
    last_vwap = float(rolling_vwap.iloc[-1])

    spread = max(ask - bid, 0.0)
    spread_bps = (spread / last_close) * 10_000 if last_close else 0.0

    bar_range_pct = (last_high - last_low) / last_close if last_close else 0.0
    median_range_pct = (
        ((recent["high"] - recent["low"]) / recent["close"]).median()
        if len(recent) > 0
        else 0.0001
    )
    median_range_pct = max(float(median_range_pct), 0.0001)
    bar_range_expansion_ratio = bar_range_pct / median_range_pct

    volume_spike_ratio = last_volume / avg_volume
    vwap_displacement_pct = abs(last_close - last_vwap) / last_vwap if last_vwap else 0.0

    risk_on = 0.0
    risk_off = 0.0
    macro = macro_proxies or {}
    risk_on += max(macro.get("SPY", 0.0), 0.0)
    risk_off += max(macro.get("VXX", 0.0), 0.0)
    risk_off += max(macro.get("UUP", 0.0), 0.0)
    risk_off += max(macro.get("TLT", 0.0), 0.0)
    greed_score = 50 + (risk_on - risk_off) * 50
    greed_score = min(100.0, max(0.0, greed_score))

    regime = _calc_regime(spread, bar_range_pct, volume_spike_ratio, vwap_displacement_pct)

    return FeatureSnapshot(
        symbol=symbol,
        timestamp=now,
        data_source=data_source,
        data_age_sec=float(max(data_age_sec, 0.0)),
        last_price=last_close,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_bps=spread_bps,
        volume=last_volume,
        avg_volume=avg_volume,
        volume_spike_ratio=volume_spike_ratio,
        vwap=last_vwap,
        vwap_displacement_pct=vwap_displacement_pct,
        bar_range_pct=bar_range_pct,
        rolling_median_range_pct=median_range_pct,
        bar_range_expansion_ratio=bar_range_expansion_ratio,
        macro_proxies=macro,
        greed_score=greed_score,
        regime=regime,
        is_rth=is_rth(now),
        data_fresh=True,
    )


def should_wake_ai(
    snapshot: FeatureSnapshot,
    day_state: DailyState,
    min_volume_spike_ratio: float = 3.0,
    min_vwap_displacement_pct: float = 0.0015,
    min_bar_range_expansion: float = 1.2,
    max_spread: float = 0.50,
    open_warmup_minutes: int | None = None,
    macro_event_active: bool = False,
    macro_event_label: str | None = None,
    trading_mode: str = "daily_scalp",
) -> tuple[bool, str, dict]:
    warmup = open_warmup_minutes
    if warmup is None:
        warmup = int(os.getenv("OPEN_WARMUP_MINUTES", "5"))

    context = {
        "spread": snapshot.spread,
        "max_spread_threshold": max_spread,
        "volume_spike_ratio": snapshot.volume_spike_ratio,
        "vwap_displacement_pct": snapshot.vwap_displacement_pct,
        "bar_range_expansion_ratio": snapshot.bar_range_expansion_ratio,
        "data_source": snapshot.data_source.value,
        "data_age_sec": snapshot.data_age_sec,
        "macro_event_label": macro_event_label,
        "trading_mode": trading_mode,
    }

    if day_state.hard_lock:
        return False, "hard_lock", context
    if not snapshot.is_rth:
        return False, "outside_rth", context
    # In spike mode, macro events are the reason to trade — do not block.
    # In daily_scalp mode, macro events mean high volatility without context — block.
    if trading_mode != "spike" and macro_event_active:
        return False, "macro_event_window", context
    if in_open_warmup(snapshot.timestamp, warmup):
        return False, "open_warmup", context
    if not snapshot.data_fresh:
        return False, "stale_data", context
    if snapshot.spread > max_spread:
        return False, "spread_too_wide", context
    if snapshot.volume_spike_ratio < min_volume_spike_ratio:
        return False, "volume_not_expanded", context
    if snapshot.vwap_displacement_pct < min_vwap_displacement_pct:
        return False, "vwap_displacement_low", context
    if snapshot.bar_range_expansion_ratio < min_bar_range_expansion:
        return False, "range_not_expanded", context
    return True, "triggered", context

