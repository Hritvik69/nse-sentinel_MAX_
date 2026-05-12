"""
Mode 7 support/resistance structure analysis.

Uses only the already-loaded OHLCV dataframe.  No API calls are made here.
The analysis is deliberately lightweight so it can run during scans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_engines.constants import (
    MODE7_BASE_TIGHT_HIGH,
    MODE7_BASE_TIGHT_MEDIUM,
    debug_log,
)

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _quality(high: bool, medium: bool = False) -> str:
    if high:
        return "HIGH"
    if medium:
        return "MEDIUM"
    return "LOW"


def analyze_mode7_structure(df: pd.DataFrame | None) -> dict[str, object]:
    """
    Return compact price-action features for Mode 7.

    Output is safe to merge directly into a scan row.  On any bad input or
    unexpected error, it returns neutral defaults.
    """
    neutral = {
        "Pivot Support": np.nan,
        "Pivot Resistance": np.nan,
        "Support Touches": 0,
        "Resistance Touches": 0,
        "Resistance Rejections": 0,
        "Higher Lows": 0,
        "Base Tightness (%)": np.nan,
        "ATR Contraction": "NO",
        "Liquidity Sweep": "NO",
        "Breakout Retest": "NO",
        "Wick Rejection": "LOW",
        "Pivot Support Quality": "LOW",
        "Pivot Resistance Quality": "LOW",
        "S&R Structure Score": 50.0,
    }
    try:
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 30:
            return neutral

        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = work.columns.get_level_values(0)
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(set(work.columns)):
            return neutral
        work = work.dropna(subset=["Open", "High", "Low", "Close"]).tail(80)
        if len(work) < 30:
            return neutral

        close = pd.to_numeric(work["Close"], errors="coerce")
        high = pd.to_numeric(work["High"], errors="coerce")
        low = pd.to_numeric(work["Low"], errors="coerce")
        open_ = pd.to_numeric(work["Open"], errors="coerce")
        volume = pd.to_numeric(work.get("Volume", pd.Series(index=work.index, dtype=float)), errors="coerce")

        last_close = _safe_float(close.iloc[-1], 0.0)
        if last_close <= 0:
            return neutral

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=7).mean()
        atr_pct = _safe_float((atr14.iloc[-1] / last_close) * 100.0, 1.8)
        tolerance_pct = float(np.clip(max(1.0, atr_pct * 0.85), 1.0, 3.0))
        tolerance = last_close * tolerance_pct / 100.0

        pivot_high = (high.shift(1) < high) & (high.shift(-1) < high)
        pivot_low = (low.shift(1) > low) & (low.shift(-1) > low)
        recent_highs = high[pivot_high].tail(10)
        recent_lows = low[pivot_low].tail(10)
        higher_lows = 0
        if len(recent_lows) >= 2:
            lows_for_sequence = recent_lows.tail(6).astype(float).to_numpy()
            higher_lows = int(np.sum(np.diff(lows_for_sequence) > 0))

        below_supports = recent_lows[recent_lows <= last_close * 1.03]
        above_resistances = recent_highs[recent_highs >= last_close * 0.97]
        support = _safe_float(below_supports.iloc[-1] if len(below_supports) else low.tail(20).min(), np.nan)
        resistance = _safe_float(above_resistances.iloc[-1] if len(above_resistances) else high.tail(20).max(), np.nan)

        support_touches = 0
        resistance_touches = 0
        rejection_count = 0
        lower_wick_good = False
        upper_wick_bad = False
        liquidity_sweep = False
        breakout_retest = False

        if np.isfinite(support) and support > 0:
            support_touches = int((low.tail(60).sub(support).abs() <= tolerance).sum())
            body_low = pd.concat([open_, close], axis=1).min(axis=1)
            lower_wick = (body_low - low).clip(lower=0)
            lower_wick_good = bool(((low.tail(5).sub(support).abs() <= tolerance) & (lower_wick.tail(5) > tr.tail(5) * 0.35)).any())
            liquidity_sweep = bool((low.iloc[-1] < support - tolerance * 0.4) and (close.iloc[-1] > support))

        if np.isfinite(resistance) and resistance > 0:
            resistance_touches = int((high.tail(60).sub(resistance).abs() <= tolerance).sum())
            body_high = pd.concat([open_, close], axis=1).max(axis=1)
            upper_wick = (high - body_high).clip(lower=0)
            near_res = high.tail(20).sub(resistance).abs() <= tolerance
            rejection_count = int((near_res & (upper_wick.tail(20) > tr.tail(20) * 0.35)).sum())
            upper_wick_bad = bool((abs(high.iloc[-1] - resistance) <= tolerance) and (upper_wick.iloc[-1] > tr.iloc[-1] * 0.42))
            breakout_retest = bool(
                close.iloc[-1] > resistance
                and (low.tail(8).sub(resistance).abs() <= tolerance).any()
                and close.tail(3).min() >= resistance - tolerance
            )

        base_high = _safe_float(high.tail(12).max(), last_close)
        base_low = _safe_float(low.tail(12).min(), last_close)
        base_tightness = ((base_high - base_low) / last_close) * 100.0 if last_close > 0 else np.nan
        atr_base = atr14.tail(8).mean()
        atr_prior = atr14.tail(40).head(24).mean()
        atr_contraction = bool(pd.notna(atr_base) and pd.notna(atr_prior) and atr_prior > 0 and atr_base <= atr_prior * 0.88)

        avg_vol_20 = volume.tail(20).mean()
        vol_building = bool(pd.notna(avg_vol_20) and avg_vol_20 > 0 and volume.tail(5).mean() >= avg_vol_20 * 0.95)

        support_quality = _quality(
            support_touches >= 2 and (lower_wick_good or liquidity_sweep or breakout_retest),
            support_touches >= 2 or lower_wick_good,
        )
        resistance_quality = _quality(
            resistance_touches >= 2 and base_tightness <= MODE7_BASE_TIGHT_MEDIUM and not upper_wick_bad,
            resistance_touches >= 1 and base_tightness <= 11.0,
        )
        wick_quality = "HIGH" if lower_wick_good and not upper_wick_bad else ("LOW" if upper_wick_bad else "MEDIUM")

        score = 50.0
        score += min(support_touches, 3) * 5.0
        score += min(resistance_touches, 3) * 4.0
        score += 8.0 if base_tightness <= MODE7_BASE_TIGHT_HIGH else (3.0 if base_tightness <= 10.0 else -5.0)
        score += 6.0 if atr_contraction else 0.0
        score += 5.0 if lower_wick_good else 0.0
        score += 6.0 if breakout_retest else 0.0
        score += 4.0 if liquidity_sweep else 0.0
        score += 3.0 if vol_building else 0.0
        score -= 8.0 if upper_wick_bad else 0.0

        return {
            "Pivot Support": round(support, 2) if np.isfinite(support) else np.nan,
            "Pivot Resistance": round(resistance, 2) if np.isfinite(resistance) else np.nan,
            "Support Touches": int(support_touches),
            "Resistance Touches": int(resistance_touches),
            "Resistance Rejections": int(rejection_count),
            "Higher Lows": int(higher_lows),
            "Base Tightness (%)": round(float(base_tightness), 2) if np.isfinite(base_tightness) else np.nan,
            "ATR Contraction": "YES" if atr_contraction else "NO",
            "Liquidity Sweep": "YES" if liquidity_sweep else "NO",
            "Breakout Retest": "YES" if breakout_retest else "NO",
            "Wick Rejection": wick_quality,
            "Pivot Support Quality": support_quality,
            "Pivot Resistance Quality": resistance_quality,
            "S&R Structure Score": round(float(np.clip(score, 0.0, 100.0)), 1),
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        debug_log("Mode 7 structure fallback activated: %s", exc, exc_info=True)
        return neutral


__all__ = ["analyze_mode7_structure"]
