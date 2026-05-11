"""
enhanced_logic_engine.py
─────────────────────────
Phase 3 intelligence layer for NSE Sentinel.

Adds classification columns to any scan DataFrame that has already
passed through enhance_results().

    "Volume Trend"   –  STRONG / BUILDING / NORMAL / WEAK
    "Setup Quality"  –  HIGH / MEDIUM / LOW
    "Entry Timing"   –  EARLY / GOOD / LATE
    "Trap Risk"      –  HIGH / MEDIUM / LOW
    "Breakout Quality" / "Support Strength" / "Resistance Distance" /
    "Structure Quality" plus Mode 7 momentum quality helpers

Design principles
─────────────────
• Zero API calls — purely in-memory DataFrame logic.
• NOT strict — adds intelligence without tightening filters.
• Never filters / removes rows — DO NOT drop any row.
• Never modifies existing columns.
• Never crashes — every path returns df unchanged on any error.
• Works after any mode without mode-specific branching.

Trap Risk clarification
───────────────────────
HIGH   requires TWO or more independent risk conditions to fire.
       A single overbought or low-volume reading is NOT enough.
MEDIUM one risk condition present — a caution flag, not a blocker.
LOW    no significant risk conditions.

Public entry point
──────────────────
    from enhanced_logic_engine import apply_enhanced_logic
    df = apply_enhanced_logic(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _sf(v: object, default: float = 0.0) -> float:
    """safe float — never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _get(row: "pd.Series", *keys: str, default: float = 0.0) -> float:
    """Return first matching key from row as a safe float."""
    for k in keys:
        v = row.get(k)
        if v is not None:
            return _sf(v, default)
    return default


def _numeric_series(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    """Vectorized safe-float extraction for scan columns."""
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    col = df[name]
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").fillna(default).astype(float)
    cleaned = (
        col.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "-": np.nan, "—": np.nan})
        .str.replace(r"[%xX×,]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(default).astype(float)


def _text_series(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[name].where(df[name].notna(), default).astype(str).str.strip().str.upper()


# ─────────────────────────────────────────────────────────────────────
# CLASSIFICATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def _volume_trend(vol_avg: float) -> str:
    """
    Classify volume relative to its 20-day average.

    vol_avg = Vol / Avg  (already computed in scan row)

    > 1.5  → STRONG    (explosive / institutional interest)
    1.2–1.5 → BUILDING  (accumulation building up)
    0.9–1.2 → NORMAL    (no signal either way)
    < 0.9  → WEAK      (distribution / disinterest)
    """
    if vol_avg > 1.5:
        return "STRONG"
    if vol_avg >= 1.2:
        return "BUILDING"
    if vol_avg >= 0.9:
        return "NORMAL"
    return "WEAK"


def _entry_timing(rsi: float, delta_ema20: float) -> str:
    """
    Classify entry timing based on RSI momentum and EMA20 distance.

    EARLY  → RSI 50–60  AND  Δ EMA20 < 4%   (best risk/reward)
    GOOD   → RSI 55–65  (sweet spot — may overlap EARLY)
    LATE   → RSI > 70   OR  Δ EMA20 > 6%    (overextended)

    Priority: LATE > EARLY > GOOD
    (LATE is always flagged regardless of other conditions)
    """
    is_late  = rsi > 70 or delta_ema20 > 6.0
    is_early = 50.0 <= rsi <= 60.0 and delta_ema20 < 4.0
    is_good  = 55.0 <= rsi <= 65.0

    if is_late:
        return "LATE"
    if is_early:
        return "EARLY"
    if is_good:
        return "GOOD"
    return "NEUTRAL"  # fallback — no strong signal in either direction


def _setup_quality(
    vol_trend: str,
    rsi: float,
    delta_ema20: float,
) -> str:
    """
    Classify setup quality by combining volume, RSI zone and EMA extension.

    HIGH   → (STRONG or BUILDING volume) AND RSI 50–65 AND Δ EMA20 < 5%
    LOW    → Weak volume  OR  overextended (Δ EMA20 > 7% or RSI > 70)
    MEDIUM → everything else (mixed signals)
    """
    vol_ok   = vol_trend in ("STRONG", "BUILDING")
    vol_norm = vol_trend == "NORMAL"
    rsi_ok   = 50.0 <= rsi <= 65.0
    ema_ok   = delta_ema20 < 5.0

    overext  = delta_ema20 > 7.0 or rsi > 70.0
    vol_weak = vol_trend == "WEAK"

    if vol_ok and rsi_ok and ema_ok:
        return "HIGH"
    # Normal volume + healthy RSI + not overextended = MEDIUM (was incorrectly LOW before)
    if vol_norm and rsi_ok and ema_ok:
        return "MEDIUM"
    if overext or vol_weak:
        return "LOW"
    return "MEDIUM"


def _trap_risk(
    rsi: float,
    vol_avg: float,
    delta_ema20: float,
    ret_5d: float,
) -> str:
    """
    Classify bull-trap risk.

    Conditions (each is an independent risk flag):
        C1: RSI > 72  AND  Vol/Avg < 1.2   (overbought on thin volume)
        C2: Δ EMA20 > 7%                   (price too far from mean)
        C3: 5D Return > 9%                 (already pumped)

    HIGH   → TWO or more conditions fire  (was: any one — too strict)
    MEDIUM → exactly ONE condition fires  (new tier — caution, not a blocker)
    LOW    → no conditions fire

    Requiring two conditions prevents a single overbought reading from
    blocking an otherwise healthy setup.
    """
    c1 = rsi > 72.0 and vol_avg < 1.2
    c2 = delta_ema20 > 7.0
    c3 = ret_5d > 9.0

    count = sum([c1, c2, c3])

    if count >= 2:
        return "HIGH"
    if count == 1:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def apply_enhanced_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four intelligence columns to the scan DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output from enhance_results() / apply_universal_grading().
        Required source columns (always present after a scan):
            "RSI"            – current RSI(14)
            "Vol / Avg"      – volume ratio vs 20-day average
            "Δ vs EMA20 (%)" – price distance from EMA20
            "5D Return (%)"  – 5-day price return

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four new columns:
            "Volume Trend"  : str  STRONG / BUILDING / NORMAL / WEAK
            "Setup Quality" : str  HIGH / MEDIUM / LOW
            "Entry Timing"  : str  EARLY / GOOD / LATE / NEUTRAL
            "Trap Risk"     : str  HIGH / MEDIUM / LOW
        Sorted by "Final Score" descending (if that column exists),
        otherwise order is unchanged.
        Zero rows removed.
    """
    # ── Guard ──────────────────────────────────────────────────────────
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
    except Exception:
        return df

    try:
        out = df.copy()

        rsi = _numeric_series(out, "RSI", 50.0)
        vol_avg = _numeric_series(out, "Vol / Avg", 1.0)
        delta_ema20 = _numeric_series(out, "Δ vs EMA20 (%)", 0.0)
        ret_5d = _numeric_series(out, "5D Return (%)", 0.0)

        vol_trend = np.select(
            [vol_avg > 1.5, vol_avg >= 1.2, vol_avg >= 0.9],
            ["STRONG", "BUILDING", "NORMAL"],
            default="WEAK",
        )

        is_late = (rsi > 70.0) | (delta_ema20 > 6.0)
        is_early = (rsi >= 50.0) & (rsi <= 60.0) & (delta_ema20 < 4.0)
        is_good = (rsi >= 55.0) & (rsi <= 65.0)
        entry_timing = np.select(
            [is_late, is_early, is_good],
            ["LATE", "EARLY", "GOOD"],
            default="NEUTRAL",
        )

        vol_ok = (vol_trend == "STRONG") | (vol_trend == "BUILDING")
        vol_norm = vol_trend == "NORMAL"
        rsi_ok = (rsi >= 50.0) & (rsi <= 65.0)
        ema_ok = delta_ema20 < 5.0
        overext = (delta_ema20 > 7.0) | (rsi > 70.0)
        vol_weak = vol_trend == "WEAK"
        setup_quality = np.select(
            [vol_ok & rsi_ok & ema_ok, vol_norm & rsi_ok & ema_ok, overext | vol_weak],
            ["HIGH", "MEDIUM", "LOW"],
            default="MEDIUM",
        )

        trap_count = (
            ((rsi > 72.0) & (vol_avg < 1.2)).astype(int)
            + (delta_ema20 > 7.0).astype(int)
            + (ret_5d > 9.0).astype(int)
        )
        trap_risk = np.select(
            [trap_count >= 2, trap_count == 1],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )

        out["Volume Trend"] = vol_trend
        out["Setup Quality"] = setup_quality
        out["Entry Timing"] = entry_timing
        out["Trap Risk"] = trap_risk

        high_dist = _numeric_series(out, "Δ vs 20D High (%)", -5.0)
        price = _numeric_series(out, "Price (₹)", 0.0)
        ema20 = _numeric_series(out, "EMA 20", 0.0)
        ema50 = _numeric_series(out, "EMA 50", 0.0)
        ret_20d = _numeric_series(out, "20D Return (%)", 0.0)
        support_touches = _numeric_series(out, "Support Touches", 0.0)
        resistance_touches = _numeric_series(out, "Resistance Touches", 0.0)
        resistance_rejections = _numeric_series(out, "Resistance Rejections", 0.0)
        base_tightness = _numeric_series(out, "Base Tightness (%)", 99.0)
        sr_score = _numeric_series(out, "S&R Structure Score", 50.0)
        pivot_support = _text_series(out, "Pivot Support Quality", "LOW")
        pivot_resistance = _text_series(out, "Pivot Resistance Quality", "LOW")
        atr_contraction = _text_series(out, "ATR Contraction", "NO").eq("YES")
        breakout_retest = _text_series(out, "Breakout Retest", "NO").eq("YES")
        liquidity_sweep = _text_series(out, "Liquidity Sweep", "NO").eq("YES")
        wick_rejection = _text_series(out, "Wick Rejection", "MEDIUM")
        channel_detected = _text_series(out, "Ascending Channel", "NO").eq("YES")
        channel_entry = _text_series(out, "Channel Entry Zone", "NO").eq("YES")
        channel_score = _numeric_series(out, "Channel Score", 0.0)

        ema_stack = (price > ema20) & (ema20 > ema50) & (ema50 > 0)
        ideal_resistance = (high_dist >= -2.0) & (high_dist <= 1.5)
        medium_resistance = (
            ((high_dist >= -5.0) & (high_dist < -2.0))
            | ((high_dist > 1.5) & (high_dist <= 3.0))
        )
        real_support = (support_touches >= 2) | pivot_support.isin(["HIGH", "MEDIUM"]) | breakout_retest | liquidity_sweep | channel_entry
        real_resistance = (resistance_touches >= 2) | pivot_resistance.isin(["HIGH", "MEDIUM"]) | channel_detected
        tight_base = base_tightness <= 8.0

        resistance_distance = np.select(
            [channel_entry & channel_score.ge(60.0), ideal_resistance & (real_resistance | tight_base), ideal_resistance | medium_resistance],
            ["HIGH", "HIGH", "MEDIUM"],
            default="LOW",
        )

        volume_confirmation = np.select(
            [(vol_avg >= 1.4) & (vol_avg <= 2.8), (vol_avg >= 1.1) & (vol_avg <= 4.0)],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )

        support_strength = np.select(
            [channel_entry & channel_score.ge(60.0),
             ema_stack & (delta_ema20 >= -1.0) & (delta_ema20 <= 3.0) & real_support,
             (price > ema20) & (delta_ema20.abs() <= 5.0) & (ema20 > 0) & (real_support | (support_touches >= 1))],
            ["HIGH", "HIGH", "MEDIUM"],
            default="LOW",
        )

        breakout_quality = np.select(
            [ideal_resistance & (vol_avg >= 1.3) & (vol_avg <= 3.2) & (rsi >= 52.0) & (rsi <= 70.0) & (delta_ema20 <= 7.0) & (real_resistance | tight_base | atr_contraction),
             (ideal_resistance | medium_resistance) & (vol_avg >= 1.1) & (rsi >= 50.0) & (rsi <= 72.0) & (delta_ema20 <= 8.0)],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )

        momentum_continuation = np.select(
            [(ret_5d >= 2.0) & (ret_5d <= 9.0) & (ret_20d >= 5.0) & (ret_20d <= 18.0) & (rsi >= 52.0) & (rsi <= 70.0) & (delta_ema20 <= 7.0),
             (ret_5d > 0.0) & (ret_20d > 0.0) & (rsi >= 48.0) & (rsi <= 74.0) & (delta_ema20 <= 9.0)],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )

        mode7_trap_count = (
            (rsi > 76.0).astype(int)
            + (vol_avg < 1.0).astype(int)
            + (delta_ema20 > 8.0).astype(int)
            + (ret_5d > 12.0).astype(int)
            + ((high_dist > 1.5) & (vol_avg < 1.2)).astype(int)
            + ((resistance_rejections >= 2) & (vol_avg < 1.25)).astype(int)
            + (wick_rejection.eq("LOW") & (high_dist > -1.0)).astype(int)
        )
        trap_probability = np.select(
            [mode7_trap_count >= 3, mode7_trap_count >= 1],
            ["HIGH", "MEDIUM"],
            default="LOW",
        )

        structure_quality = np.select(
            [
                channel_entry
                & channel_score.ge(70.0)
                & (trap_probability == "LOW"),
                ema_stack
                & (breakout_quality != "LOW")
                & (support_strength != "LOW")
                & (volume_confirmation != "LOW")
                & (momentum_continuation != "LOW")
                & (trap_probability == "LOW"),
                channel_detected
                & channel_score.ge(50.0)
                & (trap_probability != "HIGH"),
                (ema_stack & (trap_probability != "HIGH") & (delta_ema20 <= 8.0))
                | ((sr_score >= 62.0) & (trap_probability != "HIGH")),
            ],
            ["HIGH", "HIGH", "MEDIUM", "MEDIUM"],
            default="LOW",
        )

        out["Breakout Quality"] = breakout_quality
        out["Support Strength"] = support_strength
        out["Resistance Distance"] = resistance_distance
        out["Structure Quality"] = structure_quality
        out["Volume Confirmation"] = volume_confirmation
        out["Trap Probability"] = trap_probability
        out["Momentum Continuation"] = momentum_continuation

        # Preserve existing sort order (Final Score desc if present)
        if "Final Score" in out.columns:
            out = out.sort_values("Final Score", ascending=False).reset_index(drop=True)

        return out

    except Exception:
        # Absolute fail-safe — never crash the app
        return df
