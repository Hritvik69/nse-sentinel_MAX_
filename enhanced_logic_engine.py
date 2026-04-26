"""
enhanced_logic_engine.py
─────────────────────────
Phase 3 intelligence layer for NSE Sentinel.

Adds FOUR classification columns to any scan DataFrame that has already
passed through enhance_results().

    "Volume Trend"   –  STRONG / BUILDING / NORMAL / WEAK
    "Setup Quality"  –  HIGH / MEDIUM / LOW
    "Entry Timing"   –  EARLY / GOOD / LATE
    "Trap Risk"      –  HIGH / MEDIUM / LOW

Design principles
─────────────────
• Zero API calls — purely in-memory DataFrame logic.
• NOT strict — adds intelligence without tightening filters.
• Never filters / removes rows — DO NOT drop any row.
• Never modifies existing columns.
• Never crashes — every path returns df unchanged on any error.
• Works after any mode (1-6) without mode-specific branching.

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

        vol_trends:    list[str] = []
        setup_quals:   list[str] = []
        entry_timings: list[str] = []
        trap_risks:    list[str] = []

        for idx in out.index:
            row = out.loc[idx]

            rsi         = _get(row, "RSI",            default=50.0)
            vol_avg     = _get(row, "Vol / Avg",       default=1.0)
            delta_ema20 = _get(row, "Δ vs EMA20 (%)", default=0.0)
            ret_5d      = _get(row, "5D Return (%)",   default=0.0)

            # ── 1. Volume Trend ────────────────────────────────────────
            vt = _volume_trend(vol_avg)
            vol_trends.append(vt)

            # ── 2. Entry Timing ────────────────────────────────────────
            et = _entry_timing(rsi, delta_ema20)
            entry_timings.append(et)

            # ── 3. Setup Quality (uses volume trend computed above) ────
            sq = _setup_quality(vt, rsi, delta_ema20)
            setup_quals.append(sq)

            # ── 4. Trap Risk (now uses 2-condition threshold) ──────────
            tr = _trap_risk(rsi, vol_avg, delta_ema20, ret_5d)
            trap_risks.append(tr)

        out["Volume Trend"]  = vol_trends
        out["Setup Quality"] = setup_quals
        out["Entry Timing"]  = entry_timings
        out["Trap Risk"]     = trap_risks

        # Preserve existing sort order (Final Score desc if present)
        if "Final Score" in out.columns:
            out = out.sort_values("Final Score", ascending=False).reset_index(drop=True)

        return out

    except Exception:
        # Absolute fail-safe — never crash the app
        return df