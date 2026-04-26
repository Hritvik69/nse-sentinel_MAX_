"""
phase4_logic_engine.py
───────────────────────
Phase 4.1 intelligence layer for NSE Sentinel.

Adds FOUR classification columns to any scan DataFrame that has already
passed through:
    enhance_results()
    apply_enhanced_logic()
    apply_universal_grading()

New columns added:
    "Setup Type"    –  Breakout / Pullback / Reversal /
                       Momentum Continuation / Weak Setup
    "Reason"        –  Human-readable confirmation string
    "Risk Score"    –  0–100 float (higher = riskier)
    "Final Signal"  –  STRONG BUY / BUY / WATCH / AVOID / TRAP

Design rules
─────────────
• Zero API calls — purely in-memory DataFrame logic.
• Never filters / removes rows.
• Never modifies or renames existing columns.
• Never crashes — full try/except wrapping at every level.
• Works for ALL scan modes (1-6) and CSV mode.
• All column access is safe (no KeyError possible).
• Market bias bearish adjusts score (via grading); Final Signal uses
  the score. No additional hard downgrade here.

Signal softening notes
──────────────────────
• Trap Risk MEDIUM (1 condition) does NOT trigger TRAP label.
  Only Trap Risk HIGH (2+ conditions) triggers TRAP.
• Advanced trap "WEAK VOLUME" is informational — no signal downgrade.
• Advanced trap "FAKE BREAKOUT" or "EXHAUSTION" → downgrade one level.
• Risk Score threshold raised from 75 to 80 for downgrade trigger.

Public entry point
──────────────────
    from phase4_logic_engine import apply_phase4_logic

    df = apply_phase4_logic(df, market_bias_dict)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# SAFE COLUMN ACCESS  (mandatory pattern — no KeyErrors possible)
# ─────────────────────────────────────────────────────────────────────

def get_safe(
    row: "pd.Series",
    keys: list[str],
    default: float,
) -> float:
    """
    Return the first key found in row as a float.
    Falls back to `default` if all keys are missing, null, or non-numeric.
    Never raises.
    """
    for k in keys:
        try:
            if k in row and pd.notna(row[k]):
                return float(row[k])
        except Exception:
            continue
    return default


def get_str_safe(row: "pd.Series", key: str, default: str = "") -> str:
    """Return row[key] as a stripped string, or default. Never raises."""
    try:
        v = row.get(key)
        if v is not None and pd.notna(v):
            return str(v).strip()
    except Exception:
        pass
    return default


# ─────────────────────────────────────────────────────────────────────
# CLASSIFICATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def _setup_type(
    high_dist: float,
    vol: float,
    delta_ema20: float,
    rsi: float,
) -> str:
    """
    Priority order (first match wins):
        1. Breakout
        2. Pullback
        3. Reversal
        4. Momentum Continuation
        5. Weak Setup (default)
    """
    # 1. Breakout
    if -2.0 <= high_dist <= 0.0 and vol > 1.5:
        return "Breakout"

    # 2. Pullback
    if abs(delta_ema20) < 3.0 and 50.0 <= rsi <= 60.0:
        return "Pullback"

    # 3. Reversal
    if rsi < 45.0:
        return "Reversal"

    # 4. Momentum Continuation
    if 55.0 <= rsi <= 70.0 and vol > 1.2:
        return "Momentum Continuation"

    # 5. Default
    return "Weak Setup"


def _reason(
    vol: float,
    rsi: float,
    high_dist: float,
    delta_ema20: float,
) -> str:
    """
    Build a comma-joined list of confirmation reasons.
    Returns a fallback string when no confirmations fire.
    """
    parts: list[str] = []

    if vol > 1.5:
        parts.append("Strong volume")
    if 50.0 <= rsi <= 65.0:
        parts.append("Healthy RSI")
    if high_dist > -2.0:
        parts.append("Near breakout level")
    if delta_ema20 < 4.0:
        parts.append("Not overextended")
    if rsi < 45.0:
        parts.append("Low RSI reversal zone")

    return ", ".join(parts) if parts else "Weak setup or missing confirmation"


def _risk_score(
    delta_ema20: float,
    rsi: float,
    vol: float,
) -> float:
    """
    Compute a 0–100 risk score.

        risk += abs(delta_ema20) * 1.2
        risk += max(0, RSI - 65)  * 1.2
        risk += max(0, 1 - vol)   * 15
        clamped to [0, 100]
    """
    risk = 0.0
    risk += abs(delta_ema20) * 1.2
    risk += max(0.0, rsi - 68.0) * 1.2
    risk += max(0.0, 1.0 - vol) * 15.0
    return float(np.clip(risk, 0.0, 100.0))


def _final_signal(
    trap_risk: str,
    setup_quality: str,
    entry_timing: str,
    volume_trend: str,
) -> str:
    """
    Derive the base Final Signal from Phase 3 columns.

    Only Trap Risk HIGH (2+ conditions) triggers TRAP label.
    Trap Risk MEDIUM is NOT a TRAP — it flows through normally.

    Priority:
        TRAP         → Trap Risk == "HIGH"  (requires 2 conditions)
        STRONG BUY   → Setup Quality HIGH + Entry Timing EARLY + Volume Trend STRONG
        BUY          → Setup Quality HIGH + Volume Trend != WEAK
        WATCH        → Setup Quality MEDIUM
        AVOID        → everything else
    """
    if trap_risk == "HIGH":
        return "TRAP"

    if setup_quality == "HIGH" and entry_timing == "EARLY" and volume_trend == "STRONG":
        return "STRONG BUY"

    if setup_quality == "HIGH" and volume_trend != "WEAK":
        return "BUY"

    # Medium-quality BUY: decent setup with volume confirmation and not a late entry
    # Prevents near-all results collapsing to WATCH even when conditions are good
    if (setup_quality == "MEDIUM"
            and volume_trend in ("STRONG", "BUILDING")
            and entry_timing not in ("LATE",)):
        return "BUY"

    if setup_quality == "MEDIUM":
        return "WATCH"

    return "AVOID"


def _parse_bias(market_bias: dict | None) -> str:
    """
    Normalise market_bias dict → "Bullish" / "Bearish" / "Sideways".
    Handles both:
        app.py local  → "Bullish bias", "Bearish bias", "Sideways / no edge"
        market_bias_engine.py → "Bullish", "Bearish", "Sideways"
    """
    if not market_bias or not isinstance(market_bias, dict):
        return "Sideways"
    raw = str(market_bias.get("bias", "")).strip().lower()
    if "bullish" in raw:
        return "Bullish"
    if "bearish" in raw:
        return "Bearish"
    return "Sideways"


# ─────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def apply_phase4_logic(
    df: pd.DataFrame,
    market_bias: dict | None = None,
) -> pd.DataFrame:
    """
    Add Phase 4.1 intelligence columns to the scan DataFrame.

    Must be called AFTER:
        df = apply_enhanced_logic(df)
        df = apply_universal_grading(df, mb)
        df = apply_phase4_logic(df, mb)   ← this function

    Parameters
    ----------
    df : pd.DataFrame
        Scan output. All columns are read safely — no KeyError possible.

    market_bias : dict | None
        Output of compute_market_bias() from any source.
        None → treated as Sideways (no adjustment).
        Note: market bias influence is already captured in Final Score
        via grading_engine. This function reads it for context but does
        NOT apply an additional hard downgrade.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with FOUR new columns:
            "Setup Type"   str   Breakout / Pullback / Reversal /
                                 Momentum Continuation / Weak Setup
            "Reason"       str   Human-readable confirmation string
            "Risk Score"   float 0–100
            "Final Signal" str   STRONG BUY / BUY / WATCH / AVOID / TRAP
        No rows removed. No sort order changed. No existing column modified.
    """
    # ── Guard ──────────────────────────────────────────────────────────
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
    except Exception:
        return df

    try:
        out = df.copy()

        # bias_token is parsed but used only informally; grading already
        # priced in the market regime via score adjustment.
        # We keep _parse_bias for potential future use and back-compat.
        _bias_token = _parse_bias(market_bias)  # noqa: F841

        setup_types:   list[str]   = []
        reasons:       list[str]   = []
        risk_scores:   list[float] = []
        final_signals: list[str]   = []

        for idx in out.index:
            try:
                row = out.loc[idx]

                # ── Safe source-column reads ───────────────────────────
                rsi         = get_safe(row, ["RSI"],                           50.0)
                vol         = get_safe(row, ["Vol / Avg"],                      1.0)
                delta_ema20 = get_safe(row, ["Δ vs EMA20 (%)"],                 0.0)
                ret_5d      = get_safe(row, ["5D Return (%)"],                  0.0)  # noqa: F841
                high_dist   = get_safe(row, ["Δ vs 20D High (%)", "Near High (%)"], -5.0)

                # ── Phase 3 classification columns (safe string read) ──
                trap_risk    = get_str_safe(row, "Trap Risk",    "LOW")
                setup_qual   = get_str_safe(row, "Setup Quality", "MEDIUM")
                entry_timing = get_str_safe(row, "Entry Timing",  "NEUTRAL")
                vol_trend    = get_str_safe(row, "Volume Trend",  "NORMAL")

                # ── 1. Setup Type ──────────────────────────────────────
                st_val = _setup_type(high_dist, vol, delta_ema20, rsi)

                # ── 2. Reason ──────────────────────────────────────────
                rs_val = _reason(vol, rsi, high_dist, delta_ema20)

                # ── 3. Risk Score ──────────────────────────────────────
                rk_val = _risk_score(delta_ema20, rsi, vol)

                # ── 4. Final Signal (no additional market downgrade) ───
                fs_val = _final_signal(trap_risk, setup_qual, entry_timing, vol_trend)

            except Exception:
                # Row-level fail-safe — never crash the whole loop
                st_val = "Weak Setup"
                rs_val = "Weak setup or missing confirmation"
                rk_val = 50.0
                fs_val = "AVOID"

            setup_types.append(st_val)
            reasons.append(rs_val)
            risk_scores.append(round(rk_val, 2))
            final_signals.append(fs_val)

        # ── Assign columns (no existing column touched) ────────────────
        out["Setup Type"]   = setup_types
        out["Reason"]       = reasons
        out["Risk Score"]   = risk_scores
        out["Final Signal"] = final_signals

        return out

    except Exception:
        # Absolute fail-safe — return original df unchanged
        return df


# ─────────────────────────────────────────────────────────────────────
# PHASE 4.2 — ADVANCED TRAP / EXPECTED MOVE / ADJUSTED SIGNAL
# ─────────────────────────────────────────────────────────────────────

def _advanced_trap(high_dist: float, vol: float, rsi: float) -> str:
    """
    Additional trap layer — does NOT touch or replace "Trap Risk".

    Priority (first match wins):
        FAKE BREAKOUT  → near 20D high but thin volume
        EXHAUSTION     → overbought RSI with drying volume
        WEAK VOLUME    → below-average volume (informational only)
        NONE           → no trap signal
    """
    if high_dist > -1.0 and vol < 1.2:
        return "FAKE BREAKOUT"
    if rsi > 70.0 and vol < 1.0:
        return "EXHAUSTION"
    if vol < 0.9:
        return "WEAK VOLUME"
    return "NONE"


def _expected_move(vol: float, rsi: float) -> str:
    """
    Estimate expected price move range based on volume and RSI zone.

    +5% to +10%  → explosive volume in healthy RSI zone
    +2% to +5%   → strong volume
    +0% to +2%   → mild volume
    Uncertain    → below-average volume
    """
    if vol > 2.0 and 55.0 <= rsi <= 65.0:
        return "+5% to +10%"
    if vol > 1.5:
        return "+2% to +5%"
    if vol > 1.0:
        return "+0% to +2%"
    return "Uncertain"


def _adjusted_signal(
    final_signal: str,
    risk_score: float,
    advanced_trap: str,
) -> str:
    """
    Refine "Final Signal" using Risk Score and Advanced Trap.

    Softened rules vs previous version:
        1. FAKE BREAKOUT or EXHAUSTION → downgrade one level (was: auto-AVOID)
        2. WEAK VOLUME                 → no change (informational only)
        3. Risk Score > 80             → downgrade one level (was: > 75)
        4. Risk Score < 30 + BUY       → upgrade to STRONG BUY
        5. Default                     → keep Final Signal unchanged

    Hard floors: TRAP stays TRAP.  AVOID stays AVOID (no further downgrade).
    Signal hierarchy: STRONG BUY > BUY > WATCH > AVOID > TRAP
    """
    # TRAP is always preserved — highest-severity state
    if final_signal == "TRAP":
        return "TRAP"

    _downgrade = {
        "STRONG BUY": "BUY",
        "BUY":        "WATCH",
        "WATCH":      "AVOID",
        "AVOID":      "AVOID",  # floor
    }

    # Rule 1: significant traps → downgrade one level (not auto-AVOID)
    # WEAK VOLUME is purely informational and does NOT trigger a downgrade.
    if advanced_trap in ("FAKE BREAKOUT", "EXHAUSTION"):
        return _downgrade.get(final_signal, "AVOID")

    # Rule 2: high risk → downgrade one level (threshold raised to 80)
    if risk_score > 80.0:
        return _downgrade.get(final_signal, "AVOID")

    # Rule 3: low risk + BUY → promote to STRONG BUY
    if risk_score < 30.0 and final_signal == "BUY":
        return "STRONG BUY"

    # Rule 4: keep as-is
    return final_signal


def apply_phase42_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Phase 4.2 intelligence columns to the scan DataFrame.

    Must be called AFTER apply_phase4_logic():
        df = apply_phase4_logic(df, mb)
        df = apply_phase42_logic(df)      ← this function

    Parameters
    ----------
    df : pd.DataFrame
        Scan output. All columns are read safely — no KeyError possible.
        Phase 4.1 columns ("Final Signal", "Risk Score") are consumed if
        present; graceful defaults are used if absent.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with THREE new columns:
            "Advanced Trap"    str   FAKE BREAKOUT / EXHAUSTION /
                                     WEAK VOLUME / NONE
            "Expected Move"    str   +5% to +10% / +2% to +5% /
                                     +0% to +2% / Uncertain
            "Adjusted Signal"  str   STRONG BUY / BUY / WATCH / AVOID / TRAP
        No rows removed. No sort order changed. No existing column modified.
    """
    # ── Guard ──────────────────────────────────────────────────────────
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
    except Exception:
        return df

    try:
        out = df.copy()

        adv_traps:   list[str] = []
        exp_moves:   list[str] = []
        adj_signals: list[str] = []

        for idx in out.index:
            try:
                row = out.loc[idx]

                # ── Safe source-column reads ───────────────────────────
                rsi       = get_safe(row, ["RSI"],            50.0)
                vol       = get_safe(row, ["Vol / Avg"],       1.0)
                high_dist = get_safe(
                    row,
                    ["Δ vs 20D High (%)", "Near High (%)"],
                    -5.0,
                )

                # Phase 4.1 outputs (safe string/float read)
                final_sig  = get_str_safe(row, "Final Signal", "AVOID")
                risk_score = get_safe(row, ["Risk Score"],     50.0)

                # ── 1. Advanced Trap ───────────────────────────────────
                at_val = _advanced_trap(high_dist, vol, rsi)

                # ── 2. Expected Move ───────────────────────────────────
                em_val = _expected_move(vol, rsi)

                # ── 3. Adjusted Signal (softened logic) ────────────────
                as_val = _adjusted_signal(final_sig, risk_score, at_val)

            except Exception:
                # Row-level fail-safe
                at_val = "NONE"
                em_val = "Uncertain"
                as_val = "AVOID"

            adv_traps.append(at_val)
            exp_moves.append(em_val)
            adj_signals.append(as_val)

        # ── Assign columns (no existing column touched) ────────────────
        out["Advanced Trap"]   = adv_traps
        out["Expected Move"]   = exp_moves
        out["Adjusted Signal"] = adj_signals

        return out

    except Exception:
        # Absolute fail-safe — return original df unchanged
        return df