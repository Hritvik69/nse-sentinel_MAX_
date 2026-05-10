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
• Works for ALL scan modes and CSV mode.
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

from strategy_engines.mode_helpers import resolve_mode_id


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


def _numeric_series(df: pd.DataFrame, keys: list[str], default: float) -> pd.Series:
    """Vectorized safe numeric column access."""
    for key in keys:
        if key not in df.columns:
            continue
        col = df[key]
        if pd.api.types.is_numeric_dtype(col):
            return pd.to_numeric(col, errors="coerce").fillna(default).astype(float)
        cleaned = (
            col.astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "-": np.nan, "—": np.nan})
            .str.replace(r"[%xX×,]", "", regex=True)
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _string_series(df: pd.DataFrame, key: str, default: str = "") -> pd.Series:
    """Vectorized safe string column access."""
    if key not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[key].where(df[key].notna(), default).astype(str).str.strip().replace("", default)


def _mode_id_series(df: pd.DataFrame) -> pd.Series:
    """Resolve strategy mode id once per dataframe."""
    if "Mode ID" in df.columns:
        return pd.to_numeric(df["Mode ID"], errors="coerce").fillna(-1).astype(int)
    if "Mode" in df.columns:
        return df["Mode"].apply(lambda value: resolve_mode_id(value, -1)).fillna(-1).astype(int)
    return pd.Series(-1, index=df.index, dtype=int)


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


def _setup_type_mode7(
    high_dist: float,
    vol: float,
    delta_ema20: float,
    rsi: float,
    ret_5d: float,
    ret_20d: float,
    trap_risk: str,
) -> str:
    """Mode 7 setup labels: structure-first momentum language."""
    trap = str(trap_risk or "LOW").strip().upper()
    if trap == "HIGH" or (high_dist > 1.5 and vol < 1.2):
        return "FAKE BREAKOUT RISK"
    if delta_ema20 > 7.0 or rsi > 74.0 or ret_5d > 14.0:
        return "OVEREXTENDED"
    if -2.0 <= high_dist <= 0.5 and vol >= 1.3 and 52.0 <= rsi <= 70.0:
        return "BREAKOUT READY"
    if 0.5 < high_dist <= 2.0 and vol >= 1.4 and delta_ema20 <= 6.0 and 52.0 <= rsi <= 70.0:
        return "EARLY BREAKOUT"
    if -6.0 <= high_dist < -2.0 and vol >= 1.1 and delta_ema20 <= 4.5 and 50.0 <= rsi <= 67.0:
        return "RESISTANCE COMPRESSION"
    if abs(delta_ema20) <= 3.0 and -2.0 <= ret_5d <= 7.0 and ret_20d > 3.0 and 50.0 <= rsi <= 64.0:
        return "SUPPORT BOUNCE"
    if 2.0 <= ret_5d <= 9.0 and 5.0 <= ret_20d <= 18.0 and vol >= 1.2 and 52.0 <= rsi <= 70.0:
        return "MOMENTUM CONTINUATION"
    return "MOMENTUM CONTINUATION"


def _reason_mode7(
    setup_type: str,
    vol: float,
    rsi: float,
    high_dist: float,
    delta_ema20: float,
    ret_5d: float,
    ret_20d: float,
) -> str:
    parts: list[str] = []
    if setup_type == "BREAKOUT READY":
        parts.append("Clean resistance breakout zone")
    elif setup_type == "SUPPORT BOUNCE":
        parts.append("Pullback holding EMA20 support")
    elif setup_type == "RESISTANCE COMPRESSION":
        parts.append("Price compressing below resistance")
    elif setup_type == "EARLY BREAKOUT":
        parts.append("Fresh breakout with controlled extension")
    elif setup_type == "FAKE BREAKOUT RISK":
        parts.append("Breakout lacks clean confirmation")
    elif setup_type == "OVEREXTENDED":
        parts.append("Late-stage or stretched momentum")
    else:
        parts.append("Trend continuation structure")

    if 1.4 <= vol <= 2.8:
        parts.append("institutional volume confirmed")
    elif vol < 1.0:
        parts.append("weak volume")
    if 55.0 <= rsi <= 67.0:
        parts.append("controlled RSI")
    elif rsi > 74.0:
        parts.append("RSI exhaustion risk")
    if -2.0 <= high_dist <= 1.5:
        parts.append("near resistance zone")
    if delta_ema20 <= 5.0:
        parts.append("not overextended")
    if 2.0 <= ret_5d <= 9.0 and 5.0 <= ret_20d <= 18.0:
        parts.append("measured 5D/20D momentum")
    return ", ".join(parts)


def _risk_score_mode7(
    delta_ema20: float,
    rsi: float,
    vol: float,
    high_dist: float,
    ret_5d: float,
    trap_risk: str,
) -> float:
    risk = 0.0
    risk += max(0.0, delta_ema20 - 4.0) * 4.0
    risk += max(0.0, rsi - 68.0) * 2.0
    risk += max(0.0, 1.2 - vol) * 18.0
    risk += max(0.0, high_dist - 2.0) * 5.0
    risk += max(0.0, ret_5d - 9.0) * 3.0
    if str(trap_risk or "").strip().upper() == "HIGH":
        risk += 35.0
    elif str(trap_risk or "").strip().upper() == "MEDIUM":
        risk += 12.0
    return float(np.clip(risk, 0.0, 100.0))


def _final_signal_mode7(
    setup_type: str,
    trap_risk: str,
    setup_quality: str,
    volume_trend: str,
    structure_quality: str,
) -> str:
    setup = str(setup_type or "").strip().upper()
    trap = str(trap_risk or "LOW").strip().upper()
    quality = str(setup_quality or "MEDIUM").strip().upper()
    volume = str(volume_trend or "NORMAL").strip().upper()
    structure = str(structure_quality or "MEDIUM").strip().upper()

    if trap == "HIGH" or setup == "FAKE BREAKOUT RISK":
        return "TRAP"
    if setup == "OVEREXTENDED":
        return "AVOID" if quality == "LOW" or trap == "MEDIUM" else "WATCH"
    if (
        setup in ("BREAKOUT READY", "SUPPORT BOUNCE", "EARLY BREAKOUT")
        and quality == "HIGH"
        and structure == "HIGH"
        and volume == "STRONG"
        and trap == "LOW"
    ):
        return "STRONG BUY"
    if (
        setup in ("BREAKOUT READY", "SUPPORT BOUNCE", "MOMENTUM CONTINUATION", "RESISTANCE COMPRESSION", "EARLY BREAKOUT")
        and quality in ("HIGH", "MEDIUM")
        and structure in ("HIGH", "MEDIUM")
        and volume != "WEAK"
    ):
        return "BUY" if quality == "HIGH" or setup in ("BREAKOUT READY", "SUPPORT BOUNCE") else "WATCH"
    return "WATCH" if structure == "MEDIUM" else "AVOID"


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

        rsi = _numeric_series(out, ["RSI"], 50.0)
        vol = _numeric_series(out, ["Vol / Avg"], 1.0)
        delta_ema20 = _numeric_series(out, ["Δ vs EMA20 (%)"], 0.0)
        high_dist = _numeric_series(out, ["Δ vs 20D High (%)", "Near High (%)"], -5.0)
        ret_5d = _numeric_series(out, ["5D Return (%)"], 0.0)
        ret_20d = _numeric_series(out, ["20D Return (%)"], 0.0)
        trap_risk = _string_series(out, "Trap Risk", "LOW").str.upper()
        setup_qual = _string_series(out, "Setup Quality", "MEDIUM").str.upper()
        entry_timing = _string_series(out, "Entry Timing", "NEUTRAL").str.upper()
        vol_trend = _string_series(out, "Volume Trend", "NORMAL").str.upper()
        mode_ids = _mode_id_series(out)
        structure_qual = _string_series(out, "Structure Quality", "MEDIUM").str.upper()

        setup_types = [
            _setup_type(h, v, d, r)
            for h, v, d, r in zip(high_dist, vol, delta_ema20, rsi)
        ]
        reasons = [
            _reason(v, r, h, d)
            for v, r, h, d in zip(vol, rsi, high_dist, delta_ema20)
        ]
        risk_scores = [
            round(_risk_score(d, r, v), 2)
            for d, r, v in zip(delta_ema20, rsi, vol)
        ]
        final_signals = [
            _final_signal(tr, sq, et, vt)
            for tr, sq, et, vt in zip(trap_risk, setup_qual, entry_timing, vol_trend)
        ]

        for i, (m, h, v, d, r, r5, r20, tr, sq, vt, stq) in enumerate(
            zip(
                mode_ids,
                high_dist,
                vol,
                delta_ema20,
                rsi,
                ret_5d,
                ret_20d,
                trap_risk,
                setup_qual,
                vol_trend,
                structure_qual,
            )
        ):
            if int(m) != 7:
                continue
            setup7 = _setup_type_mode7(h, v, d, r, r5, r20, tr)
            setup_types[i] = setup7
            reasons[i] = _reason_mode7(setup7, v, r, h, d, r5, r20)
            risk_scores[i] = round(_risk_score_mode7(d, r, v, h, r5, tr), 2)
            final_signals[i] = _final_signal_mode7(setup7, tr, sq, vt, stq)

        out["Setup Type"] = setup_types
        out["Reason"] = reasons
        out["Risk Score"] = risk_scores
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


def _advanced_trap_mode7(
    high_dist: float,
    vol: float,
    rsi: float,
    delta_ema20: float,
    ret_5d: float,
    setup_type: str,
) -> str:
    """Mode 7 fake-breakout/exhaustion detection."""
    setup = str(setup_type or "").strip().upper()
    if setup == "FAKE BREAKOUT RISK" or (high_dist > 1.5 and vol < 1.2):
        return "FAKE BREAKOUT"
    if setup == "OVEREXTENDED" or rsi > 76.0 or delta_ema20 > 8.0 or ret_5d > 14.0:
        return "EXHAUSTION"
    if vol < 1.0:
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

        rsi = _numeric_series(out, ["RSI"], 50.0)
        vol = _numeric_series(out, ["Vol / Avg"], 1.0)
        high_dist = _numeric_series(out, ["Δ vs 20D High (%)", "Near High (%)"], -5.0)
        delta_ema20 = _numeric_series(out, ["Δ vs EMA20 (%)"], 0.0)
        ret_5d = _numeric_series(out, ["5D Return (%)"], 0.0)
        final_sig = _string_series(out, "Final Signal", "AVOID").str.upper()
        risk_score = _numeric_series(out, ["Risk Score"], 50.0)
        setup_type = _string_series(out, "Setup Type", "").str.upper()
        mode_ids = _mode_id_series(out)

        adv_traps = [
            _advanced_trap(h, v, r)
            for h, v, r in zip(high_dist, vol, rsi)
        ]
        for i, (m, h, v, r, d, r5, st) in enumerate(
            zip(mode_ids, high_dist, vol, rsi, delta_ema20, ret_5d, setup_type)
        ):
            if int(m) == 7:
                adv_traps[i] = _advanced_trap_mode7(h, v, r, d, r5, st)
        out["Advanced Trap"] = adv_traps
        out["Expected Move"] = [
            _expected_move(v, r)
            for v, r in zip(vol, rsi)
        ]
        out["Adjusted Signal"] = [
            _adjusted_signal(fs, rs, at)
            for fs, rs, at in zip(final_sig, risk_score, adv_traps)
        ]
        return out

    except Exception:
        # Absolute fail-safe — return original df unchanged
        return df
