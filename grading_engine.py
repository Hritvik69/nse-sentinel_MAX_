"""
grading_engine.py
─────────────────
Universal Grading Engine for NSE Sentinel.

Accepts the scan DataFrame that has already been through enhance_results()
and (optionally) apply_enhanced_logic(), and adds SIX new columns WITHOUT
removing any existing ones:

    "Final Score"      – re-computed composite (score + bt + ml + market adj)
    "Grade"            – A+ / A / B / C / D
    "Signal"           – STRONG BUY / BUY / WATCH / AVOID
    "Confidence"       – (bt + ml) / 2
    "Market Bias"      – human-readable label from the supplied bias dict
    "Market Regime"    – regime string from bias dict (e.g. "Trending Up")
    "Prediction Score" – probability-style combined rank (0–100)
    "Conviction Tier"  – High / Medium / Low (prediction + trap-aware)

Design rules
────────────
• Zero API calls — all logic operates on the in-memory DataFrame.
• Never crashes — every path is wrapped in try/except; returns df unchanged
  on any unexpected error.
• Never removes rows — DO NOT filter; only rank via sort.
• Column names are detected flexibly (Score vs Final Score, ML % vs ML Prob,
  Backtest % vs BT Prob / Backtest).
• market_bias dict values are normalised with `in` checks so both the
  app.py local version ("Bullish bias", "Bearish bias", "Sideways / no edge")
  and the market_bias_engine.py version ("Bullish", "Bearish", "Sideways")
  are handled identically.
• Market bias acts as a SOFT score adjustment (confidence-scaled ±0–10 pts),
  NOT a hard signal blocker.  The signal is derived purely from the adjusted
  score so the bias influence is already priced in.
• Bias adjustment is now proportional to market_bias confidence:
    confidence 50  →  ±0 pts   (low-confidence bias has minimal effect)
    confidence 75  →  ±6.25 pts
    confidence 88+ →  ±10 pts  (high-confidence regime has full effect)
• Market regime adds a soft context bonus/penalty to Prediction Score only
  (not Final Score), via _REGIME_ADJ — scaled by bias confidence so weak
  macro reads move ranks less than high-confidence ones.
• Prediction Score uses Setup Quality / Volume Trend / Trap Risk when those
  columns are present (i.e. when apply_enhanced_logic ran before this).

Public entry point
──────────────────
    from grading_engine import apply_universal_grading

    mb = compute_market_bias()          # any source — dict with "bias" key
    df = apply_universal_grading(df, mb)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _safe_float(v: object, default: float = 0.0) -> float:
    """Return float(v) if finite, else default. Never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _get_col(df: pd.DataFrame, *names: str, default: float = 50.0) -> pd.Series:
    """
    Return the first matching column from df (case-insensitive search).
    Falls back to a constant Series of `default` if none of the names exist.
    """
    df_cols_lower = {c.lower(): c for c in df.columns}
    for name in names:
        actual = df_cols_lower.get(name.lower())
        if actual is not None:
            return df[actual].apply(lambda v: _safe_float(v, default))
    return pd.Series([default] * len(df), index=df.index)


def _parse_bias(market_bias: dict | None) -> str:
    """
    Normalise the market_bias dict into one of three tokens:
        "Bullish"  – any bias string containing "Bullish" (case-insensitive)
        "Bearish"  – any bias string containing "Bearish"
        "Sideways" – everything else (including None / missing dict)

    Handles both:
      • app.py local version  → "Bullish bias", "Bearish bias", "Sideways / no edge"
      • market_bias_engine.py → "Bullish", "Bearish", "Sideways"
    """
    if not market_bias or not isinstance(market_bias, dict):
        return "Sideways"
    raw = str(market_bias.get("bias", "")).strip()
    if not raw:
        return "Sideways"
    rl = raw.lower()
    if "bullish" in rl:
        return "Bullish"
    if "bearish" in rl:
        return "Bearish"
    return "Sideways"


def _parse_bias_confidence(market_bias: dict | None) -> float:
    """
    Extract confidence value (0–100) from market_bias dict.
    Defaults to 50 (neutral / no boost) when missing or invalid.
    """
    if not market_bias or not isinstance(market_bias, dict):
        return 50.0
    try:
        return float(np.clip(market_bias.get("confidence", 50), 0.0, 100.0))
    except Exception:
        return 50.0


def _parse_regime(market_bias: dict | None) -> str:
    """
    Extract market regime string from market_bias dict.
    Returns "Ranging" as the neutral default.

    Accepts both:
      • market_bias_engine.py → "regime" key
      • Fallback when not present → "Ranging"
    """
    if not market_bias or not isinstance(market_bias, dict):
        return "Ranging"
    return str(market_bias.get("regime", "Ranging")).strip() or "Ranging"


def _grade(score: float) -> str:
    """Map final_score → letter grade."""
    if score >= 80:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 60:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _signal(score: float) -> str:
    """
    Map final_score → trading signal.

    The score already incorporates the market bias adjustment,
    so no additional hard downgrade is applied here.
    Using the score directly avoids double-penalising bearish markets.

        score ≥ 80 → STRONG BUY
        score ≥ 70 → BUY
        score ≥ 55 → WATCH
        score <  55 → AVOID
    """
    if score >= 80:
        return "STRONG BUY"
    if score >= 70:
        return "BUY"
    if score >= 55:
        return "WATCH"
    return "AVOID"


# ─────────────────────────────────────────────────────────────────────
# SOFT MODIFIERS FROM ENHANCED LOGIC COLUMNS (optional — safe fallback)
# ─────────────────────────────────────────────────────────────────────

# Applied only when apply_enhanced_logic() ran before grading.
# If columns are absent, adjustments default to 0 → no effect.
_SQ_BONUS = {"HIGH": 5.0,  "MEDIUM": 0.0, "LOW": -5.0}
_VT_BONUS = {"STRONG": 4.0, "BUILDING": 2.0, "NORMAL": 0.0, "WEAK": -4.0}
# Trap: stronger distinction — real danger vs caution (still soft, not a filter)
_TR_PEN   = {"HIGH": -14.0, "MEDIUM": -6.0, "LOW": 0.0}

# Regime context soft bonus/penalty applied to Prediction Score only.
# These adjust ONLY prediction ranking, not Final Score or Signal.
# Conservative: bonuses are small and capped; no hard blocks.
_REGIME_ADJ: dict[str, float] = {
    "Trending Up":                    3.0,   # clear bull trend confirmation
    "Trending Down":                 -5.0,   # bear regime lowers ranking
    "Breakout Pending (Squeeze)":     2.0,   # indeterminate but interesting
    "Oversold Bounce Zone":           4.0,   # mean-reversion opportunity
    "Overbought Pullback Risk":      -3.0,   # caution near exhaustion
    "High Volatility / Choppy":      -2.0,   # noisy signals
    "Ranging":                        0.0,   # neutral
}

# Entry timing nudges (Prediction Score only; from enhanced_logic_engine)
_ET_BONUS = {"EARLY": 1.5, "GOOD": 1.0, "NEUTRAL": 0.0, "LATE": -3.0}


def _conviction_tier(pred: float, conf: float, trap: str) -> str:
    """High / Medium / Low — rank-lowering when trap is real danger, not binary hide."""
    try:
        t = str(trap or "LOW").strip().upper()
        if pred >= 66 and conf >= 52 and t != "HIGH":
            return "High"
        if pred >= 66 and conf >= 52 and t == "HIGH":
            return "Medium"
        if pred >= 42:
            return "Medium"
        return "Low"
    except Exception:
        return "Low"


def _prediction_score(
    adj_fs: float,
    conf: float,
    row: "pd.Series",
    regime_adj: float = 0.0,
    bias_token: str = "Sideways",
    bias_conf: float = 50.0,
    regime_label: str = "Ranging",
) -> float:
    """
    Probability-style prediction score (0–100).

    Combines:
      • 60% weight on the bias-adjusted Final Score
      • 25% weight on Confidence (backtest+ML average)
      • Up to ±5  pts from Setup Quality  (if column present)
      • Up to ±4  pts from Volume Trend   (if column present)
      • Up to -10 pts Trap Risk penalty   (if column present)
      • Up to ±5  pts from market Regime context (passed in)
      • Overextension / weak participation (soft, clearly felt on rank)
      • Macro alignment nudge: bearish bias pulls prediction rank more when
        confidence is high; bullish nudge is small and regime-conditional
    """
    try:
        # ── Detect Mode 6 (Swing) safely ──────────────────────────────
        # Mode can be an int, "6", or a label like "🔴 Swing".
        is_mode6 = False
        try:
            _m_raw = row.get("Mode", row.get("mode", None))
            if _m_raw is not None:
                if isinstance(_m_raw, (int, float)):
                    is_mode6 = int(_m_raw) == 6
                else:
                    _m_s = str(_m_raw).strip().lower()
                    is_mode6 = (_m_s == "6") or ("swing" in _m_s) or ("m6" in _m_s) or ("mode 6" in _m_s)
        except Exception:
            is_mode6 = False

        sq = str(row.get("Setup Quality", "MEDIUM") or "MEDIUM").strip().upper()
        vt = str(row.get("Volume Trend",  "NORMAL") or "NORMAL").strip().upper()
        tr = str(row.get("Trap Risk",     "LOW")    or "LOW").strip().upper()
        et = str(row.get("Entry Timing",  "NEUTRAL") or "NEUTRAL").strip().upper()

        sq_adj = _SQ_BONUS.get(sq, 0.0)
        vt_adj = _VT_BONUS.get(vt, 0.0)
        tr_adj = _TR_PEN.get(tr,   0.0)
        # Mode 6: soften trap penalties (trend continuation can look "trappy")
        if is_mode6:
            if tr == "HIGH":
                tr_adj = -5.0
            elif tr == "MEDIUM":
                tr_adj = -2.0
        et_adj = _ET_BONUS.get(et, 0.0)

        # Weighted blend: Final Score is the anchor
        base = 0.60 * adj_fs + 0.25 * conf
        # Mode 6: dampen regime impact
        _reg_adj = regime_adj * 0.5 if is_mode6 else regime_adj
        pred = base + sq_adj + vt_adj + tr_adj + _reg_adj + et_adj

        # Soft stock-specific timing / participation (Prediction Score only)
        rsi = _safe_float(row.get("RSI", 50), 50.0)
        if is_mode6:
            # Swing trends tolerate higher RSI; penalize only extreme extension
            if rsi > 85:
                pred -= 4.0
            elif rsi > 75:
                pred -= 1.5
        else:
            if rsi > 75:
                pred -= 6
            elif rsi > 70:
                pred -= 3

        vol = _safe_float(row.get("Vol / Avg", 1), 1.0)
        if is_mode6:
            # Early swing accumulation can be quieter
            if vol < 0.7:
                pred -= 1.5
        else:
            if vol < 1:
                pred -= 2.5

        # Prediction-layer macro alignment (does not change Final Score)
        cf = float(np.clip((bias_conf - 50.0) / 40.0, 0.0, 1.0))
        # Mode 6: dampen macro noise in prediction rank
        if is_mode6:
            cf *= 0.6
        if bias_token == "Bearish":
            pred -= 3.5 * cf
        elif bias_token == "Bullish":
            if regime_label in ("Trending Up", "Oversold Bounce Zone", "Breakout Pending (Squeeze)"):
                pred += 1.2 * cf
            else:
                pred += 0.35 * cf

        # Optional Mode 6 trend bonus (safe add; only if data is present)
        if is_mode6:
            try:
                price = _safe_float(
                    row.get("Price (₹)", row.get("Close (₹)", row.get("Close", row.get("Price", 0.0)))),
                    0.0,
                )
                ema20 = _safe_float(row.get("EMA 20", row.get("EMA20", 0.0)), 0.0)
                slope_ok = False
                slope_val = row.get("EMA20 Slope", row.get("ema20_slope", None))
                if slope_val is not None:
                    try:
                        slope_ok = float(slope_val) > 0
                    except Exception:
                        slope_ok = "rising" in str(slope_val).strip().lower()
                if not slope_ok:
                    br = row.get("_breakdown", None)
                    if isinstance(br, dict):
                        # tolerate slightly different key spelling
                        for k in ("EMA20 Slope", "EMA 20 Slope", "EMA20 slope"):
                            if k in br:
                                slope_ok = "rising" in str(br.get(k, "")).strip().lower()
                                break
                if price > ema20 > 0 and slope_ok:
                    pred += 3.0
            except Exception:
                pass

        return float(np.clip(round(pred, 2), 0.0, 100.0))
    except Exception:
        return float(np.clip(round(adj_fs, 2), 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def apply_universal_grading(
    df: pd.DataFrame,
    market_bias: dict | None = None,
) -> pd.DataFrame:
    """
    Add Universal Grading columns to the scan result DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Scan output from enhance_results() — must not be empty.
        Expected columns (auto-detected with fallbacks):
            Score / Final Score  →  base strategy score (0–100)
            Backtest % / BT Prob / Backtest  →  historical win-rate (0–100)
            ML % / ML Prob       →  ML probability (0–100)
        Optional (used when apply_enhanced_logic ran first):
            Setup Quality, Volume Trend, Trap Risk

    market_bias : dict | None
        Output of compute_market_bias() from either app.py or
        market_bias_engine.py.  None ⇒ treated as Sideways.

        Used for three adjustments:
          1. Final Score bias adjustment (confidence-scaled ±0–10 pts)
          2. Market Bias column label
          3. Market Regime column + regime_adj in Prediction Score

    Returns
    -------
    pd.DataFrame
        Same DataFrame with SEVEN new columns appended (existing cols untouched):
            "Final Score"      float  0–100
            "Grade"            str    A+ / A / B / C / D
            "Signal"           str    STRONG BUY / BUY / WATCH / AVOID
            "Confidence"       float  0–100
            "Market Bias"      str    Bullish / Bearish / Sideways
            "Market Regime"    str    regime label (Trending Up / Ranging / etc.)
            "Prediction Score" float  0–100  (probability-style rank)

        Sorted by "Prediction Score" descending.
        No rows are removed.
    """
    # ── Guard: return unchanged on empty or non-DataFrame input ────────
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
    except Exception:
        return df

    try:
        out = df.copy()

        # ── 1. Detect source columns ───────────────────────────────────
        score_raw = _get_col(out, "Score", "Final Score", default=0.0)
        bt_raw    = _get_col(out, "Backtest %", "BT Prob", "Backtest", default=50.0)
        ml_raw    = _get_col(out, "ML %", "ML Prob", "ML Score", default=50.0)

        # ── 2. Parse market bias — direction, confidence, and regime ───
        bias_token   = _parse_bias(market_bias)
        bias_conf    = _parse_bias_confidence(market_bias)   # 0–100
        regime       = _parse_regime(market_bias)
        bias_display = bias_token  # "Bullish" / "Bearish" / "Sideways"

        # ── Confidence-scaled bias adjustment ─────────────────────────
        # At confidence=50  →  scale_factor=0   →  ±0 pts  (neutral)
        # At confidence=70  →  scale_factor=0.5 →  ±5 pts
        # At confidence=90  →  scale_factor=1.0 →  ±10 pts (max)
        # This prevents low-confidence bias readings from distorting scores.
        conf_factor = float(np.clip((bias_conf - 50.0) / 40.0, 0.0, 1.0))
        _bias_base  = {"Bullish": 10.0, "Bearish": -10.0, "Sideways": 0.0}
        mkt_adj     = _bias_base.get(bias_token, 0.0) * conf_factor

        # ── Regime adjustment for Prediction Score only (not Final Score)
        regime_adj = _REGIME_ADJ.get(regime, 0.0)
        # Suppress bullish regime bonuses slightly in bearish bias context
        if bias_token == "Bearish" and regime_adj > 0.0:
            regime_adj *= 0.3
        # Suppress bearish regime penalties slightly in bullish bias context
        if bias_token == "Bullish" and regime_adj < 0.0:
            regime_adj *= 0.5
        # Low macro confidence → dampen regime effect on prediction rank (not Sideways flatten)
        regime_adj *= 0.15 + 0.85 * conf_factor

        # ── 3. Compute per-row values ──────────────────────────────────
        final_scores:    list[float] = []
        grades:          list[str]   = []
        signals:         list[str]   = []
        confidences:     list[float] = []
        pred_scores:     list[float] = []
        conviction_tiers: list[str]   = []

        for idx in out.index:
            sc = _safe_float(score_raw.loc[idx], 0.0)
            bt = _safe_float(bt_raw.loc[idx],    50.0)
            ml = _safe_float(ml_raw.loc[idx],    50.0)

            # Core formula (score + backtest + ML, market-adjusted, clamped)
            raw_fs = 0.5 * sc + 0.3 * bt + 0.2 * ml
            adj_fs = float(np.clip(raw_fs + mkt_adj, 0.0, 100.0))
            conf   = float(np.clip((bt + ml) / 2.0, 0.0, 100.0))

            row_series = out.loc[idx]
            pred = _prediction_score(
                adj_fs,
                conf,
                row_series,
                regime_adj,
                bias_token,
                bias_conf,
                regime,
            )
            tr_lbl = str(row_series.get("Trap Risk", "LOW") or "LOW").strip()
            conviction_tiers.append(_conviction_tier(pred, conf, tr_lbl))

            final_scores.append(round(adj_fs, 2))
            grades.append(_grade(adj_fs))
            signals.append(_signal(adj_fs))      # no hard bias downgrade here
            confidences.append(round(conf, 2))
            pred_scores.append(pred)

        # ── 4. Attach new columns (overwrite if they exist) ────────────
        out["Final Score"]      = final_scores
        out["Grade"]            = grades
        out["Signal"]           = signals
        out["Confidence"]       = confidences
        out["Market Bias"]      = bias_display
        out["Market Regime"]    = regime       # new column — non-breaking
        out["Prediction Score"] = pred_scores
        out["Conviction Tier"]  = conviction_tiers

        # ── FIX 4 — Regime-aware grade cap ────────────────────────────
        # High-confidence bearish market: cap Grade at "A" (A+ → A).
        # Does NOT touch Final Score — label only.
        try:
            if "bearish" in bias_token.lower() and bias_conf >= 70.0:
                out["Grade"] = out["Grade"].apply(
                    lambda g: "A" if g == "A+" else g
                )
        except Exception:
            pass

        # ── FIX 3 — Conviction Tier post-pass using Setup Quality + Entry Timing
        # Upgrade: HIGH quality + EARLY timing → bump Low→Medium, Medium→High
        # Downgrade: LOW quality + LATE timing → nudge High→Medium (one step max)
        # Never downgrades to Low, never changes score.
        try:
            if "Setup Quality" in out.columns and "Entry Timing" in out.columns:
                _upgrade_mask = (
                    (out["Setup Quality"] == "HIGH") & (out["Entry Timing"] == "EARLY")
                )
                _downgrade_mask = (
                    (out["Setup Quality"] == "LOW") & (out["Entry Timing"] == "LATE")
                )
                _tier_up = {"Low": "Medium", "Medium": "High", "High": "High"}
                _tier_dn = {"High": "Medium", "Medium": "Medium", "Low": "Low"}
                out.loc[_upgrade_mask, "Conviction Tier"] = (
                    out.loc[_upgrade_mask, "Conviction Tier"].map(_tier_up)
                )
                out.loc[_downgrade_mask, "Conviction Tier"] = (
                    out.loc[_downgrade_mask, "Conviction Tier"].map(_tier_dn)
                )
        except Exception:
            pass

        # ── 5. Sort by Prediction Score descending; never filter rows ──
        out = out.sort_values("Prediction Score", ascending=False).reset_index(drop=True)

        return out

    except Exception:
        # Absolute fail-safe: return original df unchanged, never crash.
        return df