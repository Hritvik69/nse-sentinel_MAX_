"""
sector_mtf_engine.py
═════════════════════
Module 2 — Multi-Timeframe Confirmation.

Computes alignment between:
  • Long-term view  (20-bar daily lookback — same as main signals)
  • Short-term view (5-bar daily lookback — proxy for intraday momentum)

When both timeframes agree → alignment_score high → confidence boosted.
When they disagree         → alignment_score low  → confidence penalised.

Public API
──────────
  compute_mtf_alignment(ohlc) → MTFAlignment
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MTFAlignment:
    alignment_score: float = 50.0   # 0–100; 50 = neutral
    short_bias:      str   = "NEUTRAL"   # BULLISH | BEARISH | NEUTRAL
    long_bias:       str   = "NEUTRAL"
    agreement:       bool  = False
    short_score:     float = 50.0
    long_score:      float = 50.0
    note:            str   = ""


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _norm(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 50.0
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))


def _bias_from_score(score: float) -> str:
    if score >= 60:
        return "BULLISH"
    if score <= 40:
        return "BEARISH"
    return "NEUTRAL"


# ── Short-term view (5-bar) ───────────────────────────────────────────

def _short_term_score(ohlc: pd.DataFrame) -> float:
    """
    5-bar lookback features:
      • Net candle direction (bull/bear fraction)
      • EMA5 slope
      • Volume trend (last 2 bars vs prior 3)
    Returns 0–100.
    """
    if len(ohlc) < 8:
        return 50.0

    last5 = ohlc.tail(5)
    close = ohlc["Close"]
    e5    = _ema(close, 5)

    # Candle direction
    bulls  = (last5["Close"] >= last5["Open"]).sum()
    dir_s  = float(bulls / len(last5) * 100)

    # EMA5 slope
    slope  = (e5.iloc[-1] - e5.iloc[-4]) / (e5.iloc[-4] + 1e-9) * 100
    slope_s = _norm(slope, -2.5, 2.5)

    # Volume expansion (last 2 bars vs prev 3)
    v_recent = ohlc["Volume"].iloc[-2:].mean()
    v_prior  = ohlc["Volume"].iloc[-5:-2].mean()
    vol_ratio = v_recent / (v_prior + 1e-9)
    vol_s    = _norm(vol_ratio - 1, -0.5, 0.5)

    return float(np.clip(dir_s * 0.40 + slope_s * 0.35 + vol_s * 0.25, 0, 100))


# ── Long-term view (20-bar) ───────────────────────────────────────────

def _long_term_score(ohlc: pd.DataFrame) -> float:
    """
    20-bar lookback features:
      • EMA20 vs EMA50 relationship
      • 20-day momentum (close / close[−20] − 1)
      • Average candle direction (bullish fraction)
    Returns 0–100.
    """
    if len(ohlc) < 55:
        return 50.0

    close = ohlc["Close"]
    e20   = _ema(close, 20)
    e50   = _ema(close, 50)

    # EMA cross
    sep   = (e20.iloc[-1] - e50.iloc[-1]) / (close.iloc[-1] + 1e-9) * 100
    ema_s = _norm(sep, -4.0, 4.0)

    # 20-day momentum
    mom   = (close.iloc[-1] / (close.iloc[-21] + 1e-9) - 1) * 100
    mom_s = _norm(mom, -8.0, 8.0)

    # Average direction
    last20 = ohlc.tail(20)
    bulls  = (last20["Close"] >= last20["Open"]).mean() * 100
    dir_s  = float(bulls)

    return float(np.clip(ema_s * 0.40 + mom_s * 0.35 + dir_s * 0.25, 0, 100))


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def compute_mtf_alignment(ohlc: pd.DataFrame) -> MTFAlignment:
    """
    Compute multi-timeframe alignment from sector OHLC.

    Parameters
    ----------
    ohlc : pd.DataFrame  Sector OHLC (aggregated synthetic or index)

    Returns
    -------
    MTFAlignment
    """
    if ohlc is None or len(ohlc) < 8:
        return MTFAlignment(note="Insufficient data for MTF analysis.")

    try:
        short_s = _short_term_score(ohlc)
        long_s  = _long_term_score(ohlc)

        short_bias = _bias_from_score(short_s)
        long_bias  = _bias_from_score(long_s)

        # Agreement = both biases are NOT opposing
        agree = not (
            (short_bias == "BULLISH" and long_bias == "BEARISH") or
            (short_bias == "BEARISH" and long_bias == "BULLISH")
        )

        # Alignment score:
        # Perfect agreement + same direction → high score
        # Disagreement (bull vs bear)       → pulled toward 50
        if agree:
            alignment = float(short_s * 0.45 + long_s * 0.55)
            if short_bias == long_bias and short_bias != "NEUTRAL":
                # Same non-neutral direction → boost
                direction_agreement = abs(short_s - 50) + abs(long_s - 50)
                alignment = float(np.clip(alignment + direction_agreement * 0.10, 0, 100))
        else:
            # Disagreement: blend toward 50
            alignment = float(50.0 + (long_s - 50) * 0.35)

        note_parts = []
        if agree and short_bias == long_bias and short_bias != "NEUTRAL":
            note_parts.append(f"Both timeframes {short_bias} — strong confirmation.")
        elif not agree:
            note_parts.append(
                f"Timeframe conflict: short-term {short_bias} vs long-term {long_bias}. Confidence reduced."
            )
        elif short_bias == "NEUTRAL" or long_bias == "NEUTRAL":
            note_parts.append("One timeframe is neutral — mixed signal.")

        return MTFAlignment(
            alignment_score = round(float(np.clip(alignment, 0, 100)), 1),
            short_bias      = short_bias,
            long_bias       = long_bias,
            agreement       = agree,
            short_score     = round(short_s, 1),
            long_score      = round(long_s, 1),
            note            = " ".join(note_parts),
        )
    except Exception as exc:
        return MTFAlignment(note=f"MTF error: {exc}")