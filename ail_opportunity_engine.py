"""
Opportunity-preservation layer for A-I-L IN ONE.

This module keeps high-upside, speculative, and early-stage setups visible.
It annotates opportunity quality; it does not hard-accept or hard-reject rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return lo


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "").replace(",", "")
            if cleaned.lower() in {"", "nan", "none", "null", "-", "n/a", "na"}:
                return default
            value = cleaned
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _get(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = getter(key, None)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null", "-", "n/a", "na"}:
            return value
    return None


def _num(row: pd.Series | dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = _get(row, key)
        if value is not None:
            return _clip(_safe_float(value, default))
    return default


def _text(row: pd.Series | dict[str, Any], *keys: str) -> str:
    return str(_get(row, *keys) or "").strip()


def _categories(row: pd.Series | dict[str, Any]) -> set[str]:
    raw = _text(row, "AIL Categories", "AIL Category")
    return {part.strip().upper() for part in raw.replace("|", ",").split(",") if part.strip()}


def detect_asymmetric_opportunity(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    cats = _categories(row)
    momentum = _num(row, "Momentum Quality", "Bullish Probability", "Prediction Score", default=50.0)
    volume = _num(row, "Volume Quality", default=50.0)
    setup = _num(row, "Setup Cleanliness", "Setup Quality", default=50.0)
    risk_reward = _num(row, "Risk Reward Score", "AIL Risk Adjusted Score", default=50.0)
    breakout = _num(row, "Breakout Quality", "Structure Quality", default=setup)
    trap = _num(row, "Trap Risk Score", default=50.0)
    rsi = _safe_float(_get(row, "RSI"), 50.0)
    timing = _text(row, "Entry Timing", "Setup Type").upper()
    score = 0.0
    reasons: list[str] = []

    if {"MOMENTUM", "BREAKOUT"} & cats and momentum >= 72.0 and volume >= 60.0:
        score += 26.0
        reasons.append("breakout or momentum ignition")
    if risk_reward >= 70.0:
        score += 20.0
        reasons.append("strong risk-reward asymmetry")
    if "RELAXED" in cats and (46.0 <= rsi <= 60.0 or "EARLY" in timing or "ACCUM" in timing):
        score += 20.0
        reasons.append("early accumulation visibility")
    if breakout >= 68.0 and volume >= 58.0:
        score += 18.0
        reasons.append("pressure build-up with confirmation")
    if trap <= 45.0:
        score += 10.0
    elif trap >= 70.0:
        score -= 10.0

    return {
        "score": round(_clip(score), 2),
        "is_asymmetric": score >= 34.0,
        "reasons": reasons,
    }


def compute_controlled_speculative_score(row: pd.Series | dict[str, Any]) -> float:
    opp = detect_asymmetric_opportunity(row)
    trap = _num(row, "Trap Risk Score", default=50.0)
    confidence = _num(row, "AIL Calibrated Confidence", "AIL Confidence", "Smart Confidence", default=50.0)
    score = float(opp["score"]) * 0.62 + max(0.0, confidence - 45.0) * 0.22 + max(0.0, 70.0 - trap) * 0.16
    return round(_clip(score), 2)


def preserve_high_upside_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    scores: list[float] = []
    speculative: list[float] = []
    notes: list[str] = []
    floor_boosts: list[float] = []
    for _, row in out.iterrows():
        opp = detect_asymmetric_opportunity(row)
        spec = compute_controlled_speculative_score(row)
        boost = 0.0
        if opp["is_asymmetric"]:
            boost = min(6.0, float(opp["score"]) * 0.08)
        scores.append(float(opp["score"]))
        speculative.append(spec)
        notes.append("; ".join(opp["reasons"]) or "No special asymmetry")
        floor_boosts.append(round(boost, 2))
    out["AIL Opportunity Score"] = scores
    out["AIL Speculative Score"] = speculative
    out["AIL Opportunity Notes"] = notes
    out["AIL Opportunity Boost"] = floor_boosts
    return out


__all__ = [
    "detect_asymmetric_opportunity",
    "preserve_high_upside_candidates",
    "compute_controlled_speculative_score",
]
