"""
Metric-backed reasoning text for A-I-L IN ONE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
            return _safe_float(value, default)
    return default


def summarize_alignment(row: pd.Series | dict[str, Any]) -> str:
    parts: list[str] = []
    agreement = _num(row, "AIL Agreement Score", default=0.0)
    temporal = _num(row, "AIL Temporal Fit", default=0.0)
    regime = _num(row, "AIL Regime Strategy Fit", "AIL Regime Alignment", default=0.0)
    sector = _num(row, "Sector Support", "Sector Strength", default=0.0)
    if agreement >= 68:
        parts.append("mode agreement is strong")
    elif agreement > 0 and agreement < 52:
        parts.append("mode agreement is mixed")
    if temporal >= 66:
        parts.append("market-session timing supports the setup")
    elif temporal > 0 and temporal < 50:
        parts.append("market-session timing is cautious")
    if regime >= 66:
        parts.append("regime preference supports the category")
    elif regime > 0 and regime < 50:
        parts.append("regime fit is weak")
    if sector >= 65:
        parts.append("sector support is present")
    return "; ".join(parts)


def summarize_risk_profile(row: pd.Series | dict[str, Any]) -> str:
    parts: list[str] = []
    trap = _num(row, "Trap Risk Score", default=50.0)
    conflict = _num(row, "AIL Conflict Score", default=0.0)
    confidence = _num(row, "AIL Calibrated Confidence", "AIL Confidence", default=0.0)
    if trap >= 68:
        parts.append("high trap risk")
    elif trap <= 42:
        parts.append("trap risk controlled")
    if conflict >= 45:
        parts.append("cross-mode conflicts need confirmation")
    elif conflict > 0 and conflict < 20:
        parts.append("few cross-mode conflicts")
    if confidence >= 70:
        parts.append("calibrated confidence is high")
    elif confidence > 0 and confidence < 52:
        parts.append("confidence is not yet strong")
    return "; ".join(parts)


def build_orchestration_reasoning(row: pd.Series | dict[str, Any]) -> str:
    notes: list[str] = []
    alignment = summarize_alignment(row)
    risk = summarize_risk_profile(row)
    conflict_notes = str(_get(row, "AIL Conflict Notes") or "")
    align_notes = str(_get(row, "AIL Alignment Notes") or "")
    confidence_drivers = str(_get(row, "AIL Confidence Drivers") or "")
    temporal_notes = str(_get(row, "AIL Temporal Notes") or "")
    learning = str(_get(row, "AIL Learning Drivers") or "")
    for item in (alignment, align_notes, risk, conflict_notes, temporal_notes, confidence_drivers, learning):
        if item and item.lower() not in {"no major mode conflict", "no special cross-mode alignment"}:
            notes.append(item)
    if not notes:
        fallback = str(_get(row, "AIL Reasoning", "Smart Notes", "Battle Notes") or "")
        if fallback:
            notes.append(fallback)
    return "; ".join(dict.fromkeys(notes[:5])) or "Ranked from available scanner, risk, regime, and confidence metrics"


def apply_orchestration_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["AIL Orchestration Reasoning"] = [build_orchestration_reasoning(row) for _, row in out.iterrows()]
    out["AIL Reasoning"] = out["AIL Orchestration Reasoning"]
    return out


__all__ = [
    "build_orchestration_reasoning",
    "summarize_alignment",
    "summarize_risk_profile",
    "apply_orchestration_reasoning",
]
