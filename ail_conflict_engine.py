"""
Cross-mode conflict resolution for A-I-L IN ONE.

Conflicts are interpreted as soft confidence and ranking context.  Rows are not
hard rejected here.
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


def _symbol(row: pd.Series | dict[str, Any]) -> str:
    value = str(_get(row, "Symbol", "Ticker", "symbol", "ticker", "Stock") or "").strip().upper()
    return value[:-3] if value.endswith(".NS") else value


def _categories(row: pd.Series | dict[str, Any]) -> set[str]:
    raw = _text(row, "AIL Categories", "AIL Category")
    return {part.strip().upper() for part in raw.replace("|", ",").split(",") if part.strip()}


def compute_mode_agreement(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["Symbol", "AIL Agreement Score", "AIL Mode Spread"])
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cats = _categories(row)
        mode_count = _safe_float(_get(row, "AIL Mode Count"), 0.0)
        if mode_count <= 0:
            mode_count = float(max(1, len(cats)))
        agreement = 48.0 + min(24.0, max(0.0, mode_count - 1.0) * 8.0) + min(16.0, max(0, len(cats) - 1) * 4.0)
        if {"RELAXED", "INSTITUTIONAL", "BREAKOUT"} <= cats:
            agreement += 8.0
        if {"SWING", "INSTITUTIONAL"} <= cats:
            agreement += 5.0
        rows.append({"Symbol": _symbol(row), "AIL Agreement Score": round(_clip(agreement), 2), "AIL Mode Spread": int(mode_count)})
    return pd.DataFrame(rows)


def detect_signal_conflicts(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    cats = _categories(row)
    momentum = _num(row, "Momentum Quality", "Bullish Probability", "Prediction Score", default=50.0)
    volume = _num(row, "Volume Quality", default=50.0)
    setup = _num(row, "Setup Cleanliness", "Setup Quality", default=50.0)
    structure = _num(row, "Structure Quality", "AIL Risk Adjusted Score", default=setup)
    trap = _num(row, "Trap Risk Score", default=50.0)
    regime = _num(row, "AIL Regime Alignment", "Regime Alignment", default=50.0)
    sector = _num(row, "Sector Support", "Sector Strength", default=50.0)
    rsi = _safe_float(_get(row, "RSI"), 50.0)
    dist_ema = _safe_float(_get(row, "Delta vs EMA20 (%)", "Δ vs EMA20 (%)"), 0.0)
    bullish = _num(row, "Bullish Probability", "Prediction Score", default=50.0)
    conflicts: list[str] = []
    alignments: list[str] = []
    opportunity = _num(row, "AIL Opportunity Score", "AIL Speculative Score", default=0.0)

    if momentum >= 74.0 and structure < 54.0:
        conflicts.append("momentum ahead of structure")
    if momentum >= 72.0 and (rsi > 72.0 or trap >= 62.0):
        conflicts.append("momentum with exhaustion risk")
    if "BREAKOUT" in cats and volume < 56.0:
        conflicts.append("breakout lacks volume confirmation")
    if bullish >= 66.0 and trap >= 60.0:
        conflicts.append("bullish score conflicts with trap risk")
    if "SWING" in cats and dist_ema > 7.0:
        conflicts.append("swing setup extended above EMA20")
    if "INSTITUTIONAL" in cats and regime < 52.0:
        conflicts.append("institutional setup lacks regime support")

    if {"RELAXED", "INSTITUTIONAL", "BREAKOUT"} <= cats and trap < 50.0 and structure >= 58.0:
        alignments.append("accumulation and institutional breakout alignment")
    if "SWING" in cats and trap < 45.0 and setup >= 64.0:
        alignments.append("durable swing with controlled trap risk")
    if "MOMENTUM" in cats and momentum >= 68.0 and volume >= 62.0 and structure >= 58.0:
        alignments.append("momentum supported by volume and structure")
    if sector >= 65.0 and regime >= 62.0:
        alignments.append("sector and regime aligned")

    conflict_score = min(100.0, len(conflicts) * 14.0 + max(0.0, trap - 62.0) * 0.28)
    if opportunity >= 42.0 and conflict_score < 55.0:
        conflict_score *= 0.78
        alignments.append("asymmetric opportunity keeps conflict soft")
    alignment_bonus = min(30.0, len(alignments) * 8.0)
    agreement = _clip(62.0 + alignment_bonus - conflict_score * 0.35)
    return {
        "agreement": round(agreement, 2),
        "conflict_score": round(_clip(conflict_score), 2),
        "conflicts": conflicts,
        "alignments": alignments,
    }


def compute_alignment_quality(row: pd.Series | dict[str, Any]) -> float:
    return float(detect_signal_conflicts(row).get("agreement", 0.0))


def apply_conflict_penalties(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    agreement_map = compute_mode_agreement(out)
    if not agreement_map.empty and "Symbol" in agreement_map.columns:
        out["_AIL_SYMBOL_KEY"] = out.apply(_symbol, axis=1)
        out = out.merge(agreement_map, how="left", left_on="_AIL_SYMBOL_KEY", right_on="Symbol", suffixes=("", " Agreement"))
        out = out.drop(columns=["_AIL_SYMBOL_KEY", "Symbol Agreement"], errors="ignore")
    agreements: list[float] = []
    conflicts: list[float] = []
    notes: list[str] = []
    align_notes: list[str] = []
    penalties: list[float] = []
    for _, row in out.iterrows():
        signal = detect_signal_conflicts(row)
        base_agreement = _safe_float(row.get("AIL Agreement Score"), 55.0)
        agreement = _clip((base_agreement * 0.45) + (float(signal["agreement"]) * 0.55))
        conflict_score = float(signal["conflict_score"])
        opportunity = _safe_float(row.get("AIL Opportunity Score"), 0.0)
        penalty_cap = 7.0 if opportunity >= 42.0 else 9.0
        penalty = round(min(penalty_cap, conflict_score * 0.09), 2)
        agreements.append(round(agreement, 2))
        conflicts.append(round(conflict_score, 2))
        notes.append("; ".join(signal["conflicts"]) or "No major mode conflict")
        align_notes.append("; ".join(signal["alignments"]) or "No special cross-mode alignment")
        penalties.append(penalty)
    out["AIL Agreement Score"] = agreements
    out["AIL Conflict Score"] = conflicts
    out["AIL Conflict Notes"] = notes
    out["AIL Alignment Notes"] = align_notes
    out["AIL Conflict Penalty"] = penalties
    out["AIL Conflict Multiplier"] = [round(float(np.clip(1.02 - p / 100.0, 0.90, 1.03)), 4) for p in penalties]
    if "AIL Confidence" in out.columns:
        out["AIL Confidence Before Conflict"] = pd.to_numeric(out["AIL Confidence"], errors="coerce")
        out["AIL Confidence"] = (out["AIL Confidence Before Conflict"] - out["AIL Conflict Penalty"]).clip(lower=0, upper=100)
    return out


__all__ = [
    "compute_mode_agreement",
    "detect_signal_conflicts",
    "compute_alignment_quality",
    "apply_conflict_penalties",
]
