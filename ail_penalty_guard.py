"""
Penalty guardrails for A-I-L IN ONE.

The guard caps cumulative orchestration suppression so scanner conviction
remains dominant and high-upside setups stay visible.
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def compute_safe_modifier_bounds(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    scanner = _safe_float(row.get("Smart Potential Score", row.get("Final Score", row.get("Prediction Score", 0.0))), 0.0)
    opportunity = _safe_float(row.get("AIL Opportunity Score"), 0.0)
    philosophy = _safe_float(row.get("AIL Philosophy Score"), 55.0)
    floor_drop = 16.0
    if scanner >= 76.0:
        floor_drop = 10.0
    if opportunity >= 45.0:
        floor_drop = min(floor_drop, 8.0)
    if philosophy >= 68.0:
        floor_drop = min(floor_drop, 10.0)
    return {
        "min_score": round(max(0.0, scanner - floor_drop), 2),
        "max_penalty": round(floor_drop, 2),
        "scanner_score": round(scanner, 2),
    }


def cap_total_penalty(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    guarded_scores: list[float] = []
    caps: list[float] = []
    indexes: list[float] = []
    notes: list[str] = []
    for _, row in out.iterrows():
        bounds = compute_safe_modifier_bounds(row)
        current = _safe_float(row.get("AIL Master Score", row.get("Smart Potential Score", 0.0)), 0.0)
        scanner = bounds["scanner_score"]
        min_score = bounds["min_score"]
        capped = max(current, min_score)
        raw_positive_boost = (
            _safe_float(row.get("AIL Opportunity Boost"), 0.0)
            + _safe_float(row.get("AIL Philosophy Boost"), 0.0)
            + _safe_float(row.get("AIL Confidence Health Boost"), 0.0)
        )
        boost_ledger = str(row.get("AIL Boost Ledger", "") or "").lower()
        boosts_already_applied = _truthy(row.get("AIL Boosts Applied In Master")) or "master" in boost_ledger
        positive_boost = 0.0 if boosts_already_applied else raw_positive_boost
        capped = _clip(capped + positive_boost)
        raw_penalty_index = max(0.0, scanner - current)
        penalty_index = max(0.0, scanner - capped)
        guarded_scores.append(round(capped, 2))
        caps.append(bounds["max_penalty"])
        indexes.append(round(penalty_index, 2))
        if current < min_score:
            notes.append(f"Penalty capped to preserve scanner conviction near {scanner:.1f}")
        elif boosts_already_applied and raw_positive_boost > 0:
            notes.append("Positive boosts already applied in master score")
        elif positive_boost > 0:
            notes.append(f"Opportunity/philosophy boost {positive_boost:.1f}")
        else:
            notes.append("Within safe orchestration bounds")
    out["AIL Unguarded Master Score"] = out.get("AIL Master Score", pd.Series(guarded_scores, index=out.index))
    out["AIL Master Score"] = guarded_scores
    out["AIL Max Allowed Penalty"] = caps
    out["AIL Suppression Index"] = indexes
    out["AIL Raw Suppression Index"] = [
        round(max(0.0, _safe_float(row.get("Smart Potential Score", row.get("Final Score", 0.0)), 0.0) - _safe_float(row.get("AIL Unguarded Master Score"), 0.0)), 2)
        for _, row in out.iterrows()
    ]
    out["AIL Penalty Guard Notes"] = notes
    return out


def prevent_confidence_collapse(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "AIL Confidence" not in out.columns and "AIL Calibrated Confidence" not in out.columns:
        return out
    conf_col = "AIL Calibrated Confidence" if "AIL Calibrated Confidence" in out.columns else "AIL Confidence"
    adjusted: list[float] = []
    notes: list[str] = []
    for _, row in out.iterrows():
        confidence = _safe_float(row.get(conf_col), 0.0)
        scanner = _safe_float(row.get("Smart Potential Score", row.get("Final Score", 0.0)), 0.0)
        opportunity = _safe_float(row.get("AIL Opportunity Score"), 0.0)
        floor = 0.0
        if scanner >= 76.0:
            floor = 62.0
        if opportunity >= 45.0:
            floor = max(floor, 58.0)
        new_conf = max(confidence, floor) if floor else confidence
        adjusted.append(round(_clip(new_conf), 2))
        notes.append("confidence floor protected" if new_conf > confidence else "confidence unchanged")
    out[conf_col] = adjusted
    if conf_col != "AIL Confidence":
        out["AIL Confidence"] = adjusted
    out["AIL Confidence Guard Notes"] = notes
    return out


def detect_over_suppression(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "avg_suppression": 0.0, "over_suppressed_share": 0.0}
    suppression = pd.to_numeric(df.get("AIL Suppression Index", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    share = float(suppression.gt(12.0).mean() * 100.0) if len(suppression) else 0.0
    avg = float(suppression.mean()) if len(suppression) else 0.0
    status = "over_suppressed" if share >= 35.0 or avg >= 10.0 else "healthy"
    return {"status": status, "avg_suppression": round(avg, 2), "over_suppressed_share": round(share, 2)}


__all__ = [
    "cap_total_penalty",
    "compute_safe_modifier_bounds",
    "prevent_confidence_collapse",
    "detect_over_suppression",
]
