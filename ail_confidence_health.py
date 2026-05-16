"""
Confidence distribution health checks for A-I-L IN ONE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _confidence_series(df: pd.DataFrame) -> pd.Series:
    for col in ("AIL Calibrated Confidence", "AIL Confidence", "Smart Confidence", "Confidence"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def analyze_confidence_distribution(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "mean": 0.0, "spread": 0.0, "high_share": 0.0, "speculative_share": 0.0}
    values = _confidence_series(df)
    if values.empty:
        return {"status": "missing", "mean": 0.0, "spread": 0.0, "high_share": 0.0, "speculative_share": 0.0}
    spread = float(values.quantile(0.85) - values.quantile(0.15)) if len(values) > 2 else float(values.max() - values.min())
    high_share = float(values.ge(74.0).mean() * 100.0)
    speculative_share = float(values.between(45.0, 58.0, inclusive="both").mean() * 100.0)
    status = "compressed" if len(values) >= 4 and spread < 12.0 and high_share < 20.0 else "healthy"
    return {
        "status": status,
        "mean": round(float(values.mean()), 2),
        "spread": round(spread, 2),
        "high_share": round(high_share, 2),
        "speculative_share": round(speculative_share, 2),
    }


def detect_confidence_compression(df: pd.DataFrame) -> bool:
    return analyze_confidence_distribution(df).get("status") == "compressed"


def preserve_high_conviction(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    base = _confidence_series(out)
    if base.empty or len(base) < 4:
        out["AIL Confidence Health Boost"] = 0.0
        return out
    health = analyze_confidence_distribution(out)
    boosts: list[float] = []
    for _, row in out.iterrows():
        confidence = _safe_float(row.get("AIL Calibrated Confidence", row.get("AIL Confidence", 0.0)), 0.0)
        opportunity = _safe_float(row.get("AIL Opportunity Score", 0.0), 0.0)
        master = _safe_float(row.get("AIL Master Score", row.get("Smart Potential Score", 0.0)), 0.0)
        boost = 0.0
        if health["status"] == "compressed" and (opportunity >= 40.0 or master >= 72.0):
            boost = min(4.0, max(0.0, 74.0 - confidence) * 0.12)
        boosts.append(round(boost, 2))
    out["AIL Confidence Health Boost"] = boosts
    return out


def preserve_speculative_conviction(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "AIL Confidence Health Boost" not in out.columns:
        out["AIL Confidence Health Boost"] = 0.0
    spec_boosts: list[float] = []
    for _, row in out.iterrows():
        speculative = _safe_float(row.get("AIL Speculative Score", 0.0), 0.0)
        conflict = _safe_float(row.get("AIL Conflict Score", 0.0), 0.0)
        boost = min(3.0, speculative * 0.04) if speculative >= 34.0 and conflict < 55.0 else 0.0
        spec_boosts.append(round(boost, 2))
    out["AIL Speculative Confidence Boost"] = spec_boosts
    out["AIL Confidence Health Boost"] = (
        pd.to_numeric(out["AIL Confidence Health Boost"], errors="coerce").fillna(0.0)
        + pd.Series(spec_boosts, index=out.index)
    ).round(2)
    return out


__all__ = [
    "analyze_confidence_distribution",
    "detect_confidence_compression",
    "preserve_high_conviction",
    "preserve_speculative_conviction",
]
