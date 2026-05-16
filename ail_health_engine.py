"""
Health diagnostics for A-I-L IN ONE orchestration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ail_confidence_health import analyze_confidence_distribution
from ail_penalty_guard import detect_over_suppression
from ail_philosophy_guard import detect_philosophy_flattening


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def detect_confidence_inflation(df: pd.DataFrame, calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "avg_confidence": 0.0, "high_confidence_share": 0.0}
    confidence = pd.to_numeric(df.get("AIL Calibrated Confidence", df.get("AIL Confidence", pd.Series(0.0, index=df.index))), errors="coerce").fillna(0.0)
    avg_conf = float(confidence.mean()) if len(confidence) else 0.0
    high_share = float(confidence.ge(78.0).mean() * 100.0) if len(confidence) else 0.0
    drift = calibration.get("drift", {}) if isinstance(calibration, dict) else {}
    drift_status = str(drift.get("status", "") or "")
    status = "watch"
    if high_share >= 65.0 and drift_status == "drift":
        status = "inflated"
    elif avg_conf < 45.0:
        status = "low"
    elif high_share < 65.0:
        status = "stable"
    return {"status": status, "avg_confidence": round(avg_conf, 2), "high_confidence_share": round(high_share, 2)}


def detect_ranking_collapse(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "unique_symbols": 0, "top_sector_share": 0.0}
    symbols = df.get("Symbol", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.replace(".NS", "", regex=False)
    unique_symbols = int(symbols.nunique())
    sectors = df.get("Sector", pd.Series("UNMAPPED", index=df.index)).fillna("UNMAPPED").astype(str)
    top_sector_share = float(sectors.value_counts(normalize=True).iloc[0] * 100.0) if len(sectors) else 0.0
    status = "collapsed" if unique_symbols <= 1 and len(df) > 1 else "concentrated" if top_sector_share >= 70.0 and len(df) >= 4 else "healthy"
    return {"status": status, "unique_symbols": unique_symbols, "top_sector_share": round(top_sector_share, 2)}


def _opportunity_preservation(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "opportunity_share": 0.0, "avg_opportunity": 0.0}
    opp = pd.to_numeric(df.get("AIL Opportunity Score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    share = float(opp.ge(34.0).mean() * 100.0) if len(opp) else 0.0
    avg = float(opp.mean()) if len(opp) else 0.0
    status = "collapsed" if len(opp) >= 4 and share < 8.0 else "healthy"
    return {"status": status, "opportunity_share": round(share, 2), "avg_opportunity": round(avg, 2)}


def _ranking_flattening(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "no_candidates", "spread": 0.0}
    score = pd.to_numeric(df.get("AIL Master Score", pd.Series(0.0, index=df.index)), errors="coerce").dropna()
    if len(score) < 3:
        return {"status": "insufficient_candidates", "spread": float(score.max() - score.min()) if len(score) else 0.0}
    spread = float(score.quantile(0.85) - score.quantile(0.15))
    return {"status": "flat" if spread < 8.0 else "healthy", "spread": round(spread, 2)}


def detect_stale_market_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {"status": "unknown", "reason": "No market state available"}
    label = str(state.get("state", "") or "").upper()
    source = str(state.get("source_label", "") or "")
    if label in {"PRE_MARKET", "WEEKEND", "POST_CLOSE"} and not state.get("use_snapshot") and not state.get("snapshot_saved"):
        return {"status": "watch", "reason": f"{label} without loaded/saved snapshot"}
    if label == "LIVE" and state.get("use_snapshot"):
        return {"status": "stale", "reason": "Live state is using snapshot"}
    return {"status": "fresh", "reason": source or label}


def compute_orchestration_health(
    result: Any,
    state: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    conflict_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    final_df = getattr(result, "final_ranked_df", pd.DataFrame())
    health = dict(getattr(result, "health", {}) or {})
    conflict_source = conflict_df if isinstance(conflict_df, pd.DataFrame) and not conflict_df.empty else final_df
    conflicts = pd.to_numeric(conflict_source.get("AIL Conflict Score", pd.Series(dtype=float)), errors="coerce").dropna()
    agreement = pd.to_numeric(conflict_source.get("AIL Agreement Score", pd.Series(dtype=float)), errors="coerce").dropna()
    calibration_drift = calibration.get("drift", {}) if isinstance(calibration, dict) else {}
    confidence = detect_confidence_inflation(final_df, calibration)
    confidence_distribution = analyze_confidence_distribution(final_df)
    collapse = detect_ranking_collapse(final_df)
    suppression = detect_over_suppression(final_df)
    philosophy = detect_philosophy_flattening(final_df)
    opportunity = _opportunity_preservation(final_df)
    flattening = _ranking_flattening(final_df)
    stale = detect_stale_market_state(state)
    flags: list[str] = []
    if len(conflicts) and float(conflicts.mean()) >= 35.0:
        flags.append("mode disagreement elevated")
    if confidence["status"] in {"inflated", "low"}:
        flags.append(f"confidence {confidence['status']}")
    if collapse["status"] != "healthy":
        flags.append(f"ranking {collapse['status']}")
    if suppression["status"] != "healthy":
        flags.append("excessive penalty stacking")
    if confidence_distribution["status"] == "compressed":
        flags.append("confidence compression")
    if philosophy["status"] == "flattening":
        flags.append("philosophy flattening")
    if opportunity["status"] == "collapsed":
        flags.append("opportunity collapse")
    if flattening["status"] == "flat":
        flags.append("ranking flattening")
    if stale["status"] != "fresh":
        flags.append(f"market state {stale['status']}")
    if str(calibration_drift.get("status", "")) == "drift":
        flags.append("calibration drift")
    health.update(
        {
            "market_state": str((state or {}).get("state", "")),
            "runtime_sec": _safe_float(getattr(result, "elapsed_sec", 0.0), 0.0),
            "avg_conflict_score": round(float(conflicts.mean()), 2) if len(conflicts) else 0.0,
            "avg_agreement_score": round(float(agreement.mean()), 2) if len(agreement) else 0.0,
            "calibration_status": str(calibration_drift.get("status", "insufficient_data")),
            "calibration_max_gap": _safe_float(calibration_drift.get("max_gap"), 0.0),
            "confidence_health": confidence,
            "confidence_distribution_health": confidence_distribution,
            "ranking_diversity_health": collapse,
            "orchestration_suppression": suppression,
            "philosophy_integrity": philosophy,
            "opportunity_preservation": opportunity,
            "ranking_flattening": flattening,
            "market_state_health": stale,
            "AIL Health Flags": "; ".join(flags) if flags else "healthy",
        }
    )
    return health


__all__ = [
    "compute_orchestration_health",
    "detect_confidence_inflation",
    "detect_ranking_collapse",
    "detect_stale_market_state",
]
