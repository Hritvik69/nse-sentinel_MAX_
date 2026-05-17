"""
Probabilistic confidence calibration for A-I-L IN ONE.

Calibration is based only on logged outcomes.  When there are not enough
outcomes, the current confidence is preserved and marked uncalibrated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_BUCKETS = [0, 45, 55, 65, 75, 85, 101]
_LABELS = ["0-45", "45-55", "55-65", "65-75", "75-85", "85+"]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "").replace(",", "")
            if cleaned.lower() in {"", "nan", "none", "null", "-", "n/a", "na"}:
                return default
            value = cleaned
        out = float(value)
        return float(out) if np.isfinite(out) else default
    except Exception:
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return lo


def _is_correct(row: pd.Series) -> bool | None:
    correct = str(row.get("correct", "") or "").strip().lower()
    if correct in {"true", "1", "yes", "y"}:
        return True
    if correct in {"false", "0", "no", "n"}:
        return False
    actual = _safe_float(row.get("actual_next_return_pct"), None)
    if actual is None:
        return None
    direction = str(row.get("prediction_direction", "") or "").strip().lower()
    pred_bullish = str(row.get("pred_bullish", "") or "").strip().lower()
    wants_bull = pred_bullish in {"true", "1", "yes", "bullish"} or "bull" in direction or "buy" in direction
    return actual > 0 if wants_bull else actual <= 0


def _read_feedback() -> pd.DataFrame:
    try:
        from prediction_feedback_store import read_feedback_log

        return read_feedback_log()
    except Exception:
        return pd.DataFrame()


def build_confidence_buckets(feedback_df: pd.DataFrame | None = None, *, min_rows: int = 5) -> pd.DataFrame:
    df = _read_feedback() if feedback_df is None else feedback_df
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["Confidence Bin", "Observations", "Expected %", "Observed Win %", "Calibration Gap", "Usable"])
    work = df.copy()
    source = work.get("import_source", pd.Series("", index=work.index)).fillna("").astype(str).str.upper()
    ail_rows = work[source.eq("A-I-L IN ONE")].copy()
    if not ail_rows.empty:
        work = ail_rows
    work["_ail_score"] = work.get("ail_confidence", work.get("prediction_score", pd.Series(np.nan, index=work.index))).map(lambda value: _safe_float(value, None))
    work["_ail_correct"] = work.apply(_is_correct, axis=1)
    work = work[work["_ail_score"].notna() & work["_ail_correct"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["Confidence Bin", "Observations", "Expected %", "Observed Win %", "Calibration Gap", "Usable"])
    work["_bucket"] = pd.cut(work["_ail_score"].astype(float), bins=_BUCKETS, labels=_LABELS, right=False)
    rows: list[dict[str, Any]] = []
    for label, grp in work.groupby("_bucket", observed=False):
        if grp.empty:
            continue
        observations = int(len(grp))
        expected = float(grp["_ail_score"].mean())
        observed = 100.0 * float(grp["_ail_correct"].astype(bool).sum()) / float(observations)
        rows.append(
            {
                "Confidence Bin": str(label),
                "Observations": observations,
                "Expected %": round(expected, 2),
                "Observed Win %": round(observed, 2),
                "Calibration Gap": round(observed - expected, 2),
                "Usable": observations >= int(min_rows),
            }
        )
    return pd.DataFrame(rows)


def _filter_regime_rows(df: pd.DataFrame, regime: str | None) -> pd.DataFrame:
    if not regime or df is None or df.empty or "regime" not in df.columns:
        return df
    target = str(regime or "").strip().upper()
    if not target:
        return df
    matched = df[df["regime"].fillna("").astype(str).str.upper().str.contains(target[:6], regex=False)].copy()
    return matched if len(matched) >= 10 else df


def compute_confidence_calibration(
    feedback_df: pd.DataFrame | None = None,
    *,
    min_rows: int = 5,
    regime: str | None = None,
    rolling_rows: int = 250,
) -> dict[str, Any]:
    raw_df = _read_feedback() if feedback_df is None else feedback_df
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        raw_df = _filter_regime_rows(raw_df.tail(max(50, int(rolling_rows))), regime)
    buckets = build_confidence_buckets(raw_df, min_rows=min_rows)
    usable_rows = int(buckets.loc[buckets.get("Usable", pd.Series(False, index=buckets.index)).astype(bool), "Observations"].sum()) if not buckets.empty else 0
    return {
        "buckets": buckets,
        "usable_rows": usable_rows,
        "enough_data": usable_rows >= min_rows,
        "drift": detect_calibration_drift(buckets, min_rows=min_rows),
        "regime": regime or "",
    }


def _bucket_for_score(score: float, buckets: pd.DataFrame) -> pd.Series | None:
    if buckets is None or not isinstance(buckets, pd.DataFrame) or buckets.empty:
        return None
    labels = pd.cut(pd.Series([score]), bins=_BUCKETS, labels=_LABELS, right=False)
    label = str(labels.iloc[0])
    match = buckets[buckets["Confidence Bin"].astype(str).eq(label)]
    if match.empty:
        return None
    return match.iloc[0]


def calibrate_confidence_score(score: float, buckets: pd.DataFrame | None, *, max_adjustment: float = 8.0) -> tuple[float, float, str]:
    score = _clip(score)
    bucket = _bucket_for_score(score, buckets) if isinstance(buckets, pd.DataFrame) else None
    if bucket is None or not bool(bucket.get("Usable", False)):
        return round(score, 2), 0.0, "Calibration pending real outcomes"
    gap = _safe_float(bucket.get("Calibration Gap"), 0.0) or 0.0
    observations = int(_safe_float(bucket.get("Observations"), 0.0) or 0.0)
    reliability = float(np.clip(observations / (observations + 20.0), 0.0, 1.0))
    adjustment = float(np.clip(gap * 0.25 * reliability, -max_adjustment, max_adjustment))
    if adjustment < 0.0 and score >= 76.0:
        adjustment = max(adjustment, -5.0)
    elif adjustment < 0.0:
        adjustment = max(adjustment, -6.5)
    if adjustment > 0.0:
        adjustment = min(adjustment, 6.0)
    calibrated = _clip(score + adjustment)
    return (
        round(calibrated, 2),
        round(adjustment, 2),
        f"Bucket {bucket.get('Confidence Bin')} gap {gap:.1f} pts; {observations} obs shrink {reliability:.2f}",
    )


def detect_calibration_drift(buckets: pd.DataFrame | None, *, min_rows: int = 5) -> dict[str, Any]:
    if buckets is None or not isinstance(buckets, pd.DataFrame) or buckets.empty:
        return {"status": "insufficient_data", "max_gap": 0.0, "drifted_bins": 0}
    usable = buckets[buckets.get("Usable", pd.Series(False, index=buckets.index)).astype(bool)].copy()
    if usable.empty or int(usable["Observations"].sum()) < min_rows:
        return {"status": "insufficient_data", "max_gap": 0.0, "drifted_bins": 0}
    gaps = pd.to_numeric(usable["Calibration Gap"], errors="coerce").fillna(0.0).abs()
    drifted = int(gaps.ge(12.0).sum())
    status = "drift" if drifted else "stable"
    return {"status": status, "max_gap": round(float(gaps.max()), 2), "drifted_bins": drifted}


def apply_confidence_calibration(df: pd.DataFrame, calibration: dict[str, Any] | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    buckets = calibration.get("buckets") if isinstance(calibration, dict) else pd.DataFrame()
    fallback_base = pd.to_numeric(
        out.get("AIL Confidence", out.get("Smart Confidence", out.get("Confidence", pd.Series(0.0, index=out.index)))),
        errors="coerce",
    ).fillna(0.0)
    if "AIL Calibration Base Confidence" in out.columns:
        base = pd.to_numeric(out["AIL Calibration Base Confidence"], errors="coerce").fillna(fallback_base)
    else:
        base = fallback_base
        out["AIL Calibration Base Confidence"] = base
    calibrated: list[float] = []
    adjustments: list[float] = []
    notes: list[str] = []
    for value in base.tolist():
        score, adjustment, note = calibrate_confidence_score(float(value), buckets)
        calibrated.append(score)
        adjustments.append(adjustment)
        notes.append(note)
    out["AIL Calibrated Confidence"] = calibrated
    out["AIL Calibration Adjustment"] = adjustments
    out["AIL Calibration Notes"] = notes
    return out


__all__ = [
    "compute_confidence_calibration",
    "build_confidence_buckets",
    "calibrate_confidence_score",
    "detect_calibration_drift",
    "apply_confidence_calibration",
]
