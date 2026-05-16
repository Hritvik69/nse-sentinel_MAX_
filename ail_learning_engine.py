"""
Controlled learning profile for A-I-L IN ONE.

This module reads the existing prediction feedback store and converts observed
outcomes into soft reliability inputs.  It never self-modifies code and never
overwrites scanner weights; the orchestration layer uses these scores as small
calibration signals during ranking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
    outcome = str(row.get("outcome_label", "") or "").strip().lower()
    if outcome in {"win", "correct", "hit", "success"}:
        return True
    if outcome in {"loss", "incorrect", "miss", "fail", "failed"}:
        return False
    actual = _safe_float(row.get("actual_next_return_pct"), None)
    pred_bullish = str(row.get("pred_bullish", "") or "").strip().lower()
    direction = str(row.get("prediction_direction", "") or "").strip().lower()
    if actual is None:
        return None
    wants_bull = pred_bullish in {"true", "1", "yes", "bullish"} or "bull" in direction or "buy" in direction
    if wants_bull:
        return actual > 0
    return actual <= 0


def _reliability_score(wins: int, observations: int, *, prior: float = 0.52, strength: int = 8) -> float:
    if observations <= 0:
        return 0.0
    smoothed = (wins + prior * strength) / float(observations + strength)
    confidence_lift = min(1.0, observations / 40.0)
    score = 50.0 + (smoothed - 0.50) * 100.0 * (0.55 + 0.45 * confidence_lift)
    return _clip(score, 35.0, 85.0)


def _build_bucket(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if df.empty or column not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, grp in df.groupby(df[column].fillna("").astype(str).str.strip()):
        if not key:
            continue
        observations = int(len(grp))
        wins = int(grp["_ail_correct_bool"].fillna(False).sum())
        win_rate = 100.0 * float(wins) / float(observations) if observations else 0.0
        out[str(key)] = {
            "observations": observations,
            "wins": wins,
            "win_rate_pct": round(win_rate, 2),
            "score": round(_reliability_score(wins, observations), 2),
        }
    return out


def _calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "prediction_score" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["_score"] = work["prediction_score"].map(lambda value: _safe_float(value, None))
    work = work[work["_score"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    bins = [0, 45, 55, 65, 75, 101]
    labels = ["0-45", "45-55", "55-65", "65-75", "75+"]
    work["_confidence_bin"] = pd.cut(work["_score"].astype(float), bins=bins, labels=labels, right=False)
    rows: list[dict[str, Any]] = []
    for label, grp in work.groupby("_confidence_bin", observed=False):
        if grp.empty:
            continue
        wins = int(grp["_ail_correct_bool"].fillna(False).sum())
        observations = int(len(grp))
        expected = float(grp["_score"].mean())
        actual = 100.0 * wins / observations if observations else 0.0
        rows.append(
            {
                "Confidence Bin": str(label),
                "Observations": observations,
                "Expected %": round(expected, 2),
                "Observed Win %": round(actual, 2),
                "Calibration Gap": round(actual - expected, 2),
            }
        )
    return pd.DataFrame(rows)


def build_ail_learning_profile(feedback_df: pd.DataFrame | None = None) -> dict[str, Any]:
    try:
        if feedback_df is None:
            from prediction_feedback_store import read_feedback_log

            feedback_df = read_feedback_log()
    except Exception:
        feedback_df = pd.DataFrame()

    profile: dict[str, Any] = {
        "total_rows": 0,
        "ail_rows": 0,
        "outcome_rows": 0,
        "mode_reliability": {},
        "category_reliability": {},
        "sector_reliability": {},
        "regime_reliability": {},
        "trap_reliability": {},
        "confidence_calibration": pd.DataFrame(),
        "notes": [],
    }
    if feedback_df is None or not isinstance(feedback_df, pd.DataFrame) or feedback_df.empty:
        profile["notes"].append("No feedback rows available yet.")
        return profile

    df = feedback_df.copy()
    profile["total_rows"] = int(len(df))
    source = df.get("import_source", pd.Series("", index=df.index)).fillna("").astype(str)
    ail_df = df[source.str.upper().eq("A-I-L IN ONE")].copy()
    profile["ail_rows"] = int(len(ail_df))
    work = ail_df if not ail_df.empty else df
    if ail_df.empty:
        profile["notes"].append("No completed A-I-L outcomes yet; using global feedback reliability softly.")

    work["_ail_correct_bool"] = work.apply(_is_correct, axis=1)
    outcome_df = work[work["_ail_correct_bool"].notna()].copy()
    profile["outcome_rows"] = int(len(outcome_df))
    if outcome_df.empty:
        profile["notes"].append("Feedback rows exist, but outcomes are not backfilled yet.")
        return profile

    profile["mode_reliability"] = _build_bucket(outcome_df, "mode")
    profile["category_reliability"] = _build_bucket(outcome_df, "import_category")
    profile["sector_reliability"] = _build_bucket(outcome_df, "sector")
    profile["regime_reliability"] = _build_bucket(outcome_df, "regime")
    profile["trap_reliability"] = _build_bucket(outcome_df, "trap_risk")
    profile["confidence_calibration"] = _calibration_table(outcome_df)

    wins = int(outcome_df["_ail_correct_bool"].fillna(False).sum())
    win_rate = 100.0 * wins / float(len(outcome_df))
    profile["overall_reliability"] = {
        "observations": int(len(outcome_df)),
        "wins": wins,
        "win_rate_pct": round(win_rate, 2),
        "score": round(_reliability_score(wins, int(len(outcome_df))), 2),
    }
    return profile


def learning_adjustment_for_row(row: pd.Series | dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {"score": None, "multiplier": 1.0, "drivers": ""}
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    candidates: list[tuple[str, float]] = []

    def pull(bucket: str, key: Any, label: str) -> None:
        section = profile.get(bucket, {})
        if not isinstance(section, dict):
            return
        for candidate in (key, str(key), str(key).upper(), str(key).title()):
            item = section.get(candidate)
            if isinstance(item, dict):
                score = _safe_float(item.get("score"), None)
                if score is not None:
                    candidates.append((label, _clip(score)))
                    return

    pull("mode_reliability", getter("Mode ID", getter("Mode", "")), "mode")
    pull("sector_reliability", getter("Sector", ""), "sector")
    pull("regime_reliability", getter("Market Regime", getter("Regime", "")), "regime")
    pull("trap_reliability", getter("Trap Risk", ""), "trap history")
    for category in str(getter("AIL Categories", getter("AIL Category", "")) or "").replace("|", ",").split(","):
        pull("category_reliability", category.strip(), "category")

    if not candidates:
        overall = profile.get("overall_reliability", {})
        if isinstance(overall, dict) and _safe_float(overall.get("score"), None) is not None:
            candidates.append(("overall", _clip(float(overall["score"]))))
    if not candidates:
        return {"score": None, "multiplier": 1.0, "drivers": ""}

    score = float(np.mean([value for _, value in candidates]))
    multiplier = float(np.clip(0.94 + (score / 100.0) * 0.12, 0.94, 1.06))
    drivers = "; ".join(f"{label} {value:.1f}" for label, value in candidates[:4])
    return {"score": round(score, 2), "multiplier": round(multiplier, 4), "drivers": drivers}


def learning_profile_table(profile: dict[str, Any] | None) -> pd.DataFrame:
    if not isinstance(profile, dict):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bucket, label in (
        ("mode_reliability", "Mode"),
        ("category_reliability", "Category"),
        ("sector_reliability", "Sector"),
        ("regime_reliability", "Regime"),
        ("trap_reliability", "Trap Risk"),
    ):
        section = profile.get(bucket, {})
        if not isinstance(section, dict):
            continue
        for name, item in section.items():
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "Layer": label,
                    "Name": name,
                    "Observations": int(item.get("observations", 0) or 0),
                    "Win Rate %": item.get("win_rate_pct", 0.0),
                    "Reliability Score": item.get("score", 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Layer", "Reliability Score", "Observations"], ascending=[True, False, False])


__all__ = [
    "build_ail_learning_profile",
    "learning_adjustment_for_row",
    "learning_profile_table",
]
