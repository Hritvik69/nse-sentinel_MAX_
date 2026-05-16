"""
Mode philosophy preservation for A-I-L IN ONE.

The guard prevents every category from converging into a single "safe swing"
style by rewarding rows that still express their scanner's intended behavior.
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


def _mode_id(row: pd.Series | dict[str, Any]) -> int:
    return int(_safe_float(_get(row, "Mode ID", "Mode"), 0.0))


def compute_mode_style_integrity(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    cats = _categories(row)
    mode_id = _mode_id(row)
    momentum = _num(row, "Momentum Quality", "Bullish Probability", "Prediction Score", default=50.0)
    volume = _num(row, "Volume Quality", default=50.0)
    setup = _num(row, "Setup Cleanliness", "Setup Quality", default=50.0)
    structure = _num(row, "Structure Quality", "AIL Risk Adjusted Score", default=setup)
    regime = _num(row, "AIL Regime Alignment", "Regime Alignment", default=50.0)
    trap = _num(row, "Trap Risk Score", default=50.0)
    rsi = _safe_float(_get(row, "RSI"), 50.0)
    timing = _text(row, "Entry Timing", "Setup Type").upper()
    scores: list[float] = []
    notes: list[str] = []

    if "RELAXED" in cats or mode_id == 3:
        early = 78.0 if 46.0 <= rsi <= 60.0 or "EARLY" in timing or "ACCUM" in timing else 56.0
        scores.append(_clip(0.52 * early + 0.30 * structure + 0.18 * (100.0 - trap)))
        notes.append("Relaxed identity: early accumulation")
    if "INTRADAY" in cats or mode_id == 5:
        scores.append(_clip(0.56 * volume + 0.34 * momentum + 0.10 * (100.0 - trap)))
        notes.append("Intraday identity: volume expansion")
    if "SWING" in cats or mode_id == 6:
        scores.append(_clip(0.52 * setup + 0.28 * momentum + 0.20 * (100.0 - trap)))
        notes.append("Swing identity: controlled continuation")
    if "INSTITUTIONAL" in cats or mode_id == 4:
        scores.append(_clip(0.48 * regime + 0.30 * setup + 0.22 * structure))
        notes.append("Institutional identity: durable trend quality")
    if "MOMENTUM" in cats or mode_id in {1, 7}:
        scores.append(_clip(0.44 * momentum + 0.34 * structure + 0.22 * volume))
        notes.append("Momentum/Mode7 identity: structure-supported thrust")
    if "BREAKOUT" in cats:
        scores.append(_clip(0.42 * structure + 0.34 * volume + 0.24 * momentum))
        notes.append("Breakout identity: pressure and confirmation")

    score = float(np.mean(scores)) if scores else 55.0
    return {"score": round(_clip(score), 2), "notes": "; ".join(dict.fromkeys(notes[:4]))}


def detect_philosophy_flattening(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "AIL Philosophy Score" not in df.columns:
        return {"status": "unknown", "avg_integrity": 0.0, "low_integrity_share": 0.0}
    values = pd.to_numeric(df["AIL Philosophy Score"], errors="coerce").dropna()
    if values.empty:
        return {"status": "unknown", "avg_integrity": 0.0, "low_integrity_share": 0.0}
    avg = float(values.mean())
    low_share = float(values.lt(52.0).mean() * 100.0)
    status = "flattening" if avg < 56.0 or low_share > 45.0 else "healthy"
    return {"status": status, "avg_integrity": round(avg, 2), "low_integrity_share": round(low_share, 2)}


def preserve_mode_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    scores: list[float] = []
    notes: list[str] = []
    boosts: list[float] = []
    for _, row in out.iterrows():
        integrity = compute_mode_style_integrity(row)
        score = float(integrity["score"])
        scores.append(score)
        notes.append(str(integrity["notes"]))
        boosts.append(round(min(5.0, max(0.0, score - 62.0) * 0.09), 2))
    out["AIL Philosophy Score"] = scores
    out["AIL Philosophy Notes"] = notes
    out["AIL Philosophy Boost"] = boosts
    return out


__all__ = [
    "preserve_mode_identity",
    "compute_mode_style_integrity",
    "detect_philosophy_flattening",
]
