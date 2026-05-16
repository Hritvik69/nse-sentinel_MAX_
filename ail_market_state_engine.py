"""
Temporal market-state intelligence for A-I-L IN ONE.

This module reads the existing market-session plan and turns it into bounded
orchestration weights.  It never pretends closed-market data is live.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd


_CLOSING_START = time(15, 15)
_MARKET_CLOSE = time(16, 0)


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


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return lo


def _now_ist() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now()


def _categories(row: pd.Series | dict[str, Any]) -> set[str]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    raw = str(getter("AIL Categories", getter("AIL Category", "")) or "")
    return {part.strip().upper() for part in raw.replace("|", ",").split(",") if part.strip()}


def _num(row: pd.Series | dict[str, Any], *keys: str, default: float = 0.0) -> float:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = getter(key, None)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return _clip(_safe_float(value, default))
    return default


def _text(row: pd.Series | dict[str, Any], *keys: str) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = getter(key, None)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return str(value).strip()
    return ""


def detect_market_state(
    session_plan: dict[str, Any] | None = None,
    preload_stats: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    plan = dict(session_plan or {})
    if not plan and isinstance(preload_stats, dict):
        raw_plan = preload_stats.get("plan")
        if isinstance(raw_plan, dict):
            plan = dict(raw_plan)
    if not plan:
        try:
            from data_session_manager import get_scan_data_plan

            plan = dict(get_scan_data_plan() or {})
        except Exception:
            plan = {}

    current = now or _now_ist()
    explicit_state = str(plan.get("state", "") or "").upper()
    window = str(plan.get("window", "") or "").upper()
    state = explicit_state or window or "UNKNOWN"
    if window == "LIVE":
        local_time = current.timetz().replace(tzinfo=None) if hasattr(current, "timetz") else current.time()
        state = "CLOSING" if _CLOSING_START <= local_time <= _MARKET_CLOSE else "LIVE"
    elif window == "CLOSED":
        state = "POST_CLOSE"
    elif window == "PRE_MARKET":
        state = "PRE_MARKET"
    elif window == "WEEKEND":
        state = "WEEKEND"

    snapshot_loaded = int(_safe_float((preload_stats or {}).get("snapshot_loaded"), 0.0))
    snapshot_saved = int(_safe_float((preload_stats or {}).get("snapshot_saved"), 0.0))
    use_snapshot = bool(plan.get("use_snapshot", False))
    return {
        "state": state,
        "window": window or state,
        "expected_date": plan.get("expected_date"),
        "source_label": str(plan.get("source_label", "") or ""),
        "summary": str(plan.get("summary", "") or ""),
        "use_snapshot": use_snapshot,
        "force_live_refresh": bool(plan.get("force_live_refresh", False)),
        "save_snapshot_after_scan": bool(plan.get("save_snapshot_after_scan", False)),
        "snapshot_exists": bool(plan.get("snapshot_exists", False)),
        "snapshot_loaded": snapshot_loaded,
        "snapshot_saved": snapshot_saved,
        "is_live_like": state in {"LIVE", "CLOSING"},
        "is_frozen": state in {"POST_CLOSE", "PRE_MARKET", "WEEKEND"} and use_snapshot,
    }


def get_market_state_weights(state: dict[str, Any] | str | None) -> dict[str, float]:
    label = str(state.get("state") if isinstance(state, dict) else state or "UNKNOWN").upper()
    base = {
        "momentum": 1.00,
        "intraday_volume": 1.00,
        "breakout": 1.00,
        "closing_strength": 1.00,
        "structure": 1.00,
        "swing": 1.00,
        "institutional": 1.00,
        "accumulation": 1.00,
        "trap_control": 1.00,
    }
    if label == "LIVE":
        base.update({"momentum": 1.08, "intraday_volume": 1.12, "breakout": 1.08})
    elif label == "CLOSING":
        base.update({"closing_strength": 1.18, "structure": 1.10, "breakout": 1.04, "intraday_volume": 1.03})
    elif label == "POST_CLOSE":
        base.update({"momentum": 0.92, "intraday_volume": 0.82, "breakout": 0.94, "structure": 1.14, "swing": 1.10, "institutional": 1.08, "trap_control": 1.12})
    elif label == "WEEKEND":
        base.update({"momentum": 0.86, "intraday_volume": 0.70, "breakout": 0.86, "structure": 1.16, "swing": 1.16, "institutional": 1.14, "accumulation": 1.12, "trap_control": 1.16})
    elif label == "PRE_MARKET":
        base.update({"momentum": 0.82, "intraday_volume": 0.68, "breakout": 0.84, "structure": 1.18, "swing": 1.12, "institutional": 1.10, "accumulation": 1.10, "trap_control": 1.14})
    return base


def _temporal_fit(row: pd.Series, weights: dict[str, float], label: str) -> tuple[float, str, float]:
    cats = _categories(row)
    momentum = _num(row, "Momentum Quality", "Bullish Probability", "Prediction Score", default=55.0)
    volume = _num(row, "Volume Quality", default=55.0)
    setup = _num(row, "Setup Cleanliness", "Setup Quality", default=55.0)
    trap = _num(row, "Trap Risk Score", default=50.0)
    structure = _num(row, "Structure Quality", "AIL Risk Adjusted Score", default=setup)
    regime = _num(row, "AIL Regime Alignment", "Regime Alignment", default=55.0)
    rsi = _safe_float(_text(row, "RSI"), 50.0)

    fit = 50.0
    notes: list[str] = []
    if "MOMENTUM" in cats:
        fit += (momentum - 55.0) * 0.16 * weights["momentum"]
        notes.append("momentum timed")
    if "INTRADAY" in cats:
        fit += (volume - 55.0) * 0.20 * weights["intraday_volume"]
        notes.append("volume timing")
    if "BREAKOUT" in cats:
        fit += ((volume + structure) / 2.0 - 55.0) * 0.18 * weights["breakout"]
        notes.append("breakout timing")
    if "SWING" in cats:
        fit += (setup - 55.0) * 0.18 * weights["swing"]
        notes.append("swing timing")
    if "INSTITUTIONAL" in cats:
        fit += (regime - 55.0) * 0.18 * weights["institutional"]
        notes.append("institutional timing")
    if "RELAXED" in cats:
        fit += (structure - 55.0) * 0.14 * weights["accumulation"]
        notes.append("accumulation timing")

    fit += (100.0 - trap - 45.0) * 0.12 * weights["trap_control"]
    if label in {"POST_CLOSE", "WEEKEND", "PRE_MARKET"} and ("MOMENTUM" in cats or "INTRADAY" in cats) and structure < 58.0:
        fit -= 8.0
        notes.append("closed-market momentum reduced")
    if label == "CLOSING":
        if setup >= 62.0 and volume >= 58.0:
            fit += 6.0
            notes.append("closing confirmation")
        if rsi > 72.0:
            fit -= 5.0
            notes.append("closing extension risk")

    fit = _clip(fit)
    multiplier = float(np.clip(0.94 + (fit / 100.0) * 0.12, 0.94, 1.06))
    return round(fit, 2), "; ".join(dict.fromkeys(notes[:4])), round(multiplier, 4)


def apply_market_state_adjustments(df: pd.DataFrame, state: dict[str, Any] | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    state = detect_market_state(state if isinstance(state, dict) and "window" in state else None, {"plan": state} if isinstance(state, dict) else None)
    label = str(state.get("state", "UNKNOWN") or "UNKNOWN").upper()
    weights = get_market_state_weights(state)
    fits: list[float] = []
    notes: list[str] = []
    multipliers: list[float] = []
    for _, row in out.iterrows():
        fit, note, multiplier = _temporal_fit(row, weights, label)
        fits.append(fit)
        notes.append(note)
        multipliers.append(multiplier)
    out["AIL Market State"] = label
    out["AIL Temporal Fit"] = fits
    out["AIL Temporal Notes"] = notes
    out["AIL Temporal Multiplier"] = multipliers
    return out


__all__ = [
    "detect_market_state",
    "get_market_state_weights",
    "apply_market_state_adjustments",
]
