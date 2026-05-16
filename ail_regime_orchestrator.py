"""
Regime-adaptive orchestration preferences for A-I-L IN ONE.
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


def _categories(row: pd.Series | dict[str, Any]) -> set[str]:
    raw = str(_get(row, "AIL Categories", "AIL Category") or "")
    return {part.strip().upper() for part in raw.replace("|", ",").split(",") if part.strip()}


def _state_text(market_bias: dict[str, Any] | None, market_state: dict[str, Any] | None) -> str:
    pieces: list[str] = []
    if isinstance(market_bias, dict):
        pieces.extend(str(market_bias.get(key, "") or "") for key in ("bias", "trend", "market_pressure", "regime"))
    if isinstance(market_state, dict):
        pieces.append(str(market_state.get("state", "") or ""))
    return " ".join(pieces).upper()


def compute_regime_strategy_bias(
    market_bias: dict[str, Any] | None = None,
    market_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _state_text(market_bias, market_state)
    multipliers = {
        "momentum": 1.00,
        "breakout": 1.00,
        "swing": 1.00,
        "institutional": 1.00,
        "accumulation": 1.00,
        "trap_control": 1.00,
        "volume": 1.00,
    }
    labels: list[str] = []
    if any(token in text for token in ("TRENDING_UP", "TRENDING UP", "BULL", "UPTREND", "BULLISH")):
        multipliers.update({"momentum": 1.08, "breakout": 1.08, "institutional": 1.06, "volume": 1.04})
        labels.append("trend-up favors continuation")
    if any(token in text for token in ("RANGE", "SIDEWAYS", "CHOP", "NEUTRAL")):
        multipliers.update({"breakout": 0.92, "momentum": 0.94, "swing": 1.08, "accumulation": 1.08, "trap_control": 1.06})
        labels.append("range favors controlled swing")
    if any(token in text for token in ("HIGH_VOL", "HIGH VOL", "VOLATILE")):
        multipliers.update({"momentum": 0.90, "breakout": 0.92, "trap_control": 1.14, "institutional": 1.06})
        labels.append("volatility increases trap control")
    if any(token in text for token in ("WEAK", "BEAR", "DOWN", "SELL")):
        multipliers.update({"momentum": 0.88, "breakout": 0.88, "accumulation": 1.10, "institutional": 1.08, "trap_control": 1.12})
        labels.append("weak market favors defensive setups")
    return {"multipliers": multipliers, "notes": "; ".join(labels) or "neutral regime preference", "state_text": text}


def _strategy_fit(row: pd.Series, bias: dict[str, Any]) -> tuple[float, str, float]:
    multipliers = bias.get("multipliers", {}) if isinstance(bias, dict) else {}
    cats = _categories(row)
    momentum = _num(row, "Momentum Quality", "Bullish Probability", default=55.0)
    volume = _num(row, "Volume Quality", default=55.0)
    setup = _num(row, "Setup Cleanliness", default=55.0)
    trap_quality = 100.0 - _num(row, "Trap Risk Score", default=50.0)
    regime = _num(row, "AIL Regime Alignment", "Regime Alignment", default=55.0)
    score = 55.0
    notes: list[str] = []
    if "MOMENTUM" in cats:
        score += (momentum - 55.0) * 0.18 * float(multipliers.get("momentum", 1.0))
        notes.append("momentum regime fit")
    if "BREAKOUT" in cats:
        score += ((volume + setup) / 2.0 - 55.0) * 0.18 * float(multipliers.get("breakout", 1.0))
        notes.append("breakout regime fit")
    if "SWING" in cats:
        score += (setup - 55.0) * 0.18 * float(multipliers.get("swing", 1.0))
        notes.append("swing regime fit")
    if "INSTITUTIONAL" in cats:
        score += (regime - 55.0) * 0.18 * float(multipliers.get("institutional", 1.0))
        notes.append("institutional regime fit")
    if "RELAXED" in cats:
        score += (setup - 55.0) * 0.14 * float(multipliers.get("accumulation", 1.0))
        notes.append("accumulation regime fit")
    score += (trap_quality - 50.0) * 0.12 * float(multipliers.get("trap_control", 1.0))
    score = _clip(score)
    multiplier = float(np.clip(0.94 + (score / 100.0) * 0.12, 0.94, 1.06))
    return round(score, 2), "; ".join(dict.fromkeys(notes[:4])) or str(bias.get("notes", "")), round(multiplier, 4)


def apply_regime_preference(df: pd.DataFrame, bias: dict[str, Any] | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    fits: list[float] = []
    notes: list[str] = []
    multipliers: list[float] = []
    for _, row in out.iterrows():
        fit, note, multiplier = _strategy_fit(row, bias or {})
        fits.append(fit)
        notes.append(note)
        multipliers.append(multiplier)
    out["AIL Regime Strategy Fit"] = fits
    out["AIL Regime Strategy Notes"] = notes
    out["AIL Regime Multiplier"] = multipliers
    return out


def compute_regime_compatibility(row: pd.Series | dict[str, Any], bias: dict[str, Any] | None = None) -> float:
    return float(_strategy_fit(pd.Series(row), bias or {})[0])


__all__ = [
    "compute_regime_strategy_bias",
    "apply_regime_preference",
    "compute_regime_compatibility",
]
