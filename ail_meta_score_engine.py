"""
Meta-scoring engine for A-I-L IN ONE.

This layer calibrates existing scanner and Battle outputs with market context,
mode identity, risk, structure, learning reliability, and multi-mode agreement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ail_confidence_engine import compute_smart_confidence
from ail_learning_engine import learning_adjustment_for_row


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return lo


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


def _get(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = getter(key, None)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null", "-", "n/a", "na"}:
            return value
    return None


def _numeric(row: pd.Series | dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(_get(row, key), None)
        if value is not None:
            return _clip(value)
    return None


def _text(row: pd.Series | dict[str, Any], *keys: str) -> str:
    return str(_get(row, *keys) or "").strip()


def _quality_from_text(text: Any, default: float | None = None) -> float | None:
    raw = str(text or "").upper()
    if not raw:
        return default
    mapping = {
        "A+": 94.0,
        "A": 88.0,
        "EXCELLENT": 92.0,
        "VERY STRONG": 88.0,
        "STRONG": 82.0,
        "B+": 78.0,
        "GOOD": 74.0,
        "B": 70.0,
        "BUILDING": 66.0,
        "MEDIUM": 58.0,
        "NORMAL": 56.0,
        "C": 54.0,
        "LOW": 38.0,
        "WEAK": 34.0,
        "HIGH": 28.0,
        "TRAP": 16.0,
    }
    for key, score in mapping.items():
        if key in raw:
            return score
    return default


def _categories(row: pd.Series | dict[str, Any]) -> list[str]:
    raw = _text(row, "AIL Categories", "AIL Category")
    return [part.strip() for part in raw.replace("|", ",").split(",") if part.strip()]


def _market_state(market_bias: dict[str, Any] | None) -> dict[str, bool]:
    text = ""
    if isinstance(market_bias, dict):
        text = " ".join(str(market_bias.get(key, "") or "") for key in ("bias", "trend", "market_pressure", "regime")).upper()
    return {
        "bullish": any(token in text for token in ("BULL", "UP", "STRONG")),
        "weak": any(token in text for token in ("BEAR", "DOWN", "WEAK", "SELL")),
        "range": any(token in text for token in ("RANGE", "SIDE", "NEUTRAL", "CHOP")),
        "volatile": any(token in text for token in ("VOLATILE", "HIGH_VOL", "HIGH VOL")),
        "trending": any(token in text for token in ("TREND", "UP", "DOWN")),
    }


def compute_regime_alignment(row: pd.Series | dict[str, Any], market_bias: dict[str, Any] | None = None) -> float:
    existing = _numeric(row, "Regime Alignment", "AIL Regime Alignment")
    if existing is not None:
        return round(existing, 2)

    state = _market_state(market_bias)
    cats = {cat.upper() for cat in _categories(row)}
    mode_id = int(_safe_float(_get(row, "Mode ID", "Mode"), 0) or 0)
    score = 56.0
    if state["bullish"] and ({"MOMENTUM", "BREAKOUT", "INSTITUTIONAL"} & cats or mode_id in {1, 4, 7}):
        score += 12.0
    if state["weak"] and ({"BREAKOUT", "MOMENTUM"} & cats or mode_id in {1, 7}):
        score -= 12.0
    if state["weak"] and ({"RELAXED", "INSTITUTIONAL"} & cats or mode_id in {3, 4}):
        score += 5.0
    if state["range"] and ({"SWING", "RELAXED"} & cats or mode_id in {3, 6}):
        score += 8.0
    if state["volatile"]:
        trap = _numeric(row, "Trap Risk Score") or 50.0
        score += 5.0 if trap < 45.0 else -8.0
    return round(_clip(score), 2)


def compute_market_compatibility(row: pd.Series | dict[str, Any], market_bias: dict[str, Any] | None = None) -> float:
    existing = _numeric(row, "AIL Market Compatibility", "Market Compatibility")
    if existing is not None:
        return round(existing, 2)
    state = _market_state(market_bias)
    bullish = _numeric(row, "Bullish Probability", "Prediction Score", "Final Score")
    wants_bull = bullish is not None and bullish >= 55.0
    signal = _text(row, "Smart Verdict", "Final Signal", "Adjusted Signal", "Signal").upper()
    if any(token in signal for token in ("BUY", "BULL", "GREEN")):
        wants_bull = True
    score = 58.0
    if state["bullish"]:
        score = 74.0 if wants_bull else 48.0
    elif state["weak"]:
        score = 44.0 if wants_bull else 64.0
    elif state["range"]:
        score = 63.0
    if state["volatile"]:
        score -= 5.0
    return round(_clip(score), 2)


def compute_risk_adjusted_potential(row: pd.Series | dict[str, Any]) -> float:
    components: list[tuple[float, float]] = []
    for value, weight in (
        (_numeric(row, "Risk Reward Score"), 0.26),
        (_numeric(row, "Setup Cleanliness"), 0.22),
        (_numeric(row, "Momentum Quality"), 0.16),
        (_numeric(row, "Volume Quality"), 0.14),
        (_numeric(row, "Bullish Probability", "Prediction Score"), 0.12),
    ):
        if value is not None:
            components.append((value, weight))
    trap = _numeric(row, "Trap Risk Score")
    if trap is not None:
        components.append((_clip(100.0 - trap), 0.18))
    rr = _safe_float(_get(row, "RR", "Risk Reward", "Reward Risk"), None)
    if rr is not None:
        components.append((_clip(45.0 + min(rr, 4.0) * 10.0), 0.08))
    if not components:
        return 0.0
    total = sum(weight for _, weight in components)
    return round(_clip(sum(value * weight for value, weight in components) / total), 2)


def _mode_philosophy_score(row: pd.Series | dict[str, Any], market_bias: dict[str, Any] | None = None) -> tuple[float, str]:
    cats = {cat.upper() for cat in _categories(row)}
    mode_id = int(_safe_float(_get(row, "Mode ID", "Mode"), 0) or 0)
    rsi = _safe_float(_get(row, "RSI"), 50.0) or 50.0
    trap = _numeric(row, "Trap Risk Score")
    trap = trap if trap is not None else 50.0
    volume = _numeric(row, "Volume Quality", "Vol / Avg")
    setup = _numeric(row, "Setup Cleanliness")
    momentum = _numeric(row, "Momentum Quality")
    regime = compute_regime_alignment(row, market_bias)
    score = 58.0
    reasons: list[str] = []

    if "RELAXED" in cats or mode_id == 3:
        early = 46.0 <= rsi <= 60.0 or "EARLY" in _text(row, "Entry Timing", "Setup Type").upper()
        score += 12.0 if early else 2.0
        score += 5.0 if trap < 50.0 else -4.0
        reasons.append("relaxed accumulation fit")
    if "INTRADAY" in cats or mode_id == 5:
        if volume is not None:
            score += (volume - 55.0) * 0.22
            reasons.append("intraday volume expansion")
    if "MOMENTUM" in cats or mode_id in {1, 7}:
        if momentum is not None:
            score += (momentum - 55.0) * 0.20
        if rsi > 72.0 or trap >= 65.0:
            score -= 10.0
            reasons.append("momentum exhaustion check")
        else:
            reasons.append("momentum sustainability")
    if "SWING" in cats or mode_id == 6:
        if setup is not None:
            score += (setup - 55.0) * 0.24
        if rsi > 72.0:
            score -= 8.0
        reasons.append("swing setup cleanliness")
    if "INSTITUTIONAL" in cats or mode_id == 4:
        score += (regime - 55.0) * 0.28
        reasons.append("institutional regime alignment")
    if "BREAKOUT" in cats:
        volume_term = (volume - 55.0) * 0.18 if volume is not None else 0.0
        score += volume_term + (8.0 if trap < 52.0 else -8.0)
        reasons.append("breakout confirmation")

    return round(_clip(score), 2), "; ".join(dict.fromkeys(reasons[:3])) or "mode-context calibrated"


def _dynamic_weights(row: pd.Series | dict[str, Any], market_bias: dict[str, Any] | None = None) -> dict[str, float]:
    state = _market_state(market_bias)
    cats = {cat.upper() for cat in _categories(row)}
    weights = {
        "base_score": 0.18,
        "prediction": 0.13,
        "confidence": 0.13,
        "risk_adjusted": 0.12,
        "mode_fit": 0.10,
        "regime": 0.09,
        "sector": 0.07,
        "learning": 0.07,
        "agreement": 0.06,
        "risk_control": 0.05,
    }
    if state["weak"] or state["volatile"]:
        weights["risk_control"] += 0.05
        weights["sector"] += 0.03
        weights["base_score"] -= 0.03
        weights["prediction"] -= 0.02
        weights["agreement"] -= 0.03
    if state["bullish"] or state["trending"]:
        weights["regime"] += 0.03
        weights["agreement"] += 0.02
        if {"MOMENTUM", "BREAKOUT", "INSTITUTIONAL"} & cats:
            weights["prediction"] += 0.02
    if state["range"] or "RELAXED" in cats:
        weights["mode_fit"] += 0.03
        weights["risk_adjusted"] += 0.02
        weights["prediction"] -= 0.02
    total = sum(max(0.0, value) for value in weights.values())
    return {key: max(0.0, value) / total for key, value in weights.items()}


def _multi_mode_bonus(row: pd.Series | dict[str, Any]) -> float:
    count = _safe_float(_get(row, "AIL Mode Count"), None)
    if count is None:
        cats = _categories(row)
        count = float(len(cats)) if cats else 1.0
    return round(_clip((max(1.0, count) - 1.0) * 6.0, 0.0, 18.0), 2)


def _reason_text(row: pd.Series | dict[str, Any], values: dict[str, float], philosophy_reason: str, confidence: dict[str, Any]) -> str:
    parts: list[str] = []
    if values.get("base_score", 0.0) >= 68.0:
        parts.append("strong upstream score")
    if values.get("risk_control", 0.0) >= 60.0:
        parts.append("trap risk controlled")
    if values.get("regime", 0.0) >= 65.0:
        parts.append("market regime aligned")
    if values.get("sector", 0.0) >= 65.0:
        parts.append("sector support present")
    if values.get("agreement", 0.0) >= 60.0:
        parts.append("multi-mode agreement")
    if philosophy_reason:
        parts.append(philosophy_reason)
    drivers = str(confidence.get("drivers", "") or "")
    if drivers:
        parts.append("confidence: " + drivers)
    return "; ".join(dict.fromkeys(parts[:5])) or "ranked from available real metrics"


def compute_ail_master_score(
    df: pd.DataFrame,
    *,
    market_bias: dict[str, Any] | None = None,
    learning_profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    context = {"market_bias": market_bias or {}, "learning_profile": learning_profile or {}}
    for _, row in df.iterrows():
        out = row.to_dict()
        confidence = compute_smart_confidence(row, context)
        learning = learning_adjustment_for_row(row, learning_profile)
        risk_adjusted = compute_risk_adjusted_potential(row)
        regime = compute_regime_alignment(row, market_bias)
        market_compat = compute_market_compatibility(row, market_bias)
        mode_fit, philosophy_reason = _mode_philosophy_score(row, market_bias)
        agreement = _clip(52.0 + _multi_mode_bonus(row) * 2.0)
        sector = _numeric(row, "Sector Support", "Sector Strength") or market_compat
        trap = _numeric(row, "Trap Risk Score")
        risk_control = _clip(100.0 - trap) if trap is not None else _quality_from_text(_text(row, "Trap Risk", "Trap Warning"), 56.0)
        base_score = _numeric(row, "Smart Potential Score", "Battle Score", "Final Score", "AIL Top3 Score") or 0.0
        prediction = _numeric(row, "Bullish Probability", "Prediction Score", "Battle Probability") or base_score
        learning_score = learning.get("score")
        if learning_score is None:
            learning_score = confidence.get("components", {}).get("learning probability")
        values = {
            "base_score": base_score,
            "prediction": prediction,
            "confidence": float(confidence.get("score", 0.0) or 0.0),
            "risk_adjusted": risk_adjusted,
            "mode_fit": mode_fit,
            "regime": regime,
            "sector": sector,
            "learning": float(learning_score) if learning_score is not None else 0.0,
            "agreement": agreement,
            "risk_control": risk_control,
        }
        weights = _dynamic_weights(row, market_bias)
        available = [(key, values[key], weight) for key, weight in weights.items() if values.get(key, 0.0) > 0.0]
        if available:
            score = sum(value * weight for key, value, weight in available) / sum(weight for _, _, weight in available)
        else:
            score = 0.0
        score = _clip(score + _multi_mode_bonus(row) * 0.35)
        score = _clip(score * float(learning.get("multiplier", 1.0) or 1.0))

        out["AIL Master Score"] = round(score, 2)
        out["AIL Confidence"] = round(float(confidence.get("score", 0.0) or 0.0), 2)
        out["AIL Confidence Label"] = confidence.get("label", "")
        out["AIL Confidence Drivers"] = confidence.get("drivers", "")
        out["AIL Confidence Coverage"] = confidence.get("coverage", 0.0)
        out["AIL Risk Adjusted Score"] = risk_adjusted
        out["AIL Regime Alignment"] = regime
        out["AIL Market Compatibility"] = market_compat
        out["AIL Mode Philosophy Score"] = mode_fit
        out["AIL Multi Mode Bonus"] = _multi_mode_bonus(row)
        out["AIL Learning Reliability"] = learning.get("score", "")
        out["AIL Learning Adjustment"] = round((float(learning.get("multiplier", 1.0) or 1.0) - 1.0) * 100.0, 2)
        out["AIL Learning Drivers"] = learning.get("drivers", "")
        out["AIL Reasoning"] = _reason_text(row, values, philosophy_reason, confidence)
        out["AIL Component Weights"] = ", ".join(f"{key}:{weight:.2f}" for key, weight in weights.items())
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    return out_df.sort_values(
        ["AIL Master Score", "AIL Confidence", "AIL Risk Adjusted Score"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


__all__ = [
    "compute_ail_master_score",
    "compute_regime_alignment",
    "compute_market_compatibility",
    "compute_risk_adjusted_potential",
]
