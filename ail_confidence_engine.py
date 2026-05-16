"""
Confidence fusion helpers for A-I-L IN ONE.

The functions here intentionally consume existing NSE Sentinel columns.  They
do not create a replacement model; they blend the scanner, Battle, learning,
sector, regime, volume, and structure signals already produced upstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_BLANKS = {"", "nan", "none", "null", "-", "n/a", "na"}


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
            if cleaned.lower() in _BLANKS:
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
        if value is not None and str(value).strip().lower() not in _BLANKS:
            return value
    return None


def _numeric(row: pd.Series | dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(_get(row, key), None)
        if value is not None:
            return _clip(value)
    return None


def _text(row: pd.Series | dict[str, Any], *keys: str) -> str:
    value = _get(row, *keys)
    return str(value or "").strip()


def _score_text(value: Any, *, context: str = "quality") -> float | None:
    text = str(value or "").strip().upper()
    if not text or text.lower() in _BLANKS:
        return None
    if context == "trap":
        mapping = {
            "LOW": 84.0,
            "CLEAN": 84.0,
            "NONE": 84.0,
            "MEDIUM": 58.0,
            "MODERATE": 58.0,
            "HIGH": 26.0,
            "TRAP": 16.0,
            "AVOID": 18.0,
        }
    else:
        mapping = {
            "A+": 94.0,
            "A": 88.0,
            "B+": 78.0,
            "B": 70.0,
            "C": 54.0,
            "D": 34.0,
            "EXCELLENT": 92.0,
            "VERY STRONG": 88.0,
            "STRONG": 82.0,
            "GOOD": 74.0,
            "BUILDING": 66.0,
            "OK": 60.0,
            "NORMAL": 56.0,
            "MEDIUM": 56.0,
            "AVERAGE": 52.0,
            "LOW": 38.0,
            "WEAK": 34.0,
            "POOR": 26.0,
            "EARLY": 66.0,
            "FORMING": 64.0,
            "WAIT": 54.0,
            "LATE": 42.0,
            "OVEREXTENDED": 30.0,
        }
    for key, score in mapping.items():
        if key in text:
            return score
    return None


def _mode_categories(row: pd.Series | dict[str, Any]) -> list[str]:
    raw = _text(row, "AIL Categories", "AIL Category")
    if not raw:
        return []
    return [part.strip() for part in raw.replace("|", ",").split(",") if part.strip()]


def _profile_score(profile: dict[str, Any] | None, bucket: str, key: Any) -> float | None:
    if not isinstance(profile, dict):
        return None
    section = profile.get(bucket, {})
    if not isinstance(section, dict):
        return None
    for candidate in (key, str(key), str(key).upper(), str(key).title()):
        item = section.get(candidate)
        if isinstance(item, dict):
            score = _safe_float(item.get("score", item.get("win_rate_pct")), None)
            if score is not None:
                return _clip(score)
        score = _safe_float(item, None)
        if score is not None:
            return _clip(score)
    return None


def confidence_from_learning(
    row: pd.Series | dict[str, Any],
    learning_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components: list[tuple[str, float]] = []
    learned = _numeric(row, "Learned Prob %", "Learning Probability", "ML Success %", "ML %")
    if learned is not None:
        components.append(("learning probability", learned))

    mode_id = _safe_float(_get(row, "Mode ID", "Mode"), None)
    if mode_id is not None:
        mode_score = _profile_score(learning_profile, "mode_reliability", int(mode_id))
        if mode_score is not None:
            components.append((f"mode {int(mode_id)} reliability", mode_score))

    for category in _mode_categories(row):
        category_score = _profile_score(learning_profile, "category_reliability", category)
        if category_score is not None:
            components.append((f"{category} reliability", category_score))

    sector = _text(row, "Sector")
    sector_score = _profile_score(learning_profile, "sector_reliability", sector)
    if sector_score is not None:
        components.append((f"{sector} reliability", sector_score))

    regime = _text(row, "Market Regime", "Regime")
    regime_score = _profile_score(learning_profile, "regime_reliability", regime)
    if regime_score is not None:
        components.append((f"{regime} reliability", regime_score))

    if not components:
        return {"score": None, "drivers": []}
    score = float(np.mean([score for _, score in components]))
    return {"score": round(_clip(score), 2), "drivers": [name for name, _ in components[:4]]}


def confidence_from_alignment(
    row: pd.Series | dict[str, Any],
    market_bias: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sector = _numeric(row, "Sector Support", "Sector Strength", "Sector Accuracy")
    regime = _numeric(row, "Regime Alignment", "AIL Regime Alignment")
    market = _numeric(row, "AIL Market Compatibility", "Market Compatibility")

    if market is None and isinstance(market_bias, dict) and market_bias:
        bias_text = " ".join(
            str(market_bias.get(key, "") or "")
            for key in ("bias", "trend", "market_pressure", "regime")
        ).upper()
        bullish = _numeric(row, "Bullish Probability", "Prediction Score", "Final Score")
        signal_text = _text(row, "Final Signal", "Adjusted Signal", "Signal", "Smart Verdict").upper()
        wants_bull = (bullish is not None and bullish >= 55.0) or any(token in signal_text for token in ("BUY", "BULL", "GREEN"))
        if "BEAR" in bias_text or "WEAK" in bias_text or "DOWN" in bias_text:
            market = 42.0 if wants_bull else 62.0
        elif "BULL" in bias_text or "UP" in bias_text or "STRONG" in bias_text:
            market = 72.0 if wants_bull else 48.0
        elif "RANGE" in bias_text or "SIDE" in bias_text or "NEUTRAL" in bias_text:
            market = 58.0
        if market is not None and ("HIGH_VOL" in bias_text or "HIGH VOL" in bias_text):
            market = _clip(market - 5.0)

    out = {
        "sector": None if sector is None else round(sector, 2),
        "regime": None if regime is None else round(regime, 2),
        "market": None if market is None else round(market, 2),
    }
    values = [value for value in out.values() if value is not None]
    out["score"] = round(float(np.mean(values)), 2) if values else None
    out["drivers"] = [name for name, value in out.items() if name in {"sector", "regime", "market"} and value is not None]
    return out


def confidence_from_structure(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    structure = _numeric(row, "Structure Quality", "Breakout Quality")
    if structure is None:
        structure = _score_text(_text(row, "Structure Quality", "Breakout Quality"))
    setup = _numeric(row, "Setup Cleanliness", "Setup Quality")
    if setup is None:
        setup = _score_text(_text(row, "Setup Quality", "Entry Timing", "Grade"))
    volume = _numeric(row, "Volume Quality", "Volume Score")
    if volume is None:
        volume = _score_text(_text(row, "Volume Trend", "Volume Confirmation", "Volume Strength"))
    momentum = _numeric(row, "Momentum Quality", "Momentum Score")
    values = [value for value in (structure, setup, volume, momentum) if value is not None]
    return {
        "score": round(float(np.mean(values)), 2) if values else None,
        "structure": None if structure is None else round(_clip(structure), 2),
        "setup": None if setup is None else round(_clip(setup), 2),
        "volume": None if volume is None else round(_clip(volume), 2),
        "momentum": None if momentum is None else round(_clip(momentum), 2),
        "drivers": [
            name
            for name, value in (
                ("structure", structure),
                ("setup", setup),
                ("volume", volume),
                ("momentum", momentum),
            )
            if value is not None
        ],
    }


def compute_smart_confidence(
    row: pd.Series | dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    learning_profile = context.get("learning_profile") if isinstance(context, dict) else None
    market_bias = context.get("market_bias") if isinstance(context, dict) else None

    prediction = _numeric(row, "Prediction Score", "Bullish Probability", "Battle Probability", "AIL Top3 Score")
    grading = _numeric(row, "Smart Confidence", "Battle Confidence", "Confidence", "AI Confidence")
    trap_score = _numeric(row, "Trap Risk Score")
    trap = _clip(100.0 - trap_score) if trap_score is not None else _score_text(_text(row, "Trap Risk", "Trap Warning"), context="trap")
    setup = _numeric(row, "Setup Cleanliness")
    if setup is None:
        setup = _score_text(_text(row, "Setup Quality", "Entry Timing", "Grade"))
    learning = confidence_from_learning(row, learning_profile)
    alignment = confidence_from_alignment(row, market_bias)
    structure = confidence_from_structure(row)
    volume = structure.get("volume")

    mode_score = None
    mode_id = _safe_float(_get(row, "Mode ID", "Mode"), None)
    if mode_id is not None:
        mode_score = _profile_score(learning_profile, "mode_reliability", int(mode_id))

    components = [
        ("prediction", prediction, 0.18),
        ("grading", grading, 0.14),
        ("trap quality", trap, 0.14),
        ("setup cleanliness", setup, 0.12),
        ("mode reliability", mode_score, 0.09),
        ("learning probability", learning.get("score"), 0.12),
        ("sector alignment", alignment.get("sector"), 0.07),
        ("regime alignment", alignment.get("regime"), 0.07),
        ("market compatibility", alignment.get("market"), 0.04),
        ("volume quality", volume, 0.06),
        ("structure quality", structure.get("structure"), 0.07),
    ]
    available = [(name, _clip(float(value)), weight) for name, value, weight in components if value is not None]
    total_weight = sum(weight for _, _, weight in components)
    available_weight = sum(weight for _, _, weight in available)
    if not available or available_weight <= 0:
        return {
            "score": 0.0,
            "label": "Insufficient evidence",
            "coverage": 0.0,
            "drivers": "No measurable confidence inputs",
            "components": {},
        }

    score = sum(value * weight for _, value, weight in available) / available_weight
    score = _clip(score)
    coverage = _clip(100.0 * available_weight / total_weight)
    if score >= 78.0 and coverage >= 55.0:
        label = "Institutional grade"
    elif score >= 68.0:
        label = "High conviction"
    elif score >= 56.0:
        label = "Balanced"
    elif score >= 42.0:
        label = "Cautious"
    else:
        label = "Low confidence"

    sorted_components = sorted(available, key=lambda item: item[1] * item[2], reverse=True)
    driver_text = "; ".join(f"{name} {value:.1f}" for name, value, _ in sorted_components[:4])
    return {
        "score": round(score, 2),
        "label": label,
        "coverage": round(coverage, 2),
        "drivers": driver_text,
        "components": {name: round(value, 2) for name, value, _ in available},
    }


__all__ = [
    "compute_smart_confidence",
    "confidence_from_learning",
    "confidence_from_alignment",
    "confidence_from_structure",
]
