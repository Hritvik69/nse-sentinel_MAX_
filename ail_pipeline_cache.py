"""
Shared lightweight feature cache for A-I-L orchestration passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


_NUMERIC_COLUMNS = (
    "AIL Master Score",
    "AIL Confidence",
    "AIL Calibrated Confidence",
    "AIL Temporal Fit",
    "AIL Agreement Score",
    "AIL Conflict Score",
    "AIL Regime Strategy Fit",
    "Smart Potential Score",
    "Bullish Probability",
    "Prediction Score",
    "Final Score",
    "Momentum Quality",
    "Volume Quality",
    "Setup Cleanliness",
    "Trap Risk Score",
    "Regime Alignment",
    "Sector Support",
)


@dataclass
class AILFeatureCache:
    built_at: str
    row_count: int
    numeric: dict[str, pd.Series] = field(default_factory=dict)
    categories_by_symbol: dict[str, set[str]] = field(default_factory=dict)
    mode_count_by_symbol: dict[str, int] = field(default_factory=dict)
    confidence_components: dict[str, dict[str, Any]] = field(default_factory=dict)
    market_state: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    learning_profile: dict[str, Any] = field(default_factory=dict)


def _symbol(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in ("Symbol", "Ticker", "symbol", "ticker", "Stock"):
        value = str(getter(key, "") or "").strip().upper()
        if value:
            return value[:-3] if value.endswith(".NS") else value
    return ""


def _categories(raw: Any) -> set[str]:
    return {part.strip() for part in str(raw or "").replace("|", ",").split(",") if part.strip()}


def cache_orchestration_features(
    df: pd.DataFrame,
    *,
    market_state: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    learning_profile: dict[str, Any] | None = None,
) -> AILFeatureCache:
    cache = AILFeatureCache(
        built_at=datetime.now().isoformat(timespec="seconds"),
        row_count=int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        market_state=market_state or {},
        calibration=calibration or {},
        learning_profile=learning_profile or {},
    )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return cache
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            cache.numeric[col] = pd.to_numeric(df[col], errors="coerce")
    for _, row in df.iterrows():
        symbol = _symbol(row)
        if not symbol:
            continue
        cats = _categories(row.get("AIL Categories", row.get("AIL Category", "")))
        cache.categories_by_symbol.setdefault(symbol, set()).update(cats)
        mode_count = pd.to_numeric(pd.Series([row.get("AIL Mode Count", 0)]), errors="coerce").fillna(0).iloc[0]
        cache.mode_count_by_symbol[symbol] = max(cache.mode_count_by_symbol.get(symbol, 0), int(mode_count or len(cats) or 1))
    return cache


def get_cached_alignment(cache: AILFeatureCache | None, symbol: str) -> dict[str, Any]:
    if cache is None:
        return {"categories": set(), "mode_count": 0}
    key = str(symbol or "").strip().upper().replace(".NS", "")
    return {
        "categories": cache.categories_by_symbol.get(key, set()),
        "mode_count": cache.mode_count_by_symbol.get(key, 0),
    }


def memoize_confidence_components(cache: AILFeatureCache | None, symbol: str, components: dict[str, Any]) -> None:
    if cache is None:
        return
    key = str(symbol or "").strip().upper().replace(".NS", "")
    if key:
        cache.confidence_components[key] = dict(components or {})


__all__ = [
    "AILFeatureCache",
    "cache_orchestration_features",
    "get_cached_alignment",
    "memoize_confidence_components",
]
