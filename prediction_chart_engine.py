from __future__ import annotations

from typing import Any

import pandas as pd

from feature_data_manager import feature_manager


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    return raw if raw.endswith(".NS") else f"{raw}.NS"


def fetch_chart_data(
    symbol: str,
    *,
    period: str = "2mo",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    ticker = _normalize_symbol(symbol)
    if not ticker:
        return None
    return feature_manager.get_stock_data(
        ticker,
        period=period,
        interval=interval,
        force_refresh=force_refresh,
    )


def get_chart_status(symbol: str) -> dict[str, Any] | None:
    ticker = _normalize_symbol(symbol)
    if not ticker:
        return None
    return feature_manager.get_last_status(ticker)
