from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pandas as pd

from feature_data_manager import feature_manager

COMPARE_STOCK_LIMIT = 19
_SYMBOL_KEYS = ("Symbol", "Ticker", "ticker", "symbol", "Stock", "stock")


def normalize_compare_symbols(
    values: list[object] | tuple[object, ...] | None,
    limit: int = COMPARE_STOCK_LIMIT,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values or []:
        symbol = str(raw or "").strip().upper()
        if not symbol.endswith(".NS"):
            symbol = f"{symbol}.NS"
        base = symbol[:-3].strip()
        if not base:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
        if len(ordered) >= max(1, int(limit)):
            break
    return ordered


def normalize_compare_plain_symbols(
    values: list[object] | tuple[object, ...] | None,
    limit: int = COMPARE_STOCK_LIMIT,
) -> list[str]:
    return [symbol.replace(".NS", "") for symbol in normalize_compare_symbols(values, limit=limit)]


def _symbol_from_row(row: dict[str, Any]) -> str:
    for key in _SYMBOL_KEYS:
        value = row.get(key)
        symbol = normalize_compare_plain_symbols([value], limit=1)
        if symbol:
            return symbol[0]
    return ""


def _iter_source_symbols(source: object) -> list[object]:
    if source is None:
        return []

    if isinstance(source, pd.DataFrame):
        return [_symbol_from_row(row.to_dict()) for _, row in source.iterrows()]

    if isinstance(source, pd.Series):
        return [_symbol_from_row(source.to_dict())]

    if isinstance(source, dict):
        symbols: list[object] = []
        row_symbol = _symbol_from_row(source)
        if row_symbol:
            symbols.append(row_symbol)

        sections = source.get("sections")
        if isinstance(sections, dict):
            for values in sections.values():
                if isinstance(values, (list, tuple)):
                    symbols.extend(values)

        for key in ("picks", "symbols", "tickers", "records", "predictions"):
            values = source.get(key)
            if isinstance(values, pd.DataFrame):
                symbols.extend(_iter_source_symbols(values))
            elif isinstance(values, (list, tuple)):
                for item in values:
                    if isinstance(item, dict):
                        symbols.extend(_iter_source_symbols(item))
                    else:
                        symbols.append(item)
        return symbols

    if isinstance(source, (list, tuple, set)):
        symbols = []
        for item in source:
            if isinstance(item, (dict, pd.Series, pd.DataFrame)):
                symbols.extend(_iter_source_symbols(item))
            else:
                symbols.append(item)
        return symbols

    return [source]


def collect_compare_import_symbols(
    *sources: object,
    limit: int = COMPARE_STOCK_LIMIT,
) -> list[str]:
    symbols: list[object] = []
    for source in sources:
        symbols.extend(_iter_source_symbols(source))
    return normalize_compare_plain_symbols(symbols, limit=limit)


def build_compare_source_statuses(symbols: list[str]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for symbol in normalize_compare_symbols(symbols):
        status = feature_manager.get_last_status(symbol) or {}
        statuses.append(
            {
                "symbol": symbol.replace(".NS", ""),
                "source_kind": str(status.get("source_kind", "CACHED") or "CACHED"),
                "as_of": str(status.get("as_of", "") or ""),
                "source": str(status.get("source", "") or ""),
            }
        )
    return statuses


def summarize_compare_sources(statuses: list[dict[str, Any]] | None) -> str:
    if not statuses:
        return ""
    parts: list[str] = []
    for item in statuses:
        symbol = str(item.get("symbol", "") or "").strip()
        source_kind = str(item.get("source_kind", "CACHED") or "CACHED").strip().upper()
        as_of = str(item.get("as_of", "") or "").strip()
        text = f"{symbol} {source_kind}".strip()
        if as_of:
            text = f"{text} ({as_of})"
        parts.append(text)
    return " | ".join(parts)


def load_compare_results(symbols: list[str]) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    payload = feature_manager.load_compare_cache(normalize_compare_symbols(symbols))
    if not payload:
        return None, None
    frame_json = str(payload.get("battle_df", "") or "")
    if not frame_json:
        return None, payload
    try:
        df = pd.read_json(StringIO(frame_json), orient="split")
    except Exception:
        return None, payload
    return df, payload


def save_compare_results(
    symbols: list[str],
    df: pd.DataFrame,
    *,
    statuses: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "symbols": normalize_compare_symbols(symbols),
        "source_statuses": statuses or build_compare_source_statuses(symbols),
        "battle_df": df.to_json(orient="split", date_format="iso"),
    }
    feature_manager.save_compare_cache(normalize_compare_symbols(symbols), payload)
