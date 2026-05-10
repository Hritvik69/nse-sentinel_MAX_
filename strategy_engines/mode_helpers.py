"""
Small reusable helpers for mode parsing and mode metadata access.
"""

from __future__ import annotations

from collections.abc import Mapping

from strategy_engines.constants import MODE_ID_COLUMN
from strategy_engines.mode_registry import (
    MODE_METADATA,
    get_mode_color,
    get_mode_description,
    get_mode_display,
    get_mode_label,
    get_mode_metadata,
    get_mode_name,
)


def _coerce_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        raw = str(value).strip().lower()
        if raw.replace(".", "", 1).isdigit():
            return int(float(raw))
    except Exception:
        return None
    return None


def _reverse_label_map() -> dict[str, int]:
    labels: dict[str, int] = {}
    for mode, meta in MODE_METADATA.items():
        for key in ("name", "short_name", "display_name", "ui_label"):
            value = meta.get(key)
            if value:
                labels[str(value).strip().lower()] = int(mode)
        labels[get_mode_label(mode).strip().lower()] = int(mode)
        labels[f"mode {mode}"] = int(mode)
        labels[f"m{mode}"] = int(mode)
    return labels


def resolve_mode_id(value: object, default: int | None = None) -> int | None:
    """Resolve a mode id from an int, row-like object, or known registry label."""
    try:
        if isinstance(value, Mapping) or hasattr(value, "get"):
            for key in (MODE_ID_COLUMN, "ModeID", "mode_id", "mode"):
                resolved = _coerce_int(value.get(key))  # type: ignore[attr-defined]
                if resolved is not None:
                    return resolved
            value = value.get("Mode")  # type: ignore[attr-defined]
    except Exception:
        pass

    direct = _coerce_int(value)
    if direct is not None:
        return direct

    raw = str(value or "").strip().lower()
    if not raw:
        return default
    labels = _reverse_label_map()
    if raw in labels:
        return labels[raw]
    for label, mode in labels.items():
        if label and label in raw:
            return mode
    return default


def safe_mode_int(value: object, default: int | None = None) -> int | None:
    return resolve_mode_id(value, default)


def safe_mode_match(value: object, mode: int) -> bool:
    return resolve_mode_id(value) == int(mode)


def is_mode6(value: object) -> bool:
    return safe_mode_match(value, 6)


def is_mode7(value: object) -> bool:
    return safe_mode_match(value, 7)


__all__ = [
    "safe_mode_int",
    "resolve_mode_id",
    "safe_mode_match",
    "is_mode6",
    "is_mode7",
    "get_mode_label",
    "get_mode_color",
    "get_mode_description",
    "get_mode_metadata",
    "get_mode_name",
    "get_mode_display",
]
