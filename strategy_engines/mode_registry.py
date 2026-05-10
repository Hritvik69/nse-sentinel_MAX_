"""
Central mode metadata registry for NSE Sentinel.

This file is intentionally small and side-effect-free.  It is the single
source of truth for mode names, labels, colors, sidebar labels, and filter
descriptions used by the Streamlit UI and companion engines.
"""

from __future__ import annotations

from copy import deepcopy


MODE_METADATA: dict[int, dict[str, object]] = {
    1: {
        "name": "Momentum",
        "short_name": "Momentum",
        "emoji": "\U0001f7e2",
        "color": "#00d4a8",
        "pill_class": "pill-m1",
        "description": "Strict momentum continuation scanner",
        "display_name": "Momentum",
        "display_num": 1,
        "ui_label": None,
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("Volume", "> 1.5x avg"),
            ("RSI", "52 - 74"),
            ("Use Case", "Strict Momentum"),
        ],
    },
    2: {
        "name": "Balanced",
        "short_name": "Balanced",
        "emoji": "\U0001f535",
        "color": "#0094ff",
        "pill_class": "pill-m2",
        "description": "Balanced momentum and quality scanner",
        "display_name": "Balanced",
        "display_num": 2,
        "ui_label": None,
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("Volume", "> 1.3x avg"),
            ("RSI", "50 - 72"),
            ("Use Case", "Balanced Scan"),
        ],
    },
    3: {
        "name": "Relaxed",
        "short_name": "Relaxed",
        "emoji": "\U0001f7e1",
        "color": "#f0b429",
        "pill_class": "pill-m3",
        "description": "Relaxed wide-scan momentum mode",
        "display_name": "Relaxed",
        "display_num": 1,
        "ui_label": "\U0001f7e1  Mode 1 - Relaxed (Wide Scan)",
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("Volume", "> 1.3x avg"),
            ("RSI", "50 - 72"),
            ("Price Floor", "Rs 50"),
            ("20D High", "Within 5%"),
            ("Use Case", "Wide Scan"),
        ],
    },
    4: {
        "name": "Institutional",
        "short_name": "Institutional",
        "emoji": "\U0001f7e3",
        "color": "#b08cff",
        "pill_class": "pill-m3",
        "description": "Institutional relative-strength scanner",
        "display_name": "Institutional",
        "display_num": 4,
        "ui_label": None,
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("Volume", "> 1.3x avg"),
            ("RSI", "52 - 72"),
            ("Use Case", "Institutional RS"),
        ],
    },
    5: {
        "name": "Intraday",
        "short_name": "Intraday",
        "emoji": "\U0001f7e0",
        "color": "#ff8c00",
        "pill_class": "pill-m5",
        "description": "Tomorrow push and intraday momentum scanner",
        "display_name": "Intraday",
        "display_num": 3,
        "ui_label": "\U0001f7e2  Mode 3 - Intraday",
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("Volume", "> 1.1x avg"),
            ("RSI", "52 - 60"),
            ("Price Floor", "Rs 20"),
            ("10D High", "Break above"),
            ("Use Case", "Tomorrow Push"),
        ],
    },
    6: {
        "name": "Swing",
        "short_name": "Swing",
        "emoji": "\U0001f534",
        "color": "#ff4d6d",
        "pill_class": "pill-m6",
        "description": "Swing continuation scanner",
        "display_name": "Swing",
        "display_num": 2,
        "ui_label": "\U0001f534  Mode 2 - Swing",
        "filter_rules": [
            ("EMA Trend", "Close > EMA20 > EMA50"),
            ("EMA20 Slope", "Rising"),
            ("Volume", "> 1.3x avg & > prev"),
            ("RSI", "53 - 59"),
            ("Price Floor", "Rs 40"),
            ("10D High", "Break above"),
        ],
    },
    7: {
        "name": "Momentum (S&R)",
        "short_name": "Momentum S&R",
        "emoji": "\U0001f7e3",
        "color": "#b08cff",
        "pill_class": "pill-m7",
        "description": "Support + Resistance Momentum Scanner",
        "display_name": "Momentum (S&R)",
        "display_num": 3,
        "ui_label": "\U0001f7e3  Mode 3 - MOMENTUM (S&R)",
        "tooltip": (
            "Detects clean breakout structures, support bounces, and "
            "institutional momentum with volume confirmation."
        ),
        "filter_rules": [
            ("S&R Structure", "Support + resistance clean"),
            ("Volume", "1.3x+ confirmation"),
            ("RSI", "52 - 70 controlled"),
            ("Breakout Zone", "-5% to +2.5%"),
            ("EMA Trend", "Price > EMA20 > EMA50"),
            ("Use Case", "Momentum S&R"),
        ],
    },
}

UI_MODE_ORDER: tuple[int, ...] = (3, 6, 5, 7)


def get_mode_metadata(mode: int, *, copy: bool = True) -> dict[str, object]:
    meta = MODE_METADATA.get(int(mode), MODE_METADATA[3])
    return deepcopy(meta) if copy else meta


def get_mode_name(mode: int) -> str:
    return str(get_mode_metadata(mode, copy=False).get("name", f"Mode {mode}"))


def get_mode_description(mode: int) -> str:
    return str(get_mode_metadata(mode, copy=False).get("description", ""))


def get_mode_color(mode: int) -> str:
    return str(get_mode_metadata(mode, copy=False).get("color", "#00d4a8"))


def get_mode_pill_class(mode: int) -> str:
    return str(get_mode_metadata(mode, copy=False).get("pill_class", "pill-m1"))


def get_mode_display(mode: int) -> dict[str, object]:
    meta = get_mode_metadata(mode, copy=False)
    return {
        "display_num": meta.get("display_num", mode),
        "display_name": meta.get("display_name", meta.get("name", f"Mode {mode}")),
    }


def get_mode_filter_rules(mode: int) -> list[tuple[str, str]]:
    rules = get_mode_metadata(mode, copy=False).get("filter_rules", [])
    return list(rules) if isinstance(rules, list) else []


def get_mode_label(mode: int) -> str:
    meta = get_mode_metadata(mode, copy=False)
    return f"{meta.get('emoji', '')} {meta.get('name', f'Mode {mode}')}".strip()


def get_mode_label_map() -> dict[int, str]:
    return {mode: get_mode_label(mode) for mode in sorted(MODE_METADATA)}


def get_mode_names() -> dict[int, str]:
    return {mode: str(meta["name"]) for mode, meta in MODE_METADATA.items()}


def get_mode_colors() -> dict[int, str]:
    return {mode: str(meta["color"]) for mode, meta in MODE_METADATA.items()}


def get_mode_pill_classes() -> dict[int, str]:
    return {mode: str(meta["pill_class"]) for mode, meta in MODE_METADATA.items()}


def get_ui_mode_meta() -> dict[int, dict[str, object]]:
    return {mode: get_mode_display(mode) for mode in UI_MODE_ORDER}


def get_mode_map() -> dict[str, int]:
    out: dict[str, int] = {}
    for mode in UI_MODE_ORDER:
        label = get_mode_metadata(mode, copy=False).get("ui_label")
        if label:
            out[str(label)] = mode
    return out


__all__ = [
    "MODE_METADATA",
    "UI_MODE_ORDER",
    "get_mode_metadata",
    "get_mode_name",
    "get_mode_description",
    "get_mode_color",
    "get_mode_pill_class",
    "get_mode_display",
    "get_mode_filter_rules",
    "get_mode_label",
    "get_mode_label_map",
    "get_mode_names",
    "get_mode_colors",
    "get_mode_pill_classes",
    "get_ui_mode_meta",
    "get_mode_map",
]
