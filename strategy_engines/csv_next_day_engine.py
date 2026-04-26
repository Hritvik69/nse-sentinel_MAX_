"""
Compatibility wrapper for the canonical CSV next-day engine.

This keeps the root-level implementation as the single source of truth while
preserving older imports that still reference strategy_engines.
"""

from __future__ import annotations

from importlib import import_module

_ENGINE = import_module("csv_next_day_engine")

run_csv_next_day = _ENGINE.run_csv_next_day
get_csv_next_day_cache_status = getattr(_ENGINE, "get_csv_next_day_cache_status")
CSV_NEXT_DAY_RESULT_COLUMNS = list(getattr(_ENGINE, "CSV_NEXT_DAY_RESULT_COLUMNS", []))

__all__ = [
    "run_csv_next_day",
    "get_csv_next_day_cache_status",
    "CSV_NEXT_DAY_RESULT_COLUMNS",
]
