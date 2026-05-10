from __future__ import annotations

import contextvars
import functools
import threading
from datetime import date, datetime, time as dtime
from typing import Callable, TypeVar

import pandas as pd

try:
    from strategy_engines._engine_utils import ALL_DATA, _ALL_DATA_LOCK
except Exception:
    ALL_DATA = {}  # type: ignore[assignment]
    _ALL_DATA_LOCK = threading.RLock()  # type: ignore[assignment]

T = TypeVar("T")

_TT_DATE: contextvars.ContextVar[date | None] = contextvars.ContextVar(
    "nse_sentinel_time_travel_date",
    default=None,
)
_TOKEN_LOCAL = threading.local()
_CACHE_LOCK = threading.RLock()
_TRUNCATED_CACHE: dict[tuple[str, str, tuple[int, str]], pd.DataFrame] = {}
_MAX_TRUNCATED_CACHE = 2000


def _coerce_date(value: object) -> date | None:
    try:
        if value is None:
            return None
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _frame_signature(df: pd.DataFrame | None) -> tuple[int, str]:
    try:
        if df is None or df.empty:
            return (0, "")
        last = pd.to_datetime(df.index[-1], errors="coerce")
        last_text = "" if pd.isna(last) else pd.Timestamp(last).isoformat()
        return (int(len(df)), last_text)
    except Exception:
        return (0, "")


def _cache_key(ticker: str, cutoff: date, df: pd.DataFrame | None) -> tuple[str, str, tuple[int, str]]:
    ticker_key = str(ticker or "").strip().upper()
    return (ticker_key, cutoff.isoformat(), _frame_signature(df))


def _evict_cache_if_needed() -> None:
    overflow = len(_TRUNCATED_CACHE) - _MAX_TRUNCATED_CACHE
    if overflow <= 0:
        return
    for key in list(_TRUNCATED_CACHE.keys())[:overflow]:
        _TRUNCATED_CACHE.pop(key, None)


def _clear_all_bt_caches() -> None:
    import sys

    module_bases = {
        "mode1_engine",
        "mode2_engine",
        "mode3_engine",
        "mode4_engine",
        "mode5_engine",
        "mode6_engine",
        "mode7_engine",
        "app",
    }
    try:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            base = mod_name.split(".")[-1]
            if base not in module_bases:
                continue
            cache = getattr(mod, "_BT_CACHE", None)
            if isinstance(cache, dict):
                cache.clear()
    except Exception:
        pass

    for mod_name in (
        "multi_index_market_bias_engine",
        "strategy_engines.multi_index_market_bias_engine",
    ):
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            for attr in ("_DASHBOARD_STOCK_ROW_CACHE", "_SECTOR_ROW_CACHE"):
                cache = getattr(mod, attr, None)
                if isinstance(cache, dict):
                    cache.clear()
        except Exception:
            continue


def truncate_df(df: pd.DataFrame | None, cutoff: date, min_rows: int = 10) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    cutoff_date = _coerce_date(cutoff)
    if cutoff_date is None:
        return None
    try:
        out = df.copy()
        parsed_index = pd.to_datetime(out.index, errors="coerce")
        if pd.isna(parsed_index).any():
            return None
        idx_dates = parsed_index.date
        out = out.loc[idx_dates <= cutoff_date].copy()
        if len(out) < max(1, int(min_rows)):
            return None
        out.attrs.update(dict(getattr(df, "attrs", {}) or {}))
        out.attrs["_nse_tt_cutoff"] = cutoff_date.isoformat()
        source = str(out.attrs.get("_nse_data_source", "") or "")
        if source and not source.endswith("_time_travel"):
            out.attrs["_nse_data_source"] = f"{source}_time_travel"
        elif not source:
            out.attrs["_nse_data_source"] = "time_travel"
        out.attrs["_nse_market_date"] = cutoff_date.isoformat()
        return out
    except Exception:
        return None


def get_cached_frame(
    ticker: str,
    df: pd.DataFrame | None,
    cutoff: date | None = None,
) -> pd.DataFrame | None:
    cutoff_date = _coerce_date(cutoff) or get_reference_date()
    if cutoff_date is None or df is None:
        return None
    key = _cache_key(ticker, cutoff_date, df)
    with _CACHE_LOCK:
        cached = _TRUNCATED_CACHE.get(key)
        return cached.copy() if isinstance(cached, pd.DataFrame) else None


def get_cached_for_ticker(ticker: str, cutoff: date | None = None) -> pd.DataFrame | None:
    cutoff_date = _coerce_date(cutoff) or get_reference_date()
    if cutoff_date is None:
        return None
    ticker_key = str(ticker or "").strip().upper()
    cutoff_text = cutoff_date.isoformat()
    with _CACHE_LOCK:
        for key in reversed(list(_TRUNCATED_CACHE.keys())):
            if key[0] == ticker_key and key[1] == cutoff_text:
                cached = _TRUNCATED_CACHE.get(key)
                return cached.copy() if isinstance(cached, pd.DataFrame) else None
    return None


def cache_frame(
    ticker: str,
    df: pd.DataFrame | None,
    cutoff: date | None = None,
    *,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    cutoff_date = _coerce_date(cutoff) or get_reference_date()
    if cutoff_date is None or df is None:
        return df
    cached = get_cached_frame(ticker, df, cutoff_date)
    if cached is not None:
        return cached
    trimmed = truncate_df(df, cutoff_date, min_rows=min_rows)
    if trimmed is None:
        return None
    key = _cache_key(ticker, cutoff_date, df)
    with _CACHE_LOCK:
        _TRUNCATED_CACHE[key] = trimmed.copy()
        _evict_cache_if_needed()
    return trimmed


def clear_cache(cutoff: date | None = None) -> None:
    cutoff_date = _coerce_date(cutoff)
    with _CACHE_LOCK:
        if cutoff_date is None:
            _TRUNCATED_CACHE.clear()
            return
        cutoff_text = cutoff_date.isoformat()
        for key in list(_TRUNCATED_CACHE.keys()):
            if key[1] == cutoff_text:
                _TRUNCATED_CACHE.pop(key, None)


def activate(cutoff: date) -> int:
    cutoff_date = _coerce_date(cutoff)
    if cutoff_date is None:
        return 0

    token = _TT_DATE.set(cutoff_date)
    stack = getattr(_TOKEN_LOCAL, "tokens", None)
    if stack is None:
        stack = []
        _TOKEN_LOCAL.tokens = stack
    stack.append(token)

    warmed = 0
    try:
        with _ALL_DATA_LOCK:
            items = list(ALL_DATA.items())
        for ticker, df in items:
            if cache_frame(ticker, df, cutoff_date) is not None:
                warmed += 1
    except Exception:
        warmed = 0

    _clear_all_bt_caches()
    return warmed


def restore() -> None:
    stack = getattr(_TOKEN_LOCAL, "tokens", None)
    try:
        if stack:
            token = stack.pop()
            _TT_DATE.reset(token)
        else:
            _TT_DATE.set(None)
    except Exception:
        _TT_DATE.set(None)
    _clear_all_bt_caches()


def is_active() -> bool:
    return get_reference_date() is not None


def get_reference_date() -> date | None:
    return _coerce_date(_TT_DATE.get())


def get_reference_datetime() -> datetime:
    ref = get_reference_date()
    if ref is None:
        return datetime.now()
    return datetime.combine(ref, dtime(16, 0, 0))


def apply_time_travel_cutoff(df: pd.DataFrame | None, min_rows: int = 10) -> pd.DataFrame | None:
    cutoff = get_reference_date()
    if cutoff is None or df is None or df.empty:
        return df
    return truncate_df(df, cutoff, min_rows=min_rows)


def context_callable(fn: Callable[..., T], /, *args, **kwargs) -> Callable[[], T]:
    ctx = contextvars.copy_context()
    return functools.partial(ctx.run, fn, *args, **kwargs)


def submit_with_context(executor, fn: Callable[..., T], /, *args, **kwargs):
    return executor.submit(context_callable(fn, *args, **kwargs))


def format_banner() -> str:
    ref = get_reference_date()
    if ref is None:
        return ""
    return (
        f"TIME TRAVEL - Simulating Market Date: "
        f"{ref.strftime('%d-%b-%Y')} ({ref.strftime('%A')}) Post-Market Close"
    )
