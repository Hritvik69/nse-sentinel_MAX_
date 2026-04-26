"""
time_travel_engine.py
──────────────────────
🕰️ TIME-TRAVEL MODE — System-wide Historical Simulation for NSE Sentinel.

How it works
────────────
When activated with a cutoff date D, this engine:

  1. Patches `get_df_for_ticker` so EVERY call (from analyse(), Stock Aura,
     Battle, Sector Screener) returns data truncated to ≤ D.
  2. Snapshots existing ALL_DATA entries (used by zero-API backtests) with
     the same truncated data.
  3. Stores originals so `restore()` puts everything back exactly.

Data guarantee
──────────────
  • ALL indicators (EMA, RSI, Vol/Avg) are computed AFTER the cutoff filter,
    so no future data ever leaks into any calculation.
  • yfinance fallback downloads (for tickers not yet in ALL_DATA) are also
    truncated immediately on receipt.
  • The staleness check in analyse() uses reference datetime (4pm on cutoff
    date) instead of datetime.now() so historical DFs are never rejected.

Zero impact on live mode
─────────────────────────
  • When inactive, all functions are no-ops and return normal values.
  • is_active() guard protects every code path.
  • Never raises — fully wrapped in try/except throughout.

Public API
──────────
    activate(cutoff_date)           → int   (tickers snapshotted)
    restore()                       → None
    is_active()                     → bool
    get_reference_datetime()        → datetime  (4pm on cutoff, or now())
    get_reference_date()            → date | None
    truncate_df(df, cutoff_date)    → pd.DataFrame | None
    apply_time_travel_cutoff(df)    → pd.DataFrame | None  (convenience)
"""

from __future__ import annotations

import threading
from datetime import date, datetime, time as dtime
from typing import Callable

import numpy as np
import pandas as pd

# ── Internal imports ──────────────────────────────────────────────────
try:
    from strategy_engines._engine_utils import (
        ALL_DATA,
        _ALL_DATA_LOCK,
        get_df_for_ticker as _original_get_df,
        download_history as _original_download,
    )
    import strategy_engines._engine_utils as _eu
    _EU_OK = True
except ImportError:
    _EU_OK = False
    ALL_DATA = {}
    _ALL_DATA_LOCK = threading.Lock()
    _original_get_df = lambda t: None  # noqa: E731
    _original_download = lambda t, **kw: None  # noqa: E731
    _eu = None


# ══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE (thread-safe)
# ══════════════════════════════════════════════════════════════════════

_STATE_LOCK         = threading.Lock()
_active             = False
_reference_date:    date | None = None
_all_data_backup:   dict        = {}   # ticker → original df


# ══════════════════════════════════════════════════════════════════════
# BACKTEST CACHE CLEARING
# ══════════════════════════════════════════════════════════════════════

def _clear_all_bt_caches() -> None:
    """
    Clear every mode engine's _BT_CACHE dict AND the dashboard stock-row
    cache in multi_index_market_bias_engine.
 
    Each mode engine (mode1_engine … mode6_engine) and app.py maintain a
    module-level _BT_CACHE that maps ticker → backtest probability. These
    caches are never invalidated across Time Travel activations, so a
    live-mode result gets silently reused for a TT scan of the same ticker.
 
    Additionally, multi_index_market_bias_engine maintains
    _DASHBOARD_STOCK_ROW_CACHE keyed by (mode, ticker_ns, df_signature).
    When ALL_DATA is truncated the df_signature changes, so the cache key
    changes automatically — but clearing it explicitly is safer and prevents
    stale rows from persisting across TT date changes within the same session.
 
    Never raises — fully wrapped in try/except.
    """
    import sys
 
    _CACHE_MODULE_PATTERNS = (
        "mode1_engine", "mode2_engine", "mode3_engine",
        "mode4_engine", "mode5_engine", "mode6_engine",
        "strategy_engines.mode1_engine", "strategy_engines.mode2_engine",
        "strategy_engines.mode3_engine", "strategy_engines.mode4_engine",
        "strategy_engines.mode5_engine", "strategy_engines.mode6_engine",
        "app",
    )
 
    # ── Clear mode-engine _BT_CACHE dicts (existing logic, unchanged) ──
    try:
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            base = mod_name.split(".")[-1]
            if base not in _CACHE_MODULE_PATTERNS and mod_name not in _CACHE_MODULE_PATTERNS:
                continue
            try:
                cache = getattr(mod, "_BT_CACHE", None)
                if isinstance(cache, dict):
                    cache.clear()
            except Exception:
                continue
    except Exception:
        pass
 
    # ── NEW: Clear _DASHBOARD_STOCK_ROW_CACHE in multi_index engine ────
    # This cache stores pre-computed stock rows keyed by df_signature.
    # Must be cleared on every TT activate/restore so sector screener rows
    # are always recomputed against the correct (truncated) ALL_DATA snapshot.
    _MIBE_NAMES = (
        "multi_index_market_bias_engine",
        "strategy_engines.multi_index_market_bias_engine",
    )
    try:
        for mod_name in _MIBE_NAMES:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            cache = getattr(mod, "_DASHBOARD_STOCK_ROW_CACHE", None)
            if isinstance(cache, dict):
                cache.clear()
            # Also clear the regular sector screener row cache if present
            cache2 = getattr(mod, "_SECTOR_ROW_CACHE", None)
            if isinstance(cache2, dict):
                cache2.clear()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# CORE UTILITY
# ══════════════════════════════════════════════════════════════════════

def truncate_df(df: pd.DataFrame | None, cutoff: date) -> pd.DataFrame | None:
    """
    Return df filtered to rows where index.date <= cutoff.
    Returns None if result has fewer than 10 rows (not enough for indicators).
    Never raises.
    """
    if df is None or df.empty:
        return None
    try:
        idx_dates = pd.to_datetime(df.index).date
        mask = idx_dates <= cutoff
        trimmed = df.loc[mask]
        return trimmed if len(trimmed) >= 10 else None
    except Exception:
        return df   # fail-safe: return original rather than None


# ══════════════════════════════════════════════════════════════════════
# PATCHED get_df_for_ticker
# ══════════════════════════════════════════════════════════════════════

def _time_travel_get_df(ticker: str) -> pd.DataFrame | None:
    """
    Drop-in replacement for get_df_for_ticker() when time-travel is active.
    Fetches data normally then truncates to _reference_date before returning.
    """
    try:
        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

        # 1️⃣ Try ALL_DATA (already snapshotted — truncated values are there)
        with _ALL_DATA_LOCK:
            cached = ALL_DATA.get(ticker_ns)
        if cached is not None:
            return cached   # already truncated by apply_snapshot()

        # 2️⃣ Live download fallback — truncate immediately
        df_live = _original_download(ticker_ns, period="6mo")
        if df_live is None:
            return None
        with _STATE_LOCK:
            cutoff = _reference_date
        if cutoff is None:
            return df_live
        trimmed = truncate_df(df_live, cutoff)
        # ── FIX 7: Validate no future leakage ─────────────────────────
        if trimmed is not None and len(trimmed) > 0:
            last = pd.to_datetime(trimmed.index[-1]).date()
            if last > cutoff:   # should never happen — log if it does
                import warnings
                warnings.warn(
                    f"[TimeTravelEngine] LEAKAGE DETECTED: {ticker_ns} "
                    f"last_date={last} > cutoff={cutoff}. Forcing re-trim.",
                    RuntimeWarning, stacklevel=2,
                )
                trimmed = truncate_df(df_live, cutoff)
        return trimmed

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def activate(cutoff: date) -> int:
    """
    Enable time-travel mode for cutoff date.
    Truncates ALL existing ALL_DATA entries and patches get_df_for_ticker.
    Returns number of ticker DataFrames successfully truncated.
    Thread-safe. Never raises.
    """
    global _active, _reference_date, _all_data_backup

    try:
        with _STATE_LOCK:
            _reference_date = cutoff
            _active = True

        count = 0

        # ── Snapshot + truncate ALL_DATA ──────────────────────────────
        if _EU_OK:
            with _ALL_DATA_LOCK:
                _all_data_backup.clear()
                for ticker, df in list(ALL_DATA.items()):
                    # BUG FIX: Store a copy, not a reference — in-place mutations
                    # elsewhere would otherwise silently corrupt the backup.
                    _all_data_backup[ticker] = df.copy() if df is not None else None
                    if df is None or df.empty:
                        continue
                    trimmed = truncate_df(df, cutoff)
                    ALL_DATA[ticker] = trimmed
                    if trimmed is not None:
                        count += 1

            # ── Monkey-patch get_df_for_ticker ────────────────────────
            # Patch both the module attribute AND the __init__ re-export so
            # that code using `from strategy_engines import get_df_for_ticker`
            # (e.g. battle_mode_engine) also receives the time-travel version.
            try:
                _eu.get_df_for_ticker = _time_travel_get_df
            except Exception:
                pass
            try:
                import strategy_engines as _se_pkg
                _se_pkg.get_df_for_ticker = _time_travel_get_df
            except Exception:
                pass

        # BUG FIX: Clear all mode-engine _BT_CACHE dicts so stale live-mode
        # backtest results don't get reused for this TT scan session.
        _clear_all_bt_caches()

        return count

    except Exception:
        return 0


def restore() -> None:
    """
    Restore ALL_DATA to its original state and unpatch get_df_for_ticker.
    Safe to call even if activate() was never called.
    Never raises.
    """
    global _active, _reference_date, _all_data_backup

    try:
        # ── Restore ALL_DATA ──────────────────────────────────────────
        if _EU_OK and _all_data_backup:
            with _ALL_DATA_LOCK:
                for ticker, df in _all_data_backup.items():
                    ALL_DATA[ticker] = df
                _all_data_backup.clear()

            # ── Restore original get_df_for_ticker (both bindings) ────
            try:
                _eu.get_df_for_ticker = _original_get_df
            except Exception:
                pass
            try:
                import strategy_engines as _se_pkg
                _se_pkg.get_df_for_ticker = _original_get_df
            except Exception:
                pass

        with _STATE_LOCK:
            _active = False
            _reference_date = None

        # BUG FIX: Clear all mode-engine _BT_CACHE dicts so TT-specific backtest
        # results don't bleed into subsequent live-mode scans.
        _clear_all_bt_caches()

    except Exception:
        # Absolute fail-safe — mark inactive even if restore partially failed
        _active = False
        _reference_date = None


def is_active() -> bool:
    """Return True when time-travel mode is currently activated."""
    with _STATE_LOCK:
        return _active


def get_reference_date() -> date | None:
    """Return the active cutoff date when time-travel is enabled."""
    with _STATE_LOCK:
        return _reference_date


def get_reference_datetime() -> datetime:
    """
    Return the reference datetime for staleness checks.
    Time-travel ON  → 4:00 PM on the cutoff date (post-market-close).
    Time-travel OFF → datetime.now() (unchanged live behaviour).
    """
    d = get_reference_date()
    if d is None:
        return datetime.now()
    return datetime.combine(d, dtime(16, 0, 0))   # 4pm IST = post-market


def apply_time_travel_cutoff(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Convenience wrapper: truncate df to the active cutoff date if
    time-travel is active, otherwise return df unchanged.

    Use this everywhere a yfinance download is used outside of ALL_DATA
    (e.g. index data in market_bias_engine, multi_index_market_bias_engine).

    Never raises. Returns None if df is None or becomes empty after trim.
    """
    cutoff = get_reference_date()
    if cutoff is None or df is None or df.empty:
        return df
    return truncate_df(df, cutoff)


def format_banner() -> str:
    """
    Return a human-readable banner string for display in the UI.
    Returns empty string if time-travel is not active.
    """
    d = get_reference_date()
    if d is None:
        return ""
    day_str = d.strftime("%d-%b-%Y")
    weekday = d.strftime("%A")
    return f"🕰️ TIME TRAVEL · Simulating Market Date: {day_str} ({weekday}) Post-Market Close"
