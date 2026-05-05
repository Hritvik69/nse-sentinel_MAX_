r"""
NSE Sentinel — Production-Ready Streamlit App  (Enhanced Edition)
Dark terminal aesthetic | Multi-strategy scanner | 1000+ NSE stocks

Run from the APP3 root folder (the folder that contains app.py):
    cd C:\Users\HP\Downloads\app3
    
.\.venv\Scripts\python.exe -m streamlit run app.py

CHANGES vs original:
  • Scoring layer  (compute_score)         — added AFTER scan, never touches filters
  • Backtest prob  (compute_backtest_prob)  — added AFTER scan
  • ML probability (train_model_once /
                    predict_ml_probability) — added AFTER scan
  • Final rank     (enhance_results)        — combines the three
  • Bull-trap warning                       — display only
  • Top Picks section                       — display only
  • All existing analyse() / run_scan()
    logic is 100 % untouched.
"""
#https://nse-sentinelmax-msrfjdkwmksf6jama4jvmx.streamlit.app/
#.\.venv\Scripts\python.exe -m streamlit run app.py
from __future__ import annotations

# ── PATH FIX: ensure this file's own directory is always on sys.path ──
# Fixes "No module named 'app_sector_screener_dashboard'" and similar
# errors when Streamlit is launched from a different working directory.
import importlib as _importlib
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
# If app.py is inside a sub-folder, also add its parent so that
# strategy_engines/ package is importable from the parent level.
_PARENT = _os.path.dirname(_HERE)
if _os.path.isdir(_os.path.join(_PARENT, "strategy_engines")) and _PARENT not in _sys.path:
    _sys.path.insert(0, _PARENT)

def _default_learning_status() -> dict:
    return {
        "trained": False,
        "samples": 0,
        "stock_samples": 0,
        "sector_samples": 0,
        "accuracy_pct": None,
        "last_trained": "",
        "source": "none",
        "message": "Learning engine unavailable.",
        "regime_encoder": {},
        "sector_encoder": {},
    }


def _load_learning_engine_module():
    """
    Load the project learning engine without leaving a half-imported module
    behind in sys.modules. Streamlit Cloud can otherwise surface that as
    "cannot import name ..." on reruns.
    """
    for module_name in ("learning_engine", "trade_decision_engine"):
        try:
            if module_name == "learning_engine":
                _sys.modules.pop("learning_engine", None)
            module = _importlib.import_module(module_name)
            if module_name != "learning_engine":
                _sys.modules["learning_engine"] = module
            return module
        except Exception:
            if module_name == "learning_engine":
                _sys.modules.pop("learning_engine", None)
            continue
    return None


_learning_engine = _load_learning_engine_module()


def _fallback_train_learning_model():
    return {
        "model": None,
        "scaler": None,
        "status": _default_learning_status(),
    }

import io
import html
import json
import re
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

if _learning_engine is not None:
    get_training_status = getattr(_learning_engine, "get_training_status", _default_learning_status)
    predict_success = getattr(_learning_engine, "predict_success", lambda row: 50.0)
    restore_learning_bundle = getattr(_learning_engine, "restore_learning_bundle", lambda bundle: False)
    train_learning_model = getattr(_learning_engine, "train_learning_model", _fallback_train_learning_model)
else:
    get_training_status = _default_learning_status
    predict_success = lambda row: 50.0
    restore_learning_bundle = lambda bundle: False
    train_learning_model = _fallback_train_learning_model

from nse_learning_brain import run_learning_cycle
try:
    import data_session_manager as _dsm
except Exception:
    _dsm = None  # type: ignore[assignment]


def _fallback_get_current_window() -> str:
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today = now_ist.date()
    current_time = now_ist.time()
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    if today.weekday() >= 5:
        return "WEEKEND"
    if market_open <= current_time <= market_close:
        return "LIVE"
    if current_time > market_close:
        return "CLOSED"
    return "PRE_MARKET"


def _fallback_previous_weekday(day):
    cur = day
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def _fallback_get_expected_data_date():
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    today = now_ist.date()
    window = _fallback_get_current_window()
    if window in ("LIVE", "CLOSED"):
        return today
    if window == "PRE_MARKET":
        return _fallback_previous_weekday(today - timedelta(days=1))
    return _fallback_previous_weekday(today)


def _fallback_get_snapshot_path(market_date):
    try:
        snap_day = pd.to_datetime(market_date).date()
    except Exception:
        snap_day = _fallback_get_expected_data_date()
    return Path(_HERE) / "data" / "snapshots" / snap_day.isoformat()


def _fallback_snapshot_exists(market_date) -> bool:
    try:
        snap_dir = _fallback_get_snapshot_path(market_date)
        return snap_dir.exists() and len(list(snap_dir.glob("*.csv"))) >= 100
    except Exception:
        return False


def _fallback_get_scan_data_plan() -> dict:
    window = _fallback_get_current_window()
    expected_date = _fallback_get_expected_data_date()
    has_snapshot = _fallback_snapshot_exists(expected_date)
    live_window = "09:30 AM - 04:00 PM IST"
    plan = {
        "window": window,
        "expected_date": expected_date,
        "snapshot_exists": has_snapshot,
        "snapshot_path": _fallback_get_snapshot_path(expected_date),
        "live_window_label": live_window,
        "use_snapshot": False,
        "force_live_refresh": False,
        "save_snapshot_after_scan": False,
        "source_label": "",
        "summary": "",
    }
    plan.update(
        {
            "use_snapshot": False,
            "force_live_refresh": True,
            "save_snapshot_after_scan": False,
            "source_label": "Always live refresh",
            "summary": "Main scanner refreshes live data on every run and skips snapshot startup.",
        }
    )
    return plan


def _fallback_get_data_status_label() -> str:
    plan = _fallback_get_scan_data_plan()
    day_text = str(plan.get("expected_date"))
    return f"Always Live Refresh - {day_text}"


get_current_window = getattr(_dsm, "get_current_window", _fallback_get_current_window)
get_expected_data_date = getattr(_dsm, "get_expected_data_date", _fallback_get_expected_data_date)
get_data_status_label = getattr(_dsm, "get_data_status_label", _fallback_get_data_status_label)
get_scan_data_plan = getattr(_dsm, "get_scan_data_plan", _fallback_get_scan_data_plan)
snapshot_exists = getattr(_dsm, "snapshot_exists", _fallback_snapshot_exists)
save_closing_snapshot = getattr(_dsm, "save_closing_snapshot", lambda *_args, **_kwargs: 0)
load_snapshot_into_ALL_DATA = getattr(_dsm, "load_snapshot_into_ALL_DATA", lambda *_args, **_kwargs: 0)
get_snapshot_path = getattr(_dsm, "get_snapshot_path", _fallback_get_snapshot_path)
read_snapshot_metadata = getattr(_dsm, "read_snapshot_metadata", lambda *_args, **_kwargs: {})

_PREDICTION_FEEDBACK_PATH = Path(_HERE) / "data" / "prediction_feedback_log.csv"
_SECTOR_PREDICTION_PATH = Path(_HERE) / "data" / "sector_predictions.csv"


def _safe_file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime) if path.exists() else 0.0
    except Exception:
        return 0.0


def _weight_bar_html(label: str, value_pct: float, *, warn: bool = False) -> str:
    try:
        safe_value = max(0.0, min(100.0, float(value_pct)))
    except Exception:
        safe_value = 0.0
    color = "#ffb347" if warn else "#00d4ff"
    return (
        f'<div style="margin:6px 0 8px 0;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;">'
        f'<span style="font-size:11px;color:#8ab4d8;">{html.escape(str(label))}</span>'
        f'<span style="font-size:11px;color:{color};">{safe_value:.1f}%</span>'
        f'</div>'
        f'<div style="background:#111926;border-radius:999px;height:7px;margin-top:4px;overflow:hidden;">'
        f'<div style="background:{color};height:7px;width:{safe_value}%;"></div>'
        f'</div>'
        f'</div>'
    )


def _restore_learning_bundle_from_session() -> None:
    try:
        bundle = st.session_state.get("_learning_model_bundle")
        if isinstance(bundle, dict):
            restore_learning_bundle(bundle)
    except Exception:
        pass


def _legacy_learning_status_from_brain(brain_status: dict | None) -> dict:
    try:
        brain = brain_status if isinstance(brain_status, dict) else {}
        meta = brain.get("meta_model", {}) if isinstance(brain.get("meta_model", {}), dict) else {}
        feedback = brain.get("feedback", {}) if isinstance(brain.get("feedback", {}), dict) else {}
        legacy = dict(meta)
        legacy["validated_today"] = int(feedback.get("filled_stock", 0) or 0)
        legacy["sector_validated_today"] = int(feedback.get("filled_sector", 0) or 0)
        legacy["feedback_summary"] = brain.get("feedback_summary", {})
        legacy["brain_status"] = brain
        legacy["message"] = str(meta.get("message", "") or brain.get("error", "") or "Learning status unavailable.")
        return legacy
    except Exception:
        return {"trained": False, "feedback_summary": {}, "brain_status": {}, "message": "Learning status unavailable."}


def _signal_weight_status_from_brain(brain_status: dict | None) -> dict:
    default = {
        "processed": 0,
        "top_signal": "",
        "top_weight": 0.0,
        "weakest_signal": "",
        "weakest_weight": 0.0,
        "report": pd.DataFrame(),
    }
    try:
        brain = brain_status if isinstance(brain_status, dict) else {}
        weights = brain.get("weights", {}) if isinstance(brain.get("weights", {}), dict) else {}
        report = weights.get("report")
        top_signal = weights.get("top_signal", {}) if isinstance(weights.get("top_signal", {}), dict) else {}
        weak_signal = weights.get("weakest_signal", {}) if isinstance(weights.get("weakest_signal", {}), dict) else {}
        return {
            "processed": int(weights.get("processed", 0) or 0),
            "top_signal": str(top_signal.get("signal", "") or ""),
            "top_weight": float(top_signal.get("weight_pct", 0.0) or 0.0),
            "weakest_signal": str(weak_signal.get("signal", "") or ""),
            "weakest_weight": float(weak_signal.get("weight_pct", 0.0) or 0.0),
            "report": report if isinstance(report, pd.DataFrame) else pd.DataFrame(),
        }
    except Exception:
        return default


def _refresh_signal_weight_status(force_update: bool = False) -> dict:
    default = {
        "processed": 0,
        "top_signal": "",
        "top_weight": 0.0,
        "weakest_signal": "",
        "weakest_weight": 0.0,
        "report": pd.DataFrame(),
    }
    try:
        sig = (
            str(get_expected_data_date()),
            round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
            round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        )
        cached_sig = st.session_state.get("_signal_weight_sig")
        cached_status = st.session_state.get("_signal_weight_status")
        if (not force_update) and cached_sig == sig and isinstance(cached_status, dict):
            return cached_status

        brain_status = st.session_state.get("learning_cycle_status")
        if not isinstance(brain_status, dict) or force_update:
            _refresh_feedback_learning_system(force=force_update)
            brain_status = st.session_state.get("learning_cycle_status")

        status = _signal_weight_status_from_brain(brain_status)
        if not status.get("top_signal") and isinstance(cached_status, dict):
            status = cached_status
        fresh_sig = (
            str(get_expected_data_date()),
            round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
            round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        )
        st.session_state["_signal_weight_status"] = status
        st.session_state["_signal_weight_sig"] = fresh_sig
        return status
    except Exception:
        return default


def _refresh_feedback_learning_system(force: bool = False) -> dict:
    try:
        _restore_learning_bundle_from_session()
    except Exception:
        pass

    if bool(st.session_state.get("tt_toggle_val")) or (_TIME_TRAVEL_OK and getattr(_tt, "is_active", lambda: False)()):
        status = dict(get_training_status())
        status["message"] = "Learning refresh paused during Time Travel."
        status["validated_today"] = 0
        st.session_state["_learning_status"] = status
        return status

    current_sig = (
        str(get_expected_data_date()),
        round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
    )
    cached_sig = st.session_state.get("_learning_refresh_sig")
    cached_status = st.session_state.get("_learning_status")
    if (not force) and cached_sig == current_sig and isinstance(cached_status, dict):
        return cached_status

    try:
        from strategy_engines._engine_utils import ALL_DATA
    except Exception:
        ALL_DATA = {}

    try:
        if not isinstance(ALL_DATA, dict):
            ALL_DATA = {}
        if not ALL_DATA:
            load_snapshot_into_ALL_DATA(get_expected_data_date())
    except Exception:
        pass

    try:
        brain_status = run_learning_cycle(ALL_DATA, force=force)
    except Exception:
        brain_status = {}

    bundle = st.session_state.get("_learning_model_bundle")
    if isinstance(bundle, dict):
        try:
            restore_learning_bundle(bundle)
        except Exception:
            pass

    status = _legacy_learning_status_from_brain(brain_status)
    fresh_sig = (
        str(get_expected_data_date()),
        round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
    )
    st.session_state["_learning_status"] = status
    st.session_state["_learning_refresh_sig"] = fresh_sig

    total_validated = int(status.get("validated_today", 0) or 0) + int(status.get("sector_validated_today", 0) or 0)
    if total_validated > 0:
        toast_key = f"{datetime.now().date().isoformat()}|{total_validated}|{fresh_sig}"
        badge_text = f"✅ {total_validated} predictions validated today"
        st.session_state["_validated_today_badge"] = badge_text
        if st.session_state.get("_validated_today_toast_key") != toast_key:
            st.toast(badge_text)
            st.session_state["_validated_today_toast_key"] = toast_key
    else:
        st.session_state.setdefault("_validated_today_badge", "")

    return status

try:
    import gspread
    from google.oauth2.service_account import Credentials
    _GOOGLE_SHEETS_IMPORT_OK = True
except Exception:
    gspread = None  # type: ignore[assignment]
    Credentials = None  # type: ignore[assignment]
    _GOOGLE_SHEETS_IMPORT_OK = False

_VISIBLE_RESULT_LIMIT = 10
from strategy_engines import (
    get_engine_functions,
    get_train_function,
    preload_all,
    backtest_with_preloaded,
    get_df_for_ticker,
)
try:
    import strategy_engines._engine_utils as _engine_utils  # type: ignore[import]
except Exception:
    _engine_utils = None  # type: ignore[assignment]


def _fallback_is_fresh_enough(df, strict: bool = False) -> bool:
    if df is None or getattr(df, "empty", True):
        return False
    if not strict:
        return True
    try:
        last_seen = pd.to_datetime(df.index[-1]).date()
        return last_seen == get_expected_data_date()
    except Exception:
        return True


def _fallback_get_shared_market_frame(
    symbol: str,
    *,
    period: str = "6mo",
    min_rows: int = 30,
    append_nse_suffix: bool = True,
    allow_csv_cache: bool | None = None,
    require_volume: bool = True,
) -> "pd.DataFrame | None":
    raw = str(symbol or "").strip().upper()
    if not raw:
        return None

    ticker = raw if (not append_nse_suffix or raw.endswith(".NS")) else f"{raw}.NS"

    try:
        cached = get_df_for_ticker(ticker)
        if cached is not None and len(cached) >= max(1, int(min_rows)):
            return cached
    except Exception:
        pass

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=12,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.copy()
        df.columns = [str(col).strip().title() for col in df.columns]
        needed = ["Open", "High", "Low", "Close"]
        if not set(needed).issubset(df.columns):
            return None
        if "Volume" not in df.columns:
            if require_volume:
                return None
            df["Volume"] = 0.0
        cols = needed + ["Volume"]
        df = df[cols].copy()
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=needed)
        if require_volume:
            df = df.dropna(subset=["Volume"])
        else:
            df["Volume"] = df["Volume"].fillna(0.0)
        return df.sort_index() if len(df) >= max(1, int(min_rows)) else None
    except Exception:
        return None


_is_fresh_enough = getattr(_engine_utils, "is_fresh_enough", _fallback_is_fresh_enough)
_get_shared_market_frame = getattr(
    _engine_utils,
    "get_shared_market_frame",
    _fallback_get_shared_market_frame,
)
from strategy_engines.nse_autocomplete import (
    configure_nse_stock_search,
    render_nse_stock_input,
)
# preload_history_batch removed — use preload_all() directly

try:
    from strategy_engines.app_sector_screener_dashboard import (  # type: ignore[import]
        render_sector_screener_dashboard,
    )
    _SECTOR_SCREENER_UI_OK = True
except Exception:
    try:
        from app_sector_screener_dashboard import (  # type: ignore[import]
            render_sector_screener_dashboard,
        )
        _SECTOR_SCREENER_UI_OK = True
    except Exception:
        _SECTOR_SCREENER_UI_OK = False

try:
    from strategy_engines.app_sector_explorer_section import (  # type: ignore[import]
        render_sector_explorer_section,
    )
    _SECTOR_EXPLORER_UI_OK = True
    _SECTOR_EXPLORER_UI_ERR = ""
except Exception as _sector_explorer_exc:
    try:
        from app_sector_explorer_section import (  # type: ignore[import]
            render_sector_explorer_section,
        )
        _SECTOR_EXPLORER_UI_OK = True
        _SECTOR_EXPLORER_UI_ERR = ""
    except Exception:
        _SECTOR_EXPLORER_UI_OK = False
        _SECTOR_EXPLORER_UI_ERR = str(_sector_explorer_exc).strip() or "sector explorer import failed"

        def render_sector_explorer_section(ticker_universe=None) -> None:  # type: ignore[misc]
            return None

try:
    from app_sector_prediction_section import render_sector_prediction_section
    _SECTOR_PREDICTION_UI_OK = True
    _SECTOR_PREDICTION_UI_ERR = ""
except Exception as _sector_prediction_exc:
    _SECTOR_PREDICTION_UI_OK = False
    _SECTOR_PREDICTION_UI_ERR = str(_sector_prediction_exc).strip() or "sector prediction import failed"

    def render_sector_prediction_section(*args, **kwargs) -> None:  # type: ignore[misc]
        return None

try:
    from strategy_engines.app_sector_intelligence_section import render_sector_intelligence_section
    _SECTOR_INTELLIGENCE_UI_OK = True
    _SECTOR_INTELLIGENCE_UI_ERR = ""
except Exception as _sector_intelligence_exc:
    try:
        from app_sector_intelligence_section import render_sector_intelligence_section  # type: ignore[import]
        _SECTOR_INTELLIGENCE_UI_OK = True
        _SECTOR_INTELLIGENCE_UI_ERR = ""
    except Exception:
        _SECTOR_INTELLIGENCE_UI_OK = False
        _SECTOR_INTELLIGENCE_UI_ERR = str(_sector_intelligence_exc).strip() or "sector intelligence import failed"

        def render_sector_intelligence_section(*args, **kwargs) -> None:  # type: ignore[misc]
            return None

try:
    from app_breakout_radar_section import render_breakout_radar_section
    _BREAKOUT_SECTION_OK = True
except Exception:
    _BREAKOUT_SECTION_OK = False

    def render_breakout_radar_section(*args, **kwargs):  # type: ignore[misc]
        return None

try:
    from app_live_breakout_pulse_section import render_live_breakout_pulse
    _LIVE_PULSE_SECTION_OK = True
except Exception:
    _LIVE_PULSE_SECTION_OK = False

    def render_live_breakout_pulse(*args, **kwargs):  # type: ignore[misc]
        return None

from app_prediction_chart_section import render_prediction_chart_section
from app_compare_stocks_section import (
    build_compare_source_statuses as _build_compare_source_statuses,
    load_compare_results as _load_compare_results,
    normalize_compare_symbols as _normalize_compare_symbols,
    save_compare_results as _save_compare_results,
    summarize_compare_sources as _summarize_compare_sources,
)

try:
    from nse_animations import inject_animations
    _NSE_ANIMATIONS_OK = True
except Exception:
    _NSE_ANIMATIONS_OK = False

    def inject_animations() -> None:  # type: ignore[misc]
        return None

# AFTER the csv_next_day import block, add:
try:
    from breakout_radar_engine import run_breakout_radar, radar_summary
    _BREAKOUT_RADAR_OK = True
except Exception:
    _BREAKOUT_RADAR_OK = False
    def run_breakout_radar(df=None, cutoff_date=None): return pd.DataFrame()
    def radar_summary(df): return {}



try:
    from csv_next_day_engine import run_csv_next_day  # type: ignore[import]
    _CSV_NEXT_DAY_ENGINE_OK = True
except Exception:
    _CSV_NEXT_DAY_ENGINE_OK = False

    def run_csv_next_day(df=None, cutoff_date=None):  # type: ignore[misc]
        return pd.DataFrame()

warnings.filterwarnings("ignore")

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_TOMORROW_STORE_PATH = Path(_HERE) / "data" / "tomorrow_picks_store.json"
_TICKER_MASTER_STORE_PATH = Path(_HERE) / "data" / "ticker_master_list.json"

_TOMORROW_SECTION_ORDER = ("relax", "swing", "intraday", "breakout")
_TOMORROW_SECTION_META = {
    "relax": {
        "label": "Relax",
        "accent": "#22c55e",
        "caption": "Wide-scan continuation and easier entries",
    },
    "swing": {
        "label": "Swing",
        "accent": "#f0b429",
        "caption": "Multi-day momentum and cleaner swing structure",
    },
    "intraday": {
        "label": "Intraday",
        "accent": "#00d4a8",
        "caption": "Fast tactical setups and tighter timing",
    },
    "breakout": {
        "label": "Breakout",
        "accent": "#ff6b6b",
        "caption": "Radar and live-breakout momentum names",
    },
}


@st.cache_resource(ttl=0)
def _get_sheet():
    """
    Returns the gspread worksheet object.
    Same connection reused across all users.
    Returns None if credentials not configured.
    """
    try:
        if not _GOOGLE_SHEETS_IMPORT_OK or gspread is None or Credentials is None:
            return None
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            creds_dict, scopes=_SCOPES
        )
        client = gspread.authorize(creds)
        sheet_id = st.secrets["google_sheets"]["sheet_id"]
        sh = client.open_by_key(sheet_id)
        # Use first worksheet
        return sh.sheet1
    except Exception:
        return None


def _load_picks() -> dict:
    """
    Load picks and notes from Google Sheet.
    Sheet layout:
      Row 1: headers — "picks" | "notes"
      Row 2: JSON list of picks | notes text
    Returns {"picks": [...], "notes": "..."}
    """
    default = _normalize_tomorrow_store(None)
    try:
        ws = _get_sheet()
        if ws is None:
            return default
        # Read row 2 (data row)
        picks_json = ws.cell(2, 1).value or "[]"
        notes_text = ws.cell(2, 2).value or ""
        payload = json.loads(picks_json)
        if isinstance(payload, dict):
            if not str(payload.get("notes", "") or "").strip() and notes_text:
                payload["notes"] = notes_text
            return _normalize_tomorrow_store(payload)
        return _normalize_tomorrow_store({"picks": payload, "notes": notes_text})
    except Exception:
        return default


def _save_picks(store: dict) -> None:
    """
    Save picks and notes back to Google Sheet.
    Writes row 2: JSON(picks) | notes
    """
    try:
        ws = _get_sheet()
        if ws is None:
            return
        normalized = _normalize_tomorrow_store(store)
        # Write headers if empty
        if not ws.cell(1, 1).value:
            ws.update("A1:B1", [["picks", "notes"]])
        ws.update("A2:B2", [[
            json.dumps(normalized, ensure_ascii=False),
            normalized["notes"],
        ]])
    except Exception:
        pass


def _normalize_tomorrow_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _normalize_tomorrow_symbols(values: list[object] | tuple[object, ...] | None, limit: int = 20) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        symbol = _normalize_tomorrow_symbol(raw)
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
        if len(symbols) >= limit:
            break
    return symbols


def _tomorrow_section_defaults() -> dict[str, list[str]]:
    return {bucket: [] for bucket in _TOMORROW_SECTION_ORDER}


def _normalize_tomorrow_bucket(bucket: object) -> str:
    raw = str(bucket or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in _TOMORROW_SECTION_META:
        return raw
    return "relax"


def _tomorrow_bucket_for_mode(mode_value: object) -> str:
    try:
        mode_int = int(mode_value)
    except Exception:
        mode_int = 3
    return {
        3: "relax",
        6: "swing",
        5: "intraday",
    }.get(mode_int, "relax")


def _tomorrow_section_label(bucket: object) -> str:
    bucket_key = _normalize_tomorrow_bucket(bucket)
    return str(_TOMORROW_SECTION_META.get(bucket_key, {}).get("label", "Relax"))


def _tomorrow_flatten_sections(sections: dict[str, list[object]] | None, limit: int = 20) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    bucket_map = sections if isinstance(sections, dict) else {}
    for bucket in _TOMORROW_SECTION_ORDER:
        for raw in list(bucket_map.get(bucket, [])):
            symbol = _normalize_tomorrow_symbol(raw)
            if symbol and symbol not in seen:
                merged.append(symbol)
                seen.add(symbol)
            if len(merged) >= limit:
                return merged
    return merged


def _apply_tomorrow_sections_limit(sections: dict[str, list[object]] | None, limit: int = 20) -> dict[str, list[str]]:
    limited = _tomorrow_section_defaults()
    seen: set[str] = set()
    bucket_map = sections if isinstance(sections, dict) else {}
    total = 0
    for bucket in _TOMORROW_SECTION_ORDER:
        for raw in list(bucket_map.get(bucket, [])):
            symbol = _normalize_tomorrow_symbol(raw)
            if not symbol or symbol in seen:
                continue
            if total >= limit:
                return limited
            limited[bucket].append(symbol)
            seen.add(symbol)
            total += 1
    return limited


def _tomorrow_section_membership(sections: dict[str, list[object]] | None) -> dict[str, str]:
    membership: dict[str, str] = {}
    bucket_map = sections if isinstance(sections, dict) else {}
    for bucket in _TOMORROW_SECTION_ORDER:
        for symbol in _normalize_tomorrow_symbols(bucket_map.get(bucket, []), limit=20):
            membership[symbol] = bucket
    return membership


def _normalize_tomorrow_store(store: dict | None) -> dict:
    default = {"picks": [], "notes": "", "sections": _tomorrow_section_defaults()}
    if not isinstance(store, dict):
        return default.copy()

    raw_picks = store.get("picks", [])
    if not isinstance(raw_picks, list):
        raw_picks = []

    raw_sections = store.get("sections", {})
    section_seed = _tomorrow_section_defaults()
    if isinstance(raw_sections, dict):
        for bucket in _TOMORROW_SECTION_ORDER:
            raw_values = raw_sections.get(bucket, [])
            if isinstance(raw_values, (list, tuple)):
                section_seed[bucket] = list(raw_values)

    known_symbols = set(_tomorrow_flatten_sections(section_seed, limit=20))
    for item in _normalize_tomorrow_symbols(raw_picks, limit=20):
        if item not in known_symbols:
            section_seed["relax"].append(item)
            known_symbols.add(item)

    sections = _apply_tomorrow_sections_limit(section_seed, limit=20)
    picks = _tomorrow_flatten_sections(sections, limit=20)

    notes_text = store.get("notes", "")
    return {
        "picks": picks,
        "notes": str(notes_text or ""),
        "sections": sections,
    }


def _merge_tomorrow_symbols(existing_symbols: list[object], incoming_symbols: list[object], limit: int = 20) -> tuple[list[str], int]:
    merged: list[str] = []
    seen: set[str] = set()

    for raw in list(existing_symbols or []):
        symbol = _normalize_tomorrow_symbol(raw)
        if symbol and symbol not in seen:
            merged.append(symbol)
            seen.add(symbol)
        if len(merged) >= limit:
            return merged[:limit], 0

    base_len = len(merged)
    for raw in list(incoming_symbols or []):
        symbol = _normalize_tomorrow_symbol(raw)
        if symbol and symbol not in seen:
            merged.append(symbol)
            seen.add(symbol)
        if len(merged) >= limit:
            break

    return merged[:limit], max(0, len(merged[:limit]) - base_len)


def _load_local_tomorrow_store() -> dict:
    default = {"picks": [], "notes": ""}
    try:
        if not _TOMORROW_STORE_PATH.exists():
            return default
        payload = json.loads(_TOMORROW_STORE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else default
    except Exception:
        return default


def _save_local_tomorrow_store(store: dict) -> bool:
    try:
        _TOMORROW_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOMORROW_STORE_PATH.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def _read_tomorrow_store_source() -> tuple[dict, str]:
    try:
        if _get_sheet() is not None:
            return _normalize_tomorrow_store(_load_picks()), "cloud"
    except Exception:
        pass
    return _normalize_tomorrow_store(_load_local_tomorrow_store()), "local"


def _cache_tomorrow_store_session(store: dict, storage_mode: str) -> tuple[dict, str]:
    normalized = _normalize_tomorrow_store(store)
    mode = "cloud" if str(storage_mode or "").strip().lower() == "cloud" else "local"
    st.session_state["tomorrow_picks_store"] = normalized
    st.session_state["tomorrow_picks_storage_mode"] = mode
    st.session_state["tomorrow_picks_store_loaded"] = True
    return normalized, mode


def _get_cached_tomorrow_store() -> tuple[dict | None, str]:
    if not bool(st.session_state.get("tomorrow_picks_store_loaded", False)):
        return None, ""
    cached = st.session_state.get("tomorrow_picks_store")
    if not isinstance(cached, dict):
        return None, ""
    mode = str(st.session_state.get("tomorrow_picks_storage_mode", "local") or "local")
    return _normalize_tomorrow_store(cached), ("cloud" if mode == "cloud" else "local")


def _load_tomorrow_store(force_refresh: bool = False) -> tuple[dict, str]:
    cached_store, cached_mode = _get_cached_tomorrow_store()
    if cached_store is not None and not force_refresh:
        return cached_store, cached_mode or "local"

    default = _normalize_tomorrow_store(None)
    session_store = cached_store or _normalize_tomorrow_store(
        st.session_state.get("tomorrow_picks_store", default)
    )
    sheets_ready = _get_sheet() is not None
    if sheets_ready:
        store = _normalize_tomorrow_store(_load_picks())
        storage_mode = "cloud"
        if (not store["picks"] and not store["notes"]) and (session_store["picks"] or session_store["notes"]):
            store = session_store
            _save_picks(store)
    else:
        local_store = _normalize_tomorrow_store(_load_local_tomorrow_store())
        if local_store["picks"] or local_store["notes"]:
            store = local_store
            storage_mode = "local"
        else:
            store = session_store
            storage_mode = "local"
            if store["picks"] or store["notes"]:
                _save_local_tomorrow_store(store)
    return _cache_tomorrow_store_session(store, storage_mode)


def _persist_tomorrow_store(store: dict) -> None:
    normalized = _normalize_tomorrow_store(store)
    storage_mode = "cloud" if _get_sheet() is not None else "local"
    _cache_tomorrow_store_session(normalized, storage_mode)
    if storage_mode == "cloud":
        _save_picks(normalized)
    else:
        _save_local_tomorrow_store(normalized)


def _assign_symbols_to_tomorrow_bucket(
    sections: dict[str, list[object]] | None,
    incoming_symbols: list[object] | tuple[object, ...] | None,
    *,
    bucket: str,
    limit: int = 20,
) -> tuple[dict[str, list[str]], int, int, int]:
    target_bucket = _normalize_tomorrow_bucket(bucket)
    current_sections = _apply_tomorrow_sections_limit(sections, limit=limit)
    membership_before = _tomorrow_section_membership(current_sections)
    normalized_incoming = _normalize_tomorrow_symbols(incoming_symbols, limit=limit)
    if not normalized_incoming:
        return current_sections, 0, 0, 0

    working_sections = {name: list(values) for name, values in current_sections.items()}
    for name in _TOMORROW_SECTION_ORDER:
        working_sections[name] = [
            symbol
            for symbol in working_sections.get(name, [])
            if symbol not in normalized_incoming
        ]

    existing_total = len(_tomorrow_flatten_sections(working_sections, limit=limit))
    target_values = list(working_sections.get(target_bucket, []))
    added = 0
    moved = 0
    overflow = 0

    for symbol in normalized_incoming:
        previous_bucket = membership_before.get(symbol)
        if previous_bucket is None:
            if existing_total >= limit:
                overflow += 1
                continue
            added += 1
            existing_total += 1
        elif previous_bucket != target_bucket:
            moved += 1
        if symbol not in target_values:
            target_values.append(symbol)

    working_sections[target_bucket] = target_values
    return _apply_tomorrow_sections_limit(working_sections, limit=limit), added, moved, overflow


def _save_symbols_to_tomorrow_store(
    symbols: list[object] | tuple[object, ...] | None,
    *,
    bucket: str = "relax",
) -> dict[str, object]:
    store, storage_mode = _load_tomorrow_store()
    sections_before = store.get("sections", _tomorrow_section_defaults())
    sections_after, added, moved, overflow = _assign_symbols_to_tomorrow_bucket(
        sections_before,
        symbols,
        bucket=bucket,
        limit=20,
    )
    target_bucket = _normalize_tomorrow_bucket(bucket)
    changed = sections_after != store.get("sections", _tomorrow_section_defaults())
    if changed:
        store["sections"] = sections_after
        store["picks"] = _tomorrow_flatten_sections(sections_after, limit=20)
        _persist_tomorrow_store(store)

    return {
        "added": int(added),
        "moved": int(moved),
        "changed": bool(changed),
        "total": len(_tomorrow_flatten_sections(sections_after, limit=20)),
        "section_total": len(sections_after.get(target_bucket, [])),
        "storage_mode": storage_mode,
        "limit_reached": bool(overflow > 0),
        "bucket": target_bucket,
        "bucket_label": _tomorrow_section_label(target_bucket),
    }


def _set_tomorrow_notes_feedback(kind: str, message: str) -> None:
    st.session_state["tmr_notes_feedback"] = {
        "kind": str(kind or "info"),
        "message": str(message or "").strip(),
        "at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


def _verify_tomorrow_notes_saved(expected_notes: str) -> tuple[bool, str]:
    try:
        source_store, storage_mode = _read_tomorrow_store_source()
        saved_notes = str(source_store.get("notes", "") or "")
        return saved_notes == str(expected_notes or ""), storage_mode
    except Exception:
        return False, ""


def _render_add_in_picks_actions(
    symbols: list[object] | tuple[object, ...] | None,
    *,
    key_prefix: str,
    scope_label: str,
    bucket: str = "relax",
    helper_text: str = "",
) -> None:
    normalized = _normalize_tomorrow_symbols(symbols)
    if helper_text:
        st.markdown(
            f'<div style="font-size:11px;color:#88a6c4;margin:10px 0 12px 0;">{helper_text}</div>',
            unsafe_allow_html=True,
        )

    add_col, open_col = st.columns([1.15, 1], gap="small")
    with add_col:
        add_clicked = st.button(
            "ADD IN PICKS",
            key=f"{key_prefix}_add_in_picks",
            use_container_width=True,
            disabled=not normalized,
        )
    with open_col:
        open_clicked = st.button(
            "OPEN PICKS",
            key=f"{key_prefix}_open_picks",
            use_container_width=True,
        )

    if add_clicked:
        if not normalized:
            st.info("No valid stock is available to add right now.")
        else:
            summary = _save_symbols_to_tomorrow_store(normalized, bucket=bucket)
            bucket_label = str(summary.get("bucket_label", _tomorrow_section_label(bucket)))
            if summary["added"] or summary["moved"]:
                target = "Google Sheets" if summary["storage_mode"] == "cloud" else "local storage"
                parts: list[str] = []
                if summary["added"]:
                    noun = "stock" if int(summary["added"]) == 1 else "stocks"
                    parts.append(f"added {summary['added']} new {noun}")
                if summary["moved"]:
                    noun = "stock" if int(summary["moved"]) == 1 else "stocks"
                    parts.append(f"moved {summary['moved']} existing {noun}")
                joined = " and ".join(parts)
                st.success(f"{joined.title()} from {scope_label} into the {bucket_label} strip in {target}.")
            elif summary["limit_reached"]:
                st.info("Tomorrow's Picks is already full at 20 stocks, or these symbols are already saved.")
            else:
                st.info(f"These symbols are already saved in the {bucket_label} strip.")

    if open_clicked:
        _activate_sidebar_panel("tomorrow_picks_show_panel")

try:
    import scan_diagnostics as _scan_diag
    _SCAN_DIAGNOSTICS_OK = True
except Exception:
    _SCAN_DIAGNOSTICS_OK = False

    class _ScanDiagnosticsStub:
        @staticmethod
        def reset() -> None:
            return None

        @staticmethod
        def record_attempt(ticker: str) -> None:
            return None

        @staticmethod
        def record_success(ticker: str) -> None:
            return None

        @staticmethod
        def record_failure(ticker: str, reason: str) -> None:
            return None

        @staticmethod
        def get_report() -> dict:
            return {
                "attempted": 0,
                "succeeded": 0,
                "failed_data": 0,
                "scan_filtered": 0,
                "reasons": {},
                "failed_tickers": [],
                "success_rate_pct": 0.0,
                "data_ok_pct": 0.0,
            }

    _scan_diag = _ScanDiagnosticsStub()

# ── TradingView symbol helper ─────────────────────────────────────────
def tv_symbol(ticker: str) -> str:
    """
    Convert yfinance NSE ticker to TradingView symbol.
    e.g. RELIANCE.NS → NSE:RELIANCE
    """
    return "NSE:" + ticker.replace(".NS", "")


def tv_chart_url(symbol_no_ns: str) -> str:
    """Return TradingView chart URL for a bare symbol (no .NS suffix)."""
    return f"https://www.tradingview.com/chart/?symbol=NSE:{symbol_no_ns}"


# ── Data downloader (optional, graceful if missing) ───────────────────
try:
    from data_downloader import (
        update_data_if_old,
        update_all_data,
        data_status_summary,
        bulk_download,
        load_csv,
    )
    _DATA_DOWNLOADER_OK = True
except ImportError:
    _DATA_DOWNLOADER_OK = False

    def update_data_if_old(tickers, **kw):  # type: ignore[misc]
        return 0

    def update_all_data(tickers, **kw):  # type: ignore[misc]
        return {"updated": 0, "skipped": 0, "failed": 0}

    def data_status_summary(tickers):  # type: ignore[misc]
        return {}

    def bulk_download(tickers, **kw):  # type: ignore[misc]
        return {}

    def load_csv(ticker):  # type: ignore[misc]
        return None

# ── Grading engine (optional, graceful if missing) ────────────────────
try:
    from grading_engine import apply_universal_grading
    _GRADING_OK = True
except ImportError:
    _GRADING_OK = False

    def apply_universal_grading(df, market_bias=None):  # type: ignore[misc]
        return df

# ── Enhanced logic engine (optional, graceful if missing) ─────────────
try:
    from enhanced_logic_engine import apply_enhanced_logic
    _ENHANCED_LOGIC_OK = True
except ImportError:
    _ENHANCED_LOGIC_OK = False

    def apply_enhanced_logic(df):  # type: ignore[misc]
        return df

# ── Phase 4 logic engine (optional, graceful if missing) ──────────────
try:
    from phase4_logic_engine import apply_phase4_logic, apply_phase42_logic
    _PHASE4_LOGIC_OK = True
except ImportError:
    _PHASE4_LOGIC_OK = False

    def apply_phase4_logic(df, market_bias=None):  # type: ignore[misc]
        return df

    def apply_phase42_logic(df):  # type: ignore[misc]
        return df

# ── Time Travel engine (optional, graceful if missing) ────────────────
try:
    import time_travel_engine as _tt
    _TIME_TRAVEL_OK = True
except ImportError:
    _TIME_TRAVEL_OK = False

    class _tt:  # type: ignore[no-redef]
        """Silent stub — all calls are no-ops when engine file is missing."""
        @staticmethod
        def is_active() -> bool:          return False
        @staticmethod
        def get_reference_datetime():     return datetime.now()
        @staticmethod
        def get_reference_date():         return None
        @staticmethod
        def format_banner() -> str:       return ""
        @staticmethod
        def activate(d) -> int:           return 0
        @staticmethod
        def restore() -> None:            pass


def _get_pending_time_travel_date():
    """Return the sidebar-selected TT date even before a scan activates the engine."""
    try:
        pending = st.session_state.get("tt_date_val") or st.session_state.get("tt_date_picker")
    except Exception:
        return None
    return pending if pending not in (None, "") else None


def _get_dashboard_reference_datetime() -> datetime:
    """Use the selected TT date for dashboard labels before scan-time activation."""
    pending = _get_pending_time_travel_date()
    if pending is not None:
        try:
            return datetime(pending.year, pending.month, pending.day, 16, 0, 0)
        except Exception:
            pass
    return _tt.get_reference_datetime()


def _get_dashboard_status_label() -> str:
    """Show TT-selected status instead of a misleading live-refresh caption."""
    pending = _get_pending_time_travel_date()
    if pending is None:
        return get_data_status_label()
    try:
        day_text = pending.isoformat()
    except Exception:
        day_text = str(pending)
    return f"🕰️ Time Travel Selected — {day_text}"

# ── Stock Aura — fully inlined (no external file dependency) ─────────

def _aura_ema(s: "pd.Series", n: int) -> "pd.Series":
    return s.ewm(span=n, adjust=False).mean()

def _aura_rsi_last(close: "pd.Series", period: int = 14) -> float:
    try:
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])
    except Exception:
        return 50.0

def _aura_fetch(symbol: str) -> "pd.DataFrame | None":
    """Fetch OHLCV; always truncates to TT cutoff if active — no leakage."""
    ticker_ns = symbol.upper().strip()
    if not ticker_ns.endswith(".NS"):
        ticker_ns += ".NS"

    # Use the engine's authoritative cutoff (works even in worker threads)
    cutoff = _tt.get_reference_date()  # None when TT is off

    def _cut(df):
        if df is None or df.empty or cutoff is None:
            return df
        try:
            mask = pd.to_datetime(df.index).date <= cutoff
            t    = df.loc[mask]
            return t if len(t) >= 10 else None
        except Exception:
            return df

    # 1️⃣ Try ALL_DATA cache — _cut enforces cutoff even if not pre-truncated.
    # Note: after the get_df_for_ticker Bug 7 fix, the cached frame is already
    # truncated when TT is active, so _cut() is a belt-and-suspenders guard.
    try:
        from feature_data_manager import feature_manager as _feature_manager

        df = _feature_manager.get_stock_data(
            ticker_ns,
            period="6mo",
            interval="1d",
            force_refresh=False,
        )
        if df is not None and len(df) >= 10:
            return _cut(df)
    except Exception:
        pass

    # 2️⃣ Live yfinance fallback — truncate BEFORE returning so no future data
    # leaks. Also store the truncated frame back into ALL_DATA so subsequent
    # calls (backtest, signal, etc.) see the correct historical data.
    try:
        df = _get_shared_market_frame(
            ticker_ns,
            period="6mo",
            min_rows=10,
            append_nse_suffix=False,
            allow_csv_cache=True,
            require_volume=True,
        )
        if df is None or len(df) < 10:
            return None
        truncated = _cut(df)
        # BUG FIX: Cache the truncated (not the full) frame so that any
        # subsequent get_df_for_ticker call for this ticker during TT also
        # gets the historically-correct data, not future-contaminated data.
        if truncated is not None and cutoff is not None:
            try:
                from strategy_engines._engine_utils import ALL_DATA, _ALL_DATA_LOCK
                with _ALL_DATA_LOCK:
                    ALL_DATA[ticker_ns] = truncated
            except Exception:
                pass
        return truncated
    except Exception:
        return None


def _aura_engine(df: "pd.DataFrame", symbol: str, market_bias: dict) -> dict:
    """
    Run all 8 Aura checks; return a result dict.
    Never raises — returns AVOID on any error.
    """
    r = dict(
        symbol=symbol.upper().replace(".NS", ""),
        price=0.0, rsi=50.0, ema20=0.0, ema50=0.0,
        vol_ratio=1.0, delta_ema20=0.0, delta_20h=0.0,
        ret_5d=0.0, ret_20d=0.0, rr_ratio=0.0,
        verdict="❌ AVOID", timing="NO TRADE", verdict_color="#ff4d6d",
        setup_type="None", trend_ok=False, volume_ok=False,
        momentum_ok=False, entry_ok=False, sl_quality="Poor",
        rr_ok=False, market_note="",
        pos=[], warn=[], rej=[],
    )
    try:
        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        high_s = df["High"].dropna()  if "High"  in df.columns else close
        low_s  = df["Low"].dropna()   if "Low"   in df.columns else close

        if len(close) < 30:
            r["rej"].append("Insufficient price history")
            return r

        lc       = float(close.iloc[-1])
        e20      = float(_aura_ema(close, 20).iloc[-1])
        e50      = float(_aura_ema(close, 50).iloc[-1])
        prev_e20 = float(_aura_ema(close, 20).iloc[-2]) if len(close) >= 2 else e20
        rsi_v    = _aura_rsi_last(close)

        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        lv      = float(volume.iloc[-1])
        vol_r   = round(lv / avg_vol, 2) if avg_vol > 0 else 1.0

        h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
        ret_5d  = (lc / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
        ret_20d = (lc / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        d_ema20 = (lc / e20  - 1) * 100 if e20  > 0 else 0.0
        d_20h   = (lc / h20  - 1) * 100 if h20  > 0 else 0.0

        # Risk-reward
        if d_20h >= -1.5:
            target = lc * 1.06          # breakout: project 6% continuation
        else:
            target = h20                # pullback: prior high
        downside = max(lc - e20, 0.01) if lc > e20 > 0 else 0.01
        rr       = max(target - lc, 0.0) / downside

        r.update(price=round(lc,2), rsi=round(rsi_v,1), ema20=round(e20,2),
                 ema50=round(e50,2), vol_ratio=round(vol_r,2),
                 delta_ema20=round(d_ema20,2), delta_20h=round(d_20h,2),
                 ret_5d=round(ret_5d,2), ret_20d=round(ret_20d,2),
                 rr_ratio=round(rr,2))

        # 1 — Trend
        if lc > e20 > e50:
            r["trend_ok"] = True
            r["pos"].append("Strong uptrend (Price > EMA20 > EMA50)")
            if e20 > prev_e20:
                r["pos"].append("EMA20 slope rising — momentum intact")
        elif lc > e20:
            r["warn"].append("Price above EMA20 but EMA20 < EMA50 — weak structure")
        else:
            r["rej"].append("Downtrend — price below EMA20")

        # 2 — Setup
        at_zone   = -1.5 <= d_20h <= 0.5
        pb_zone   = -6.0 <= d_20h <  -1.5
        if at_zone and vol_r >= 1.5:
            r["setup_type"] = "Breakout"
            r["pos"].append("Breakout setup — price at 20D high with volume")
        elif at_zone:
            r["setup_type"] = "Pullback"
            r["warn"].append("Near 20D high but volume not confirming — wait for vol")
        elif pb_zone and e20 > prev_e20:
            r["setup_type"] = "Pullback"
            r["pos"].append("Healthy pullback to EMA support — potential re-entry")
        elif d_20h < -6.0:
            r["setup_type"] = "None"
            r["rej"].append(f"Too far from 20D high ({d_20h:.1f}%) — no valid entry")
        else:
            r["setup_type"] = "Pullback"
            r["warn"].append("Setup not fully formed — borderline zone")

        # 3 — Volume
        if vol_r >= 1.5:
            r["volume_ok"] = True
            r["pos"].append(f"Volume strong ({vol_r:.1f}× avg) — institutional participation")
        elif vol_r >= 1.3:
            r["volume_ok"] = True
            r["pos"].append(f"Volume valid ({vol_r:.1f}× avg) — acceptable participation")
        elif vol_r >= 1.0:
            r["warn"].append(f"Volume weak ({vol_r:.1f}× avg) — no conviction")
        else:
            r["rej"].append(f"Volume below average ({vol_r:.1f}×) — distribution risk")

        # 4 — Momentum
        if rsi_v > 75:
            r["rej"].append(f"RSI overbought ({rsi_v:.0f}) — late-stage entry risk")
        elif ret_5d > 12.0:
            r["rej"].append(f"5D return {ret_5d:.1f}% — short-term exhaustion risk")
        elif 50 <= rsi_v <= 70 and 2 <= ret_5d <= 10:
            r["momentum_ok"] = True
            r["pos"].append(f"RSI healthy ({rsi_v:.0f}) with strong 5D return ({ret_5d:.1f}%)")
        elif 50 <= rsi_v <= 70:
            r["momentum_ok"] = True
            r["pos"].append(f"RSI healthy ({rsi_v:.0f}) — momentum not stretched")
        elif 70 < rsi_v <= 75:
            r["warn"].append(f"RSI elevated ({rsi_v:.0f}) — caution zone")
        else:
            r["momentum_ok"] = True
            r["warn"].append(f"RSI low ({rsi_v:.0f}) — early accumulation stage")

        # 5 — Entry quality
        if d_ema20 <= 3.0:
            r["entry_ok"] = True
            r["pos"].append(f"Close to EMA20 ({d_ema20:.1f}%) — tight entry")
        elif d_ema20 <= 6.0:
            r["entry_ok"] = True
            r["warn"].append(f"Moderately extended from EMA20 ({d_ema20:.1f}%)")
        else:
            r["rej"].append(f"Overextended from EMA20 ({d_ema20:.1f}%) — late entry")

        # 6 — Stop quality
        if d_ema20 <= 3.0:
            r["sl_quality"] = "Tight"
            r["pos"].append(f"Tight stop ({d_ema20:.1f}% to EMA20)")
        elif d_ema20 <= 6.0:
            r["sl_quality"] = "Medium"
            r["warn"].append(f"Medium stop distance ({d_ema20:.1f}% to EMA20)")
        else:
            r["sl_quality"] = "Poor"
            r["rej"].append(f"Wide stop ({d_ema20:.1f}% to EMA20) — poor structure")

        # 7 — Risk-reward
        if rr >= 2.0:
            r["rr_ok"] = True
            r["pos"].append(f"Risk-reward {rr:.1f}:1 — excellent setup")
        elif rr >= 1.5:
            r["rr_ok"] = True
            r["pos"].append(f"Risk-reward {rr:.1f}:1 — acceptable")
        elif rr >= 1.0:
            r["warn"].append(f"Risk-reward {rr:.1f}:1 — marginal, prefer ≥2:1")
        else:
            r["rej"].append(f"Risk-reward {rr:.1f}:1 — unfavorable")

        # 8 — Market context
        mb  = market_bias if isinstance(market_bias, dict) else {}
        lbl = str(mb.get("bias", mb.get("regime", ""))).strip()
        if lbl:
            if any(w in lbl.lower() for w in ("bearish","weak","caution")):
                r["market_note"] = f"Market: {lbl} ⚠️"
                r["warn"].append(f"Market is {lbl} — reduce position size")
            elif any(w in lbl.lower() for w in ("bullish","trending up","strong")):
                r["market_note"] = f"Market: {lbl} ✅"
                r["pos"].append(f"Favorable market backdrop ({lbl})")
            else:
                r["market_note"] = f"Market: {lbl}"
        else:
            r["market_note"] = "Market context unavailable — run Market Bias first"

        # Verdict
        rej_n  = len(r["rej"])
        warn_n = len(r["warn"])
        rb = r

        is_buy_now = (
            rb["setup_type"] == "Breakout" and rb["trend_ok"] and
            rb["volume_ok"] and rb["momentum_ok"] and
            rb["entry_ok"] and rb["rr_ok"] and rej_n == 0
        )
        is_buy_conf = (
            rb["setup_type"] in ("Breakout","Pullback") and
            rb["trend_ok"] and rb["momentum_ok"] and
            (rb["rr_ok"] or rr >= 1.0) and rej_n == 0
        )
        is_watch = (
            rb["trend_ok"] and rb["setup_type"] != "None" and
            rej_n <= 1 and rsi_v <= 75
        )

        if is_buy_now:
            r["verdict"]       = "🔥 BUY RIGHT NOW"
            r["timing"]        = "BUY NOW"
            r["verdict_color"] = "#00d4a8"
        elif is_buy_conf:
            r["verdict"]       = "✅ BUY (ON CONFIRMATION)"
            r["timing"]        = "BUY TOMORROW"
            r["verdict_color"] = "#0094ff"
        elif is_watch:
            r["verdict"]       = "👀 WATCH"
            r["timing"]        = "WAIT"
            r["verdict_color"] = "#f0b429"
        else:
            r["verdict"]       = "❌ AVOID"
            r["timing"]        = "NO TRADE"
            r["verdict_color"] = "#ff4d6d"

    except Exception as exc:
        r["rej"].append(f"Engine error: {exc}")
    return r


def _aura_timing_badge(timing: str, vc: str) -> str:
    return (
        f'<span style="background:{vc}20;border:1px solid {vc};border-radius:6px;'
        f'padding:3px 10px;font-size:12px;font-weight:700;color:{vc};">{timing}</span>'
    )

def _aura_factor_row(label: str, value: str, color: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480;">{label}</span>'
        f'<span style="font-size:12px;font-weight:700;color:{color};">{value}</span></div>'
    )

def render_stock_aura_panel() -> None:
    """Fully self-contained Stock Aura panel — no external file needed."""
    if not st.session_state.get("aura_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # Header
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:22px;'
        'color:#ccd9e8;margin-bottom:4px;">🔮 Stock Aura</h2>'
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        "Single stock decision engine — a trader's brain, not a screener</div>",
        unsafe_allow_html=True,
    )

    # TT banner inside Aura
    _aura_tt = st.session_state.get("aura_tt_date")
    if _aura_tt is not None:
        try:
            _aura_tt_str = _aura_tt.strftime("%d %b %Y")
        except Exception:
            _aura_tt_str = str(_aura_tt)
        st.markdown(
            f'<div style="background:#1a0a00;border:1.5px solid #f0b429;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:14px;font-size:12px;color:#f0b429;">'
            f'🕰️ <b>TIME TRAVEL ACTIVE</b> — Evaluating {_aura_tt_str} post-market · '
            f'No future data used</div>',
            unsafe_allow_html=True,
        )

    # Input row
    configure_nse_stock_search(fetch_nse_tickers())
    c_form, c_cls = st.columns([4, 1])
    with c_form:
        with st.form("stock_aura_form", clear_on_submit=False):
            c_in, c_btn = st.columns([3, 1])
            with c_in:
                ticker_raw = render_nse_stock_input(
                    "Stock Symbol",
                    key="aura_ticker_input",
                    placeholder="e.g. RELIANCE or search company name",
                    label_visibility="collapsed",
                )
            with c_btn:
                analyze_clicked = st.form_submit_button(
                    "🧠 Analyze Aura",
                    use_container_width=True,
                )
    with c_cls:
        if st.button("✕ Close", key="aura_close_btn"):
            st.session_state["aura_show_panel"] = False
            st.rerun()

    if not (analyze_clicked and ticker_raw.strip()):
        return

    sym = ticker_raw.strip().upper().replace(".NS", "")
    _spin = (
        f"🕰️ Historical aura for {sym} ({_aura_tt_str})…"
        if _aura_tt else f"🔮 Reading aura for {sym}…"
    )
    with st.spinner(_spin):
        df = _aura_fetch(sym)

    if df is None or df.empty:
        st.error(
            f"❌ No data for **{sym}**. Check the symbol "
            "(e.g. RELIANCE, not RELIANCE.NS) and try again."
        )
        return

    mb = dict(st.session_state.get("market_bias_result") or {})
    if _aura_tt and not mb.get("bias"):
        mb["bias"] = f"Historical ({_aura_tt_str}) — run Market Bias for that date"

    res = _aura_engine(df, sym, mb)
    vc  = res["verdict_color"]

    # ── Verdict card ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#0b1017;border:2px solid {vc};border-radius:14px;'
        f'padding:20px 24px;margin:12px 0 20px;">'
        f'<div style="font-size:13px;color:#4a6480;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px;">🔮 STOCK AURA RESULT</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;'
        f'color:#ccd9e8;margin-bottom:2px;">{res["symbol"]}</div>'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:14px;">'
        f'₹{res["price"]:.2f} &nbsp;|&nbsp; RSI {res["rsi"]:.0f} &nbsp;|&nbsp; '
        f'Vol {res["vol_ratio"]:.1f}× &nbsp;|&nbsp; EMA20 {res["delta_ema20"]:+.1f}% '
        f'&nbsp;|&nbsp; 5D {res["ret_5d"]:+.1f}%</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:22px;font-weight:900;'
        f'color:{vc};margin-bottom:10px;">{res["verdict"]}</div>'
        f'Timing: {_aura_timing_badge(res["timing"], vc)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([3, 2])

    # ── Why / Warnings ────────────────────────────────────────────────
    with col_l:
        if res["pos"]:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#00d4a8;'
                'letter-spacing:.5px;margin-bottom:8px;">WHY ✔</div>' +
                "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:#00d4a8;font-weight:700;">✔</span> &nbsp;{t}</div>'
                    for t in res["pos"]
                ) + '</div>',
                unsafe_allow_html=True,
            )

        issues = [(t, "#f0b429") for t in res["warn"]] + [(t, "#ff4d6d") for t in res["rej"]]
        if issues:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #3a1e1e;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#ff4d6d;'
                'letter-spacing:.5px;margin-bottom:8px;">WARNINGS / REJECTIONS ✖</div>' +
                "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:{c};font-weight:700;">✖</span> &nbsp;{t}</div>'
                    for t, c in issues
                ) + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;font-size:12px;color:#00d4a8;">'
                'Warnings: None ✔</div>',
                unsafe_allow_html=True,
            )

    # ── Factor scorecard ──────────────────────────────────────────────
    with col_r:
        def _g(ok, lt="PASS", lf="FAIL"):
            return (lt, "#00d4a8") if ok else (lf, "#ff4d6d")

        tl, tc = _g(res["trend_ok"],     "ALIGNED",   "WEAK")
        sc2    = "#00d4a8" if res["setup_type"] != "None" else "#ff4d6d"
        vl, vc2 = _g(res["volume_ok"],   "STRONG",    "WEAK")
        ml, mc2 = _g(res["momentum_ok"], "HEALTHY",   "STRETCHED")
        el, ec  = _g(res["entry_ok"],    "GOOD",      "EXTENDED")
        slc     = {"Tight":"#00d4a8","Medium":"#f0b429","Poor":"#ff4d6d"}.get(res["sl_quality"],"#4a6480")
        rrl, rrc = _g(res["rr_ok"],
                      f'{res["rr_ratio"]:.1f}:1 ✔',
                      f'{res["rr_ratio"]:.1f}:1 ✖')

        factors = (
            _aura_factor_row("Trend",        tl,   tc)  +
            _aura_factor_row("Setup",        res["setup_type"], sc2) +
            _aura_factor_row("Volume",       f'{res["vol_ratio"]:.1f}× — {vl}', vc2) +
            _aura_factor_row("Momentum RSI", f'{res["rsi"]:.0f} — {ml}', mc2) +
            _aura_factor_row("Entry Quality",f'{res["delta_ema20"]:+.1f}% — {el}', ec) +
            _aura_factor_row("Stop Quality", res["sl_quality"], slc) +
            _aura_factor_row("Risk-Reward",  rrl,  rrc)
        )
        st.markdown(
            '<div style="background:#0f1923;border:1px solid #1e3a5f;'
            'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
            '<div style="font-size:11px;font-weight:700;color:#8ab4d8;'
            'letter-spacing:.5px;margin-bottom:8px;">FACTOR SCORECARD</div>'
            + factors + '</div>',
            unsafe_allow_html=True,
        )
        if res["market_note"]:
            nc = "#f0b429" if "caution" in res["market_note"].lower() else "#4a6480"
            st.markdown(
                f'<div style="font-size:11px;color:{nc};padding:4px 0;">'
                f'🌐 {res["market_note"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="font-size:10px;color:#2a3f58;margin-top:12px;text-align:center;">'
        '⚠️ Stock Aura is for educational purposes only. Not financial advice.</div>',
        unsafe_allow_html=True,
    )

def render_stock_aura_panel() -> None:
    """Render the Stock Aura panel with persistent result actions."""
    if not st.session_state.get("aura_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:22px;'
        'color:#ccd9e8;margin-bottom:4px;">Stock Aura</h2>'
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        "Single-stock decision engine with a direct add-to-picks action.</div>",
        unsafe_allow_html=True,
    )

    aura_tt = st.session_state.get("aura_tt_date")
    aura_tt_str = ""
    if aura_tt is not None:
        try:
            aura_tt_str = aura_tt.strftime("%d %b %Y")
        except Exception:
            aura_tt_str = str(aura_tt)
        st.markdown(
            f'<div style="background:#1a0a00;border:1.5px solid #f0b429;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:14px;font-size:12px;color:#f0b429;">'
            f'<b>TIME TRAVEL ACTIVE</b> - Evaluating {aura_tt_str} post-market. '
            f'No future data used.</div>',
            unsafe_allow_html=True,
        )

    configure_nse_stock_search(fetch_nse_tickers())
    form_col, close_col = st.columns([4, 1])
    with form_col:
        with st.form("stock_aura_form", clear_on_submit=False):
            input_col, button_col = st.columns([3, 1])
            with input_col:
                ticker_raw = render_nse_stock_input(
                    "Stock Symbol",
                    key="aura_ticker_input",
                    placeholder="e.g. RELIANCE or search company name",
                    label_visibility="collapsed",
                )
            with button_col:
                analyze_clicked = st.form_submit_button(
                    "Analyze Aura",
                    use_container_width=True,
                )
    with close_col:
        if st.button("Close", key="aura_close_btn"):
            st.session_state["aura_show_panel"] = False
            st.rerun()

    ticker_text = str(ticker_raw or "").strip()
    aura_payload = st.session_state.get("aura_last_payload")
    aura_error = str(st.session_state.get("aura_last_error", "") or "")

    if analyze_clicked and ticker_text:
        symbol = ticker_text.upper().replace(".NS", "")
        spinner_text = (
            f"Historical aura for {symbol} ({aura_tt_str})..."
            if aura_tt
            else f"Reading aura for {symbol}..."
        )
        with st.spinner(spinner_text):
            aura_df = _aura_fetch(symbol)

        if aura_df is None or aura_df.empty:
            aura_payload = None
            aura_error = (
                f"No data for {symbol}. Check the symbol "
                "(for example RELIANCE, not RELIANCE.NS) and try again."
            )
        else:
            market_bias = dict(st.session_state.get("market_bias_result") or {})
            if aura_tt and not market_bias.get("bias"):
                market_bias["bias"] = f"Historical ({aura_tt_str}) - run Market Bias for that date"

            aura_result = _aura_engine(aura_df, symbol, market_bias)
            aura_payload = {"symbol": symbol, "result": aura_result}
            aura_error = ""

        st.session_state["aura_last_payload"] = aura_payload
        st.session_state["aura_last_error"] = aura_error

    if aura_error:
        st.error(aura_error)

    if not aura_payload:
        return

    res = dict(aura_payload.get("result") or {})
    symbol = _normalize_tomorrow_symbol(aura_payload.get("symbol", res.get("symbol", "")))
    if not symbol or not res:
        return

    verdict_color = str(res.get("verdict_color", "#00d4a8") or "#00d4a8")
    verdict_text = str(res.get("verdict", "NO VERDICT") or "NO VERDICT")
    timing_text = str(res.get("timing", "WAIT") or "WAIT")
    price = float(res.get("price", 0) or 0)
    rsi = float(res.get("rsi", 0) or 0)
    vol_ratio = float(res.get("vol_ratio", 0) or 0)
    delta_ema20 = float(res.get("delta_ema20", 0) or 0)
    ret_5d = float(res.get("ret_5d", 0) or 0)

    st.markdown(
        f'<div style="background:#0b1017;border:2px solid {verdict_color};border-radius:14px;'
        f'padding:20px 24px;margin:12px 0 20px;">'
        f'<div style="font-size:13px;color:#4a6480;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px;">STOCK AURA RESULT</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;'
        f'color:#ccd9e8;margin-bottom:2px;">{symbol}</div>'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:14px;">'
        f'Rs {price:.2f} | RSI {rsi:.0f} | Vol {vol_ratio:.1f}x | EMA20 {delta_ema20:+.1f}% '
        f'| 5D {ret_5d:+.1f}%</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:22px;font-weight:900;'
        f'color:{verdict_color};margin-bottom:10px;">{verdict_text}</div>'
        f'Timing: {_aura_timing_badge(timing_text, verdict_color)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 2])
    with left_col:
        positives = list(res.get("pos") or [])
        if positives:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#00d4a8;'
                'letter-spacing:.5px;margin-bottom:8px;">WHY YES</div>'
                + "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:#00d4a8;font-weight:700;">+</span> {text}</div>'
                    for text in positives
                )
                + '</div>',
                unsafe_allow_html=True,
            )

        issues = [(text, "#f0b429") for text in list(res.get("warn") or [])]
        issues += [(text, "#ff4d6d") for text in list(res.get("rej") or [])]
        if issues:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #3a1e1e;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#ff4d6d;'
                'letter-spacing:.5px;margin-bottom:8px;">WARNINGS / REJECTIONS</div>'
                + "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:{color};font-weight:700;">-</span> {text}</div>'
                    for text, color in issues
                )
                + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;font-size:12px;color:#00d4a8;">'
                'Warnings: None</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        def _aura_gate(ok: bool, pass_text: str = "PASS", fail_text: str = "FAIL") -> tuple[str, str]:
            return (pass_text, "#00d4a8") if ok else (fail_text, "#ff4d6d")

        trend_label, trend_color = _aura_gate(bool(res.get("trend_ok")), "ALIGNED", "WEAK")
        setup_type = str(res.get("setup_type", "None") or "None")
        setup_color = "#00d4a8" if setup_type != "None" else "#ff4d6d"
        volume_label, volume_color = _aura_gate(bool(res.get("volume_ok")), "STRONG", "WEAK")
        momentum_label, momentum_color = _aura_gate(bool(res.get("momentum_ok")), "HEALTHY", "STRETCHED")
        entry_label, entry_color = _aura_gate(bool(res.get("entry_ok")), "GOOD", "EXTENDED")
        sl_quality = str(res.get("sl_quality", "Unknown") or "Unknown")
        sl_color = {"Tight": "#00d4a8", "Medium": "#f0b429", "Poor": "#ff4d6d"}.get(sl_quality, "#4a6480")
        rr_ratio = float(res.get("rr_ratio", 0) or 0)
        rr_ok = bool(res.get("rr_ok"))
        rr_label, rr_color = _aura_gate(rr_ok, f"{rr_ratio:.1f}:1", f"{rr_ratio:.1f}:1")

        factors = (
            _aura_factor_row("Trend", trend_label, trend_color)
            + _aura_factor_row("Setup", setup_type, setup_color)
            + _aura_factor_row("Volume", f"{vol_ratio:.1f}x - {volume_label}", volume_color)
            + _aura_factor_row("Momentum RSI", f"{rsi:.0f} - {momentum_label}", momentum_color)
            + _aura_factor_row("Entry Quality", f"{delta_ema20:+.1f}% - {entry_label}", entry_color)
            + _aura_factor_row("Stop Quality", sl_quality, sl_color)
            + _aura_factor_row("Risk-Reward", rr_label, rr_color)
        )
        st.markdown(
            '<div style="background:#0f1923;border:1px solid #1e3a5f;'
            'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
            '<div style="font-size:11px;font-weight:700;color:#8ab4d8;'
            'letter-spacing:.5px;margin-bottom:8px;">FACTOR SCORECARD</div>'
            + factors
            + '</div>',
            unsafe_allow_html=True,
        )

        market_note = str(res.get("market_note", "") or "")
        if market_note:
            note_color = "#f0b429" if "caution" in market_note.lower() else "#4a6480"
            st.markdown(
                f'<div style="font-size:11px;color:{note_color};padding:4px 0;">'
                f'Market note: {market_note}</div>',
                unsafe_allow_html=True,
            )

    _render_add_in_picks_actions(
        [symbol],
        key_prefix=f"aura_result_{symbol or 'stock'}",
        scope_label="Stock Aura",
        bucket=_tomorrow_bucket_for_mode(st.session_state.get("mode", 3)),
        helper_text="Add this Stock Aura result into Tomorrow's Picks and keep it saved until you delete it.",
    )

    st.markdown(
        '<div style="font-size:10px;color:#2a3f58;margin-top:12px;text-align:center;">'
        'Stock Aura is for educational purposes only. Not financial advice.</div>',
        unsafe_allow_html=True,
    )


def render_tomorrow_picks_panel() -> None:
    if not st.session_state.get("tomorrow_picks_show_panel", False):
        return

    store, storage_mode = _load_tomorrow_store()
    saved_picks = list(store.get("picks", []))
    saved_notes = str(store.get("notes", "") or "")
    saved_count = len(saved_picks)
    slots_left = max(0, 20 - saved_count)
    notes_words = len(saved_notes.split()) if saved_notes.strip() else 0
    notes_feedback = st.session_state.get("tmr_notes_feedback", {}) or {}
    notes_feedback_kind = str(notes_feedback.get("kind", "") or "").strip().lower()
    notes_feedback_msg = str(notes_feedback.get("message", "") or "").strip()
    notes_feedback_at = str(notes_feedback.get("at", "") or "").strip()
    if notes_feedback_kind not in {"success", "error", "info"}:
        notes_feedback_kind = "info"

    if st.session_state.get("tmr_notes_area") != store["notes"]:
        st.session_state["tmr_notes_area"] = store["notes"]

    sync_caption = "Saved permanently across all devices until you delete them" if storage_mode == "cloud" else "Saved permanently on this machine until you delete them"
    lead_copy = (
        "Add stocks once and keep them saved until you remove them manually."
        if storage_mode == "cloud"
        else "Add stocks once and keep them saved locally until you remove them manually."
    )
    status_tone = "cloud" if storage_mode == "cloud" else "local"
    status_copy = (
        "Cloud sync active via Google Sheets. Picks stay saved until deleted."
        if storage_mode == "cloud"
        else "Local persistence active on this machine. Picks stay saved until deleted."
    )
    notes_status_value = "Verified Saved" if saved_notes.strip() else "Empty"
    notes_status_caption = f"{notes_words} word{'s' if notes_words != 1 else ''}" if saved_notes.strip() else "No saved note yet"
    storage_label = "Cloud Sync" if storage_mode == "cloud" else "Local Save"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(.tmr-panel-anchor) {
          background:
            radial-gradient(circle at top right, rgba(240,180,41,0.12), transparent 32%),
            radial-gradient(circle at top left, rgba(0,148,255,0.10), transparent 28%),
            linear-gradient(180deg, rgba(9,13,20,0.96), rgba(5,8,13,0.99));
          border:1px solid rgba(88,120,154,0.34);
          border-radius:18px;
          padding:24px 24px 20px 24px;
          box-shadow:
            0 22px 48px rgba(0,0,0,0.42),
            inset 0 1px 0 rgba(255,255,255,0.04),
            inset 0 0 0 1px rgba(0,148,255,0.05);
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-left-panel-anchor),
        div[data-testid="stVerticalBlock"]:has(.tmr-notes-panel-anchor) {
          background:
            linear-gradient(180deg, rgba(14,20,30,0.94), rgba(9,14,22,0.98));
          border:1px solid rgba(61,86,114,0.42);
          border-radius:16px;
          padding:18px 18px 20px 18px;
          height:100%;
          box-shadow:
            inset 0 0 0 1px rgba(36,53,80,0.24),
            0 16px 34px rgba(0,0,0,0.20);
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-left-panel-anchor) div[data-testid="stTextInput"] input,
        div[data-testid="stVerticalBlock"]:has(.tmr-notes-panel-anchor) div[data-testid="stTextArea"] textarea {
          background:rgba(9,17,27,0.94) !important;
          color:#e0e8f0 !important;
          border:1px solid rgba(56,86,116,0.52) !important;
          box-shadow:none !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-left-panel-anchor) div[data-testid="stTextInput"] label p,
        div[data-testid="stVerticalBlock"]:has(.tmr-notes-panel-anchor) div[data-testid="stTextArea"] label p {
          color:#dbe8f6 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-notes-panel-anchor) div[data-testid="stTextArea"] textarea {
          min-height:400px !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-panel-anchor) div[data-testid="stAlert"] {
          background:linear-gradient(90deg, rgba(79,67,18,0.44), rgba(126,109,31,0.30)) !important;
          border:1px solid rgba(240,180,41,0.42) !important;
          border-left:3px solid #f0b429 !important;
          border-radius:12px !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-panel-anchor) div[data-testid="stAlert"] * {
          color:#fff2bf !important;
        }
        .tmr-status-pill {
          display:flex;
          align-items:center;
          gap:10px;
          border-radius:12px;
          padding:12px 16px;
          margin:0 0 20px 0;
          font-size:13px;
          line-height:1.45;
          border:1px solid rgba(96,132,167,0.30);
          color:#d8e6f4;
          background:linear-gradient(90deg, rgba(13,23,34,0.94), rgba(11,18,28,0.98));
          box-shadow:inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .tmr-status-pill strong {
          color:#f4f8ff;
        }
        .tmr-status-pill-local {
          border-color:rgba(240,180,41,0.34);
          background:linear-gradient(90deg, rgba(46,38,12,0.62), rgba(20,24,18,0.94));
        }
        .tmr-status-pill-local .tmr-status-icon {
          color:#f0b429;
        }
        .tmr-status-pill-cloud {
          border-color:rgba(0,212,168,0.34);
          background:linear-gradient(90deg, rgba(7,40,34,0.68), rgba(10,21,30,0.96));
        }
        .tmr-status-pill-cloud .tmr-status-icon {
          color:#00d4a8;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-remove-scope) .stButton > button {
          border-color:#ff4d6d !important;
          color:#ff4d6d !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-remove-scope) .stButton > button:hover {
          background:#ff4d6d !important;
          color:#060a0f !important;
          box-shadow:0 0 18px rgba(255,77,109,0.35) !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-add-scope) .stButton > button {
          white-space:nowrap !important;
          letter-spacing:0.2px !important;
          padding:12px 16px !important;
          min-height:54px !important;
        }
        .tmr-main-title {
          font-family:'Syne',sans-serif;
          font-size:48px;
          font-weight:800;
          letter-spacing:-1px;
          color:#f0b429;
          margin:0 0 8px 0;
        }
        .tmr-panel-lead {
          font-size:14px;
          color:#8eb0cf;
          margin-bottom:18px;
        }
        .tmr-metric-grid {
          display:grid;
          grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
          gap:14px;
          margin:0 0 22px 0;
        }
        .tmr-metric-card {
          border:1px solid rgba(69,102,136,0.34);
          border-radius:15px;
          padding:16px 16px 14px 16px;
          background:
            linear-gradient(180deg, rgba(12,18,28,0.96), rgba(8,13,22,0.98));
          box-shadow:
            inset 0 0 0 1px rgba(255,255,255,0.03),
            0 14px 28px rgba(0,0,0,0.18);
        }
        .tmr-metric-label {
          font-size:11px;
          letter-spacing:1px;
          text-transform:uppercase;
          color:#7d9abb;
          margin-bottom:8px;
        }
        .tmr-metric-value {
          font-family:'Syne',sans-serif;
          font-size:28px;
          font-weight:800;
          line-height:1;
          color:#f4f8ff;
          margin-bottom:6px;
        }
        .tmr-metric-caption {
          font-size:12px;
          color:#97b1cc;
          line-height:1.45;
        }
        .tmr-metric-value-accent {
          color:#00d4a8;
        }
        .tmr-metric-value-warn {
          color:#f0b429;
        }
        .tmr-picks-toolbar {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:12px;
          flex-wrap:wrap;
          margin:0 0 16px 0;
          padding:12px 14px;
          border:1px solid rgba(50,81,112,0.42);
          border-radius:14px;
          background:linear-gradient(180deg, rgba(11,18,27,0.92), rgba(8,13,20,0.98));
        }
        .tmr-picks-toolbar-main {
          font-size:13px;
          color:#dce7f4;
          font-weight:700;
        }
        .tmr-picks-toolbar-sub {
          font-size:11px;
          color:#88a6c4;
          margin-top:3px;
        }
        .tmr-slot-pill {
          display:inline-flex;
          align-items:center;
          gap:8px;
          padding:8px 12px;
          border-radius:999px;
          border:1px solid rgba(240,180,41,0.35);
          color:#ffe39a;
          background:rgba(64,49,15,0.48);
          font-size:11px;
          font-weight:700;
          letter-spacing:0.5px;
          text-transform:uppercase;
        }
        .tmr-pick-row {
          display:flex;
          align-items:center;
          gap:14px;
          background:linear-gradient(180deg, rgba(10,21,32,0.98), rgba(8,16,25,0.98));
          border:1px solid rgba(50,79,108,0.34);
          border-left:3px solid #00d4a8;
          border-radius:14px;
          padding:14px 16px;
          margin-bottom:10px;
          box-shadow:inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .tmr-pick-badge {
          width:34px;
          height:34px;
          border-radius:50%;
          display:flex;
          align-items:center;
          justify-content:center;
          font-size:12px;
          font-weight:800;
          color:#06111a;
          background:linear-gradient(180deg, #00d4a8, #00a88d);
          flex:0 0 34px;
        }
        .tmr-pick-symbol {
          color:#00d4a8;
          font-weight:800;
          font-size:18px;
          letter-spacing:0.3px;
        }
        .tmr-pick-meta {
          font-size:11px;
          color:#8da9c5;
          margin-top:4px;
        }
        .tmr-section-title {
          font-family:'Syne',sans-serif;
          font-size:34px;
          font-weight:800;
          line-height:1.12;
          letter-spacing:-0.7px;
          color:#f4f8ff;
          margin-bottom:8px;
          text-shadow:0 8px 24px rgba(0,0,0,0.18);
        }
        .tmr-section-caption {
          font-size:13px;
          color:#9ab4d0;
          margin:0 0 22px 0;
        }
        .tmr-notes-title {
          font-family:'Syne',sans-serif;
          font-size:28px;
          font-weight:800;
          color:#f4f8ff;
          letter-spacing:-0.5px;
          margin-bottom:8px;
        }
        .tmr-notes-caption {
          font-size:12px;
          color:#8eaed0;
          margin:0 0 14px 0;
        }
        .tmr-notes-status {
          border-radius:13px;
          padding:12px 14px;
          margin:0 0 14px 0;
          border:1px solid rgba(59,90,120,0.38);
          font-size:12px;
          line-height:1.5;
          background:linear-gradient(180deg, rgba(10,17,26,0.96), rgba(8,13,20,0.98));
          color:#dce7f4;
        }
        .tmr-notes-status strong {
          color:#f4f8ff;
        }
        .tmr-notes-status-success {
          border-color:rgba(0,212,168,0.34);
          background:linear-gradient(180deg, rgba(8,36,30,0.86), rgba(7,16,23,0.98));
        }
        .tmr-notes-status-error {
          border-color:rgba(255,77,109,0.34);
          background:linear-gradient(180deg, rgba(48,18,25,0.88), rgba(11,14,22,0.98));
        }
        .tmr-notes-status-info {
          border-color:rgba(240,180,41,0.30);
          background:linear-gradient(180deg, rgba(50,39,14,0.82), rgba(10,15,24,0.98));
        }
        .tmr-notes-helper {
          font-size:11px;
          color:#7e9aba;
          margin:10px 0 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="tmr-panel-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tmr-main-title">📈 Tomorrow\'s Picks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tmr-panel-lead">'
            f'{lead_copy}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tmr-status-pill tmr-status-pill-'
            f'{status_tone}">'
            '<span class="tmr-status-icon">'
            f'{"☁" if storage_mode == "cloud" else "💾"}</span>'
            f'<span><strong>{status_copy}</strong> '
            f'{"Changes sync across sessions and devices." if storage_mode == "cloud" else "You can still enable Google Sheets later for cross-device sync."}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                '<div class="tmr-metric-grid">'
                f'<div class="tmr-metric-card"><div class="tmr-metric-label">Saved Picks</div>'
                f'<div class="tmr-metric-value tmr-metric-value-accent">{saved_count}</div>'
                f'<div class="tmr-metric-caption">Stocks pinned in your permanent list</div></div>'
                f'<div class="tmr-metric-card"><div class="tmr-metric-label">Slots Left</div>'
                f'<div class="tmr-metric-value tmr-metric-value-warn">{slots_left}</div>'
                f'<div class="tmr-metric-caption">Up to 20 total saved picks</div></div>'
                f'<div class="tmr-metric-card"><div class="tmr-metric-label">Notes</div>'
                f'<div class="tmr-metric-value">{notes_status_value}</div>'
                f'<div class="tmr-metric-caption">{notes_status_caption}</div></div>'
                f'<div class="tmr-metric-card"><div class="tmr-metric-label">Storage</div>'
                f'<div class="tmr-metric-value">{storage_label}</div>'
                f'<div class="tmr-metric-caption">{sync_caption}</div></div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            with st.container():
                st.markdown('<div class="tmr-left-panel-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="tmr-section-title">📈 Tomorrow\'s Trending Stocks</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="tmr-section-caption">📡 {sync_caption}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        '<div class="tmr-picks-toolbar">'
                        '<div>'
                        '<div class="tmr-picks-toolbar-main">Permanent picks list</div>'
                        '<div class="tmr-picks-toolbar-sub">Anything saved here stays until you remove it yourself.</div>'
                        '</div>'
                        f'<div class="tmr-slot-pill">{saved_count}/20 saved</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                with st.form("tmr_add_form", clear_on_submit=True):
                    add_input_col, add_btn_col = st.columns([4, 1.3], gap="small")
                    with add_input_col:
                        new_symbol = st.text_input(
                            "List Your Trending Stocks",
                            key="tmr_symbol_input",
                            placeholder="e.g. RELIANCE",
                        )
                    with add_btn_col:
                        st.markdown('<div class="tmr-add-scope"></div>', unsafe_allow_html=True)
                        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                        add_clicked = st.form_submit_button(
                            "＋ Add",
                            use_container_width=True,
                            disabled=len(store["picks"]) >= 20,
                        )

                if add_clicked:
                    symbol = _normalize_tomorrow_symbol(new_symbol)
                    if symbol:
                        _save_result = _save_symbols_to_tomorrow_store([symbol])
                        if _save_result["added"]:
                            st.rerun()
                        elif _save_result["limit_reached"]:
                            st.warning("Maximum 20 stocks reached. Remove one before adding a new symbol.")
                        else:
                            st.info(f"{symbol} is already saved.")

                if len(store["picks"]) >= 20:
                    st.caption("Maximum 20 stocks reached.")

                if not store["picks"]:
                    st.markdown(
                        '<div style="font-size:13px;color:#4a6480;padding:18px 0 8px 0;">'
                        'No stocks added yet.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for idx, symbol in enumerate(store["picks"]):
                        with st.container():
                            st.markdown('<div class="tmr-remove-scope"></div>', unsafe_allow_html=True)
                            card_col, remove_col = st.columns([6, 1], gap="small")
                            with card_col:
                                st.markdown(
                                    f'<div class="tmr-pick-row">'
                                    f'<div class="tmr-pick-badge">{idx + 1}</div>'
                                    f'<div><div class="tmr-pick-symbol">{html.escape(symbol)}</div>'
                                    f'<div class="tmr-pick-meta">Saved permanently until you delete this stock</div></div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            with remove_col:
                                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                                remove_clicked = st.button(
                                    "✕",
                                    key=f"tmr_remove_{idx}",
                                    use_container_width=True,
                                )
                            if remove_clicked:
                                del store["picks"][idx]
                                _persist_tomorrow_store(store)
                                st.rerun()

        with right_col:
            with st.container():
                st.markdown('<div class="tmr-notes-panel-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="tmr-notes-title">Notes</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="tmr-notes-caption">Keep your thesis, levels, risks, and reminders here. Notes stay saved with your picks.</div>',
                    unsafe_allow_html=True,
                )
                if notes_feedback_msg:
                    st.markdown(
                        (
                            f'<div class="tmr-notes-status tmr-notes-status-{notes_feedback_kind}">'
                            f'<strong>{html.escape(notes_feedback_msg)}</strong>'
                            f'{f"<br><span>Checked at {html.escape(notes_feedback_at)}</span>" if notes_feedback_at else ""}'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    _default_note_status = (
                        "Saved note is available and ready to edit."
                        if saved_notes.strip()
                        else "No saved note yet. Add one and click Save Notes to persist it."
                    )
                    st.markdown(
                        (
                            '<div class="tmr-notes-status tmr-notes-status-info">'
                            f'<strong>{_default_note_status}</strong>'
                            '<br><span>The panel verifies storage after every note save.</span>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )
                with st.form("tmr_notes_form", clear_on_submit=False):
                    notes_value = st.text_area(
                        "Notes",
                        key="tmr_notes_area",
                        height=400,
                        placeholder="Example: Breakout above 20D high, watch volume confirmation, invalidation below EMA20...",
                        label_visibility="collapsed",
                    )
                    notes_btn_col, notes_reset_col = st.columns(2, gap="small")
                    with notes_btn_col:
                        save_notes_clicked = st.form_submit_button(
                            "Save Notes",
                            use_container_width=True,
                        )
                    with notes_reset_col:
                        reset_notes_clicked = st.form_submit_button(
                            "Revert Saved",
                            use_container_width=True,
                        )
                if reset_notes_clicked:
                    st.session_state["tmr_notes_area"] = saved_notes
                    _set_tomorrow_notes_feedback("info", "Reverted notes back to the last saved version.")
                    st.rerun()
                if save_notes_clicked:
                    if notes_value == store["notes"]:
                        _set_tomorrow_notes_feedback("info", "Notes already match the saved version.")
                    else:
                        store["notes"] = notes_value
                        _persist_tomorrow_store(store)
                        _verified, _verified_mode = _verify_tomorrow_notes_saved(notes_value)
                        if _verified:
                            _target_name = "Google Sheets" if _verified_mode == "cloud" else "local storage"
                            _set_tomorrow_notes_feedback("success", f"Notes saved and verified in {_target_name}.")
                        else:
                            _set_tomorrow_notes_feedback("error", "Save was attempted, but verification did not confirm the notes in storage.")
                    st.rerun()
                st.markdown(
                    '<div class="tmr-notes-helper">Tip: Save Notes after any change. The panel will confirm whether the note was really written.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "Close",
            key="tmr_picks_close_btn",
            use_container_width=True,
            on_click=_close_tomorrow_picks_panel,
        )


def render_tomorrow_picks_ticker_strip() -> None:
    """Render a news-style rolling ticker using Tomorrow's Picks symbols."""
    store, _storage_mode = _load_tomorrow_store()
    picks = [
        str(symbol).strip().upper()
        for symbol in store.get("picks", [])
        if str(symbol).strip()
    ]

    if not picks:
        picks = ["ADD STOCKS IN TOMORROW'S PICKS PANEL TO START THE LIVE TICKER"]

    if len(picks) == 1:
        ticker_items = picks * 6
    elif len(picks) == 2:
        ticker_items = picks * 4
    elif len(picks) == 3:
        ticker_items = picks * 3
    else:
        ticker_items = picks

    sequence_html = "".join(
        (
            f'<span class="tmr-roll-item">{html.escape(symbol)}</span>'
            '<span class="tmr-roll-sep">•</span>'
        )
        for symbol in ticker_items
    )
    duration_s = max(18, min(48, int(len(" ".join(ticker_items)) * 0.42) + 8))

    st.markdown(
        (
            '<div class="tmr-roll-shell">'
            '<div class="tmr-roll-label">Tomorrow&apos;s Picks</div>'
            '<div class="tmr-roll-viewport">'
            f'<div class="tmr-roll-track" style="--tmr-roll-duration:{duration_s}s;">'
            f'<div class="tmr-roll-sequence">{sequence_html}</div>'
            f'<div class="tmr-roll-sequence" aria-hidden="true">{sequence_html}</div>'
            '</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

_STOCK_AURA_OK = True   # always True — no external dependency

# ── optional sklearn (graceful fallback if missing) ───────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# NOTE:
# External module imports (scoring_engine/backtest_engine/ml_engine/ui_components)
# intentionally removed. Mode-specific strategy engines are used directly.

# ─────────────────────────────────────────────────────────────────────
# YFINANCE THROTTLING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────
MAX_YF_CONCURRENCY = 12
_YF_SEM = threading.BoundedSemaphore(MAX_YF_CONCURRENCY)

_MKT_LOCK   = threading.Lock()
_MKT_CACHE: dict[str, float] = {}
_NIFTY_LOCK = threading.Lock()
_NIFTY_20D_RET: float | None = None

# SPEED FIX — restore mktcap cache from session_state on each rerun so
# mode 1/2 re-scans don't re-fetch tickers already looked up this session.
try:
    _ss_mkt = st.session_state.get("_mkt_cache_store", {})
    if _ss_mkt:
        _MKT_CACHE.update(_ss_mkt)
except Exception:
    pass


def get_mktcap_cr(ticker: str) -> float:
    """DO NOT MODIFY — strategy rule"""
    with _MKT_LOCK:
        if ticker in _MKT_CACHE:
            return _MKT_CACHE[ticker]
    try:
        with _YF_SEM:
            info = yf.Ticker(ticker).fast_info
            raw  = getattr(info, "market_cap", 0) or 0
    except Exception:
        raw = 0
    mc_cr = float(raw) / 1e7 if raw else 0.0
    with _MKT_LOCK:
        _MKT_CACHE[ticker] = mc_cr
    # SPEED FIX — persist new entry to session_state so next rerun skips API call
    try:
        if "_mkt_cache_store" not in st.session_state:
            st.session_state["_mkt_cache_store"] = {}
        st.session_state["_mkt_cache_store"][ticker] = mc_cr
    except Exception:
        pass
    return mc_cr


def get_nifty_20d_return() -> float | None:
    """20-day return for Nifty (^NSEI), shared across all stocks.
    BUG FIX: Applies Time Travel cutoff so Mode 4 relative-strength
    comparison uses historical Nifty data, not live current data.
    """
    global _NIFTY_20D_RET
    with _NIFTY_LOCK:
        if _NIFTY_20D_RET is not None:
            return _NIFTY_20D_RET
    try:
        with _YF_SEM:
            df_n = yf.download(
                "^NSEI", period="2mo", interval="1d",
                auto_adjust=True, progress=False, timeout=10, threads=False,
            )
        # BUG FIX: Without this, Mode 4 compares a historical stock return
        # against today's Nifty return, corrupting relative-strength logic
        # in every Time Travel scan.
        if _TIME_TRAVEL_OK and hasattr(_tt, "apply_time_travel_cutoff"):
            df_n = _tt.apply_time_travel_cutoff(df_n)
        if df_n is None or len(df_n) < 21:
            return None
        close_n = df_n["Close"].dropna()
        if len(close_n) < 21:
            return None
        n_today = float(close_n.iloc[-1])
        n_ago20 = float(close_n.iloc[-21])
        if n_ago20 <= 0:
            return None
        ret = (n_today - n_ago20) / n_ago20
    except Exception:
        return None
    with _NIFTY_LOCK:
        _NIFTY_20D_RET = ret
    return ret


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
def _tomorrow_section_stat_card(bucket: str, count: int) -> str:
    meta = dict(_TOMORROW_SECTION_META.get(bucket, {}))
    label = html.escape(str(meta.get("label", bucket.title()) or bucket.title()))
    caption = html.escape(str(meta.get("caption", "") or ""))
    accent = str(meta.get("accent", "#00d4a8") or "#00d4a8")
    noun = "stock" if int(count) == 1 else "stocks"
    return (
        '<div class="tmr-v2-stat-card" style="--tmr-accent:{accent};">'
        '<div class="tmr-v2-stat-label">{label}</div>'
        '<div class="tmr-v2-stat-value">{count}</div>'
        '<div class="tmr-v2-stat-caption">{caption}</div>'
        '<div class="tmr-v2-stat-foot">{count} saved {noun}</div>'
        '</div>'
    ).format(
        accent=accent,
        label=label,
        count=int(count),
        caption=caption,
        noun=noun,
    )


def render_tomorrow_picks_panel() -> None:
    if not st.session_state.get("tomorrow_picks_show_panel", False):
        return

    store, storage_mode = _load_tomorrow_store()
    sections = _apply_tomorrow_sections_limit(store.get("sections", {}), limit=20)
    store["sections"] = sections
    store["picks"] = _tomorrow_flatten_sections(sections, limit=20)

    saved_count = len(store["picks"])
    slots_left = max(0, 20 - saved_count)
    active_sections = sum(1 for bucket in _TOMORROW_SECTION_ORDER if sections.get(bucket))
    saved_notes = str(store.get("notes", "") or "")
    notes_words = len(saved_notes.split()) if saved_notes.strip() else 0
    notes_feedback = st.session_state.get("tmr_notes_feedback", {}) or {}
    notes_feedback_kind = str(notes_feedback.get("kind", "info") or "info").strip().lower()
    notes_feedback_msg = str(notes_feedback.get("message", "") or "").strip()
    notes_feedback_at = str(notes_feedback.get("at", "") or "").strip()
    if notes_feedback_kind not in {"success", "error", "info"}:
        notes_feedback_kind = "info"

    if "tmr_notes_area" not in st.session_state:
        st.session_state["tmr_notes_area"] = saved_notes

    status_copy = (
        "Cloud sync is active. Picks stay saved until you delete them."
        if storage_mode == "cloud"
        else "Local persistence is active. Picks stay saved until you delete them."
    )
    sync_caption = (
        "Google Sheets keeps your four strips synced across sessions."
        if storage_mode == "cloud"
        else "This machine stores your four strips permanently until removal."
    )
    storage_label = "Cloud Sync" if storage_mode == "cloud" else "Local Save"
    default_bucket = _tomorrow_bucket_for_mode(st.session_state.get("mode", 3))
    default_bucket_index = (
        _TOMORROW_SECTION_ORDER.index(default_bucket)
        if default_bucket in _TOMORROW_SECTION_ORDER
        else 0
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-anchor) {
          background:
            radial-gradient(circle at top right, rgba(240,180,41,0.12), transparent 30%),
            radial-gradient(circle at top left, rgba(0,212,168,0.10), transparent 28%),
            linear-gradient(180deg, rgba(9,13,20,0.97), rgba(5,8,13,0.99));
          border:1px solid rgba(77,107,140,0.34);
          border-radius:22px;
          padding:24px 24px 20px 24px;
          box-shadow:
            0 24px 54px rgba(0,0,0,0.42),
            inset 0 1px 0 rgba(255,255,255,0.04);
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-left-anchor),
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-notes-anchor) {
          background:linear-gradient(180deg, rgba(12,18,28,0.96), rgba(8,13,21,0.98));
          border:1px solid rgba(60,88,118,0.40);
          border-radius:18px;
          padding:18px 18px 20px 18px;
          height:100%;
          box-shadow:
            inset 0 0 0 1px rgba(255,255,255,0.02),
            0 16px 34px rgba(0,0,0,0.18);
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-left-anchor) div[data-testid="stTextInput"] input,
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-left-anchor) div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-notes-anchor) div[data-testid="stTextArea"] textarea {
          background:rgba(9,17,27,0.95) !important;
          color:#dce7f4 !important;
          border:1px solid rgba(56,86,116,0.52) !important;
          box-shadow:none !important;
        }
        div[data-testid="stVerticalBlock"]:has(.tmr-v2-notes-anchor) div[data-testid="stTextArea"] textarea {
          min-height:360px !important;
        }
        .tmr-v2-title {
          font-family:'Syne',sans-serif;
          font-size:44px;
          font-weight:800;
          letter-spacing:-1px;
          color:#f0b429;
          margin:0 0 8px 0;
        }
        .tmr-v2-lead {
          font-size:14px;
          line-height:1.55;
          color:#90b0cd;
          margin:0 0 16px 0;
        }
        .tmr-v2-status {
          border:1px solid rgba(78,110,143,0.34);
          border-left:4px solid #f0b429;
          border-radius:14px;
          padding:12px 14px;
          margin:0 0 20px 0;
          font-size:12px;
          color:#dce7f4;
          background:linear-gradient(90deg, rgba(18,28,41,0.95), rgba(10,16,25,0.98));
        }
        .tmr-v2-status strong {
          color:#f4f8ff;
        }
        .tmr-v2-metrics {
          display:grid;
          grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
          gap:14px;
          margin:0 0 18px 0;
        }
        .tmr-v2-metric {
          border:1px solid rgba(65,96,127,0.34);
          border-radius:16px;
          padding:16px 16px 14px 16px;
          background:linear-gradient(180deg, rgba(10,17,27,0.96), rgba(7,12,20,0.99));
        }
        .tmr-v2-metric-label {
          font-size:11px;
          letter-spacing:1px;
          text-transform:uppercase;
          color:#7d9abb;
          margin-bottom:8px;
        }
        .tmr-v2-metric-value {
          font-family:'Syne',sans-serif;
          font-size:26px;
          font-weight:800;
          color:#f4f8ff;
          line-height:1;
          margin-bottom:6px;
        }
        .tmr-v2-metric-caption {
          font-size:12px;
          line-height:1.45;
          color:#93afcb;
        }
        .tmr-v2-stat-grid {
          display:grid;
          grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
          gap:14px;
          margin:0 0 18px 0;
        }
        .tmr-v2-stat-card {
          border:1px solid color-mix(in srgb, var(--tmr-accent) 30%, rgba(70,96,128,0.35));
          border-radius:16px;
          padding:15px 15px 13px 15px;
          background:linear-gradient(180deg, rgba(11,19,29,0.96), rgba(8,13,21,0.98));
          box-shadow:inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .tmr-v2-stat-label {
          font-size:11px;
          letter-spacing:1px;
          text-transform:uppercase;
          color:var(--tmr-accent);
          margin-bottom:7px;
        }
        .tmr-v2-stat-value {
          font-family:'Syne',sans-serif;
          font-size:28px;
          font-weight:800;
          color:#f4f8ff;
          line-height:1;
          margin-bottom:6px;
        }
        .tmr-v2-stat-caption {
          font-size:12px;
          color:#dce7f4;
          line-height:1.45;
          min-height:34px;
        }
        .tmr-v2-stat-foot {
          margin-top:8px;
          font-size:11px;
          color:#89a8c7;
        }
        .tmr-v2-add-box {
          border:1px solid rgba(70,100,132,0.34);
          border-radius:16px;
          padding:14px 14px 12px 14px;
          margin:0 0 18px 0;
          background:linear-gradient(180deg, rgba(10,17,27,0.94), rgba(8,13,21,0.98));
        }
        .tmr-v2-add-title {
          font-size:13px;
          font-weight:700;
          color:#f4f8ff;
          margin-bottom:4px;
        }
        .tmr-v2-add-caption {
          font-size:11px;
          color:#8aa9c8;
          margin-bottom:10px;
        }
        .tmr-v2-strip-card {
          border:1px solid color-mix(in srgb, var(--tmr-accent) 34%, rgba(55,84,114,0.34));
          border-radius:18px;
          padding:14px 14px 12px 14px;
          margin:0 0 14px 0;
          background:
            radial-gradient(circle at top right, color-mix(in srgb, var(--tmr-accent) 12%, transparent), transparent 38%),
            linear-gradient(180deg, rgba(10,17,27,0.97), rgba(7,12,20,0.99));
          box-shadow:
            inset 0 0 0 1px rgba(255,255,255,0.02),
            0 14px 30px rgba(0,0,0,0.16);
        }
        .tmr-v2-strip-head {
          display:flex;
          align-items:flex-start;
          justify-content:space-between;
          gap:12px;
          margin-bottom:10px;
        }
        .tmr-v2-strip-kicker {
          font-size:10px;
          letter-spacing:1.1px;
          text-transform:uppercase;
          color:var(--tmr-accent);
          margin-bottom:5px;
        }
        .tmr-v2-strip-title {
          font-family:'Syne',sans-serif;
          font-size:22px;
          font-weight:800;
          color:#f4f8ff;
          margin:0 0 4px 0;
        }
        .tmr-v2-strip-caption {
          font-size:12px;
          line-height:1.45;
          color:#8daecf;
          max-width:540px;
        }
        .tmr-v2-strip-count {
          display:inline-flex;
          align-items:center;
          justify-content:center;
          min-width:74px;
          padding:10px 12px;
          border-radius:999px;
          border:1px solid color-mix(in srgb, var(--tmr-accent) 32%, rgba(255,255,255,0.12));
          color:#f4f8ff;
          font-size:12px;
          font-weight:700;
          background:rgba(10,17,27,0.86);
        }
        .tmr-v2-row-symbol {
          font-weight:800;
          color:#f4f8ff;
          font-size:17px;
          letter-spacing:0.2px;
        }
        .tmr-v2-row-meta {
          font-size:11px;
          color:#8daac8;
          margin-top:2px;
        }
        .tmr-v2-empty {
          padding:14px 12px;
          border-radius:14px;
          border:1px dashed rgba(82,112,143,0.30);
          color:#6f8ba8;
          font-size:12px;
          background:rgba(8,13,20,0.58);
          margin:0 0 14px 0;
        }
        .tmr-v2-notes-title {
          font-family:'Syne',sans-serif;
          font-size:28px;
          font-weight:800;
          color:#f4f8ff;
          margin:0 0 8px 0;
        }
        .tmr-v2-notes-caption {
          font-size:12px;
          line-height:1.5;
          color:#8eaed0;
          margin:0 0 14px 0;
        }
        .tmr-v2-notes-status {
          border-radius:14px;
          padding:12px 14px;
          margin:0 0 14px 0;
          border:1px solid rgba(59,90,120,0.38);
          font-size:12px;
          line-height:1.5;
          background:linear-gradient(180deg, rgba(10,17,26,0.96), rgba(8,13,20,0.98));
          color:#dce7f4;
        }
        .tmr-v2-notes-status strong {
          color:#f4f8ff;
        }
        .tmr-v2-notes-status-success {
          border-color:rgba(0,212,168,0.34);
          background:linear-gradient(180deg, rgba(8,36,30,0.86), rgba(7,16,23,0.98));
        }
        .tmr-v2-notes-status-error {
          border-color:rgba(255,77,109,0.34);
          background:linear-gradient(180deg, rgba(48,18,25,0.88), rgba(11,14,22,0.98));
        }
        .tmr-v2-notes-status-info {
          border-color:rgba(240,180,41,0.30);
          background:linear-gradient(180deg, rgba(50,39,14,0.82), rgba(10,15,24,0.98));
        }
        .tmr-v2-notes-helper {
          font-size:11px;
          color:#7e9aba;
          margin-top:10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="tmr-v2-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="tmr-v2-title">Tomorrow\'s Picks</div>', unsafe_allow_html=True)
        st.markdown(
            (
                '<div class="tmr-v2-lead">'
                'Four dedicated strips keep Relax, Swing, Intraday, and Breakout ideas separate. '
                'Any ADD IN PICKS action now lands in the correct strip automatically, and each pick stays saved until you remove it.'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                '<div class="tmr-v2-status">'
                f'<strong>{html.escape(status_copy)}</strong><br>{html.escape(sync_caption)}'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                '<div class="tmr-v2-metrics">'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Saved Picks</div><div class="tmr-v2-metric-value">{saved_count}</div><div class="tmr-v2-metric-caption">Total stocks saved across all four strips</div></div>'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Slots Left</div><div class="tmr-v2-metric-value">{slots_left}</div><div class="tmr-v2-metric-caption">Maximum 20 saved picks in total</div></div>'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Active Strips</div><div class="tmr-v2-metric-value">{active_sections}/4</div><div class="tmr-v2-metric-caption">How many strategy lanes currently hold picks</div></div>'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Storage</div><div class="tmr-v2-metric-value">{html.escape(storage_label)}</div><div class="tmr-v2-metric-caption">{html.escape(sync_caption)}</div></div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="tmr-v2-stat-grid">'
            + "".join(
                _tomorrow_section_stat_card(bucket, len(sections.get(bucket, [])))
                for bucket in _TOMORROW_SECTION_ORDER
            )
            + '</div>',
            unsafe_allow_html=True,
        )

        left_col, right_col = st.columns([3.2, 2], gap="large")

        with left_col:
            with st.container():
                st.markdown('<div class="tmr-v2-left-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="tmr-v2-add-box">'
                    '<div class="tmr-v2-add-title">Quick Add to a Strip</div>'
                    '<div class="tmr-v2-add-caption">Choose the lane first, then save a manual symbol into that strategy bucket.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                with st.form("tmr_add_form_v2", clear_on_submit=True):
                    add_input_col, add_bucket_col, add_button_col = st.columns([2.2, 1.2, 1], gap="small")
                    with add_input_col:
                        new_symbol = st.text_input(
                            "Add Symbol",
                            key="tmr_symbol_input_v2",
                            placeholder="e.g. RELIANCE",
                        )
                    with add_bucket_col:
                        manual_bucket = st.selectbox(
                            "Strip",
                            options=list(_TOMORROW_SECTION_ORDER),
                            index=default_bucket_index,
                            format_func=_tomorrow_section_label,
                            key="tmr_manual_bucket",
                        )
                    with add_button_col:
                        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                        add_clicked = st.form_submit_button(
                            "Add",
                            use_container_width=True,
                            disabled=saved_count >= 20,
                        )

                if add_clicked:
                    symbol = _normalize_tomorrow_symbol(new_symbol)
                    if not symbol:
                        st.info("Enter a valid stock symbol first.")
                    else:
                        add_summary = _save_symbols_to_tomorrow_store([symbol], bucket=manual_bucket)
                        bucket_label = str(add_summary.get("bucket_label", _tomorrow_section_label(manual_bucket)))
                        if add_summary["added"] or add_summary["moved"]:
                            st.success(f"{symbol} is now saved in the {bucket_label} strip.")
                            store, storage_mode = _load_tomorrow_store()
                            sections = _apply_tomorrow_sections_limit(store.get("sections", {}), limit=20)
                            store["sections"] = sections
                            store["picks"] = _tomorrow_flatten_sections(sections, limit=20)
                            saved_count = len(store["picks"])
                            slots_left = max(0, 20 - saved_count)
                            active_sections = sum(1 for bucket in _TOMORROW_SECTION_ORDER if sections.get(bucket))
                        elif add_summary["limit_reached"]:
                            st.warning("Tomorrow's Picks already has 20 saved stocks. Remove one before adding another.")
                        else:
                            st.info(f"{symbol} is already in the {bucket_label} strip.")

                if saved_count >= 20:
                    st.caption("Tomorrow's Picks is full at 20 total saved stocks.")

                for bucket in _TOMORROW_SECTION_ORDER:
                    meta = dict(_TOMORROW_SECTION_META.get(bucket, {}))
                    label = str(meta.get("label", bucket.title()) or bucket.title())
                    caption = str(meta.get("caption", "") or "")
                    accent = str(meta.get("accent", "#00d4a8") or "#00d4a8")
                    bucket_symbols = list(sections.get(bucket, []))

                    st.markdown(
                        (
                            f'<div class="tmr-v2-strip-card" style="--tmr-accent:{accent};">'
                            '<div class="tmr-v2-strip-head">'
                            '<div>'
                            '<div class="tmr-v2-strip-kicker">Strategy Strip</div>'
                            f'<div class="tmr-v2-strip-title">{html.escape(label)}</div>'
                            f'<div class="tmr-v2-strip-caption">{html.escape(caption)}</div>'
                            '</div>'
                            f'<div class="tmr-v2-strip-count">{len(bucket_symbols)} saved</div>'
                            '</div>'
                            '</div>'
                        ),
                        unsafe_allow_html=True,
                    )

                    if not bucket_symbols:
                        st.markdown(
                            f'<div class="tmr-v2-empty">No stocks saved in the {html.escape(label)} strip yet. Use ADD IN PICKS from that mode, or add a manual symbol above.</div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    for idx, symbol in enumerate(bucket_symbols):
                        row_col, meta_col, remove_col = st.columns([2.1, 2.8, 1], gap="small")
                        with row_col:
                            st.markdown(
                                (
                                    f'<div class="tmr-v2-row-symbol">{html.escape(symbol)}</div>'
                                    f'<div class="tmr-v2-row-meta">{html.escape(label)} strip</div>'
                                ),
                                unsafe_allow_html=True,
                            )
                        with meta_col:
                            st.markdown(
                                (
                                    '<div class="tmr-v2-row-meta">'
                                    f'Auto-routed from {html.escape(label)} results and stored permanently until removed.'
                                    '</div>'
                                ),
                                unsafe_allow_html=True,
                            )
                        with remove_col:
                            remove_clicked = st.button(
                                "Remove",
                                key=f"tmr_v2_remove_{bucket}_{idx}",
                                use_container_width=True,
                            )
                        if remove_clicked:
                            updated_sections = {
                                name: list(values)
                                for name, values in sections.items()
                            }
                            updated_sections[bucket] = [
                                saved_symbol
                                for saved_symbol in bucket_symbols
                                if saved_symbol != symbol
                            ]
                            store["sections"] = _apply_tomorrow_sections_limit(updated_sections, limit=20)
                            store["picks"] = _tomorrow_flatten_sections(store["sections"], limit=20)
                            _persist_tomorrow_store(store)
                            st.rerun()

        with right_col:
            with st.container():
                st.markdown('<div class="tmr-v2-notes-anchor"></div>', unsafe_allow_html=True)
                st.markdown('<div class="tmr-v2-notes-title">Notes</div>', unsafe_allow_html=True)
                st.markdown(
                    (
                        '<div class="tmr-v2-notes-caption">'
                        'Keep your thesis, entry logic, invalidation, or reminders here. '
                        'Notes stay linked to the four-strip Tomorrow Picks board.'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                if notes_feedback_msg:
                    status_html = (
                        f'<div class="tmr-v2-notes-status tmr-v2-notes-status-{notes_feedback_kind}">'
                        f'<strong>{html.escape(notes_feedback_msg)}</strong>'
                        f'{f"<br><span>Checked at {html.escape(notes_feedback_at)}</span>" if notes_feedback_at else ""}'
                        '</div>'
                    )
                else:
                    default_note_status = (
                        "Saved notes are ready to edit."
                        if saved_notes.strip()
                        else "No saved notes yet. Add one and click Save Notes."
                    )
                    status_html = (
                        '<div class="tmr-v2-notes-status tmr-v2-notes-status-info">'
                        f'<strong>{html.escape(default_note_status)}</strong>'
                        '<br><span>The panel verifies storage after each note save.</span>'
                        '</div>'
                    )
                st.markdown(status_html, unsafe_allow_html=True)

                with st.form("tmr_notes_form_v2", clear_on_submit=False):
                    notes_value = st.text_area(
                        "Notes",
                        key="tmr_notes_area",
                        height=360,
                        placeholder="Example: Relax strip for slower continuation names, Breakout strip for radar/live pulse setups, invalidation below EMA20, partial profit zones...",
                        label_visibility="collapsed",
                    )
                    notes_btn_col, notes_reset_col = st.columns(2, gap="small")
                    with notes_btn_col:
                        save_notes_clicked = st.form_submit_button(
                            "Save Notes",
                            use_container_width=True,
                        )
                    with notes_reset_col:
                        reset_notes_clicked = st.form_submit_button(
                            "Revert Saved",
                            use_container_width=True,
                        )

                if reset_notes_clicked:
                    st.session_state["tmr_notes_area"] = saved_notes
                    _set_tomorrow_notes_feedback("info", "Reverted notes back to the last saved version.")
                    st.rerun()
                if save_notes_clicked:
                    if notes_value == store["notes"]:
                        _set_tomorrow_notes_feedback("info", "Notes already match the saved version.")
                    else:
                        store["notes"] = notes_value
                        _persist_tomorrow_store(store)
                        verified, verified_mode = _verify_tomorrow_notes_saved(notes_value)
                        if verified:
                            target_name = "Google Sheets" if verified_mode == "cloud" else "local storage"
                            _set_tomorrow_notes_feedback("success", f"Notes saved and verified in {target_name}.")
                        else:
                            _set_tomorrow_notes_feedback("error", "Save was attempted, but verification did not confirm the notes in storage.")
                    st.rerun()

                st.markdown(
                    f'<div class="tmr-v2-notes-helper">Words saved: {notes_words}. Tip: keep strip-specific notes here so Relax, Swing, Intraday, and Breakout decisions stay in one place.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "Close",
            key="tmr_picks_close_btn_v2",
            use_container_width=True,
            on_click=_close_tomorrow_picks_panel,
        )


def render_tomorrow_picks_ticker_strip() -> None:
    store, _storage_mode = _load_tomorrow_store()
    sections = _apply_tomorrow_sections_limit(store.get("sections", {}), limit=20)

    st.markdown(
        """
        <style>
        div[data-testid="stElementContainer"]:has(.tmr-board-shell) {
          margin-top:-66px !important;
          margin-bottom:2px !important;
        }
        .tmr-board-shell {
          border:1px solid rgba(86,118,150,0.34);
          border-radius:18px;
          padding:12px 14px 10px 14px;
          background:
            radial-gradient(circle at top right, rgba(240,180,41,0.12), transparent 26%),
            linear-gradient(180deg, rgba(10,17,27,0.97), rgba(6,10,17,0.99));
          box-shadow:
            0 14px 28px rgba(0,0,0,0.16),
            inset 0 0 0 1px rgba(255,255,255,0.02);
          margin:-18px 0 8px 0;
        }
        .tmr-board-header {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:10px;
          flex-wrap:wrap;
          margin-bottom:8px;
        }
        .tmr-board-title {
          font-family:'Syne',sans-serif;
          font-size:16px;
          font-weight:800;
          letter-spacing:0.8px;
          text-transform:uppercase;
          color:#f0b429;
        }
        .tmr-board-copy {
          font-size:11px;
          color:#88a8c7;
        }
        .tmr-board-row {
          display:grid;
          grid-template-columns:minmax(120px, 150px) 1fr;
          gap:10px;
          align-items:center;
          padding:8px 0;
          border-top:1px solid rgba(42,61,86,0.46);
        }
        .tmr-board-row:first-of-type {
          border-top:none;
          padding-top:0;
        }
        .tmr-board-label-wrap {
          display:flex;
          align-items:center;
          gap:8px;
          flex-wrap:wrap;
        }
        .tmr-board-label {
          display:inline-flex;
          align-items:center;
          gap:8px;
          width:max-content;
          padding:6px 10px;
          border-radius:999px;
          border:1px solid color-mix(in srgb, var(--tmr-accent) 34%, rgba(255,255,255,0.12));
          background:rgba(9,15,24,0.88);
          color:#f4f8ff;
          font-size:11px;
          font-weight:800;
          letter-spacing:0.4px;
          text-transform:uppercase;
        }
        .tmr-board-label-count {
          color:var(--tmr-accent);
        }
        .tmr-board-items {
          display:flex;
          flex-wrap:wrap;
          gap:6px;
          align-items:center;
        }
        .tmr-board-chip {
          display:inline-flex;
          align-items:center;
          gap:8px;
          padding:7px 10px;
          border-radius:999px;
          border:1px solid color-mix(in srgb, var(--tmr-accent) 28%, rgba(72,102,134,0.30));
          background:linear-gradient(180deg, rgba(13,22,33,0.94), rgba(9,15,24,0.98));
          color:#dce7f4;
          font-size:11px;
          font-weight:700;
        }
        .tmr-board-chip-badge {
          display:inline-flex;
          align-items:center;
          justify-content:center;
          min-width:30px;
          padding:3px 7px;
          border-radius:999px;
          background:rgba(9,28,36,0.92);
          color:var(--tmr-accent);
          border:1px solid color-mix(in srgb, var(--tmr-accent) 32%, rgba(255,255,255,0.10));
          font-size:9px;
          letter-spacing:0.7px;
          text-transform:uppercase;
        }
        .tmr-board-empty {
          display:inline-flex;
          align-items:center;
          min-height:28px;
          padding:0 2px;
          color:#6f8ba8;
          font-size:11px;
        }
        .tmr-board-overflow {
          display:inline-flex;
          align-items:center;
          justify-content:center;
          min-width:36px;
          padding:7px 10px;
          border-radius:999px;
          border:1px dashed rgba(106,130,158,0.38);
          color:#8fb0cf;
          font-size:11px;
          font-weight:700;
        }
        @media (max-width: 900px) {
          div[data-testid="stElementContainer"]:has(.tmr-board-shell) {
            margin-top:-18px !important;
          }
          .tmr-board-shell {
            margin:-8px 0 8px 0;
          }
          .tmr-board-row {
            grid-template-columns:1fr;
            gap:6px;
          }
          .tmr-board-label-wrap {
            justify-content:flex-start;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    rows_html: list[str] = []
    for bucket in _TOMORROW_SECTION_ORDER:
        meta = dict(_TOMORROW_SECTION_META.get(bucket, {}))
        label = html.escape(str(meta.get("label", bucket.title()) or bucket.title()))
        accent = str(meta.get("accent", "#00d4a8") or "#00d4a8")
        bucket_symbols = list(sections.get(bucket, []))
        if bucket_symbols:
            visible_symbols = bucket_symbols[:4]
            hidden_count = max(0, len(bucket_symbols) - len(visible_symbols))
            items_html = "".join(
                (
                    '<span class="tmr-board-chip">'
                    '<span class="tmr-board-chip-badge">NSE</span>'
                    f'{html.escape(symbol)}'
                    '</span>'
                )
                for symbol in visible_symbols
            )
            if hidden_count:
                items_html += f'<span class="tmr-board-overflow">+{hidden_count}</span>'
        else:
            items_html = '<span class="tmr-board-empty">Empty strip</span>'

        rows_html.append(
            (
                f'<div class="tmr-board-row" style="--tmr-accent:{accent};">'
                '<div class="tmr-board-label-wrap">'
                f'<div class="tmr-board-label">{label} <span class="tmr-board-label-count">{len(bucket_symbols)}</span></div>'
                '</div>'
                f'<div class="tmr-board-items">{items_html}</div>'
                '</div>'
            )
        )

    st.markdown(
        (
            '<div class="tmr-board-shell">'
            '<div class="tmr-board-header">'
            '<div class="tmr-board-title">Tomorrow\'s Picks</div>'
            '<div class="tmr-board-copy">Compact 4-lane view: Relax, Swing, Intraday, Breakout.</div>'
            '</div>'
            + "".join(rows_html)
            + '</div>'
        ),
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="NSE Sentinel — Stock Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  (Space Mono + Syne, terminal/Bloomberg aesthetic)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #060a0f; --bg2: #0b1017; --bg3: #0f1823;
  --border: #1a2840; --border2: #243550;
  --accent: #00d4a8; --accent2: #0094ff; --accent3: #f0b429;
  --red: #ff4d6d; --text: #ccd9e8; --muted: #4a6480;
  --mono: 'Space Mono', monospace; --sans: 'Syne', sans-serif;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--mono) !important; }
.stApp::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px); }
.stApp::after { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(0,212,168,0.018) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,168,0.018) 1px,transparent 1px);
  background-size:40px 40px; }
[data-testid="stSidebar"] { background-color:var(--bg2) !important; border-right:1px solid var(--border) !important; font-family:var(--mono) !important; }
[data-testid="stSidebar"] * { color:var(--text) !important; }
section[data-testid="stSidebar"] > div { padding-top:4px !important; }
@media (min-width: 781px) {
  [data-testid="stHeader"],
  .stApp > header {
    display: block !important;
    height: auto !important;
    min-height: 3.75rem !important;
    background: transparent !important;
    border: 0 !important;
    overflow: visible !important;
  }
  section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 26rem !important;
    min-width: 26rem !important;
    max-width: 26rem !important;
  }
  section[data-testid="stSidebar"][aria-expanded="true"] > div {
    width: 26rem !important;
    min-width: 26rem !important;
  }
  section[data-testid="stMain"] {
    padding-top: 0.5rem !important;
  }
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; font-family:var(--mono) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:var(--accent) !important; box-shadow:0 0 8px var(--accent) !important; }
[data-testid="stMetric"] { background:var(--bg2) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:18px 20px !important; }
[data-testid="stMetricValue"] { font-family:var(--sans) !important; font-weight:800 !important; font-size:2rem !important; color:var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family:var(--sans) !important; font-size:10px !important; letter-spacing:2px !important; text-transform:uppercase !important; color:var(--muted) !important; }
[data-testid="stMetricDelta"] { color:var(--accent3) !important; }
.stButton > button { background:transparent !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:var(--mono) !important; font-size:13px !important; font-weight:700 !important; letter-spacing:1px !important; border-radius:10px !important; padding:14px 28px !important; width:100% !important; transition:transform 0.14s ease, box-shadow 0.18s ease, background 0.18s ease, color 0.18s ease, border-color 0.18s ease !important; position:relative !important; overflow:hidden !important; isolation:isolate !important; }
.stButton > button::before,
.stDownloadButton > button::before { content:""; position:absolute; inset:-2px auto -2px -40%; width:32%; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.16),transparent); transform:skewX(-22deg); opacity:0; pointer-events:none; z-index:0; }
.stButton > button::after,
.stDownloadButton > button::after { content:""; position:absolute; width:18px; height:18px; left:50%; top:50%; border-radius:999px; background:rgba(255,255,255,0.24); transform:translate(-50%,-50%) scale(0); opacity:0; pointer-events:none; z-index:0; }
.stButton > button > div,
.stDownloadButton > button > div { position:relative; z-index:1; }
.stButton > button:hover::before,
.stDownloadButton > button:hover::before { animation:btn-sheen 0.55s ease; opacity:1; }
.stButton > button:hover { background:var(--accent) !important; color:var(--bg) !important; box-shadow:0 0 18px rgba(0,0,0,0.18), 0 0 18px color-mix(in srgb, var(--accent) 45%, transparent) !important; transform:translateY(-1px); }
.stButton > button:disabled { border-color:var(--muted) !important; color:var(--muted) !important; }
.stButton > button:active,
.stDownloadButton > button:active { transform:scale(0.975) translateY(1px) !important; }
.stButton > button.btn-clicked,
.stDownloadButton > button.btn-clicked { animation:btn-pop 0.34s ease-out; }
.stButton > button.btn-clicked::after,
.stDownloadButton > button.btn-clicked::after { animation:btn-ripple 0.6s ease-out; }
.stDownloadButton > button { background:rgba(0,148,255,0.1) !important; border:1px solid var(--accent2) !important; color:var(--accent2) !important; font-family:var(--mono) !important; font-weight:700 !important; border-radius:8px !important; width:100% !important; transition:transform 0.14s ease, box-shadow 0.18s ease, background 0.18s ease, color 0.18s ease, border-color 0.18s ease !important; position:relative !important; overflow:hidden !important; isolation:isolate !important; }
.stDownloadButton > button:hover { background:var(--accent2) !important; color:var(--bg) !important; box-shadow:0 0 18px rgba(0,0,0,0.18), 0 0 18px rgba(0,148,255,0.35) !important; transform:translateY(-1px); }
.stProgress > div > div { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; box-shadow:0 0 10px var(--accent) !important; }
.stProgress > div { background:var(--border) !important; border-radius:3px !important; height:6px !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:12px !important; overflow:hidden !important; font-family:var(--mono) !important; }
.stDataFrame thead tr th { background:var(--bg3) !important; color:var(--muted) !important; font-family:var(--sans) !important; font-size:10px !important; letter-spacing:1.5px !important; text-transform:uppercase !important; }
.stDataFrame tbody tr:hover td { background:rgba(0,212,168,0.04) !important; }
.stAlert { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:10px !important; font-family:var(--mono) !important; }
h1,h2,h3,h4 { font-family:var(--sans) !important; }
h1 { color:var(--accent) !important; font-weight:800 !important; }
h2,h3 { color:#79c0ff !important; font-weight:700 !important; }
hr { border-color:var(--border) !important; }

@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
@keyframes btn-sheen { 0% { left:-40%; opacity:0; } 30% { opacity:1; } 100% { left:120%; opacity:0; } }
@keyframes btn-ripple { 0% { transform:translate(-50%,-50%) scale(0); opacity:0.34; } 100% { transform:translate(-50%,-50%) scale(14); opacity:0; } }
@keyframes btn-pop { 0% { transform:scale(1); } 35% { transform:scale(0.965); } 70% { transform:scale(1.02); } 100% { transform:scale(1); } }
.live-dot { width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);animation:pdot 2s ease infinite;display:inline-block;margin-right:8px; }
.section-lbl { font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border); }
.mode-pill { display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid currentColor;font-family:var(--mono); }
.pill-m1 { color:#00d4a8;background:rgba(0,212,168,0.08); }
.pill-m2 { color:#0094ff;background:rgba(0,148,255,0.08); }
.pill-m3 { color:#f0b429;background:rgba(240,180,41,0.08); }
.pill-m5 { color:#ff8c00;background:rgba(255,140,0,0.08); }
.pill-m6 { color:#ff4d6d;background:rgba(255,77,109,0.08); }
.top-banner { display:flex;align-items:center;gap:16px;padding:20px 0 8px 0; }
.banner-logo { font-family:var(--sans);font-weight:800;font-size:26px;color:var(--accent);letter-spacing:-0.5px; }
.count-pill { background:rgba(0,212,168,0.1);border:1px solid var(--accent);color:var(--accent);border-radius:20px;padding:2px 12px;font-size:13px;font-weight:700;font-family:var(--mono); }
.result-hdr { display:flex;align-items:center;gap:12px;padding:14px 0; }
.result-hdr h3 { margin:0 !important; }
.status-line { display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;font-size:12px;color:var(--muted); }
.sdot { width:7px;height:7px;border-radius:50%;display:inline-block; }
.sdot-green { background:var(--accent);box-shadow:0 0 6px var(--accent); }

/* ── NEW: score badge + top-pick cards ─────────────────────────── */
.score-green  { color:#00d4a8;font-weight:700; }
.score-blue   { color:#0094ff;font-weight:700; }
.score-yellow { color:#f0b429;font-weight:700; }
.score-red    { color:#ff4d6d;font-weight:700; }
.pick-card {
  background:#0b1017;border:1px solid #1a2840;border-radius:14px;
  padding:18px 20px;transition:border 0.2s;
}
.pick-card:hover { border-color:#243550; }
.pick-rank  { font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#00d4a8;line-height:1; }
.pick-sym   { font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#ccd9e8; }
.pick-score { font-size:11px;color:#4a6480;margin-top:4px; }
.trap-badge { color:#ff4d6d;font-weight:700;font-size:12px; }
.breakdown-box {
  background:#060a0f;border:1px solid #1a2840;border-radius:8px;
  padding:10px 14px;font-size:11px;line-height:1.9;color:#4a6480;
}
.tmr-roll-shell {
  display:flex;align-items:center;gap:16px;
  margin:8px 0 18px 0;padding:12px 16px;
  border:1px solid rgba(240,180,41,0.22);
  border-radius:14px;
  background:linear-gradient(90deg, rgba(10,20,32,0.96), rgba(12,28,46,0.88));
  box-shadow:inset 0 0 0 1px rgba(27,47,72,0.42), 0 12px 28px rgba(0,0,0,0.16);
  overflow:hidden;
}
.tmr-roll-label {
  flex:0 0 auto;
  font-family:var(--sans);
  font-size:11px;
  font-weight:800;
  letter-spacing:1.9px;
  text-transform:uppercase;
  color:var(--accent3);
  white-space:nowrap;
}
.tmr-roll-viewport {
  position:relative;
  flex:1 1 auto;
  min-width:0;
  overflow:hidden;
}
.tmr-roll-viewport::before,
.tmr-roll-viewport::after {
  content:"";
  position:absolute;
  top:0;
  bottom:0;
  width:42px;
  pointer-events:none;
  z-index:2;
}
.tmr-roll-viewport::before {
  left:0;
  background:linear-gradient(90deg, rgba(10,20,32,0.98), rgba(10,20,32,0));
}
.tmr-roll-viewport::after {
  right:0;
  background:linear-gradient(270deg, rgba(12,28,46,0.96), rgba(12,28,46,0));
}
.tmr-roll-track {
  display:flex;
  width:max-content;
  min-width:100%;
  animation:tmr-roll-scroll var(--tmr-roll-duration, 28s) linear infinite;
  will-change:transform;
}
.tmr-roll-shell:hover .tmr-roll-track {
  animation-play-state:paused;
}
.tmr-roll-sequence {
  display:flex;
  align-items:center;
  flex:0 0 auto;
}
.tmr-roll-item {
  display:inline-flex;
  align-items:center;
  font-size:13px;
  font-weight:700;
  color:#dfe9f5;
  white-space:nowrap;
}
.tmr-roll-item::before {
  content:"NSE";
  display:inline-flex;
  align-items:center;
  justify-content:center;
  margin-right:10px;
  padding:2px 7px;
  border-radius:999px;
  border:1px solid rgba(0,212,168,0.26);
  background:rgba(0,212,168,0.08);
  color:var(--accent);
  font-size:10px;
  font-weight:700;
  letter-spacing:1px;
}
.tmr-roll-sep {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  margin:0 18px;
  color:rgba(240,180,41,0.72);
  font-size:16px;
}

@keyframes tmr-roll-scroll {
  0% { transform:translateX(0); }
  100% { transform:translateX(-50%); }
}

@media (max-width: 780px) {
  .tmr-roll-shell {
    align-items:flex-start;
    flex-direction:column;
    gap:10px;
    padding:12px 14px;
  }
  .tmr-roll-label {
    font-size:10px;
    letter-spacing:1.5px;
  }
  .tmr-roll-item {
    font-size:12px;
  }
  .tmr-roll-sep {
    margin:0 14px;
  }
}
</style>
""", unsafe_allow_html=True)

components.html(
    """
    <script>
    const wireAnimatedButtons = () => {
      const selectors = [".stButton > button", ".stDownloadButton > button"];
      document.querySelectorAll(selectors.join(",")).forEach((btn) => {
        if (btn.dataset.animReady === "1") return;
        btn.dataset.animReady = "1";
        btn.addEventListener("click", () => {
          btn.classList.remove("btn-clicked");
          void btn.offsetWidth;
          btn.classList.add("btn-clicked");
          window.setTimeout(() => btn.classList.remove("btn-clicked"), 650);
        });
      });
    };
    wireAnimatedButtons();
    const observer = new MutationObserver(() => wireAnimatedButtons());
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    height=0,
    width=0,
)

inject_animations()

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
      transition: none !important;
      filter: none !important;
    }
    [data-testid="stMainBlockContainer"] {
      padding-top: 1.1rem !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
      opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# NSE TICKER LOADER
# ─────────────────────────────────────────────────────────────────────

_TICKER_GOOD_COUNT = 2000
_TICKER_FALLBACK_MIN_COUNT = 500
_TICKER_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{0,20}\.NS$")
_HARDCODED_TICKER_FALLBACK = [
    "20MICRONS.NS",
    "21STCENMGM.NS",
    "360ONE.NS",
    "3BBLACKBIO.NS",
    "3IINFOLTD.NS",
    "3MINDIA.NS",
    "3PLAND.NS",
    "5PAISA.NS",
    "63MOONS.NS",
    "A2ZINFRA.NS",
    "AAATECH.NS",
    "AADHARHFC.NS",
    "AAKASH.NS",
    "AAREYDRUGS.NS",
    "AARNAV.NS",
    "AARON.NS",
    "AARTECH.NS",
    "AARTI.NS",
    "AARTIDRUGS.NS",
    "AARTIIND.NS",
    "AARTIPHARM.NS",
    "AARTISURF.NS",
    "AARVI.NS",
    "AAVAS.NS",
    "ABAN.NS",
    "ABANSENT.NS",
    "ABB.NS",
    "ABBOTINDIA.NS",
    "ABCAPITAL.NS",
    "ABCOTS.NS",
    "ABDL.NS",
    "ABDPL.NS",
    "ABFRL.NS",
    "ABGSHIP.NS",
    "ABHICAP.NS",
    "ABHIINFRAS.NS",
    "ABINBEV.NS",
    "ABINFRA.NS",
    "ABLBL.NS",
    "ABMINTL.NS",
    "ABMINTLLTD.NS",
    "ABMKNO.NS",
    "ABREL.NS",
    "ABSLAMC.NS",
    "ACC.NS",
    "ACCELYA.NS",
    "ACCURACY.NS",
    "ACE.NS",
    "ACEINTEG.NS",
    "ACI.NS",
    "ACL.NS",
    "ACME.NS",
    "ACMESOLAR.NS",
    "ACROPETAL.NS",
    "ACROW.NS",
    "ACRYSIL.NS",
    "ACSTECH.NS",
    "ACUTAAS.NS",
    "ADANI.NS",
    "ADANIENSOL.NS",
    "ADANIENT.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "ADANIPOWER.NS",
    "ADANITRANS.NS",
    "ADANIWILMAR.NS",
    "ADANIWIND.NS",
    "ADFFOODS.NS",
    "ADHUNIK.NS",
    "ADIPURIYA.NS",
    "ADL.NS",
    "ADOR.NS",
    "ADORWELD.NS",
    "ADPBIO.NS",
    "ADROITINFO.NS",
    "ADSL.NS",
    "ADVAIT.NS",
    "ADVANCE.NS",
    "ADVANCICON.NS",
    "ADVANIHOTR.NS",
    "ADVENTHTL.NS",
    "ADVENZYMES.NS",
    "AECS.NS",
    "AEGISLOG.NS",
    "AEGISVOPAK.NS",
    "AEKTRA.NS",
    "AELPCL.NS",
    "AEPL.NS",
    "AEQUS.NS",
    "AEROENTER.NS",
    "AEROFLEX.NS",
    "AERONEU.NS",
    "AETHER.NS",
    "AFCONS.NS",
    "AFFLE.NS",
    "AFFORDABLE.NS",
    "AFIL.NS",
    "AFRO.NS",
    "AFSL.NS",
    "AGARIND.NS",
    "AGARWALEYE.NS",
    "AGCNET.NS",
    "AGI.NS",
    "AGIIL.NS",
    "AGIO.NS",
    "AGRITECH.NS",
    "AGROPHOS.NS",
    "AGSL.NS",
    "AHCL.NS",
    "AHFL.NS",
    "AHIMSA.NS",
    "AHLADA.NS",
    "AHLEAST.NS",
    "AHLUCONT.NS",
    "AHLWEST.NS",
    "AHMEDABADSTEEL.NS",
    "AIAENG.NS",
    "AIIL.NS",
    "AIMCO.NS",
    "AINDIA.NS",
    "AINDRA.NS",
    "AINTGEN.NS",
    "AIRAN.NS",
    "AIRLINE.NS",
    "AIRO.NS",
    "AIROLAM.NS",
    "AIRTELPP.NS",
    "AJANTPHARM.NS",
    "AJAXENG.NS",
    "AJAXENGG.NS",
    "AJMERA.NS",
    "AJOONI.NS",
    "AKASH.NS",
    "AKASHDEEP.NS",
    "AKASHINFRA.NS",
    "AKCAPIT.NS",
    "AKG.NS",
    "AKGSL.NS",
    "AKHIL.NS",
    "AKI.NS",
    "AKSHAR.NS",
    "AKSHARCHEM.NS",
    "AKSHOPTFBR.NS",
    "AKUMS.NS",
    "AKZOINDIA.NS",
    "ALANKIT.NS",
    "ALBERTDAVD.NS",
    "ALEMBICLTD.NS",
    "ALENNOV.NS",
    "ALESAIND.NS",
    "ALEXISLGO.NS",
    "ALGOQUANT.NS",
    "ALICON.NS",
    "ALIVUS.NS",
    "ALKALI.NS",
    "ALKEM.NS",
    "ALKYLAMINE.NS",
    "ALLCARGO.NS",
    "ALLDIGI.NS",
    "ALLENSOLVE.NS",
    "ALLSEC.NS",
    "ALLTIME.NS",
    "ALMONDZ.NS",
    "ALOKINDS.NS",
    "ALPA.NS",
    "ALPHAGEO.NS",
    "ALPHAICON.NS",
    "ALPHAREALM.NS",
    "ALTIMETRICS.NS",
    "ALTIUS.NS",
    "ALUFLUORIDE.NS",
    "AMAGI.NS",
    "AMANTA.NS",
    "AMARARAJA.NS",
    "AMBALALRES.NS",
    "AMBALALSA.NS",
    "AMBASSADOR.NS",
    "AMBER.NS",
    "AMBICAAGAR.NS",
    "AMBIKCO.NS",
    "AMBUJACEM.NS",
    "AMCO.NS",
    "AMCORIS.NS",
    "AMDIND.NS",
    "AMFIL.NS",
    "AMFORGE.NS",
    "AMINES.NS",
    "AMIORG.NS",
    "AMIRCHAND.NS",
    "AMJLAND.NS",
    "AMJUMBO.NS",
    "AMMAPET.NS",
    "AMNPLST.NS",
    "AMPERE.NS",
    "AMRUTANJAN.NS",
    "AMTL.NS",
    "ANANDRATHI.NS",
    "ANANTRAJ.NS",
    "ANDHRAPAP.NS",
    "ANDHRPAPER.NS",
    "ANDHRPAPMILL.NS",
    "ANDHRSUGAR.NS",
    "ANGELONE.NS",
    "ANGIND.NS",
    "ANIKINDS.NS",
    "ANKIT.NS",
    "ANKITMETAL.NS",
    "ANMOL.NS",
    "ANNPURNA.NS",
    "ANNTL.NS",
    "ANSALAPI.NS",
    "ANTELOPUS.NS",
    "ANTGRAPHIC.NS",
    "ANTHEM.NS",
    "ANTONY.NS",
    "ANUHPHR.NS",
    "ANUP.NS",
    "ANUPAM.NS",
    "ANURAS.NS",
    "APAR.NS",
    "APARINDS.NS",
    "APCL.NS",
    "APCOTEXIND.NS",
    "APEX.NS",
    "APIIND.NS",
    "APL.NS",
    "APLAPOLLO.NS",
    "APLLTD.NS",
    "APOLLO.NS",
    "APOLLOFINVEST.NS",
    "APOLLOHOSP.NS",
    "APOLLOHSP.NS",
    "APOLLOPIPE.NS",
    "APOLLOTYRE.NS",
    "APOLSINHOT.NS",
    "APPLEIND.NS",
    "APPLIEDDNA.NS",
    "APT.NS",
    "APTECHT.NS",
    "APTUS.NS",
    "AQUA.NS",
    "AQUALITE.NS",
    "AQYLON.NS",
    "ARCHIDPLY.NS",
    "ARCHIES.NS",
    "ARCOTECH.NS",
    "ARE&M.NS",
    "AREL.NS",
    "ARENTERP.NS",
    "AREV.NS",
    "ARFIN.NS",
    "ARIES.NS",
    "ARIHANT.NS",
    "ARIHANTCAP.NS",
    "ARIHANTSUP.NS",
    "ARIS.NS",
    "ARKADE.NS",
    "ARMAN.NS",
    "ARMANFIN.NS",
    "ARNITJ.NS",
    "AROGRANITE.NS",
    "AROHAN.NS",
    "ARROWGREEN.NS",
    "ARROWHEAD.NS",
    "ARSHIYA.NS",
    "ARSSBL.NS",
    "ARTEMISMED.NS",
    "ARTNIRMAN.NS",
    "ARTSON.NS",
    "ARVEE.NS",
    "ARVIND.NS",
    "ARVINDFASHN.NS",
    "ARVINDFASN.NS",
    "ARVSMART.NS",
    "ASAHIINDIA.NS",
    "ASAHISONG.NS",
    "ASAL.NS",
    "ASALCBR.NS",
    "ASCOM.NS",
    "ASEL.NS",
    "ASHAPURMIN.NS",
    "ASHARI.NS",
    "ASHIANA.NS",
    "ASHIKA.NS",
    "ASHIMASYN.NS",
    "ASHOKA.NS",
    "ASHOKAMET.NS",
    "ASHOKLEY.NS",
    "ASIANENE.NS",
    "ASIANHOTNR.NS",
    "ASIANPAINT.NS",
    "ASIANTILES.NS",
    "ASKAUTOLTD.NS",
    "ASMS.NS",
    "ASPINWALL.NS",
    "ASSOCEMEN.NS",
    "ASTAR.NS",
    "ASTEC.NS",
    "ASTECINDIA.NS",
    "ASTER.NS",
    "ASTERDM.NS",
    "ASTHAGRAPH.NS",
    "ASTRA.NS",
    "ASTRAL.NS",
    "ASTRAMICRO.NS",
    "ASTRAZEN.NS",
    "ASTRON.NS",
    "ASTTRAL.NS",
    "ATALREAL.NS",
    "ATAM.NS",
    "ATGL.NS",
    "ATHERENERG.NS",
    "ATISHAY.NS",
    "ATL.NS",
    "ATLANTAA.NS",
    "ATLANTAELE.NS",
    "ATLAS.NS",
    "ATLASCOPC.NS",
    "ATLASCYCLE.NS",
    "ATLP.NS",
    "ATSS.NS",
    "ATTICUS.NS",
    "ATUL.NS",
    "ATULAUTO.NS",
    "AUBANK.NS",
    "AURIGROW.NS",
    "AURIONPRO.NS",
    "AUROPHARMA.NS",
    "AURUM.NS",
    "AUSOME.NS",
    "AUSOMENT.NS",
    "AUSTIN.NS",
    "AUTOAXLES.NS",
    "AUTOBEES.NS",
    "AUTOCORP.NS",
    "AUTOIND.NS",
    "AUTONC.NS",
    "AVADHSUGAR.NS",
    "AVAILFC.NS",
    "AVALON.NS",
    "AVANTEL.NS",
    "AVANTIFEED.NS",
    "AVG.NS",
    "AVL.NS",
    "AVONMORE.NS",
    "AVROIND.NS",
    "AVSL.NS",
    "AVTNPL.NS",
    "AWFIS.NS",
    "AWHCL.NS",
    "AWL.NS",
    "AXISBANK.NS",
    "AXISCADES.NS",
    "AXITA.NS",
    "AYE.NS",
    "AYMSYNTEX.NS",
    "AYUDHAN.NS",
    "AZAD.NS",
    "AZIMUTH.NS",
    "BAFNAPH.NS",
    "BAGFILMS.NS",
    "BAIDFIN.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJCON.NS",
    "BAJAJELEC.NS",
    "BAJAJFINSV.NS",
    "BAJAJHCARE.NS",
    "BAJAJHFL.NS",
    "BAJAJHHL.NS",
    "BAJAJHIND.NS",
    "BAJAJHLDNG.NS",
    "BAJAJHOUSING.NS",
    "BAJAJINDEF.NS",
    "BAJAJSFL.NS",
    "BAJAJST.NS",
    "BAJEL.NS",
    "BAJELECTR.NS",
    "BAJFINANCE.NS",
    "BALAJEE.NS",
    "BALAJITELE.NS",
    "BALAMAR.NS",
    "BALAMERCANTILE.NS",
    "BALAMINES.NS",
    "BALASORE.NS",
    "BALAXI.NS",
    "BALI.NS",
    "BALKRISHIND.NS",
    "BALKRISHNA.NS",
    "BALKRISIND.NS",
    "BALMLAWRIE.NS",
    "BALPHARMA.NS",
    "BALRAMCHIN.NS",
    "BALUFORGE.NS",
    "BAMBINO.NS",
    "BANARBEADS.NS",
    "BANARISUG.NS",
    "BANCO.NS",
    "BANCOINDIA.NS",
    "BANDHANBNK.NS",
    "BANG.NS",
    "BANGALHOT.NS",
    "BANKA.NS",
    "BANKBARODA.NS",
    "BANKINDIA.NS",
    "BANKMAHA.NS",
    "BANNARI.NS",
    "BANSALWIRE.NS",
    "BANSWRAS.NS",
    "BARBEQUE.NS",
    "BARCLAYS.NS",
    "BARODA.NS",
    "BASF.NS",
    "BASML.NS",
    "BATAINDIA.NS",
    "BATLIBOI.NS",
    "BAYER.NS",
    "BAYERCROP.NS",
    "BBL.NS",
    "BBOX.NS",
    "BBTC.NS",
    "BBTCL.NS",
    "BCG.NS",
    "BCLIND.NS",
    "BCONCEPTS.NS",
    "BCPL.NS",
    "BDL.NS",
    "BEARDSELL.NS",
    "BECTORFOOD.NS",
    "BEDMUTHA.NS",
    "BEEKAY.NS",
    "BEEKAYST.NS",
    "BEL.NS",
    "BELLACASA.NS",
    "BELRISE.NS",
    "BEML.NS",
    "BENGALASM.NS",
    "BEPL.NS",
    "BERGEPAINT.NS",
    "BESTAGRO.NS",
    "BETA.NS",
    "BFINVEST.NS",
    "BFUTILITIE.NS",
    "BGRENERGY.NS",
    "BHAGCHEM.NS",
    "BHAGERIA.NS",
    "BHAGWATI.NS",
    "BHAGYANAGAR.NS",
    "BHAGYANGR.NS",
    "BHANDARI.NS",
    "BHARAT.NS",
    "BHARATCOAL.NS",
    "BHARATFORG.NS",
    "BHARATGEAR.NS",
    "BHARATRAS.NS",
    "BHARATSE.NS",
    "BHARATWIRE.NS",
    "BHARTIARTL.NS",
    "BHARTIAXML.NS",
    "BHARTIGAS.NS",
    "BHARTIHEXA.NS",
    "BHAVYA.NS",
    "BHEL.NS",
    "BHILWARA.NS",
    "BHORUKA.NS",
    "BI.NS",
    "BIGBLOC.NS",
    "BIGSHARE.NS",
    "BIKAJI.NS",
    "BIL.NS",
    "BIMETAL.NS",
    "BINANIIND.NS",
    "BINDAL.NS",
    "BINDHYA.NS",
    "BIOCON.NS",
    "BIOFILCHEM.NS",
    "BIOPAC.NS",
    "BIRLACABLE.NS",
    "BIRLACORPN.NS",
    "BIRLAMONEY.NS",
    "BIRLANU.NS",
    "BIRLANUVO.NS",
    "BIRLAPREC.NS",
    "BIRLASOFT.NS",
    "BLACKBUCK.NS",
    "BLACKROSE.NS",
    "BLAL.NS",
    "BLBLIMITED.NS",
    "BLIL.NS",
    "BLISS.NS",
    "BLISSGVS.NS",
    "BLKASHYAP.NS",
    "BLS.NS",
    "BLSE.NS",
    "BLUECHIP.NS",
    "BLUECOAST.NS",
    "BLUEDART.NS",
    "BLUEJET.NS",
    "BLUESTAR.NS",
    "BLUESTARCO.NS",
    "BLUESTONE.NS",
]


def _sanitize_ticker_list(tickers) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    try:
        iterable = tickers or []
    except Exception:
        iterable = []

    for raw in iterable:
        bare = str(raw or "").strip().upper().replace(".NS", "")
        if not bare:
            continue

        formatted = f"{bare}.NS"
        if not _TICKER_SYMBOL_RE.fullmatch(formatted):
            continue
        if formatted in seen:
            continue

        seen.add(formatted)
        cleaned.append(formatted)

    return cleaned


def _merge_ticker_lists(*sources) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for source in sources:
        for ticker in _sanitize_ticker_list(source):
            if ticker in seen:
                continue
            seen.add(ticker)
            merged.append(ticker)

    return merged


def _load_local_ticker_master() -> list[str]:
    try:
        if not _TICKER_MASTER_STORE_PATH.exists():
            return []
        payload = json.loads(_TICKER_MASTER_STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            tickers = payload.get("tickers", [])
        else:
            tickers = payload
        return _sanitize_ticker_list(tickers)
    except Exception:
        return []


def _save_local_ticker_master(tickers: list[str]) -> bool:
    normalized = _sanitize_ticker_list(tickers)
    if not normalized:
        return False

    try:
        _TICKER_MASTER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(normalized),
            "tickers": normalized,
        }
        _TICKER_MASTER_STORE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def _save_tickers_to_gsheets(tickers: list) -> None:
    """
    Save full ticker list to Google Sheets tickers tab.
    Called once after a successful live fetch.
    Never crashes — fully wrapped.
    """
    try:
        normalized = sorted({
            symbol if symbol.endswith(".NS") else f"{symbol}.NS"
            for symbol in [
                str(t).strip().upper().replace(".NS", "")
                for t in (tickers or [])
                if str(t).strip()
            ]
        })
        if not normalized:
            return

        import gspread
        from google.oauth2.service_account import Credentials

        _SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=_SCOPES,
        )
        client = gspread.authorize(creds)
        sh = client.open_by_key(
            st.secrets["google_sheets"]["sheet_id"]
        )
        try:
            ws = sh.worksheet("tickers")
        except Exception:
            ws = sh.add_worksheet("tickers", rows=3500, cols=1)

        rows = [["# NSE Tickers — auto-updated"]] + [
            [ticker] for ticker in normalized
        ]
        ws.clear()
        ws.update("A1", rows)
    except Exception:
        pass


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_nse_tickers() -> list:
    # Keep the biggest known universe instead of letting a later
    # partial fetch shrink the scan list after cache/session expiry.
    best_known = _merge_ticker_lists(
        _load_local_ticker_master(),
        st.session_state.get("_ticker_master_list", []),
    )

    # ── LAYER 1: Google Sheets backup (optional permanent storage) ────
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        _SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=_SCOPES,
        )
        client = gspread.authorize(creds)
        sh = client.open_by_key(
            st.secrets["google_sheets"]["sheet_id"]
        )
        try:
            ws = sh.worksheet("tickers")
        except Exception:
            ws = sh.add_worksheet("tickers", rows=5000, cols=1)

        values = ws.col_values(1)
        gs_tickers = [
            value.strip().upper() for value in values
            if value.strip() and not value.strip().startswith("#")
        ]
        best_known = _merge_ticker_lists(best_known, gs_tickers)
    except Exception:
        pass

    # ── LAYER 2: nse_ticker_universe module ───────────────────────────
    try:
        from nse_ticker_universe import (
            get_all_tickers,
            invalidate_cache,
        )

        live_tickers = _sanitize_ticker_list(get_all_tickers(live=True) or [])
        best_known = _merge_ticker_lists(best_known, live_tickers)

        if len(live_tickers) < _TICKER_GOOD_COUNT:
            invalidate_cache()
            fallback_tickers = _sanitize_ticker_list(get_all_tickers(live=False) or [])
            best_known = _merge_ticker_lists(best_known, fallback_tickers)
    except Exception:
        pass

    # ── LAYER 3: nse_tickers.txt in repo ──────────────────────────────
    try:
        f = Path(__file__).with_name("nse_tickers.txt")
        if f.exists():
            repo_symbols = [
                f"{line.strip().upper().replace('.NS', '')}.NS"
                for line in f.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                if line.strip()
            ]
            best_known = _merge_ticker_lists(best_known, repo_symbols)
    except Exception:
        pass

    # ── LAYER 4: hardcoded emergency fallback ─────────────────────────
    # This layer ensures we still have a usable universe if every other
    # source is unavailable, but it should never suppress a larger list.
    best_known = _merge_ticker_lists(best_known, _HARDCODED_TICKER_FALLBACK)

    if len(best_known) < _TICKER_FALLBACK_MIN_COUNT:
        best_known = _sanitize_ticker_list(_HARDCODED_TICKER_FALLBACK)

    if best_known:
        st.session_state["_ticker_master_list"] = best_known
        _save_local_ticker_master(best_known)
        if len(best_known) >= _TICKER_GOOD_COUNT:
            _save_tickers_to_gsheets(best_known)

    return best_known


# ─────────────────────────────────────────────────────────────────────
# TECHNICAL HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    r     = 100 - (100 / (1 + rs))
    return float(r.iloc[-1]) if not r.empty else np.nan


# ─────────────────────────────────────────────────────────────────────
# MARKET BIAS ENGINE (add-on; does not affect scan/mode engines)
# ─────────────────────────────────────────────────────────────────────
def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def compute_market_bias(include_bank: bool = True) -> dict:
    """
    Compute probabilistic market bias for next day using only free yfinance data.
    Fail-safe: returns conservative "Sideways / no edge" output if index data missing.
    """
    try:
        def _fetch_index(symbol: str) -> pd.DataFrame | None:
            try:
                df_i = yf.download(
                    symbol, period="3mo", interval="1d",
                    auto_adjust=True, progress=False, timeout=12, threads=False,
                )
                if df_i is None or df_i.empty:
                    return None
                if isinstance(df_i.columns, pd.MultiIndex):
                    df_i.columns = df_i.columns.get_level_values(0)
                # ── Time-Travel: truncate to historical cutoff ─────────
                if _TIME_TRAVEL_OK:
                    df_i = _tt.apply_time_travel_cutoff(df_i) if hasattr(_tt, "apply_time_travel_cutoff") else df_i
                    if df_i is None or df_i.empty:
                        return None
                return df_i
            except Exception:
                return None

        def _index_features(df_i: pd.DataFrame) -> dict:
            close = df_i["Close"].dropna() if "Close" in df_i.columns else pd.Series(dtype=float)
            if len(close) < 40:
                return {}

            ema20_s = ema(close, 20)
            ema50_s = ema(close, 50)
            rsi14_v = rsi(close, 14)
            c_last  = _safe_float(close.iloc[-1], 0.0)
            e20_last = _safe_float(ema20_s.iloc[-1], 0.0)
            e50_last = _safe_float(ema50_s.iloc[-1], 0.0)

            ret5d = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 and float(close.iloc[-6]) != 0 else np.nan
            ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 and float(close.iloc[-21]) != 0 else np.nan

            # Vol relative to 20D average (if volume exists; indices may lack Volume)
            vol_ratio = None
            if "Volume" in df_i.columns:
                vol = df_i["Volume"].dropna()
                if len(vol) >= 21:
                    avg20 = float(vol.iloc[-21:-1].mean()) if len(vol.iloc[-21:-1]) >= 20 else float(vol.mean())
                    lastv = float(vol.iloc[-1])
                    vol_ratio = (lastv / avg20) if avg20 > 0 else None

            # Realized volatility proxy for expected move range
            ret_1d = close.pct_change().dropna().tail(20) * 100.0
            sigma_pct = float(ret_1d.std()) if not ret_1d.empty else 0.0

            features = {
                "close": c_last,
                "ema20": e20_last,
                "ema50": e50_last,
                "rsi14": _safe_float(rsi14_v, 50.0),
                "ret5d": _safe_float(ret5d, 0.0),
                "ret20d": _safe_float(ret20d, 0.0),
                "vol_ratio": (float(vol_ratio) if vol_ratio is not None else None),
                "sigma_pct": max(0.0, sigma_pct),
            }
            return features

        df_nifty = _fetch_index("^NSEI")
        if df_nifty is None:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "\u00b10.30% to \u00b10.70%",
                "breakdown": ["Nifty data unavailable (fallback)."],
                "regime": "Ranging",
            }

        nifty_feat = _index_features(df_nifty)
        if not nifty_feat:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "\u00b10.30% to \u00b10.70%",
                "breakdown": ["Nifty indicators insufficient (fallback)."],
                "regime": "Ranging",
            }

        bank_feat = None
        if include_bank:
            df_bn = _fetch_index("^NSEBANK")
            if df_bn is not None:
                bf = _index_features(df_bn)
                bank_feat = bf if bf else None

        # Score / interpret (nifty dominates; bank is confirmation)
        return interpret_market_bias(nifty_feat, bank_feat)
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "\u00b10.30% to \u00b10.70%",
            "breakdown": ["Market bias computation failed (fallback)."],
            "regime": "Ranging",
        }




def _classify_regime_nifty(feat: dict) -> str:
    """
    Regime label aligned with grading_engine._REGIME_ADJ keys (soft context only).
    """
    try:
        if not feat:
            return "Ranging"
        c = _safe_float(feat.get("close", 0.0), 0.0)
        e20 = _safe_float(feat.get("ema20", 0.0), 0.0)
        e50 = _safe_float(feat.get("ema50", 0.0), 0.0)
        rsi14_v = _safe_float(feat.get("rsi14", 50.0), 50.0)
        r5 = _safe_float(feat.get("ret5d", 0.0), 0.0)
        sig = _safe_float(feat.get("sigma_pct", 0.5), 0.5)
        if rsi14_v > 72 and r5 > 0.15:
            return "Overbought Pullback Risk"
        if rsi14_v < 32:
            return "Oversold Bounce Zone"
        if c > e20 > e50 and r5 > 0:
            return "Trending Up"
        if c < e20 < e50 and r5 < 0:
            return "Trending Down"
        if sig > 1.15:
            return "High Volatility / Choppy"
        return "Ranging"
    except Exception:
        return "Ranging"


def interpret_market_bias(nifty_feat: dict, bank_feat: dict | None = None) -> dict:
    """
    Convert index features into conservative bias/confidence/expected-range.
    """
    try:
        def _signal(feat: dict) -> tuple[float, dict]:
            close = _safe_float(feat.get("close", 0.0), 0.0)
            ema20_v = _safe_float(feat.get("ema20", 0.0), 0.0)
            ema50_v = _safe_float(feat.get("ema50", 0.0), 0.0)
            rsi14_v = _safe_float(feat.get("rsi14", 50.0), 50.0)
            ret5d_v = _safe_float(feat.get("ret5d", 0.0), 0.0)
            ret20d_v = _safe_float(feat.get("ret20d", 0.0), 0.0)
            vol_ratio = feat.get("vol_ratio", None)

            bull_trend = close > ema20_v > ema50_v
            bear_trend = close < ema20_v < ema50_v
            trend_sig = 1.0 if bull_trend else (-1.0 if bear_trend else 0.0)

            momentum_sig = 1.0 if ret5d_v > 0.30 else (-1.0 if ret5d_v < -0.30 else 0.0)
            rsi_sig = 1.0 if rsi14_v >= 55.0 else (-1.0 if rsi14_v <= 45.0 else 0.0)

            if vol_ratio is None:
                volume_sig = 0.0
            else:
                volume_sig = 1.0 if vol_ratio >= 1.10 else (-1.0 if vol_ratio <= 0.90 else 0.0)

            breakdown = (close < ema20_v and ret20d_v < 0.0)
            support = (close > ema20_v and ret20d_v > 0.0)

            base_score = 0.35 * trend_sig + 0.25 * momentum_sig + 0.20 * rsi_sig + 0.20 * volume_sig

            details = {
                "trend_sig": trend_sig,
                "momentum_sig": momentum_sig,
                "rsi_sig": rsi_sig,
                "volume_sig": volume_sig,
                "breakdown": breakdown,
                "support": support,
                "close": close,
                "ema20": ema20_v,
                "ema50": ema50_v,
                "rsi14": rsi14_v,
                "ret5d": ret5d_v,
                "ret20d": ret20d_v,
                "vol_ratio": vol_ratio,
            }
            return base_score, details

        nifty_score, nf = _signal(nifty_feat)

        bank_score = 0.0
        bf = None
        if bank_feat:
            bank_score, bf = _signal(bank_feat)

        combined = nifty_score
        bank_used = bf is not None
        if bank_used:
            combined = 0.80 * nifty_score + 0.20 * bank_score

        trend_pos = nf["trend_sig"] > 0
        trend_neg = nf["trend_sig"] < 0
        mom_pos = nf["momentum_sig"] > 0
        mom_neg = nf["momentum_sig"] < 0
        rsi_pos = nf["rsi_sig"] > 0
        rsi_neg = nf["rsi_sig"] < 0
        # BUG FIX: When volume data is absent (vol_ratio is None), volume_sig == 0.0
        # (set in _signal()). Absent volume must be NEUTRAL — not bullish, not bearish.
        # Original code set vol_neg = True when vol_ratio is None, causing
        # bearish_strict to fire incorrectly on volume-less indexes (e.g. ^NSEI).
        vol_pos = nf["volume_sig"] > 0 or nf["vol_ratio"] is None
        vol_neg = nf["volume_sig"] < 0   # absent volume is neutral, NOT bearish

        bullish_strict = trend_pos and mom_pos and vol_pos and rsi_pos
        bearish_strict = trend_neg and mom_neg and vol_neg and rsi_neg

        bullish_relaxed = (bullish_strict or ((trend_pos or nf["support"]) and mom_pos and rsi_pos and vol_pos))
        bearish_relaxed = (bearish_strict or ((trend_neg or nf["breakdown"]) and mom_neg and rsi_neg and vol_neg))

        if bank_used and bf is not None:
            bank_trend_pos = bf["trend_sig"] > 0
            bank_trend_neg = bf["trend_sig"] < 0
            # Only veto if bank is meaningfully negative (score < -0.25), not just mildly soft
            if bullish_relaxed and bank_trend_neg and bf["momentum_sig"] < 0 and bank_score < -0.25:
                bullish_relaxed = False
            if bearish_relaxed and bank_trend_pos and bf["momentum_sig"] > 0 and bank_score > 0.25:
                bearish_relaxed = False

        if bullish_strict:
            bias = "Bullish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bearish_strict:
            bias = "Bearish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bullish_relaxed and not bearish_relaxed:
            bias = "Bullish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        elif bearish_relaxed and not bullish_relaxed:
            bias = "Bearish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        else:
            bias = "Sideways / no edge"
            conf = min(58, int(round(45 - abs(combined) * 18)))
            conf = max(conf, 25)

        sigma_pct = _safe_float(nifty_feat.get("sigma_pct", 0.0), 0.0)
        sigma_pct = max(sigma_pct, 0.10)
        if bias == "Bullish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"+{low_mag:.2f}% to +{high_mag:.2f}%"
        elif bias == "Bearish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"-{low_mag:.2f}% to -{high_mag:.2f}%"
        else:
            side = sigma_pct * 0.55 * (0.80 + conf / 220.0)
            expected_range = f"±{side:.2f}% to ±{(side * 1.2):.2f}%"

        breakdown = []
        trend_txt = "Bullish trend (Close > EMA20 > EMA50)" if nf["trend_sig"] > 0 else (
            "Bearish trend (Close < EMA20 < EMA50)" if nf["trend_sig"] < 0 else "Neutral trend (EMA alignment mixed)"
        )
        breakdown.append(f"Trend: {trend_txt}.")
        breakdown.append(f"Momentum: 5D return {nf['ret5d']:+.2f}%.")
        breakdown.append(f"RSI(14): {nf['rsi14']:.1f}.")
        if nf["vol_ratio"] is None:
            breakdown.append("Volume: not available for index (confidence kept conservative).")
        else:
            breakdown.append(f"Volume: Vol/20D avg {nf['vol_ratio']:.2f}x.")
        breakdown.append(f"20D return: {nf['ret20d']:+.2f}%.")

        if bank_used and bf is not None:
            bn_txt = "BankNifty confirms bullishness" if bf["trend_sig"] > 0 and bf["momentum_sig"] > 0 else (
                "BankNifty confirms bearishness" if bf["trend_sig"] < 0 and bf["momentum_sig"] < 0 else
                "BankNifty mixed / neutral."
            )
            breakdown.append(bn_txt)

        return {
            "bias": bias,
            "confidence": conf,
            "expected_range": expected_range,
            "breakdown": breakdown[:6],
            "regime": _classify_regime_nifty(nifty_feat),
        }
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "\u00b10.30% to \u00b10.70%",
            "breakdown": ["Interpretation failed (fallback)."],
            "regime": "Ranging",
        }


# ── FRESH ISOLATED MARKET BIAS ENGINE (Task 4.3 — UI Version) ─────────
@st.cache_data(ttl=600, show_spinner=False)
def compute_market_bias_ui(_tt_cache_key: str = "live") -> dict:
    """
    Independent function for the 'Market Bias Engine' UI button.
    Does not touch scanner/strategy mode logic.
    _tt_cache_key is passed as a cache-buster — callers pass the active
    TT date string (or "live") so Streamlit re-runs when the date changes.
    """
    try:
        # 1. Fetch data
        df = yf.download("^NSEI", period="4mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return {
                "bias": "Sideways / No Edge",
                "confidence": 50,
                "expected_move": "±0.5% (fallback)",
                "reasons": ["Insufficient data from yfinance for Nifty (^NSEI)."]
            }
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # ── Time-Travel: truncate to historical cutoff ─────────────────
        if _TIME_TRAVEL_OK and hasattr(_tt, "apply_time_travel_cutoff"):
            df = _tt.apply_time_travel_cutoff(df)
            if df is None or df.empty or len(df) < 50:
                return {
                    "bias": "Sideways / No Edge",
                    "confidence": 50,
                    "expected_move": "±0.5% (fallback)",
                    "reasons": ["Insufficient historical data for selected Time Travel date."]
                }

        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()

        # 2. Indicators
        e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        # BUG FIX: Use EWM-based RSI (Wilder's smoothing) consistent with the
        # main rsi() function. The old SMA rolling(14).mean() produced NaN for
        # the first 13 rows and gave different values from every other RSI calc.
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_val = float(100 - (100 / (1 + rs.iloc[-1]))) if np.isfinite(rs.iloc[-1]) else 50.0

        c_last = float(close.iloc[-1])
        ret5d  = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
        ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else 0.0
        
        avg_vol = vol.iloc[-21:-1].mean() if len(vol) >= 21 else 1.0
        vol_r   = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        # 3. Bias Logic (Strictly Isolated)
        bullish = (c_last > e20 > e50) and (rsi_val > 52) and (ret5d > 0.1)
        bearish = (c_last < e20 < e50) and (rsi_val < 48) and (ret5d < -0.1)
        
        score = 50
        if bullish:
            score += 15
            if ret20d > 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bullish / Strong Bias" if score > 70 else "Bullish"
        elif bearish:
            score += 15
            if ret20d < 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bearish / Negative Bias" if score > 70 else "Bearish"
        else:
            bias = "Sideways / No Clear Edge"
        
        score = min(95, max(5, score))

        # 4. Volatility based move range
        daily_returns = close.pct_change().tail(20)
        volatility = daily_returns.std() * 100.0 if not daily_returns.empty else 0.5
        move_pct = round(volatility * 0.8, 2)
        expected_move = f"±{move_pct}% to ±{round(move_pct*1.5, 2)}%"

        # 5. Reason Breakdown
        reasons = []
        trend_txt = "Bullish stack" if c_last > e20 > e50 else ("Bearish stack" if c_last < e20 < e50 else "Neutral trend")
        reasons.append(f"Trend: {trend_txt} (Close={c_last:.0f}, EMA20={e20:.0f}).")
        reasons.append(f"RSI(14): {rsi_val:.1f} ({'Overbought' if rsi_val > 70 else ('Oversold' if rsi_val < 30 else 'Neutral')}).")
        reasons.append(f"Momentum: 5-day return {ret5d:+.2f}%.")
        reasons.append(f"Volume: Recent volume is {vol_r:.2f}x of 20-day average.")
        reasons.append(f"Structure: 20-day return is {ret20d:+.2f}%.")

        return {
            "bias": bias,
            "confidence": int(score),
            "expected_move": expected_move,
            "reasons": reasons
        }
    except Exception as e:
        return {
            "bias": "Sideways / Unknown",
            "confidence": 50,
            "expected_move": "±0.5% (error)",
            "reasons": [f"Market analysis error: {str(e)}"]
        }


# ─────────────────────────────────────────────────────────────────────
# STOCK ANALYSER  (Zero-API Refactored)
# ─────────────────────────────────────────────────────────────────────
def analyse(ticker, mode, retries=2):
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    _scan_diag.record_attempt(ticker_ns)
    try:
        df = get_df_for_ticker(ticker_ns)

        if df is None or df.empty:
            _scan_diag.record_failure(ticker_ns, "NO_DATA")
            return None
        try:
            if not _is_fresh_enough(df, strict=True):
                _scan_diag.record_failure(ticker_ns, "STALE")
                return None
        except Exception:
            pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "Close", "Volume"])
        if df.empty or len(df) < 30:
            _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
            return None

        # ── 🕰️ TIME TRAVEL: truncate to cutoff date (guaranteed no leakage) ──
        # This runs REGARDLESS of whether the ticker was in ALL_DATA or came
        # from a live yfinance fallback — the explicit slice here is the true
        # data-leakage guard. The monkey-patch in time_travel_engine only covers
        # ALL_DATA hits; this covers everything else.
        try:
            _tt_cut = _tt.get_reference_date()
            if _tt_cut is not None:
                _tt_mask = pd.to_datetime(df.index).date <= _tt_cut
                df = df.loc[_tt_mask]
                if df.empty or len(df) < 30:
                    _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
                    return None
        except Exception:
            pass  # fail-safe: continue with untruncated data rather than crash

        try:
            last_idx = df.index[-1]
            last_dt  = pd.to_datetime(last_idx).to_pydatetime()
        except Exception:
            _scan_diag.record_failure(ticker_ns, "STALE")
            return None
        if (_tt.get_reference_datetime() - last_dt).days > 7:
            _scan_diag.record_failure(ticker_ns, "STALE")
            return None

        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        open_p = df["Open"].dropna()
        if len(close) < 25:
            _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
            return None

        lc  = float(close.iloc[-1])
        lo  = float(open_p.iloc[-1])
        lv  = float(volume.iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])
        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ri  = rsi(close)

        if not (1 < lc <= 100000):
            _scan_diag.record_failure(ticker_ns, "BAD_PRICE")
            return None
        if lv <= 0:
            _scan_diag.record_failure(ticker_ns, "ZERO_VOLUME")
            return None
        if any(np.isnan(v) for v in (ri, e20, e50)):
            _scan_diag.record_failure(ticker_ns, "NAN_INDICATORS")
            return None

        # SPEED FIX — mktcap only fetched for modes that actually use it (1 & 2).
        # Modes 3-6 never reference mktcap_cr so skipping saves ~1000 API calls/scan.
        ok = False
        if mode == 1:
            mktcap_cr = get_mktcap_cr(ticker)
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            ok  = (lc > e20 and e20 > e50 and lv > 1.5 * avg_vol
                   and 52 <= ri <= 74 and lc >= 0.99 * h10 and lc > lo
                   and lc > 30 and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 2:
            mktcap_cr = get_mktcap_cr(ticker)
            h15 = float(close.iloc[-16:-1].max()) if len(close) >= 16 else float(close.max())
            ok  = (lc > 30 and lc > e20 and e20 > e50
                   and lv > 1.3 * avg_vol and 50 <= ri <= 72
                   and lc >= 0.96 * h15 and lc > lo
                   and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 3:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            ok  = (lc > e20 and lv > 1.1 * avg_vol
                   and 48 <= ri <= 74 and lc >= 0.90 * h20 and lc > 20)
        elif mode == 4:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            if len(close) < 21:
                _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
                return None
            base_20 = float(close.iloc[-21])
            if base_20 <= 0:
                _scan_diag.record_failure(ticker_ns, "BAD_PRICE")
                return None
            stock_ret_20d = (lc - base_20) / base_20
            nifty_ret_20d = get_nifty_20d_return()
            if nifty_ret_20d is None:
                nifty_ret_20d = 0.0  # fallback: compare vs flat market instead of blocking all results
            ok = (
                lc > e20 and e20 > e50 and
                lv > 1.3 * avg_vol and 52 <= ri <= 72 and
                lc >= 0.97 * h20 and stock_ret_20d > nifty_ret_20d and lc > lo
            )
        elif mode == 5:
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and lv > 1.1 * avg_vol_sma and
                lc >= 0.99 * h10 and 50 <= ri <= 65 and lc > lo and lc > 20
            )
        elif mode == 6:
            if len(close) < 2:
                _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
                return None
            prev_e20     = float(ema(close, 20).iloc[-2])
            h10          = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma  = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and e20 > prev_e20 and
                lv > 1.1 * avg_vol_sma and
                lc >= 0.97 * h10 and 50 <= ri <= 68 and lc > lo and lc > 40
            )

        if ok:
            h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
            dist_ema20    = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
            ret_5d  = (lc / float(close.iloc[-6]) - 1.0) * 100.0  if len(close) >= 6  else np.nan
            ret_20d = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else np.nan

            return {
                "Symbol":            ticker.replace(".NS", ""),
                "Price (₹)":         round(lc, 2),
                "Volume":            int(lv),
                "RSI":               round(ri, 2),
                "EMA 20":            round(e20, 2),
                "EMA 50":            round(e50, 2),
                "Vol / Avg":         round(lv / avg_vol, 2) if avg_vol > 0 else 0,
                "Mode":              {1:"🟢 Momentum",2:"🔵 Balanced",3:"🟡 Relaxed",
                                      4:"🟣 Institutional",5:"🟠 Intraday",6:"🔴 Swing"}[mode],
                "Δ vs 20D High (%)": round(dist_20d_high, 2),
                "Δ vs EMA20 (%)":    round(dist_ema20, 2),
                "5D Return (%)":     round(ret_5d, 2)  if not np.isnan(ret_5d)  else np.nan,
                "20D Return (%)":    round(ret_20d, 2) if not np.isnan(ret_20d) else np.nan,
            }
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# PARALLEL SCANNER  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def _start_stage_feedback(label: str):
    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        status = st.empty()
    with col_b:
        eta_box = st.empty()
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )
    eta_box.markdown(
        '<div class="status-line" style="justify-content:center">'
        'Elapsed <b style="color:#8ab4d8">0s</b>'
        ' &nbsp;·&nbsp; ETA <b style="color:#f0b429">Calibrating...</b></div>',
        unsafe_allow_html=True,
    )
    return progress_bar, status, eta_box, time.time()


def _format_scan_duration(seconds: float | None) -> str:
    try:
        if seconds is None:
            return "--"
        seconds = max(float(seconds), 0.0)
    except Exception:
        return "--"

    total_seconds = int(round(seconds))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _update_stage_feedback(
    progress_bar,
    status,
    eta_box,
    started_at: float,
    done: int,
    total: int,
    found: int,
    unit_label: str,
    found_label: str,
    ui_state: dict | None = None,
    phase_label: str | None = None,
    extra_html: str = "",
) -> None:
    pct = done / total if total > 0 else 0.0
    elapsed = max(time.time() - started_at, 0.001)
    inst_rate = done / elapsed if elapsed > 0 else 0.0

    if ui_state is not None:
        prev_rate = ui_state.get("rate")
        rate = inst_rate if prev_rate is None else (prev_rate * 0.82 + inst_rate * 0.18)
        ui_state["rate"] = rate
    else:
        rate = inst_rate

    min_samples = min(80, max(20, total // 40)) if total > 0 else 0
    eta_ready = done >= min_samples and elapsed >= 8 and rate > 0.05
    remaining = (total - done) / rate if eta_ready else None
    pct_text = f"{pct * 100:.1f}%"
    stage_text = phase_label or unit_label
    progress_bar.progress(min(pct, 1.0))
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;{stage_text} &nbsp;·&nbsp; {pct_text}'
        f' &nbsp;·&nbsp; {unit_label} <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
        f' &nbsp;·&nbsp; {found_label} <b style="color:#00d4a8">{found:,}</b>'
        f' &nbsp;·&nbsp; Speed <b style="color:#8ab4d8">{rate:.1f}/s</b>'
        f'{extra_html}</div>',
        unsafe_allow_html=True,
    )
    eta_box.markdown(
        f'<div class="status-line" style="justify-content:center">'
        f'Elapsed <b style="color:#8ab4d8">{_format_scan_duration(elapsed)}</b>'
        f' &nbsp;·&nbsp; ETA <b style="color:#f0b429">{_format_scan_duration(remaining) if remaining is not None else "Calibrating..."}</b></div>',
        unsafe_allow_html=True,
    )


def _finish_stage_feedback(
    progress_bar,
    status,
    eta_box,
    started_at: float,
    total: int,
    found: int,
    found_label: str,
) -> None:
    elapsed = max(time.time() - started_at, 0.001)
    avg_speed = total / elapsed if elapsed > 0 else 0.0
    progress_bar.progress(1.0)
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} stocks in'
        f' <b style="color:#f0b429">{elapsed:.1f}s</b>'
        f' &nbsp;·&nbsp; {found_label} <b style="color:#00d4a8">{found:,}</b>'
        f' &nbsp;·&nbsp; Avg speed <b style="color:#8ab4d8">{avg_speed:.1f}/s</b></div>',
        unsafe_allow_html=True,
    )
    eta_box.empty()


_SCAN_REASON_MEANINGS: dict[str, str] = {
    "NO_DATA": "preloaded/cache lookup returned no usable frame",
    "TOO_SHORT": "not enough usable history for the current scan logic",
    "STALE": "latest candle is older than the required market date for this scan",
    "BAD_PRICE": "closing price is outside the allowed scan range",
    "ZERO_VOLUME": "latest session volume is zero or negative",
    "NAN_INDICATORS": "EMA20 / EMA50 / RSI could not be computed cleanly",
    "SCAN_FILTER": "data was valid but the stock did not match the mode",
    "EXCEPTION": "unexpected runtime exception inside analyse()",
}


def _render_scan_diagnostics_panel() -> None:
    report = st.session_state.get("_scan_diag_report")
    if not isinstance(report, dict) or int(report.get("attempted", 0) or 0) <= 0:
        return

    attempted = int(report.get("attempted", 0) or 0)
    succeeded = int(report.get("succeeded", 0) or 0)
    failed_data = int(report.get("failed_data", 0) or 0)
    scan_filtered = int(report.get("scan_filtered", 0) or 0)
    reasons = report.get("reasons", {}) if isinstance(report.get("reasons"), dict) else {}
    failed_tickers = report.get("failed_tickers", [])
    scan_mode = st.session_state.get("_scan_diag_mode")
    scan_stamp = st.session_state.get("_scan_diag_scan_time", st.session_state.get("scan_time", "—"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#8ab4d8;letter-spacing:1.2px;'
        'text-transform:uppercase;margin-bottom:10px;">Scan Diagnostics</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Mode {scan_mode} diagnostics · {scan_stamp}"
        if scan_mode is not None
        else f"Diagnostics · {scan_stamp}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Attempted", f"{attempted:,}")
    c2.metric("Signals Found", f"{succeeded:,}")
    c3.metric("Data Failed", f"{failed_data:,}")
    c4.metric("Scan Filtered", f"{scan_filtered:,}")
    c5.metric("Data OK", f"{report.get('data_ok_pct', 0.0):.1f}%")

    with st.expander("Failure Breakdown", expanded=False):
        data_problem_reasons = {
            "NO_DATA",
            "TOO_SHORT",
            "STALE",
            "BAD_PRICE",
            "ZERO_VOLUME",
            "NAN_INDICATORS",
            "EXCEPTION",
        }
        rows = [
            {
                "Reason": reason,
                "Count": count,
                "Meaning": _SCAN_REASON_MEANINGS.get(reason, reason),
            }
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
            if reason in data_problem_reasons
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.success("No data-quality failures were recorded in the last scan.")

    if isinstance(failed_tickers, list) and failed_tickers:
        with st.expander("Failed Tickers", expanded=False):
            fail_df = pd.DataFrame(failed_tickers, columns=["Ticker", "Reason"])
            fail_df["Meaning"] = fail_df["Reason"].map(lambda reason: _SCAN_REASON_MEANINGS.get(reason, reason))
            st.dataframe(fail_df, width="stretch", hide_index=True)


def run_scan(tickers, mode, workers=20):
    results = []
    total   = len(tickers)
    done    = 0

    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a: status  = st.empty()
    with col_b: eta_box = st.empty()
    t0 = time.time()
    scan_feedback = {"rate": None}
    render_step = max(12, total // 120) if total else 12
    last_render_done = 0
    last_render_ts = 0.0

    status.markdown(
        '<div class="status-line"><span class="sdot sdot-green"></span>'
        '&nbsp;Stage 2 of 2 &nbsp;·&nbsp; Running strategy scan</div>',
        unsafe_allow_html=True,
    )
    eta_box.markdown(
        '<div class="status-line" style="justify-content:center">'
        'Elapsed <b style="color:#8ab4d8">0s</b>'
        ' &nbsp;·&nbsp; ETA <b style="color:#f0b429">Calibrating...</b></div>',
        unsafe_allow_html=True,
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyse, t, mode): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                results.append(r)
            now = time.time()
            should_render = (
                done == total
                or done == 1
                or (done - last_render_done) >= render_step
                or (now - last_render_ts) >= 0.25
            )
            if should_render:
                _update_stage_feedback(
                    progress_bar,
                    status,
                    eta_box,
                    t0,
                    done,
                    total,
                    len(results),
                    "Scanned",
                    "Found",
                    ui_state=scan_feedback,
                    phase_label="Stage 2 of 2",
                )
                last_render_done = done
                last_render_ts = now

    progress_bar.progress(1.0)
    elapsed_total = time.time() - t0
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} stocks in'
        f' <b style="color:#f0b429">{elapsed_total:.1f}s</b>'
        f' &nbsp;·&nbsp; <b style="color:#00d4a8">{len(results)}</b> found'
        f' &nbsp;·&nbsp; Avg speed <b style="color:#8ab4d8">{(total / elapsed_total) if elapsed_total > 0 else 0:.1f}/s</b></div>',
        unsafe_allow_html=True)
    eta_box.empty()
    return results, elapsed_total


# ═════════════════════════════════════════════════════════════════════
# ▼▼▼  NEW LAYER — SCORING / BACKTEST / ML  (added AFTER scan) ▼▼▼
# ═════════════════════════════════════════════════════════════════════

# ── mode-specific weight configs ─────────────────────────────────────
_MODE_WEIGHTS = {
    # (vol_bonus, breakout_bonus, ema_bonus, rsi_bonus, penalty_scale)
    1: dict(vol=1.4,  breakout=1.5, ema=1.0, rsi=1.0, pen=1.0),   # Momentum
    2: dict(vol=1.0,  breakout=1.0, ema=1.0, rsi=1.0, pen=1.0),   # Balanced
    3: dict(vol=0.8,  breakout=0.8, ema=0.8, rsi=0.8, pen=0.5),   # Relaxed
    4: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Institutional
    5: dict(vol=1.5,  breakout=1.2, ema=0.7, rsi=0.8, pen=0.9),   # Intraday
    6: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Swing
}


def _safe(v, default=0.0):
    """Return v if finite, else default."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def compute_score(row: dict, mode: int = 2) -> tuple[float, dict]:
    """
    Returns (score_0_100, breakdown_dict).
    breakdown_dict is used for the tooltip.
    """
    w   = _MODE_WEIGHTS.get(mode, _MODE_WEIGHTS[2])
    pts = {}

    ri       = _safe(row.get("RSI",            50))
    vol_r    = _safe(row.get("Vol / Avg",       1))
    d20h     = _safe(row.get("Δ vs 20D High (%)", -5))
    d_ema20  = _safe(row.get("Δ vs EMA20 (%)",   0))
    r5d      = _safe(row.get("5D Return (%)",    0))
    price    = _safe(row.get("Price (₹)",        0))
    e20      = _safe(row.get("EMA 20",           0))
    e50      = _safe(row.get("EMA 50",           0))

    # ── RSI zone ─────────────────────────────────────────────────────
    if 55 <= ri <= 65:
        pts["RSI 55-65"] = round(15 * w["rsi"])
    elif 65 < ri <= 70:
        pts["RSI 65-70"] = round(10 * w["rsi"])

    # ── Volume ratio ──────────────────────────────────────────────────
    if vol_r > 2.0:
        pts["Vol >2×"] = round(25 * w["vol"])
    elif vol_r > 1.5:
        pts["Vol >1.5×"] = round(20 * w["vol"])

    # ── Near 20D breakout ─────────────────────────────────────────────
    if -2.0 <= d20h <= 0.0:
        pts["Near 20D High"] = round(15 * w["breakout"])

    # ── Above EMA20 ───────────────────────────────────────────────────
    if price > e20 > 0:
        pts["Price > EMA20"] = round(10 * w["ema"])

    # ── EMA stack ─────────────────────────────────────────────────────
    if e20 > e50 > 0:
        pts["EMA20 > EMA50"] = round(10 * w["ema"])

    # ── 5D return zone ────────────────────────────────────────────────
    if 1.0 <= r5d <= 5.0:
        pts["5D Return 1-5%"] = round(10 * w["rsi"])

    # ── PENALTIES ─────────────────────────────────────────────────────
    if ri > 72:
        pts["RSI Overbought"] = round(-20 * w["pen"])
    if d_ema20 > 6.0:
        pts["Overextended EMA"] = round(-15 * w["pen"])
    if r5d > 8.0:
        pts["5D Return >8%"] = round(-10 * w["pen"])
    if vol_r < 1.2:
        pts["Low Volume"] = round(-15 * w["pen"])

    raw = sum(pts.values())
    score = float(np.clip(raw, 0, 100))
    return score, pts


# ── Backtest cache: ticker → float ───────────────────────────────────
_BT_CACHE: dict[str, float] = {}
_BT_LOCK = threading.Lock()


def _download_history(ticker_ns: str, period: str = "6mo") -> pd.DataFrame | None:
    """Download history; returns None on failure."""
    try:
        with _YF_SEM:
            df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12, threads=False,
            )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "Volume"])
        return df if len(df) >= 30 else None
    except Exception:
        return None


def compute_backtest_probability(row: dict, ticker: str, mode: int = 2) -> float:
    """
    Mode-aware backtest probability (Task 7).
    Each mode uses different RSI/vol tolerances + extra conditions that
    mirror what the live scanner filter actually checks.
    Cache key is ticker+mode so each strategy gets its own probability.
    Returns 50 (neutral) if fewer than 20 matching samples found.
    """
    # per-mode matching config: rsi_tol, vol_tol, ema_trend required,
    # near-high required, rolling-high window, near-high threshold
    _MBTCFG: dict[int, dict] = {
        1: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=10, hp=0.02),
        2: dict(rsi_tol=4,  vol_tol=0.25, ema_trend=True,  near_high=False, hw=15, hp=0.03),
        3: dict(rsi_tol=5,  vol_tol=0.30, ema_trend=False, near_high=False, hw=20, hp=0.05),
        4: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=20, hp=0.02),
        5: dict(rsi_tol=2,  vol_tol=0.15, ema_trend=True,  near_high=True,  hw=10, hp=0.01),
        6: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=False, hw=10, hp=0.02),
    }
    cfg = _MBTCFG.get(mode, _MBTCFG[2])
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    # BUG FIX: Include TT date in cache key so live and TT results are stored
    # separately. Without this, a live-mode cached result is returned for a TT
    # scan of the same ticker+mode, giving completely wrong backtest numbers.
    _bt_tt_key = str(_tt.get_reference_date()) if _TIME_TRAVEL_OK else "live"
    cache_key = f"{ticker_ns}|m{mode}|{_bt_tt_key}"

    with _BT_LOCK:
        if cache_key in _BT_CACHE:
            return _BT_CACHE[cache_key]

    result = 50.0
    try:
        # BUG FIX: Use get_df_for_ticker (TT-patched) instead of _download_history
        # which bypassed Time Travel and always fetched live data, corrupting TT backtests.
        df = get_df_for_ticker(ticker_ns)
        # Belt-and-suspenders: explicitly truncate to TT cutoff even if patch missed it.
        try:
            _bt_tt_cut = _tt.get_reference_date()
            if _bt_tt_cut is not None and df is not None and not df.empty:
                _bt_mask = pd.to_datetime(df.index).date <= _bt_tt_cut
                df = df.loc[_bt_mask]
        except Exception:
            pass
        if df is None or len(df) < 40:
            raise ValueError("insufficient data")

        close  = df["Close"].copy()
        volume = df["Volume"].copy()

        e20s = ema(close, 20)
        e50s = ema(close, 50)
        # vectorised RSI — no per-row loop, avoids pandas 2.x FutureWarning
        _d  = close.diff()
        _g  = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l  = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_series = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))

        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        roll_high = close.rolling(cfg["hw"], min_periods=max(1, cfg["hw"] // 2)).max().shift(1)

        target_rsi  = _safe(row.get("RSI",      50))
        target_volr = _safe(row.get("Vol / Avg", 1))

        mask = (
            rsi_series.notna() &
            (rsi_series >= target_rsi  - cfg["rsi_tol"]) &
            (rsi_series <= target_rsi  + cfg["rsi_tol"]) &
            (vol_ratio  >= target_volr * (1 - cfg["vol_tol"])) &
            (vol_ratio  <= target_volr * (1 + cfg["vol_tol"]))
        )
        if cfg["ema_trend"]:
            mask &= (e20s > e50s)
        if cfg["near_high"]:
            mask &= (roll_high.notna() & (close >= roll_high * (1 - cfg["hp"])))

        # Mode-specific extra matching conditions (Task 7)
        if mode == 4:
            mask &= (close.pct_change(20) > 0)          # positive 20D return (institutional)
        elif mode == 5:
            mask &= (vol_ratio > 1.5)                    # strong vol spike (intraday)
        elif mode == 6:
            mask &= (e20s > e20s.shift(1))               # rising EMA20 slope (swing)

        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]         # exclude last row

        if len(idx) < 20:
            raise ValueError(f"too few samples: {len(idx)}")

        close_vals  = close.values
        green_count = int(sum(close_vals[i + 1] > close_vals[i] for i in idx))
        result = round((green_count / len(idx)) * 100, 1)

    except Exception:
        result = 50.0

    with _BT_LOCK:
        _BT_CACHE[cache_key] = result
    return result


# ── ML model cache ────────────────────────────────────────────────────
_ML_MODEL: "LogisticRegression | None" = None
_ML_SCALER: "StandardScaler | None"   = None
_ML_LOCK = threading.Lock()
_ML_TICKERS_TRAINED: list[str] = []


def _build_ml_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Build training feature matrix for one ticker.
    All computations vectorised — no per-row Python loop (BUG 3 fix).
    """
    try:
        if len(close) < 30:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        ema_dist  = (close / e20s.replace(0, np.nan) - 1.0) * 100
        # vectorised RSI — no per-row loop
        _d        = close.diff()
        _g        = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l        = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_col   = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))
        ret5      = close.pct_change(5)  * 100
        ret20     = close.pct_change(20) * 100
        target    = (close.shift(-1) > close).astype(int)
        ema_trend = (e20s > e50s).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_col,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ret_5d":    ret5,
            "ret_20d":   ret20,
            "ema_trend": ema_trend,
            "target":    target,
        }).dropna()
        return df if len(df) >= 10 else None
    except Exception:
        return None


# 50-stock training universe (Task 4 — was only 15 stocks)
_ML_TRAIN_UNIVERSE: list[str] = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","AXISBANK.NS",
    "TATAMOTORS.NS","MARUTI.NS","LT.NS","NTPC.NS","ADANIPORTS.NS",
    "HINDALCO.NS","JSWSTEEL.NS","COALINDIA.NS","ONGC.NS","POWERGRID.NS",
    "BHARTIARTL.NS","TITAN.NS","NESTLEIND.NS","ULTRACEMCO.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","EICHERMOT.NS","M&M.NS","TATACONSUM.NS","BRITANNIA.NS",
    "TECHM.NS","INDUSINDBK.NS","KOTAKBANK.NS","ASIANPAINT.NS","GRASIM.NS",
    "DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","SUNPHARMA.NS","APOLLOHOSP.NS",
    "ITC.NS","BPCL.NS","IOC.NS","GAIL.NS","VEDL.NS",
    "ZOMATO.NS","NAUKRI.NS","IRCTC.NS","DMART.NS","TRENT.NS",
]


def train_model_once(tickers_sample: list[str] | None = None) -> bool:
    """
    Train LogisticRegression on up to 50 NSE stocks (1-year history).
    Task 4 upgrades:
      • 80/20 stratified train/test split
      • Prints test accuracy to stdout
      • class_weight='balanced' to reduce overfitting
      • C=0.5 (moderate regularisation)
      • 50-stock training universe (was 15)
    Model + scaler cached in module globals; re-entrant safe.
    """
    global _ML_MODEL, _ML_SCALER, _ML_TICKERS_TRAINED

    if not _SKLEARN_OK:
        return False

    with _ML_LOCK:
        if _ML_MODEL is not None:
            return True

    sample = (list(tickers_sample)[:50]
              if (tickers_sample and len(tickers_sample) >= 5)
              else _ML_TRAIN_UNIVERSE[:50])

    all_rows: list[pd.DataFrame] = []
    for t in sample:
        # BUG FIX: Use get_df_for_ticker (TT-patched) instead of
        # _download_history (bypasses TT), then apply cutoff explicitly.
        # Ensures ML model training in TT mode uses only historical data.
        df_h = get_df_for_ticker(t)
        if df_h is not None and _TIME_TRAVEL_OK and hasattr(_tt, "apply_time_travel_cutoff"):
            df_h = _tt.apply_time_travel_cutoff(df_h)
        if df_h is None:
            continue
        rows = _build_ml_features(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False

    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 100:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ret_5d", "ret_20d", "ema_trend"]
    X = data[FEAT].values
    y = data["target"].values

    try:
        # 80/20 stratified split (Task 4)
        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
        except Exception:
            split = int(len(data) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        scaler    = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_tr)
        X_te_sc   = scaler.transform(X_te)

        model = LogisticRegression(
            max_iter=500, C=0.5, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )
        model.fit(X_tr_sc, y_tr)

        acc = model.score(X_te_sc, y_te)
        print(
            f"[ML] Model trained on {len(data)} samples "
            f"({len(sample)} tickers) — test accuracy: {acc:.3f}"
        )

        with _ML_LOCK:
            _ML_MODEL            = model
            _ML_SCALER           = scaler
            _ML_TICKERS_TRAINED  = sample
        return True
    except Exception as exc:
        print(f"[ML] Training failed: {exc}")
        return False


def predict_ml_probability(row: dict, mode: int = 2) -> float:
    """
    Returns next-day-green probability (0-100) from the trained LR model.
    Task 7: base probability is adjusted by mode-specific signal weights
    so each strategy context influences the final ML score differently:
      Mode 1 (Momentum) → boosts high-vol + near-breakout
      Mode 3 (Relaxed)  → slight confidence haircut
      Mode 4 (Institutional) → rewards RSI + 20D return
      Mode 5 (Intraday)  → rewards vol spike, penalises high RSI
      Mode 6 (Swing)     → rewards EMA distance + 5D return
    Falls back to 50 if model not ready or features invalid.
    """
    if not _SKLEARN_OK:
        return 50.0

    with _ML_LOCK:
        model  = _ML_MODEL
        scaler = _ML_SCALER

    if model is None or scaler is None:
        return 50.0

    try:
        ri    = _safe(row.get("RSI",               50))
        vol_r = _safe(row.get("Vol / Avg",           1))
        de20  = _safe(row.get("Δ vs EMA20 (%)",      0))
        r5d   = _safe(row.get("5D Return (%)",        0))
        r20d  = _safe(row.get("20D Return (%)",       0))
        d20h  = _safe(row.get("Δ vs 20D High (%)", -10))

        feat    = np.array([[ri, vol_r, de20, r5d, r20d, 1.0]])
        feat_sc = scaler.transform(feat)
        base_p  = float(model.predict_proba(feat_sc)[0][1])

        # ── mode-specific probability adjustment (Task 7) ─────────────
        adj = 0.0
        if mode == 1:                              # Momentum
            if vol_r > 1.7:              adj += 0.05
            if -2.0 <= d20h <= 0.0:      adj += 0.03
        elif mode == 2:                            # Balanced — no adjustment
            pass
        elif mode == 3:                            # Relaxed — confidence haircut
            adj -= 0.03
        elif mode == 4:                            # Institutional
            if ri > 58 and r20d > 3.0:   adj += 0.04
            if vol_r > 1.5:              adj += 0.02
        elif mode == 5:                            # Intraday
            if vol_r > 1.5:              adj += 0.06
            if ri > 60:                  adj -= 0.03
        elif mode == 6:                            # Swing
            if r5d > 1.5:               adj += 0.04
            if de20 < 3.0:              adj += 0.02   # not overextended

        final_p = float(np.clip(base_p + adj, 0.01, 0.99))
        return round(final_p * 100, 1)
    except Exception:
        return 50.0


def apply_phase43_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4.3 — Dynamic intelligence (additive only).
    Adds:
      - "Dynamic Score"
      - "Confidence Level"
    No API calls; never filters/removes rows; safe fallbacks on missing columns.
    """
    try:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return df
        except Exception:
            return df

        def _sf(row: pd.Series, keys: list[str], default: float = 0.0) -> float:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        f = float(v)
                        return f if np.isfinite(f) else default
                except Exception:
                    continue
            return default

        def _ss(row: pd.Series, keys: list[str], default: str = "") -> str:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        return str(v).strip()
                except Exception:
                    continue
            return default

        out = df.copy()

        dyn_scores: list[float] = []
        conf_levels: list[str] = []

        for idx in out.index:
            try:
                row = out.loc[idx]

                score = _sf(row, ["Score"], 0.0)
                ml_p  = _sf(row, ["ML %", "ML"], 0.0)
                bt_p  = _sf(row, ["Backtest %", "Backtest"], 0.0)
                vol_r  = _sf(row, ["Vol / Avg", "Vol/Avg", "Volume"], 1.0)

                # Market bias is optional input; we read it safely but do not invent new effects.
                _mb_raw = _ss(row, ["Market Bias"], default="")
                _ = _mb_raw  # keep read-only; intentionally not used in formula

                if vol_r > 2.0:
                    weight_score, weight_ml, weight_bt = 0.6, 0.2, 0.2
                elif 1.2 <= vol_r <= 2.0:
                    weight_score, weight_ml, weight_bt = 0.5, 0.25, 0.25
                else:
                    weight_score, weight_ml, weight_bt = 0.3, 0.3, 0.4

                dynamic_score = score * weight_score + ml_p * weight_ml + bt_p * weight_bt
                dynamic_score = float(np.clip(dynamic_score, 0.0, 100.0))

                avg_prob = (bt_p + ml_p) / 2.0
                if avg_prob > 60.0:
                    conf = "HIGH"
                elif avg_prob > 52.0:
                    conf = "MEDIUM"
                else:
                    conf = "LOW"

            except Exception:
                dynamic_score = 0.0
                conf = "LOW"

            dyn_scores.append(round(dynamic_score, 2))
            conf_levels.append(conf)

        out["Dynamic Score"] = dyn_scores
        out["Confidence Level"] = conf_levels
        return out
    except Exception:
        return df


def apply_phase44_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4.4 — Feedback tracking (additive only).
    Adds:
      - "Next Day Return (%)"
      - "Signal Outcome"
      - "System Accuracy"
      - "Weight Suggestion" (optional, based on System Accuracy)
    No API calls; safe fallbacks on missing outcome/price columns.
    """
    try:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return df
        except Exception:
            return df

        def _sf(row: pd.Series, keys: list[str], default: float | None = None) -> float | None:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        f = float(v)
                        if np.isfinite(f):
                            return f
                except Exception:
                    continue
            return default

        def _ss(row: pd.Series, keys: list[str], default: str = "") -> str:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        return str(v).strip()
                except Exception:
                    continue
            return default

        out = df.copy()

        next_returns: list[float] = []
        outcomes: list[str] = []
        accuracies: list[float] = []
        weight_suggestions: list[str] = []

        running_win = 0
        running_total = 0

        for idx in out.index:
            try:
                row = out.loc[idx]

                current_close = _sf(row, ["Price (₹)", "Close", "Current Close", "Price"], default=None)
                next_close = _sf(row, ["Next Close", "Next Close (₹)", "Next Close (Rs)"], default=None)

                next_ret = np.nan
                if current_close is not None and next_close is not None:
                    if current_close != 0:
                        next_ret = float(((next_close - current_close) / current_close) * 100.0)

                final_signal = _ss(row, ["Final Signal"], default="") or "AVOID"

                if final_signal in ["STRONG BUY", "BUY"]:
                    if pd.notna(next_ret):
                        if next_ret > 0:
                            outcome = "WIN"
                        elif next_ret < 0:
                            outcome = "LOSS"
                        else:
                            outcome = "NEUTRAL"
                    else:
                        outcome = "NEUTRAL"
                else:
                    outcome = "NEUTRAL"

                # Running accuracy (only count rows where we have an outcome)
                if final_signal in ["STRONG BUY", "BUY"] and pd.notna(next_ret):
                    running_total += 1
                    if outcome == "WIN":
                        running_win += 1

                if running_total > 0:
                    acc = (running_win / running_total) * 100.0
                else:
                    acc = 50.0

                if acc < 50.0:
                    w_s = "Reduce ML weight"
                elif acc > 65.0:
                    w_s = "Increase Score weight"
                else:
                    w_s = "Balanced"

            except Exception:
                next_ret = np.nan
                outcome = "NEUTRAL"
                acc = 50.0
                w_s = "Balanced"

            next_returns.append(next_ret)
            outcomes.append(outcome)
            accuracies.append(round(float(acc), 2))
            weight_suggestions.append(w_s)

        out["Next Day Return (%)"] = next_returns
        out["Signal Outcome"] = outcomes
        out["System Accuracy"] = accuracies
        out["Weight Suggestion"] = weight_suggestions
        return out
    except Exception:
        return df


def compute_next_day_signal(row: dict, df: pd.DataFrame | None) -> str:
    """Compute short-term confirmational signal using last 10 days geometry."""
    if df is None or len(df) < 10:
        return "❌ No Data"
    try:
        last_10 = df.tail(10)
        closes = last_10["Close"].values
        vols = last_10["Volume"].values
        highs = last_10["High"].values if "High" in last_10.columns else closes

        if len(closes) < 10:
            return "❌ No Data"

        last_3 = closes[-3:]
        vol_last = vols[-1]
        vol_avg10 = vols.mean()
        high_10 = highs.max()
        close_today = closes[-1]

        momentum = (last_3[0] < last_3[1] < last_3[2])
        vol_spike = vol_last > 1.3 * vol_avg10
        near_breakout = close_today >= high_10 * 0.98
        overextended = _safe(row.get("Δ vs EMA20 (%)", 0)) > 7.0

        if momentum and vol_spike and near_breakout and not overextended:
            return "🔥 Strong Green"
        elif momentum and near_breakout:
            return "🟢 Possible Up"
        elif overextended:
            return "⚠️ Risky (Late Entry)"
        else:
            return "❌ Weak Setup"
    except Exception:
        return "❌ Error"


def check_bull_trap(row: dict) -> str:
    """Return warning string or empty string."""
    ri    = _safe(row.get("RSI",         50))
    vol_r = _safe(row.get("Vol / Avg",    1))
    de20  = _safe(row.get("Δ vs EMA20 (%)", 0))

    traps = []
    if ri > 72:
        traps.append("RSI overbought")
    if vol_r < 1.0:
        traps.append("vol declining")
    if de20 > 6.5:
        traps.append("far from EMA20")

    return "⚠️ Bull Trap" if len(traps) >= 2 else ""


def _trap_check_label(row: pd.Series | dict) -> str:
    """Collapse all trap layers into one user-facing status label."""
    try:
        trap_risk = str(row.get("Trap Risk", "") or "").strip().upper()
        advanced_trap = str(row.get("Advanced Trap", "") or "").strip().upper()
        legacy_trap = str(row.get("Trap", "") or "").strip().upper()
        final_signal = str(
            row.get("Adjusted Signal", row.get("Final Signal", "")) or ""
        ).strip().upper()
    except Exception:
        return "✅ Clean"

    if (
        trap_risk == "HIGH"
        or "BULL TRAP" in legacy_trap
        or final_signal == "TRAP"
        or advanced_trap in {"FAKE BREAKOUT", "EXHAUSTION"}
    ):
        return "⚠️ Trap"

    if trap_risk == "MEDIUM" or advanced_trap == "WEAK VOLUME":
        return "🟡 Caution"

    return "✅ Clean"


def enhance_results(results: list[dict], mode: int) -> pd.DataFrame:
    """
    Given raw scan results, attach Score / Backtest% / ML% / FinalRank.
    Uses the central ALL_DATA store for zero-API backtest computation.
    Returns a DataFrame sorted by FinalScore DESC.
    """
    if not results:
        return pd.DataFrame()

    _eng_score, _eng_bt, _eng_ml, _eng_trap = get_engine_functions(mode)
    max_workers = 10
    # NOTE: tickers list removed — was computed but never used anywhere.
    # Data is already preloaded before run_scan via preload_all().

    # ── Step 1: score all rows (fast, no I/O) ─────────────────────────
    pre_rows: list[dict] = []
    for r in results:
        sym = r.get("Ticker") or r.get("Symbol") or ""
        try:
            score, breakdown = _eng_score(r)
        except Exception:
            score, breakdown = 0.0, {}
        score = _safe(score, 0.0)
        pre_rows.append({
            "row":       r,
            "sym":       sym,
            "score":     round(score, 2),
            "breakdown": breakdown if isinstance(breakdown, dict) else {},
        })

    # ── Step 2: backtest top 50 only (zero-API via ALL_DATA) ──────────
    top_bt = {
        id(x["row"])
        for x in sorted(pre_rows, key=lambda x: x["score"], reverse=True)[:50]
    }

    def _process_enriched(pr: dict) -> dict:
        r         = pr["row"]
        sym       = pr["sym"]
        score     = pr["score"]
        breakdown = pr["breakdown"]

        # Backtest: zero-API — reads from ALL_DATA via backtest_with_preloaded
        if id(r) in top_bt:
            try:
                bt_prob = float(backtest_with_preloaded(mode, r, sym))
            except Exception:
                bt_prob = 50.0
        else:
            bt_prob = 50.0

        try:
            ml_prob = float(_eng_ml(r))
        except Exception:
            ml_prob = 50.0
        try:
            trap = _eng_trap(r)
        except Exception:
            trap = ""

        try:
            df_for_signal = get_df_for_ticker(sym)
            # TT guard: truncate signal df to cutoff so next-day signal
            # is computed on historical data only
            try:
                _tt_sig_cut = _tt.get_reference_date()
                if _tt_sig_cut is not None and df_for_signal is not None:
                    _sig_mask = pd.to_datetime(df_for_signal.index).date <= _tt_sig_cut
                    df_for_signal = df_for_signal.loc[_sig_mask]
                    if df_for_signal.empty:
                        df_for_signal = None
            except Exception:
                pass
            nd_signal = compute_next_day_signal(r, df_for_signal)
        except Exception:
            nd_signal = "❌ Error"

        bt_prob = _safe(bt_prob, 50.0)
        ml_prob = _safe(ml_prob, 50.0)
        final   = round(0.5 * score + 0.3 * bt_prob + 0.2 * ml_prob, 2)
        return {
            **r,
            "Score":       score,
            "_breakdown":  breakdown,
            "Backtest %":  round(bt_prob, 2),
            "ML %":        round(ml_prob, 2),
            "Final Score": final,
            "Trap":        trap,
            "Next-Day Signal": nd_signal,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_enriched, pr): pr for pr in pre_rows}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                pr    = futs[fut]
                r     = pr["row"]
                score = _safe(pr.get("score", 0.0), 0.0)
                rows.append({
                    **r,
                    "Score":       round(score, 2),
                    "_breakdown":  pr.get("breakdown", {}),
                    "Backtest %":  50.0,
                    "ML %":        50.0,
                    "Final Score": round(0.5 * score + 0.3 * 50.0 + 0.2 * 50.0, 2),
                    "Trap":        "",
                    "Next-Day Signal": "❌ Error",
                })

    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def _score_color(v: float) -> str:
    if v > 75:   return "#00d4a8"
    if v > 60:   return "#0094ff"
    if v > 40:   return "#f0b429"
    return "#ff4d6d"


def _score_label(v: float) -> str:
    if v > 75:   return "score-green"
    if v > 60:   return "score-blue"
    if v > 40:   return "score-yellow"
    return "score-red"


def render_top_picks(df: pd.DataFrame, n: int = 5) -> None:
    """Render the Top N pick cards in a horizontal strip."""
    st.markdown('<div class="section-lbl">🏅 Top Picks</div>', unsafe_allow_html=True)
    cols = st.columns(min(n, len(df)))
    for i, (col, (_, row)) in enumerate(zip(cols, df.head(n).iterrows())):
        sc   = row.get("Score",      0)
        bt   = row.get("Backtest %", 50)
        ml   = row.get("ML %",       50)
        fin  = row.get("Final Score", 0)
        trap = row.get("Trap",        "")
        sym  = row.get("Symbol",     "—")
        nd_sig= row.get("Next-Day Signal", "❌ No Data")
        c    = _score_color(sc)
        bd   = row.get("_breakdown", {})
        bd_html = "".join(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span>{k}</span>'
            f'<span style="color:{"#00d4a8" if v>=0 else "#ff4d6d"}">{v:+d}</span></div>'
            for k, v in bd.items()
        )
        with col:
            st.markdown(
                f'<div class="pick-card">'
                f'<div class="pick-rank">#{i+1}</div>'
                f'<div class="pick-sym">{sym}</div>'
                f'<div class="pick-score">'
                f'Score <span style="color:{c};font-weight:700">{sc:.0f}</span> &nbsp;|&nbsp; '
                f'BT <span style="color:#0094ff">{bt:.0f}%</span> &nbsp;|&nbsp; '
                f'ML <span style="color:#b08cff">{ml:.0f}%</span><br>'
                f'<b style="color:{c}">Final {fin:.1f}</b>'
                f'<br><span style="font-size:12px;color:#ccd9e8;font-weight:bold;">{nd_sig}</span>'
                f'{"&nbsp;&nbsp;<span class=trap-badge>" + trap + "</span>" if trap else ""}'
                f'</div>'
                f'<details style="margin-top:8px">'
                f'<summary style="font-size:11px;color:#4a6480;cursor:pointer">Score breakdown ▾</summary>'
                f'<div class="breakdown-box" style="margin-top:6px">{bd_html}</div>'
                f'</details>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR  (unchanged logic, unchanged options)
# ═════════════════════════════════════════════════════════════════════
mode_names  = {1:"Momentum", 2:"Balanced", 3:"Relaxed", 4:"Institutional", 5:"Intraday", 6:"Swing"}
mode_colors = {1:"#00d4a8",  2:"#0094ff",  3:"#f0b429", 4:"#b08cff", 5:"#00d4a8", 6:"#ff4d6d"}
pill_cls    = {1:"pill-m1",  2:"pill-m2",  3:"pill-m3", 4:"pill-m3", 5:"pill-m5", 6:"pill-m6"}
ui_mode_meta = {
    3: {"display_num": 1, "display_name": "Relaxed"},
    6: {"display_num": 2, "display_name": "Swing"},
    5: {"display_num": 3, "display_name": "Intraday"},
}

_SIDEBAR_PANEL_KEYS = (
    "show_sector_screener",
    "battle_show_panel",
    "aura_show_panel",
    "tomorrow_picks_show_panel",
    "pred_chart_show_panel",
    "csv_next_day_show_panel",
    "live_pulse_show_panel",
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    try:
        value = hex_color.lstrip("#")
        if len(value) != 6:
            raise ValueError("expected 6-digit hex")
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(0,212,168,{alpha})"


def _activate_sidebar_panel(active_key: str | None = None) -> None:
    for key in _SIDEBAR_PANEL_KEYS:
        st.session_state[key] = (key == active_key)
    st.rerun()


def _close_tomorrow_picks_panel() -> None:
    st.session_state["tomorrow_picks_show_panel"] = False

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:20px;'
        'color:#00d4a8;letter-spacing:-0.5px;padding:4px 0 16px 0;">'
        '<span class="live-dot"></span>NSE SENTINEL</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Strategy Mode</div>', unsafe_allow_html=True)

    # ── FIX 8 — Mode hint from cached market regime ───────────────────
    try:
        _hint_bias = st.session_state.get("market_bias_result") or {}
        _hint_regime = str(_hint_bias.get("regime", "")).strip()
        _REGIME_HINTS = {
            "Trending Up": "💡 Swing or Relaxed recommended",
            "Ranging":     "💡 Relaxed or Swing recommended",
            "High Vol":    "💡 Intraday recommended — tight stops",
            "Bearish":     "💡 Caution — all modes show elevated risk",
        }
        _hint_text = _REGIME_HINTS.get(_hint_regime, "")
        if _hint_text:
            st.info(_hint_text)
    except Exception:
        pass

    mode_map = {
        "\U0001F7E1  Mode 1 - Relaxed (Wide Scan)": 3,
        "\U0001F534  Mode 2 - Swing":               6,
        "\U0001F7E2  Mode 3 - Intraday":            5,
    }
    if st.session_state.get("strategy_mode") not in mode_map:
        st.session_state["strategy_mode"] = next(iter(mode_map))
    mode_label = st.selectbox(
        "Strategy mode",
        list(mode_map.keys()),
        label_visibility="collapsed",
        key="strategy_mode",
    )
    mode = mode_map[mode_label]
    mode_display = ui_mode_meta.get(mode, {"display_num": mode, "display_name": mode_names.get(mode, "Mode")})

    filter_data = {
        3: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.3× avg"),
            ("RSI","50 – 72"),("Price Floor","₹50"),
            ("20D High","Within 5%"),("Use Case","Wide Scan")],
        5: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.1× avg"),
            ("RSI","52 – 60"),("Price Floor","₹20"),
            ("10D High","Break above"),("Use Case","Tomorrow Push")],
        6: [("EMA Trend","Close > EMA20 > EMA50"),("EMA20 Slope","Rising"),
            ("Volume","> 1.3× avg & > prev"),("RSI","53 – 59"),
            ("Price Floor","₹40"),("10D High","Break above")],
    }
    colors = {1:"#00d4a8",2:"#0094ff",3:"#f0b429",4:"#b08cff",5:"#ff8c00",6:"#ff4d6d"}
    mc = colors[mode]

    params_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480">{k}</span>'
        f'<span style="font-size:11px;color:{mc};font-weight:700">{v}</span></div>'
        for k, v in filter_data[mode]
    ])
    st.markdown(
        f'<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;'
        f'padding:12px 14px;margin-bottom:16px;">{params_html}</div>',
        unsafe_allow_html=True)

    workers = 12

    st.markdown("<br>", unsafe_allow_html=True)
    scan_clicked = False
    sector_screener_clicked = st.button("🔭 Sector Screener Dashboard", key="sector_screener_dashboard_btn")
    battle_compare_clicked = st.button("⚔️ Compare Stocks", key="battle_compare_btn")
    aura_clicked = st.button("🔮 Stock Aura", key="stock_aura_btn")
    tomorrow_picks_clicked = st.button("📈 Tomorrow's Picks", key="tomorrow_picks_btn")

    pred_chart_clicked = st.button("📊 Prediction Chart Tomorrow", key="pred_chart_btn")

    if sector_screener_clicked:
        _activate_sidebar_panel("show_sector_screener")
    if battle_compare_clicked:
        _activate_sidebar_panel("battle_show_panel")
    if aura_clicked:
        _activate_sidebar_panel("aura_show_panel")
    if tomorrow_picks_clicked:
        _activate_sidebar_panel("tomorrow_picks_show_panel")
    if pred_chart_clicked:
        _activate_sidebar_panel("pred_chart_show_panel")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 🕰️ Time Travel Mode ───────────────────────────────────────
    st.markdown('<div class="section-lbl">🕰️ Time Travel Mode</div>', unsafe_allow_html=True)
    _tt_toggle = st.toggle(
        "Simulate a past market date",
        value=st.session_state.get("tt_toggle_val", False),
        key="tt_toggle",
    )
    st.session_state["tt_toggle_val"] = _tt_toggle

    if _tt_toggle:
        _tt_min     = datetime(2023, 1, 1).date()
        _tt_max     = (datetime.now() - timedelta(days=1)).date()
        _tt_default = st.session_state.get("tt_date_val", _tt_max)
        _tt_selected = st.date_input(
            "Market date to simulate",
            value=_tt_default,
            min_value=_tt_min,
            max_value=_tt_max,
            key="tt_date_picker",
            label_visibility="collapsed",
        )
        if _tt_selected is None:
            _tt_selected = _tt_max
        st.session_state["tt_date_val"] = _tt_selected
        st.markdown(
            f'<div style="background:#1a0a00;border:1px solid #f0b429;border-radius:8px;'
            f'padding:8px 12px;font-size:11px;color:#f0b429;margin-top:4px;line-height:1.6;">'
            f'🕰️ <b>SIMULATING</b><br>'
            f'{_tt_selected.strftime("%d %b %Y")} (Post-Market Close)<br>'
            f'<span style="color:#4a6480;">All scans use data up to this date only</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state["tt_date_val"] = None
        st.session_state["tt_date_picker"] = None
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;">Live mode — using current market data</div>',
            unsafe_allow_html=True,
        )
    _aura_tt_date = st.session_state.get("tt_date_val")
    st.session_state["aura_tt_date"] = (
        _aura_tt_date if (_aura_tt_date is not None and _TIME_TRAVEL_OK) else None
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">📡 Data Session</div>', unsafe_allow_html=True)
    if _tt_toggle:
        st.markdown(
            '<div style="font-size:11px;color:#f0b429;line-height:1.7;">'
            'Time Travel is active, so the normal live/close routing is paused until simulation is turned off.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        _session_plan = get_scan_data_plan()
        _session_window = str(_session_plan.get("window", "") or "").upper()
        _session_date = _session_plan.get("expected_date")
        _session_date_text = (
            _session_date.strftime("%d %b %Y")
            if hasattr(_session_date, "strftime")
            else str(_session_date)
        )
        _session_source = html.escape(str(_session_plan.get("source_label", "Auto")).upper())
        _session_summary = html.escape(str(_session_plan.get("summary", "")).strip())
        _live_window_label = html.escape(str(_session_plan.get("live_window_label", "")).strip())
        _has_snapshot = bool(_session_plan.get("snapshot_exists", False))
        _snapshot_state = "READY" if _has_snapshot else "PENDING"
        _snapshot_color = "#00d4a8" if _has_snapshot else "#f0b429"
        _window_color = {
            "LIVE": "#00d4a8",
            "CLOSED": "#4da3ff",
            "PRE_MARKET": "#f0b429",
            "WEEKEND": "#f0b429",
        }.get(_session_window, "#4da3ff")
        _snapshot_meta = read_snapshot_metadata(_session_date) if _has_snapshot else {}
        _captured_raw = str(_snapshot_meta.get("captured_at", "") or "").strip()
        _captured_text = ""
        if _captured_raw:
            _captured_text = (
                _captured_raw.replace("T", " ")
                .replace("+05:30", " IST")
                .replace("+0530", " IST")
            )
        _captured_row = ""
        if _captured_text:
            _captured_row = (
                '<div style="display:flex;justify-content:space-between;gap:12px;padding-top:8px;'
                'border-top:1px solid #1a2840;margin-top:8px;">'
                '<span style="font-size:10px;color:#4a6480;">Captured</span>'
                f'<span style="font-size:10px;color:#ccd9e8;text-align:right;">{html.escape(_captured_text)}</span>'
                '</div>'
            )
        st.markdown(
            f'<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;'
            f'padding:12px 14px;margin-bottom:4px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">'
            f'<span style="font-size:10px;letter-spacing:1.2px;color:#4a6480;">SESSION</span>'
            f'<span style="font-size:10px;font-weight:700;color:{_window_color};">{_session_window}</span>'
            f'</div>'
            f'<div style="margin-top:8px;font-size:12px;font-weight:700;color:{_window_color};">{_session_source}</div>'
            f'<div style="display:flex;justify-content:space-between;gap:12px;padding-top:10px;">'
            f'<span style="font-size:10px;color:#4a6480;">Live Window</span>'
            f'<span style="font-size:10px;color:#ccd9e8;text-align:right;">{_live_window_label}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;gap:12px;padding-top:8px;">'
            f'<span style="font-size:10px;color:#4a6480;">Scan Date</span>'
            f'<span style="font-size:10px;color:#ccd9e8;text-align:right;">{html.escape(_session_date_text)}</span>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;gap:12px;padding-top:8px;">'
            f'<span style="font-size:10px;color:#4a6480;">Snapshot</span>'
            f'<span style="font-size:10px;font-weight:700;color:{_snapshot_color};text-align:right;">{_snapshot_state}</span>'
            f'</div>'
            f'{_captured_row}'
            f'<div style="margin-top:10px;font-size:10px;line-height:1.7;color:#4a6480;">{_session_summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Data Management Panel ─────────────────────────────────────
    st.markdown('<div class="section-lbl">📦 Local Data Cache</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#4a6480;line-height:1.7;margin:-4px 0 10px 0;">'
        'Refresh the offline CSV cache, then run a focused scanner below.</div>',
        unsafe_allow_html=True,
    )
    if _DATA_DOWNLOADER_OK:
        if st.button("🔄 Refresh Local Data Cache", key="refresh_data_btn"):
            with st.spinner("Updating data..."):
                try:
                    _tickers_for_dl = fetch_nse_tickers()
                    update_all_data(_tickers_for_dl)
                    st.success("Data updated")
                except Exception as _e:
                    st.error(f"Data update failed: {_e}")
        st.markdown(
            '<div style="font-size:10px;color:#4a6480;letter-spacing:1.4px;'
            'text-transform:uppercase;margin:12px 0 8px 0;">Focused Scanners</div>',
            unsafe_allow_html=True,
        )
        csv_scan_clicked = st.button("⚡ Breakout Radar (CSV)", key="csv_next_day_btn")
        if csv_scan_clicked:
            _activate_sidebar_panel("csv_next_day_show_panel")
        live_pulse_clicked = st.button("📡 Live Breakout Pulse", key="live_pulse_btn")
        if live_pulse_clicked:
            st.session_state["live_pulse_autorun"] = True
            _activate_sidebar_panel("live_pulse_show_panel")
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;line-height:1.7;margin:6px 0 2px 0;">'
            '⚡ Cached CSV scan for pre-move setups<br>'
            '📡 Live scan for real-time momentum bursts</div>',
            unsafe_allow_html=True,
        )
        # Show cache status
        try:
            _status = data_status_summary(fetch_nse_tickers())
            st.markdown(
                f'<div style="font-size:11px;color:#4a6480;line-height:1.9;">'
                f'Fresh: <b style="color:#00d4a8">{_status.get("fresh", "?")}</b> &nbsp;'
                f'Stale: <b style="color:#f0b429">{_status.get("stale", "?")}</b> &nbsp;'
                f'Missing: <b style="color:#ff4d6d">{_status.get("missing", "?")}</b></div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass
    else:
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;">'
            'data_downloader.py not found — using live yfinance.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="font-size:10px;color:#4a6480;letter-spacing:1.4px;'
            'text-transform:uppercase;margin:12px 0 8px 0;">Focused Scanners</div>',
            unsafe_allow_html=True,
        )
        csv_scan_clicked = st.button("⚡ Breakout Radar (CSV)", key="csv_next_day_btn")
        if csv_scan_clicked:
            _activate_sidebar_panel("csv_next_day_show_panel")
        live_pulse_clicked = st.button("📡 Live Breakout Pulse", key="live_pulse_btn")
        if live_pulse_clicked:
            st.session_state["live_pulse_autorun"] = True
            _activate_sidebar_panel("live_pulse_show_panel")
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;line-height:1.7;margin:6px 0 2px 0;">'
            '⚡ Uses local CSV data when available<br>'
            '📡 Uses live market data directly</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#4a6480;line-height:1.7;">'
        'Data: Yahoo Finance (NSE)<br>Indicators: EMA · RSI · Volume<br>'
        'Universe: Current NSE listed equities<br><br>'
        '⚠️ Educational use only.<br>Not financial advice.</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FIX 1 — Startup background ML training (fires once on app boot)
# Non-blocking daemon thread — UI never waits for this.
# The scan handler still has its own fallback call if model isn't ready.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────
learning_status = _refresh_feedback_learning_system(force=False)
signal_weight_status = _refresh_signal_weight_status(force_update=False)

mc = mode_colors[mode]
_mc_soft = _hex_to_rgba(mc, 0.10)
_mc_border = _hex_to_rgba(mc, 0.28)
_show_sector_screener = st.session_state.get("show_sector_screener", False) or sector_screener_clicked
_show_live_pulse_panel = bool(st.session_state.get("live_pulse_show_panel", False)) or live_pulse_clicked
_show_tomorrow_picks_panel = bool(st.session_state.get("tomorrow_picks_show_panel", False))
_show_pred_chart_panel = bool(st.session_state.get("pred_chart_show_panel", False))
_show_home_scanner = not (
    _show_sector_screener
    or _show_live_pulse_panel
    or _show_tomorrow_picks_panel
    or _show_pred_chart_panel
)

st.markdown(
    f"""
    <style>
    :root {{
      --accent: {mc};
      --accent2: {mc};
      --accent3: {mc};
    }}
    h2, h3 {{
      color: var(--accent) !important;
    }}
    .section-lbl {{
      color: var(--accent) !important;
      border-bottom-color: {_mc_border} !important;
    }}
    .count-pill {{
      color: var(--accent) !important;
      border-color: var(--accent) !important;
      background: {_mc_soft} !important;
    }}
    .pick-rank {{
      color: var(--accent) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if _show_home_scanner:
    _dashboard_tt_date = _get_pending_time_travel_date()
    if _dashboard_tt_date is not None:
        try:
            _dashboard_ref_dt = datetime(
                _dashboard_tt_date.year,
                _dashboard_tt_date.month,
                _dashboard_tt_date.day,
                16,
                0,
                0,
            )
        except Exception:
            _dashboard_ref_dt = _get_dashboard_reference_datetime()
        try:
            _dashboard_status_label = f"🕰️ Time Travel Selected — {_dashboard_tt_date.isoformat()}"
        except Exception:
            _dashboard_status_label = f"🕰️ Time Travel Selected — {_dashboard_tt_date}"
    else:
        _dashboard_ref_dt = _get_dashboard_reference_datetime()
        _dashboard_status_label = get_data_status_label()
    _dashboard_tt_badge = (
        "  🕰️ TIME TRAVEL SELECTED"
        if _dashboard_tt_date is not None
        else ("  🕰️ TIME TRAVEL" if _tt.is_active() else "")
    )
    render_tomorrow_picks_ticker_strip()
    st.markdown(
        f'<div class="top-banner">'
        f'<div class="banner-logo"><span class="live-dot"></span>NSE SENTINEL</div>'
        f'<div style="margin-left:auto">'
        f'<span class="mode-pill {pill_cls[mode]}">MODE {mode_display["display_num"]} · {mode_display["display_name"].upper()}</span>'
        f'</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#4a6480;font-size:12px;font-family:\'Space Mono\',monospace;'
        f'margin-top:-8px;margin-bottom:20px;">'
        f'Automated multi-strategy scanner for NSE equities · '
        f'{_dashboard_ref_dt.strftime("%d %b %Y, %H:%M")}'
        f'{_dashboard_tt_badge}</p>',
        unsafe_allow_html=True)
    st.caption(_dashboard_status_label)

_ui_cached_tickers = st.session_state.get("_ui_all_tickers", [])
if isinstance(_ui_cached_tickers, list) and _ui_cached_tickers:
    all_tickers = list(_ui_cached_tickers)
else:
    with st.spinner("Loading NSE ticker list..."):
        all_tickers = fetch_nse_tickers()
    st.session_state["_ui_all_tickers"] = list(all_tickers)
n = len(all_tickers)

with st.sidebar:
    ticker_count = len(all_tickers)
    if ticker_count < _TICKER_GOOD_COUNT:
        st.warning(
            f"⚠️ Only {ticker_count} tickers loaded. "
            f"Click below to restore full list."
        )
        if st.button("🔄 Restore Full Ticker List", key="restore_full_ticker_list_btn"):
            st.cache_data.clear()
            st.session_state.pop("_ticker_master_list", None)
            st.session_state.pop("_ui_all_tickers", None)
            st.rerun()
    else:
        st.caption(f"✅ {ticker_count:,} tickers loaded")

with st.sidebar:
    try:
        _ls = learning_status if isinstance(learning_status, dict) else {}
        _fs = _ls.get("feedback_summary", {}) if isinstance(_ls.get("feedback_summary", {}), dict) else {}
        _logged = int(_fs.get("total_logged", 0) or 0)
        _validated = int(_fs.get("rows_with_outcome", 0) or 0)
        _validated_pct = round((_validated / max(_logged, 1)) * 100.0, 1) if _logged else 0.0
        _trained_samples = int(_ls.get("samples", 0) or 0)
        _train_acc = _ls.get("accuracy_pct")
        _train_acc_txt = f"{float(_train_acc):.1f}%" if _train_acc is not None else "n/a"
        _last_trained = str(_ls.get("last_trained", "") or "").replace("T", " ")
        _brain = _ls.get("brain_status", {}) if isinstance(_ls.get("brain_status", {}), dict) else {}
        _mode_models = _brain.get("mode_models", {}) if isinstance(_brain.get("mode_models", {}), dict) else {}
        _mode_by_mode = _mode_models.get("by_mode", {}) if isinstance(_mode_models.get("by_mode", {}), dict) else {}
        _best_mode = _mode_models.get("best_mode", {}) if isinstance(_mode_models.get("best_mode", {}), dict) else {}
        _worst_mode = _mode_models.get("worst_mode", {}) if isinstance(_mode_models.get("worst_mode", {}), dict) else {}
        _calibration = _brain.get("calibration", {}) if isinstance(_brain.get("calibration", {}), dict) else {}
        _regime_brain = _brain.get("regime", {}) if isinstance(_brain.get("regime", {}), dict) else {}
        _eval = _brain.get("evaluation", {}) if isinstance(_brain.get("evaluation", {}), dict) else {}
        _eval_report = _eval.get("report", None)
        _best_mode_txt = "n/a"
        if _best_mode:
            _best_mode_txt = f"Mode{int(_best_mode.get('mode', 0) or 0)} {float(_best_mode.get('accuracy_pct', 0.0) or 0.0):.1f}%"
        _worst_mode_txt = "n/a"
        if _worst_mode:
            _worst_mode_txt = f"Mode{int(_worst_mode.get('mode', 0) or 0)} {float(_worst_mode.get('accuracy_pct', 0.0) or 0.0):.1f}%"
        _calibration_txt = "n/a"
        if _calibration:
            _calibration_txt = (
                f"{float(_calibration.get('error_pct', 0.0) or 0.0):.1f}% "
                f"(x{float(_calibration.get('factor', 1.0) or 1.0):.2f})"
            )
        _badge = str(st.session_state.get("_validated_today_badge", "") or "").strip()
        if _badge:
            st.caption(_badge)

        if _ls.get("trained"):
            st.caption(f"ðŸ§  Model trained on {_trained_samples} samples | Accuracy: {_train_acc_txt}")
        else:
            st.caption(f"ðŸ§  {str(_ls.get('message', 'Model not trained yet.'))}")

        _sw = signal_weight_status if isinstance(signal_weight_status, dict) else {}
        _top_signal = str(_sw.get("top_signal", "") or "")
        _weak_signal = str(_sw.get("weakest_signal", "") or "")
        if _top_signal and _weak_signal:
            st.caption(
                f"ðŸ“Š Top signal: {_top_signal} ({float(_sw.get('top_weight', 0.0) or 0.0):.1f}%) | "
                f"Weakest: {_weak_signal} ({float(_sw.get('weakest_weight', 0.0) or 0.0):.1f}%)"
            )

        _recent_bull_acc_txt = "n/a"
        _regime_txt = "n/a"
        _wf_txt = "n/a"
        if _regime_brain:
            _regime_txt = (
                f"{str(_regime_brain.get('regime', 'n/a'))} "
                f"(conf: {float(_regime_brain.get('confidence', 0.0) or 0.0):.0f}%)"
            )
        try:
            _wf_score = float(getattr(_eval_report, "wf_stability_score", 0.0) or 0.0)
            if _wf_score > 0:
                _wf_txt = f"{_wf_score:.1f}"
        except Exception:
            pass
        try:
            from prediction_feedback_store import read_feedback_log

            _feedback_df = read_feedback_log()
            if isinstance(_feedback_df, pd.DataFrame) and not _feedback_df.empty and "correct" in _feedback_df.columns:
                _valid_df = _feedback_df[_feedback_df["correct"].isin(["True", "False"])].copy()
                if not _valid_df.empty:
                    _valid_df = _valid_df.sort_values("logged_at", ascending=False).head(20)
                    _bull_df = _valid_df[_valid_df["pred_bullish"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])]
                    if not _bull_df.empty:
                        _recent_bull_acc_txt = f"{(_bull_df['correct'].eq('True').mean() * 100.0):.0f}% correct"
        except Exception:
            pass

        try:
            from strategy_engines._engine_utils import ALL_DATA as _learning_all_data
            from sector_regime_engine import detect_regime as _detect_learning_regime

            _regime_state = _detect_learning_regime(_learning_all_data)
            _regime_txt = (
                f"{str(getattr(_regime_state, 'regime', 'n/a'))} "
                f"(conf: {float(getattr(_regime_state, 'confidence', 0.0) or 0.0):.0f}%)"
            )
        except Exception:
            pass

        try:
            from sector_evaluation_engine import compute_full_evaluation
            from sector_prediction_tracker import read_log as _read_sector_prediction_log

            _sector_log_df = _read_sector_prediction_log()
            if isinstance(_sector_log_df, pd.DataFrame) and not _sector_log_df.empty:
                _sector_eval = compute_full_evaluation(_sector_log_df)
                _wf_score = float(getattr(_sector_eval, "wf_stability_score", 0.0) or 0.0)
                if _wf_score > 0:
                    _wf_txt = f"{_wf_score:.1f}"
        except Exception:
            pass

        with st.expander("🧠 AI Learning Status", expanded=False):
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #1e3a5f;border-radius:12px;'
                f'padding:14px 14px 10px 14px;">'
                f'<div style="font-size:12px;color:#8ab4d8;line-height:1.8;">'
                f'Predictions logged: <span style="color:#ccd9e8;">{_logged}</span><br>'
                f'Validated: <span style="color:#ccd9e8;">{_validated} ({_validated_pct:.1f}%)</span><br>'
                f'Filled today: <span style="color:#00d4ff;">{int(_ls.get("validated_today", 0) or 0) + int(_ls.get("sector_validated_today", 0) or 0)}</span><br>'
                f'Meta accuracy: <span style="color:#00d4ff;">{_train_acc_txt}</span><br>'
                f'Best mode: <span style="color:#ccd9e8;">{html.escape(_best_mode_txt)}</span><br>'
                f'Worst mode: <span style="color:#ccd9e8;">{html.escape(_worst_mode_txt)}</span><br>'
                f'Calibration: <span style="color:#ccd9e8;">{html.escape(_calibration_txt)}</span><br>'
                f'Last trained: <span style="color:#ccd9e8;">{html.escape(_last_trained or "n/a")}</span><br>'
                f'Last learning cycle: <span style="color:#ccd9e8;">{html.escape(str(_brain.get("completed_at", "n/a") or "n/a").replace("T", " "))}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            _report = _sw.get("report") if isinstance(_sw, dict) else pd.DataFrame()
            if isinstance(_report, pd.DataFrame) and not _report.empty:
                st.markdown(
                    '<div style="font-size:11px;color:#8ab4d8;margin:10px 0 6px 0;">TOP SIGNALS (Dynamic Weights)</div>',
                    unsafe_allow_html=True,
                )
                for _, _sig_row in _report.head(4).iterrows():
                    _sig_name = str(_sig_row.get("Signal", "") or "")
                    _sig_weight = float(_sig_row.get("Dynamic Weight", 0.0) or 0.0)
                    st.markdown(
                        _weight_bar_html(_sig_name, _sig_weight, warn=_sig_weight <= 3.0),
                        unsafe_allow_html=True,
                    )

            st.markdown(
                f'<div style="font-size:11px;color:#8ab4d8;margin:10px 0 6px 0;">RECENT ACCURACY (Last 20)</div>'
                f'<div style="font-size:12px;color:#ccd9e8;line-height:1.8;">'
                f'Bullish calls: {_recent_bull_acc_txt}<br>'
                f'Regime: {_regime_txt}<br>'
                f'Walk-forward stability: {_wf_txt}'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

if _show_pred_chart_panel:
    render_prediction_chart_section(ticker_list=all_tickers)
    st.stop()

if _show_home_scanner:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📋 NSE Tickers Loaded", f"{n:,}")
    with c2: st.metric("🎯 Active Mode", f"M{mode_display['display_num']} · {mode_display['display_name']}")
    with c3: st.metric("⚡ Workers", workers)
    with c4:
        found_val  = len(st.session_state.get("results", [])) if "results" in st.session_state else None
        elapsed_v  = st.session_state.get("elapsed", None)
        if found_val is not None:
            st.metric("✅ Last Scan Found", found_val,
                      delta=f"{elapsed_v:.1f}s" if elapsed_v else None)
        else:
            st.metric("✅ Last Scan Found", "—")

    st.markdown("<hr>", unsafe_allow_html=True)

main_scan_clicked = False
if _show_home_scanner:
    _scan_cta_cols = st.columns([1.5, 3.2, 1.5])
    with _scan_cta_cols[1]:
        main_scan_clicked = st.button("▶  SCAN MARKET NOW", key="main_panel_scan_btn", width="stretch")

# ── SCAN ──────────────────────────────────────────────────────────────

# ── 🕰️ Time-travel banner (shown whenever TT is active) ───────────────
_tt_banner = _tt.format_banner()
if _tt_banner and _show_home_scanner:
    st.markdown(
        f'<div style="background:#1a0a00;border:2px solid #f0b429;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-family:\'Space Mono\',monospace;'
        f'font-size:13px;font-weight:700;color:#f0b429;letter-spacing:0.3px;">'
        f'{_tt_banner}</div>',
        unsafe_allow_html=True,
    )

if scan_clicked or main_scan_clicked:
    st.markdown(
        f'<div class="section-lbl">⏳ Scanning {n:,} NSE Equities — Mode {mode_display["display_num"]}: {mode_display["display_name"]}</div>',
        unsafe_allow_html=True)

    # ── 🕰️ Activate time-travel BEFORE scan if toggle is on ───────────
    _tt_active_date = st.session_state.get("tt_date_val")
    if _tt_active_date is not None and _TIME_TRAVEL_OK:
        with st.spinner(f"🕰️ Preparing historical snapshot for {_tt_active_date.strftime('%d %b %Y')}…"):
            _snapped = _tt.activate(_tt_active_date)
        st.caption(f"🕰️ Time-travel active — {_snapped} ticker(s) snapshotted to {_tt_active_date}")
        # BUG FIX: Reset Nifty cache so get_nifty_20d_return() re-fetches
        # with the TT cutoff applied, not a previously cached live value.
        _NIFTY_20D_RET = None

    window = get_current_window()
    expected_date = get_expected_data_date()
    _snapshot_hint_path = get_snapshot_path(expected_date)
    skip_preload = False
    do_snapshot = False
    force_live_refresh = False

    if _tt_active_date is None:
        _session_plan = get_scan_data_plan()
        window = str(_session_plan.get("window", window) or window)
        expected_date = _session_plan.get("expected_date", expected_date)
        _snapshot_hint_path = _session_plan.get("snapshot_path", _snapshot_hint_path)
        do_snapshot = bool(_session_plan.get("save_snapshot_after_scan", False))
        force_live_refresh = bool(_session_plan.get("force_live_refresh", False))
        if bool(_session_plan.get("use_snapshot", False)) and snapshot_exists(expected_date):
            loaded = load_snapshot_into_ALL_DATA(expected_date)
            if loaded > 0:
                total_universe = len(all_tickers)
                missing = max(0, total_universe - loaded)
                if missing > 0:
                    # Snapshot is partial -- preload_all fills the gaps via
                    # historical OHLCV (CSV cache / yfinance period=6mo).
                    # Tickers already in ALL_DATA are cache hits; skipped.
                    st.info(
                        f"📂 Loaded {loaded} tickers from "
                        f"{expected_date} snapshot. "
                        f"Fetching history for {missing} remaining tickers..."
                    )
                    skip_preload = False
                else:
                    # Full snapshot -- nothing left to fetch.
                    st.info(
                        f"📂 Loaded {loaded} tickers from "
                        f"{expected_date} snapshot. "
                        f"No live refresh needed."
                    )
                    skip_preload = True
            else:
                st.warning(
                    f"⚠️ Snapshot found at {_snapshot_hint_path} "
                    "but could not be loaded. Falling back to live/cache preload."
                )

    if False and _tt_active_date is None:
        if window in ("PRE_MARKET", "WEEKEND"):
            if snapshot_exists(expected_date):
                loaded = load_snapshot_into_ALL_DATA(expected_date)
                if loaded > 0:
                    st.info(
                        f"📂 Loaded {loaded} tickers from "
                        f"{expected_date} closing snapshot. "
                        f"No download needed."
                    )
                    skip_preload = True
                else:
                    st.warning(
                        f"⚠️ Snapshot found at {_snapshot_hint_path} "
                        "but could not be loaded. Falling back to preload."
                    )
                    skip_preload = False
            else:
                skip_preload = False
        elif window == "CLOSED":
            if snapshot_exists(expected_date):
                loaded = load_snapshot_into_ALL_DATA(expected_date)
                if loaded > 0:
                    st.info(
                        f"📂 Loaded {loaded} tickers from "
                        f"{expected_date} closing snapshot. "
                        f"No download needed."
                    )
                    skip_preload = True
                    do_snapshot = False
                else:
                    skip_preload = False
                    do_snapshot = False
            else:
                skip_preload = False
                do_snapshot = True
        else:
            skip_preload = False
            do_snapshot = False

    preload_stats = {
        "total": len(all_tickers),
        "loaded": 0,
        "downloaded": 0,
        "fallback_used": 0,
        "cache_hits": 0,
        "force_live_refresh": force_live_refresh,
    }

    if not skip_preload:
        preload_message = "Preparing price-history preload..."
        if force_live_refresh and window == "LIVE":
            preload_message = "Refreshing live market data..."
        elif force_live_refresh and window == "CLOSED":
            preload_message = "Refreshing latest post-close data..."
        elif window in ("PRE_MARKET", "WEEKEND"):
            preload_message = "Preparing latest close data..."
        preload_bar, preload_status, preload_eta, preload_started = _start_stage_feedback(
            preload_message
        )
        _preload_state = {
            "done": 0,
            "total": len(all_tickers),
            "loaded": 0,
        }
        _preload_render = {
            "done": 0,
            "ts": 0.0,
            "step": max(12, len(all_tickers) // 120) if all_tickers else 12,
        }

        def _update_preload(done: int, total: int, loaded: int) -> None:
            _preload_state["done"] = done
            _preload_state["total"] = total
            _preload_state["loaded"] = loaded
            now = time.time()
            should_render = (
                done == total
                or done == 1
                or (done - _preload_render["done"]) >= _preload_render["step"]
                or (now - _preload_render["ts"]) >= 0.25
            )
            if not should_render:
                return
            _preload_render["done"] = done
            _preload_render["ts"] = now
            _update_stage_feedback(
                preload_bar,
                preload_status,
                preload_eta,
                preload_started,
                done,
                total,
                loaded,
                "Preloaded",
                "Ready",
            )

        preload_stats = preload_all(
            all_tickers,
            period="6mo",
            workers=workers,
            progress_callback=_update_preload,
            force_live_refresh=force_live_refresh,
        )
        _finish_stage_feedback(
            preload_bar,
            preload_status,
            preload_eta,
            preload_started,
            _preload_state["total"] if _preload_state["total"] > 0 else len(all_tickers),
            _preload_state["loaded"],
            "Ready",
        )

    if _tt_active_date is None and do_snapshot and window == "CLOSED" and not snapshot_exists(expected_date):
        try:
            from strategy_engines._engine_utils import ALL_DATA

            saved = save_closing_snapshot(ALL_DATA, expected_date, require_live_source=True)
            if saved > 0:
                st.success(f"💾 Closing snapshot saved: {saved} tickers for {expected_date}")
        except Exception:
            pass

    # Warm the active-mode ML model only after preload so it can reuse the
    # loaded history and avoid duplicate network work during scan startup.
    if _SKLEARN_OK:
        try:
            get_train_function(mode)()
        except Exception:
            pass

    try:
        results, elapsed = run_scan(all_tickers, mode, workers=workers)
    finally:
        # Always restore — even if scan raised an exception
        if _tt_active_date is not None and _TIME_TRAVEL_OK:
            _tt.restore()
            # BUG FIX: Reset Nifty cache after restore so next live scan
            # does not reuse the TT-truncated Nifty return value.
            _NIFTY_20D_RET = None

    _scan_time_label = (
        _tt_active_date.strftime("%d %b %Y (TT)") if _tt_active_date
        else datetime.now().strftime("%H:%M:%S")
    )
    st.session_state.update({
        "results":       results,
        "scan_time":     _scan_time_label,
        "elapsed":       elapsed,
        "mode":          mode,
        "tt_was_active": _tt_active_date is not None,
        "tt_scan_date":  str(_tt_active_date) if _tt_active_date else "",
    })

    # FIX 6 — Auto-backfill actual returns for past predictions (background)
    st.rerun()

# ── SECTOR SCREENER DASHBOARD ─────────────────────────────────────
if sector_screener_clicked:
    st.session_state["show_sector_screener"] = True

if st.session_state.get("show_sector_screener", False):
    if _SECTOR_SCREENER_UI_OK:
        render_sector_screener_dashboard(
            mode=mode,
            enhance_results_fn=enhance_results,
            apply_enhanced_logic_fn=apply_enhanced_logic,
            apply_universal_grading_fn=apply_universal_grading,
            apply_phase4_logic_fn=apply_phase4_logic,
            apply_phase42_logic_fn=apply_phase42_logic,
            compute_market_bias_fn=compute_market_bias,
        )
    else:
        # ── Auto-retry: try importing again in case file was added after startup ──
        _retry_ok = False
        try:
            import importlib, sys
            for _mod in [
                "app_sector_screener_dashboard",
                "strategy_engines.app_sector_screener_dashboard",
            ]:
                if _mod in sys.modules:
                    del sys.modules[_mod]
            try:
                from strategy_engines.app_sector_screener_dashboard import render_sector_screener_dashboard as _rsd  # type: ignore[import]
            except Exception:
                from app_sector_screener_dashboard import render_sector_screener_dashboard as _rsd  # type: ignore[import]
            _retry_ok = True
        except Exception as _retry_err:
            _retry_err_msg = str(_retry_err)
        if _retry_ok:
            st.session_state["_sector_ui_ok_runtime"] = True
            _rsd(  # type: ignore[possibly-undefined]
                mode=mode,
                enhance_results_fn=enhance_results,
                apply_enhanced_logic_fn=apply_enhanced_logic,
                apply_universal_grading_fn=apply_universal_grading,
                apply_phase4_logic_fn=apply_phase4_logic,
                apply_phase42_logic_fn=apply_phase42_logic,
                compute_market_bias_fn=compute_market_bias,
            )
        else:
            st.error(
                "❌ **Sector Screener could not load.**\n\n"
                "**Checklist:**\n"
                "1. `strategy_engines/app_sector_screener_dashboard.py` must exist\n"
                "2. `strategy_engines/multi_index_market_bias_engine.py` must also exist\n"
                "3. If you keep these files next to `app.py` instead, that layout is also supported\n"
                "4. After placing or changing the files, **fully restart** Streamlit:\n"
                "   - Press `Ctrl+C` in the terminal\n"
                "   - Run `streamlit run app.py` again\n\n"
                f"*Import error: `{_retry_err_msg}`*"  # type: ignore[possibly-undefined]
            )

    if _SECTOR_EXPLORER_UI_OK:
        render_sector_explorer_section(fetch_nse_tickers())
    else:
        st.warning(
            "Sector Explorer is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_EXPLORER_UI_ERR}"
        )

    _sector_intel_ready = isinstance(st.session_state.get("last_scan_df"), pd.DataFrame) and not st.session_state.get("last_scan_df").empty
    if _SECTOR_INTELLIGENCE_UI_OK and _sector_intel_ready:
        render_sector_intelligence_section()
    elif _sector_intel_ready:
        st.warning(
            "Sector Intelligence is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_INTELLIGENCE_UI_ERR}"
        )

# ── NEW: MARKET BIAS UI PANEL (Isolated) ──────────────────────────────
    try:
        from strategy_engines._engine_utils import ALL_DATA as _sector_prediction_all_data
    except Exception:
        _sector_prediction_all_data = {}

    if _SECTOR_PREDICTION_UI_OK:
        render_sector_prediction_section(
            scan_df=st.session_state.get("last_scan_df"),
            all_data=_sector_prediction_all_data,
        )
    else:
        st.warning(
            "Sector Prediction is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_PREDICTION_UI_ERR}"
        )

if st.session_state.get("battle_show_panel", False):
    st.markdown("<hr>", unsafe_allow_html=True)
    _battle_hdr_col, _battle_close_col = st.columns([6, 1])
    with _battle_hdr_col:
        st.markdown('<h2>⚔️ Compare Stocks</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
            'This panel now opens in the main UI. Enter up to 10 stocks and run the full comparison here.</div>',
            unsafe_allow_html=True,
        )
    with _battle_close_col:
        st.markdown("<br>", unsafe_allow_html=True)
        _battle_close_panel = st.button("Close", key="battle_close_panel_btn", use_container_width=True)

    if _battle_close_panel:
        st.session_state["battle_show_panel"] = False
        st.rerun()

    configure_nse_stock_search(fetch_nse_tickers())
    with st.form("battle_mode_form", clear_on_submit=False):
        _battle_input_col1, _battle_input_col2 = st.columns(2)
        with _battle_input_col1:
            _t1  = render_nse_stock_input("Stock 1",  key="battle_t1",  placeholder="e.g. RELIANCE")
            _t2  = render_nse_stock_input("Stock 2",  key="battle_t2",  placeholder="e.g. TCS")
            _t3  = render_nse_stock_input("Stock 3",  key="battle_t3",  placeholder="e.g. INFY")
            _t4  = render_nse_stock_input("Stock 4",  key="battle_t4",  placeholder="e.g. HDFCBANK")
            _t5  = render_nse_stock_input("Stock 5",  key="battle_t5",  placeholder="e.g. SBIN")
        with _battle_input_col2:
            _t6  = render_nse_stock_input("Stock 6",  key="battle_t6",  placeholder="e.g. ICICIBANK")
            _t7  = render_nse_stock_input("Stock 7",  key="battle_t7",  placeholder="e.g. AXISBANK")
            _t8  = render_nse_stock_input("Stock 8",  key="battle_t8",  placeholder="e.g. BAJFINANCE")
            _t9  = render_nse_stock_input("Stock 9",  key="battle_t9",  placeholder="e.g. TATAMOTORS")
            _t10 = render_nse_stock_input("Stock 10", key="battle_t10", placeholder="e.g. MARUTI")

        _battle_main_run = st.form_submit_button("Run Battle Analysis", use_container_width=True)
    if _battle_main_run:
        _all_inputs = [_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10]
        _battle_tickers = [t.strip() for t in _all_inputs if t and t.strip()][:10]
        if not _battle_tickers:
            st.warning("Please enter at least 1 stock.")
        else:
            st.session_state["battle_mode_request"] = mode
            st.session_state["battle_tickers_request"] = _battle_tickers

render_tomorrow_picks_panel()

if st.session_state.get("show_bias_engine"):
    st.markdown('<div class="section-lbl">📊 Market Bias Engine (Analytics)</div>', unsafe_allow_html=True)
    with st.spinner("Crunching latest Nifty (^NSEI) indicators..."):
        _ui_tt_key = str(st.session_state.get("tt_date_val") or "live")
        _bias_data = compute_market_bias_ui(_ui_tt_key)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Bias", _bias_data["bias"])
    with col2:
        st.metric("Confidence", f"{_bias_data['confidence']}%")
    with col3:
        st.metric("Expected Move", _bias_data["expected_move"])

    with st.expander("Short Reason Breakdown", expanded=True):
        for r in _bias_data["reasons"]:
            st.markdown(f"- {r}")

    if st.button("Close Bias Panel"):
        st.session_state["show_bias_engine"] = False
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────────────
if _show_home_scanner and "results" in st.session_state:
    results     = st.session_state["results"]
    stored_mode = st.session_state.get("mode", mode)
    stored_mode_display = ui_mode_meta.get(
        stored_mode,
        {"display_num": stored_mode, "display_name": mode_names.get(stored_mode, "Mode")},
    )
    elapsed_d   = st.session_state.get("elapsed", 0)
    scan_time_d = st.session_state.get("scan_time", "—")
    mc2         = mode_colors.get(stored_mode, "#00d4a8")

    st.markdown(
        f'<div class="result-hdr">'
        f'<h3>🏆 Bullish Candidates</h3>'
        f'<span class="mode-pill {pill_cls[stored_mode]}">M{stored_mode_display["display_num"]} · {stored_mode_display["display_name"].upper()}</span>'
        f'<span class="count-pill">{len(results)}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:#4a6480;">'
        f'Scanned at {scan_time_d} · {elapsed_d:.1f}s</span></div>',
        unsafe_allow_html=True)

    if results:
        # ── Summary metrics (unchanged) ──────────────────────────────
        df_raw = pd.DataFrame(results)
        i1, i2, i3 = st.columns(3)
        with i1: st.metric("📊 Avg RSI",      f"{df_raw['RSI'].mean():.1f}")
        with i2: st.metric("💰 Avg Price",    f"₹{df_raw['Price (₹)'].mean():,.0f}")
        with i3: st.metric("⚡ Avg Vol / Avg", f"{df_raw['Vol / Avg'].mean():.2f}×")

        # ── Enhance with scoring / backtest / ML ─────────────────────
        with st.spinner("🔢 Computing Smart Scores, Backtest & ML probabilities …"):
            df = enhance_results(results, stored_mode)

        # ── Enhanced Logic Engine — runs FIRST so Setup Quality /
        # ── Volume Trend / Trap Risk are available for Prediction Score
        try:
            df = apply_enhanced_logic(df)
        except Exception:
            pass

        # ── Universal Grading Engine ──────────────────────────────────
        try:
            # FIX 2: reuse cached bias if younger than 30 min
            _now_ts = time.time()
            _cached_mb   = st.session_state.get("market_bias_result")
            _cached_ts   = st.session_state.get("market_bias_ts", 0.0)
            _cached_ttkey = st.session_state.get("market_bias_tt_key", "live")
            _cur_ttkey   = str(st.session_state.get("tt_date_val") or "live")
            _cache_valid = (
                _cached_mb
                and (_now_ts - float(_cached_ts)) < 1800
                and _cached_ttkey == _cur_ttkey   # bust cache on TT date change
            )
            if _cache_valid:
                _mb = _cached_mb
            else:
                _mb = compute_market_bias()
                st.session_state["market_bias_result"]  = _mb
                st.session_state["market_bias_ts"]      = _now_ts
                st.session_state["market_bias_tt_key"]  = _cur_ttkey
            df = apply_universal_grading(df, _mb)
        except Exception:
            pass

        # ── Phase 4 Logic Engine (Setup Type, Reason, Risk, Final Signal)
        try:
            df = apply_phase4_logic(df, _mb)
        except Exception:
            pass

        try:
            from trade_decision_simple import apply_trade_decision_simple
            df = apply_trade_decision_simple(df)
        except Exception:
            pass

        # ── Learning prediction (added column only) ───────────────────
        try:
            from learning_engine import predict_success
            df["Learned Prob %"] = df.apply(lambda row: predict_success(row), axis=1)
        except Exception:
            pass

        # ── Phase 4.2 Logic Engine (Advanced Trap, Expected Move, Adjusted Signal)
        try:
            df = apply_phase42_logic(df)
        except Exception:
            pass

        try:
            df["Trap Check"] = df.apply(_trap_check_label, axis=1)
        except Exception:
            pass

        # ── Phase 4.3/4.4 (Dynamic Intelligence + Feedback Tracking) ─
        try:
            df = apply_phase43_logic(df)
        except Exception:
            pass
        try:
            df = apply_phase44_logic(df)
        except Exception:
            pass

        st.session_state["last_scan_df"] = df.copy()

        try:
            from prediction_feedback_store import feedback_summary, log_scan_predictions

            _log_key = f"{stored_mode}|{scan_time_d}|{len(df)}"
            if st.session_state.get("_prediction_log_key") != _log_key:
                log_scan_predictions(df, stored_mode, st.session_state.get("market_bias_result"))
                st.session_state["_prediction_log_key"] = _log_key
            _fs = feedback_summary()
            if _fs.get("total_logged", 0):
                _cap = f"📒 Prediction log: {_fs['total_logged']} row(s) stored"
                if _fs.get("rows_with_outcome"):
                    _cap += f"; {_fs.get('rows_with_outcome', 0)} with outcomes"
                if _fs.get("accuracy_pct") is not None and _fs.get("rows_with_outcome", 0):
                    _cap += f". Recent accuracy: {_fs.get('accuracy_pct')}%"
                _cap += "."
                st.caption(_cap)
        except Exception:
            pass

        try:
            learning_status = _refresh_feedback_learning_system(force=True)
        except Exception:
            pass
        try:
            signal_weight_status = _refresh_signal_weight_status(force_update=True)
        except Exception:
            pass

        if "Next-Day Signal" in df.columns:
            with st.sidebar:
                counts = df["Next-Day Signal"].value_counts().to_dict()
                sg = counts.get("🔥 Strong Green", 0)
                pu = counts.get("🟢 Possible Up", 0)
                ri = counts.get("⚠️ Risky (Late Entry)", 0)
                we = counts.get("❌ Weak Setup", 0)

                st.markdown('<hr><div class="section-lbl">📊 Next-Day Signals Summary</div>', unsafe_allow_html=True)
                _nd_summary = (
                    '<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;padding:12px 14px;margin-bottom:16px;">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">🔥 Strong Green</span>'
                    '<span style="font-size:13px;color:#00d4a8;font-weight:700">{sg}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">🟢 Possible Up</span>'
                    '<span style="font-size:13px;color:#0094ff;font-weight:700">{pu}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">⚠️ Risky (Late Entry)</span>'
                    '<span style="font-size:13px;color:#f0b429;font-weight:700">{ri}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">❌ Weak Setup</span>'
                    '<span style="font-size:13px;color:#ff4d6d;font-weight:700">{we}</span></div>'
                    '</div>'
                ).format(sg=sg, pu=pu, ri=ri, we=we)
                st.markdown(_nd_summary, unsafe_allow_html=True)

        # ── TASK 4: Section Headers & Spacing ─────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2>🔥 Top Picks</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 1: Top Picks Cards ───────────────────────────────────
        top_n = min(5, len(df))
        cols = st.columns(top_n)
        for i, (col, (_, row)) in enumerate(zip(cols, df.head(top_n).iterrows())):
            sym = row.get("Symbol", "—")
            fin   = row.get("Final Score", 0)
            bt    = row.get("Backtest %", 50)
            ml    = row.get("ML %", 50)
            rsi_v = float(row.get("RSI", 0))
            vol   = row.get("Vol / Avg", 0)
            trap_status = row.get("Trap Check", _trap_check_label(row))
            tv_link = tv_chart_url(sym)
            nd_sig  = row.get("Next-Day Signal", "❌ No Data")

            # Score colour
            if fin > 75:     color = "#00d4a8"
            elif fin >= 60:  color = "#0094ff"
            elif fin >= 40:  color = "#f0b429"
            else:            color = "#ff4d6d"

            # FIX 7 — Stop-loss and target (display-only, not added to df)
            # SL  = Price × (1 − (Δ vs EMA20 % / 100) × 0.5)  half EMA distance below
            # Tgt = Price × (1 + (Δ vs EMA20 % / 100) × 1.5)  1.5× EMA distance above
            try:
                _price   = float(row.get("Price (₹)", 0) or 0)
                _de20    = float(row.get("Δ vs EMA20 (%)", 0) or 0)
                if _price > 0 and _de20 != 0:
                    _sl  = round(_price * (1 - (_de20 / 100) * 0.5), 2)
                    _tgt = round(_price * (1 + (_de20 / 100) * 1.5), 2)
                    _sl_tgt_html = (
                        f'<div style="font-size:11px;color:#4a6480;margin-top:8px;">'
                        f'SL ₹{_sl:,.2f}&nbsp;&nbsp;|&nbsp;&nbsp;Tgt ₹{_tgt:,.2f}</div>'
                    )
                else:
                    _sl_tgt_html = ""
            except Exception:
                _sl_tgt_html = ""

            with col:
                if "Trap" in str(trap_status):
                    _trap_bg = "rgba(255,77,109,0.12)"
                    _trap_fg = "#ff4d6d"
                elif "Caution" in str(trap_status):
                    _trap_bg = "rgba(240,180,41,0.12)"
                    _trap_fg = "#f0b429"
                else:
                    _trap_bg = "rgba(0,212,168,0.10)"
                    _trap_fg = "#00d4a8"
                trap_html = (
                    f'<div style="margin-top:8px;color:{_trap_fg};font-size:12px;'
                    f'font-weight:bold;background:{_trap_bg};'
                    f'padding:4px 8px;border-radius:4px;display:inline-block;">'
                    f'{trap_status}</div>'
                )
                st.markdown(
                    f'<div style="border:1px solid #1a2840;padding:16px;border-radius:8px;'
                    f'background:#0f1823;position:relative;">'
                    f'<div style="font-size:20px;font-weight:bold;margin-bottom:8px;">{sym}</div>'
                    f'<div style="font-size:32px;font-weight:bold;color:{color};margin-bottom:12px;">{fin:.1f}</div>'
                    f'<div style="font-size:14px;font-weight:bold;color:#ccd9e8;margin-bottom:8px;">{nd_sig}</div>'
                    f'<div style="font-size:14px;color:#ccd9e8;line-height:1.6;">'
                    f'<b>BT:</b> {bt:.1f}%&nbsp; '
                    f'<b>ML:</b> {ml:.1f}%&nbsp; '
                    f'<b>RSI:</b> {rsi_v:.1f}&nbsp; '
                    f'<b>Vol:</b> {vol:.1f}x'
                    f'</div>{_sl_tgt_html}{trap_html}</div>',
                    unsafe_allow_html=True,
                )
                st.link_button("📈 TradingView", tv_link, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2>📊 Full Rankings</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 2 & 5: Clean Table with TradingView Link ─────────────
        table_df = df.copy()
        try:
            from strategy_engines._engine_utils import add_rank_score_columns
            table_df = add_rank_score_columns(table_df)
            if "rank_score" in table_df.columns:
                table_df = table_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
                table_df["Rank Score"] = table_df["rank_score"]
        except Exception:
            pass
        if "Rank Score" not in table_df.columns:
            table_df["Rank Score"] = 0.0
        table_df.insert(0, "Rank", range(1, len(table_df) + 1))
        table_df["Ticker"] = table_df["Symbol"]
        table_df["TradingView"] = table_df["Symbol"].apply(lambda s: tv_chart_url(s))

        display_cols = [
            "Rank", "Rank Score", "Ticker", "Score", "Backtest %", "ML %",
            "Final Score", "Prediction Score", "Conviction Tier", "Trap Check", "Next-Day Signal", "TradingView",
            "Learned Prob %",
            "Action", "Hold Days",
        ]
        display_cols = [c for c in display_cols if c in table_df.columns]

        st.dataframe(
            table_df[display_cols],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank"),
                "Rank Score": st.column_config.NumberColumn("Rank Score", format="%.2f"),
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Backtest %": st.column_config.NumberColumn("Backtest %", format="%.1f%%"),
                "ML %": st.column_config.NumberColumn("ML %", format="%.1f%%"),
                "Final Score": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "Prediction Score": st.column_config.NumberColumn("Pred Score", format="%.1f"),
                "Conviction Tier": st.column_config.TextColumn("Conviction"),
                "Trap Check": st.column_config.TextColumn("Trap Check"),
                "Next-Day Signal": st.column_config.TextColumn("Signal"),
                "TradingView": st.column_config.LinkColumn("TradingView Link", display_text="📈 Open Chart"),
                "Action": st.column_config.TextColumn("Action"),
                "Hold Days": st.column_config.TextColumn("Hold Days"),
            },
            use_container_width=True,
            hide_index=True,
        )

        # ── TASK 3: Expandable Details ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _visible_details_df = table_df.head(_VISIBLE_RESULT_LIMIT).copy()
        st.caption(
            f"Details panel limited to top {_VISIBLE_RESULT_LIMIT} stocks to keep the page shorter."
        )
        for _, row in _visible_details_df.iterrows():
            sym = row.get("Symbol", "—")
            fin = row.get("Final Score", 0)
            with st.expander(f"🔍 {sym} Details (Final Score: {fin:.1f})"):
                brk_col, ind_col = st.columns([1, 2])
                with brk_col:
                    st.markdown("**Score Breakdown**")
                    st.json(row.get("_breakdown", {}))
                with ind_col:
                    st.markdown("**Key Indicators**")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("RSI", f"{row.get('RSI', 0):.1f}")
                    ic2.metric("EMA 20", f"₹{row.get('EMA 20', 0):.2f}")
                    ic3.metric("EMA 50", f"₹{row.get('EMA 50', 0):.2f}")
                    ic4.metric("Vol / Avg", f"{row.get('Vol / Avg', 0):.2f}x")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("5D Return", f"{row.get('5D Return (%)', 0):+.2f}%")
                    rc2.metric("20D Return", f"{row.get('20D Return (%)', 0):+.2f}%")
                    
                    # Handle possible missing columns gracefully
                    h_val = row.get('Δ vs 20D High (%)', 0)
                    rc3.metric("Δ vs 20D Hi", f"{h_val:+.2f}%" if pd.notna(h_val) else "—")

        # ── NEW: Clean Export Helper ──────────────────────────────────
        def get_clean_export_df(df):
            """Return a clean, emoji-free copy of df for CSV/Excel export.
            Only includes columns shown in the UI table. Safe if any column
            is missing. Does NOT modify the original df.
            """
            import re

            _EXPORT_COLS = [
                "Rank", "Rank Score", "Ticker", "Score",
                "Backtest %", "ML %", "Final Score",
                "Prediction Score", "Conviction Tier", "Trap Check", "Next-Day Signal",
            ]

            def _strip_emoji(val):
                """Remove emoji / icon characters from a string value."""
                if not isinstance(val, str):
                    return val
                # Remove all emoji and non-ASCII symbols
                cleaned = re.sub(
                    r"[\U00010000-\U0010ffff"
                    r"\U00002600-\U000027BF"
                    r"\U0001F300-\U0001F9FF"
                    r"\u2700-\u27BF"
                    r"\u2300-\u23FF"
                    r"\u2B50-\u2B55"
                    r"\u231A-\u231B"
                    r"\u25AA-\u25FE"
                    r"\u2614-\u2615"
                    r"\u2648-\u2653"
                    r"\u26AA-\u26AB"
                    r"\u26BD-\u26BE"
                    r"\u26C4-\u26C5"
                    r"\u26CE-\u26CE"
                    r"\u26D4-\u26D4"
                    r"\u26EA-\u26EA"
                    r"\u26F2-\u26F3"
                    r"\u26F5-\u26F5"
                    r"\u26FA-\u26FA"
                    r"\u26FD-\u26FD]",
                    "",
                    val,
                    flags=re.UNICODE,
                )
                return cleaned.strip()

            _copy = df.copy()
            # Keep only columns that exist in this df
            _cols = [c for c in _EXPORT_COLS if c in _copy.columns]
            _copy = _copy[_cols]

            # Round numeric percentage columns to 1 decimal place
            for _pct_col in ["Backtest %", "ML %", "Final Score", "Prediction Score", "Rank Score"]:
                if _pct_col in _copy.columns:
                    _copy[_pct_col] = pd.to_numeric(_copy[_pct_col], errors="coerce").round(1)

            # Round Score to 0 decimal places
            if "Score" in _copy.columns:
                _copy["Score"] = pd.to_numeric(_copy["Score"], errors="coerce").round(0)

            # Strip emojis from all string columns
            for _col in _copy.select_dtypes(include="object").columns:
                _copy[_col] = _copy[_col].apply(_strip_emoji)

            return _copy

        # ── CSV download (uses clean export layer) ────────────────────
        st.markdown("<br><br>", unsafe_allow_html=True)
        _clean_export = get_clean_export_df(table_df)
        _csv_buf = io.StringIO()
        _clean_export.to_csv(_csv_buf, index=False)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="⬇ Download Results CSV",
                data=_csv_buf.getvalue().encode("utf-8-sig"),
                file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}_mode{stored_mode}.csv",
                mime="text/csv",
                use_container_width=True,
                key="main_scan_csv_download",
            )

        # ── ML status note ───────────────────────────────────────────
        if not _SKLEARN_OK:
            st.info("ℹ️  scikit-learn not installed — ML % column shows neutral 50. "
                    "Run `pip install scikit-learn` to enable ML probability.")

        try:
            from strategy_engines._engine_utils import get_tomorrow_top_picks
            _tomorrow_df = get_tomorrow_top_picks(df, source="main", top_n=3)
        except Exception:
            _tomorrow_df = pd.DataFrame()

        if isinstance(_tomorrow_df, pd.DataFrame) and not _tomorrow_df.empty:
            _tomorrow_df = _tomorrow_df.copy()
            try:
                from trade_decision_simple import apply_trade_decision_simple_any
                _tomorrow_df = apply_trade_decision_simple_any(_tomorrow_df)
            except Exception:
                pass
            _signal_col = "Adjusted Signal" if "Adjusted Signal" in _tomorrow_df.columns else "Next-Day Signal"
            _tomorrow_df["Chart"] = _tomorrow_df["Symbol"].apply(lambda s: tv_chart_url(str(s)))

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<h2>Top 3 Buyable For Tomorrow</h2>', unsafe_allow_html=True)
            st.caption("Best next-day buy candidates from this mode scan. Save them once to keep them until you delete them.")

            _tomorrow_symbols = [
                _normalize_tomorrow_symbol(symbol)
                for symbol in _tomorrow_df.get("Symbol", pd.Series(dtype=object)).tolist()
                if _normalize_tomorrow_symbol(symbol)
            ]
            _tomorrow_cols = [
                "Symbol", "Tomorrow Pick Score", "Final Score", "Prediction Score",
                _signal_col, "Conviction Tier", "Trap Check", "Chart",
                "Action", "Hold Days",
            ]
            _tomorrow_cols = [c for c in _tomorrow_cols if c in _tomorrow_df.columns]

            st.dataframe(
                _tomorrow_df[_tomorrow_cols],
                column_config={
                    "Symbol": st.column_config.TextColumn("Ticker"),
                    "Tomorrow Pick Score": st.column_config.NumberColumn("Tomorrow Score", format="%.1f"),
                    "Final Score": st.column_config.NumberColumn("Final Score", format="%.1f"),
                    "Prediction Score": st.column_config.NumberColumn("Pred Score", format="%.1f"),
                    "Adjusted Signal": st.column_config.TextColumn("Signal", width="medium"),
                    "Next-Day Signal": st.column_config.TextColumn("Signal", width="medium"),
                    "Conviction Tier": st.column_config.TextColumn("Conviction"),
                    "Trap Check": st.column_config.TextColumn("Trap Check"),
                    "Chart": st.column_config.LinkColumn("Chart", display_text="Open Chart"),
                    "Action": st.column_config.TextColumn("Action"),
                    "Hold Days": st.column_config.TextColumn("Hold Days"),
                },
                use_container_width=True,
                hide_index=True,
            )
            _render_add_in_picks_actions(
                _tomorrow_symbols,
                key_prefix=f"scan_top3_mode_{stored_mode}",
                scope_label="main scan",
                bucket=_tomorrow_bucket_for_mode(stored_mode),
                helper_text="Add these top scan picks into Tomorrow's Picks and keep them saved until you remove them.",
            )

    else:
        if _show_home_scanner:
            st.markdown(
                f'<div style="text-align:center;padding:60px 24px;background:#0f1823;'
                f'border:1px solid #1a2840;border-radius:12px;">'
                f'<div style="font-size:48px;opacity:0.3;margin-bottom:16px;">📡</div>'
                f'<div style="color:#4a6480;font-size:13px;line-height:1.7;">'
                f'No stocks matched Mode {mode_display["display_num"]} ({mode_display["display_name"]}) criteria.<br>'
                f'Try <b style="color:#ccd9e8">Mode 1 (Relaxed)</b> for a broader scan.</div></div>',
                unsafe_allow_html=True)
else:
    if _show_home_scanner:
        st.markdown(
            f'<div style="text-align:center;padding:64px 24px;background:#0f1823;'
            f'border:1px solid #1a2840;border-radius:12px;">'
            f'<div style="font-size:52px;opacity:0.25;margin-bottom:18px;">📡</div>'
            f'<div style="color:#4a6480;font-size:14px;line-height:1.8;">'
            f'Select a <b style="color:#ccd9e8">strategy mode</b> in the sidebar<br>'
            f'then click <b style="color:{mc}">▶ SCAN MARKET NOW</b> to begin.'
            f'</div></div>',
            unsafe_allow_html=True)


# ── BREAKOUT / CSV RADAR SECTION ──────────────────────────────────────
if _BREAKOUT_SECTION_OK:
    try:
        render_breakout_radar_section(
            csv_scan_clicked=csv_scan_clicked,
            _CSV_NEXT_DAY_ENGINE_OK=_CSV_NEXT_DAY_ENGINE_OK,
            _DATA_DOWNLOADER_OK=_DATA_DOWNLOADER_OK,
            _BREAKOUT_RADAR_OK=_BREAKOUT_RADAR_OK,
            render_add_in_picks_actions=_render_add_in_picks_actions,
        )
    except TypeError as _breakout_call_exc:
        if "render_add_in_picks_actions" not in str(_breakout_call_exc):
            raise
        render_breakout_radar_section(
            csv_scan_clicked=csv_scan_clicked,
            _CSV_NEXT_DAY_ENGINE_OK=_CSV_NEXT_DAY_ENGINE_OK,
            _DATA_DOWNLOADER_OK=_DATA_DOWNLOADER_OK,
            _BREAKOUT_RADAR_OK=_BREAKOUT_RADAR_OK,
        )
else:
    _csv_panel_open = bool(st.session_state.get("csv_next_day_show_panel", False))
    if csv_scan_clicked or _csv_panel_open:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h2>📂 CSV Next-Day Potential</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
            'Fast local scan using cached CSV data · Focused on tomorrow-up probability and stricter buy quality</div>',
            unsafe_allow_html=True
        )

        if not _DATA_DOWNLOADER_OK or not _CSV_NEXT_DAY_ENGINE_OK:
            st.warning("CSV next-day engine is not available. Check `data_downloader.py` and `csv_next_day_engine.py`.")
        else:
            if csv_scan_clicked:
                # Pass TT cutoff so CSV engine slices data before computing indicators.
                _csv_tt_cut = st.session_state.get("tt_date_val")
                try:
                    _csv_fresh_df = run_csv_next_day(None, cutoff_date=_csv_tt_cut)
                    st.session_state["csv_next_day_results_df"] = (
                        _csv_fresh_df.copy() if isinstance(_csv_fresh_df, pd.DataFrame) else pd.DataFrame()
                    )
                    st.session_state["csv_next_day_last_error"] = ""
                    _ts_label = (
                        _csv_tt_cut.strftime("%d %b %Y (TT)")
                        if _csv_tt_cut else datetime.now().strftime("%d %b %Y, %H:%M")
                    )
                    st.session_state["csv_next_day_last_scan_at"] = _ts_label
                except Exception as _csv_err:
                    st.session_state["csv_next_day_last_error"] = str(_csv_err)

            csv_df = st.session_state.get("csv_next_day_results_df", pd.DataFrame())
            csv_last_error = str(st.session_state.get("csv_next_day_last_error", "") or "").strip()
            csv_last_scan_at = str(st.session_state.get("csv_next_day_last_scan_at", "") or "").strip()

            if csv_last_scan_at:
                st.caption(f"Last CSV scan: {csv_last_scan_at}")

            if csv_last_error:
                st.error(f"CSV scan failed: {csv_last_error}")

            if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
                st.success(f"✅ {len(csv_df)} buy-ready setups matched the stricter tomorrow-up CSV criteria")
                _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                with _m1:
                    st.metric("Matches", f"{len(csv_df):,}")
                with _m2:
                    st.metric("Avg Prob", f"{csv_df['Next Day Prob'].mean():.1f}%")
                with _m3:
                    st.metric("Avg Conf", f"{csv_df['Confidence'].mean():.1f}%")
                with _m4:
                    _ready_count = int((csv_df["Buy Readiness"] == "BUY READY").sum()) if "Buy Readiness" in csv_df.columns else 0
                    st.metric("Buy Ready", f"{_ready_count:,}")
                with _m5:
                    _best_grade = "-"
                    if "Grade" in csv_df.columns:
                        _grade_order = ["A", "B", "C", "D"]
                        _grade_values = csv_df["Grade"].astype(str).tolist()
                        _best_grade = next((g for g in _grade_order if g in _grade_values), _grade_values[0] if _grade_values else "-")
                    st.metric("Best Grade", _best_grade)

                _download_col, _grade_col = st.columns([0.32, 0.68])
                with _download_col:
                    _csv_download_data = csv_df.to_csv(index=False).encode("utf-8-sig")
                    _scan_stamp = csv_last_scan_at.replace(" ", "_").replace(",", "").replace(":", "-") if csv_last_scan_at else datetime.now().strftime("%Y-%m-%d_%H-%M")
                    st.download_button(
                        "⬇️ Download CSV Results",
                        data=_csv_download_data,
                        file_name=f"csv_next_day_results_{_scan_stamp}.csv",
                        mime="text/csv",
                        key="csv_next_day_download_btn",
                    )
                with _grade_col:
                    if "Grade" in csv_df.columns:
                        _grade_counts = csv_df["Grade"].fillna("-").astype(str).value_counts()
                        _grade_summary = " | ".join(
                            f"{_grade}: {_grade_counts.get(_grade, 0)}" for _grade in ["A", "B", "C", "D"]
                        )
                        st.markdown(
                            '<div style="font-size:12px;color:#4a6480;padding-top:8px;">'
                            f'Grading System: <b>A</b> strongest setup · <b>B</b> good setup · '
                            f'<b>C</b> watchlist quality · <b>D</b> weak setup'
                            f'<br>Grade Distribution: {_grade_summary}</div>',
                            unsafe_allow_html=True,
                        )

                _csv_display_df = csv_df.copy()
                try:
                    from trade_decision_simple import apply_trade_decision_simple_any
                    _csv_display_df = apply_trade_decision_simple_any(_csv_display_df)
                except Exception:
                    pass

                st.dataframe(
                    _csv_display_df,
                    column_config={
                        "Symbol":           st.column_config.TextColumn("Ticker"),
                        "Price (₹)":        st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
                        "Next Day Prob":    st.column_config.NumberColumn("Tomorrow Up %", format="%.1f%%"),
                        "Confidence":       st.column_config.NumberColumn("Confidence %", format="%.1f%%"),
                        "Grade":            st.column_config.TextColumn("Grade"),
                        "Buy Readiness":    st.column_config.TextColumn("Buy Verdict"),
                        "Signal":           st.column_config.TextColumn("Signal"),
                        "Setup":            st.column_config.TextColumn("Setup"),
                        "Historical Win %": st.column_config.NumberColumn("Hist Win %", format="%.1f%%"),
                        "Downside Risk %":  st.column_config.NumberColumn("Downside Risk %", format="%.1f%%"),
                        "Analog Count":     st.column_config.NumberColumn("Analogs", format="%d"),
                        "Analog Avg Ret %": st.column_config.NumberColumn("Analog Avg %", format="%.2f%%"),
                        "Setup Quality":    st.column_config.NumberColumn("Setup Q", format="%.1f"),
                        "Trigger Quality":  st.column_config.NumberColumn("Trigger Q", format="%.1f"),
                        "RSI":              st.column_config.NumberColumn("RSI", format="%.1f"),
                        "Vol / Avg":        st.column_config.NumberColumn("Vol/Avg", format="%.2fx"),
                        "Volume Strength":  st.column_config.TextColumn("Volume"),
                        "Bull Trap":        st.column_config.TextColumn("Trap"),
                        "Risk Notes":       st.column_config.TextColumn("Risk Notes", width="large"),
                        "Chart Link":       st.column_config.LinkColumn("Chart", display_text="📈 Open"),
                        "Action":           st.column_config.TextColumn("Action"),
                        "Hold Days":        st.column_config.TextColumn("Hold Days"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            elif not csv_last_error:
                st.info("No clean buy-ready setups were found for tomorrow in the current CSV universe. That usually means wait for better structure instead of forcing a trade.")

if _LIVE_PULSE_SECTION_OK:
    render_live_breakout_pulse(
        live_pulse_clicked=live_pulse_clicked,
        tt_date_val=st.session_state.get("tt_date_val"),
        render_add_in_picks_actions=_render_add_in_picks_actions,
    )

# ══════════════════════════════════════════════════════════
# ⚔️  MULTI-STOCK BATTLE MODE
# ══════════════════════════════════════════════════════════
try:
    from battle_mode_engine import run_battle_mode, compute_battle_scores
    _BATTLE_OK = True
except ImportError:
    _BATTLE_OK = False

if not _BATTLE_OK:
    st.warning("⚠️ battle_mode_engine.py not found. Place it in the same folder as app.py.")
else:
    _battle_request_tickers = st.session_state.get("battle_tickers_request", None)
    _battle_mode = st.session_state.get("battle_mode_request", mode)

    # Execute the battle pipeline only when the sidebar requested it.
    if isinstance(_battle_request_tickers, list) and _battle_request_tickers:
        with st.spinner(f"⚔️ Analysing {len(_battle_request_tickers)} stock(s)…"):
            # ── 🕰️ Activate time-travel for battle if toggle is on ─────
            _tt_battle_date = st.session_state.get("tt_date_val")
            if _tt_battle_date is not None and _TIME_TRAVEL_OK:
                _tt.activate(_tt_battle_date)
            try:
                _battle_symbols = _normalize_compare_symbols(_battle_request_tickers)
                _battle_window = str(get_current_window() or "CLOSED").upper()
                _cached_battle_df = None
                _cached_battle_payload = None
                if _battle_window != "LIVE" and _tt_battle_date is None:
                    _cached_battle_df, _cached_battle_payload = _load_compare_results(_battle_symbols)

                if isinstance(_cached_battle_df, pd.DataFrame) and not _cached_battle_df.empty:
                    st.session_state["battle_results_df"] = _cached_battle_df
                    st.session_state["battle_source_statuses"] = list(_cached_battle_payload.get("source_statuses", []) or [])
                else:
                    _battle_raw = run_battle_mode(_battle_request_tickers, _battle_mode)
                    if not _battle_raw:
                        st.error("No valid data found. Check symbols and try again.")
                        st.session_state["battle_results_df"] = pd.DataFrame()
                        st.session_state["battle_source_statuses"] = []
                    else:
                        _battle_df = enhance_results(_battle_raw, _battle_mode)
                        try:
                            _battle_df = apply_enhanced_logic(_battle_df)
                        except Exception:
                            pass
                        try:
                            _mb = st.session_state.get("market_bias_result", None)
                            _mb_ttkey = st.session_state.get("market_bias_tt_key", "live")
                            _battle_ttkey = str(st.session_state.get("tt_date_val") or "live")
                            if _mb is None or _mb_ttkey != _battle_ttkey:
                                _mb = compute_market_bias()
                                st.session_state["market_bias_result"]  = _mb
                                st.session_state["market_bias_tt_key"]  = _battle_ttkey
                            _battle_df = apply_universal_grading(_battle_df, _mb)
                        except Exception:
                            pass
                        try:
                            _mb2 = st.session_state.get("market_bias_result", None)
                            _battle_df = apply_phase4_logic(_battle_df, _mb2)
                            _battle_df = apply_phase42_logic(_battle_df)
                        except Exception:
                            pass
                        _battle_df = compute_battle_scores(_battle_df)
                        st.session_state["battle_results_df"] = _battle_df
                        _battle_statuses = _build_compare_source_statuses(_battle_symbols)
                        st.session_state["battle_source_statuses"] = _battle_statuses
                        try:
                            _save_compare_results(_battle_symbols, _battle_df, statuses=_battle_statuses)
                        except Exception:
                            pass
            except Exception as _battle_err:
                st.error(f"Battle Mode error: {_battle_err}. Check your tickers and try again.")
                st.session_state["battle_results_df"] = pd.DataFrame()
                st.session_state["battle_source_statuses"] = []
            finally:
                st.session_state["battle_tickers_request"] = None
                if _tt_battle_date is not None and _TIME_TRAVEL_OK:
                    _tt.restore()

    _battle_df = st.session_state.get("battle_results_df", None)
    if isinstance(_battle_df, pd.DataFrame) and not _battle_df.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h2>⚔️ Multi-Stock Battle Mode</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
            'Compare up to 10 stocks head-to-head · Full pipeline per ticker · Ranks by battle probability, quality and risk-adjusted strength</div>',
            unsafe_allow_html=True,
        )

        # ── 🥇 Winner Card ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">🥇 Battle Winner</div>', unsafe_allow_html=True)
        _battle_sources = _summarize_compare_sources(st.session_state.get("battle_source_statuses"))
        if _battle_sources:
            st.caption(f"Data sources: {_battle_sources}")
        _w = _battle_df.iloc[0]
        _w_sym    = _w.get("Symbol", "—")
        _w_score  = _w.get("Final Score", 0)
        _w_conf   = _w.get("Confidence", 50)
        _w_signal = _w.get("Signal", _w.get("Final Signal", "—"))
        _w_setup  = _w.get("Setup Type", _w.get("Volume Trend", "—"))
        _w_bat    = _w.get("Battle Score", 0)
        _w_prob   = _w.get("Battle Probability", _w_bat)
        _w_bconf  = _w.get("Battle Confidence", _w_conf)
        _w_bqual  = _w.get("Battle Quality", _w_score)
        _w_verdict = _w.get("Battle Verdict", "BETTER PICK")
        _w_edge   = _w.get("Battle Edge", 0)
        _w_notes  = _w.get("Battle Notes", "")
        _w_grade  = _w.get("Grade", "—")
        _wc       = "#00d4a8" if _w_bat >= 65 else ("#f0b429" if _w_bat >= 45 else "#ff4d6d")
        st.markdown(
            f'<div style="background:#0b1017;border:2px solid {_wc};border-radius:16px;padding:24px 28px;">'
            f'<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
            f'<div style="font-size:42px;">🥇</div>'
            f'<div>'
            f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;color:#ccd9e8;">{_w_sym}</div>'
            f'<div style="font-size:12px;color:#4a6480;margin-top:4px;">Battle Winner · Grade: <b style="color:{_wc}">{_w_grade}</b></div>'
            f'</div>'
            f'<div style="margin-left:auto;text-align:right;">'
            f'<div style="font-size:32px;font-weight:800;color:{_wc};">{_w_bat:.1f}</div>'
            f'<div style="font-size:11px;color:#4a6480;">Battle Score</div>'
            f'</div></div>'
            f'<div style="display:flex;gap:32px;margin-top:18px;flex-wrap:wrap;">'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Final Score</div>'
            f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_score:.1f}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Battle Probability</div>'
            f'<div style="font-size:18px;font-weight:700;color:{_wc};">{_w_prob:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Confidence</div>'
            f'<div style="font-size:18px;font-weight:700;color:#0094ff;">{_w_conf:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Compare Confidence</div>'
            f'<div style="font-size:18px;font-weight:700;color:#7fd1ff;">{_w_bconf:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Battle Quality</div>'
            f'<div style="font-size:18px;font-weight:700;color:#8cf08c;">{_w_bqual:.1f}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Signal</div>'
            f'<div style="font-size:18px;font-weight:700;color:#f0b429;">{_w_signal}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Setup Type</div>'
            f'<div style="font-size:18px;font-weight:700;color:#b08cff;">{_w_setup}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Verdict</div>'
            f'<div style="font-size:18px;font-weight:700;color:{_wc};">{_w_verdict}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Lead Margin</div>'
            f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_edge:.1f}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if _w_notes:
            st.caption(f"Winner notes: {_w_notes}")
        _render_add_in_picks_actions(
            [_w_sym],
            key_prefix=f"battle_winner_{_normalize_tomorrow_symbol(_w_sym) or 'stock'}",
            scope_label="Compare Stocks winner",
            bucket=_tomorrow_bucket_for_mode(st.session_state.get("mode", 3)),
            helper_text="Add the current battle winner into Tomorrow's Picks without reopening the compare view.",
        )

        # ── 📊 Comparison Table ───────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">📊 Head-to-Head Comparison</div>', unsafe_allow_html=True)
        _table_rows = []
        for _, _br in _battle_df.iterrows():
            _trap_r = str(_br.get("Trap Risk", "")).strip()
            _trap_w = str(_br.get("Trap", "")).strip()
            _trap_flag = "⚠️ Potential Bull Trap" if (_trap_r == "HIGH" or "Bull Trap" in _trap_w) else "✅ Clean"
            _table_rows.append({
                "Rank":         int(_br.get("Battle Rank", 0)),
                "Stock":        _br.get("Symbol", "—"),
                "Verdict":      _br.get("Battle Verdict", "WATCHLIST"),
                "Battle Score": round(float(_br.get("Battle Score", 0)), 1),
                "Probability %": round(float(_br.get("Battle Probability", _br.get("Battle Score", 0))), 1),
                "Compare Conf %": round(float(_br.get("Battle Confidence", _br.get("Confidence", 50))), 1),
                "Quality":      round(float(_br.get("Battle Quality", _br.get("Final Score", 0))), 1),
                "Signal":       _br.get("Signal", _br.get("Final Signal", "—")),
                "Grade":        _br.get("Grade", "—"),
                "Risk Score":   round(float(_br.get("Risk Score", 50)), 1),
                "Edge":         round(float(_br.get("Battle Edge", 0)), 1),
                "⚠️ Trap Check": _trap_flag,
                "Notes":        _br.get("Battle Notes", ""),
            })
        st.dataframe(
            pd.DataFrame(_table_rows),
            column_config={
                "Rank":         st.column_config.NumberColumn("Rank", format="%d"),
                "Stock":        st.column_config.TextColumn("Stock"),
                "Verdict":      st.column_config.TextColumn("Verdict"),
                "Battle Score": st.column_config.NumberColumn("Battle Score", format="%.1f"),
                "Probability %": st.column_config.NumberColumn("Probability %", format="%.1f%%"),
                "Compare Conf %": st.column_config.NumberColumn("Compare Conf %", format="%.1f%%"),
                "Quality":      st.column_config.NumberColumn("Quality", format="%.1f"),
                "Signal":       st.column_config.TextColumn("Signal"),
                "Grade":        st.column_config.TextColumn("Grade"),
                "Risk Score":   st.column_config.NumberColumn("Risk Score", format="%.1f"),
                "Edge":         st.column_config.NumberColumn("Edge", format="%.1f"),
                "⚠️ Trap Check": st.column_config.TextColumn("⚠️ Trap Check"),
                "Notes":        st.column_config.TextColumn("Notes", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("🧾 Full Battle Diagnostics", expanded=False):
            st.dataframe(_battle_df, use_container_width=True, hide_index=True)

        # ── ⚠️ Trap Warnings ─────────────────────────────
        _trap_stocks = [
            str(_r.get("Symbol", "?"))
            for _, _r in _battle_df.iterrows()
            if (str(_r.get("Trap Risk", "")).strip() == "HIGH"
                or "Bull Trap" in str(_r.get("Trap", "")))
        ]
        if _trap_stocks:
            st.warning(
                f"⚠️ **Potential Bull Trap** detected in: {', '.join(_trap_stocks)}  —  "
                "RSI overbought and/or volume declining. Proceed with caution."
            )

# ── 🔮 Stock Aura Panel ───────────────────────────────────────────────
if _STOCK_AURA_OK:
    render_stock_aura_panel()
elif st.session_state.get("aura_show_panel", False):
    st.warning("⚠️ app_stock_aura_section.py not found — place it next to app.py and restart.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:\'Space Mono\',monospace;'
    'font-size:11px;color:#2a3f58;padding:8px 0 16px;">'
    'NSE SENTINEL · Python + Streamlit + yFinance &nbsp;|&nbsp; '
    'For educational purposes only — not financial advice</div>',
    unsafe_allow_html=True)
