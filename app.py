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
from typing import Any
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
        "validation_accuracy_pct": None,
        "training_accuracy_pct": None,
        "last_trained": "",
        "source": "none",
        "message": "Learning engine unavailable.",
        "regime_encoder": {},
        "sector_encoder": {},
    }


def _learning_engine_module_is_usable(module) -> bool:
    try:
        required = (
            "get_training_status",
            "predict_success",
            "restore_learning_bundle",
            "train_learning_model",
        )
        return module is not None and all(callable(getattr(module, name, None)) for name in required)
    except Exception:
        return False


def _load_learning_engine_module():
    """
    Load the project learning engine without mutating sys.modules during
    normal Streamlit execution.
    """
    existing = _sys.modules.get("learning_engine")
    if _learning_engine_module_is_usable(existing):
        return existing

    for module_name in ("learning_engine", "trade_decision_engine"):
        try:
            module = _importlib.import_module(module_name)
            if module_name != "learning_engine":
                return module
            return module
        except Exception:
            continue
    return None


def _fallback_train_learning_model():
    return {
        "model": None,
        "scaler": None,
        "status": _default_learning_status(),
    }


def _module_has_required_attrs(module, required_attrs: tuple[str, ...]) -> bool:
    try:
        if module is None:
            return False
        return all(hasattr(module, attr) for attr in required_attrs)
    except Exception:
        return False


def _load_optional_module(module_name: str, required_attrs: tuple[str, ...] = ()):
    """
    Import an optional local module without popping sys.modules during normal
    app execution.
    """
    existing = _sys.modules.get(module_name)
    if _module_has_required_attrs(existing, required_attrs):
        return existing

    try:
        module = _importlib.import_module(module_name)
        if required_attrs and not _module_has_required_attrs(module, required_attrs):
            return None
        return module
    except Exception:
        return None

import io
import html
import json
import logging
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
import yfinance as yf
from atomic_io import atomic_write_json

_LOG = logging.getLogger(__name__)

try:
    from persistent_store import push_file as _push_persistent_file
except Exception:
    def _push_persistent_file(*a, **kw):  # type: ignore[misc]
        return False


def _queue_persistent_file(path: Path) -> bool:
    try:
        ok = bool(_push_persistent_file(path))
        if not ok:
            _LOG.error("app: queueing persistent sync failed for %s", Path(path).name)
        return ok
    except Exception:
        _LOG.exception("app: persistent sync failed for %s", Path(path).name)
        return False

# Persistent store: pull remote data before any local data readers run.
if not st.session_state.get("_persistence_pulled", False):
    st.session_state["_persistence_pulled"] = True
    try:
        from persistent_store import health_check as _persistence_health_check, pull_all

        _pulled = pull_all()
        _persistence_health = _persistence_health_check()
        try:
            from learning_engine import load_persisted_model

            load_persisted_model()
        except Exception:
            pass
        st.session_state["_persistence_health"] = _persistence_health
        if _pulled > 0:
            st.session_state["_persistence_msg"] = f"Restored {_pulled} data file(s) from cloud storage."
        elif not _persistence_health.get("connected"):
            st.session_state["_persistence_warning"] = (
                "Cloud persistence is NOT active. Add the [github_store] secrets below or Streamlit reboot will wipe saved picks."
            )
    except Exception:
        _LOG.exception("Cloud persistence startup pull failed")
        st.session_state["_persistence_warning"] = "Cloud persistence startup pull failed; continuing with local data."

_learning_engine = None
_run_learning_cycle = None


def _get_learning_engine_module():
    global _learning_engine
    if _learning_engine_module_is_usable(_learning_engine):
        return _learning_engine
    _learning_engine = _load_learning_engine_module()
    return _learning_engine


def get_training_status():
    try:
        module = _get_learning_engine_module()
        fn = getattr(module, "get_training_status", None) if module is not None else None
        result = fn() if callable(fn) else _default_learning_status()
        return result if isinstance(result, dict) else _default_learning_status()
    except Exception:
        return _default_learning_status()


def predict_success(row):
    try:
        module = _get_learning_engine_module()
        fn = getattr(module, "predict_success", None) if module is not None else None
        return float(fn(row)) if callable(fn) else 50.0
    except Exception:
        return 50.0


def restore_learning_bundle(bundle):
    try:
        module = _get_learning_engine_module()
        fn = getattr(module, "restore_learning_bundle", None) if module is not None else None
        return bool(fn(bundle)) if callable(fn) else False
    except Exception:
        return False


def train_learning_model():
    try:
        module = _get_learning_engine_module()
        fn = getattr(module, "train_learning_model", None) if module is not None else None
        result = fn() if callable(fn) else _fallback_train_learning_model()
        return result if isinstance(result, dict) else _fallback_train_learning_model()
    except Exception:
        return _fallback_train_learning_model()


def _get_learning_cycle_runner():
    global _run_learning_cycle
    if callable(_run_learning_cycle):
        return _run_learning_cycle
    try:
        from nse_learning_brain import run_learning_cycle as _brain_runner

        _run_learning_cycle = _brain_runner
    except Exception:
        _run_learning_cycle = None
    return _run_learning_cycle


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
_LEARNING_STATUS_SNAPSHOT_PATH = Path(_HERE) / "data" / "learning_status_snapshot.json"


def _safe_file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime) if path.exists() else 0.0
    except Exception:
        return 0.0


def _json_safe_snapshot(value):
    try:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (datetime, pd.Timestamp)):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.DataFrame):
            return [_json_safe_snapshot(row) for row in value.head(12).to_dict("records")]
        if isinstance(value, dict):
            return {str(key): _json_safe_snapshot(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe_snapshot(item) for item in value]
        return str(value)
    except Exception:
        return str(value)


def _compact_brain_status(brain_status: dict | None) -> dict:
    try:
        brain = brain_status if isinstance(brain_status, dict) else {}
        compact = {
            "started_at": brain.get("started_at", ""),
            "completed_at": brain.get("completed_at", ""),
            "feedback": brain.get("feedback", {}),
            "feedback_summary": brain.get("feedback_summary", {}),
            "meta_model": brain.get("meta_model", {}),
            "mode_models": brain.get("mode_models", {}),
            "sector_model": brain.get("sector_model", {}),
            "weights": brain.get("weights", {}),
            "regime": brain.get("regime", {}),
            "calibration": brain.get("calibration", {}),
            "predictions": brain.get("predictions", {}),
            "error": brain.get("error", ""),
        }
        return _json_safe_snapshot(compact)
    except Exception:
        return {}


def _serialize_signal_weight_status(status: dict | None) -> dict:
    try:
        payload = dict(status or {})
        report = payload.pop("report", pd.DataFrame())
        payload["report_rows"] = (
            report.head(12).to_dict("records")
            if isinstance(report, pd.DataFrame) and not report.empty
            else []
        )
        return _json_safe_snapshot(payload)
    except Exception:
        return {"processed": 0, "top_signal": "", "top_weight": 0.0, "weakest_signal": "", "weakest_weight": 0.0, "report_rows": []}


def _deserialize_signal_weight_status(payload: dict | None) -> dict:
    default = {
        "processed": 0,
        "top_signal": "",
        "top_weight": 0.0,
        "weakest_signal": "",
        "weakest_weight": 0.0,
        "report": pd.DataFrame(),
    }
    try:
        if not isinstance(payload, dict):
            return default
        out = dict(payload)
        report_rows = out.pop("report_rows", [])
        out["report"] = pd.DataFrame(report_rows) if isinstance(report_rows, list) else pd.DataFrame()
        return out
    except Exception:
        return default


def _write_learning_status_snapshot(
    learning_status: dict,
    signal_status: dict,
    *,
    signature: tuple[str, float, float],
) -> None:
    try:
        _LEARNING_STATUS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "signature": list(signature),
            "learning_status": _json_safe_snapshot(learning_status),
            "signal_weight_status": _serialize_signal_weight_status(signal_status),
        }
        atomic_write_json(_LEARNING_STATUS_SNAPSHOT_PATH, payload, indent=2)
        _queue_persistent_file(_LEARNING_STATUS_SNAPSHOT_PATH)
    except Exception:
        return


def _write_learning_status_snapshot_async(
    learning_status: dict,
    signal_status: dict,
    *,
    signature: tuple[str, float, float],
) -> None:
    def _bg_write_snapshot() -> None:
        try:
            _write_learning_status_snapshot(learning_status, signal_status, signature=signature)
        except Exception:
            pass

    threading.Thread(target=_bg_write_snapshot, daemon=True).start()


def _read_learning_status_snapshot() -> dict:
    try:
        if not _LEARNING_STATUS_SNAPSHOT_PATH.exists():
            return {}
        payload = json.loads(_LEARNING_STATUS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        signature = payload.get("signature", [])
        signature_tuple = tuple(signature[:3]) if isinstance(signature, list) else tuple()
        return {
            "saved_at": str(payload.get("saved_at", "") or ""),
            "signature": signature_tuple,
            "learning_status": payload.get("learning_status", {}) if isinstance(payload.get("learning_status"), dict) else {},
            "signal_weight_status": _deserialize_signal_weight_status(payload.get("signal_weight_status")),
        }
    except Exception:
        return {}


def _bootstrap_learning_status() -> tuple[dict, dict]:
    default_signal_status = {
        "processed": 0,
        "top_signal": "",
        "top_weight": 0.0,
        "weakest_signal": "",
        "weakest_weight": 0.0,
        "report": pd.DataFrame(),
    }
    current_sig = (
        str(get_expected_data_date()),
        round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
    )
    cached_status = st.session_state.get("_learning_status")
    cached_weights = st.session_state.get("_signal_weight_status")
    if (
        st.session_state.get("_learning_refresh_sig") == current_sig
        and isinstance(cached_status, dict)
        and isinstance(cached_weights, dict)
    ):
        return cached_status, cached_weights

    snapshot = _read_learning_status_snapshot()
    snapshot_status = snapshot.get("learning_status")
    snapshot_weights = snapshot.get("signal_weight_status")
    if isinstance(snapshot_status, dict):
        st.session_state["_learning_status"] = snapshot_status
        st.session_state["_learning_refresh_sig"] = snapshot.get("signature") or current_sig
        st.session_state["learning_cycle_status"] = snapshot_status.get("brain_status", {})
        if isinstance(snapshot_weights, dict):
            st.session_state["_signal_weight_status"] = snapshot_weights
            st.session_state["_signal_weight_sig"] = snapshot.get("signature") or current_sig
            return snapshot_status, snapshot_weights
        st.session_state["_signal_weight_status"] = default_signal_status
        st.session_state["_signal_weight_sig"] = snapshot.get("signature") or current_sig
        return snapshot_status, default_signal_status

    persisted_status = get_training_status()
    if isinstance(persisted_status, dict) and persisted_status.get("trained"):
        st.session_state["_learning_status"] = persisted_status
        st.session_state["_learning_refresh_sig"] = current_sig
        st.session_state["_signal_weight_status"] = default_signal_status
        st.session_state["_signal_weight_sig"] = current_sig
        return persisted_status, default_signal_status

    fallback_status = _default_learning_status()
    fallback_status["message"] = "Learning warm-up pending. Full refresh runs after the next scan."
    st.session_state["_learning_status"] = fallback_status
    st.session_state["_learning_refresh_sig"] = current_sig
    st.session_state["_signal_weight_status"] = default_signal_status
    st.session_state["_signal_weight_sig"] = current_sig
    return fallback_status, default_signal_status


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
        runner = _get_learning_cycle_runner()
        brain_status = runner(ALL_DATA, force=force) if callable(runner) else {}
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
    signal_snapshot = _signal_weight_status_from_brain(brain_status)
    st.session_state["_learning_status"] = status
    st.session_state["_learning_refresh_sig"] = fresh_sig
    st.session_state["_signal_weight_status"] = signal_snapshot
    st.session_state["_signal_weight_sig"] = fresh_sig
    status["brain_status"] = _compact_brain_status(brain_status)
    _write_learning_status_snapshot_async(status, signal_snapshot, signature=fresh_sig)

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


def _sync_cached_learning_feedback_summary(feedback_stats: dict | None = None) -> dict:
    stats = dict(feedback_stats) if isinstance(feedback_stats, dict) else {}
    if not stats:
        try:
            from prediction_feedback_store import feedback_summary as _feedback_summary

            stats = dict(_feedback_summary())
        except Exception:
            stats = {}
    try:
        cached_status = st.session_state.get("_learning_status")
        if isinstance(cached_status, dict):
            updated_status = dict(cached_status)
            updated_status["feedback_summary"] = dict(stats)
            brain_status = updated_status.get("brain_status", {})
            if isinstance(brain_status, dict):
                brain_copy = dict(brain_status)
                brain_copy["feedback_summary"] = dict(stats)
                updated_status["brain_status"] = brain_copy
                st.session_state["learning_cycle_status"] = brain_copy
            st.session_state["_learning_status"] = updated_status
            try:
                _sig = st.session_state.get("_learning_refresh_sig") or ("", 0.0, 0.0)
                _write_learning_status_snapshot_async(
                    updated_status,
                    st.session_state.get("_signal_weight_status", {}),
                    signature=_sig,
                )
            except Exception:
                pass
    except Exception:
        pass
    return stats


def _feedback_event_signature(feedback_stats: dict | None = None) -> tuple:
    stats = dict(feedback_stats) if isinstance(feedback_stats, dict) else {}
    return (
        str(get_expected_data_date()),
        int(stats.get("total_logged", 0) or 0),
        int(stats.get("rows_with_outcome", 0) or 0),
        str(stats.get("accuracy_pct", "")),
        str(stats.get("bullish_precision_pct", "")),
        str(stats.get("bearish_precision_pct", "")),
        round(_safe_file_mtime(_PREDICTION_FEEDBACK_PATH), 3),
        round(_safe_file_mtime(_SECTOR_PREDICTION_PATH), 3),
    )


def _should_run_full_learning_refresh(feedback_stats: dict | None = None) -> bool:
    stats = dict(feedback_stats) if isinstance(feedback_stats, dict) else {}
    cached_status = st.session_state.get("_learning_status")
    cached_summary = (
        cached_status.get("feedback_summary", {})
        if isinstance(cached_status, dict) and isinstance(cached_status.get("feedback_summary", {}), dict)
        else {}
    )
    brain_status = st.session_state.get("learning_cycle_status")
    try:
        refresh_sig = st.session_state.get("_learning_refresh_sig")
        current_date = str(get_expected_data_date())
        cached_date = str(refresh_sig[0]) if isinstance(refresh_sig, tuple) and refresh_sig else str(refresh_sig or "")
    except Exception:
        cached_date = ""

    if not isinstance(cached_status, dict) or not isinstance(brain_status, dict):
        return True
    if cached_date != str(get_expected_data_date()):
        return True

    current_outcomes = int(stats.get("rows_with_outcome", 0) or 0)
    cached_outcomes = int(cached_summary.get("rows_with_outcome", 0) or 0)
    current_total = int(stats.get("total_logged", 0) or 0)
    cached_total = int(cached_summary.get("total_logged", 0) or 0)

    if current_outcomes > cached_outcomes:
        return True
    if cached_total <= 0 and current_total > 0:
        return True
    if not st.session_state.get("_signal_weight_status"):
        return True
    return False


def _refresh_learning_after_prediction_log(feedback_stats: dict | None = None) -> bool:
    stats = _sync_cached_learning_feedback_summary(feedback_stats)
    feedback_sig = _feedback_event_signature(stats)
    if not _should_run_full_learning_refresh(stats):
        st.session_state["_last_feedback_learning_event_sig"] = feedback_sig
        return False
    if st.session_state.get("_last_feedback_learning_event_sig") == feedback_sig:
        return False
    try:
        _refresh_feedback_learning_system(force=True)
    except Exception:
        pass
    try:
        _refresh_signal_weight_status(force_update=True)
    except Exception:
        pass
    st.session_state["_last_feedback_learning_event_sig"] = feedback_sig
    return True

_POST_CLOSE_OUTCOME_INTERVAL_SEC = 15 * 60
_POST_CLOSE_OUTCOME_SYMBOL_LIMIT = 160


def _is_blank_outcome_value(value: object) -> bool:
    try:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return str(value).strip().lower() in {"", "nan", "none", "nat", "null"}
    except Exception:
        return True


def _is_valid_correct_value(value: object) -> bool:
    return str(value).strip() in {"True", "False"}


def _post_close_outcome_window() -> bool:
    try:
        if bool(st.session_state.get("tt_toggle_val")):
            return False
        tt_mod = globals().get("_tt")
        if globals().get("_TIME_TRAVEL_OK", False) and getattr(tt_mod, "is_active", lambda: False)():
            return False
        window = str(get_current_window() or "").upper()
        return window in {"CLOSED", "PRE_MARKET", "WEEKEND"}
    except Exception:
        return False


def _pending_outcome_symbols(limit: int = _POST_CLOSE_OUTCOME_SYMBOL_LIMIT) -> dict[str, object]:
    symbols: list[str] = []
    seen: set[str] = set()
    pending_stock = 0
    pending_sector = 0

    def _add_symbol(value: object) -> None:
        try:
            sym = _normalize_tomorrow_symbol(value)
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
        except Exception:
            pass

    try:
        from prediction_feedback_store import read_feedback_log

        feedback_df = read_feedback_log()
        if isinstance(feedback_df, pd.DataFrame) and not feedback_df.empty:
            for _, row in feedback_df.iterrows():
                ret_missing = (
                    "actual_next_return_pct" not in feedback_df.columns
                    or _is_blank_outcome_value(row.get("actual_next_return_pct"))
                )
                correct_missing = (
                    "correct" not in feedback_df.columns
                    or not _is_valid_correct_value(row.get("correct"))
                )
                if ret_missing or correct_missing:
                    pending_stock += 1
                    _add_symbol(row.get("symbol", row.get("ticker", "")))
    except Exception:
        pass

    try:
        from sector_prediction_tracker import read_log as read_sector_log

        sector_df = read_sector_log()
        if isinstance(sector_df, pd.DataFrame) and not sector_df.empty:
            for _, row in sector_df.iterrows():
                exit_missing = "exit_price" not in sector_df.columns or _is_blank_outcome_value(row.get("exit_price"))
                ret_missing = "return_pct" not in sector_df.columns or _is_blank_outcome_value(row.get("return_pct"))
                correct_missing = "correct" not in sector_df.columns or not _is_valid_correct_value(row.get("correct"))
                if not (exit_missing or ret_missing or correct_missing):
                    continue
                pending_sector += 1
                _add_symbol(row.get("ohlc_symbol", ""))
                _add_symbol(row.get("leader_ticker", ""))
                try:
                    members = json.loads(str(row.get("stocks_used_json", "") or "[]"))
                except Exception:
                    members = []
                if isinstance(members, list):
                    for member in members:
                        _add_symbol(member)
    except Exception:
        pass

    limited = len(symbols) > int(limit or 0) > 0
    if limited:
        symbols = symbols[:limit]
    return {
        "symbols": symbols,
        "pending_stock": pending_stock,
        "pending_sector": pending_sector,
        "limited": limited,
    }


def _build_outcome_backfill_data(symbols: list[str]) -> dict[str, Any]:
    all_data: dict[str, Any] = {}
    try:
        from strategy_engines._engine_utils import ALL_DATA as ENGINE_ALL_DATA
    except Exception:
        ENGINE_ALL_DATA = {}

    if isinstance(ENGINE_ALL_DATA, dict):
        all_data.update(ENGINE_ALL_DATA)

    for raw_symbol in symbols:
        try:
            symbol = _normalize_tomorrow_symbol(raw_symbol)
            if not symbol:
                continue
            ticker_ns = symbol if symbol.startswith("^") or symbol.endswith(".NS") else f"{symbol}.NS"
            hist = None
            for key in (ticker_ns, symbol):
                try:
                    if isinstance(ENGINE_ALL_DATA, dict) and ENGINE_ALL_DATA.get(key) is not None:
                        hist = ENGINE_ALL_DATA.get(key)
                        break
                except Exception:
                    pass
            if hist is None:
                hist = None if symbol.startswith("^") else get_df_for_ticker(ticker_ns)
            if hist is None:
                continue
            all_data[symbol] = hist
            all_data[ticker_ns] = hist
        except Exception:
            continue
    return all_data


def _run_post_close_outcome_refresh(*, force: bool = False, allow_open_session: bool = False) -> dict[str, object]:
    """
    After market close, fill pending prediction outcomes without requiring a
    manual scan. This drives Imported AI's Last Outcome/Correct columns.
    """
    default = {
        "ran": False,
        "filled_stock": 0,
        "filled_sector": 0,
        "pending_stock": 0,
        "pending_sector": 0,
        "message": "",
        "checked_at": "",
    }
    try:
        if not allow_open_session and not _post_close_outcome_window():
            return default

        now = time.time()
        last_checked = float(st.session_state.get("_post_close_outcome_checked_at", 0.0) or 0.0)
        if not force and last_checked and (now - last_checked) < _POST_CLOSE_OUTCOME_INTERVAL_SEC:
            cached = st.session_state.get("_post_close_outcome_status")
            return cached if isinstance(cached, dict) else default
        st.session_state["_post_close_outcome_checked_at"] = now

        pending = _pending_outcome_symbols()
        symbols = list(pending.get("symbols", []) or [])
        status = dict(default)
        status["ran"] = True
        status["pending_stock"] = int(pending.get("pending_stock", 0) or 0)
        status["pending_sector"] = int(pending.get("pending_sector", 0) or 0)
        status["checked_at"] = datetime.now().isoformat(timespec="minutes")

        if not symbols:
            status["message"] = "Post-close validation: no pending logged predictions."
            st.session_state["_post_close_outcome_status"] = status
            return status

        outcome_data = _build_outcome_backfill_data(symbols)
        if not outcome_data:
            status["message"] = "Post-close validation: waiting for next close data."
            st.session_state["_post_close_outcome_status"] = status
            return status

        filled_stock = 0
        filled_sector = 0
        try:
            from prediction_feedback_store import backfill_actual_returns

            filled_stock = int(backfill_actual_returns(outcome_data) or 0)
        except Exception:
            filled_stock = 0
        try:
            from sector_prediction_tracker import backfill_outcomes

            filled_sector = int(backfill_outcomes(outcome_data) or 0)
        except Exception:
            filled_sector = 0

        status["filled_stock"] = filled_stock
        status["filled_sector"] = filled_sector
        total_filled = filled_stock + filled_sector
        if total_filled > 0:
            _refresh_feedback_learning_system(force=True)
            _refresh_signal_weight_status(force_update=True)
            status["message"] = f"Post-close validation filled {total_filled} outcome(s)."
        else:
            status["message"] = "Post-close validation checked; outcomes are waiting for next-session close."
        st.session_state["_post_close_outcome_status"] = status
        return status
    except Exception:
        return default


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
from strategy_engines.mode_registry import (
    get_mode_color,
    get_mode_colors,
    get_mode_display,
    get_mode_filter_rules,
    get_mode_label,
    get_mode_map,
    get_mode_metadata,
    get_mode_pill_classes,
)
from strategy_engines.mode_helpers import resolve_mode_id
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
        _scan_diag.record_failure(ticker, "EXCEPTION")
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

_SECTOR_EXPLORER_UI_OK = True
_SECTOR_EXPLORER_UI_ERR = ""
_sector_explorer_renderer = None


def render_sector_explorer_section(ticker_universe=None) -> None:  # type: ignore[misc]
    global _SECTOR_EXPLORER_UI_OK, _SECTOR_EXPLORER_UI_ERR, _sector_explorer_renderer
    if callable(_sector_explorer_renderer):
        return _sector_explorer_renderer(ticker_universe)
    try:
        try:
            from strategy_engines.app_sector_explorer_section import (  # type: ignore[import]
                render_sector_explorer_section as _render_sector_explorer_section,
            )
        except Exception:
            from app_sector_explorer_section import (  # type: ignore[import]
                render_sector_explorer_section as _render_sector_explorer_section,
            )
        _sector_explorer_renderer = _render_sector_explorer_section
        _SECTOR_EXPLORER_UI_OK = True
        _SECTOR_EXPLORER_UI_ERR = ""
        return _sector_explorer_renderer(ticker_universe)
    except Exception as exc:
        _SECTOR_EXPLORER_UI_OK = False
        _SECTOR_EXPLORER_UI_ERR = str(exc).strip() or "sector explorer import failed"
        st.warning(
            "Sector Explorer is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_EXPLORER_UI_ERR}"
        )
        return None


_SECTOR_PREDICTION_UI_OK = True
_SECTOR_PREDICTION_UI_ERR = ""
_sector_prediction_renderer = None


def render_sector_prediction_section(*args, **kwargs) -> None:  # type: ignore[misc]
    global _SECTOR_PREDICTION_UI_OK, _SECTOR_PREDICTION_UI_ERR, _sector_prediction_renderer
    if callable(_sector_prediction_renderer):
        return _sector_prediction_renderer(*args, **kwargs)
    try:
        from app_sector_prediction_section import (
            render_sector_prediction_section as _render_sector_prediction_section,
        )

        _sector_prediction_renderer = _render_sector_prediction_section
        _SECTOR_PREDICTION_UI_OK = True
        _SECTOR_PREDICTION_UI_ERR = ""
        return _sector_prediction_renderer(*args, **kwargs)
    except Exception as exc:
        _SECTOR_PREDICTION_UI_OK = False
        _SECTOR_PREDICTION_UI_ERR = str(exc).strip() or "sector prediction import failed"
        st.warning(
            "Sector Prediction is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_PREDICTION_UI_ERR}"
        )
        return None


_SECTOR_INTELLIGENCE_UI_OK = True
_SECTOR_INTELLIGENCE_UI_ERR = ""
_sector_intelligence_renderer = None


def render_sector_intelligence_section(*args, **kwargs) -> None:  # type: ignore[misc]
    global _SECTOR_INTELLIGENCE_UI_OK, _SECTOR_INTELLIGENCE_UI_ERR, _sector_intelligence_renderer
    if callable(_sector_intelligence_renderer):
        return _sector_intelligence_renderer(*args, **kwargs)
    try:
        try:
            from strategy_engines.app_sector_intelligence_section import (
                render_sector_intelligence_section as _render_sector_intelligence_section,
            )
        except Exception:
            from app_sector_intelligence_section import (  # type: ignore[import]
                render_sector_intelligence_section as _render_sector_intelligence_section,
            )
        _sector_intelligence_renderer = _render_sector_intelligence_section
        _SECTOR_INTELLIGENCE_UI_OK = True
        _SECTOR_INTELLIGENCE_UI_ERR = ""
        return _sector_intelligence_renderer(*args, **kwargs)
    except Exception as exc:
        _SECTOR_INTELLIGENCE_UI_OK = False
        _SECTOR_INTELLIGENCE_UI_ERR = str(exc).strip() or "sector intelligence import failed"
        st.warning(
            "Sector Intelligence is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_INTELLIGENCE_UI_ERR}"
        )
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


_prediction_chart_renderer = None


def render_prediction_chart_section(*args, **kwargs):
    """
    Lazy-load the Prediction Chart panel so Plotly and its helpers do not block
    the initial app shell render.
    """
    global _prediction_chart_renderer
    try:
        import importlib
        import app_prediction_chart_section as _prediction_chart_section_module

        _prediction_chart_section_module = importlib.reload(_prediction_chart_section_module)
        _render_prediction_chart_section = _prediction_chart_section_module.render_prediction_chart_section

        _prediction_chart_renderer = _render_prediction_chart_section
        return _prediction_chart_renderer(*args, **kwargs)
    except Exception:
        _prediction_chart_renderer = None
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
_IMPORTED_AI_STORE_PATH = Path(_HERE) / "data" / "imported_ai_learning_store.json"
_TICKER_MASTER_STORE_PATH = Path(_HERE) / "data" / "ticker_master_list.json"

_TOMORROW_SECTION_ORDER = ("relax", "swing", "intraday", "momentum", "breakout")
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
    "momentum": {
        "label": "Momentum",
        "accent": "#b08cff",
        "caption": "Mode 7 Momentum S&R and structure-first continuation",
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
    mode_int = resolve_mode_id(mode_value, 3) or 3
    return {
        1: "momentum",
        3: "relax",
        6: "swing",
        5: "intraday",
        7: "momentum",
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
        atomic_write_json(_TOMORROW_STORE_PATH, store, indent=2)
        _queue_persistent_file(_TOMORROW_STORE_PATH)
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
    local_store = _normalize_tomorrow_store(_load_local_tomorrow_store())
    sheets_ready = _get_sheet() is not None
    if sheets_ready:
        cloud_store = _normalize_tomorrow_store(_load_picks())
        storage_mode = "cloud"
        if cloud_store["picks"] or cloud_store["notes"]:
            store = cloud_store
            try:
                _save_local_tomorrow_store(store)
            except Exception:
                pass
        elif local_store["picks"] or local_store["notes"]:
            store = local_store
            _save_picks(store)
        else:
            store = session_store
            if store["picks"] or store["notes"]:
                _save_picks(store)
                try:
                    _save_local_tomorrow_store(store)
                except Exception:
                    pass
    else:
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
    try:
        _save_local_tomorrow_store(normalized)
    except Exception:
        pass
    if storage_mode == "cloud":
        try:
            _save_picks(normalized)
        except Exception:
            pass


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


def _clear_tomorrow_picks_store(store: dict) -> int:
    normalized = _normalize_tomorrow_store(store)
    removed_count = len(normalized.get("picks", []))
    normalized["sections"] = _tomorrow_section_defaults()
    normalized["picks"] = []
    _persist_tomorrow_store(normalized)
    return removed_count


def _set_tomorrow_picks_feedback(kind: str, message: str) -> None:
    st.session_state["tmr_picks_feedback"] = {
        "kind": str(kind or "info"),
        "message": str(message or "").strip(),
        "at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
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
        st.caption(helper_text)

    add_col, open_col = st.columns([1.15, 1], gap="small")
    with add_col:
        add_clicked = st.button(
            "ADD IN PICKS",
            key=f"{key_prefix}_add_in_picks",
            width="stretch",
            disabled=not normalized,
        )
    with open_col:
        open_clicked = st.button(
            "OPEN PICKS",
            key=f"{key_prefix}_open_picks",
            width="stretch",
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


def _normalize_prediction_chart_imports(values: list[object] | tuple[object, ...] | None, limit: int = 40) -> list[str]:
    return _normalize_tomorrow_symbols(values, limit=limit)


def _pretty_learning_signal_name(signal: object) -> str:
    try:
        text = str(signal or "").strip().replace("_", " ")
        return text.title() if text else "Momentum"
    except Exception:
        return "Momentum"


def _import_category_from_context(
    *,
    source_label: object = "",
    source_bucket: object = "",
    mode_value: object = 0,
) -> str:
    text = str(source_label or "").strip().lower()
    raw_bucket = str(source_bucket or "").strip()
    bucket_key = _normalize_tomorrow_bucket(raw_bucket) if raw_bucket else ""
    if "pulse" in text:
        return "Breakout Pulse"
    if "radar" in text and "csv" in text:
        return "Breakout Radar CSV"
    if "radar" in text:
        return "Breakout Radar"
    if bucket_key == "breakout":
        return "Breakout"
    if bucket_key:
        return _tomorrow_section_label(bucket_key)
    try:
        return _tomorrow_section_label(_tomorrow_bucket_for_mode(mode_value))
    except Exception:
        return "Imported"


def _normalize_import_mode_list(values: object) -> list[int]:
    modes: list[int] = []
    seen: set[int] = set()
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    for raw in list(raw_values):
        try:
            mode_int = int(raw)
        except Exception:
            continue
        if mode_int not in seen and mode_int >= 0:
            modes.append(mode_int)
            seen.add(mode_int)
    return modes


def _normalize_import_text_list(values: object) -> list[str]:
    items = values if isinstance(values, (list, tuple, set)) else [values]
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(items):
        text = str(raw or "").strip()
        if not text:
            continue
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _best_import_mode_for_records(
    records: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
    *,
    focus_symbol: object = "",
    fallback: object = 0,
) -> int:
    try:
        focus = _normalize_tomorrow_symbol(focus_symbol)
        if focus:
            for record in list(records or []):
                if _normalize_tomorrow_symbol(record.get("ticker")) != focus:
                    continue
                modes = [mode for mode in _normalize_import_mode_list(record.get("modes", [])) if mode > 0]
                if modes:
                    return int(modes[0])

        positive_modes: list[int] = []
        for record in list(records or []):
            positive_modes.extend(
                [mode for mode in _normalize_import_mode_list(record.get("modes", [])) if mode > 0]
            )
        if positive_modes:
            return int(positive_modes[0])
    except Exception:
        pass
    try:
        return int(fallback or 0)
    except Exception:
        return 0


def _sanitize_import_snapshot_value(value: object) -> object:
    try:
        if value is None:
            return ""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat(timespec="seconds")
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else ""
        if isinstance(value, str):
            return value
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        return str(value or "").strip()[:240]
    except Exception:
        return ""


def _build_import_snapshot_map(source_rows: object) -> dict[str, dict[str, object]]:
    snapshot_map: dict[str, dict[str, object]] = {}
    try:
        if isinstance(source_rows, pd.DataFrame):
            iterable = [row.to_dict() for _, row in source_rows.iterrows()]
        elif isinstance(source_rows, dict):
            iterable = [dict(source_rows)]
        elif isinstance(source_rows, (list, tuple)):
            iterable = []
            for item in source_rows:
                if isinstance(item, pd.Series):
                    iterable.append(item.to_dict())
                elif isinstance(item, dict):
                    iterable.append(dict(item))
        else:
            iterable = []

        for row in iterable:
            symbol = _normalize_tomorrow_symbol(row.get("Symbol") or row.get("Ticker"))
            if not symbol:
                continue
            snapshot: dict[str, object] = {}
            for key, raw_value in row.items():
                if key in {"_learn_symbol", "_panel_symbol"}:
                    continue
                snapshot[str(key)] = _sanitize_import_snapshot_value(raw_value)
            snapshot["Symbol"] = symbol
            snapshot_map[symbol] = snapshot
    except Exception:
        return {}
    return snapshot_map


def _legacy_imported_ai_learning_records() -> list[dict[str, object]]:
    symbols = _normalize_prediction_chart_imports(
        st.session_state.get("prediction_chart_imported_symbols", []),
        limit=40,
    )
    if not symbols:
        return []
    origin = str(st.session_state.get("prediction_chart_import_origin", "AI Prediction imports") or "AI Prediction imports")
    mode_value = st.session_state.get("prediction_chart_import_mode", st.session_state.get("mode", 0))
    category = _import_category_from_context(source_label=origin, mode_value=mode_value)
    timestamp = str(st.session_state.get("prediction_chart_imported_at", "") or "")
    records: list[dict[str, object]] = []
    for symbol in symbols:
        records.append(
            {
                "ticker": symbol,
                "categories": [category],
                "sources": [origin],
                "modes": _normalize_import_mode_list(mode_value),
                "last_imported_at": timestamp,
            }
        )
    return records


def _normalize_imported_ai_learning_records(raw: object) -> list[dict[str, object]]:
    records_map: dict[str, dict[str, object]] = {}
    raw_items = list(raw or []) if isinstance(raw, (list, tuple)) else []
    if not raw_items:
        raw_items = _legacy_imported_ai_learning_records()

    for item in raw_items:
        if isinstance(item, dict):
            ticker = _normalize_tomorrow_symbol(item.get("ticker") or item.get("symbol"))
            if not ticker:
                continue
            categories = _normalize_import_text_list(item.get("categories", item.get("category", [])))
            sources = _normalize_import_text_list(item.get("sources", item.get("source", [])))
            modes = _normalize_import_mode_list(item.get("modes", item.get("mode", [])))
            if 7 in modes:
                categories = _normalize_import_text_list(categories + ["Momentum"])
            imported_at = str(item.get("last_imported_at", item.get("imported_at", "")) or "")
            snapshot = _build_import_snapshot_map(item.get("snapshot", {})).get(ticker, {})
        else:
            ticker = _normalize_tomorrow_symbol(item)
            if not ticker:
                continue
            origin = str(st.session_state.get("prediction_chart_import_origin", "AI Prediction imports") or "AI Prediction imports")
            mode_value = st.session_state.get("prediction_chart_import_mode", st.session_state.get("mode", 0))
            categories = [_import_category_from_context(source_label=origin, mode_value=mode_value)]
            sources = [origin]
            modes = _normalize_import_mode_list(mode_value)
            if 7 in modes:
                categories = _normalize_import_text_list(categories + ["Momentum"])
            imported_at = str(st.session_state.get("prediction_chart_imported_at", "") or "")
            snapshot = {}

        record = records_map.setdefault(
            ticker,
            {
                "ticker": ticker,
                "categories": [],
                "sources": [],
                "modes": [],
                "last_imported_at": imported_at,
                "snapshot": {},
            },
        )
        record["categories"] = _normalize_import_text_list(list(record.get("categories", [])) + categories)
        record["sources"] = _normalize_import_text_list(list(record.get("sources", [])) + sources)
        record["modes"] = _normalize_import_mode_list(list(record.get("modes", [])) + modes)
        if imported_at:
            record["last_imported_at"] = imported_at
        if isinstance(snapshot, dict) and snapshot:
            record["snapshot"] = dict(snapshot)

    return list(records_map.values())


def _normalize_imported_ai_learning_payload(payload: object) -> dict[str, object]:
    default = {"records": [], "updated_at": ""}
    try:
        if not isinstance(payload, dict):
            return default.copy()
        records = _normalize_imported_ai_learning_records(payload.get("records", []))
        updated_at = str(payload.get("updated_at", "") or "")
        return {
            "records": records,
            "updated_at": updated_at,
        }
    except Exception:
        return default.copy()


def _load_local_imported_ai_learning_store() -> dict[str, object]:
    default = {"records": [], "updated_at": ""}
    try:
        if not _IMPORTED_AI_STORE_PATH.exists():
            return default
        payload = json.loads(_IMPORTED_AI_STORE_PATH.read_text(encoding="utf-8"))
        return _normalize_imported_ai_learning_payload(payload)
    except Exception:
        return default


def _save_local_imported_ai_learning_store(payload: dict[str, object]) -> bool:
    try:
        normalized = _normalize_imported_ai_learning_payload(payload)
        _IMPORTED_AI_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_IMPORTED_AI_STORE_PATH, normalized, indent=2)
        _queue_persistent_file(_IMPORTED_AI_STORE_PATH)
        return True
    except Exception:
        return False


def _load_imported_ai_learning_cloud_store() -> dict[str, object]:
    default = {"records": [], "updated_at": ""}
    try:
        ws = _get_sheet()
        if ws is None:
            return default
        records_json = ws.cell(2, 3).value or "[]"
        updated_at = ws.cell(2, 4).value or ""
        payload = json.loads(records_json)
        return _normalize_imported_ai_learning_payload(
            {
                "records": payload,
                "updated_at": updated_at,
            }
        )
    except Exception:
        return default


def _save_imported_ai_learning_cloud_store(payload: dict[str, object]) -> bool:
    try:
        ws = _get_sheet()
        if ws is None:
            return False
        normalized = _normalize_imported_ai_learning_payload(payload)
        if not ws.cell(1, 3).value:
            ws.update("C1:D1", [["imported_ai_records", "imported_ai_updated_at"]])
        ws.update(
            "C2:D2",
            [[
                json.dumps(normalized.get("records", []), ensure_ascii=False),
                str(normalized.get("updated_at", "") or ""),
            ]],
        )
        return True
    except Exception:
        return False


def _cache_imported_ai_learning_store_session(payload: dict[str, object], storage_mode: str) -> dict[str, object]:
    normalized = _normalize_imported_ai_learning_payload(payload)
    mode = "cloud" if str(storage_mode or "").strip().lower() == "cloud" else "local"
    st.session_state["imported_ai_learning_records"] = list(normalized.get("records", []))
    st.session_state["imported_ai_learning_updated_at"] = str(normalized.get("updated_at", "") or "")
    st.session_state["imported_ai_learning_storage_mode"] = mode
    st.session_state["imported_ai_learning_store_loaded"] = True
    return normalized


def _get_cached_imported_ai_learning_store() -> tuple[dict[str, object] | None, str]:
    if not bool(st.session_state.get("imported_ai_learning_store_loaded", False)):
        return None, ""
    payload = {
        "records": st.session_state.get("imported_ai_learning_records", []),
        "updated_at": st.session_state.get("imported_ai_learning_updated_at", ""),
    }
    mode = str(st.session_state.get("imported_ai_learning_storage_mode", "local") or "local")
    return _normalize_imported_ai_learning_payload(payload), ("cloud" if mode == "cloud" else "local")


def _load_imported_ai_learning_store(force_refresh: bool = False) -> tuple[dict[str, object], str]:
    cached_payload, cached_mode = _get_cached_imported_ai_learning_store()
    if cached_payload is not None and not force_refresh:
        return cached_payload, cached_mode or "local"

    session_payload = cached_payload or _normalize_imported_ai_learning_payload(
        {
            "records": st.session_state.get("imported_ai_learning_records", _legacy_imported_ai_learning_records()),
            "updated_at": st.session_state.get("imported_ai_learning_updated_at", ""),
        }
    )
    local_payload = _load_local_imported_ai_learning_store()
    sheets_ready = _get_sheet() is not None
    if sheets_ready:
        cloud_payload = _load_imported_ai_learning_cloud_store()
        if cloud_payload.get("records"):
            payload = cloud_payload
            storage_mode = "cloud"
            _save_local_imported_ai_learning_store(payload)
        elif local_payload.get("records"):
            payload = local_payload
            storage_mode = "cloud"
            _save_imported_ai_learning_cloud_store(payload)
        else:
            payload = session_payload
            storage_mode = "cloud"
            if payload.get("records"):
                _save_imported_ai_learning_cloud_store(payload)
                _save_local_imported_ai_learning_store(payload)
    else:
        if local_payload.get("records"):
            payload = local_payload
            storage_mode = "local"
        else:
            payload = session_payload
            storage_mode = "local"
            if payload.get("records"):
                _save_local_imported_ai_learning_store(payload)
    return _cache_imported_ai_learning_store_session(payload, storage_mode), storage_mode


def _persist_imported_ai_learning_store(records: list[dict[str, object]], *, updated_at: str | None = None) -> dict[str, object]:
    normalized_records = _normalize_imported_ai_learning_records(records)
    payload = {
        "records": normalized_records,
        "updated_at": (
            datetime.now().isoformat(timespec="seconds")
            if updated_at is None
            else str(updated_at or "")
        ),
    }
    storage_mode = "cloud" if _get_sheet() is not None else "local"
    normalized_payload = _cache_imported_ai_learning_store_session(payload, storage_mode)
    try:
        _save_local_imported_ai_learning_store(normalized_payload)
    except Exception:
        pass
    if storage_mode == "cloud":
        try:
            _save_imported_ai_learning_cloud_store(normalized_payload)
        except Exception:
            pass
    return normalized_payload


def _get_imported_ai_learning_records() -> list[dict[str, object]]:
    payload, _storage_mode = _load_imported_ai_learning_store()
    records = _normalize_imported_ai_learning_records(payload.get("records", []))
    if records:
        st.session_state["imported_ai_learning_records"] = records
    return records


def _sync_imported_ai_store_to_prediction_chart(*, focus_symbol: str | None = None) -> dict[str, object]:
    records = _get_imported_ai_learning_records()
    symbols = [str(record.get("ticker", "")).strip() for record in records if str(record.get("ticker", "")).strip()]
    symbols = _normalize_prediction_chart_imports(symbols, limit=40)
    if not symbols:
        return {"symbols": [], "mode": None, "origin": ""}

    focus = _normalize_tomorrow_symbol(focus_symbol) if focus_symbol else symbols[0]
    latest_record = records[-1] if records else {}
    latest_sources = _normalize_import_text_list(latest_record.get("sources", []))
    focus_mode = _best_import_mode_for_records(
        records,
        focus_symbol=focus,
        fallback=st.session_state.get("mode", 0),
    )

    st.session_state["prediction_chart_imported_symbols"] = symbols
    st.session_state["prediction_chart_import_origin"] = "Imported AI Stocks Basket"
    st.session_state["prediction_chart_import_mode"] = focus_mode
    st.session_state["prediction_chart_focus_symbol"] = focus
    st.session_state["pc_loaded_symbol"] = focus
    st.session_state["prediction_chart_imported_at"] = str(datetime.now().isoformat(timespec="seconds"))
    st.session_state["prediction_chart_import_context"] = {
        "latest_sources": latest_sources,
        "latest_modes": [focus_mode] if focus_mode > 0 else [],
        "count": len(symbols),
    }
    return {
        "symbols": symbols,
        "mode": focus_mode,
        "origin": st.session_state.get("prediction_chart_import_origin", ""),
    }


def _store_symbols_in_imported_ai_learning(
    symbols: list[object] | tuple[object, ...] | None,
    *,
    mode_value: object,
    source_label: str,
    source_bucket: object = "",
    source_rows: object = None,
) -> dict[str, object]:
    normalized = _normalize_prediction_chart_imports(symbols, limit=40)
    if not normalized:
        return {"symbols": [], "added": 0, "updated": 0, "mode": None, "source_label": str(source_label or ""), "category": ""}

    existing_records = _get_imported_ai_learning_records()
    snapshot_map = _build_import_snapshot_map(source_rows)
    records_map = {
        str(record.get("ticker", "")).strip(): dict(record)
        for record in existing_records
        if str(record.get("ticker", "")).strip()
    }
    ordered_symbols = [str(record.get("ticker", "")).strip() for record in existing_records if str(record.get("ticker", "")).strip()]
    category = _import_category_from_context(
        source_label=source_label,
        source_bucket=source_bucket,
        mode_value=mode_value,
    )
    source_text = str(source_label or category or "Imported AI Stocks")
    mode_list = _normalize_import_mode_list(mode_value)
    ts = datetime.now().isoformat(timespec="seconds")
    added = 0
    updated = 0
    for symbol in normalized:
        record = records_map.get(symbol)
        if record is None:
            records_map[symbol] = {
                "ticker": symbol,
                "categories": [category] if category else [],
                "sources": [source_text],
                "modes": mode_list,
                "last_imported_at": ts,
                "snapshot": dict(snapshot_map.get(symbol) or {}),
            }
            ordered_symbols.append(symbol)
            added += 1
        else:
            before_categories = list(record.get("categories", []))
            before_sources = list(record.get("sources", []))
            before_modes = list(record.get("modes", []))
            before_snapshot = dict(record.get("snapshot", {}) or {})
            record["categories"] = _normalize_import_text_list(before_categories + ([category] if category else []))
            record["sources"] = _normalize_import_text_list(before_sources + [source_text])
            record["modes"] = _normalize_import_mode_list(before_modes + mode_list)
            record["last_imported_at"] = ts
            if symbol in snapshot_map and snapshot_map.get(symbol):
                record["snapshot"] = dict(snapshot_map.get(symbol) or {})
            records_map[symbol] = record
            if (
                record.get("categories", []) != before_categories
                or record.get("sources", []) != before_sources
                or record.get("modes", []) != before_modes
                or dict(record.get("snapshot", {}) or {}) != before_snapshot
            ):
                updated += 1
        if len(ordered_symbols) >= 40:
            break

    merged_records = [records_map[symbol] for symbol in ordered_symbols if symbol in records_map][:40]
    _persist_imported_ai_learning_store(merged_records, updated_at=ts)
    return {
        "symbols": [str(record.get("ticker", "")).strip() for record in merged_records if str(record.get("ticker", "")).strip()],
        "added": added,
        "updated": updated,
        "mode": mode_list[0] if mode_list else st.session_state.get("mode", 0),
        "source_label": source_text,
        "category": category,
    }


def _log_imported_symbols_for_self_learning() -> dict[str, object]:
    result: dict[str, object] = {
        "symbols": [],
        "matched_symbols": [],
        "missing_symbols": [],
        "added": 0,
        "already_logged": 0,
        "message": "",
        "mode": 0,
    }
    try:
        imported_records = _get_imported_ai_learning_records()
        imported = [
            str(record.get("ticker", "")).strip()
            for record in imported_records
            if str(record.get("ticker", "")).strip()
        ]
        record_map = {
            str(record.get("ticker", "")).strip(): dict(record)
            for record in imported_records
            if str(record.get("ticker", "")).strip()
        }
        result["symbols"] = imported
        st.session_state["self_learning_imported_symbols"] = imported
        if not imported:
            result["message"] = "Add some stocks into Imported AI Stocks first."
            return result

        frames: list[pd.DataFrame] = []
        scan_df = st.session_state.get("last_scan_df")
        if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
            symbol_col = "Symbol" if "Symbol" in scan_df.columns else "Ticker" if "Ticker" in scan_df.columns else ""
            if symbol_col:
                scan_working = scan_df.copy()
                scan_working["_learn_symbol"] = scan_working[symbol_col].map(_normalize_tomorrow_symbol)
                scan_working = scan_working[scan_working["_learn_symbol"].isin(imported)].copy()
                if not scan_working.empty:
                    frames.append(scan_working)

        snapshot_rows: list[dict[str, object]] = []
        seen_snapshot_symbols: set[str] = set()
        for symbol in imported:
            record = record_map.get(symbol, {})
            snapshot = dict(record.get("snapshot", {}) or {})
            if not snapshot:
                continue
            snap_symbol = _normalize_tomorrow_symbol(snapshot.get("Symbol") or symbol)
            if not snap_symbol or snap_symbol in seen_snapshot_symbols:
                continue
            snapshot["_learn_symbol"] = snap_symbol
            snapshot_rows.append(snapshot)
            seen_snapshot_symbols.add(snap_symbol)
        if snapshot_rows:
            frames.append(pd.DataFrame(snapshot_rows))

        if not frames:
            result["message"] = "Imported stocks are saved, but no stored rows are available to log yet."
            return result

        working = pd.concat(frames, ignore_index=True, sort=False)
        working = working[working["_learn_symbol"].isin(imported)].copy()
        if working.empty:
            result["missing_symbols"] = imported
            result["message"] = "These imported stocks are not available in the saved basket data yet."
            return result

        working = working.drop_duplicates(subset=["_learn_symbol"], keep="first")
        matched_symbols = [sym for sym in working["_learn_symbol"].tolist() if sym]
        result["matched_symbols"] = matched_symbols
        result["missing_symbols"] = [sym for sym in imported if sym not in matched_symbols]

        mode_value = _best_import_mode_for_records(
            imported_records,
            fallback=st.session_state.get("mode", 0),
        )
        try:
            mode_int = int(mode_value)
        except Exception:
            mode_int = 0
        result["mode"] = mode_int

        def _record_categories(symbol: object) -> str:
            record = record_map.get(_normalize_tomorrow_symbol(symbol), {})
            categories = _normalize_import_text_list(record.get("categories", []))
            return " | ".join(categories)

        def _record_sources(symbol: object) -> str:
            record = record_map.get(_normalize_tomorrow_symbol(symbol), {})
            sources = _normalize_import_text_list(record.get("sources", []))
            return " | ".join(sources)

        def _record_mode(symbol: object) -> int:
            record = record_map.get(_normalize_tomorrow_symbol(symbol), {})
            modes = _normalize_import_mode_list(record.get("modes", []))
            return modes[0] if modes else mode_int

        def _record_logged_at(symbol: object) -> str:
            record = record_map.get(_normalize_tomorrow_symbol(symbol), {})
            raw = str(record.get("last_imported_at", "") or "").strip()
            if raw:
                return raw
            return datetime.now().isoformat(timespec="seconds")

        def _logged_date(value: object):
            try:
                parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
                if pd.isnull(parsed):
                    return get_expected_data_date()
                return parsed.date()
            except Exception:
                return get_expected_data_date()

        working["Import Category"] = working["_learn_symbol"].map(_record_categories)
        working["Import Source"] = working["_learn_symbol"].map(_record_sources)
        working["Import Mode"] = working["_learn_symbol"].map(_record_mode)
        working["Logged At"] = working["_learn_symbol"].map(_record_logged_at)
        working["_learn_logged_date"] = working["Logged At"].map(_logged_date)

        existing_keys: set[tuple[str, int, object]] = set()
        try:
            from prediction_feedback_store import read_feedback_log

            existing_log = read_feedback_log()
            if isinstance(existing_log, pd.DataFrame) and not existing_log.empty:
                existing = existing_log.copy()
                existing["_symbol_norm"] = existing.get("symbol", "").map(_normalize_tomorrow_symbol)
                existing["_mode_norm"] = pd.to_numeric(existing.get("mode", 0), errors="coerce").fillna(0).astype(int)
                existing["_logged_date"] = pd.to_datetime(existing.get("logged_at", ""), errors="coerce").dt.date
                existing_keys = {
                    (str(row.get("_symbol_norm", "") or ""), int(row.get("_mode_norm", 0) or 0), row.get("_logged_date"))
                    for _, row in existing.iterrows()
                    if str(row.get("_symbol_norm", "") or "")
                }
        except Exception:
            existing_keys = set()

        row_keys = working.apply(
            lambda row: (
                str(row.get("_learn_symbol", "") or ""),
                int(row.get("Import Mode", mode_int) or 0),
                row.get("_learn_logged_date"),
            ),
            axis=1,
        )
        to_log = working[[key not in existing_keys for key in row_keys]].copy()
        result["already_logged"] = int(len(working) - len(to_log))
        if to_log.empty:
            result["message"] = "These imported stocks are already being tracked by the self-learning engine."
            return result

        try:
            from prediction_feedback_store import log_scan_predictions

            log_scan_predictions(
                to_log.drop(columns=["_learn_symbol"], errors="ignore"),
                mode_int,
                st.session_state.get("market_bias_result"),
            )
            result["added"] = int(len(to_log))
        except Exception:
            result["message"] = "The learning log could not be updated right now."
            return result

        try:
            from prediction_feedback_store import feedback_summary as _feedback_summary

            _refresh_learning_after_prediction_log(_feedback_summary())
        except Exception:
            pass
        try:
            _run_post_close_outcome_refresh(force=True)
        except Exception:
            pass

        if result["added"]:
            result["message"] = (
                f"Added {result['added']} imported stock(s) to self-learning. "
                "They will train the model after the next-session outcome is available."
            )
        else:
            result["message"] = "Imported stocks were reviewed, but nothing new needed to be logged."
        return result
    except Exception:
        result["message"] = "Imported AI stocks could not be sent to self-learning."
        return result


def _refresh_imported_ai_last_outcomes() -> str:
    try:
        status = _run_post_close_outcome_refresh(force=True, allow_open_session=True)
    except Exception:
        status = {}
    try:
        from prediction_feedback_store import feedback_summary as _feedback_summary

        _refresh_learning_after_prediction_log(_feedback_summary())
    except Exception:
        pass

    filled = int(status.get("filled_stock", 0) or 0) + int(status.get("filled_sector", 0) or 0)
    pending = int(status.get("pending_stock", 0) or 0) + int(status.get("pending_sector", 0) or 0)
    if filled > 0:
        return f"Imported {filled} last outcome/correct row(s) from available next-session data."
    msg = str(status.get("message", "") or "").strip()
    if msg:
        return msg
    return f"Checked last outcomes by date. Pending rows: {pending}."


def _run_imported_ai_self_improve_update(*, include_outcome_refresh: bool = True) -> dict[str, object]:
    summary = _log_imported_symbols_for_self_learning()
    learning_message = str(summary.get("message", "") or "Self-learning update finished.").strip()
    outcome_message = ""
    if include_outcome_refresh:
        try:
            outcome_message = _refresh_imported_ai_last_outcomes()
        except Exception:
            outcome_message = ""

    parts = [msg for msg in (learning_message, outcome_message) if msg]
    summary["message"] = " ".join(parts) if parts else "Self-learning update finished."
    summary["outcome_message"] = outcome_message
    return summary


def _render_sidebar_imported_ai_learning_button() -> None:
    records = _get_imported_ai_learning_records()
    imported = [
        str(record.get("ticker", "")).strip()
        for record in records
        if str(record.get("ticker", "")).strip()
    ]
    count = len(imported)
    preview = ", ".join(imported[:4])
    if count > 4:
        preview = f"{preview} +{count - 4} more"
    categories = []
    for record in records:
        categories.extend(_normalize_import_text_list(record.get("categories", [])))
    categories = _normalize_import_text_list(categories)
    origin = ", ".join(categories[:3]) if categories else "Imported AI Stocks"
    if len(categories) > 3:
        origin = f"{origin} +{len(categories) - 3} more"
    helper_text = (
        f"Ready for self-learning: <b style='color:#ccd9e8;'>{count}</b><br>"
        f"<span style='color:#8ab4d8;'>{preview}</span>"
        if count
        else "Import stocks into Imported AI Stocks first, then use this permanently saved basket for self-learning from here."
    )
    st.markdown(
        '<div style="margin-top:14px;padding:12px 12px 10px 12px;'
        'border:1px solid rgba(0,212,255,0.20);border-radius:14px;'
        'background:linear-gradient(180deg, rgba(8,16,28,0.96), rgba(8,13,20,0.92));">'
        '<div style="font-size:10px;color:#4a6480;letter-spacing:1.3px;'
        'text-transform:uppercase;margin-bottom:6px;">Imported AI Learning</div>'
        f'<div style="font-size:11px;color:#4a6480;line-height:1.7;margin-bottom:10px;">'
        f'Source: <span style="color:#8ab4d8;">{origin}</span><br>{helper_text}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "🧠 Use Imported Stocks For Self-Learning",
        key="sidebar_imported_ai_learning_btn",
        width="stretch",
        disabled=not imported,
        type="secondary",
    ):
        summary = _run_imported_ai_self_improve_update()
        added = int(summary.get("added", 0) or 0)
        already_logged = int(summary.get("already_logged", 0) or 0)
        missing = int(len(list(summary.get("missing_symbols", []) or [])))
        message = str(summary.get("message", "") or "Self-learning update finished.")
        if added > 0:
            try:
                st.toast(message)
            except Exception:
                pass
            st.success(message)
        elif already_logged > 0 and added == 0:
            st.info(message)
        elif missing > 0:
            st.warning(message)
        else:
            st.info(message)


def _render_sidebar_imported_ai_learning_entry_button() -> None:
    records = _get_imported_ai_learning_records()
    imported = [
        str(record.get("ticker", "")).strip()
        for record in records
        if str(record.get("ticker", "")).strip()
    ]
    count = len(imported)
    preview = ", ".join(imported[:4])
    if count > 4:
        preview = f"{preview} +{count - 4} more"
    categories = []
    for record in records:
        categories.extend(_normalize_import_text_list(record.get("categories", [])))
    categories = _normalize_import_text_list(categories)
    origin = ", ".join(categories[:3]) if categories else "Imported AI Stocks"
    if len(categories) > 3:
        origin = f"{origin} +{len(categories) - 3} more"
    helper_text = (
        f"Stored imported stocks: <b style='color:#ccd9e8;'>{count}</b><br>"
        f"<span style='color:#8ab4d8;'>{preview}</span>"
        if count
        else "Import stocks into Imported AI Stocks from any Top 3 area, then open the permanently saved self-learning basket from here."
    )
    st.markdown(
        '<div style="margin-top:14px;padding:12px 12px 10px 12px;'
        'border:1px solid rgba(0,212,255,0.20);border-radius:14px;'
        'background:linear-gradient(180deg, rgba(8,16,28,0.96), rgba(8,13,20,0.92));">'
        '<div style="font-size:10px;color:#4a6480;letter-spacing:1.3px;'
        'text-transform:uppercase;margin-bottom:6px;">Imported AI Learning</div>'
        f'<div style="font-size:11px;color:#4a6480;line-height:1.7;margin-bottom:10px;">'
        f'Categories: <span style="color:#8ab4d8;">{origin}</span><br>{helper_text}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Open Imported AI Stocks",
        key="sidebar_imported_ai_learning_entry_btn",
        width="stretch",
        type="secondary",
    ):
        _activate_sidebar_panel("imported_ai_learning_show_panel")
    if st.button(
        "Refresh Last Outcome And Correct",
        key="sidebar_imported_ai_learning_refresh_outcome_btn",
        width="stretch",
        type="primary",
        disabled=not imported,
    ):
        summary = _run_imported_ai_self_improve_update()
        st.session_state["_imported_ai_outcome_refresh_msg"] = str(
            summary.get("message", "") or "Self-learning/outcome refresh finished."
        )
        _activate_sidebar_panel("imported_ai_learning_show_panel")


def _build_imported_ai_learning_panel_data() -> dict[str, object]:
    payload: dict[str, object] = {
        "imported": [],
        "origin": "Imported AI Stocks",
        "mode": st.session_state.get("mode", 0),
        "table": pd.DataFrame(),
        "matched_count": 0,
        "logged_today_count": 0,
        "validated_count": 0,
        "missing_symbols": [],
        "categories": [],
        "records": [],
        "storage_mode": "local",
        "updated_at": "",
    }
    try:
        store_payload, storage_mode = _load_imported_ai_learning_store()
        records = _normalize_imported_ai_learning_records(store_payload.get("records", []))
        payload["storage_mode"] = storage_mode
        payload["updated_at"] = str(store_payload.get("updated_at", "") or "")
        imported = [
            str(record.get("ticker", "")).strip()
            for record in records
            if str(record.get("ticker", "")).strip()
        ]
        categories = []
        for record in records:
            categories.extend(_normalize_import_text_list(record.get("categories", [])))
        payload["records"] = records
        payload["imported"] = imported
        payload["categories"] = _normalize_import_text_list(categories)
        if payload["categories"]:
            payload["origin"] = ", ".join(payload["categories"])
        payload["mode"] = _best_import_mode_for_records(
            records,
            fallback=payload.get("mode", st.session_state.get("mode", 0)),
        )
        if not imported:
            return payload

        record_map = {
            str(record.get("ticker", "")).strip(): dict(record)
            for record in records
            if str(record.get("ticker", "")).strip()
        }
        scan_df = st.session_state.get("last_scan_df")
        scan_map: dict[str, dict[str, object]] = {}
        if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
            symbol_col = "Symbol" if "Symbol" in scan_df.columns else "Ticker" if "Ticker" in scan_df.columns else ""
            if symbol_col:
                working = scan_df.copy()
                working["_panel_symbol"] = working[symbol_col].map(_normalize_tomorrow_symbol)
                working = working[working["_panel_symbol"].isin(imported)].copy()
                working = working.drop_duplicates(subset=["_panel_symbol"], keep="first")
                for _, row in working.iterrows():
                    symbol = _normalize_tomorrow_symbol(row.get("_panel_symbol"))
                    if symbol:
                        scan_map[symbol] = row.to_dict()
        snapshot_map: dict[str, dict[str, object]] = {
            str(record.get("ticker", "")).strip(): dict(record.get("snapshot", {}) or {})
            for record in records
            if str(record.get("ticker", "")).strip()
        }
        stored_symbols = {
            sym for sym in imported
            if sym in scan_map or bool(snapshot_map.get(sym))
        }
        payload["matched_count"] = int(len(stored_symbols))

        latest_log_map: dict[str, dict[str, object]] = {}
        latest_validated_log_map: dict[str, dict[str, object]] = {}
        today_logged: set[str] = set()
        try:
            from prediction_feedback_store import read_feedback_log

            log_df = read_feedback_log()
            if isinstance(log_df, pd.DataFrame) and not log_df.empty:
                log_work = log_df.copy()
                fallback_blank = pd.Series([""] * len(log_work), index=log_work.index, dtype=object)
                fallback_zero = pd.Series([0] * len(log_work), index=log_work.index, dtype=float)
                log_symbol_series = (
                    log_work.get("symbol")
                    if "symbol" in log_work.columns
                    else log_work.get("ticker", fallback_blank)
                )
                log_time_series = (
                    log_work.get("logged_at")
                    if "logged_at" in log_work.columns
                    else log_work.get("prediction_date", fallback_blank)
                )
                log_work["_panel_symbol"] = log_symbol_series.map(_normalize_tomorrow_symbol)
                log_work["_logged_dt"] = pd.to_datetime(log_time_series, errors="coerce")
                log_work["_logged_date"] = log_work["_logged_dt"].dt.date
                target_date = get_expected_data_date()
                today_logged = {
                    sym
                    for sym in log_work.loc[
                        log_work["_logged_date"] == target_date,
                        "_panel_symbol",
                    ].tolist()
                    if sym
                }
                latest = log_work.sort_values("_logged_dt").drop_duplicates("_panel_symbol", keep="last")
                for _, row in latest.iterrows():
                    symbol = _normalize_tomorrow_symbol(row.get("_panel_symbol"))
                    if symbol:
                        latest_log_map[symbol] = row.to_dict()
                valid_mask = log_work.get("correct", fallback_blank).astype(str).str.strip().isin(["True", "False"])
                latest_validated = (
                    log_work.loc[valid_mask]
                    .sort_values("_logged_dt")
                    .drop_duplicates("_panel_symbol", keep="last")
                )
                for _, row in latest_validated.iterrows():
                    symbol = _normalize_tomorrow_symbol(row.get("_panel_symbol"))
                    if symbol:
                        latest_validated_log_map[symbol] = row.to_dict()
                payload["logged_today_count"] = int(len(today_logged & set(imported)))
                payload["validated_count"] = int(len(set(imported) & set(latest_validated_log_map)))
        except Exception:
            latest_log_map = {}
            latest_validated_log_map = {}

        rows: list[dict[str, object]] = []
        for symbol in imported:
            scan_row = dict(scan_map.get(symbol) or {})
            snapshot_row = dict(snapshot_map.get(symbol) or {})
            source_row = scan_row if scan_row else snapshot_row
            latest_pending_or_current_log = dict(latest_log_map.get(symbol) or {})
            outcome_log_row = dict(latest_validated_log_map.get(symbol) or latest_pending_or_current_log)
            record = record_map.get(symbol, {})
            actual_return = _to_float(outcome_log_row.get("actual_next_return_pct"))
            correct_value = outcome_log_row.get("correct")
            correct_text = str(correct_value).strip()
            if correct_text.lower() in {"", "nan", "none"}:
                correct_text = "-"
            trap_value = (
                source_row.get("Trap Risk")
                if source_row.get("Trap Risk") is not None and str(source_row.get("Trap Risk")).strip().lower() not in {"", "nan", "none"}
                else source_row.get("Trap Check")
            )
            if trap_value is None or str(trap_value).strip().lower() in {"", "nan", "none"}:
                trap_value = source_row.get("Trap")
            record_categories = _normalize_import_text_list(record.get("categories", []))
            record_sources = _normalize_import_text_list(record.get("sources", []))
            record_modes = _normalize_import_mode_list(record.get("modes", []))
            visible_modes = [mode for mode in record_modes if mode > 0]
            stored_state = (
                "Live Scan" if symbol in scan_map
                else "Imported Snapshot" if snapshot_row
                else "Imported Only"
            )
            rows.append(
                {
                    "Ticker": symbol,
                    "Categories": " | ".join(record_categories) if record_categories else "-",
                    "Sources": " | ".join(record_sources[:3]) if record_sources else "-",
                    "Modes": " | ".join([f"M{mode}" for mode in visible_modes]) if visible_modes else "-",
                    "Imported At": str(record.get("last_imported_at", "") or "-").replace("T", " "),
                    "Stored Data": stored_state,
                    "Sector": str(source_row.get("Sector", "-") or "-"),
                    "Final Score": round(_safe(source_row.get("Final Score", np.nan), np.nan), 1) if source_row else np.nan,
                    "Pred Score": round(
                        _safe(
                            source_row.get(
                                "Prediction Score",
                                source_row.get(
                                    "Pred Score",
                                    source_row.get(
                                        "Next Day Prob",
                                        source_row.get("Tomorrow Pick Score", np.nan),
                                    ),
                                ),
                            ),
                            np.nan,
                        ),
                        1,
                    ) if source_row else np.nan,
                    "Signal": str(source_row.get("Signal", "-") or "-"),
                    "Conviction": str(
                        source_row.get(
                            "Conviction Tier",
                            source_row.get("Conviction", source_row.get("Confidence", source_row.get("Grade", "-"))),
                        ) or "-"
                    ),
                    "Trap": str(trap_value or "-"),
                    "Logged Today": "Yes" if symbol in today_logged else "No",
                    "Last Outcome": f"{actual_return:.2f}%" if actual_return is not None else "-",
                    "Correct": correct_text,
                }
            )

        payload["missing_symbols"] = [sym for sym in imported if sym not in stored_symbols]
        payload["table"] = pd.DataFrame(rows)
        return payload
    except Exception:
        return payload


def _build_imported_ai_top3_source_rows(records: list[dict[str, object]]) -> pd.DataFrame:
    imported = [
        str(record.get("ticker", "")).strip()
        for record in records
        if str(record.get("ticker", "")).strip()
    ]
    if not imported:
        return pd.DataFrame()

    scan_map: dict[str, dict[str, object]] = {}
    scan_df = st.session_state.get("last_scan_df")
    if isinstance(scan_df, pd.DataFrame) and not scan_df.empty:
        symbol_col = "Symbol" if "Symbol" in scan_df.columns else "Ticker" if "Ticker" in scan_df.columns else ""
        if symbol_col:
            working = scan_df.copy()
            working["_top3_symbol"] = working[symbol_col].map(_normalize_tomorrow_symbol)
            working = working[working["_top3_symbol"].isin(imported)].copy()
            working = working.drop_duplicates(subset=["_top3_symbol"], keep="first")
            for _, row in working.iterrows():
                symbol = _normalize_tomorrow_symbol(row.get("_top3_symbol"))
                if symbol:
                    scan_map[symbol] = row.drop(labels=["_top3_symbol"], errors="ignore").to_dict()

    rows: list[dict[str, object]] = []
    for record in records:
        symbol = str(record.get("ticker", "")).strip()
        if not symbol:
            continue
        snapshot = dict(record.get("snapshot", {}) or {})
        source_row = dict(scan_map.get(symbol) or snapshot)
        if not source_row:
            continue
        source_row.setdefault("Ticker", symbol)
        source_row.setdefault("Symbol", symbol)
        rows.append(source_row)

    return pd.DataFrame(rows)


def _render_imported_ai_top3_prompt_panel(panel: dict[str, object]) -> None:
    records = list(panel.get("records", []) or [])
    source_df = _build_imported_ai_top3_source_rows(records)

    with st.expander("NSE Sentinel Top 3 Picker", expanded=not source_df.empty):
        st.caption(
            "Uses the master prompt formula locally on the saved Imported AI rows. "
            "No new market data is fetched here."
        )
        tab_output, tab_prompt, tab_reality = st.tabs(
            ["Top 3 Output", "Master Prompt", "Self Improve Reality"]
        )

        with tab_output:
            if source_df.empty:
                st.info("No stored scan rows are available yet for the imported basket.")
            else:
                try:
                    from nse_sentinel_top3 import rank_top3_from_rows

                    top3 = rank_top3_from_rows(
                        source_df,
                        as_of=get_expected_data_date(),
                        market_context=st.session_state.get("market_bias_result"),
                    )
                    output_text = str(top3.get("text", "") or "")
                    st.code(output_text, language="text")
                    st.download_button(
                        "Download Top 3 Output",
                        data=output_text.encode("utf-8-sig"),
                        file_name=f"nse_sentinel_top3_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        key="imported_ai_top3_output_download",
                        width="stretch",
                    )
                except Exception as exc:
                    st.warning(f"Top 3 picker could not run right now: {exc}")

        with tab_prompt:
            try:
                from nse_sentinel_top3 import load_top3_prompt_text

                prompt_text = load_top3_prompt_text()
            except Exception:
                prompt_text = "NSE Sentinel Top 3 prompt is unavailable."
            st.download_button(
                "Download Master Prompt",
                data=prompt_text.encode("utf-8-sig"),
                file_name="nse_sentinel_top3_prompt.txt",
                mime="text/plain",
                key="imported_ai_top3_prompt_download",
                width="stretch",
            )
            st.text_area(
                "Reusable prompt",
                value=prompt_text,
                height=360,
                key="imported_ai_top3_prompt_text",
            )

        with tab_reality:
            logged_today = int(panel.get("logged_today_count", 0) or 0)
            if logged_today <= 0:
                st.warning(
                    "Logged Today is 0. Self Improve is currently a data logger, not an immediate trainer."
                )
            else:
                st.success(f"Logged Today: {logged_today}. These rows are now feeding the tracking log.")
            st.markdown(
                """
- Self Improve logs imported snapshots into the prediction feedback CSV.
- It records signal values, scores, and later outcomes for history.
- It does not retrain a model or change weights immediately.
- Meaningful model updates need enough validated next-session outcomes first.
                """.strip()
            )


def render_imported_ai_learning_panel() -> None:
    if not st.session_state.get("imported_ai_learning_show_panel", False):
        return

    try:
        _run_post_close_outcome_refresh()
    except Exception:
        pass

    panel = _build_imported_ai_learning_panel_data()
    imported = list(panel.get("imported", []) or [])
    table = panel.get("table")
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame()

    st.divider()
    _hdr_col, _close_col = st.columns([6, 1])
    with _hdr_col:
        st.header("Imported AI Stocks")
        st.caption(
            "This screen shows the imported AI stocks already saved in permanent storage, their saved scan or snapshot data, "
            "their import category/source, and their learning-log status. Use Self Improve here when you want this basket to feed the learning engine."
        )
    with _close_col:
        st.write("")
        if st.button("Close", key="imported_ai_learning_close_btn", width="stretch"):
            _activate_sidebar_panel(None)

    _mode_value = panel.get("mode", 0)
    try:
        _mode_int = int(_mode_value or 0)
    except Exception:
        _mode_int = 0
    _mode_label = f"Mode M{_mode_int}" if _mode_int > 0 else "Mixed / Imported"
    _category_text = ", ".join(list(panel.get("categories", []) or [])[:4]) or "Imported AI Stocks"
    _storage_mode = str(panel.get("storage_mode", "local") or "local")
    _storage_label = "Cloud + Local backup" if _storage_mode == "cloud" else "Local persistent store"
    _persistence_health = st.session_state.get("_persistence_health", {})
    _github_backup_label = (
        "GitHub backup ON"
        if isinstance(_persistence_health, dict) and _persistence_health.get("connected")
        else "GitHub backup OFF"
    )
    _updated_at = str(panel.get("updated_at", "") or "").strip()
    st.caption(
        f"Categories: {_category_text} | {_mode_label} | Storage: {_storage_label} | {_github_backup_label} | Imported stocks: {len(imported)}"
    )
    if _updated_at:
        st.caption(f"Last saved: {_updated_at}")
    _outcome_refresh_msg = st.session_state.pop("_imported_ai_outcome_refresh_msg", "")
    if _outcome_refresh_msg:
        st.info(str(_outcome_refresh_msg))

    _m1, _m2, _m3, _m4 = st.columns(4)
    with _m1:
        st.metric("Imported", f"{len(imported):,}")
    with _m2:
        st.metric("Stored Rows", f"{int(panel.get('matched_count', 0) or 0):,}")
    with _m3:
        st.metric("Logged Today", f"{int(panel.get('logged_today_count', 0) or 0):,}")
    with _m4:
        st.metric("Validated", f"{int(panel.get('validated_count', 0) or 0):,}")

    _action1, _action2, _action3, _action4 = st.columns(4)
    with _action1:
        _self_improve_clicked = st.button(
            "Self Improve",
            key="imported_ai_learning_self_improve_btn",
            width="stretch",
            type="primary",
            help="Logs this imported basket into the self-learning feedback log and refreshes available next-session outcomes.",
        )
    with _action2:
        if st.button("Open AI Prediction", key="imported_ai_learning_open_chart_btn", width="stretch"):
            summary = _sync_imported_ai_store_to_prediction_chart()
            if list(summary.get("symbols", []) or []):
                _activate_sidebar_panel("pred_chart_show_panel")
            else:
                st.info("No imported AI stocks are stored yet in the permanent basket.")
    with _action3:
        if st.button("Clear Imported", key="imported_ai_learning_clear_btn", width="stretch"):
            _persist_imported_ai_learning_store([], updated_at="")
            for key in (
                "imported_ai_learning_records",
                "imported_ai_learning_storage_mode",
                "imported_ai_learning_store_loaded",
                "self_learning_imported_symbols",
                "prediction_chart_imported_symbols",
                "prediction_chart_import_origin",
                "prediction_chart_import_mode",
                "prediction_chart_focus_symbol",
                "prediction_chart_imported_at",
                "prediction_chart_import_context",
                "imported_ai_learning_updated_at",
            ):
                st.session_state.pop(key, None)
            st.rerun()
    with _action4:
        if st.button("Back To Scanner", key="imported_ai_learning_back_btn", width="stretch"):
            _activate_sidebar_panel(None)

    if _self_improve_clicked:
        summary = _run_imported_ai_self_improve_update()
        added = int(summary.get("added", 0) or 0)
        already_logged = int(summary.get("already_logged", 0) or 0)
        missing = int(len(list(summary.get("missing_symbols", []) or [])))
        message = str(summary.get("message", "") or "Self-learning update finished.")
        if added > 0:
            try:
                st.toast(message)
            except Exception:
                pass
            st.success(message)
        elif already_logged > 0 and added == 0:
            st.info(message)
        elif missing > 0:
            st.warning(message)
        else:
            st.info(message)

    if not imported:
        st.info("No imported AI stocks are stored yet in the permanent basket. Add some symbols from a Mode Top 3, Breakout Radar, CSV Breakout, or Live Breakout Pulse panel first.")
        return

    missing_symbols = list(panel.get("missing_symbols", []) or [])
    if missing_symbols:
        _missing_preview = ", ".join(missing_symbols[:8])
        if len(missing_symbols) > 8:
            _missing_preview += " ..."
        st.caption(
            f"Stored row data is still missing for some imported symbols: {_missing_preview}"
        )

    _render_imported_ai_top3_prompt_panel(panel)

    if not table.empty:
        st.dataframe(
            table,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Categories": st.column_config.TextColumn("Categories", width="medium"),
                "Sources": st.column_config.TextColumn("Sources", width="large"),
                "Modes": st.column_config.TextColumn("Modes"),
                "Imported At": st.column_config.TextColumn("Imported At", width="medium"),
                "Stored Data": st.column_config.TextColumn("Stored Data"),
                "Sector": st.column_config.TextColumn("Sector"),
                "Final Score": st.column_config.NumberColumn("Final Score", format="%.1f"),
                "Pred Score": st.column_config.NumberColumn("Pred Score", format="%.1f"),
                "Signal": st.column_config.TextColumn("Signal"),
                "Conviction": st.column_config.TextColumn("Conviction"),
                "Trap": st.column_config.TextColumn("Trap"),
                "Logged Today": st.column_config.TextColumn("Logged Today"),
                "Last Outcome": st.column_config.TextColumn("Last Outcome"),
                "Correct": st.column_config.TextColumn("Correct"),
            },
            width="stretch",
            hide_index=True,
        )

    st.write("")
    if st.button(
        "Refresh Last Outcome And Correct",
        key="imported_ai_learning_refresh_last_outcome_wide_btn",
        width="stretch",
        type="primary",
        help="Fetch and fill the Last Outcome and Correct columns using the latest next-session data by imported date.",
    ):
        summary = _run_imported_ai_self_improve_update()
        st.session_state["_imported_ai_outcome_refresh_msg"] = str(
            summary.get("message", "") or "Self-learning/outcome refresh finished."
        )
        st.rerun()


def _build_mode_ai_top3_preview(
    symbols: list[object] | tuple[object, ...] | None,
    mode_value: object,
) -> dict[str, object]:
    normalized = _normalize_tomorrow_symbols(symbols, limit=6)
    default = {"table": pd.DataFrame(), "summary": {}}
    if not normalized:
        return default

    try:
        mode_int = int(mode_value)
    except Exception:
        mode_int = 0

    preview_sig = (
        tuple(normalized),
        mode_int,
        str(st.session_state.get("_learning_brain_signature", "")),
        str(st.session_state.get("_learning_refresh_sig", "")),
        str(st.session_state.get("tt_scan_date", "")),
    )
    cached = st.session_state.get("_mode_ai_top3_preview_cache")
    if isinstance(cached, dict) and cached.get("sig") == preview_sig:
        payload = cached.get("payload")
        if isinstance(payload, dict):
            return payload

    prediction_map: dict[str, dict] = {}
    try:
        from nse_learning_brain import get_cached_prediction, summarize_cached_predictions

        summary = summarize_cached_predictions(normalized)
        for symbol in normalized:
            pred = get_cached_prediction(symbol)
            if isinstance(pred, dict) and pred:
                prediction_map[symbol] = dict(pred)
    except Exception:
        summary = {}
        prediction_map = {}

    if len(prediction_map) < len(normalized):
        try:
            from strategy_engines._engine_utils import ALL_DATA as _ai_all_data
            from tomorrow_prediction_engine import summarize_tomorrow_predictions

            live_summary = summarize_tomorrow_predictions(normalized, _ai_all_data, mode_int)
            if isinstance(live_summary, dict) and live_summary:
                summary = live_summary
                for pred in list(live_summary.get("predictions", []) or []):
                    symbol = _normalize_tomorrow_symbol(pred.get("ticker"))
                    if symbol:
                        prediction_map[symbol] = dict(pred)
        except Exception:
            pass

    preview = default.copy()
    rows: list[dict[str, object]] = []
    for symbol in normalized:
        pred = dict(prediction_map.get(symbol) or {})
        rows.append(
            {
                "Ticker": symbol,
                "AI Direction": str(pred.get("direction", "Sideways") or "Sideways"),
                "AI Confidence": round(_safe(pred.get("confidence", 0.0), 0.0), 1),
                "AI Action": str(pred.get("action", "Wait") or "Wait").replace("???", "-"),
                "AI Risk": str(pred.get("risk", "MEDIUM") or "MEDIUM").upper(),
                "AI Hold": str(pred.get("hold_days", "-") or "-").replace("?", "-"),
                "AI Key Signal": _pretty_learning_signal_name(pred.get("key_signal", "momentum")),
                "AI Regime": str(pred.get("regime", "UNKNOWN") or "UNKNOWN").replace("_", " ").title(),
            }
        )

    preview = {
        "table": pd.DataFrame(rows),
        "summary": summary if isinstance(summary, dict) else {},
    }
    st.session_state["_mode_ai_top3_preview_cache"] = {"sig": preview_sig, "payload": preview}
    return preview


def _render_ai_prediction_import_action(
    symbols: list[object] | tuple[object, ...] | None,
    *,
    mode_value: object,
    key_prefix: str,
    source_label: str,
    source_bucket: object = "",
    source_rows: object = None,
    helper_text: str = "",
) -> None:
    normalized = _normalize_prediction_chart_imports(symbols, limit=12)
    if helper_text:
        st.caption(helper_text)

    import_clicked = st.button(
        "ADD TO IMPORTED AI STOCKS",
        key=f"{key_prefix}_import_ai_prediction",
        width="stretch",
        disabled=not normalized,
        type="primary",
    )
    if not import_clicked:
        return

    if not normalized:
        st.info("No valid stock is available to import right now.")
        return

    summary = _store_symbols_in_imported_ai_learning(
        normalized,
        mode_value=mode_value,
        source_label=source_label,
        source_bucket=source_bucket,
        source_rows=source_rows,
    )
    imported = list(summary.get("symbols", []) or [])
    if imported:
        tracking_summary = _run_imported_ai_self_improve_update()
        st.session_state["_imported_ai_outcome_refresh_msg"] = str(
            tracking_summary.get("message", "") or "Imported stocks were added to self-learning."
        )
        try:
            st.toast(
                f"Added {len(normalized)} stock(s) to Imported AI Stocks "
                f"under {summary.get('category', 'Imported')}."
            )
        except Exception:
            pass
        _activate_sidebar_panel("imported_ai_learning_show_panel")
    else:
        st.info("The imported AI stocks list is empty right now.")

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
_grading_module = _load_optional_module("grading_engine", ("apply_universal_grading",))
if _grading_module is not None:
    apply_universal_grading = getattr(_grading_module, "apply_universal_grading")
    _GRADING_OK = True
else:
    _GRADING_OK = False

    def apply_universal_grading(df, market_bias=None):  # type: ignore[misc]
        return df

# ── Enhanced logic engine (optional, graceful if missing) ─────────────
_enhanced_logic_module = _load_optional_module("enhanced_logic_engine", ("apply_enhanced_logic",))
if _enhanced_logic_module is not None:
    apply_enhanced_logic = getattr(_enhanced_logic_module, "apply_enhanced_logic")
    _ENHANCED_LOGIC_OK = True
else:
    _ENHANCED_LOGIC_OK = False

    def apply_enhanced_logic(df):  # type: ignore[misc]
        return df

# ── Phase 4 logic engine (optional, graceful if missing) ──────────────
_phase4_logic_module = _load_optional_module(
    "phase4_logic_engine",
    ("apply_phase4_logic", "apply_phase42_logic"),
)
if _phase4_logic_module is not None:
    apply_phase4_logic = getattr(_phase4_logic_module, "apply_phase4_logic")
    apply_phase42_logic = getattr(_phase4_logic_module, "apply_phase42_logic")
    _PHASE4_LOGIC_OK = True
else:
    _PHASE4_LOGIC_OK = False

    def apply_phase4_logic(df, market_bias=None):  # type: ignore[misc]
        return df

    def apply_phase42_logic(df):  # type: ignore[misc]
        return df

# ── Time Travel engine (optional, graceful if missing) ────────────────
_tt = _load_optional_module("time_travel_engine")
if _tt is not None:
    _TIME_TRAVEL_OK = True
else:
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


def _aura_time_travel_cutoff():
    try:
        from feature_data_manager import get_time_travel_date as _feature_tt_date
        cutoff = _feature_tt_date()
        if cutoff is not None:
            return pd.to_datetime(cutoff).date()
    except Exception:
        pass
    try:
        cutoff = _tt.get_reference_date()
        if cutoff is not None:
            return pd.to_datetime(cutoff).date()
    except Exception:
        pass
    try:
        cutoff = st.session_state.get("aura_tt_date") or st.session_state.get("tt_date_val")
        if cutoff is not None:
            return pd.to_datetime(cutoff).date()
    except Exception:
        pass
    return None


def _aura_fetch(symbol: str) -> "pd.DataFrame | None":
    """Fetch OHLCV; always truncates to TT cutoff if active — no leakage."""
    ticker_ns = symbol.upper().strip()
    if not ticker_ns.endswith(".NS"):
        ticker_ns += ".NS"

    cutoff = _aura_time_travel_cutoff()

    def _cut(df):
        if df is None or df.empty or cutoff is None:
            return df
        try:
            if _TIME_TRAVEL_OK and hasattr(_tt, "truncate_df"):
                return _tt.truncate_df(df, cutoff, min_rows=10)
            return None
        except Exception:
            return None

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
        if truncated is not None and cutoff is not None:
            try:
                import time_travel_engine as _tt_cache

                _tt_cache.cache_frame(ticker_ns, df, cutoff, min_rows=10)
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
    """Render the Stock Aura panel with persistent result actions."""
    if not st.session_state.get("aura_show_panel", False):
        return

    st.divider()
    st.subheader("Stock Aura")
    st.caption("Single-stock decision engine with a direct add-to-picks action.")

    aura_tt = st.session_state.get("aura_tt_date")
    aura_tt_str = ""
    if aura_tt is not None:
        try:
            aura_tt_str = aura_tt.strftime("%d %b %Y")
        except Exception:
            aura_tt_str = str(aura_tt)
        st.warning(f"TIME TRAVEL ACTIVE - Evaluating {aura_tt_str} post-market. No future data used.")

    _aura_cache_key = str(aura_tt or "live")
    if st.session_state.get("aura_tt_cache_key") != _aura_cache_key:
        st.session_state.pop("aura_last_payload", None)
        st.session_state.pop("aura_last_error", None)
        st.session_state["aura_tt_cache_key"] = _aura_cache_key

    configure_nse_stock_search(_get_cached_nse_tickers())
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
                    width="stretch",
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
            st.caption(f"Market note: {market_note}")

    _render_add_in_picks_actions(
        [symbol],
        key_prefix=f"aura_result_{symbol or 'stock'}",
        scope_label="Stock Aura",
        bucket=_tomorrow_bucket_for_mode(st.session_state.get("mode", 3)),
        helper_text="Add this Stock Aura result into Tomorrow's Picks and keep it saved until you delete it.",
    )

    st.caption("Stock Aura is for educational purposes only. Not financial advice.")





_STOCK_AURA_OK = True   # always True — no external dependency

# ── optional sklearn (graceful fallback if missing) ───────────────────
_SKLEARN_OK = _importlib.util.find_spec("sklearn") is not None

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
_NIFTY_COMPUTING = False

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
        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        df = _engine_utils.ALL_DATA.get(ticker_ns) if _engine_utils is not None else None
        if df is not None and not df.empty and {"Close", "Volume"}.issubset(df.columns):
            close = df["Close"].dropna()
            volume = df["Volume"].dropna()
            if not close.empty and not volume.empty:
                lc = float(close.iloc[-1])
                av = float(volume.tail(20).mean())
                mc_cr = round(lc * av * 250 / 1e7, 2)
                with _MKT_LOCK:
                    _MKT_CACHE[ticker] = mc_cr
                try:
                    st.session_state.setdefault("_mkt_cache_store", {})[ticker] = mc_cr
                except Exception:
                    pass
                return mc_cr
    except Exception:
        pass
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
    global _NIFTY_20D_RET, _NIFTY_COMPUTING
    with _NIFTY_LOCK:
        if _NIFTY_20D_RET is not None:
            return _NIFTY_20D_RET
        if _NIFTY_COMPUTING:
            return None
        _NIFTY_COMPUTING = True
    try:
        try:
            if _engine_utils is not None:
                for tk in ("^NSEI", "NIFTY_50.NS", "%5ENSEI"):
                    df_pre = _engine_utils.ALL_DATA.get(tk)
                    if df_pre is None or len(df_pre) < 25 or "Close" not in df_pre.columns:
                        continue
                    if _TIME_TRAVEL_OK and hasattr(_tt, "apply_time_travel_cutoff"):
                        df_pre = _tt.apply_time_travel_cutoff(df_pre)
                    close_pre = df_pre["Close"].dropna()
                    if len(close_pre) >= 21:
                        base = float(close_pre.iloc[-21])
                        if base <= 0:
                            continue
                        ret = float(close_pre.iloc[-1] / base - 1.0)
                        with _NIFTY_LOCK:
                            _NIFTY_20D_RET = ret
                        return ret
        except Exception:
            pass
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
    finally:
        with _NIFTY_LOCK:
            _NIFTY_COMPUTING = False


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
    picks_feedback = st.session_state.get("tmr_picks_feedback", {}) or {}
    if "tmr_picks_feedback" in st.session_state:
        del st.session_state["tmr_picks_feedback"]
    picks_feedback_kind = str(picks_feedback.get("kind", "info") or "info").strip().lower()
    picks_feedback_msg = str(picks_feedback.get("message", "") or "").strip()
    picks_feedback_at = str(picks_feedback.get("at", "") or "").strip()
    if picks_feedback_kind not in {"success", "error", "info"}:
        picks_feedback_kind = "info"
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
        "Google Sheets keeps your strategy strips synced across sessions."
        if storage_mode == "cloud"
        else "This machine stores your strategy strips permanently until removal."
    )
    storage_label = "Cloud Sync" if storage_mode == "cloud" else "Local Save"
    default_bucket = _tomorrow_bucket_for_mode(st.session_state.get("mode", 3))
    default_bucket_index = (
        _TOMORROW_SECTION_ORDER.index(default_bucket)
        if default_bucket in _TOMORROW_SECTION_ORDER
        else 0
    )

    st.divider()
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
        .tmr-v2-bulk-actions {
          border:1px solid rgba(255,77,109,0.30);
          border-radius:16px;
          padding:14px;
          margin:0 0 18px 0;
          background:linear-gradient(180deg, rgba(43,18,28,0.64), rgba(8,13,21,0.98));
        }
        .tmr-v2-bulk-title {
          font-size:13px;
          font-weight:800;
          color:#f4f8ff;
          margin-bottom:4px;
        }
        .tmr-v2-bulk-caption {
          font-size:11px;
          line-height:1.45;
          color:#9fb8d0;
        }
        .tmr-v2-action-status {
          border-radius:14px;
          padding:12px 14px;
          margin:0 0 14px 0;
          border:1px solid rgba(59,90,120,0.38);
          font-size:12px;
          line-height:1.5;
          background:linear-gradient(180deg, rgba(10,17,26,0.96), rgba(8,13,20,0.98));
          color:#dce7f4;
        }
        .tmr-v2-action-status strong {
          color:#f4f8ff;
        }
        .tmr-v2-action-status-success {
          border-color:rgba(0,212,168,0.34);
          background:linear-gradient(180deg, rgba(8,36,30,0.86), rgba(7,16,23,0.98));
        }
        .tmr-v2-action-status-error {
          border-color:rgba(255,77,109,0.34);
          background:linear-gradient(180deg, rgba(48,18,25,0.88), rgba(11,14,22,0.98));
        }
        .tmr-v2-action-status-info {
          border-color:rgba(240,180,41,0.30);
          background:linear-gradient(180deg, rgba(50,39,14,0.82), rgba(10,15,24,0.98));
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
                'Dedicated strips keep Relax, Swing, Intraday, Momentum, and Breakout ideas separate. '
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
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Saved Picks</div><div class="tmr-v2-metric-value">{saved_count}</div><div class="tmr-v2-metric-caption">Total stocks saved across all strategy strips</div></div>'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Slots Left</div><div class="tmr-v2-metric-value">{slots_left}</div><div class="tmr-v2-metric-caption">Maximum 20 saved picks in total</div></div>'
                f'<div class="tmr-v2-metric"><div class="tmr-v2-metric-label">Active Strips</div><div class="tmr-v2-metric-value">{active_sections}/{len(_TOMORROW_SECTION_ORDER)}</div><div class="tmr-v2-metric-caption">How many strategy lanes currently hold picks</div></div>'
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

        if picks_feedback_msg:
            picks_status_html = (
                f'<div class="tmr-v2-action-status tmr-v2-action-status-{picks_feedback_kind}">'
                f'<strong>{html.escape(picks_feedback_msg)}</strong>'
                f'{f"<br><span>Checked at {html.escape(picks_feedback_at)}</span>" if picks_feedback_at else ""}'
                '</div>'
            )
            st.markdown(picks_status_html, unsafe_allow_html=True)

        bulk_text_col, bulk_button_col = st.columns([2.7, 1], gap="small")
        with bulk_text_col:
            st.markdown(
                (
                    '<div class="tmr-v2-bulk-actions">'
                    '<div class="tmr-v2-bulk-title">Bulk Remove</div>'
                    f'<div class="tmr-v2-bulk-caption">Delete all {saved_count} saved stock'
                    f'{"s" if saved_count != 1 else ""} from every strip in one click. Notes stay saved.</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
        with bulk_button_col:
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            clear_all_clicked = st.button(
                "Delete All Stocks",
                key="tmr_v2_clear_all_stocks",
                width="stretch",
                disabled=saved_count <= 0,
            )

        if clear_all_clicked:
            removed_count = _clear_tomorrow_picks_store(store)
            if removed_count > 0:
                removed_noun = "stock" if removed_count == 1 else "stocks"
                _set_tomorrow_picks_feedback(
                    "success",
                    f"Deleted {removed_count} saved {removed_noun} from Tomorrow's Picks.",
                )
            else:
                _set_tomorrow_picks_feedback("info", "There were no saved stocks to delete.")
            st.rerun()

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
                            width="stretch",
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
                                width="stretch",
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
                        'Notes stay linked to the five-strip Tomorrow Picks board.'
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
                            width="stretch",
                        )
                    with notes_reset_col:
                        reset_notes_clicked = st.form_submit_button(
                            "Revert Saved",
                            width="stretch",
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
                    f'<div class="tmr-v2-notes-helper">Words saved: {notes_words}. Tip: keep strip-specific notes here so Relax, Swing, Intraday, Momentum, and Breakout decisions stay in one place.</div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        st.button(
            "Close",
            key="tmr_picks_close_btn_v2",
            width="stretch",
            on_click=_close_tomorrow_picks_panel,
        )


def render_tomorrow_picks_ticker_strip(*, embedded: bool = False) -> None:
    store, _storage_mode = _load_tomorrow_store()
    sections = _apply_tomorrow_sections_limit(store.get("sections", {}), limit=20)
    container_margin_top = "0px" if embedded else "-12px"
    container_margin_bottom = "14px" if embedded else "2px"
    shell_margin = "0 0 18px 0" if embedded else "0 0 8px 0"
    mobile_container_margin_top = "0px" if embedded else "-18px"
    mobile_shell_margin = "0 0 14px 0" if embedded else "-8px 0 8px 0"

    st.markdown(
        """
        <style>
        div[data-testid="stElementContainer"]:has(.tmr-board-shell) {
          margin-top:__CONTAINER_MARGIN_TOP__ !important;
          margin-bottom:__CONTAINER_MARGIN_BOTTOM__ !important;
        }
        .tmr-board-shell {
          border:1px solid rgba(86,118,150,0.34);
          border-radius:18px;
          padding:0;
          background:
            radial-gradient(circle at top right, rgba(240,180,41,0.12), transparent 26%),
            linear-gradient(180deg, rgba(10,17,27,0.97), rgba(6,10,17,0.99));
          box-shadow:
            0 14px 28px rgba(0,0,0,0.16),
            inset 0 0 0 1px rgba(255,255,255,0.02);
          margin:__SHELL_MARGIN__;
          overflow:hidden;
        }
        .tmr-board-shell > summary {
          list-style:none;
        }
        .tmr-board-shell > summary::-webkit-details-marker {
          display:none;
        }
        .tmr-board-summary {
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:10px;
          flex-wrap:wrap;
          min-height:48px;
          padding:12px 14px;
          cursor:pointer;
          user-select:none;
          outline:none;
        }
        .tmr-board-summary:focus-visible {
          box-shadow:inset 0 0 0 2px rgba(240,180,41,0.46);
        }
        .tmr-board-title {
          font-family:'Syne',sans-serif;
          font-size:16px;
          font-weight:800;
          letter-spacing:0.8px;
          text-transform:uppercase;
          color:#f0b429;
        }
        .tmr-board-shutter {
          display:inline-flex;
          align-items:center;
          gap:8px;
          min-height:26px;
          padding:4px 8px;
          border-radius:999px;
          border:1px solid rgba(240,180,41,0.38);
          background:rgba(240,180,41,0.08);
          color:#f0b429;
          font-size:10px;
          font-weight:800;
          letter-spacing:0.6px;
          text-transform:uppercase;
        }
        .tmr-board-toggle-text::after {
          content:"Show";
        }
        .tmr-board-shell[open] .tmr-board-toggle-text::after {
          content:"Hide";
        }
        .tmr-board-arrow {
          display:inline-flex;
          align-items:center;
          justify-content:center;
          width:16px;
          height:16px;
        }
        .tmr-board-arrow::before {
          content:">";
          transform:rotate(0deg);
          transition:transform 0.16s ease;
        }
        .tmr-board-shell[open] .tmr-board-arrow::before {
          transform:rotate(90deg);
        }
        .tmr-board-body {
          padding:0 14px 10px 14px;
        }
        .tmr-board-copy {
          margin:-2px 0 8px 0;
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
        .tmr-board-copy + .tmr-board-row {
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
            margin-top:__MOBILE_CONTAINER_MARGIN_TOP__ !important;
          }
          .tmr-board-shell {
            margin:__MOBILE_SHELL_MARGIN__;
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
        """.replace("__CONTAINER_MARGIN_TOP__", container_margin_top)
        .replace("__CONTAINER_MARGIN_BOTTOM__", container_margin_bottom)
        .replace("__SHELL_MARGIN__", shell_margin)
        .replace("__MOBILE_CONTAINER_MARGIN_TOP__", mobile_container_margin_top)
        .replace("__MOBILE_SHELL_MARGIN__", mobile_shell_margin),
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
            '<details class="tmr-board-shell">'
            '<summary class="tmr-board-summary" aria-label="Toggle Tomorrow\'s Picks">'
            '<span class="tmr-board-title">Tomorrow\'s Picks</span>'
            '<span class="tmr-board-shutter" aria-hidden="true">'
            '<span class="tmr-board-toggle-text"></span>'
            '<span class="tmr-board-arrow"></span>'
            '</span>'
            '</summary>'
            '<div class="tmr-board-body">'
            '<div class="tmr-board-copy">Compact 5-lane view: Relax, Swing, Intraday, Momentum, Breakout.</div>'
            + "".join(rows_html)
            + '</div>'
            + '</details>'
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
.pill-m7 { color:#b08cff;background:rgba(176,140,255,0.12);box-shadow:inset 0 0 0 1px rgba(176,140,255,0.22); }
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
        atomic_write_json(_TICKER_MASTER_STORE_PATH, payload, indent=2)
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


class _DegradedTickerUniverse(Exception):
    pass


def _fetch_nse_tickers_uncached() -> tuple[list[str], bool]:
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
        st.session_state["_ticker_universe_degraded"] = len(best_known) < _TICKER_GOOD_COUNT
        if len(best_known) >= _TICKER_GOOD_COUNT:
            _save_local_ticker_master(best_known)
        _prev_gs_count = int(st.session_state.get("_gsheets_saved_ticker_count", 0) or 0)
        if len(best_known) >= _TICKER_GOOD_COUNT and len(best_known) > _prev_gs_count:
            _save_tickers_to_gsheets(best_known)
            st.session_state["_gsheets_saved_ticker_count"] = len(best_known)

    degraded = len(best_known) < _TICKER_GOOD_COUNT
    return best_known, degraded


@st.cache_data(ttl=43200, show_spinner=False)
def _fetch_authoritative_nse_tickers() -> list[str]:
    tickers, degraded = _fetch_nse_tickers_uncached()
    if degraded:
        raise _DegradedTickerUniverse
    return tickers


def fetch_nse_tickers() -> list[str]:
    try:
        tickers = _fetch_authoritative_nse_tickers()
        st.session_state["_ticker_universe_degraded"] = False
        return tickers
    except _DegradedTickerUniverse:
        tickers, degraded = _fetch_nse_tickers_uncached()
        st.session_state["_ticker_universe_degraded"] = degraded
        return tickers
    except Exception:
        st.session_state["_ticker_universe_degraded"] = True
        return []


def _get_cached_nse_tickers(*, show_spinner: bool = False, force_refresh: bool = False) -> list[str]:
    try:
        if force_refresh:
            try:
                _fetch_authoritative_nse_tickers.clear()
            except Exception:
                pass
            try:
                from nse_ticker_universe import invalidate_cache as _invalidate_universe_cache

                try:
                    _invalidate_universe_cache(clear_disk=True)
                except TypeError:
                    _invalidate_universe_cache()
            except Exception:
                pass
            st.session_state.pop("_ui_all_tickers", None)
            st.session_state.pop("_ticker_master_list", None)
        cached = st.session_state.get("_ui_all_tickers", [])
        if isinstance(cached, list) and cached:
            return list(cached)
    except Exception:
        pass

    try:
        if show_spinner:
            with st.spinner("Loading NSE ticker list..."):
                tickers = fetch_nse_tickers()
        else:
            tickers = fetch_nse_tickers()
    except Exception:
        tickers = []

    try:
        if isinstance(tickers, list) and tickers:
            st.session_state["_ui_all_tickers"] = list(tickers)
    except Exception:
        pass
    return list(tickers or [])


def _invalidate_sidebar_data_status_cache() -> None:
    try:
        st.session_state["_data_status_cache_version"] = int(
            st.session_state.get("_data_status_cache_version", 0) or 0
        ) + 1
        st.session_state.pop("_sidebar_data_status", None)
        st.session_state.pop("_sidebar_data_status_sig", None)
    except Exception:
        pass


def _get_sidebar_data_status(tickers: list[str]) -> dict:
    default = {"fresh": 0, "stale": 0, "missing": 0, "total": len(tickers or [])}
    try:
        tickers_ns = [
            str(t).strip().upper() if str(t).strip().upper().endswith(".NS")
            else f"{str(t).strip().upper()}.NS"
            for t in (tickers or [])
            if str(t).strip()
        ]
        version = int(st.session_state.get("_data_status_cache_version", 0) or 0)
        sig = (
            version,
            len(tickers_ns),
            tuple(tickers_ns[:3]),
            tuple(tickers_ns[-3:]) if len(tickers_ns) >= 3 else tuple(tickers_ns),
            str(get_expected_data_date()),
        )
        cached_sig = st.session_state.get("_sidebar_data_status_sig")
        cached_status = st.session_state.get("_sidebar_data_status")
        if cached_sig == sig and isinstance(cached_status, dict):
            return cached_status

        data_dir = Path(_HERE) / "data"
        now_ts = time.time()
        max_staleness_hours = 24.0
        file_mtimes: dict[str, float] = {}
        if data_dir.exists():
            try:
                for entry in _os.scandir(data_dir):
                    if entry.is_file() and entry.name.lower().endswith(".csv"):
                        file_mtimes[entry.name[:-4].upper()] = float(entry.stat().st_mtime)
            except Exception:
                file_mtimes = {}

        fresh = 0
        stale = 0
        missing = 0
        for ticker_ns in tickers_ns:
            safe_name = ticker_ns.replace(":", "_").replace("/", "_").upper()
            mtime = file_mtimes.get(safe_name)
            if not mtime:
                missing += 1
                continue
            age_hours = max((now_ts - mtime) / 3600.0, 0.0)
            if age_hours > max_staleness_hours:
                stale += 1
            else:
                fresh += 1

        status = {
            "total": len(tickers_ns),
            "fresh": fresh,
            "stale": stale,
            "missing": missing,
        }
        st.session_state["_sidebar_data_status"] = status
        st.session_state["_sidebar_data_status_sig"] = sig
        return status
    except Exception:
        return default


def _build_scan_results_signature(results: list[dict], mode: int, scan_time: str, tt_scan_date: str) -> tuple:
    try:
        compact_rows = []
        for row in results or []:
            symbol = str(row.get("Ticker") or row.get("Symbol") or "").upper().strip()
            price = round(_safe(row.get("Price (₹)", 0.0), 0.0), 4)
            rsi_value = round(_safe(row.get("RSI", 0.0), 0.0), 2)
            compact_rows.append((symbol, price, rsi_value))
        return (
            int(mode or 0),
            str(scan_time or ""),
            str(tt_scan_date or ""),
            len(compact_rows),
            tuple(compact_rows),
        )
    except Exception:
        return (int(mode or 0), str(scan_time or ""), str(tt_scan_date or ""), len(results or []))


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
        # 1. Fetch data, preferring the scan's shared cache over a live call.
        df = None
        try:
            if _engine_utils is not None:
                for _tk in ("^NSEI", "NIFTY_50.NS", "%5ENSEI"):
                    _cached = _engine_utils.ALL_DATA.get(_tk)
                    if isinstance(_cached, pd.DataFrame) and not _cached.empty:
                        df = _cached.copy()
                        break
        except Exception:
            df = None

        if df is None or df.empty:
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
def analyse(ticker, mode, retries=2):  # retries unused; kept for API compatibility
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
        # data-leakage guard. It covers cached frames and any live fallback
        # frame without mutating the shared live ALL_DATA cache.
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
        # Modes 3-7 never reference mktcap_cr so skipping saves ~1000 API calls/scan.
        ok = False
        _mode7_channel_result = None
        _mode7_channel_score = 0.0
        _mode7_channel_entry = False
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
        elif mode == 7:
            if len(close) < 21:
                _scan_diag.record_failure(ticker_ns, "TOO_SHORT")
                return None
            prev_e20 = float(ema(close, 20).iloc[-2]) if len(close) >= 2 else e20
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            avg_vol_sma = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            vol_ratio = (lv / avg_vol_sma) if avg_vol_sma > 0 else 0.0
            d20h_mode7 = (lc / h20 - 1.0) * 100.0 if h20 > 0 else 0.0
            d_ema20_mode7 = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
            ret5d_mode7 = (lc / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
            ret20d_mode7 = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else 0.0
            try:
                high_p = df["High"].dropna()
                lh = float(high_p.iloc[-1]) if len(high_p) else lc
            except Exception:
                lh = lc
            upper_wick_pct = ((lh - max(lc, lo)) / lc) * 100.0 if lc > 0 else 0.0

            trend_ok = lc > e20 > e50 > 0 and e20 >= prev_e20
            not_overextended = -2.0 <= d_ema20_mode7 <= 7.0 and ret5d_mode7 <= 14.0
            volume_ok = vol_ratio >= 1.15
            clean_candle = lc >= lo * 0.995 and upper_wick_pct <= 2.8
            breakout_ready = (
                vol_ratio > 1.3 and -3.0 <= d20h_mode7 <= 2.0
                and 52 <= ri <= 70 and clean_candle
            )
            support_bounce = (
                abs(d_ema20_mode7) <= 3.0 and 52 <= ri <= 64
                and -2.0 <= ret5d_mode7 <= 7.0 and ret20d_mode7 > 3.0
                and lc >= lo * 0.995
            )
            resistance_compression = (
                -6.0 <= d20h_mode7 < -2.0 and vol_ratio >= 1.1
                and 50 <= ri <= 67 and ret20d_mode7 > 2.0
                and d_ema20_mode7 <= 4.5 and clean_candle
            )
            try:
                from mode7_ascending_channel import detect_ascending_channel, score_channel_entry

                _mode7_channel_result = detect_ascending_channel(
                    df,
                    lookback=70,
                    near_support_pct=0.45,
                )
                _mode7_channel_score = score_channel_entry(_mode7_channel_result)
                _mode7_channel_entry = bool(
                    _mode7_channel_result.detected
                    and _mode7_channel_result.entry_zone
                    and _mode7_channel_result.higher_lows >= 1
                    and _mode7_channel_result.higher_highs >= 1
                    and _mode7_channel_result.risk_reward >= 0.8
                    and 48 <= ri <= 70
                    and clean_candle
                    and vol_ratio >= 0.85
                )
            except Exception:
                _mode7_channel_result = None
                _mode7_channel_score = 0.0
                _mode7_channel_entry = False

            classic_mode7_setup = (
                volume_ok and (breakout_ready or support_bounce or resistance_compression)
            )
            ok = (
                lc > 40 and trend_ok and not_overextended
                and (classic_mode7_setup or _mode7_channel_entry)
            )

        if ok:
            h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
            dist_ema20    = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
            ret_5d  = (lc / float(close.iloc[-6]) - 1.0) * 100.0  if len(close) >= 6  else np.nan
            ret_20d = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else np.nan

            _mode7_structure = {}
            if mode == 7:
                try:
                    from strategy_engines.mode7_structure import analyze_mode7_structure

                    _mode7_structure = analyze_mode7_structure(df)
                except Exception:
                    _mode7_structure = {}

            _scan_row = {
                "Symbol":            ticker.replace(".NS", ""),
                "Price (₹)":         round(lc, 2),
                "Volume":            int(lv),
                "RSI":               round(ri, 2),
                "EMA 20":            round(e20, 2),
                "EMA 50":            round(e50, 2),
                "Vol / Avg":         round(lv / avg_vol, 2) if avg_vol > 0 else 0,
                "Mode ID":           int(mode),
                "Mode":              get_mode_label(mode),
                "Δ vs 20D High (%)": round(dist_20d_high, 2),
                "Δ vs EMA20 (%)":    round(dist_ema20, 2),
                "5D Return (%)":     round(ret_5d, 2)  if not np.isnan(ret_5d)  else np.nan,
                "20D Return (%)":    round(ret_20d, 2) if not np.isnan(ret_20d) else np.nan,
            }
            if _mode7_structure:
                _scan_row.update(_mode7_structure)
            if mode == 7 and _mode7_channel_result is not None:
                _scan_row.update({
                    "Ascending Channel": "YES" if _mode7_channel_result.detected else "NO",
                    "Channel Entry Zone": "YES" if _mode7_channel_result.entry_zone else "NO",
                    "Channel Quality": _mode7_channel_result.quality,
                    "Channel Score": round(float(_mode7_channel_score), 1),
                    "Channel Position %": round(float(_mode7_channel_result.position_in_channel) * 100.0, 1),
                    "Channel Support": _mode7_channel_result.support_price,
                    "Channel Resistance": _mode7_channel_result.resistance_price,
                    "Channel RR": _mode7_channel_result.risk_reward,
                    "Channel Note": _mode7_channel_result.note,
                })
            return _scan_row
        _scan_diag.record_failure(ticker_ns, "SCAN_FILTER")
        return None
    except MemoryError:
        _scan_diag.record_failure(ticker_ns, "EXCEPTION")
        raise
    except Exception as exc:
        _scan_diag.record_failure(ticker_ns, "EXCEPTION")
        import logging
        logging.warning(
            "analyse(%s, mode=%s): %s: %s",
            ticker_ns, mode, type(exc).__name__, exc
        )
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
    low_quality = 0
    try:
        low_quality = len(_scan_diag.get_low_quality_tickers())
    except Exception:
        low_quality = int(reasons.get("LOW_QUALITY", 0) or 0)
    scan_mode = st.session_state.get("_scan_diag_mode")
    scan_mode_label = "3" if scan_mode == 7 else scan_mode
    scan_stamp = st.session_state.get("_scan_diag_scan_time", st.session_state.get("scan_time", "—"))

    st.divider()
    st.caption("Scan Diagnostics")
    st.caption(
        f"Mode {scan_mode_label} diagnostics · {scan_stamp}"
        if scan_mode is not None
        else f"Diagnostics · {scan_stamp}"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Attempted", f"{attempted:,}")
    c2.metric("Signals Found", f"{succeeded:,}")
    c3.metric("Data Failed", f"{failed_data:,}")
    c4.metric("Scan Filtered", f"{scan_filtered:,}")
    c5.metric("Data OK", f"{report.get('data_ok_pct', 0.0):.1f}%")
    c6.metric("Low Quality", f"{low_quality:,}")

    with st.expander("Failure Breakdown", expanded=False):
        data_problem_reasons = {
            "NO_DATA",
            "TOO_SHORT",
            "STALE",
            "BAD_PRICE",
            "ZERO_VOLUME",
            "NAN_INDICATORS",
            "EXCEPTION",
            "LOW_QUALITY",
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


def _build_ready_scan_tickers(tickers, *, strict: bool = True) -> tuple[list[str], dict[str, int]]:
    """
    Shrink the stage-2 scan universe to symbols that already have usable data
    in memory after preload/snapshot restore.
    """
    tickers_list = list(tickers or [])
    summary = {
        "requested": len(tickers_list),
        "ready": len(tickers_list),
        "skipped_no_data": 0,
        "skipped_short": 0,
        "skipped_stale": 0,
    }
    try:
        if not tickers_list or _engine_utils is None:
            return tickers_list, summary

        tickers_ns = [
            ticker if str(ticker).endswith(".NS") else f"{ticker}.NS"
            for ticker in tickers_list
        ]

        all_data_lock = getattr(_engine_utils, "_ALL_DATA_LOCK", None)
        if all_data_lock is not None:
            with all_data_lock:
                cached_frames = {ticker_ns: _engine_utils.ALL_DATA.get(ticker_ns) for ticker_ns in tickers_ns}
        else:
            cached_frames = {ticker_ns: _engine_utils.ALL_DATA.get(ticker_ns) for ticker_ns in tickers_ns}

        no_data_lock = getattr(_engine_utils, "_NO_DATA_LOCK", None)
        no_data_fn = getattr(_engine_utils, "_coerce_no_data_tickers", None)
        if callable(no_data_fn):
            if no_data_lock is not None:
                with no_data_lock:
                    known_no_data = set(no_data_fn())
            else:
                known_no_data = set(no_data_fn())
        else:
            known_no_data = set()

        ready: list[str] = []
        for raw_ticker, ticker_ns in zip(tickers_list, tickers_ns):
            if ticker_ns in known_no_data:
                summary["skipped_no_data"] += 1
                continue

            df = cached_frames.get(ticker_ns)
            if not isinstance(df, pd.DataFrame) or df.empty:
                summary["skipped_no_data"] += 1
                continue
            if len(df) < 5:
                summary["skipped_short"] += 1
                continue
            if strict:
                try:
                    if not _is_fresh_enough(df, strict=True):
                        summary["skipped_stale"] += 1
                        continue
                except Exception:
                    pass
            ready.append(raw_ticker)

        if ready:
            summary["ready"] = len(ready)
            return ready, summary
        return tickers_list, summary
    except Exception:
        return tickers_list, summary


def run_scan(tickers, mode, workers=12):
    try:
        workers = min(max(1, int(workers)), 12)
    except Exception:
        workers = 12
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

    def _submit_analyse(executor, ticker):
        try:
            if _TIME_TRAVEL_OK and getattr(_tt, "is_active", lambda: False)():
                return _tt.submit_with_context(executor, analyse, ticker, mode)
        except Exception:
            pass
        return executor.submit(analyse, ticker, mode)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {_submit_analyse(ex, t): t for t in tickers}
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
    7: dict(vol=1.2,  breakout=1.7, ema=1.3, rsi=1.0, pen=1.2),   # Momentum S&R
}


def _safe(v, default=0.0):
    """Return v if finite, else default."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _to_float(value, default=None):
    """Best-effort float conversion used by lightweight UI helpers."""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
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
        7: dict(rsi_tol=4,  vol_tol=0.25, ema_trend=True,  near_high=True,  hw=20, hp=0.03),
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
        elif mode == 7:
            ema_dist = (close / e20s.replace(0, np.nan) - 1.0) * 100.0
            mask &= (
                (e20s > e20s.shift(1))
                & (vol_ratio >= 1.3)
                & (rsi_series >= 52)
                & (rsi_series <= 70)
                & (ema_dist <= 7.0)
            )

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
_ML_MODEL: Any = None
_ML_SCALER: Any = None
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
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception:
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
      Mode 7 (Momentum S&R) → rewards clean S&R structure + volume confirmation
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
        elif mode == 7:                            # Momentum S&R
            if -2.0 <= d20h <= 1.5:      adj += 0.05
            if 1.4 <= vol_r <= 2.8:      adj += 0.04
            if 55 <= ri <= 67:           adj += 0.03
            if de20 > 7.0 or ri > 74:    adj -= 0.08
            if vol_r < 1.0:              adj -= 0.06

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

    try:
        _tt_sig_cut = _tt.get_reference_date()
    except Exception:
        _tt_sig_cut = None

    _signal_dfs: dict[str, pd.DataFrame | None] = {}
    for pr in pre_rows:
        sym = pr["sym"]
        if sym in _signal_dfs:
            continue
        try:
            df_sig = get_df_for_ticker(sym)
            if _tt_sig_cut is not None and df_sig is not None:
                _sig_mask = pd.to_datetime(df_sig.index).date <= _tt_sig_cut
                df_sig = df_sig.loc[_sig_mask]
            _signal_dfs[sym] = df_sig if df_sig is not None and not df_sig.empty else None
        except Exception:
            _signal_dfs[sym] = None

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
            nd_signal = compute_next_day_signal(r, _signal_dfs.get(sym))
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
    st.caption("🏅 Top Picks")
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
mode_colors = get_mode_colors()
pill_cls = get_mode_pill_classes()


def get_mode_display_columns(mode_value: int, base_cols: list[str]) -> list[str]:
    cols = list(base_cols)
    if mode_value == 7:
        mode7_cols = [
            "Breakout Quality",
            "Support Strength",
            "Resistance Distance",
            "Structure Quality",
            "Volume Confirmation",
            "Trap Probability",
            "Momentum Continuation",
            "Channel Score",
            "Channel Entry Zone",
            "Channel RR",
            "Mode7 Verdict",
        ]
        insert_at = cols.index("Trap Check") if "Trap Check" in cols else len(cols)
        cols = cols[:insert_at] + mode7_cols + cols[insert_at:]
    return cols

_SIDEBAR_PANEL_KEYS = (
    "show_sector_screener",
    "battle_show_panel",
    "aura_show_panel",
    "tomorrow_picks_show_panel",
    "pred_chart_show_panel",
    "imported_ai_learning_show_panel",
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
    changed = False
    for key in _SIDEBAR_PANEL_KEYS:
        new_val = (key == active_key)
        if st.session_state.get(key) != new_val:
            st.session_state[key] = new_val
            changed = True
    if changed:
        st.rerun()


def _close_tomorrow_picks_panel() -> None:
    st.session_state["tomorrow_picks_show_panel"] = False

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:20px;'
        'color:#00d4a8;letter-spacing:-0.5px;padding:4px 0 16px 0;">'
        '<span class="live-dot"></span>NSE SENTINEL</div>',
        unsafe_allow_html=True)

    if st.session_state.get("_persistence_msg"):
        st.success(st.session_state.pop("_persistence_msg"))
    if st.session_state.get("_persistence_warning"):
        st.warning(st.session_state.get("_persistence_warning"))
        with st.expander("Fix permanent storage", expanded=True):
            st.caption("Copy this into Streamlit Cloud -> App settings -> Secrets, then reboot the app.")
            st.code(
                '[github_store]\n'
                'token  = "PASTE_YOUR_GITHUB_PAT_HERE"\n'
                'owner  = "Hritvik69"\n'
                'repo   = "nse-sentinel_MAX_"\n'
                'branch = "main"',
                language="toml",
            )
            st.caption("PAT needs repo scope. Until this is added, saves are local-only and not permanent.")
    else:
        _persistence_health = st.session_state.get("_persistence_health", {})
        if isinstance(_persistence_health, dict) and _persistence_health.get("connected"):
            st.success("Cloud persistence connected.")

    st.caption("Strategy Mode")

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

    if not _GRADING_OK:
        st.warning("⚠️ grading_engine.py failed to load — grades disabled.")
    if not _ENHANCED_LOGIC_OK:
        st.warning("⚠️ enhanced_logic_engine.py failed to load — trap/timing disabled.")
    if not _PHASE4_LOGIC_OK:
        st.warning("⚠️ phase4_logic_engine.py failed to load — phase 4 signal refinements disabled.")

    mode_map = get_mode_map()
    _current_strategy_mode = st.session_state.get("strategy_mode")
    if (
        isinstance(_current_strategy_mode, str)
        and "MOMENTUM (S&R)" in _current_strategy_mode
        and _current_strategy_mode not in mode_map
    ):
        st.session_state["strategy_mode"] = get_mode_metadata(7, copy=False)["ui_label"]
    if st.session_state.get("strategy_mode") not in mode_map:
        st.session_state["strategy_mode"] = next(iter(mode_map))
    mode_label = st.selectbox(
        "Strategy mode",
        list(mode_map.keys()),
        label_visibility="collapsed",
        key="strategy_mode",
    )
    mode = mode_map[mode_label]
    mode_display = get_mode_display(mode)

    filter_data = {m: get_mode_filter_rules(m) for m in mode_map.values()}
    mc = get_mode_color(mode)

    if mode == 7:
        st.markdown(
            '<div title="Detects clean breakout structures, support bounces, and institutional momentum with volume confirmation." '
            'style="background:linear-gradient(135deg,rgba(176,140,255,0.18),rgba(74,40,130,0.16));'
            'border:1px solid rgba(176,140,255,0.44);border-radius:10px;padding:12px 14px;margin:8px 0 12px 0;'
            'box-shadow:0 0 24px rgba(176,140,255,0.16);">'
            '<div style="font-size:12px;font-weight:900;color:#b08cff;letter-spacing:0.8px;"><span style="color:#b08cff;">●</span> MODE 3 · MOMENTUM (S&amp;R)</div>'
            '<div style="font-size:11px;color:#8ab4d8;margin-top:4px;">Support + Resistance Momentum Scanner</div>'
            '</div>',
            unsafe_allow_html=True,
        )

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

    st.write("")
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

    st.divider()

    # ── 🕰️ Time Travel Mode ───────────────────────────────────────
    st.caption("🕰️ Time Travel Mode")
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
        st.warning(
            f"SIMULATING {_tt_selected.strftime('%d %b %Y')} (Post-Market Close). "
            "All scans use data up to this date only."
        )
    else:
        st.session_state["tt_date_val"] = None
        st.session_state["tt_date_picker"] = None
        st.caption("Live mode - using current market data")
    _aura_tt_date = st.session_state.get("tt_date_val")
    st.session_state["aura_tt_date"] = (
        _aura_tt_date if (_aura_tt_date is not None and _TIME_TRAVEL_OK) else None
    )

    st.divider()

    st.caption("📡 Data Session")
    if _tt_toggle:
        st.caption("Time Travel is active, so the normal live/close routing is paused until simulation is turned off.")
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

    st.divider()

    # ── Data Management Panel ─────────────────────────────────────
    st.caption("📦 Local Data Cache")
    st.caption("Refresh the offline CSV cache, then run a focused scanner below.")
    if _DATA_DOWNLOADER_OK:
        if st.button("🔄 Refresh Local Data Cache", key="refresh_data_btn"):
            with st.spinner("Updating data..."):
                try:
                    _tickers_for_dl = _get_cached_nse_tickers()
                    update_all_data(_tickers_for_dl)
                    _invalidate_sidebar_data_status_cache()
                    st.success("Data updated")
                except Exception as _e:
                    st.error(f"Data update failed: {_e}")
        st.caption("Focused Scanners")
        csv_scan_clicked = st.button("⚡ Breakout Radar (CSV)", key="csv_next_day_btn")
        if csv_scan_clicked:
            _activate_sidebar_panel("csv_next_day_show_panel")
        live_pulse_clicked = st.button("📡 Live Breakout Pulse", key="live_pulse_btn")
        if live_pulse_clicked:
            st.session_state["live_pulse_autorun"] = True
            _activate_sidebar_panel("live_pulse_show_panel")
        st.caption("⚡ Cached CSV scan for pre-move setups. 📡 Live scan for real-time momentum bursts.")
        # Show cache status
        try:
            _status = _get_sidebar_data_status(_get_cached_nse_tickers())
            st.markdown(
                f'<div style="font-size:11px;color:#4a6480;line-height:1.9;">'
                f'Fresh: <b style="color:#00d4a8">{_status.get("fresh", "?")}</b> &nbsp;'
                f'Stale: <b style="color:#f0b429">{_status.get("stale", "?")}</b> &nbsp;'
                f'Missing: <b style="color:#ff4d6d">{_status.get("missing", "?")}</b></div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass
        _render_sidebar_imported_ai_learning_entry_button()
    else:
        st.caption("data_downloader.py not found - using live yfinance.")
        st.caption("Focused Scanners")
        csv_scan_clicked = st.button("⚡ Breakout Radar (CSV)", key="csv_next_day_btn")
        if csv_scan_clicked:
            _activate_sidebar_panel("csv_next_day_show_panel")
        live_pulse_clicked = st.button("📡 Live Breakout Pulse", key="live_pulse_btn")
        if live_pulse_clicked:
            st.session_state["live_pulse_autorun"] = True
            _activate_sidebar_panel("live_pulse_show_panel")
        st.caption("⚡ Uses local CSV data when available. 📡 Uses live market data directly.")

    if not _DATA_DOWNLOADER_OK:
        _render_sidebar_imported_ai_learning_entry_button()

    st.divider()
    st.caption("Data: Yahoo Finance (NSE). Indicators: EMA, RSI, Volume. Universe: Current NSE listed equities. Educational use only. Not financial advice.")


# ─────────────────────────────────────────────────────────────────────
# FIX 1 — Startup learning status bootstrap
# Fast snapshot restore for first paint.
# The heavy learning cycle still runs after scans and other explicit refresh paths.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────
learning_status, signal_weight_status = _bootstrap_learning_status()
_post_close_outcome_status = _run_post_close_outcome_refresh()
if isinstance(_post_close_outcome_status, dict) and (
    int(_post_close_outcome_status.get("filled_stock", 0) or 0)
    + int(_post_close_outcome_status.get("filled_sector", 0) or 0)
) > 0:
    learning_status = st.session_state.get("_learning_status", learning_status)
    signal_weight_status = st.session_state.get("_signal_weight_status", signal_weight_status)

mc = mode_colors[mode]
_mc_soft = _hex_to_rgba(mc, 0.10)
_mc_border = _hex_to_rgba(mc, 0.28)
_show_sector_screener = st.session_state.get("show_sector_screener", False) or sector_screener_clicked
_show_live_pulse_panel = bool(st.session_state.get("live_pulse_show_panel", False)) or live_pulse_clicked
_show_tomorrow_picks_panel = bool(st.session_state.get("tomorrow_picks_show_panel", False))
_show_pred_chart_panel = bool(st.session_state.get("pred_chart_show_panel", False))
_show_imported_ai_learning_panel = bool(st.session_state.get("imported_ai_learning_show_panel", False))
_show_home_scanner = not (
    _show_sector_screener
    or _show_live_pulse_panel
    or _show_tomorrow_picks_panel
    or _show_pred_chart_panel
    or _show_imported_ai_learning_panel
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
        all_tickers = _get_cached_nse_tickers(show_spinner=True)
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
            try:
                from nse_ticker_universe import invalidate_cache as _invalidate_universe_cache

                try:
                    _invalidate_universe_cache(clear_disk=True)
                except TypeError:
                    _invalidate_universe_cache()
            except Exception:
                pass
            st.session_state.pop("_ticker_master_list", None)
            st.session_state.pop("_ui_all_tickers", None)
            _invalidate_sidebar_data_status_cache()
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
        _outcome_status = st.session_state.get("_post_close_outcome_status")
        if isinstance(_outcome_status, dict):
            _outcome_msg = str(_outcome_status.get("message", "") or "").strip()
            if _outcome_msg:
                st.caption(_outcome_msg)

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
                        _recent_bull_acc_txt = "{:.0f}% correct".format(
                            _bull_df["correct"].eq("True").mean() * 100.0  # pyright: ignore[reportUndefinedVariable]
                        )
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
                st.caption("TOP SIGNALS (Dynamic Weights)")
                for _, _sig_row in _report.head(4).iterrows():
                    _sig_name = str(_sig_row.get("Signal", "") or "")
                    _sig_weight = float(_sig_row.get("Dynamic Weight", 0.0) or 0.0)
                    st.markdown(
                        _weight_bar_html(_sig_name, _sig_weight, warn=_sig_weight <= 3.0),
                        unsafe_allow_html=True,
                    )

            st.caption(
                f"RECENT ACCURACY (Last 20): Bullish calls {_recent_bull_acc_txt}. "
                f"Regime: {_regime_txt}. Walk-forward stability: {_wf_txt}."
            )
    except Exception:
        pass

if _show_pred_chart_panel:
    render_prediction_chart_section(
        ticker_list=all_tickers,
        tomorrow_strip_renderer=lambda: render_tomorrow_picks_ticker_strip(embedded=True),
    )
    st.stop()

if _show_imported_ai_learning_panel:
    render_imported_ai_learning_panel()
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

    st.divider()

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

if main_scan_clicked:
    if st.session_state.get("_main_scan_running", False):
        st.warning("A market scan is already running. Please wait for it to finish.")
        main_scan_clicked = False
    else:
        st.session_state["_main_scan_running"] = True

if main_scan_clicked:
    st.caption(f"⏳ Scanning {n:,} NSE Equities - Mode {mode_display['display_num']}: {mode_display['display_name']}")

    # ── 🕰️ Activate time-travel BEFORE scan if toggle is on ───────────
    _tt_active_date = st.session_state.get("tt_date_val")
    if _tt_active_date is not None and _TIME_TRAVEL_OK:
        with st.spinner(f"🕰️ Preparing historical snapshot for {_tt_active_date.strftime('%d %b %Y')}…"):
            _snapped = _tt.activate(_tt_active_date)
        st.caption(f"🕰️ Time-travel active — {_snapped} ticker(s) snapshotted to {_tt_active_date}")
        # BUG FIX: Reset Nifty cache so get_nifty_20d_return() re-fetches
        # with the TT cutoff applied, not a previously cached live value.
        with _NIFTY_LOCK:
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
                    # Snapshot is partial -- preload_all fills what it can from
                    # the local CSV cache and skips the rest so closed-market
                    # scans stay fast instead of triggering a large live fetch.
                    st.info(
                        f"📂 Loaded {loaded} tickers from "
                        f"{expected_date} snapshot. "
                        f"Checking local history for {missing} remaining tickers..."
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
        _preload_tickers = [
            t for t in all_tickers
            if force_live_refresh
            or _engine_utils is None
            or _engine_utils.ALL_DATA.get(t if t.endswith(".NS") else f"{t}.NS") is None
        ]
        _preload_total = len(_preload_tickers)
        _preload_state = {
            "done": 0,
            "total": _preload_total,
            "loaded": 0,
        }
        _preload_render = {
            "done": 0,
            "ts": 0.0,
            "step": max(50, _preload_total // 40) if _preload_total else 50,
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
                or (now - _preload_render["ts"]) >= 0.50
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
            _preload_tickers,
            period="6mo",
            workers=min(workers, 12),
            progress_callback=_update_preload,
            force_live_refresh=force_live_refresh,
        )
        _finish_stage_feedback(
            preload_bar,
            preload_status,
            preload_eta,
            preload_started,
            _preload_state["total"],
            _preload_state["loaded"],
            "Ready",
        )

    if (
        _tt_active_date is None
        and do_snapshot
        and str(window).upper() in {"CLOSED", "WEEKEND"}
        and not snapshot_exists(expected_date)
    ):
        try:
            from strategy_engines._engine_utils import ALL_DATA

            saved = save_closing_snapshot(ALL_DATA, expected_date, require_live_source=True)
            if saved > 0:
                st.success(f"💾 Market-data snapshot saved: {saved} tickers for {expected_date}")
        except Exception:
            pass

    # Warm the active-mode ML model only after preload so it can reuse the
    # loaded history and avoid duplicate network work during scan startup.
    if _SKLEARN_OK:
        try:
            get_train_function(mode)()
        except Exception:
            pass

    scan_tickers = list(all_tickers)
    scan_scope = {
        "requested": len(scan_tickers),
        "ready": len(scan_tickers),
        "skipped_no_data": 0,
        "skipped_short": 0,
        "skipped_stale": 0,
    }
    try:
        scan_tickers, scan_scope = _build_ready_scan_tickers(
            all_tickers,
            strict=(_tt_active_date is None),
        )
        skipped_total = max(
            0,
            int(scan_scope.get("requested", 0) or 0) - int(scan_scope.get("ready", 0) or 0),
        )
        if skipped_total > 0 and int(scan_scope.get("ready", 0) or 0) > 0:
            stale_note = ""
            if int(scan_scope.get("skipped_stale", 0) or 0) > 0:
                stale_note = f" | stale skipped: {int(scan_scope.get('skipped_stale', 0) or 0):,}"
            st.caption(
                f"⚡ Fast scan mode: running stage 2 on {int(scan_scope.get('ready', 0) or 0):,} ready tickers "
                f"and skipping {skipped_total:,} unusable names "
                f"(no data: {int(scan_scope.get('skipped_no_data', 0) or 0):,}, "
                f"short history: {int(scan_scope.get('skipped_short', 0) or 0):,}{stale_note})."
            )
    except Exception:
        scan_tickers = list(all_tickers)

    try:
        results, elapsed = run_scan(scan_tickers, mode, workers=workers)
    finally:
        st.session_state.pop("_main_scan_running", None)
        # Always restore — even if scan raised an exception
        if _tt_active_date is not None and _TIME_TRAVEL_OK:
            _tt.restore()
            # BUG FIX: Reset Nifty cache after restore so next live scan
            # does not reuse the TT-truncated Nifty return value.
            with _NIFTY_LOCK:
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
    st.session_state.pop("last_scan_df", None)
    st.session_state.pop("_last_scan_df_sig", None)

    try:
        _run_post_close_outcome_refresh(force=True)
    except Exception:
        pass

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
        render_sector_explorer_section(_get_cached_nse_tickers())
    else:
        st.warning(
            "Sector Explorer is unavailable because its UI module could not be imported. "
            f"Import error: {_SECTOR_EXPLORER_UI_ERR}"
        )

    _sector_intel_ready = isinstance(st.session_state.get("last_scan_df"), pd.DataFrame) and not st.session_state.get("last_scan_df").empty
    if _SECTOR_INTELLIGENCE_UI_OK and _sector_intel_ready:
        render_sector_intelligence_section()
    elif _SECTOR_INTELLIGENCE_UI_OK and not _sector_intel_ready:
        st.info("Run a main market scan first to enable Sector Intelligence.")
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
    st.divider()
    _battle_hdr_col, _battle_close_col = st.columns([6, 1])
    with _battle_hdr_col:
        st.header("⚔️ Compare Stocks")
        st.caption("This panel now opens in the main UI. Enter up to 10 stocks and run the full comparison here.")
    with _battle_close_col:
        st.write("")
        _battle_close_panel = st.button("Close", key="battle_close_panel_btn", width="stretch")

    if _battle_close_panel:
        st.session_state["battle_show_panel"] = False
        st.rerun()

    configure_nse_stock_search(_get_cached_nse_tickers())
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

        _battle_main_run = st.form_submit_button("Run Battle Analysis", width="stretch")
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
    st.caption("📊 Market Bias Engine (Analytics)")
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

    st.divider()

# ── RESULTS ───────────────────────────────────────────────────────────
if _show_home_scanner and "results" in st.session_state:
    results     = st.session_state["results"]
    stored_mode = st.session_state.get("mode", mode)
    stored_mode_display = get_mode_display(stored_mode)
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
        _scan_df_sig = _build_scan_results_signature(
            results,
            stored_mode,
            scan_time_d,
            str(st.session_state.get("tt_scan_date", "") or ""),
        )
        _cached_scan_df = st.session_state.get("last_scan_df")
        _using_cached_scan_df = (
            isinstance(_cached_scan_df, pd.DataFrame)
            and not _cached_scan_df.empty
            and st.session_state.get("_last_scan_df_sig") == _scan_df_sig
        )
        if _using_cached_scan_df:
            df = _cached_scan_df.copy()
        else:
            with st.spinner("🔢 Computing Smart Scores, Backtest & ML probabilities ..."):
                df = enhance_results(results, stored_mode)

        # ── Enhanced Logic Engine — runs FIRST so Setup Quality /
        # ── Volume Trend / Trap Risk are available for Prediction Score
        if not _using_cached_scan_df:
            try:
                df = apply_enhanced_logic(df)
            except Exception:
                pass

        # ── Universal Grading Engine ──────────────────────────────────
        if not _using_cached_scan_df:
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
                _mb = st.session_state.get("market_bias_result", {})
        else:
            _mb = st.session_state.get("market_bias_result", {})

        # ── Phase 4 Logic Engine (Setup Type, Reason, Risk, Final Signal)
        if not _using_cached_scan_df:
            try:
                df = apply_phase4_logic(df, _mb)
            except Exception:
                pass

        if not _using_cached_scan_df:
            try:
                from trade_decision_simple import apply_trade_decision_simple
                df = apply_trade_decision_simple(df)
            except Exception:
                pass

        # ── Learning prediction (added column only) ───────────────────
        if not _using_cached_scan_df:
            try:
                from learning_engine import batch_predict_success
                df["Learned Prob %"] = batch_predict_success(df)
            except Exception:
                pass

        # ── Phase 4.2 Logic Engine (Advanced Trap, Expected Move, Adjusted Signal)
        if not _using_cached_scan_df:
            try:
                df = apply_phase42_logic(df)
            except Exception:
                pass

        # Mode 7 ranks structure first and suppresses trap-heavy prediction hype.
        if not _using_cached_scan_df and stored_mode == 7:
            try:
                from strategy_engines.mode7_ranking import apply_mode7_ranking

                df = apply_mode7_ranking(df, _mb)
            except Exception:
                pass

        if not _using_cached_scan_df:
            try:
                df["Trap Check"] = df.apply(_trap_check_label, axis=1)
            except Exception:
                pass

        # ── Phase 4.3/4.4 (Dynamic Intelligence + Feedback Tracking) ─
        if not _using_cached_scan_df:
            try:
                df = apply_phase43_logic(df)
            except Exception:
                pass
            try:
                df = apply_phase44_logic(df)
            except Exception:
                pass

            st.session_state["last_scan_df"] = df.copy()
            st.session_state["_last_scan_df_sig"] = _scan_df_sig

        _did_log_predictions = False
        _fs = {}
        try:
            from prediction_feedback_store import feedback_summary, log_scan_predictions

            _log_key = f"{stored_mode}|{scan_time_d}|{len(df)}"
            if st.session_state.get("_prediction_log_key") != _log_key:
                log_scan_predictions(df, stored_mode, st.session_state.get("market_bias_result"))
                st.session_state["_prediction_log_key"] = _log_key
                _did_log_predictions = True
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

        if _did_log_predictions:
            try:
                _full_learning_refresh = _refresh_learning_after_prediction_log(_fs)
                if _full_learning_refresh:
                    learning_status = st.session_state.get("_learning_status", learning_status)
                    signal_weight_status = st.session_state.get("_signal_weight_status", signal_weight_status)
                else:
                    st.caption(
                        "⚡ Fast result paint: learning retrain skipped because no new validated outcomes were added yet."
                    )
            except Exception:
                pass

        if "Next-Day Signal" in df.columns:
            with st.sidebar:
                counts = df["Next-Day Signal"].value_counts().to_dict()
                sg = counts.get("🔥 Strong Green", 0)
                pu = counts.get("🟢 Possible Up", 0)
                ri = counts.get("⚠️ Risky (Late Entry)", 0)
                we = counts.get("❌ Weak Setup", 0)

                st.divider()
                st.caption("📊 Next-Day Signals Summary")
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
        st.write("")
        st.header("🔥 Top Picks")
        st.write("")

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
                    _sl  = max(round(_price * (1 - (_de20 / 100) * 0.5), 2), round(_price * 0.97, 2))
                    _tgt = min(round(_price * (1 + (_de20 / 100) * 1.5), 2), round(_price * 1.15, 2))
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
                st.link_button("📈 TradingView", tv_link, width="stretch")

        st.write("")
        st.write("")
        st.header("📊 Full Rankings")
        st.write("")

        # ── TASK 2 & 5: Clean Table with TradingView Link ─────────────
        table_df = df.copy()
        try:
            if stored_mode == 7 and "Mode7 Rank Score" in table_df.columns:
                table_df["rank_score"] = pd.to_numeric(table_df["Mode7 Rank Score"], errors="coerce").fillna(0.0)
            else:
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
        display_cols = get_mode_display_columns(stored_mode, display_cols)
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
                "Breakout Quality": st.column_config.TextColumn("Breakout"),
                "Support Strength": st.column_config.TextColumn("Support"),
                "Resistance Distance": st.column_config.TextColumn("Resistance"),
                "Structure Quality": st.column_config.TextColumn("Structure"),
                "Volume Confirmation": st.column_config.TextColumn("Volume Confirm"),
                "Trap Probability": st.column_config.TextColumn("Trap Prob"),
                "Momentum Continuation": st.column_config.TextColumn("Continuation"),
                "Channel Score": st.column_config.NumberColumn("Channel", format="%.1f"),
                "Channel Entry Zone": st.column_config.TextColumn("Channel Entry"),
                "Channel RR": st.column_config.NumberColumn("Channel RR", format="%.2f"),
                "Mode7 Verdict": st.column_config.TextColumn("M7 Verdict", width="medium"),
                "Trap Check": st.column_config.TextColumn("Trap Check"),
                "Next-Day Signal": st.column_config.TextColumn("Signal"),
                "TradingView": st.column_config.LinkColumn("TradingView Link", display_text="📈 Open Chart"),
                "Action": st.column_config.TextColumn("Action"),
                "Hold Days": st.column_config.TextColumn("Hold Days"),
            },
            width="stretch",
            hide_index=True,
        )

        # ── TASK 3: Expandable Details ────────────────────────────────
        st.write("")
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
                    
                    st.write("")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("5D Return", f"{row.get('5D Return (%)', 0):+.2f}%")
                    rc2.metric("20D Return", f"{row.get('20D Return (%)', 0):+.2f}%")
                    
                    # Handle possible missing columns gracefully
                    h_val = row.get('Δ vs 20D High (%)', 0)
                    rc3.metric("Δ vs 20D Hi", f"{h_val:+.2f}%" if pd.notna(h_val) else "—")

                    if stored_mode == 7:
                        st.write("")
                        st.markdown("**Momentum S&R Structure**")
                        m71, m72, m73, m74 = st.columns(4)
                        m71.metric("Breakout", str(row.get("Breakout Quality", "—")))
                        m72.metric("Support", str(row.get("Support Strength", "—")))
                        m73.metric("Resistance", str(row.get("Resistance Distance", "—")))
                        m74.metric("Structure", str(row.get("Structure Quality", "—")))
                        if "Channel Score" in row.index:
                            c71, c72, c73, c74 = st.columns(4)
                            c71.metric("Channel", str(row.get("Channel Quality", "—")))
                            c72.metric("Entry Zone", str(row.get("Channel Entry Zone", "—")))
                            c73.metric("Channel RR", f"{float(row.get('Channel RR', 0) or 0):.2f}")
                            c74.metric("Position", f"{float(row.get('Channel Position %', 0) or 0):.1f}%")

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
                "Breakout Quality", "Support Strength", "Resistance Distance",
                "Structure Quality", "Volume Confirmation", "Trap Probability",
                "Momentum Continuation", "Channel Score", "Channel Entry Zone", "Channel RR",
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
        st.write("")
        st.write("")
        _clean_export = get_clean_export_df(table_df)
        _csv_buf = io.StringIO()
        _clean_export.to_csv(_csv_buf, index=False)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="⬇ Download Results CSV",
                data=_csv_buf.getvalue().encode("utf-8-sig"),
                file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}_mode{stored_mode_display['display_num']}.csv",
                mime="text/csv",
                width="stretch",
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

            st.write("")
            st.write("")
            st.header("Top 3 Buyable For Tomorrow")
            st.caption("Best next-day buy candidates from this mode scan. Save them once to keep them until you delete them.")

            _tomorrow_symbols = [
                _normalize_tomorrow_symbol(symbol)
                for symbol in _tomorrow_df.get("Symbol", pd.Series(dtype=object)).tolist()
                if _normalize_tomorrow_symbol(symbol)
            ]
            _ai_preview = _build_mode_ai_top3_preview(_tomorrow_symbols, stored_mode)
            _ai_preview_df = _ai_preview.get("table") if isinstance(_ai_preview, dict) else pd.DataFrame()
            _ai_summary = _ai_preview.get("summary", {}) if isinstance(_ai_preview, dict) else {}
            if isinstance(_ai_preview_df, pd.DataFrame) and not _ai_preview_df.empty:
                _ai_merge = _ai_preview_df.rename(columns={"Ticker": "Symbol"})
                _tomorrow_df = _tomorrow_df.merge(_ai_merge, on="Symbol", how="left")

                _brain_direction = str(_ai_summary.get("direction", "Sideways") or "Sideways")
                _brain_conf = float(_safe(_ai_summary.get("confidence", 0.0), 0.0))
                _brain_score = float(_safe(_ai_summary.get("score", 50.0), 50.0))
                _brain_action = str(_ai_summary.get("action", "Watch") or "Watch")
                _brain_risk = str(_ai_summary.get("risk", "MEDIUM") or "MEDIUM").upper()
                _brain_hold = str(_ai_summary.get("hold_days", "-") or "-").replace("?", "-")
                _brain_key_signal = _pretty_learning_signal_name(_ai_summary.get("key_signal", "momentum"))
                _brain_regime = str(_ai_summary.get("regime", "UNKNOWN") or "UNKNOWN").replace("_", " ").title()
                _brain_color = {"Bullish": "#00d4a8", "Bearish": "#ff4d6d", "Sideways": "#8ab4d8"}.get(_brain_direction, "#8ab4d8")
                _action_color = (
                    "#00d4a8" if "Buy Tomorrow" in _brain_action
                    else "#ff4d6d" if "Avoid" in _brain_action
                    else "#f0b429" if "Watch" in _brain_action
                    else "#4da3ff"
                )
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg,#0b1017 56%,{_brain_color}12);border:1.5px solid #1e3a5f;border-radius:16px;padding:16px 18px;margin:12px 0 14px 0;">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap;">
                        <div>
                          <div style="font-size:10px;color:#4a6480;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Self-Learning Brain For This Mode</div>
                          <div style="font-size:24px;font-weight:900;color:{_brain_color};">{html.escape(_brain_direction)} • {_brain_conf:.1f}%</div>
                          <div style="font-size:12px;color:#8ab4d8;margin-top:4px;">Mode M{stored_mode_display["display_num"]} AI read on these top 3 candidates</div>
                        </div>
                        <div style="text-align:right;">
                          <span style="background:{_action_color}22;border:1.5px solid {_action_color};border-radius:999px;padding:6px 12px;font-size:11px;font-weight:800;color:{_action_color};">{html.escape(_brain_action)}</span>
                          <div style="font-size:12px;color:#8ab4d8;margin-top:8px;">AI Score: {_brain_score:.1f}</div>
                        </div>
                      </div>
                      <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px;">
                        <div style="background:#0d1626;border:1px solid #1e3a5f;border-radius:10px;padding:10px 12px;">
                          <div style="font-size:10px;color:#4a6480;">Key Signal</div>
                          <div style="font-size:13px;font-weight:800;color:#ccd9e8;margin-top:4px;">{html.escape(_brain_key_signal)}</div>
                        </div>
                        <div style="background:#0d1626;border:1px solid #1e3a5f;border-radius:10px;padding:10px 12px;">
                          <div style="font-size:10px;color:#4a6480;">Risk</div>
                          <div style="font-size:13px;font-weight:800;color:#ccd9e8;margin-top:4px;">{html.escape(_brain_risk)}</div>
                        </div>
                        <div style="background:#0d1626;border:1px solid #1e3a5f;border-radius:10px;padding:10px 12px;">
                          <div style="font-size:10px;color:#4a6480;">Hold</div>
                          <div style="font-size:13px;font-weight:800;color:#ccd9e8;margin-top:4px;">{html.escape(_brain_hold)}</div>
                        </div>
                        <div style="background:#0d1626;border:1px solid #1e3a5f;border-radius:10px;padding:10px 12px;">
                          <div style="font-size:10px;color:#4a6480;">Regime</div>
                          <div style="font-size:13px;font-weight:800;color:#ccd9e8;margin-top:4px;">{html.escape(_brain_regime)}</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            _tomorrow_cols = [
                "Symbol", "Tomorrow Pick Score", "AI Confidence", "AI Direction",
                "Final Score", "Prediction Score", "Mode7 Verdict", _signal_col, "AI Action", "AI Key Signal",
                "Conviction Tier", "Trap Check", "Chart",
                "Action", "Hold Days",
            ]
            _tomorrow_cols = [c for c in _tomorrow_cols if c in _tomorrow_df.columns]

            st.dataframe(
                _tomorrow_df[_tomorrow_cols],
                column_config={
                    "Symbol": st.column_config.TextColumn("Ticker"),
                    "Tomorrow Pick Score": st.column_config.NumberColumn("Tomorrow Score", format="%.1f"),
                    "AI Confidence": st.column_config.NumberColumn("AI Conf %", format="%.1f%%"),
                    "AI Direction": st.column_config.TextColumn("AI Dir", width="small"),
                    "Final Score": st.column_config.NumberColumn("Final Score", format="%.1f"),
                    "Prediction Score": st.column_config.NumberColumn("Pred Score", format="%.1f"),
                    "Mode7 Verdict": st.column_config.TextColumn("M7 Verdict", width="medium"),
                    "Adjusted Signal": st.column_config.TextColumn("Signal", width="medium"),
                    "Next-Day Signal": st.column_config.TextColumn("Signal", width="medium"),
                    "AI Action": st.column_config.TextColumn("AI Action", width="medium"),
                    "AI Key Signal": st.column_config.TextColumn("AI Key", width="medium"),
                    "Conviction Tier": st.column_config.TextColumn("Conviction"),
                    "Trap Check": st.column_config.TextColumn("Trap Check"),
                    "Chart": st.column_config.LinkColumn("Chart", display_text="Open Chart"),
                    "Action": st.column_config.TextColumn("Action"),
                    "Hold Days": st.column_config.TextColumn("Hold Days"),
                },
                width="stretch",
                hide_index=True,
            )
            _render_add_in_picks_actions(
                _tomorrow_symbols,
                key_prefix=f"scan_top3_mode_{stored_mode}",
                scope_label="main scan",
                bucket=_tomorrow_bucket_for_mode(stored_mode),
                helper_text="Add these top scan picks into Tomorrow's Picks and keep them saved until you remove them.",
            )
            _render_ai_prediction_import_action(
                _tomorrow_symbols,
                mode_value=stored_mode,
                key_prefix=f"scan_top3_mode_{stored_mode}",
                source_label=f"Mode {stored_mode_display['display_num']} Top 3",
                source_bucket=_tomorrow_bucket_for_mode(stored_mode),
                source_rows=_tomorrow_df,
                helper_text="Add these mode picks into Imported AI Stocks for self-learning. Open AI Prediction later only when you want the chart.",
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
            render_imported_ai_actions=_render_ai_prediction_import_action,
        )
    except TypeError as _breakout_call_exc:
        if (
            "render_add_in_picks_actions" not in str(_breakout_call_exc)
            and "render_imported_ai_actions" not in str(_breakout_call_exc)
        ):
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
        st.divider()
        st.header("📂 CSV Next-Day Potential")
        st.caption("Fast local scan using cached CSV data. Focused on tomorrow-up probability and stricter buy quality.")

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
                        st.caption(
                            "Grading System: A strongest setup, B good setup, C watchlist quality, D weak setup. "
                            f"Grade Distribution: {_grade_summary}"
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
                    width="stretch",
                    hide_index=True,
                )
            elif not csv_last_error:
                st.info("No clean buy-ready setups were found for tomorrow in the current CSV universe. That usually means wait for better structure instead of forcing a trade.")

if _LIVE_PULSE_SECTION_OK:
    render_live_breakout_pulse(
        live_pulse_clicked=live_pulse_clicked,
        tt_date_val=st.session_state.get("tt_date_val"),
        render_add_in_picks_actions=_render_add_in_picks_actions,
        render_imported_ai_actions=_render_ai_prediction_import_action,
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
        if st.session_state.get("_battle_analysis_running", False):
            st.warning("Battle analysis is already running. Please wait for it to finish.")
            _battle_request_tickers = None
        else:
            st.session_state["_battle_analysis_running"] = True

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
                st.session_state.pop("_battle_analysis_running", None)
                st.session_state["battle_tickers_request"] = None
                if _tt_battle_date is not None and _TIME_TRAVEL_OK:
                    _tt.restore()

    _battle_df = st.session_state.get("battle_results_df", None)
    if isinstance(_battle_df, pd.DataFrame) and not _battle_df.empty:
        st.divider()
        st.header("⚔️ Multi-Stock Battle Mode")
        st.caption("Compare up to 10 stocks head-to-head. Full pipeline per ticker. Ranks by battle probability, quality and risk-adjusted strength.")

        # ── 🥇 Winner Card ────────────────────────────────
        st.write("")
        st.caption("🥇 Battle Winner")
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
        st.write("")
        st.caption("📊 Head-to-Head Comparison")
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
            width="stretch",
            hide_index=True,
        )
        with st.expander("🧾 Full Battle Diagnostics", expanded=False):
            st.dataframe(_battle_df, width="stretch", hide_index=True)

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

st.divider()
st.caption("NSE SENTINEL · Python + Streamlit + yFinance | For educational purposes only - not financial advice")
