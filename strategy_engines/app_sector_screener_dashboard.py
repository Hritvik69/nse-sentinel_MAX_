import streamlit as st
import pandas as pd

import threading
from html import escape
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any

try:
    from feature_data_manager import get_current_window as _get_feature_window
except Exception:
    def _get_feature_window() -> str:
        return "CLOSED"

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))

try:
    from strategy_engines.multi_index_market_bias_engine import (
        analyze_index,
        build_dashboard_sector_raw_rows,
        clear_index_cache,
        compute_overall_market_enhanced,
        compute_sector_prediction_enhanced,
        get_dashboard_data_signature,
        get_dashboard_index_sector,
        get_dashboard_sector_count,
        get_dashboard_sector_description,
        get_dashboard_sector_labels,
        get_dashboard_sector_signature,
        get_dashboard_sector_stocks,
        preload_dashboard_sector_data,
    )
    _SSE_OK = True
except ImportError:
    try:
        from multi_index_market_bias_engine import (
            analyze_index,
            build_dashboard_sector_raw_rows,
            clear_index_cache,
            compute_overall_market_enhanced,
            compute_sector_prediction_enhanced,
            get_dashboard_data_signature,
            get_dashboard_index_sector,
            get_dashboard_sector_count,
            get_dashboard_sector_description,
            get_dashboard_sector_labels,
            get_dashboard_sector_signature,
            get_dashboard_sector_stocks,
            preload_dashboard_sector_data,
        )
        _SSE_OK = True
    except ImportError:
        _SSE_OK = False
        def clear_index_cache() -> None:  # type: ignore[misc]
            return None

try:
    from strategy_engines._engine_utils import ALL_DATA, _ALL_DATA_LOCK
except ImportError:
    try:
        from _engine_utils import ALL_DATA, _ALL_DATA_LOCK  # type: ignore[import]
    except ImportError:
        ALL_DATA: dict[str, pd.DataFrame | None] = {}  # type: ignore[assignment]
        _ALL_DATA_LOCK = threading.Lock()


def _pred_color(pred: str) -> str:
    pred_u = str(pred).upper().strip()
    return {
        "UP": "#00d4a8",
        "BULLISH": "#00d4a8",
        "DOWN": "#ff4d6d",
        "BEARISH": "#ff4d6d",
        "SIDEWAYS": "#f0b429",
    }.get(pred_u, "#8ab4d8")


def _pred_icon(pred: str) -> str:
    pred_u = str(pred).upper().strip()
    return {
        "UP": "📈",
        "BULLISH": "📈",
        "DOWN": "📉",
        "BEARISH": "📉",
        "SIDEWAYS": "➡️",
    }.get(pred_u, "—")


def _prob_color(p: float) -> str:
    if p >= 65:
        return "#00d4a8"
    if p >= 50:
        return "#f0b429"
    return "#ff4d6d"


def _pill(text: str, color: str, bg_alpha: str = "22") -> str:
    return (
        f'<span style="background:{color}{bg_alpha};color:{color};'
        'border-radius:999px;padding:4px 10px;font-size:11px;font-weight:800;'
        f'display:inline-block;">{text}</span>'
    )


def _now_ist() -> datetime:
    try:
        return datetime.now(_IST_TZ)
    except Exception:
        return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _status_card(title: str, value: str, caption: str, accent: str) -> str:
    return (
        f'<div style="background:#0b1017;border:1px solid {accent}33;border-radius:14px;'
        'padding:14px 16px;min-height:108px;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1.2px;text-transform:uppercase;'
        f'margin-bottom:8px;">{title}</div>'
        f'<div style="font-size:22px;font-weight:900;color:{accent};line-height:1.1;margin-bottom:8px;">{value}</div>'
        f'<div style="font-size:11px;color:#8ab4d8;line-height:1.5;">{caption}</div>'
        '</div>'
    )


def _insight_card(title: str, value: str, caption: str, accent: str) -> str:
    return (
        f'<div style="background:linear-gradient(180deg,#0d1521 0%,#0b1017 100%);'
        f'border:1px solid {accent}2b;border-radius:14px;padding:14px 16px;min-height:102px;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1.2px;text-transform:uppercase;'
        f'margin-bottom:8px;">{escape(title)}</div>'
        f'<div style="font-size:18px;font-weight:900;color:{accent};line-height:1.15;margin-bottom:8px;">{escape(value)}</div>'
        f'<div style="font-size:11px;color:#8ab4d8;line-height:1.55;">{escape(caption)}</div>'
        '</div>'
    )


def _source_color(primary_source: str) -> str:
    source = str(primary_source or "").strip().upper()
    if source == "LIVE ONLY":
        return "#00d4a8"
    if source == "LIVE DOMINANT":
        return "#22c55e"
    if source == "MIXED":
        return "#0094ff"
    if source == "CSV CACHE":
        return "#f0b429"
    if source == "NO DATA":
        return "#ff4d6d"
    return "#8ab4d8"


def _safe_dt_label(value: object) -> str:
    if value in (None, "", "None"):
        return "Not yet"
    try:
        dt = pd.to_datetime(value)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.tz_localize(_IST_TZ)
        else:
            dt = dt.tz_convert(_IST_TZ)
        return dt.strftime("%d %b %Y, %I:%M:%S %p IST")
    except Exception:
        return str(value)


def _pretty_label(value: object, fallback: str = "N/A") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.replace("_", " ").replace("+", " + ").title()


def _prediction_conviction_card(pred: dict[str, Any]) -> tuple[str, str, str]:
    prob = float(pred.get("bullish_probability", 50.0) or 50.0)
    conf = float(pred.get("confidence", 50.0) or 50.0)
    coverage_quality = str(pred.get("coverage_quality", "MEDIUM") or "MEDIUM").strip().upper()
    probability_model = _pretty_label(pred.get("probability_model", "base"), fallback="Base")
    directional_gap = abs(prob - 50.0)

    if coverage_quality == "VERY_LOW":
        return (
            "Low Reliability",
            f"Coverage is very low. Treat this as a weak read. Model: {probability_model}.",
            "#ff4d6d",
        )
    if coverage_quality == "LOW":
        return (
            "Guarded Read",
            f"Coverage is low, so the directional edge can change quickly. Model: {probability_model}.",
            "#f0b429",
        )
    if conf >= 70 and directional_gap >= 12:
        return (
            "High Conviction",
            f"Strong signal agreement and cleaner coverage. Model: {probability_model}.",
            "#00d4a8",
        )
    if conf >= 58 and directional_gap >= 8:
        return (
            "Medium Conviction",
            f"Usable directional edge, but still probabilistic. Model: {probability_model}.",
            "#0094ff",
        )
    return (
        "Low Edge",
        f"Probability is near balance or confidence is modest. Model: {probability_model}.",
        "#8ab4d8",
    )


def _result_insight_cards(result: dict[str, Any], pred: dict[str, Any]) -> list[str]:
    meta = result.get("scan_meta", {}) if isinstance(result, dict) else {}
    profile = meta.get("profile", {}) if isinstance(meta, dict) else {}
    source = str(profile.get("primary_source", "NO DATA") or "NO DATA")
    source_accent = _source_color(source)
    captured_at = str(profile.get("captured_at", "") or "").strip() or str(meta.get("refreshed_at", "Not yet") or "Not yet")
    market_date = str(profile.get("market_date", "-") or "-")
    market_window = str(profile.get("window", "-") or "-")
    live_count = int(profile.get("live", 0) or 0)
    csv_count = int(profile.get("csv", 0) or 0)
    missing_count = int(profile.get("missing", 0) or 0)
    coverage_quality = _pretty_label(pred.get("coverage_quality", "MEDIUM"), fallback="Medium")
    signal_quality = _pretty_label(pred.get("signal_quality", "MEDIUM"), fallback="Medium")
    conviction_value, conviction_caption, conviction_accent = _prediction_conviction_card(pred)

    return [
        _insight_card(
            "Data Trust",
            source,
            f"Live {live_count} | CSV {csv_count} | Missing {missing_count}",
            source_accent,
        ),
        _insight_card(
            "Captured",
            captured_at,
            f"Market date {market_date} | Window {market_window}",
            "#8ab4d8",
        ),
        _insight_card(
            "Prediction Basis",
            _pretty_label(pred.get("probability_model", "base"), fallback="Base"),
            f"Coverage {coverage_quality} | Signal {signal_quality}",
            "#0094ff",
        ),
        _insight_card(
            "Conviction",
            conviction_value,
            conviction_caption,
            conviction_accent,
        ),
    ]


def _top_stock_card(stock: dict[str, Any]) -> str:
    symbol = escape(str(stock.get("symbol", "-") or "-"))
    signal = escape(str(stock.get("signal", "-") or "-"))
    grade = escape(str(stock.get("grade", "-") or "-"))
    score = float(stock.get("score", 0.0) or 0.0)
    conf = float(stock.get("conf", 0.0) or 0.0)
    accent = _prob_color(score)
    return (
        f'<div style="background:linear-gradient(180deg,#0f1923 0%,#0b1017 100%);'
        f'border:1px solid {accent}2f;border-radius:14px;padding:14px 16px;min-height:118px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">'
        f'<div style="font-size:18px;font-weight:900;color:#ccd9e8;">{symbol}</div>'
        f'<div style="font-size:22px;font-weight:900;color:{accent};">{score:.1f}</div>'
        f'</div>'
        f'<div style="font-size:11px;color:#8ab4d8;margin:8px 0 10px 0;">{signal}</div>'
        f'<div style="display:flex;gap:14px;flex-wrap:wrap;">'
        f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;">Grade</div>'
        f'<div style="font-size:13px;font-weight:800;color:#f0b429;">{grade}</div></div>'
        f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;">Conf</div>'
        f'<div style="font-size:13px;font-weight:800;color:#0094ff;">{conf:.0f}%</div></div>'
        f'</div></div>'
    )


def _top_sector_card(row: dict[str, Any]) -> str:
    sector = escape(str(row.get("Sector", "-") or "-"))
    prediction = escape(str(row.get("Prediction", "SIDEWAYS") or "SIDEWAYS"))
    probability = float(row.get("Probability %", 50.0) or 50.0)
    confidence = float(row.get("Confidence %", 40.0) or 40.0)
    coverage = float(row.get("Coverage %", 0.0) or 0.0)
    accent = _pred_color(prediction)
    return (
        f'<div style="background:linear-gradient(180deg,#0f1923 0%,#0b1017 100%);'
        f'border:1px solid {accent}30;border-radius:14px;padding:14px 16px;min-height:122px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:flex-start;">'
        f'<div style="font-size:17px;font-weight:900;color:#ccd9e8;">{sector}</div>'
        f'<div style="font-size:22px;font-weight:900;color:{_prob_color(probability)};">{probability:.0f}%</div>'
        f'</div>'
        f'<div style="font-size:11px;color:{accent};margin:8px 0 10px 0;">{prediction}</div>'
        f'<div style="display:flex;gap:14px;flex-wrap:wrap;">'
        f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;">Confidence</div>'
        f'<div style="font-size:13px;font-weight:800;color:#0094ff;">{confidence:.0f}%</div></div>'
        f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;">Coverage</div>'
        f'<div style="font-size:13px;font-weight:800;color:#22c55e;">{coverage:.0f}%</div></div>'
        f'</div></div>'
    )


def _build_data_profile(symbols: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    ordered = [str(sym).replace(".NS", "").strip().upper() for sym in (symbols or []) if str(sym).strip()]
    profile: dict[str, Any] = {
        "total": len(ordered),
        "live": 0,
        "csv": 0,
        "unknown": 0,
        "missing": 0,
        "primary_source": "NO DATA",
        "market_date": "—",
        "window": "—",
        "captured_at": "",
        "loaded": 0,
    }
    if not ordered:
        return profile

    try:
        with _ALL_DATA_LOCK:
            frames = {
                sym: ALL_DATA.get(sym if sym.endswith(".NS") else f"{sym}.NS")
                for sym in ordered
            }
    except Exception:
        frames = {}

    market_dates: set[str] = set()
    window_counts: dict[str, int] = {}
    latest_capture = None

    for sym in ordered:
        df = frames.get(sym)
        if df is None or getattr(df, "empty", True):
            profile["missing"] += 1
            continue

        source = str(getattr(df, "attrs", {}).get("_nse_data_source", "") or "").strip().lower()
        if source.startswith("live"):
            profile["live"] += 1
        elif source.startswith("csv"):
            profile["csv"] += 1
        else:
            profile["unknown"] += 1

        market_date = str(getattr(df, "attrs", {}).get("_nse_market_date", "") or "").strip()
        if market_date:
            market_dates.add(market_date)

        window = str(getattr(df, "attrs", {}).get("_nse_window", "") or "").strip().upper()
        if window:
            window_counts[window] = window_counts.get(window, 0) + 1

        captured_at = str(getattr(df, "attrs", {}).get("_nse_captured_at", "") or "").strip()
        if captured_at:
            try:
                parsed = pd.to_datetime(captured_at)
                if latest_capture is None or parsed > latest_capture:
                    latest_capture = parsed
            except Exception:
                pass

    profile["loaded"] = int(profile["total"]) - int(profile["missing"])
    if market_dates:
        profile["market_date"] = sorted(market_dates)[-1]
    if window_counts:
        profile["window"] = max(window_counts.items(), key=lambda item: item[1])[0]
    if latest_capture is not None:
        profile["captured_at"] = _safe_dt_label(latest_capture)

    loaded = max(int(profile["loaded"]), 0)
    if loaded <= 0:
        primary = "NO DATA"
    elif int(profile["live"]) == loaded and int(profile["csv"]) == 0 and int(profile["unknown"]) == 0:
        primary = "LIVE ONLY"
    elif int(profile["live"]) > 0 and int(profile["csv"]) == 0:
        primary = "LIVE DOMINANT"
    elif int(profile["live"]) > 0:
        primary = "MIXED"
    elif int(profile["csv"]) > 0:
        primary = "CSV CACHE"
    else:
        primary = "UNKNOWN"
    profile["primary_source"] = primary
    return profile


def _sector_flag_badge(pred: dict[str, Any]) -> str:
    if pred.get("is_fake_bullish"):
        return _pill("FAKE BULLISH", "#ff4d6d")
    if pred.get("index_contradicts"):
        return _pill("INDEX CONTRADICTS", "#f0b429")

    coverage_quality = str(pred.get("coverage_quality", "")).strip().upper()
    if coverage_quality == "VERY_LOW":
        return _pill("VERY LOW COVERAGE", "#ff4d6d")
    if coverage_quality == "LOW":
        return _pill("LOW COVERAGE", "#f0b429")

    signal_quality = str(pred.get("signal_quality", "")).strip().upper()
    if signal_quality == "HIGH":
        return _pill("HIGH QUALITY", "#00d4a8")
    if signal_quality == "MEDIUM":
        return _pill("MEDIUM QUALITY", "#0094ff")
    if signal_quality == "WEAK_BULLISH":
        return _pill("WEAK BULLISH", "#f0b429")
    if signal_quality == "LOW":
        return _pill("LOW QUALITY", "#ff4d6d")

    return _pill("CLEAN", "#00d4a8")


def _overall_flag_badge(pred: dict[str, Any]) -> str:
    coverage_quality = str(pred.get("coverage_quality", "")).strip().upper()
    if coverage_quality == "VERY_LOW":
        return _pill("VERY LOW COVERAGE", "#ff4d6d")
    if coverage_quality == "LOW":
        return _pill("LOW COVERAGE", "#f0b429")

    pressure = str(pred.get("market_pressure", "NEUTRAL")).upper()
    if pressure == "BULLISH_PRESSURE":
        return _pill("BULLISH PRESSURE", "#00d4a8")
    if pressure == "BEARISH_PRESSURE":
        return _pill("BEARISH PRESSURE", "#ff4d6d")
    if pred.get("nifty_contradicts_majority"):
        return _pill("NIFTY CONTRADICTS", "#f0b429")
    return _pill("BALANCED", "#0094ff")


def _sort_scan_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    for col in ("Final Score", "Prediction Score", "Score"):
        if col in out.columns:
            return out.sort_values(col, ascending=False, kind="stable")
    return out


def _cacheable_value(value: Any) -> Any:
    """Convert nested objects into a stable hashable structure for cache keys."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else round(float(value), 6)
    if isinstance(value, dict):
        return tuple((str(k), _cacheable_value(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple, set)):
        return tuple(_cacheable_value(v) for v in value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _freeze_rows(raw_rows: list[dict]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    """Freeze raw rows into a deterministic tuple so Streamlit can cache the pipeline result."""
    frozen_rows: list[tuple[tuple[str, Any], ...]] = []
    for row in raw_rows:
        frozen_rows.append(
            tuple((str(k), _cacheable_value(v)) for k, v in sorted(row.items(), key=lambda item: str(item[0])))
        )
    return tuple(frozen_rows)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_index_analysis(index_sector: str, _tt_date_key: str = "live") -> dict[str, Any]:
    return analyze_index(index_sector)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_pipeline_df(
    raw_rows_payload: tuple[tuple[tuple[str, Any], ...], ...],
    mode: int,
    market_bias_key: Any,
    _tt_date_key: str = "live",
    _market_bias: dict[str, Any] | None = None,
    _enhance_results_fn: Callable[[list[dict], int], pd.DataFrame] | None = None,
    _apply_enhanced_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    _apply_universal_grading_fn: Callable[[pd.DataFrame, dict | None], pd.DataFrame] | None = None,
    _apply_phase4_logic_fn: Callable[[pd.DataFrame, dict | None], pd.DataFrame] | None = None,
    _apply_phase42_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if not raw_rows_payload or _enhance_results_fn is None:
        return pd.DataFrame()

    raw_rows = [{k: v for k, v in row_items} for row_items in raw_rows_payload]
    try:
        df = _enhance_results_fn(raw_rows, mode)
        if _apply_enhanced_logic_fn is not None:
            df = _apply_enhanced_logic_fn(df)
        if _apply_universal_grading_fn is not None:
            df = _apply_universal_grading_fn(df, _market_bias)
        if _apply_phase4_logic_fn is not None:
            df = _apply_phase4_logic_fn(df, _market_bias)
        if _apply_phase42_logic_fn is not None:
            df = _apply_phase42_logic_fn(df)
        return _sort_scan_df(df)
    except Exception:
        return pd.DataFrame()


def render_sector_screener_dashboard(
    mode: int,
    enhance_results_fn: Callable[[list[dict], int], pd.DataFrame],
    apply_enhanced_logic_fn: Callable[[pd.DataFrame], pd.DataFrame],
    apply_universal_grading_fn: Callable[[pd.DataFrame, dict | None], pd.DataFrame],
    apply_phase4_logic_fn: Callable[[pd.DataFrame, dict | None], pd.DataFrame],
    apply_phase42_logic_fn: Callable[[pd.DataFrame], pd.DataFrame],
    compute_market_bias_fn: Callable[..., dict] | None = None,
) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-bottom:4px;">🔭 Maximum Basket Sector Screener Dashboard</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        'Click a sector card to scan the widest related basket we support. '
        'The new Overall card scans all sector baskets together and gives a weighted market probability.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not _SSE_OK:
        st.warning(
            "⚠️ Required sector screener engine functions are not available. "
            "Ensure `multi_index_market_bias_engine.py` loads successfully."
        )
        return

    try:
        import time_travel_engine as _tt_banner
        _tt_banner_date = _tt_banner.get_reference_date() if _tt_banner.is_active() else st.session_state.get("tt_date_val")
        if _tt_banner_date is not None:
            _tt_banner_str = _tt_banner_date.strftime("%d %b %Y") if _tt_banner_date else "unknown date"
            st.markdown(
                f'<div style="background:#1a0a00;border:1.5px solid #f0b429;'
                f'border-radius:8px;padding:8px 14px;margin-bottom:14px;'
                f'font-size:12px;color:#f0b429;">'
                f'🕰️ <b>TIME TRAVEL ACTIVE</b> — Sector data simulated at '
                f'<b>{_tt_banner_str}</b> post-market close. '
                f'All predictions reflect that historical date only.</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    try:
        import time_travel_engine as _tt_guard
        _tt_guard_key = (
            _tt_guard.is_active(),
            str(_tt_guard.get_reference_date()) if _tt_guard.is_active() else str(st.session_state.get("tt_date_val") or "live"),
        )
    except Exception:
        _tt_guard_key = (False, str(st.session_state.get("tt_date_val") or "live"))

    _prev_tt_guard_key = st.session_state.get("ss_screener_tt_date_guard")
    if _prev_tt_guard_key != _tt_guard_key:
        st.session_state["ss_screener_active_sector"] = None
        st.session_state["ss_screener_sector_result"] = None
        st.session_state["ss_screener_scan_all_done"] = False
        st.session_state["ss_screener_all_results"] = None
        st.session_state["ss_screener_all_overall"] = None
        st.session_state["ss_screener_sector_cache"] = {}
        st.session_state["ss_screener_pending_live_refresh"] = None
        st.session_state["ss_screener_pending_scan_all_live"] = False
        st.session_state["ss_screener_last_refresh_meta"] = {}
        st.session_state["ss_screener_tt_date_guard"] = _tt_guard_key
    else:
        st.session_state.setdefault("ss_screener_active_sector", None)
        st.session_state.setdefault("ss_screener_sector_result", None)
        st.session_state.setdefault("ss_screener_scan_all_done", False)
        st.session_state.setdefault("ss_screener_all_results", None)
        st.session_state.setdefault("ss_screener_all_overall", None)
        st.session_state.setdefault("ss_screener_sector_cache", {})
        st.session_state.setdefault("ss_screener_pending_live_refresh", None)
        st.session_state.setdefault("ss_screener_pending_scan_all_live", False)
        st.session_state.setdefault("ss_screener_last_refresh_meta", {})
        st.session_state.setdefault("ss_screener_tt_date_guard", _tt_guard_key)

    _base_sectors = get_dashboard_sector_labels(include_overall=False)
    _grid_sectors = get_dashboard_sector_labels(include_overall=True)
    _counts = {sec: get_dashboard_sector_count(sec) for sec in _grid_sectors}
    _active = st.session_state.get("ss_screener_active_sector")
    _index_to_dashboard = {
        get_dashboard_index_sector(sec): sec
        for sec in _base_sectors
    }

    def _friendly_sector_name(name: str) -> str:
        return _index_to_dashboard.get(str(name).strip(), str(name))

    def _get_market_bias() -> dict | None:
        try:
            import time_travel_engine as _tt_mb
            _mb_tt_key = str(_tt_mb.get_reference_date()) if _tt_mb.is_active() else "live"
        except Exception:
            _mb_tt_key = str(st.session_state.get("tt_date_val") or "live")

        _cached_mb = st.session_state.get("market_bias_result", None)
        _cached_mb_key = st.session_state.get("market_bias_tt_key", None)
        if isinstance(_cached_mb, dict) and _cached_mb_key == _mb_tt_key:
            return _cached_mb

        if compute_market_bias_fn is not None:
            try:
                _mb = compute_market_bias_fn(include_bank=True)
            except TypeError:
                _mb = compute_market_bias_fn()
            if isinstance(_mb, dict):
                st.session_state["market_bias_result"] = _mb
                st.session_state["market_bias_tt_key"] = _mb_tt_key
                return _mb
        return _cached_mb

    def _market_bias_key(market_bias: dict | None) -> Any:
        return _cacheable_value(market_bias) if isinstance(market_bias, dict) else None

    def _format_runtime_error(exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__

    def _run_pipeline(
        raw_rows: list[dict],
        market_bias: dict | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        if not raw_rows:
            return pd.DataFrame(), None
        mb = market_bias if isinstance(market_bias, dict) else _get_market_bias()
        try:
            import time_travel_engine as _tt_rp
            _rp_tt_key = str(_tt_rp.get_reference_date()) if _tt_rp.is_active() else "live"
        except Exception:
            _rp_tt_key = str(st.session_state.get("tt_date_val") or "live")
        try:
            df = _cached_pipeline_df(
                _freeze_rows(raw_rows),
                mode,
                _market_bias_key(mb),
                _tt_date_key=_rp_tt_key,
                _market_bias=mb,
                _enhance_results_fn=enhance_results_fn,
                _apply_enhanced_logic_fn=apply_enhanced_logic_fn,
                _apply_universal_grading_fn=apply_universal_grading_fn,
                _apply_phase4_logic_fn=apply_phase4_logic_fn,
                _apply_phase42_logic_fn=apply_phase42_logic_fn,
            )
            return df, None
        except Exception as cache_exc:
            try:
                df = enhance_results_fn(raw_rows, mode)
                df = apply_enhanced_logic_fn(df)
                df = apply_universal_grading_fn(df, mb)
                df = apply_phase4_logic_fn(df, mb)
                df = apply_phase42_logic_fn(df)
                return _sort_scan_df(df), None
            except Exception as pipeline_exc:
                return pd.DataFrame(), (
                    f"{_format_runtime_error(cache_exc)} | "
                    f"fallback: {_format_runtime_error(pipeline_exc)}"
                )

    def _apply_coverage_penalty(
        pred: dict[str, Any] | None,
        requested_count: int,
        raw_row_count: int,
        valid_row_count: int,
    ) -> dict[str, Any] | None:
        if pred is None:
            return None

        adjusted = dict(pred)
        requested = max(int(requested_count), 0)
        raw_rows = max(int(raw_row_count), 0)
        valid_rows = max(int(valid_row_count), 0)

        if requested <= 0:
            adjusted["coverage_pct"] = 0.0
            adjusted["coverage_quality"] = "LOW"
            adjusted["tomorrow_prediction"] = "SIDEWAYS"
            adjusted["sector_direction"] = "Sideways"
            adjusted["confidence"] = 35.0
            adjusted["bullish_probability"] = 50.0
            return adjusted

        raw_cov = min(1.0, raw_rows / requested)
        valid_cov = min(1.0, valid_rows / requested)
        effective_cov = min(raw_cov, valid_cov if valid_rows > 0 else raw_cov)

        base_prob = float(adjusted.get("bullish_probability", 50.0))
        base_conf = float(adjusted.get("confidence", 50.0))
        breadth_prob = float(adjusted.get("bullish_pct", base_prob))
        weighted_prob = float(adjusted.get("weighted_bullish_pct", breadth_prob))
        score_prob = float(
            adjusted.get("market_cap_weighted_score", adjusted.get("avg_score", 50.0))
        )
        pred_score_prob = float(
            adjusted.get("weighted_pred_score", adjusted.get("avg_pred_score", score_prob))
        )
        trap_pct = float(adjusted.get("trap_high_pct", 0.0))
        signal_quality = str(adjusted.get("signal_quality", "MEDIUM")).strip().upper()
        blended_prob = (
            0.36 * base_prob
            + 0.26 * weighted_prob
            + 0.20 * breadth_prob
            + 0.10 * score_prob
            + 0.08 * pred_score_prob
        )
        disagreement = (
            abs(base_prob - weighted_prob)
            + abs(base_prob - breadth_prob)
            + abs(weighted_prob - breadth_prob)
        ) / 3.0
        prob_shrink = max(0.30, min(1.0, 0.20 + 1.10 * effective_cov))
        conf_scale = max(0.35, min(1.0, 0.25 + effective_cov))
        agreement_scale = max(0.55, min(1.0, 1.0 - disagreement / 80.0))

        adjusted_prob = 50.0 + (blended_prob - 50.0) * prob_shrink * agreement_scale
        adjusted_conf = base_conf * conf_scale * agreement_scale
        min_valid_threshold = max(8, int(requested * 0.15))

        quality_prob_scale = {
            "HIGH": 1.04,
            "MEDIUM": 1.00,
            "WEAK_BULLISH": 0.84,
            "LOW": 0.74,
        }.get(signal_quality, 0.94)
        quality_conf_scale = {
            "HIGH": 1.05,
            "MEDIUM": 1.00,
            "WEAK_BULLISH": 0.84,
            "LOW": 0.76,
        }.get(signal_quality, 0.95)
        adjusted_prob = 50.0 + (adjusted_prob - 50.0) * quality_prob_scale
        adjusted_conf *= quality_conf_scale

        if 45.0 <= breadth_prob <= 55.0:
            adjusted_prob = 50.0 + (adjusted_prob - 50.0) * 0.82
            adjusted_conf *= 0.92

        if adjusted.get("index_contradicts"):
            adjusted_prob = 50.0 + (adjusted_prob - 50.0) * 0.75
            adjusted_conf *= 0.80

        if adjusted.get("is_fake_bullish"):
            adjusted_prob = min(adjusted_prob, 57.0)
            adjusted_conf *= 0.76

        if trap_pct >= 45.0:
            adjusted_prob = 50.0 + (adjusted_prob - 50.0) * 0.86
            adjusted_conf *= 0.88

        if effective_cov >= 0.75:
            coverage_quality = "HIGH"
        elif effective_cov >= 0.55:
            coverage_quality = "MEDIUM"
        elif effective_cov >= 0.35:
            coverage_quality = "LOW"
        else:
            coverage_quality = "VERY_LOW"

        if effective_cov < 0.35 or valid_rows < min_valid_threshold or adjusted_conf < 52.0:
            adjusted_direction = "Sideways"
            adjusted_tomorrow = "SIDEWAYS"
        elif adjusted_prob >= 60.0 and adjusted_conf >= 56.0:
            adjusted_direction = "Bullish"
            adjusted_tomorrow = "UP"
        elif adjusted_prob <= 40.0 and adjusted_conf >= 54.0:
            adjusted_direction = "Bearish"
            adjusted_tomorrow = "DOWN"
        else:
            adjusted_direction = "Sideways"
            adjusted_tomorrow = "SIDEWAYS"

        adjusted["bullish_probability"] = round(float(max(0.0, min(100.0, adjusted_prob))), 1)
        adjusted["confidence"] = round(float(max(0.0, min(100.0, adjusted_conf))), 1)
        adjusted["tomorrow_prediction"] = adjusted_tomorrow
        adjusted["sector_direction"] = adjusted_direction
        adjusted["coverage_pct"] = round(effective_cov * 100.0, 1)
        adjusted["raw_coverage_pct"] = round(raw_cov * 100.0, 1)
        adjusted["valid_coverage_pct"] = round(valid_cov * 100.0, 1)
        adjusted["coverage_quality"] = coverage_quality
        adjusted["signal_agreement_gap"] = round(disagreement, 1)
        adjusted["probability_model"] = "coverage+quality+agreement"

        if "weighted_bullish_pct" in adjusted:
            weighted_bp = float(adjusted.get("weighted_bullish_pct", 50.0))
            adjusted["weighted_bullish_pct"] = round(
                50.0 + (weighted_bp - 50.0) * prob_shrink * agreement_scale,
                1,
            )

        if coverage_quality in {"LOW", "VERY_LOW"}:
            adjusted["signal_quality"] = "LOW"

        return adjusted

    def _apply_overall_coverage_penalty(
        pred: dict[str, Any] | None,
        sector_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if pred is None:
            return None

        adjusted = dict(pred)
        sector_preds = [res.get("pred") for res in sector_results if res.get("pred") is not None]
        coverages = [
            float(sec_pred.get("coverage_pct", 0.0)) / 100.0
            for sec_pred in sector_preds
        ]
        if not coverages:
            adjusted["coverage_pct"] = 0.0
            adjusted["coverage_quality"] = "LOW"
            adjusted["overall_prediction"] = "SIDEWAYS"
            adjusted["tomorrow_prediction"] = "SIDEWAYS"
            adjusted["confidence"] = 35.0
            adjusted["bullish_probability"] = 50.0
            return adjusted

        avg_cov = sum(coverages) / len(coverages)
        good_ratio = sum(1 for cov in coverages if cov >= 0.55) / len(coverages)
        sector_probs = [
            float(sec_pred.get("bullish_probability", 50.0))
            for sec_pred in sector_preds
        ]
        directions = [
            str(sec_pred.get("tomorrow_prediction", "SIDEWAYS")).upper()
            for sec_pred in sector_preds
        ]
        up_count = sum(1 for direction in directions if direction == "UP")
        down_count = sum(1 for direction in directions if direction == "DOWN")
        side_count = sum(1 for direction in directions if direction == "SIDEWAYS")
        dominant_count = max(up_count, down_count, side_count)
        directional_consensus = dominant_count / len(directions) if directions else 0.0
        directional_edge = abs(up_count - down_count) / len(directions) if directions else 0.0
        base_prob = float(adjusted.get("bullish_probability", 50.0))
        base_conf = float(adjusted.get("confidence", 50.0))
        prob_shrink = max(0.35, min(1.0, 0.35 + 0.75 * avg_cov))
        conf_scale = max(0.40, min(1.0, 0.35 + 0.75 * avg_cov + 0.10 * good_ratio))
        consensus_scale = max(
            0.55,
            min(1.0, 0.30 + 0.70 * directional_consensus + 0.15 * directional_edge),
        )

        adjusted_prob = 50.0 + (base_prob - 50.0) * prob_shrink * consensus_scale
        adjusted_conf = base_conf * conf_scale * consensus_scale

        if sector_probs:
            avg_sector_prob = sum(sector_probs) / len(sector_probs)
            adjusted_prob = 0.70 * adjusted_prob + 0.30 * avg_sector_prob

        if side_count >= max(up_count, down_count):
            adjusted_prob = 50.0 + (adjusted_prob - 50.0) * 0.78
            adjusted_conf *= 0.90

        if avg_cov >= 0.75:
            coverage_quality = "HIGH"
        elif avg_cov >= 0.55:
            coverage_quality = "MEDIUM"
        elif avg_cov >= 0.35:
            coverage_quality = "LOW"
        else:
            coverage_quality = "VERY_LOW"

        if avg_cov < 0.45 or good_ratio < 0.50 or adjusted_conf < 52.0:
            adjusted_prediction = "SIDEWAYS"
        elif adjusted_prob >= 60.0 and adjusted_conf >= 55.0:
            adjusted_prediction = "BULLISH"
        elif adjusted_prob <= 40.0 and adjusted_conf >= 50.0:
            adjusted_prediction = "BEARISH"
        else:
            adjusted_prediction = "SIDEWAYS"

        adjusted["bullish_probability"] = round(float(max(0.0, min(100.0, adjusted_prob))), 1)
        adjusted["confidence"] = round(float(max(0.0, min(100.0, adjusted_conf))), 1)
        adjusted["overall_prediction"] = adjusted_prediction
        adjusted["tomorrow_prediction"] = {
            "BULLISH": "UP",
            "BEARISH": "DOWN",
        }.get(adjusted_prediction, "SIDEWAYS")
        adjusted["coverage_pct"] = round(avg_cov * 100.0, 1)
        adjusted["good_sector_ratio"] = round(good_ratio * 100.0, 1)
        adjusted["directional_consensus_pct"] = round(directional_consensus * 100.0, 1)
        adjusted["directional_edge_pct"] = round(directional_edge * 100.0, 1)
        adjusted["coverage_quality"] = coverage_quality
        adjusted["probability_model"] = "coverage+sector-consensus"
        return adjusted

    def _sector_cache_key(
        sector_name: str,
        market_bias: dict | None,
        overall: bool = False,
    ) -> tuple[Any, ...]:
        cache_kind = "overall" if overall else "sector"
        data_signature = (
            get_dashboard_data_signature(get_dashboard_sector_stocks("Overall"))
            if overall
            else get_dashboard_sector_signature(sector_name)
        )
        try:
            import time_travel_engine as _tt_ck
            tt_key = str(_tt_ck.get_reference_date()) if _tt_ck.is_active() else str(st.session_state.get("tt_date_val") or "live")
        except Exception:
            tt_key = str(st.session_state.get("tt_date_val") or "live")
        return (cache_kind, mode, sector_name, tt_key, data_signature, _market_bias_key(market_bias))

    def _build_sector_result(
        sector_name: str,
        market_bias: dict | None = None,
    ) -> dict[str, Any]:
        symbols = get_dashboard_sector_stocks(sector_name)
        raw_rows = build_dashboard_sector_raw_rows(
            sector_name,
            mode,
            preload_missing=False,
            workers=12,
        )
        raw_symbols = {
            str(row.get("Symbol", "")).upper().strip()
            for row in raw_rows
            if row.get("Symbol")
        }
        missing_symbols = [sym for sym in symbols if sym not in raw_symbols]

        if not raw_rows:
            result = {
                "kind": "sector",
                "sector": sector_name,
                "pred": None,
                "df": pd.DataFrame(),
                "symbols": symbols,
                "missing_symbols": symbols,
                "requested_count": len(symbols),
                "raw_row_count": 0,
                "valid_row_count": 0,
                "err": f"No raw rows found for {sector_name}.",
            }
            return result

        df, pipeline_err = _run_pipeline(raw_rows, market_bias=market_bias)
        if df.empty:
            result = {
                "kind": "sector",
                "sector": sector_name,
                "pred": None,
                "df": pd.DataFrame(),
                "symbols": symbols,
                "missing_symbols": missing_symbols,
                "requested_count": len(symbols),
                "raw_row_count": len(raw_rows),
                "valid_row_count": 0,
                "err": (
                    f"{sector_name} pipeline could not complete: {pipeline_err}"
                    if pipeline_err
                    else f"{sector_name} scan returned no valid rows after the full pipeline."
                ),
            }
            return result

        try:
            import time_travel_engine as _tt_cia
            _cia_tt_key = str(_tt_cia.get_reference_date()) if _tt_cia.is_active() else "live"
        except Exception:
            _cia_tt_key = str(st.session_state.get("tt_date_val") or "live")
        index_analysis = _cached_index_analysis(
            get_dashboard_index_sector(sector_name),
            _tt_date_key=_cia_tt_key,
        )
        pred = compute_sector_prediction_enhanced(
            sector_name,
            df,
            index_analysis,
        )
        pred = _apply_coverage_penalty(
            pred,
            requested_count=len(symbols),
            raw_row_count=len(raw_rows),
            valid_row_count=len(df),
        )

        result = {
            "kind": "sector",
            "sector": sector_name,
            "pred": pred,
            "df": df,
            "symbols": symbols,
            "missing_symbols": missing_symbols,
            "requested_count": len(symbols),
            "raw_row_count": len(raw_rows),
            "valid_row_count": len(df),
            "err": None,
        }
        return result

    def _scan_sector_result(
        sector_name: str,
        force_refresh: bool = False,
        market_bias: dict | None = None,
        preloaded: bool = False,
    ) -> dict[str, Any]:
        cache = st.session_state.setdefault("ss_screener_sector_cache", {})
        mb = market_bias if isinstance(market_bias, dict) else _get_market_bias()
        if not preloaded:
            preload_dashboard_sector_data(
                sector_name,
                workers=12,
                force_live_refresh=force_refresh,
            )

        cache_key = _sector_cache_key(sector_name, mb)
        if not force_refresh and cache_key in cache:
            return cache[cache_key]

        result = _build_sector_result(sector_name, market_bias=mb)
        cache[cache_key] = result
        return result

    def _compute_overall_result(sector_results: list[dict[str, Any]]) -> dict[str, Any]:
        overall_inputs: dict[str, dict] = {}
        combined_frames: list[pd.DataFrame] = []

        for sec_res in sector_results:
            sec = str(sec_res.get("sector", "")).strip()
            sec_pred = sec_res.get("pred")
            sec_df = sec_res.get("df", pd.DataFrame())
            if sec_pred is not None and isinstance(sec_df, pd.DataFrame) and not sec_df.empty:
                overall_inputs[get_dashboard_index_sector(sec)] = sec_pred
                tagged_df = sec_df.copy()
                tagged_df.insert(0, "Sector Basket", sec)
                combined_frames.append(tagged_df)

        if not overall_inputs:
            result = {
                "kind": "overall",
                "sector": "Overall",
                "pred": None,
                "df": pd.DataFrame(),
                "sector_results": sector_results,
                "symbols": get_dashboard_sector_stocks("Overall"),
                "missing_symbols": get_dashboard_sector_stocks("Overall"),
                "requested_count": get_dashboard_sector_count("Overall"),
                "raw_row_count": 0,
                "valid_row_count": 0,
                "unique_valid_count": 0,
                "err": "No sector data available for the overall scan.",
            }
            return result

        overall_data = compute_overall_market_enhanced(overall_inputs)
        combined_df = (
            _sort_scan_df(pd.concat(combined_frames, ignore_index=True))
            if combined_frames else pd.DataFrame()
        )

        overall_symbols = get_dashboard_sector_stocks("Overall")
        unique_valid_symbols = []
        if isinstance(combined_df, pd.DataFrame) and not combined_df.empty and "Symbol" in combined_df.columns:
            unique_valid_symbols = sorted({
                str(sym).upper().strip()
                for sym in combined_df["Symbol"].dropna().tolist()
            })
        missing_symbols = [sym for sym in overall_symbols if sym not in set(unique_valid_symbols)]

        overall_prediction = str(overall_data.get("overall_prediction", "SIDEWAYS")).upper()
        overall_tomorrow = {
            "BULLISH": "UP",
            "BEARISH": "DOWN",
        }.get(overall_prediction, "SIDEWAYS")

        overall_pred = {
            "tomorrow_prediction": overall_tomorrow,
            "bullish_probability": float(overall_data.get("weighted_score", 50.0)),
            "confidence": float(overall_data.get("confidence", 50.0)),
            "overall_prediction": overall_prediction,
            "strongest_sector": _friendly_sector_name(str(overall_data.get("strongest_sector", "N/A"))),
            "weakest_sector": _friendly_sector_name(str(overall_data.get("weakest_sector", "N/A"))),
            "market_pressure": str(overall_data.get("market_pressure", "NEUTRAL")),
            "dominant_sector_score": float(overall_data.get("dominant_sector_score", 50.0)),
            "bank_influence": float(overall_data.get("bank_influence", 0.0)),
            "nifty_contradicts_majority": bool(overall_data.get("nifty_contradicts_majority", False)),
            "top_sectors": [_friendly_sector_name(str(s)) for s in overall_data.get("top_sectors", [])],
            "weak_sectors": [_friendly_sector_name(str(s)) for s in overall_data.get("weak_sectors", [])],
        }
        overall_pred = _apply_overall_coverage_penalty(overall_pred, sector_results)

        result = {
            "kind": "overall",
            "sector": "Overall",
            "pred": overall_pred,
            "df": combined_df,
            "sector_results": sector_results,
            "symbols": overall_symbols,
            "missing_symbols": missing_symbols,
            "requested_count": len(overall_symbols),
            "raw_row_count": sum(int(res.get("raw_row_count", 0)) for res in sector_results),
            "valid_row_count": len(combined_df) if isinstance(combined_df, pd.DataFrame) else 0,
            "unique_valid_count": len(unique_valid_symbols),
            "err": None,
        }
        return result

    def _scan_overall_result(
        force_refresh: bool = False,
        market_bias: dict | None = None,
        precomputed_sector_results: list[dict[str, Any]] | None = None,
        preloaded: bool = False,
    ) -> dict[str, Any]:
        cache = st.session_state.setdefault("ss_screener_sector_cache", {})
        mb = market_bias if isinstance(market_bias, dict) else _get_market_bias()
        if precomputed_sector_results is None:
            if not preloaded:
                preload_dashboard_sector_data(
                    "Overall",
                    workers=12,
                    force_live_refresh=force_refresh,
                )
            cache_key = _sector_cache_key("Overall", mb, overall=True)
            if not force_refresh and cache_key in cache:
                return cache[cache_key]
            _max_workers = max(1, min(6, len(_base_sectors)))
            _sector_results_map: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=_max_workers) as ex:
                _futures = {
                    ex.submit(_build_sector_result, sec, mb): sec
                    for sec in _base_sectors
                }
                for _future in as_completed(_futures):
                    _sec = _futures[_future]
                    try:
                        _sec_result = _future.result()
                    except Exception as exc:
                        _sec_result = {
                            "kind": "sector",
                            "sector": _sec,
                            "pred": None,
                            "df": pd.DataFrame(),
                            "symbols": get_dashboard_sector_stocks(_sec),
                            "missing_symbols": get_dashboard_sector_stocks(_sec),
                            "requested_count": get_dashboard_sector_count(_sec),
                            "raw_row_count": 0,
                            "valid_row_count": 0,
                            "err": str(exc),
                        }
                    _sector_results_map[_sec] = _sec_result
                    cache[_sector_cache_key(_sec, mb)] = _sec_result
            sector_results = [_sector_results_map[_sec] for _sec in _base_sectors if _sec in _sector_results_map]
        else:
            cache_key = _sector_cache_key("Overall", mb, overall=True)
            sector_results = list(precomputed_sector_results)

        result = _compute_overall_result(sector_results)
        cache[cache_key] = result
        return result

    def _remember_scan_df(result: dict[str, Any]) -> None:
        df = result.get("df")
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.session_state["ss_screener_last_scan_df"] = df.copy()
            st.session_state["last_scan_df"] = df.copy()

    def _result_to_summary_row(result: dict[str, Any]) -> dict[str, Any]:
        pred = result.get("pred")
        if pred is None:
            requested = max(int(result.get("requested_count", 0)), 0)
            valid = max(int(result.get("valid_row_count", 0)), 0)
            coverage_pct = round((valid / requested) * 100.0, 1) if requested > 0 else 0.0
            return {
                "Sector": result.get("sector", "—"),
                "Prediction": "SIDEWAYS",
                "Probability %": 50.0,
                "Confidence %": 40.0,
                "Universe": int(result.get("requested_count", 0)),
                "Valid Rows": int(result.get("valid_row_count", 0)),
                "Coverage %": coverage_pct,
            }

        return {
            "Sector": result.get("sector", "—"),
            "Prediction": str(pred.get("tomorrow_prediction", "SIDEWAYS")).upper(),
            "Probability %": round(float(pred.get("bullish_probability", 50.0)), 1),
            "Confidence %": round(float(pred.get("confidence", 40.0)), 1),
            "Universe": int(result.get("requested_count", 0)),
            "Valid Rows": int(result.get("valid_row_count", 0)),
            "Coverage %": round(float(pred.get("coverage_pct", 0.0)), 1),
        }

    def _reset_live_runtime(clear_results: bool = False) -> None:
        try:
            _cached_pipeline_df.clear()
        except Exception:
            pass
        try:
            _cached_index_analysis.clear()
        except Exception:
            pass
        try:
            clear_index_cache()
        except Exception:
            pass
        st.session_state["ss_screener_sector_cache"] = {}
        for key in ("market_bias_result", "market_bias_ts", "market_bias_tt_key"):
            st.session_state.pop(key, None)
        if clear_results:
            st.session_state["ss_screener_sector_result"] = None
            st.session_state["ss_screener_scan_all_done"] = False
            st.session_state["ss_screener_all_results"] = None
            st.session_state["ss_screener_all_overall"] = None

    def _record_scan_meta(
        result: dict[str, Any],
        *,
        scope_label: str,
        live_requested: bool,
    ) -> dict[str, Any]:
        profile = _build_data_profile(result.get("symbols", []))
        meta = {
            "scope": scope_label,
            "live_requested": bool(live_requested),
            "refreshed_at": _safe_dt_label(_now_ist()),
            "profile": profile,
        }
        out = dict(result)
        out["scan_meta"] = meta
        st.session_state["ss_screener_last_refresh_meta"] = meta
        return out

    def _meta_pill_row(result: dict[str, Any]) -> str:
        meta = result.get("scan_meta", {}) if isinstance(result, dict) else {}
        profile = meta.get("profile", {}) if isinstance(meta, dict) else {}
        source = str(profile.get("primary_source", "NO DATA"))
        source_color = _source_color(source)
        market_date = str(profile.get("market_date", "—"))
        window = str(profile.get("window", "—"))
        refreshed_at = str(meta.get("refreshed_at", "Not yet"))
        captured_at = str(profile.get("captured_at", "") or "").strip()
        live_count = int(profile.get("live", 0))
        csv_count = int(profile.get("csv", 0))
        missing_count = int(profile.get("missing", 0))
        policy = "LIVE REFRESH" if bool(meta.get("live_requested")) else "SHARED CACHE OK"
        return (
            '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 12px 0;">'
            + _pill(source, source_color)
            + _pill(f"{live_count} live", "#00d4a8", "15")
            + _pill(f"{csv_count} csv", "#f0b429", "15")
            + _pill(f"{missing_count} missing", "#ff4d6d", "15")
            + _pill(f"Market date {market_date}", "#8ab4d8", "15")
            + _pill(f"Window {window}", "#8ab4d8", "15")
            + (_pill(f"Captured {captured_at}", "#22c55e", "15") if captured_at else "")
            + _pill(policy, "#0094ff", "15")
            + _pill(refreshed_at, "#8ab4d8", "15")
            + '</div>'
        )

    _last_refresh_meta = st.session_state.get("ss_screener_last_refresh_meta", {})
    _last_profile = _last_refresh_meta.get("profile", {}) if isinstance(_last_refresh_meta, dict) else {}
    _tt_live_blocked = bool(_tt_guard_key[0])
    _tracked_total = _counts.get("Overall", 0)
    _refresh_scope = _active or "Overall"
    _source_value = str(_last_profile.get("primary_source", "READY")) if _last_profile else "READY"
    _source_caption = (
        f"Live {int(_last_profile.get('live', 0))} • CSV {int(_last_profile.get('csv', 0))} • Missing {int(_last_profile.get('missing', 0))}"
        if _last_profile else
        "No sector scan has been run in this session yet."
    )
    _market_date_value = str(_last_profile.get("market_date", "—")) if _last_profile else "—"
    _market_date_caption = (
        f"Window {str(_last_profile.get('window', '—'))}"
        if _last_profile else
        "Will populate after the next sector scan."
    )
    if _last_profile:
        _market_date_caption = (
            f"Window {str(_last_profile.get('window', 'â€”'))} â€¢ Captured {str(_last_profile.get('captured_at', 'Not yet'))}"
        )
    _refresh_value = str(_last_refresh_meta.get("refreshed_at", "Not yet")) if isinstance(_last_refresh_meta, dict) else "Not yet"
    _refresh_caption = (
        f"{'Live refresh requested' if _last_refresh_meta.get('live_requested') else 'Shared cache reuse allowed'} • Scope {_last_refresh_meta.get('scope', '—')}"
        if isinstance(_last_refresh_meta, dict) and _last_refresh_meta else
        "Use Refresh Live to bypass same-session sector cache."
    )
    _context_value = "TIME TRAVEL" if _tt_live_blocked else "LIVE MODE"
    _context_caption = (
        "Historical simulation active — live refresh is disabled."
        if _tt_live_blocked else
        f"Primary quick refresh targets {_refresh_scope}."
    )

    _status_cols = st.columns(5)
    _status_markup = [
        _status_card("Context", _context_value, _context_caption, "#f0b429" if _tt_live_blocked else "#00d4a8"),
        _status_card("Source Mix", _source_value, _source_caption, _source_color(_source_value)),
        _status_card("Market Date", _market_date_value, _market_date_caption, "#8ab4d8"),
        _status_card("Last Refresh", _refresh_value, _refresh_caption, "#0094ff"),
        _status_card("Universe", f"{_tracked_total:,}", "Tracked names in the broad Overall basket.", "#22c55e"),
    ]
    for _col, _card_markup in zip(_status_cols, _status_markup):
        with _col:
            st.markdown(_card_markup, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _live_refresh_window = _get_feature_window() == "LIVE"
    _act_close_col, _act_refresh_col, _act_refresh_all_col, _act_note_col = st.columns([1.0, 1.6, 1.8, 3.6])
    with _act_close_col:
        _close_clicked = st.button("Close Screener", key="ss_screener_close_btn", width="stretch")
    with _act_refresh_col:
        if _live_refresh_window:
            _live_refresh_clicked = st.button(f"Refresh Live {_refresh_scope}", key="ss_screener_refresh_live_btn", disabled=_tt_live_blocked, width="stretch")
            f"🔄 Refresh Live {_refresh_scope}",
            # key="ss_screener_refresh_live_btn",
            # disabled=_tt_live_blocked,
            # width="stretch",
        else:
            _live_refresh_clicked = False
            st.caption("Live refresh opens during market hours only.")
    with _act_refresh_all_col:
        _refresh_all_live_clicked = st.button(
            "📡 Refresh Live All Sectors",
            key="ss_screener_refresh_all_live_btn",
            disabled=(not _live_refresh_window) or _tt_live_blocked,
            width="stretch",
        )
    with _act_note_col:
        st.markdown(
            '<div style="background:#0b1017;border:1px solid #1e3a5f;border-radius:12px;'
            'padding:12px 14px;font-size:12px;color:#8ab4d8;line-height:1.7;">'
            f'<b style="color:#ccd9e8;">Live refresh behavior:</b> '
            f'{"Disabled while Time Travel is active." if _tt_live_blocked else "Clears sector caches, refreshes market bias, and re-downloads the selected basket from the live loader path."}'
            f'<br><b style="color:#ccd9e8;">Prediction meaning:</b> Probability is a directional confidence score built from coverage, signal quality, and sector/index agreement. It is not a guaranteed hit rate.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="background:linear-gradient(90deg,#0b1017 0%,#101a28 100%);'
        'border:1px solid #16324a;border-radius:14px;padding:14px 16px;margin:14px 0 18px 0;">'
        '<div style="font-size:11px;color:#4a6480;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:6px;">Reality Check</div>'
        '<div style="font-size:13px;color:#ccd9e8;line-height:1.7;">'
        'Use <b>Refresh Live</b> when you want the newest loader path data. The source pills below will tell you whether the basket is live, mixed, or CSV-backed, and the probability card should be read as a directional edge rather than a guaranteed next-day outcome.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if _close_clicked:
        st.session_state["ss_screener_active_sector"] = None
        st.session_state["ss_screener_sector_result"] = None
        st.session_state["ss_screener_scan_all_done"] = False
        st.session_state["ss_screener_all_results"] = None
        st.session_state["ss_screener_all_overall"] = None
        st.session_state["ss_screener_sector_cache"] = {}
        st.session_state["ss_screener_pending_live_refresh"] = None
        st.session_state["ss_screener_pending_scan_all_live"] = False
        st.session_state["ss_screener_last_refresh_meta"] = {}
        st.session_state["ss_screener_tt_date_guard"] = None
        st.session_state["show_sector_screener"] = False
        st.rerun()

    if _live_refresh_clicked:
        _reset_live_runtime(clear_results=True)
        st.session_state["ss_screener_active_sector"] = _refresh_scope
        st.session_state["ss_screener_pending_live_refresh"] = _refresh_scope
        st.session_state["ss_screener_pending_scan_all_live"] = False
        st.rerun()

    if _refresh_all_live_clicked:
        _reset_live_runtime(clear_results=True)
        st.session_state["ss_screener_active_sector"] = None
        st.session_state["ss_screener_pending_live_refresh"] = None
        st.session_state["ss_screener_pending_scan_all_live"] = True
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:14px;font-weight:700;color:#8ab4d8;'
        'letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">'
        '📊 Sector Grid</div>',
        unsafe_allow_html=True,
    )

    _cols_per_row = 4
    _chunks = [_grid_sectors[i: i + _cols_per_row] for i in range(0, len(_grid_sectors), _cols_per_row)]
    for _chunk in _chunks:
        _grid_cols = st.columns(len(_chunk))
        for _col, _sector in zip(_grid_cols, _chunk):
            _count = _counts.get(_sector, 0)
            _desc = get_dashboard_sector_description(_sector)
            _is_active = _active == _sector
            _card_border = "#00d4a8" if _is_active else "#1e3a5f"
            _card_bg = "#0d1e16" if _is_active else "#0b1017"
            with _col:
                st.markdown(
                    f'<div style="background:{_card_bg};border:1.5px solid {_card_border};'
                    'border-radius:12px;padding:14px 16px;margin-bottom:6px;">'
                    f'<div style="font-weight:900;font-size:14px;color:#ccd9e8;">{_sector}</div>'
                    f'<div style="font-size:10px;color:#4a6480;margin:3px 0 8px;">{_desc}</div>'
                    f'<div style="font-size:12px;color:#8ab4d8;">📦 {_count} stocks</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"🔍 Scan {_sector}",
                    key=f"ss_scan_{_sector}",
                    width="stretch",
                ):
                    st.session_state["ss_screener_active_sector"] = _sector
                    st.session_state["ss_screener_sector_result"] = None
                    st.session_state["ss_screener_pending_live_refresh"] = None
                    st.session_state["ss_screener_pending_scan_all_live"] = False
                    st.rerun()

    if _active and st.session_state.get("ss_screener_sector_result") is None:
        _count = _counts.get(_active, 0)
        _force_live_active = (
            not _tt_live_blocked
            and str(st.session_state.get("ss_screener_pending_live_refresh") or "") == str(_active)
        )
        _spinner_label = (
            f"🔍 Scanning {_active} across {_count} stocks..."
            if _active != "Overall"
            else f"🔍 Scanning Overall market basket across {_count} stocks..."
        )
        if _force_live_active:
            _spinner_label = (
                f"Refreshing live data for {_active} across {_count} stocks..."
                if _active != "Overall"
                else f"Refreshing live data for Overall market basket across {_count} stocks..."
            )
        with st.spinner(_spinner_label):
            try:
                _tt_date = st.session_state.get("tt_date_val")
                _tt_activated = False
                _tt_sse = None
                try:
                    import time_travel_engine as _tt_sse
                    if _tt_date is not None and _tt_sse is not None:
                        _tt_sse.activate(_tt_date)
                        _tt_activated = True
                except Exception:
                    pass
                try:
                    _market_bias = _get_market_bias()
                    preload_dashboard_sector_data(
                        "Overall" if _active == "Overall" else _active,
                        workers=12,
                        force_live_refresh=_force_live_active,
                    )
                    result = (
                        _scan_overall_result(
                            force_refresh=_force_live_active,
                            market_bias=_market_bias,
                            preloaded=True,
                        )
                        if _active == "Overall"
                        else _scan_sector_result(
                            _active,
                            force_refresh=_force_live_active,
                            market_bias=_market_bias,
                            preloaded=True,
                        )
                    )
                finally:
                    if _tt_activated:
                        try:
                            _tt_sse.restore()
                        except Exception:
                            pass
            except Exception as exc:
                result = {
                    "kind": "overall" if _active == "Overall" else "sector",
                    "sector": _active,
                    "pred": None,
                    "df": pd.DataFrame(),
                    "symbols": [],
                    "missing_symbols": [],
                    "requested_count": _count,
                    "raw_row_count": 0,
                    "valid_row_count": 0,
                    "err": str(exc),
                }
            result = _record_scan_meta(
                result,
                scope_label=_active,
                live_requested=_force_live_active,
            )
            st.session_state["ss_screener_sector_result"] = result
            st.session_state["ss_screener_pending_live_refresh"] = None
            _remember_scan_df(result)

    if _active and st.session_state.get("ss_screener_sector_result") is not None:
        _res = st.session_state["ss_screener_sector_result"]
        _pred = _res.get("pred")
        _df = _res.get("df", pd.DataFrame())
        _err = _res.get("err")
        _symbols = _res.get("symbols", []) or []
        _missing = _res.get("missing_symbols", []) or []

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:14px;font-weight:800;color:#8ab4d8;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:10px;">📈 {_active} Scan Result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(_meta_pill_row(_res), unsafe_allow_html=True)

        if _err:
            st.error(_err)
        elif _pred is None:
            st.warning("No prediction is available for this scan.")
        elif _res.get("kind") == "overall":
            _tom = str(_pred.get("tomorrow_prediction", "SIDEWAYS")).upper()
            _prob = float(_pred.get("bullish_probability", 50.0))
            _conf = float(_pred.get("confidence", 50.0))
            _main_color = _pred_color(_tom)

            st.markdown(
                f'<div style="background:#0b1017;border:2px solid {_main_color};'
                'border-radius:16px;padding:20px 22px;margin-bottom:14px;">'
                '<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;">'
                '<div style="min-width:220px;">'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
                'Overall Market Prediction</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:34px;font-weight:900;color:{_main_color};line-height:1.0;">'
                f'{_pred_icon(_tom)} {_tom}</div>'
                '</div>'
                '<div style="margin-left:auto;text-align:right;min-width:180px;">'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Probability</div>'
                f'<div style="font-size:30px;font-weight:900;color:{_prob_color(_prob)};">{_prob:.0f}%</div>'
                '</div>'
                '</div>'
                '<div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;align-items:center;">'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Confidence</div>'
                f'<div style="font-size:20px;font-weight:800;color:#0094ff;">{_conf:.0f}%</div></div>'
                f'<div>{_overall_flag_badge(_pred)}</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            _overall_insight_cols = st.columns(4)
            for _col, _markup in zip(_overall_insight_cols, _result_insight_cards(_res, _pred)):
                with _col:
                    st.markdown(_markup, unsafe_allow_html=True)

            _mcols = st.columns(7)
            with _mcols[0]:
                st.metric("Unique Stocks", f"{int(_res.get('requested_count', 0)):,}")
            with _mcols[1]:
                st.metric("Unique Valid", f"{int(_res.get('unique_valid_count', 0)):,}")
            with _mcols[2]:
                st.metric("Combined Rows", f"{int(_res.get('valid_row_count', 0)):,}")
            with _mcols[3]:
                st.metric("Coverage", f"{float(_pred.get('coverage_pct', 0.0)):.1f}%")
            with _mcols[4]:
                st.metric("Consensus", f"{float(_pred.get('directional_consensus_pct', 0.0)):.1f}%")
            with _mcols[5]:
                st.metric("Skipped", f"{len(_missing):,}")
            with _mcols[6]:
                st.metric("Strongest", str(_pred.get("strongest_sector", "—")))
            st.caption(
                "Weakest sector: "
                + str(_pred.get("weakest_sector", "—"))
                + f" | Coverage quality: {str(_pred.get('coverage_quality', 'MEDIUM')).replace('_', ' ')}"
            )

            _sector_rows = []
            for sec_res in _res.get("sector_results", []):
                sec_pred = sec_res.get("pred")
                if sec_pred is None:
                    _sector_rows.append({
                        "Sector": sec_res.get("sector", "—"),
                        "Prediction": "SIDEWAYS",
                        "Probability %": 50.0,
                        "Confidence %": 40.0,
                        "Universe": int(sec_res.get("requested_count", 0)),
                        "Valid Rows": int(sec_res.get("valid_row_count", 0)),
                        "Coverage %": 0.0,
                    })
                else:
                    _sector_rows.append({
                        "Sector": sec_res.get("sector", "—"),
                        "Prediction": str(sec_pred.get("tomorrow_prediction", "SIDEWAYS")).upper(),
                        "Probability %": round(float(sec_pred.get("bullish_probability", 50.0)), 1),
                        "Confidence %": round(float(sec_pred.get("confidence", 40.0)), 1),
                        "Universe": int(sec_res.get("requested_count", 0)),
                        "Valid Rows": int(sec_res.get("valid_row_count", 0)),
                        "Coverage %": round(float(sec_pred.get("coverage_pct", 0.0)), 1),
                    })

            st.markdown(
                '<div style="font-size:13px;font-weight:800;color:#8ab4d8;margin:12px 0 8px;">📊 Sector Breakdown</div>',
                unsafe_allow_html=True,
            )
            if _sector_rows:
                _sector_df = pd.DataFrame(_sector_rows).sort_values("Probability %", ascending=False, kind="stable")
                _top_sector_cards = _sector_df.head(3).to_dict("records")
                if _top_sector_cards:
                    _top_sector_cols = st.columns(len(_top_sector_cards))
                    for _col, _row in zip(_top_sector_cols, _top_sector_cards):
                        with _col:
                            st.markdown(_top_sector_card(_row), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(_sector_df, width="stretch", hide_index=True)

            with st.expander(f"📋 All Covered Symbols — {len(_symbols)} unique", expanded=False):
                st.dataframe(pd.DataFrame({"Symbol": _symbols}), width="stretch", hide_index=True)

            if _missing:
                with st.expander(f"⚠️ Missing / Skipped Symbols — {len(_missing)}", expanded=False):
                    st.dataframe(pd.DataFrame({"Symbol": _missing}), width="stretch", hide_index=True)
                    try:
                        import time_travel_engine as _tt_missing
                        if _tt_missing.is_active():
                            st.caption(
                                "??? In Time Travel mode, some symbols may be missing because "
                                "they had no price data before the selected cutoff date. "
                                "This is expected - not a data error."
                            )
                    except Exception:
                        pass

            if isinstance(_df, pd.DataFrame) and not _df.empty:
                with st.expander(f"🧾 Combined scan details — {len(_df)} rows", expanded=False):
                    st.dataframe(_df, width="stretch", hide_index=True)

        else:
            _tom = str(_pred.get("tomorrow_prediction", "SIDEWAYS")).upper()
            _prob = float(_pred.get("bullish_probability", 50.0))
            _conf = float(_pred.get("confidence", 50.0))
            _main_color = _pred_color(_tom)

            st.markdown(
                f'<div style="background:#0b1017;border:2px solid {_main_color};'
                'border-radius:16px;padding:20px 22px;margin-bottom:14px;">'
                '<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;">'
                '<div style="min-width:220px;">'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
                'Tomorrow Prediction</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:34px;font-weight:900;color:{_main_color};line-height:1.0;">'
                f'{_pred_icon(_tom)} {_tom}</div>'
                '</div>'
                '<div style="margin-left:auto;text-align:right;min-width:180px;">'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Probability</div>'
                f'<div style="font-size:30px;font-weight:900;color:{_prob_color(_prob)};">{_prob:.0f}%</div>'
                '</div>'
                '</div>'
                '<div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;align-items:center;">'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Confidence</div>'
                f'<div style="font-size:20px;font-weight:800;color:#0094ff;">{_conf:.0f}%</div></div>'
                f'<div>{_sector_flag_badge(_pred)}</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            _sector_insight_cols = st.columns(4)
            for _col, _markup in zip(_sector_insight_cols, _result_insight_cards(_res, _pred)):
                with _col:
                    st.markdown(_markup, unsafe_allow_html=True)

            _mcols = st.columns(8)
            with _mcols[0]:
                st.metric("Universe Size", f"{int(_res.get('requested_count', 0)):,}")
            with _mcols[1]:
                st.metric("Raw Rows", f"{int(_res.get('raw_row_count', 0)):,}")
            with _mcols[2]:
                st.metric("Valid Rows", f"{int(_res.get('valid_row_count', 0)):,}")
            with _mcols[3]:
                st.metric("Coverage", f"{float(_pred.get('coverage_pct', 0.0)):.1f}%")
            with _mcols[4]:
                st.metric("Signal", str(_pred.get("signal_quality", "MEDIUM")).replace("_", " "))
            with _mcols[5]:
                st.metric("Skipped", f"{len(_missing):,}")
            with _mcols[6]:
                st.metric("Bullish %", f"{float(_pred.get('bullish_pct', 50.0)):.0f}%")
            with _mcols[7]:
                st.metric("Avg Score", f"{float(_pred.get('avg_score', 50.0)):.1f}")
            st.caption(
                f"Coverage quality: {str(_pred.get('coverage_quality', 'MEDIUM')).replace('_', ' ')}"
                + f" | Agreement gap: {float(_pred.get('signal_agreement_gap', 0.0)):.1f}"
                + f" | Model: {str(_pred.get('probability_model', 'base')).replace('+', ' + ')}"
            )

            _top_stocks = _pred.get("top_stocks", []) or []
            st.markdown(
                '<div style="font-size:13px;font-weight:800;color:#8ab4d8;margin:12px 0 8px;">🏆 Top Stocks</div>',
                unsafe_allow_html=True,
            )
            if _top_stocks:
                _top_stock_cards = _top_stocks[:3]
                _top_stock_cols = st.columns(len(_top_stock_cards))
                for _col, _stock in zip(_top_stock_cols, _top_stock_cards):
                    with _col:
                        st.markdown(_top_stock_card(_stock), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Symbol": s.get("symbol", "—"),
                            "Score": round(float(s.get("score", 0.0)), 1),
                            "Signal": s.get("signal", "—"),
                            "Grade": s.get("grade", ""),
                            "Conf %": round(float(s.get("conf", 0.0)), 1),
                        }
                        for s in _top_stocks
                    ]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No top stocks are available for this sector.")

            with st.expander(f"📋 Full Sector Basket — {len(_symbols)} symbols", expanded=False):
                st.dataframe(pd.DataFrame({"Symbol": _symbols}), width="stretch", hide_index=True)

            if _missing:
                with st.expander(f"⚠️ Missing / Skipped Symbols — {len(_missing)}", expanded=False):
                    st.dataframe(pd.DataFrame({"Symbol": _missing}), width="stretch", hide_index=True)
                    try:
                        import time_travel_engine as _tt_missing
                        if _tt_missing.is_active():
                            st.caption(
                                "??? In Time Travel mode, some symbols may be missing because "
                                "they had no price data before the selected cutoff date. "
                                "This is expected - not a data error."
                            )
                    except Exception:
                        pass

            if isinstance(_df, pd.DataFrame) and not _df.empty:
                with st.expander(f"🧾 Full scan details — {len(_df)} rows", expanded=False):
                    st.dataframe(_df, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:14px;font-weight:700;color:#8ab4d8;'
        'letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">'
        '🔄 All Sector + Overall Scan</div>',
        unsafe_allow_html=True,
    )

    _scan_all_col1, _scan_all_col2 = st.columns([2, 5])
    with _scan_all_col1:
        _scan_all_btn = st.button(
            "🚀 Scan All Sectors + Overall",
            key="ss_screener_scan_all_btn",
            width="stretch",
        )
    with _scan_all_col2:
        st.markdown(
            f'<div style="font-size:12px;color:#4a6480;padding-top:10px;">'
            f'Will scan {len(_base_sectors)} sector baskets plus the new Overall weighted market result. '
            f'Total tracked symbols: {get_dashboard_sector_count("Overall"):,}.</div>',
            unsafe_allow_html=True,
        )

    _scan_all_live_queued = bool(st.session_state.get("ss_screener_pending_scan_all_live", False))

    if _scan_all_btn or _scan_all_live_queued:
        _force_live_all = bool(_scan_all_live_queued)
        st.session_state["ss_screener_pending_scan_all_live"] = False
        st.session_state["ss_screener_pending_live_refresh"] = None
        st.session_state["ss_screener_scan_all_done"] = False
        st.session_state["ss_screener_all_results"] = []
        st.session_state["ss_screener_all_overall"] = None
        st.session_state["ss_screener_active_sector"] = None
        st.session_state["ss_screener_sector_result"] = None

        _tt_date = st.session_state.get("tt_date_val")
        _tt_activated_all = False
        _tt_sse_all = None
        try:
            import time_travel_engine as _tt_sse_all
            if _tt_date is not None:
                _tt_sse_all.activate(_tt_date)
                _tt_activated_all = True
        except Exception:
            pass

        try:
            _progress = st.progress(
                0,
                text="Refreshing live sector data..." if _force_live_all else "Preloading shared sector data...",
            )
            _market_bias = _get_market_bias()
            preload_dashboard_sector_data(
                "Overall",
                workers=12,
                force_live_refresh=_force_live_all,
            )

            _rows: list[dict[str, Any]] = []
            _sector_results_map: dict[str, dict[str, Any]] = {}
            _progress.progress(
                10,
                text="Running sector scans on refreshed live data..." if _force_live_all else "Running sector scans on shared data...",
            )
            _max_workers = max(1, min(6, len(_base_sectors)))

            with ThreadPoolExecutor(max_workers=_max_workers) as ex:
                _futures = {
                    ex.submit(_build_sector_result, _sec, _market_bias): _sec
                    for _sec in _base_sectors
                }
                for _done, _future in enumerate(as_completed(_futures), start=1):
                    _sec = _futures[_future]
                    try:
                        _sec_result = _future.result()
                    except Exception as exc:
                        _sec_result = {
                            "kind": "sector",
                            "sector": _sec,
                            "pred": None,
                            "df": pd.DataFrame(),
                            "symbols": get_dashboard_sector_stocks(_sec),
                            "missing_symbols": get_dashboard_sector_stocks(_sec),
                            "requested_count": get_dashboard_sector_count(_sec),
                            "raw_row_count": 0,
                            "valid_row_count": 0,
                            "err": str(exc),
                        }
                    _sector_results_map[_sec] = _sec_result
                    st.session_state["ss_screener_sector_cache"][_sector_cache_key(_sec, _market_bias)] = _sec_result
                    _progress.progress(
                        min(85, 10 + int((_done / max(1, len(_base_sectors))) * 75)),
                        text=f"Scanning {_sec} ({_done}/{len(_base_sectors)})...",
                    )

            _sector_results = [_sector_results_map[_sec] for _sec in _base_sectors if _sec in _sector_results_map]
            _rows = [_result_to_summary_row(_res) for _res in _sector_results]

            _progress.progress(90, text="Computing weighted Overall market probability...")
            _overall_result = _scan_overall_result(
                force_refresh=_force_live_all,
                market_bias=_market_bias,
                precomputed_sector_results=_sector_results,
                preloaded=True,
            )
            _overall_result = _record_scan_meta(
                _overall_result,
                scope_label="All Sectors + Overall",
                live_requested=_force_live_all,
            )
            _overall_row = _result_to_summary_row(_overall_result)

            _progress.progress(100, text="? All sectors + Overall scan complete!")
            st.session_state["ss_screener_all_results"] = [_overall_row] + sorted(
                _rows,
                key=lambda x: x.get("Probability %", 50.0),
                reverse=True,
            )
            st.session_state["ss_screener_all_overall"] = _overall_result
            st.session_state["ss_screener_scan_all_done"] = True
            st.session_state["ss_screener_pending_scan_all_live"] = False
            _remember_scan_df(_overall_result)
        finally:
            if _tt_activated_all:
                try:
                    _tt_sse_all.restore()
                except Exception:
                    pass

    if st.session_state.get("ss_screener_scan_all_done") and st.session_state.get("ss_screener_all_results"):
        _all_rows = st.session_state["ss_screener_all_results"]
        st.markdown(
            '<div style="font-size:13px;font-weight:800;color:#8ab4d8;margin:12px 0 8px;">📊 Sector Predictions</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(_all_rows), width="stretch", hide_index=True)
        _rankable_rows = [row for row in _all_rows if str(row.get("Sector", "")).strip().lower() != "overall"]
        _top_sector_summary = _rankable_rows[:3]
        if _top_sector_summary:
            st.markdown(
                '<div style="font-size:13px;font-weight:800;color:#8ab4d8;margin:12px 0 8px;">Top Sector Reads</div>',
                unsafe_allow_html=True,
            )
            _summary_cols = st.columns(len(_top_sector_summary))
            for _col, _row in zip(_summary_cols, _top_sector_summary):
                with _col:
                    st.markdown(_top_sector_card(_row), unsafe_allow_html=True)

        _overall_result = st.session_state.get("ss_screener_all_overall")
        if _overall_result and _overall_result.get("pred"):
            st.markdown(_meta_pill_row(_overall_result), unsafe_allow_html=True)
            _overall_pred = _overall_result["pred"]
            _tom = str(_overall_pred.get("tomorrow_prediction", "SIDEWAYS")).upper()
            _prob = float(_overall_pred.get("bullish_probability", 50.0))
            _conf = float(_overall_pred.get("confidence", 50.0))
            _main_color = _pred_color(_tom)

            st.markdown(
                f'<div style="background:#0b1017;border:2px solid {_main_color};'
                'border-radius:16px;padding:18px 20px;margin-top:16px;">'
                '<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;">'
                '<div>'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
                'Overall Market</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:28px;font-weight:900;color:{_main_color};line-height:1.0;">'
                f'{_pred_icon(_tom)} {_tom}</div>'
                '</div>'
                '<div style="margin-left:auto;text-align:right;">'
                '<div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Probability</div>'
                f'<div style="font-size:26px;font-weight:900;color:{_prob_color(_prob)};">{_prob:.0f}%</div>'
                '</div>'
                '</div>'
                '<div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;align-items:center;">'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Confidence</div>'
                f'<div style="font-size:18px;font-weight:800;color:#0094ff;">{_conf:.0f}%</div></div>'
                f'<div>{_overall_flag_badge(_overall_pred)}</div>'
                f'<div>{_pill("Strongest: " + str(_overall_pred.get("strongest_sector", "—")), "#00d4a8", "15")}</div>'
                f'<div>{_pill("Weakest: " + str(_overall_pred.get("weakest_sector", "—")), "#ff4d6d", "15")}</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
