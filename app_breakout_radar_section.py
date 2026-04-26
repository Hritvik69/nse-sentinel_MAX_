"""
app_breakout_radar_section.py
─────────────────────────────────────────────────────────────────────────
⚡ Next-Day Breakout Radar — Streamlit UI Section

Drop-in replacement for the old "📂 CSV Next-Day Potential" section
in app.py.  Renders TWO tabs side-by-side:

  Tab 1  ⚡ Breakout Radar      → breakout_radar_engine (NEW)
  Tab 2  📂 Next-Day Potential  → csv_next_day_engine   (existing — UNMODIFIED)

─── INTEGRATION GUIDE ─────────────────────────────────────────────────

Step 1  — Add this import near the top of app.py (with the other engine imports):

    try:
        from breakout_radar_engine import run_breakout_radar, radar_summary
        _BREAKOUT_RADAR_OK = True
    except Exception:
        _BREAKOUT_RADAR_OK = False
        def run_breakout_radar(df=None, cutoff_date=None): return pd.DataFrame()
        def radar_summary(df): return {}

Step 2  — In app.py, replace the sidebar button:

    OLD:  csv_scan_clicked = st.button("📂 CSV Next-Day Potential", key="csv_next_day_btn")
    NEW:  csv_scan_clicked = st.button("⚡ Next-Day Breakout Radar", key="csv_next_day_btn")

Step 3  — In app.py, find and replace the entire
    "# ── CSV NEXT-DAY POTENTIAL …" block with a single call:

    from app_breakout_radar_section import render_breakout_radar_section
    render_breakout_radar_section(
        csv_scan_clicked=csv_scan_clicked,
        _CSV_NEXT_DAY_ENGINE_OK=_CSV_NEXT_DAY_ENGINE_OK,
        _DATA_DOWNLOADER_OK=_DATA_DOWNLOADER_OK,
        _BREAKOUT_RADAR_OK=_BREAKOUT_RADAR_OK,
    )

─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

_VISIBLE_RESULT_LIMIT = 10

# ── Engine imports (safe stubs so file can be imported standalone) ────

try:
    from breakout_radar_engine import run_breakout_radar, radar_summary
    _RADAR_OK = True
except Exception:
    _RADAR_OK = False

    def run_breakout_radar(df=None, cutoff_date=None, progress_callback=None):  # type: ignore[misc]
        return pd.DataFrame()

    def radar_summary(df):  # type: ignore[misc]
        return {}


try:
    from csv_next_day_engine import get_csv_next_day_cache_status, run_csv_next_day  # type: ignore[import]
    _CSV_OK = True
except Exception:
    _CSV_OK = False

    def run_csv_next_day(df=None, cutoff_date=None, progress_callback=None):  # type: ignore[misc]
        return pd.DataFrame()

    def get_csv_next_day_cache_status(df=None):  # type: ignore[misc]
        return {}


try:
    from strategy_engines._engine_utils import get_tomorrow_top_picks
except Exception:
    def get_tomorrow_top_picks(df, source="main", top_n=3):  # type: ignore[misc]
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# SHARED COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────

def _signal_colour(signal: str) -> str:
    colours = {
        "HIGH PROBABILITY BREAKOUT": "#00e5aa",
        "STRONG SETUP":              "#4fc3f7",
        "WATCHLIST":                 "#ffd54f",
        "AVOID":                     "#ef9a9a",
        "TRAP":                      "#ff5252",
    }
    return colours.get(str(signal).strip(), "#8ab4d8")


def _score_bar_html(score: float, max_score: float = 100.0) -> str:
    """Inline coloured progress bar for a score value."""
    pct = int(np.clip(score / max_score * 100, 0, 100))
    colour = "#00e5aa" if pct >= 70 else "#4fc3f7" if pct >= 50 else "#ffd54f" if pct >= 35 else "#ef9a9a"
    return (
        f'<div style="background:#0b1017;border-radius:4px;height:8px;width:100%;">'
        f'<div style="background:{colour};width:{pct}%;height:8px;border-radius:4px;"></div>'
        f'</div><div style="font-size:10px;color:#4a6480;">{score:.0f}</div>'
    )


def _stat_card(label: str, value: str, colour: str = "#00d4a8") -> str:
    return (
        f'<div style="background:#0b1017;border:1px solid #1e3a5f;border-radius:10px;'
        f'padding:10px 14px;text-align:center;">'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:22px;font-weight:800;color:{colour};">{value}</div>'
        f'</div>'
    )


def _start_scan_feedback(initial_text: str):
    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        status_box = st.empty()
    with col_b:
        meta_box = st.empty()
    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;{initial_text}</div>',
        unsafe_allow_html=True,
    )
    meta_box.markdown(
        '<div class="status-line" style="justify-content:center">'
        'Elapsed <b style="color:#8ab4d8">0s</b> &nbsp;·&nbsp; ETA '
        '<b style="color:#f0b429">--</b></div>',
        unsafe_allow_html=True,
    )
    return progress_bar, status_box, meta_box, time.time()


def _update_scan_feedback(
    progress_bar,
    status_box,
    meta_box,
    started_at: float,
    done: int,
    total: int,
    found: int,
    found_label: str,
) -> None:
    pct = (done / total) if total > 0 else 0.0
    elapsed = max(time.time() - started_at, 0.001)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else 0.0
    progress_bar.progress(min(pct, 1.0))
    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;Scanned <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
        f' &nbsp;·&nbsp; Found <b style="color:#00d4a8">{found:,}</b> {found_label}'
        f' &nbsp;·&nbsp; Speed <b style="color:#8ab4d8">{rate:.1f}/s</b></div>',
        unsafe_allow_html=True,
    )
    meta_box.markdown(
        f'<div class="status-line" style="justify-content:center">'
        f'Elapsed <b style="color:#8ab4d8">{elapsed:.0f}s</b>'
        f' &nbsp;·&nbsp; ETA <b style="color:#f0b429">{remaining:.0f}s</b></div>',
        unsafe_allow_html=True,
    )


def _finish_scan_feedback(
    progress_bar,
    status_box,
    meta_box,
    started_at: float,
    total: int,
    found: int,
    found_label: str,
) -> None:
    elapsed = max(time.time() - started_at, 0.001)
    avg_rate = total / elapsed if elapsed > 0 else 0.0
    progress_bar.progress(1.0)
    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} scanned in '
        f'<b style="color:#f0b429">{elapsed:.1f}s</b>'
        f' &nbsp;·&nbsp; <b style="color:#00d4a8">{found:,}</b> {found_label}'
        f' &nbsp;·&nbsp; Avg speed <b style="color:#8ab4d8">{avg_rate:.1f}/s</b></div>',
        unsafe_allow_html=True,
    )
    meta_box.empty()


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — ⚡ BREAKOUT RADAR  (new engine)
# ─────────────────────────────────────────────────────────────────────

def _render_breakout_radar_tab(
    scan_clicked: bool,
    breakout_radar_ok: bool,
    tt_date,
) -> None:
    """Render the ⚡ Breakout Radar sub-tab."""

    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:14px;">'
        'Pre-breakout detection · ATR compression + volume build + trend alignment · '
        'Scores 0–100 · Returns 10–80 pre-move candidates</div>',
        unsafe_allow_html=True,
    )

    if not breakout_radar_ok:
        st.warning(
            "⚠️ breakout_radar_engine.py not found. "
            "Place it in the same folder as app.py and restart."
        )
        return

    _scan_now = bool(scan_clicked)
    if _scan_now:
        _pb, _status_box, _meta_box, _started_at = _start_scan_feedback(
            "Preparing breakout-radar scan..."
        )
        _latest_done = 0
        _latest_total = 0
        _latest_found = 0

        def _progress(done: int, total: int, found: int) -> None:
            nonlocal _latest_done, _latest_total, _latest_found
            _latest_done = done
            _latest_total = total
            _latest_found = found
            _update_scan_feedback(
                _pb, _status_box, _meta_box, _started_at, done, total, found, "setups"
            )

        try:
            radar_df = run_breakout_radar(
                df=None,
                cutoff_date=tt_date,
                progress_callback=_progress,
            )
            _latest_found = len(radar_df) if isinstance(radar_df, pd.DataFrame) else 0
            st.session_state["radar_results_df"] = (
                radar_df.copy() if isinstance(radar_df, pd.DataFrame) else pd.DataFrame()
            )
            st.session_state["radar_last_error"] = ""
            _ts = (
                tt_date.strftime("%d %b %Y (TT)")
                if tt_date else datetime.now().strftime("%d %b %Y, %H:%M")
            )
            st.session_state["radar_last_scan_at"] = _ts
        except Exception as _e:
            st.session_state["radar_last_error"] = str(_e)
        finally:
            _finish_scan_feedback(
                _pb,
                _status_box,
                _meta_box,
                _started_at,
                _latest_total if _latest_total > 0 else _latest_done,
                _latest_found,
                "setups",
            )
        scan_clicked = False

    if scan_clicked:
        with st.spinner("⚡ Scanning for pre-breakout setups …"):
            try:
                radar_df = run_breakout_radar(df=None, cutoff_date=tt_date)
                st.session_state["radar_results_df"]   = (
                    radar_df.copy() if isinstance(radar_df, pd.DataFrame) else pd.DataFrame()
                )
                st.session_state["radar_last_error"]   = ""
                _ts = (
                    tt_date.strftime("%d %b %Y (TT)")
                    if tt_date else datetime.now().strftime("%d %b %Y, %H:%M")
                )
                st.session_state["radar_last_scan_at"] = _ts
            except Exception as _e:
                st.session_state["radar_last_error"]   = str(_e)

    radar_df     = st.session_state.get("radar_results_df",   pd.DataFrame())
    radar_error  = str(st.session_state.get("radar_last_error",   "") or "").strip()
    radar_scanned_at = str(st.session_state.get("radar_last_scan_at", "") or "").strip()

    if radar_scanned_at:
        st.caption(f"Last scan: {radar_scanned_at}")
    if radar_error:
        st.error(f"Radar scan failed: {radar_error}")

    if not isinstance(radar_df, pd.DataFrame) or radar_df.empty:
        if not radar_error and not radar_scanned_at:
            st.info("Click **⚡ Next-Day Breakout Radar** in the sidebar to run a scan.")
        elif not radar_error:
            st.info(
                "No pre-breakout setups met the scoring threshold. "
                "Market may be in a low-compression phase — check again tomorrow."
            )
        return

    # ── Summary metrics ────────────────────────────────────────────────
    smry = radar_summary(radar_df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            _stat_card("Total Setups", str(smry.get("total", 0)), "#00e5aa"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _stat_card("🔥 High Prob", str(smry.get("high_prob", 0)), "#00e5aa"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _stat_card("💪 Strong", str(smry.get("strong", 0)), "#4fc3f7"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _stat_card("👁 Watchlist", str(smry.get("watchlist", 0)), "#ffd54f"),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            _stat_card("Avg Score", f"{smry.get('avg_score', 0):.1f}", "#8ab4d8"),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _stat_card("Top Pick", str(smry.get("top_symbol", "-")), "#00e5aa"),
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Legend ────────────────────────────────────────────────────────
    legend_html = (
        '<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:11px;'
        'color:#4a6480;margin-bottom:14px;padding:8px 14px;background:#0b1017;'
        'border-radius:8px;border:1px solid #1e3a5f;">'
    )
    for sig, clr in [
        ("HIGH PROBABILITY BREAKOUT", "#00e5aa"),
        ("STRONG SETUP",              "#4fc3f7"),
        ("WATCHLIST",                 "#ffd54f"),
        ("AVOID",                     "#ef9a9a"),
        ("TRAP",                      "#ff5252"),
    ]:
        n = int((radar_df.get("Signal", pd.Series()) == sig).sum())
        legend_html += (
            f'<span>●&nbsp;<b style="color:{clr};">{sig}</b> ({n})</span>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Signal colour column for the table ────────────────────────────
    def _sig_badge(sig: str) -> str:
        c = _signal_colour(sig)
        return f'<span style="color:{c};font-weight:700;">{sig}</span>'

    # ── Score distribution mini-bar ───────────────────────────────────
    with st.expander("📊 Score Distribution", expanded=False):
        bins = [48, 55, 65, 75, 85, 101]
        labels = ["48–54", "55–64", "65–74", "75–84", "85–100"]
        if "Final Score" in radar_df.columns:
            for i, lbl in enumerate(labels):
                n = int(((radar_df["Final Score"] >= bins[i]) & (radar_df["Final Score"] < bins[i+1])).sum())
                bar_w = max(2, int(n / max(len(radar_df), 1) * 200))
                colours = ["#ffd54f", "#4fc3f7", "#4fc3f7", "#00e5aa", "#00e5aa"]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
                    f'<span style="width:60px;font-size:11px;color:#4a6480;">{lbl}</span>'
                    f'<div style="width:{bar_w}px;height:10px;background:{colours[i]};border-radius:3px;"></div>'
                    f'<span style="font-size:11px;color:#8ab4d8;">{n}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Download button ───────────────────────────────────────────────
    _dl_col, _ = st.columns([1, 3])
    with _dl_col:
        _stamp = radar_scanned_at.replace(" ", "_").replace(",", "").replace(":", "-") if radar_scanned_at else "scan"
        st.download_button(
            "⬇️ Download CSV",
            data=radar_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"breakout_radar_{_stamp}.csv",
            mime="text/csv",
            key="radar_download_btn",
        )

    # ── Main results table ────────────────────────────────────────────
    display_cols = [
        "Symbol", "Price (₹)", "Volume Ratio", "RSI",
        "Δ EMA20 (%)", "Δ 20D High (%)", "5D Return (%)",
        "Compression Score", "Trend Score", "Volume Score",
        "RSI Score", "EMA Dist Score", "Risk Score",
        "Trap Flags", "Final Score", "Signal", "Chart Link",
    ]
    show_cols = [c for c in display_cols if c in radar_df.columns]

    st.caption(
        f"Showing top {_VISIBLE_RESULT_LIMIT} of {len(radar_df)} radar results. Download keeps all rows."
    )
    st.dataframe(
        radar_df.head(_VISIBLE_RESULT_LIMIT)[show_cols],
        column_config={
            "Symbol":            st.column_config.TextColumn("Ticker"),
            "Price (₹)":         st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
            "Volume Ratio":      st.column_config.NumberColumn("Vol/Avg", format="%.2fx"),
            "RSI":               st.column_config.NumberColumn("RSI", format="%.1f"),
            "Δ EMA20 (%)":       st.column_config.NumberColumn("Δ EMA20%", format="%.2f%%"),
            "Δ 20D High (%)":    st.column_config.NumberColumn("Δ 20D High%", format="%.2f%%"),
            "5D Return (%)":     st.column_config.NumberColumn("5D Ret%", format="%.2f%%"),
            "Compression Score": st.column_config.ProgressColumn(
                "Compression", min_value=0, max_value=25, format="%.1f"
            ),
            "Trend Score":       st.column_config.ProgressColumn(
                "Trend", min_value=0, max_value=25, format="%.1f"
            ),
            "Volume Score":      st.column_config.ProgressColumn(
                "Volume", min_value=0, max_value=20, format="%.1f"
            ),
            "RSI Score":         st.column_config.ProgressColumn(
                "RSI Scr", min_value=0, max_value=20, format="%.1f"
            ),
            "EMA Dist Score":    st.column_config.ProgressColumn(
                "EMA Dist", min_value=0, max_value=10, format="%.1f"
            ),
            "Risk Score":        st.column_config.NumberColumn("Risk", format="%.0f"),
            "Trap Flags":        st.column_config.TextColumn("Traps", width="medium"),
            "Final Score":       st.column_config.ProgressColumn(
                "⚡ Score", min_value=0, max_value=100, format="%.1f"
            ),
            "Signal":            st.column_config.TextColumn("Signal", width="large"),
            "Chart Link":        st.column_config.LinkColumn("Chart", display_text="📈 Open"),
        },
        width="stretch",
        hide_index=True,
    )

    # ── Scoring legend ────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:11px;color:#4a6480;margin-top:12px;">'
        '📐 <b>Score guide</b> — '
        'Compression (0–25): ATR tightening + proximity to 20D high · '
        'Trend (0–25): EMA20 > EMA50 alignment + slope · '
        'Volume (0–20): ratio + rising sessions · '
        'RSI (0–20): 52–65 zone ideal · '
        'EMA Dist (0–10): 0–4% ideal, >6% penalised · '
        'Trap penalty: −5 (1 flag) / −15 (2 flags) / TRAP (3+)'
        '</div>',
        unsafe_allow_html=True,
    )

    _radar_top3 = get_tomorrow_top_picks(radar_df, source="breakout", top_n=3)
    if isinstance(_radar_top3, pd.DataFrame) and not _radar_top3.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Top 3 Buyable For Tomorrow")
        st.caption("Best next-day breakout candidates from this radar scan.")

        _radar_cols = [
            "Symbol", "Tomorrow Pick Score", "Final Score", "Signal",
            "Risk Score", "Trap Flags", "Tomorrow Pick Reason", "Chart Link",
        ]
        _radar_cols = [c for c in _radar_cols if c in _radar_top3.columns]

        st.dataframe(
            _radar_top3[_radar_cols],
            column_config={
                "Symbol": st.column_config.TextColumn("Ticker"),
                "Tomorrow Pick Score": st.column_config.NumberColumn("Tomorrow Score", format="%.1f"),
                "Final Score": st.column_config.NumberColumn("Final Score", format="%.1f"),
                "Signal": st.column_config.TextColumn("Signal", width="medium"),
                "Risk Score": st.column_config.NumberColumn("Risk", format="%.0f"),
                "Trap Flags": st.column_config.TextColumn("Traps"),
                "Tomorrow Pick Reason": st.column_config.TextColumn("Why Buy Tomorrow", width="large"),
                "Chart Link": st.column_config.LinkColumn("Chart", display_text="Open"),
            },
            width="stretch",
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — 📂 NEXT-DAY POTENTIAL  (original csv_next_day_engine — intact)
# ─────────────────────────────────────────────────────────────────────

def _render_csv_next_day_tab(
    scan_clicked: bool,
    csv_next_day_engine_ok: bool,
    data_downloader_ok: bool,
    tt_date,
) -> None:
    """Render the original CSV Next-Day Potential sub-tab — zero changes to logic."""

    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:14px;">'
        'Fast local scan · tomorrow-up probability · stricter buy-readiness filter</div>',
        unsafe_allow_html=True,
    )

    if not data_downloader_ok or not csv_next_day_engine_ok:
        st.warning("CSV next-day engine is not available. Check `data_downloader.py` and `csv_next_day_engine.py`.")
        return

    _scan_now = bool(scan_clicked)
    if _scan_now:
        _pb, _status_box, _meta_box, _started_at = _start_scan_feedback(
            "Preparing CSV probability scan..."
        )
        _latest_done = 0
        _latest_total = 0
        _latest_found = 0

        def _progress(done: int, total: int, found: int) -> None:
            nonlocal _latest_done, _latest_total, _latest_found
            _latest_done = done
            _latest_total = total
            _latest_found = found
            _update_scan_feedback(
                _pb, _status_box, _meta_box, _started_at, done, total, found, "buy-ready"
            )

        try:
            _fresh = run_csv_next_day(
                None,
                cutoff_date=tt_date,
                progress_callback=_progress,
            )
            _latest_found = len(_fresh) if isinstance(_fresh, pd.DataFrame) else 0
            st.session_state["csv_next_day_results_df"] = (
                _fresh.copy() if isinstance(_fresh, pd.DataFrame) else pd.DataFrame()
            )
            _csv_attrs = getattr(_fresh, "attrs", {}) if isinstance(_fresh, pd.DataFrame) else {}
            st.session_state["csv_next_day_empty_reason"] = str(_csv_attrs.get("empty_reason", "") or "")
            st.session_state["csv_next_day_cache_status"] = dict(_csv_attrs.get("cache_status", {}) or {})
            st.session_state["csv_next_day_last_error"] = ""
            _ts = (
                tt_date.strftime("%d %b %Y (TT)")
                if tt_date else datetime.now().strftime("%d %b %Y, %H:%M")
            )
            st.session_state["csv_next_day_last_scan_at"] = _ts
        except Exception as _e:
            st.session_state["csv_next_day_last_error"] = str(_e)
            st.session_state["csv_next_day_empty_reason"] = ""
            st.session_state["csv_next_day_cache_status"] = get_csv_next_day_cache_status(None)
        finally:
            _finish_scan_feedback(
                _pb,
                _status_box,
                _meta_box,
                _started_at,
                _latest_total if _latest_total > 0 else _latest_done,
                _latest_found,
                "buy-ready",
            )
        scan_clicked = False

    if scan_clicked:
        with st.spinner("📂 Scanning local CSVs …"):
            try:
                _fresh = run_csv_next_day(None, cutoff_date=tt_date)
                st.session_state["csv_next_day_results_df"] = (
                    _fresh.copy() if isinstance(_fresh, pd.DataFrame) else pd.DataFrame()
                )
                _csv_attrs = getattr(_fresh, "attrs", {}) if isinstance(_fresh, pd.DataFrame) else {}
                st.session_state["csv_next_day_empty_reason"] = str(_csv_attrs.get("empty_reason", "") or "")
                st.session_state["csv_next_day_cache_status"] = dict(_csv_attrs.get("cache_status", {}) or {})
                st.session_state["csv_next_day_last_error"] = ""
                _ts = (
                    tt_date.strftime("%d %b %Y (TT)")
                    if tt_date else datetime.now().strftime("%d %b %Y, %H:%M")
                )
                st.session_state["csv_next_day_last_scan_at"] = _ts
            except Exception as _e:
                st.session_state["csv_next_day_last_error"] = str(_e)
                st.session_state["csv_next_day_empty_reason"] = ""
                st.session_state["csv_next_day_cache_status"] = get_csv_next_day_cache_status(None)

    csv_df          = st.session_state.get("csv_next_day_results_df",  pd.DataFrame())
    csv_last_error  = str(st.session_state.get("csv_next_day_last_error",   "") or "").strip()
    csv_last_scan_at = str(st.session_state.get("csv_next_day_last_scan_at", "") or "").strip()
    csv_empty_reason = str(st.session_state.get("csv_next_day_empty_reason", "") or "").strip()
    csv_cache_status = st.session_state.get("csv_next_day_cache_status", {}) or {}

    if csv_last_scan_at:
        st.caption(f"Last CSV scan: {csv_last_scan_at}")
    if csv_last_error:
        st.error(f"CSV scan failed: {csv_last_error}")

    if not isinstance(csv_df, pd.DataFrame) or csv_df.empty:
        if not csv_last_error and not csv_last_scan_at:
            st.info("Click **⚡ Next-Day Breakout Radar** in the sidebar to run a scan.")
            _initial_cache_status = get_csv_next_day_cache_status(None)
            if isinstance(_initial_cache_status, dict) and _initial_cache_status.get("status") == "empty_cache":
                st.caption(
                    "Local CSV cache is empty right now. Use `Refresh Local Data Cache` in the sidebar before running this panel."
                )
        elif not csv_last_error:
            if csv_empty_reason == "NO_LOCAL_CACHE" or csv_cache_status.get("status") == "empty_cache":
                st.warning(
                    "Local CSV cache is empty. Use `Refresh Local Data Cache` in the sidebar to populate the `data` folder, then rerun this panel."
                )
                if csv_cache_status.get("data_dir"):
                    st.caption(f"Cache folder: `{csv_cache_status['data_dir']}`")
            elif csv_empty_reason == "NO_MATCHING_CSVS" or csv_cache_status.get("status") == "no_matching_csvs":
                st.info(
                    "No matching cached CSV files were found for the current symbol list. Refresh the local cache or widen the selection, then rerun."
                )
            else:
                st.info(
                    "No clean buy-ready setups found for tomorrow in the current CSV universe. "
                    "Wait for better structure instead of forcing a trade."
                )
        return

    st.success(f"✅ {len(csv_df)} buy-ready setups matched the tomorrow-up criteria")

    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    with _m1:
        st.metric("Matches", f"{len(csv_df):,}")
    with _m2:
        st.metric("Avg Prob", f"{csv_df['Next Day Prob'].mean():.1f}%")
    with _m3:
        st.metric("Avg Conf", f"{csv_df['Confidence'].mean():.1f}%")
    with _m4:
        _ready = int((csv_df.get("Buy Readiness", pd.Series()) == "BUY READY").sum())
        st.metric("Buy Ready", f"{_ready:,}")
    with _m5:
        _grade_order = ["A", "B", "C", "D"]
        _gv = csv_df.get("Grade", pd.Series(dtype=str)).astype(str).tolist()
        _best = next((g for g in _grade_order if g in _gv), _gv[0] if _gv else "-")
        st.metric("Best Grade", _best)

    _dl_col, _grd_col = st.columns([0.32, 0.68])
    with _dl_col:
        _stamp = csv_last_scan_at.replace(" ", "_").replace(",", "").replace(":", "-") if csv_last_scan_at else "scan"
        st.download_button(
            "⬇️ Download CSV Results",
            data=csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"csv_next_day_results_{_stamp}.csv",
            mime="text/csv",
            key="csv_next_day_download_btn",
        )
    with _grd_col:
        if "Grade" in csv_df.columns:
            _gc = csv_df["Grade"].fillna("-").astype(str).value_counts()
            _gs = " | ".join(f"{g}: {_gc.get(g, 0)}" for g in ["A", "B", "C", "D"])
            st.markdown(
                '<div style="font-size:12px;color:#4a6480;padding-top:8px;">'
                f'<b>A</b> strongest · <b>B</b> good · <b>C</b> watchlist · <b>D</b> weak'
                f'<br>Distribution: {_gs}</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        f"Showing top {_VISIBLE_RESULT_LIMIT} of {len(csv_df)} CSV setups. Download keeps all rows."
    )
    st.dataframe(
        csv_df.head(_VISIBLE_RESULT_LIMIT),
        column_config={
            "Symbol":            st.column_config.TextColumn("Ticker"),
            "Price (₹)":         st.column_config.NumberColumn("Close (₹)",     format="₹%.2f"),
            "Next Day Prob":     st.column_config.NumberColumn("Tomorrow Up %",  format="%.1f%%"),
            "Confidence":        st.column_config.NumberColumn("Confidence %",   format="%.1f%%"),
            "Grade":             st.column_config.TextColumn("Grade"),
            "Buy Readiness":     st.column_config.TextColumn("Buy Verdict"),
            "Signal":            st.column_config.TextColumn("Signal"),
            "Setup":             st.column_config.TextColumn("Setup"),
            "Historical Win %":  st.column_config.NumberColumn("Hist Win %",     format="%.1f%%"),
            "Downside Risk %":   st.column_config.NumberColumn("Downside Risk %", format="%.1f%%"),
            "Analog Count":      st.column_config.NumberColumn("Analogs",         format="%d"),
            "Analog Avg Ret %":  st.column_config.NumberColumn("Analog Avg %",    format="%.2f%%"),
            "Setup Quality":     st.column_config.NumberColumn("Setup Q",         format="%.1f"),
            "Trigger Quality":   st.column_config.NumberColumn("Trigger Q",       format="%.1f"),
            "RSI":               st.column_config.NumberColumn("RSI",             format="%.1f"),
            "Vol / Avg":         st.column_config.NumberColumn("Vol/Avg",         format="%.2fx"),
            "Volume Strength":   st.column_config.TextColumn("Volume"),
            "Bull Trap":         st.column_config.TextColumn("Trap"),
            "Risk Notes":        st.column_config.TextColumn("Risk Notes",        width="large"),
            "Chart Link":        st.column_config.LinkColumn("Chart",             display_text="📈 Open"),
        },
        width="stretch",
        hide_index=True,
    )

    _csv_top3 = get_tomorrow_top_picks(csv_df, source="csv", top_n=3)
    if isinstance(_csv_top3, pd.DataFrame) and not _csv_top3.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Top 3 Buyable For Tomorrow")
        st.caption("Best next-day buy candidates from the CSV probability engine.")

        _csv_cols = [
            "Symbol", "Tomorrow Pick Score", "Next Day Prob", "Confidence",
            "Grade", "Signal", "Tomorrow Pick Reason", "Chart Link",
        ]
        _csv_cols = [c for c in _csv_cols if c in _csv_top3.columns]

        st.dataframe(
            _csv_top3[_csv_cols],
            column_config={
                "Symbol": st.column_config.TextColumn("Ticker"),
                "Tomorrow Pick Score": st.column_config.NumberColumn("Tomorrow Score", format="%.1f"),
                "Next Day Prob": st.column_config.NumberColumn("Tomorrow Up %", format="%.1f%%"),
                "Confidence": st.column_config.NumberColumn("Confidence %", format="%.1f%%"),
                "Grade": st.column_config.TextColumn("Grade"),
                "Signal": st.column_config.TextColumn("Signal", width="medium"),
                "Tomorrow Pick Reason": st.column_config.TextColumn("Why Buy Tomorrow", width="large"),
                "Chart Link": st.column_config.LinkColumn("Chart", display_text="Open"),
            },
            width="stretch",
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────
# PUBLIC RENDER FUNCTION — called from app.py
# ─────────────────────────────────────────────────────────────────────

def render_breakout_radar_section(
    csv_scan_clicked: bool,
    _CSV_NEXT_DAY_ENGINE_OK: bool = True,
    _DATA_DOWNLOADER_OK: bool = True,
    _BREAKOUT_RADAR_OK: bool = True,
) -> None:
    """
    Render the combined ⚡ Next-Day Breakout Radar section.

    Called from app.py exactly where the old CSV Next-Day block was.
    Accepts the same session-state flags so existing state keys are preserved.

    Parameters
    ----------
    csv_scan_clicked : bool
        True when the sidebar button was just clicked.
    _CSV_NEXT_DAY_ENGINE_OK : bool
        From app.py — whether csv_next_day_engine imported successfully.
    _DATA_DOWNLOADER_OK : bool
        From app.py — whether data_downloader imported successfully.
    _BREAKOUT_RADAR_OK : bool
        From app.py — whether breakout_radar_engine imported successfully.
    """
    _panel_open = bool(st.session_state.get("csv_next_day_show_panel", False))

    if not (csv_scan_clicked or _panel_open):
        return

    # ── Section header ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-bottom:2px;">⚡ Next-Day Breakout Radar</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        'Pre-move detection system · Two complementary engines · '
        'Use both for highest confidence</div>',
        unsafe_allow_html=True,
    )

    # ── TT date (pass-through from session state) ─────────────────────
    tt_date = st.session_state.get("tt_date_val")   # None in live mode

    # ── Dual tabs ─────────────────────────────────────────────────────
    tab_radar, tab_csv = st.tabs([
        "⚡ Breakout Radar  (Pre-Move Detection)",
        "📂 Next-Day Potential  (Probability Engine)",
    ])

    with tab_radar:
        _render_breakout_radar_tab(
            scan_clicked=csv_scan_clicked,
            breakout_radar_ok=_BREAKOUT_RADAR_OK and _RADAR_OK,
            tt_date=tt_date,
        )

    with tab_csv:
        _render_csv_next_day_tab(
            scan_clicked=csv_scan_clicked,
            csv_next_day_engine_ok=_CSV_NEXT_DAY_ENGINE_OK and _CSV_OK,
            data_downloader_ok=_DATA_DOWNLOADER_OK,
            tt_date=tt_date,
        )
