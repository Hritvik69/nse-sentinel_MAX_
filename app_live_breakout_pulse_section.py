"""
app_live_breakout_pulse_section.py
────────────────────────────────────────────────────────────────────
*️⃣  LIVE BREAKOUT PULSE — Streamlit UI section for app.py

HOW TO INTEGRATE
════════════════

STEP 1 ── TOP OF app.py  (after existing engine imports)
─────────────────────────────────────────────────────────
Add these lines near the other "try/except import" blocks:

    try:
        from live_breakout_pulse_engine import (
            run_live_breakout_pulse,
            pulse_summary,
        )
        _LIVE_PULSE_OK = True
    except Exception:
        _LIVE_PULSE_OK = False
        def run_live_breakout_pulse(cutoff_date=None, progress_callback=None):
            return pd.DataFrame()
        def pulse_summary(df):
            return {}


STEP 2 ── LEFT PANEL (sidebar)  — inside `with st.sidebar:`
─────────────────────────────────────────────────────────────
Add ONE line immediately before or after the csv_scan_clicked line:

    live_pulse_clicked = st.button("*️⃣ Live Breakout Pulse", key="live_pulse_btn")

Then in the session-state block below it:

    if live_pulse_clicked:
        st.session_state["live_pulse_show_panel"] = True


STEP 3 ── MAIN PANEL  (after the CSV Next-Day panel block)
───────────────────────────────────────────────────────────
Paste the entire render block below (or call render_live_breakout_pulse()
if you import this file as a module).


ARCHITECTURE NOTES
══════════════════
• Completely standalone — zero coupling to Modes 1-6 or existing scan.
• Respects Time-Travel: passes cutoff_date automatically when TT active.
• Stores results in st.session_state["live_pulse_results_df"] so the
  panel stays populated on re-render without re-running the scan.
• Progress bar updates in real-time as tickers complete.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

_VISIBLE_RESULT_LIMIT = 10

# ── Engine import (graceful stub if file is missing) ─────────────────
try:
    from live_breakout_pulse_engine import run_live_breakout_pulse, pulse_summary
    _LIVE_PULSE_ENGINE_OK = True
except Exception:
    _LIVE_PULSE_ENGINE_OK = False

    def run_live_breakout_pulse(cutoff_date=None, progress_callback=None):  # type: ignore[misc]
        return pd.DataFrame()

    def pulse_summary(df) -> dict:  # type: ignore[misc]
        return {}


# ── Signal badge helper ───────────────────────────────────────────────
_SIGNAL_COLORS = {
    "LIVE BREAKOUT":    "#00d4a8",
    "STRONG MOMENTUM":  "#f0b429",
    "WATCH":            "#8ab4d8",
}

def _signal_badge(sig: str) -> str:
    color = _SIGNAL_COLORS.get(sig, "#8ab4d8")
    return (
        f'<span style="background:{color}20;color:{color};'
        f'border:1px solid {color}50;border-radius:6px;'
        f'padding:2px 8px;font-size:11px;font-weight:700;">'
        f'{sig}</span>'
    )


_PRICE_DISPLAY_COL = "Price (\u20b9)"
_PRICE_ALIAS_COLS = (
    _PRICE_DISPLAY_COL,
    "Price (\u00e2\u201a\u00b9)",
    "Price (\u00c3\u00a2\u00e2\u20ac\u0161\u00c2\u00b9)",
)


def _with_live_pulse_display_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if _PRICE_DISPLAY_COL not in out.columns:
        for alias in _PRICE_ALIAS_COLS:
            if alias in out.columns:
                out[_PRICE_DISPLAY_COL] = out[alias]
                break
    return out


def _start_scan_feedback(label: str):
    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        status_box = st.empty()
    with col_b:
        meta_box = st.empty()

    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;{label}</div>',
        unsafe_allow_html=True,
    )
    meta_box.markdown(
        '<div class="status-line" style="justify-content:center">'
        'Elapsed <b style="color:#8ab4d8">0s</b>'
        ' &nbsp;·&nbsp; ETA <b style="color:#f0b429">--</b></div>',
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
    found: int | None = None,
) -> None:
    pct = (done / total) if total > 0 else 0.0
    elapsed = max(time.time() - started_at, 0.001)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else 0.0
    progress_bar.progress(min(pct, 1.0))

    found_html = ""
    if found is not None:
        found_html = (
            f' &nbsp;·&nbsp; Hits <b style="color:#00d4a8">{found}</b>'
        )

    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;Scanned <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
        f'{found_html}'
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
    found: int | None = None,
) -> None:
    elapsed = max(time.time() - started_at, 0.001)
    avg_rate = total / elapsed if elapsed > 0 else 0.0
    progress_bar.progress(1.0)

    found_html = ""
    if found is not None:
        found_html = (
            f' &nbsp;·&nbsp; <b style="color:#00d4a8">{found}</b> hits'
        )

    status_box.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} stocks in '
        f'<b style="color:#f0b429">{elapsed:.1f}s</b>'
        f'{found_html}'
        f' &nbsp;·&nbsp; Avg speed <b style="color:#8ab4d8">{avg_rate:.1f}/s</b></div>',
        unsafe_allow_html=True,
    )
    meta_box.empty()


def render_live_breakout_pulse(
    live_pulse_clicked: bool,
    tt_date_val=None,
    render_add_in_picks_actions=None,
    render_imported_ai_actions=None,
) -> None:
    """
    Render the Live Breakout Pulse panel.

    Parameters
    ----------
    live_pulse_clicked : bool
        True when the sidebar button was just pressed.
    tt_date_val : date | None
        The active Time-Travel date (or None in live mode).
    """
    _panel_open = bool(st.session_state.get("live_pulse_show_panel", False))
    if not (live_pulse_clicked or _panel_open):
        return
    _auto_run_scan = bool(st.session_state.pop("live_pulse_autorun", False))
    _run_scan_now = live_pulse_clicked or _auto_run_scan
    _tt_cache_key = str(tt_date_val or "live")
    if st.session_state.get("live_pulse_tt_cache_key") != _tt_cache_key:
        for _key in (
            "live_pulse_results_df",
            "live_pulse_last_scan_at",
            "live_pulse_last_error",
        ):
            st.session_state.pop(_key, None)
        st.session_state["live_pulse_tt_cache_key"] = _tt_cache_key

    try:
        from trade_decision_simple import apply_trade_decision_simple_any
    except Exception:
        def apply_trade_decision_simple_any(df):
            return df


    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="margin-bottom:4px;">*️⃣ Live Breakout Pulse</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
        'Session-aware breakout detection using snapshot/cache outside live hours · '
        'live refresh only when the market data policy allows it · Vol ≥ 1.5× required · RSI 78+ rejected</div>',
        unsafe_allow_html=True,
    )

    # ── Engine availability check ─────────────────────────────────────
    _close_live_cols = st.columns([5.5, 1.5])
    with _close_live_cols[1]:
        _close_live_pulse = st.button(
            "Back",
            key="live_pulse_close_panel_btn",
        )

    if _close_live_pulse:
        st.session_state["live_pulse_show_panel"] = False
        st.rerun()

    if not _LIVE_PULSE_ENGINE_OK:
        st.warning(
            "⚠️ `live_breakout_pulse_engine.py` not found. "
            "Place it in the same folder as `app.py` and restart."
        )
        return

    # ── Time-Travel banner ────────────────────────────────────────────
    if tt_date_val is not None:
        st.info(
            f"🕰️ **Time-Travel active** — data will be truncated to "
            f"**{tt_date_val.strftime('%d %b %Y')}** before indicator computation.",
            icon=None,
        )

    # ── Trigger scan ─────────────────────────────────────────────────
    if _run_scan_now and st.session_state.get("_live_pulse_scan_running", False):
        st.warning("Live Breakout Pulse is already running. Please wait for it to finish.")
        _run_scan_now = False

    if _run_scan_now:
        st.session_state["_live_pulse_scan_running"] = True
        _pb, _status_box, _meta_box, _started_at = _start_scan_feedback(
            "Preparing live breakout scan..."
        )
        _latest_done = 0
        _latest_total = 0
        _latest_found = 0

        def _update_progress(done: int, total: int, found: int | None = None) -> None:
            nonlocal _latest_done, _latest_total, _latest_found
            _latest_done = done
            _latest_total = total
            if found is not None:
                _latest_found = found
            _update_scan_feedback(
                _pb,
                _status_box,
                _meta_box,
                _started_at,
                done,
                total,
                _latest_found,
            )

        try:
            _fresh_df = run_live_breakout_pulse(
                cutoff_date=tt_date_val,
                progress_callback=_update_progress,
            )
            _latest_found = len(_fresh_df) if isinstance(_fresh_df, pd.DataFrame) else 0
            st.session_state["live_pulse_results_df"] = (
                _fresh_df.copy()
                if isinstance(_fresh_df, pd.DataFrame)
                else pd.DataFrame()
            )
            ts_label = (
                tt_date_val.strftime("%d %b %Y (TT)")
                if tt_date_val
                else datetime.now().strftime("%d %b %Y, %H:%M")
            )
            st.session_state["live_pulse_last_scan_at"] = ts_label
            st.session_state["live_pulse_last_error"]   = ""
        except Exception as _err:
            st.session_state["live_pulse_last_error"] = str(_err)
        finally:
            _finish_scan_feedback(
                _pb,
                _status_box,
                _meta_box,
                _started_at,
                _latest_total if _latest_total > 0 else _latest_done,
                _latest_found,
            )
            st.session_state.pop("_live_pulse_scan_running", None)

    # ── Retrieve stored results ───────────────────────────────────────
    pulse_df        = st.session_state.get("live_pulse_results_df", pd.DataFrame())
    last_error      = str(st.session_state.get("live_pulse_last_error", "") or "").strip()
    last_scan_at    = str(st.session_state.get("live_pulse_last_scan_at", "") or "").strip()

    if last_scan_at:
        st.caption(f"Last scan: {last_scan_at}")

    if isinstance(pulse_df, pd.DataFrame):
        data_stats = pulse_df.attrs.get("data_stats", {})
        if isinstance(data_stats, dict) and data_stats:
            plan = data_stats.get("plan", {}) if isinstance(data_stats.get("plan"), dict) else {}
            source_label = str(plan.get("source_label") or pulse_df.attrs.get("data_source") or "").strip()
            snapshot_loaded = int(data_stats.get("snapshot_loaded", 0) or 0)
            snapshot_saved = int(data_stats.get("snapshot_saved", 0) or 0)
            if source_label:
                bits = [source_label]
                if snapshot_loaded > 0:
                    bits.append(f"{snapshot_loaded:,} snapshot tickers loaded")
                if snapshot_saved > 0:
                    bits.append(f"{snapshot_saved:,} snapshot tickers saved")
                st.caption("Data source: " + " | ".join(bits))

    if last_error:
        st.error(f"Scan failed: {last_error}")
        return

    if not isinstance(pulse_df, pd.DataFrame) or pulse_df.empty:
        if _run_scan_now:
            st.info(
                "No stocks passed the Live Breakout Pulse filters right now. "
                "Market may be in a low-momentum phase — check back later."
            )
        return

    # ── Summary metrics ───────────────────────────────────────────────
    summary = pulse_summary(pulse_df)
    st.success(
        f"✅ **{summary['total']} stocks** passed all live breakout filters"
    )
    if summary.get("universe_scanned", 0):
        st.caption(f"Universe scanned: {summary['universe_scanned']:,} tickers")

    _mc1, _mc2, _mc3, _mc4, _mc5, _mc6 = st.columns(6)
    with _mc1:
        st.metric("Universe", f"{summary.get('universe_scanned', 0):,}")
    with _mc2:
        st.metric("Total Hits", f"{summary['total']:,}")
    with _mc3:
        st.metric(
            "🟢 Live Breakout",
            f"{summary['live_breakouts']:,}",
            help="Final Score ≥ 80",
        )
    with _mc4:
        st.metric(
            "🟡 Strong Momentum",
            f"{summary['strong_momentum']:,}",
            help="Final Score 65–79",
        )
    with _mc5:
        st.metric(
            "🔵 Watch",
            f"{summary['watch']:,}",
            help="Final Score 50–64",
        )
    with _mc6:
        st.metric("Avg Score", f"{summary['avg_score']}")

    st.markdown("<br>", unsafe_allow_html=True)

    top_pulse_df = pulse_df.copy()
    if "Final Score" in top_pulse_df.columns:
        top_pulse_df = top_pulse_df.sort_values("Final Score", ascending=False).reset_index(drop=True)
    top_pulse_df = apply_trade_decision_simple_any(top_pulse_df.copy())
    top_pulse_df = _with_live_pulse_display_aliases(top_pulse_df)
    top_pulse_df = top_pulse_df.head(3).copy()
    top_pulse_symbols = []
    if "Symbol" in top_pulse_df.columns:
        top_pulse_symbols = [
            str(symbol or "").strip().upper().replace(".NS", "")
            for symbol in top_pulse_df["Symbol"].tolist()
            if str(symbol or "").strip()
        ]
    top_pulse_cols = [
        "Symbol",
        "Final Score",
        "Signal",
        "Price (â‚¹)",
        "RSI",
        "Vol / Avg",
        "Action",
        "Hold Days",
        "Chart Link",
    ]
    if _PRICE_DISPLAY_COL in top_pulse_df.columns and _PRICE_DISPLAY_COL not in top_pulse_cols:
        top_pulse_cols.insert(3, _PRICE_DISPLAY_COL)
    top_pulse_cols = [col for col in top_pulse_cols if col in top_pulse_df.columns]
    if not top_pulse_df.empty and top_pulse_cols:
        st.markdown('<h3 style="margin-bottom:6px;">Top 3 Breakout Stocks</h3>', unsafe_allow_html=True)
        st.caption("Highest-scoring setups from the current Live Breakout Pulse scan.")
        st.dataframe(
            top_pulse_df[top_pulse_cols],
            column_config={
                "Symbol": st.column_config.TextColumn("Ticker"),
                "Final Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Signal": st.column_config.TextColumn("Signal"),
                _PRICE_DISPLAY_COL: st.column_config.NumberColumn(_PRICE_DISPLAY_COL, format="\u20b9%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "Vol / Avg": st.column_config.NumberColumn("Vol/Avg", format="%.2fx"),
                "Chart Link": st.column_config.LinkColumn("Chart", display_text="Open Chart"),
                "Action": st.column_config.TextColumn("Action"),
                "Hold Days": st.column_config.TextColumn("Hold Days"),
            },
            width="stretch",
            hide_index=True,
        )
        if callable(render_add_in_picks_actions):
            top_pulse_key = (last_scan_at or "latest").replace(" ", "_").replace(",", "").replace(":", "-")
            render_add_in_picks_actions(
                top_pulse_symbols,
                key_prefix=f"live_pulse_top3_{top_pulse_key}",
                scope_label="Live Breakout Pulse",
                bucket="breakout",
                helper_text="Add these top breakout stocks into Tomorrow's Picks and keep them saved until you remove them.",
            )
        if callable(render_imported_ai_actions):
            top_pulse_key = (last_scan_at or "latest").replace(" ", "_").replace(",", "").replace(":", "-")
            render_imported_ai_actions(
                top_pulse_symbols,
                mode_value=0,
                key_prefix=f"live_pulse_imported_{top_pulse_key}",
                source_label="Live Breakout Pulse Top 3",
                source_bucket="breakout",
                source_rows=top_pulse_df,
                helper_text="Add these Live Breakout Pulse names into Imported AI Stocks for self-learning.",
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: All / by Signal ─────────────────────────────────────────
    _tab_all, _tab_lb, _tab_sm, _tab_watch = st.tabs([
        f"📋 All ({summary['total']})",
        f"🟢 Live Breakout ({summary['live_breakouts']})",
        f"🟡 Strong Momentum ({summary['strong_momentum']})",
        f"🔵 Watch ({summary['watch']})",
    ])

    # ── Download button (shared) ──────────────────────────────────────
    _dl_col, _info_col = st.columns([0.3, 0.7])
    with _dl_col:
        _stamp = (last_scan_at or datetime.now().strftime("%Y-%m-%d_%H-%M")).replace(
            " ", "_"
        ).replace(",", "").replace(":", "-")
        st.download_button(
            "⬇️ Download Results CSV",
            data=pulse_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"live_breakout_pulse_{_stamp}.csv",
            mime="text/csv",
            key="live_pulse_download_btn",
        )
    with _info_col:
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;padding-top:8px;">'
            '<b style="color:#00d4a8;">LIVE BREAKOUT</b> = Score ≥ 80 · '
            '<b style="color:#f0b429;">STRONG MOMENTUM</b> = 65–79 · '
            '<b style="color:#8ab4d8;">WATCH</b> = 50–64 · '
            'Vol/Avg ≥ 1.5 mandatory · RSI 78+ hard-rejected</div>',
            unsafe_allow_html=True,
        )

    # ── Column config (shared) ────────────────────────────────────────
    _col_cfg = {
        "Symbol":           st.column_config.TextColumn("Ticker"),
        _PRICE_DISPLAY_COL:  st.column_config.NumberColumn(_PRICE_DISPLAY_COL, format="\u20b9%.2f"),
        "Price (₹)":        st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
        "RSI":              st.column_config.NumberColumn("RSI", format="%.1f"),
        "Vol / Avg":        st.column_config.NumberColumn("Vol/Avg", format="%.2f×"),
        "Dist to High (%)": st.column_config.NumberColumn("Dist to High", format="%.2f%%"),
        "Δ vs EMA20 (%)":   st.column_config.NumberColumn("Δ EMA20", format="%.2f%%"),
        "EMA20 Slope":      st.column_config.TextColumn("EMA Slope"),
        "Momentum":         st.column_config.TextColumn("Momentum"),
        "Final Score":      st.column_config.NumberColumn("Score", format="%.1f"),
        "Signal":           st.column_config.TextColumn("Signal"),
        "Chart Link":       st.column_config.LinkColumn("Chart", display_text="📈 View"),
        "Action":           st.column_config.TextColumn("Action"),
        "Hold Days":        st.column_config.TextColumn("Hold Days"),
    }

    def _show_table(data: pd.DataFrame, key_suffix: str) -> None:
        if data.empty:
            st.info("No stocks in this category.")
            return
        display_data = apply_trade_decision_simple_any(data.copy())
        display_data = _with_live_pulse_display_aliases(display_data)
        st.caption(
            f"Showing top {_VISIBLE_RESULT_LIMIT} of {len(data)} results in this tab. Download keeps all rows."
        )
        st.dataframe(
            display_data.head(_VISIBLE_RESULT_LIMIT),
            column_config=_col_cfg,
            width="stretch",
            hide_index=True,
        )

    with _tab_all:
        _show_table(pulse_df, "all")

    with _tab_lb:
        _show_table(
            pulse_df[pulse_df["Signal"] == "LIVE BREAKOUT"].reset_index(drop=True),
            "lb",
        )

    with _tab_sm:
        _show_table(
            pulse_df[pulse_df["Signal"] == "STRONG MOMENTUM"].reset_index(drop=True),
            "sm",
        )

    with _tab_watch:
        _show_table(
            pulse_df[pulse_df["Signal"] == "WATCH"].reset_index(drop=True),
            "watch",
        )

    # ── Methodology expander ──────────────────────────────────────────
    with st.expander("📖 How Live Breakout Pulse works", expanded=False):
        st.markdown("""
**Score Components (0–100)**

| Component | Weight | Signal |
|---|---|---|
| Trend Strength | 30 % | Price > EMA20 > EMA50 · EMA20 slope rising |
| Volume Strength | 25 % | Vol / 20-day avg (≥ 1.5 mandatory) |
| Breakout Proximity | 20 % | Distance to 20-day rolling high |
| RSI Quality | 15 % | Sweet spot 55–68 · Penalised > 72 |
| Momentum | 10 % | Green candle · Close near day-high · Higher-high |

**Hard Reject Gates**
- Volume / Avg < 1.2 → rejected
- RSI > 78 → rejected (exhaustion)
- Price > 6 % above EMA20 → rejected (overextended)
- Price ≤ EMA20 → rejected (no trend)

**Signal Thresholds**
- 🟢 **LIVE BREAKOUT** → Score ≥ 80
- 🟡 **STRONG MOMENTUM** → Score 65–79
- 🔵 **WATCH** → Score 50–64

**Data Source:** Live yfinance · Last 60 days · ~500 most-liquid NSE stocks
        """)
