# ═══════════════════════════════════════════════════════════════════════
# 📊 SECTOR INTELLIGENCE EXPLORER  (Advanced Layer — Upgrade 8)
# ═══════════════════════════════════════════════════════════════════════
#
# HOW TO ADD (integrated in app.py):
#   from strategy_engines.app_sector_intelligence_section import render_sector_intelligence_section
#   After the enriched scan DataFrame is ready:
#       st.session_state["last_scan_df"] = df.copy()
#   Then call: render_sector_intelligence_section()
#   (Place after Sector Explorer if you use app_sector_explorer_section.py.)
#
# This section ADDS new UI — it does NOT modify any existing section.
# ═══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import pandas as pd
import streamlit as st

# ── Safe import of the intelligence engine ────────────────────────────
try:
    from strategy_engines.sector_intelligence_engine import compute_sector_intelligence
    _SIE_OK = True
    _SIE_ERR = ""
except ImportError as exc:
    try:
        from sector_intelligence_engine import compute_sector_intelligence
        _SIE_OK = True
        _SIE_ERR = ""
    except ImportError:
        _SIE_OK = False
        _SIE_ERR = str(exc).strip() or "sector intelligence import failed"

# ── Safe import of sector_master for base data ────────────────────────
try:
    from sector_master import get_sector_description, SECTOR_DESCRIPTIONS  # BUG FIX: root-level file
    _SM_OK = True
except ImportError:
    try:
        from sector_master import get_sector_description, SECTOR_DESCRIPTIONS
        _SM_OK = True
    except ImportError:
        _SM_OK = False
        SECTOR_DESCRIPTIONS: dict = {}
        def get_sector_description(s): return s

# ── Flow signal display helpers ───────────────────────────────────────
_FLOW_ICON = {
    "MONEY_INFLOW":  "🟢 Inflow",
    "MONEY_OUTFLOW": "🔴 Outflow",
    "STABLE":        "⚪ Stable",
}
_FLOW_COLOR = {
    "MONEY_INFLOW":  "#00d4a8",
    "MONEY_OUTFLOW": "#ff4d6d",
    "STABLE":        "#8ab4d8",
}

def _strength_color(score: float) -> str:
    """Map sector_strength score to a hex color."""
    if score >= 68:
        return "#00d4a8"   # strong green
    if score >= 50:
        return "#f0b429"   # amber
    return "#ff4d6d"       # weak red

def _strength_label(score: float) -> str:
    if score >= 68:
        return "STRONG"
    if score >= 50:
        return "NEUTRAL"
    return "WEAK"


def _sector_intel_cache_key(df: pd.DataFrame) -> tuple[object, ...]:
    """Return a deterministic cache key for the current scan frame."""
    try:
        hashed = pd.util.hash_pandas_object(df.fillna(""), index=True).sum()
        return (
            len(df),
            tuple(str(col) for col in df.columns),
            int(hashed),
        )
    except Exception:
        return (
            len(df),
            tuple(str(col) for col in df.columns),
        )

# ─────────────────────────────────────────────────────────────────────

def render_sector_intelligence_section() -> None:
    """Render Sector Intelligence UI (call from app.py after scan results)."""
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-bottom:4px;">🧠 Sector Intelligence Engine</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
        'Advanced Layer · Dynamic weighting · Rotation detection · Leader stocks · '
        'Real-time sector strength — powered by scan data</div>',
        unsafe_allow_html=True,
    )

    if not _SIE_OK:
        st.warning(
            "Sector Intelligence is unavailable because `sector_intelligence_engine.py` "
            f"could not be imported. Import error: {_SIE_ERR}"
        )
    else:
        # ── Pull scan DataFrame from session state ─────────────────────────
        # Primary: enriched scan stored by app.py.
        # Fallback: raw scan `results` (if enrichment wasn't stored yet),
        # so the UI still works without extra user steps.
        _sie_df: pd.DataFrame | None = st.session_state.get("last_scan_df", None)
        _used_fallback = False
        if _sie_df is None:
            _raw_results = st.session_state.get("results", None)
            if isinstance(_raw_results, list) and _raw_results:
                _candidate_df = pd.DataFrame(_raw_results)
                if isinstance(_candidate_df, pd.DataFrame) and not _candidate_df.empty:
                    _sie_df = _candidate_df
                    _used_fallback = True

        _sie_col_run, _sie_col_info = st.columns([2, 5])
        with _sie_col_run:
            _sie_run_btn = st.button(
                "🧠 Run Sector Intelligence",
                key="sie_run_btn",
                width="stretch",
            )
        with _sie_col_info:
            if _sie_df is None or (isinstance(_sie_df, pd.DataFrame) and _sie_df.empty):
                st.markdown(
                    '<div style="font-size:12px;color:#f0b429;padding-top:10px;">'
                    '⚠️ No scan data found. Run a stock scan first, then click above.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="font-size:12px;color:#4a6480;padding-top:10px;">'
                    f'📦 Using {"fallback " if _used_fallback else "last scan"}: '
                    f'<b style="color:#00d4a8;">{len(_sie_df)} stocks</b> loaded</div>',
                    unsafe_allow_html=True,
                )

        if _sie_df is None or (isinstance(_sie_df, pd.DataFrame) and _sie_df.empty):
            st.warning("Run a stock scan first to populate scan data.")
            return

        scan_sig = _sector_intel_cache_key(_sie_df)
        _cached_sig = st.session_state.get("_sector_intel_scan_sig", None)
        _cached_intel = st.session_state.get("_sector_intel_result", None)

        _should_compute = _sie_run_btn or (_cached_intel is None) or (_cached_sig != scan_sig)
        if _should_compute:
            try:
                with st.spinner("🧠 Computing sector intelligence…"):
                    _intel = compute_sector_intelligence(_sie_df)
            except Exception as exc:
                st.warning(
                    "Sector Intelligence could not be computed for the current scan. "
                    f"Reason: {str(exc).strip() or exc.__class__.__name__}"
                )
                return
            st.session_state["_sector_intel_result"] = _intel
            st.session_state["_sector_intel_scan_sig"] = scan_sig
        else:
            _intel = _cached_intel

        if not isinstance(_intel, dict):
            st.warning("Sector Intelligence returned an invalid payload for this scan.")
            return

        _sie_details  = _intel.get("sector_details",  {})
        _sie_ranking  = _intel.get("sector_ranking",  [])
        _sie_dominant = _intel.get("dominant_sector", {})
        _sie_summary  = _intel.get("overall_summary", {})

        if not _sie_ranking:
            st.warning(
                "Sector intelligence could not extract enough data. "
                "Ensure your scan includes Symbol, Price, Volume, "
                "Prediction Score and 5D Return columns."
            )
        else:
                    # ── SUMMARY BANNER ────────────────────────────────────
                    _dom = _sie_dominant.get("dominant_sector", "")
                    _dom_str = _sie_dominant.get("strength", 0.0)
                    _avg_str = _sie_summary.get("avg_sector_strength", 0.0)
                    _top_sec = _sie_summary.get("top_sector", "")
                    _weak_sec = _sie_summary.get("weakest_sector", "")
                    _n_sec   = _sie_summary.get("sectors_analysed", 0)
                    _n_stocks = _sie_summary.get("total_stocks", 0)

                    # Dominant sector card
                    st.markdown("<br>", unsafe_allow_html=True)
                    if _dom:
                        _dom_color = _strength_color(_dom_str)
                        st.markdown(
                            f'<div style="background:#0b1017;border:2px solid {_dom_color};'
                            f'border-radius:14px;padding:16px 22px;margin-bottom:14px;'
                            f'display:flex;align-items:center;gap:18px;flex-wrap:wrap;">'
                            f'<div style="font-size:28px;">🏆</div>'
                            f'<div>'
                            f'<div style="font-family:\'Syne\',sans-serif;font-size:18px;'
                            f'font-weight:800;color:{_dom_color};">Dominant Sector: {_dom}</div>'
                            f'<div style="font-size:11px;color:#4a6480;margin-top:3px;">'
                            f'{get_sector_description(_dom)} · '
                            f'Strength {_dom_str:.1f} · '
                            f'Market bias BOOSTED by this sector</div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # ── SECTION 1 — 🚀 MARKET SUMMARY (TOP PANEL) ─────────
                    if _avg_str >= 68:
                        _market_label = "BULLISH"
                        _market_color = "#00d4a8"
                    elif _avg_str >= 50:
                        _market_label = "NEUTRAL"
                        _market_color = "#f0b429"
                    else:
                        _market_label = "BEARISH"
                        _market_color = "#ff4d6d"

                    _spread = abs(_avg_str - 50.0)
                    _confidence = max(55, min(95, 55 + _spread * 1.2))

                    st.markdown("<br>", unsafe_allow_html=True)
                    _top_row_left, _top_row_right = st.columns([2, 3])

                    with _top_row_left:
                        st.markdown(
                            f'<div style="background:#0b1017;border-radius:14px;'
                            f'border:1px solid {_market_color};padding:16px 20px;'
                            f'display:flex;flex-direction:column;gap:4px;">'
                            f'<div style="font-size:12px;color:#4a6480;">Market Condition</div>'
                            f'<div style="font-size:26px;font-weight:800;color:{_market_color};">'
                            f'{_market_label}</div>'
                            f'<div style="font-size:11px;color:#8ab4d8;">'
                            f'Avg sector strength {_avg_str:.1f}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with _top_row_right:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Market", _market_label)
                        c2.metric("Confidence", f"{_confidence:.0f}%")
                        c3.metric("Dominant Sector", _dom or "—")
                        c4.metric("Weak Sector", _weak_sec or "—")

                    # ── SECTION 2 — 📊 SECTOR RANKING TABLE ───────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<h3 style="margin-bottom:4px;">📊 Sector Ranking</h3>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div style="font-size:12px;color:#4a6480;margin-bottom:12px;">'
                        'All detected sectors sorted by composite strength score, '
                        'with momentum, money flow and signal quality.</div>',
                        unsafe_allow_html=True,
                    )

                    _rank_html = (
                        '<div style="overflow-x:auto;">'
                        '<table style="width:100%;border-collapse:collapse;'
                        'font-size:13px;color:#ccd9e8;">'
                        '<thead>'
                        '<tr style="border-bottom:1.5px solid #1e3a5f;">'
                        '<th style="text-align:left;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Rank</th>'
                        '<th style="text-align:left;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Sector</th>'
                        '<th style="text-align:right;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Strength</th>'
                        '<th style="text-align:right;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Momentum</th>'
                        '<th style="text-align:center;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Flow</th>'
                        '<th style="text-align:center;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Signal</th>'
                        '<th style="text-align:left;padding:8px 10px;color:#4a6480;'
                        'font-weight:600;">Leaders</th>'
                        '</tr></thead><tbody>'
                    )

                    for _ri, _r in enumerate(_sie_ranking, 1):
                        _sec_name = _r.get("sector", "")
                        _sec_str  = _r.get("sector_strength", 0.0)
                        _sec_mom  = _r.get("momentum_score", 0.0)
                        _sec_flow = _r.get("flow_signal", "STABLE")
                        _sec_ldr  = _r.get("leader_stocks", [])
                        _sec_desc = get_sector_description(_sec_name)
                        _col      = _strength_color(_sec_str)
                        _lbl      = _strength_label(_sec_str)
                        _flow_lbl = _FLOW_ICON.get(_sec_flow, "⚪ Stable")
                        _flow_col = _FLOW_COLOR.get(_sec_flow, "#8ab4d8")
                        _leaders_str = ", ".join(_sec_ldr) if _sec_ldr else "—"
                        _bg = "#0d1520" if _ri % 2 == 0 else "#0b1017"

                        _rank_html += (
                            f'<tr style="background:{_bg};border-bottom:1px solid #0f1e2f;">'
                            f'<td style="padding:9px 10px;font-weight:700;color:#4a6480;">#{_ri}</td>'
                            f'<td style="padding:9px 10px;">'
                            f'<div style="font-weight:700;">{_sec_name}</div>'
                            f'<div style="font-size:10px;color:#4a6480;">{_sec_desc}</div>'
                            f'</td>'
                            f'<td style="padding:9px 10px;text-align:right;">'
                            f'<span style="font-weight:800;color:{_col};font-size:15px;">'
                            f'{_sec_str:.1f}</span></td>'
                            f'<td style="padding:9px 10px;text-align:right;color:#8ab4d8;">'
                            f'{_sec_mom:.1f}</td>'
                            f'<td style="padding:9px 10px;text-align:center;">'
                            f'<span style="color:{_flow_col};font-size:12px;">{_flow_lbl}</span>'
                            f'</td>'
                            f'<td style="padding:9px 10px;text-align:center;">'
                            f'<span style="background:{_col}22;color:{_col};'
                            f'border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700;">'
                            f'{_lbl}</span></td>'
                            f'<td style="padding:9px 10px;font-size:11px;color:#8ab4d8;">'
                            f'{_leaders_str}</td>'
                            f'</tr>'
                        )

                    _rank_html += '</tbody></table></div>'
                    st.markdown(_rank_html, unsafe_allow_html=True)

                    with st.expander("📋 Export ranking as table", expanded=False):
                        _export_rows = []
                        for _ri, _r in enumerate(_sie_ranking, 1):
                            _export_rows.append({
                                "Rank":     _ri,
                                "Sector":   _r.get("sector", ""),
                                "Strength": round(_r.get("sector_strength", 0.0), 1),
                                "Momentum": round(_r.get("momentum_score", 0.0), 1),
                                "Flow":     _FLOW_ICON.get(_r.get("flow_signal", ""), "⚪ Stable"),
                                "Signal":   _strength_label(_r.get("sector_strength", 0.0)),
                                "Leaders":  ", ".join(_r.get("leader_stocks", [])),
                                "Stocks":   _r.get("stock_count", 0),
                                "Bullish%": round(_r.get("bullish_pct", 0.0), 1),
                            })
                        st.dataframe(
                            pd.DataFrame(_export_rows),
                            width="stretch",
                            hide_index=True,
                        )

                    # ── SECTION 3 — 🔄 MONEY FLOW PANEL ───────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<h3 style="margin-bottom:4px;">🔄 Money Flow</h3>',
                        unsafe_allow_html=True,
                    )

                    _inflow_secs  = [
                        r for r in _sie_ranking if r.get("flow_signal") == "MONEY_INFLOW"
                    ]
                    _outflow_secs = [
                        r for r in _sie_ranking if r.get("flow_signal") == "MONEY_OUTFLOW"
                    ]

                    _top_inflow_names = ", ".join(
                        r.get("sector", "") for r in _inflow_secs[:3]
                    ) or "—"
                    _top_outflow_names = ", ".join(
                        r.get("sector", "") for r in _outflow_secs[:3]
                    ) or "—"

                    st.markdown(
                        f'<div style="font-size:12px;color:#ccd9e8;'
                        f'background:#0b1017;border-radius:10px;'
                        f'border:1px solid #1e3a5f;padding:10px 14px;margin-bottom:10px;">'
                        f'💰 Inflow: <b style="color:#00d4a8;">{_top_inflow_names}</b><br>'
                        f'🧯 Outflow: <b style="color:#ff4d6d;">{_top_outflow_names}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    _fc1, _fc2 = st.columns(2)

                    with _fc1:
                        st.markdown(
                            '<div style="color:#00d4a8;font-weight:800;'
                            'font-size:14px;margin-bottom:8px;">🟢 MONEY INFLOW</div>',
                            unsafe_allow_html=True,
                        )
                        if _inflow_secs:
                            for _sec_r in _inflow_secs[:6]:
                                _sn  = _sec_r.get("sector", "")
                                _ss  = _sec_r.get("sector_strength", 0.0)
                                _ldr = ", ".join(_sec_r.get("leader_stocks", []))
                                _leaders_txt = _ldr or "—"
                                st.markdown(
                                    f'<div style="background:#0d1e14;border:1px solid #00d4a833;'
                                    f'border-radius:8px;padding:10px 14px;margin-bottom:8px;">'
                                    f'<span style="font-weight:700;color:#ccd9e8;">{_sn}</span> '
                                    f'<span style="color:#00d4a8;font-size:12px;">▲ {_ss:.1f}</span>'
                                    f'<div style="font-size:11px;color:#4a6480;margin-top:4px;">'
                                    f'Leaders: {_leaders_txt}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                '<div style="color:#4a6480;font-size:12px;">'
                                'No inflow signals yet · Run scan again to compare</div>',
                                unsafe_allow_html=True,
                            )

                    with _fc2:
                        st.markdown(
                            '<div style="color:#ff4d6d;font-weight:800;'
                            'font-size:14px;margin-bottom:8px;">🔴 MONEY OUTFLOW</div>',
                            unsafe_allow_html=True,
                        )
                        if _outflow_secs:
                            for _sec_r in _outflow_secs[:6]:
                                _sn  = _sec_r.get("sector", "")
                                _ss  = _sec_r.get("sector_strength", 0.0)
                                st.markdown(
                                    f'<div style="background:#1e0d10;border:1px solid #ff4d6d33;'
                                    f'border-radius:8px;padding:10px 14px;margin-bottom:8px;">'
                                    f'<span style="font-weight:700;color:#ccd9e8;">{_sn}</span> '
                                    f'<span style="color:#ff4d6d;font-size:12px;">▼ {_ss:.1f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                '<div style="color:#4a6480;font-size:12px;">'
                                'No outflow signals yet · Run scan again to compare</div>',
                                unsafe_allow_html=True,
                            )

                    # ── SECTION 4 — 🏆 SECTOR LEADERS ─────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<h3 style="margin-bottom:4px;">🏆 Sector Leaders</h3>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div style="font-size:12px;color:#4a6480;margin-bottom:10px;">'
                        'Top three sectors and their leader stocks (relative strength × volume).</div>',
                        unsafe_allow_html=True,
                    )

                    _top_sectors_for_leaders = _sie_ranking[:3]
                    _ldr_cols = st.columns(3)
                    for _li, _sec_r in enumerate(_top_sectors_for_leaders):
                        _sec_n   = _sec_r.get("sector", "")
                        _sec_str = _sec_r.get("sector_strength", 0.0)
                        _leaders = _sec_r.get("leader_stocks", [])[:3]
                        _col_c   = _strength_color(_sec_str)
                        _desc    = get_sector_description(_sec_n)

                        _card = (
                            f'<div style="background:#0b1017;border:1.5px solid #1e3a5f;'
                            f'border-radius:12px;padding:14px 16px;margin-bottom:10px;">'
                            f'<div style="display:flex;align-items:center;'
                            f'justify-content:space-between;margin-bottom:10px;">'
                            f'<div>'
                            f'<span style="font-weight:800;font-size:14px;color:#ccd9e8;">'
                            f'{_sec_n}</span>'
                            f'<span style="font-size:10px;color:#4a6480;margin-left:8px;">'
                            f'{_desc}</span>'
                            f'</div>'
                            f'<span style="font-size:13px;font-weight:800;color:{_col_c};">'
                            f'{_sec_str:.1f}</span>'
                            f'</div>'
                        )

                        if _leaders:
                            for _li_rank, _leader in enumerate(_leaders, 1):
                                _medal = ["🥇", "🥈", "🥉"][_li_rank - 1]
                                _card += (
                                    f'<div style="display:flex;align-items:center;'
                                    f'gap:8px;margin-bottom:5px;">'
                                    f'<span style="font-size:14px;">{_medal}</span>'
                                    f'<span style="font-weight:700;font-size:13px;'
                                    f'color:#ccd9e8;">{_leader}</span>'
                                    f'</div>'
                                )
                        else:
                            _card += '<div style="color:#4a6480;font-size:12px;">No leaders found</div>'

                        _card += '</div>'
                        _ldr_cols[_li % 3].markdown(_card, unsafe_allow_html=True)

                    # ── SECTION 5 — 🎯 TRADE DECISION PANEL ───────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<h3 style="margin-bottom:4px;">🎯 Trade Decision Panel</h3>',
                        unsafe_allow_html=True,
                    )

                    _best_sector = _top_sec or _dom or "—"
                    _avoid_sector = _weak_sec or "—"

                    td1, td2, td3 = st.columns(3)
                    with td1:
                        st.markdown(
                            f'<div style="background:#0b1017;border-radius:10px;'
                            f'border:1px solid #00d4a8;padding:12px 14px;">'
                            f'<div style="font-size:11px;color:#4a6480;margin-bottom:4px;">'
                            f'Trade Focus</div>'
                            f'<div style="font-size:18px;font-weight:800;color:#00d4a8;">'
                            f'{_best_sector}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with td2:
                        st.markdown(
                            f'<div style="background:#0b1017;border-radius:10px;'
                            f'border:1px solid #ff4d6d;padding:12px 14px;">'
                            f'<div style="font-size:11px;color:#4a6480;margin-bottom:4px;">'
                            f'Avoid Sector</div>'
                            f'<div style="font-size:18px;font-weight:800;color:#ff4d6d;">'
                            f'{_avoid_sector}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with td3:
                        st.markdown(
                            f'<div style="background:#0b1017;border-radius:10px;'
                            f'border:1px solid {_market_color};padding:12px 14px;">'
                            f'<div style="font-size:11px;color:#4a6480;margin-bottom:4px;">'
                            f'Market Type</div>'
                            f'<div style="font-size:18px;font-weight:800;color:{_market_color};">'
                            f'{_market_label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
