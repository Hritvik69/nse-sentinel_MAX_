# ═══════════════════════════════════════════════════════════════════════
# 🔭 INTERACTIVE SECTOR SCREENER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
#
# HOW TO ADD:
#   1. Place sector_screener_engine.py next to app.py
#   2. Paste this entire block into app.py just ABOVE the closing <hr>
#      at the bottom of the file (below the Battle Mode section).
#
# ALSO ADD at the top of app.py (if not already present):
#   from sector_screener_engine import build_sector_raw_rows, compute_sector_prediction
#
# DOES NOT modify any existing function or section.
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd

# ── Engine imports ────────────────────────────────────────────────────
try:
    from sector_screener_engine import build_sector_raw_rows, compute_sector_prediction  # type: ignore[import]
    _SSE_OK = True
except ImportError:
    _SSE_OK = False

try:
    from sector_master import (
        get_all_sectors, get_stocks_in_sector,
        get_sector_count, get_sector_description,
    )
    _SM_OK = True
except ImportError:
    _SM_OK = False
    def get_all_sectors():           return []
    def get_stocks_in_sector(s):     return []
    def get_sector_count():          return {}
    def get_sector_description(s):   return s

# ── Pipeline function stubs (resolved in app.py context) ─────────────
# These will be the real functions when pasted inside app.py.
# Left as stubs only so this file can be linted independently.
try:
    _enhance_results         = enhance_results            # type: ignore[name-defined]
    _apply_enhanced_logic    = apply_enhanced_logic       # type: ignore[name-defined]
    _apply_universal_grading = apply_universal_grading    # type: ignore[name-defined]
    _apply_phase4_logic      = apply_phase4_logic         # type: ignore[name-defined]
    _apply_phase42_logic     = apply_phase42_logic        # type: ignore[name-defined]
    _stored_mode             = stored_mode                # type: ignore[name-defined]
except NameError:
    def _enhance_results(rows, mode):        return pd.DataFrame()
    def _apply_enhanced_logic(df):           return df
    def _apply_universal_grading(df, mb=None): return df
    def _apply_phase4_logic(df, mb=None):    return df
    def _apply_phase42_logic(df):            return df
    _stored_mode = 2


# ─────────────────────────────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────

def _pred_color(pred: str) -> str:
    return {"UP": "#00d4a8", "DOWN": "#ff4d6d", "SIDEWAYS": "#f0b429"}.get(pred, "#8ab4d8")

def _pred_icon(pred: str) -> str:
    return {"UP": "📈", "DOWN": "📉", "SIDEWAYS": "➡️"}.get(pred, "—")

def _prob_color(p: float) -> str:
    if p >= 65: return "#00d4a8"
    if p >= 50: return "#f0b429"
    return "#ff4d6d"

def _flag_badge(flag: str) -> str:
    if flag == "FAKE_BULLISH":
        return '<span style="background:#ff4d6d22;color:#ff4d6d;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700;">⚠️ FAKE BULLISH</span>'
    if flag == "CAUTION":
        return '<span style="background:#f0b42922;color:#f0b429;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700;">⚡ CAUTION</span>'
    return '<span style="background:#00d4a822;color:#00d4a8;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700;">✅ CLEAN</span>'

def _run_full_pipeline(rows: list, mode: int) -> pd.DataFrame:
    """Run the full existing pipeline on raw rows. Never crashes."""
    try:
        df = _enhance_results(rows, mode)
        mb = st.session_state.get("market_bias_result", None)
        try:
            df = _apply_universal_grading(df, mb)
        except Exception:
            pass
        try:
            df = _apply_enhanced_logic(df)
        except Exception:
            pass
        try:
            df = _apply_phase4_logic(df, mb)
            df = _apply_phase42_logic(df)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# SECTION HEADER
# ─────────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<h2 style="margin-bottom:4px;">🔭 Sector Screener Dashboard</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#4a6480;margin-bottom:20px;">'
    'Click any sector card to run a full pipeline scan · Get next-day prediction · '
    'Or scan all 17 sectors at once for a complete market view</div>',
    unsafe_allow_html=True,
)

if not _SSE_OK or not _SM_OK:
    st.warning(
        "⚠️ sector_screener_engine.py or sector_master.py not found. "
        "Place both files next to app.py and restart."
    )
else:
    _ss_mode   = _stored_mode
    _ss_sectors = get_all_sectors()
    _ss_counts  = get_sector_count()

    # ── Session state keys ────────────────────────────────────────────
    if "ss_active_sector"   not in st.session_state: st.session_state["ss_active_sector"]   = None
    if "ss_sector_result"   not in st.session_state: st.session_state["ss_sector_result"]   = None
    if "ss_all_results"     not in st.session_state: st.session_state["ss_all_results"]     = []
    if "ss_scan_all_done"   not in st.session_state: st.session_state["ss_scan_all_done"]   = False

    # ════════════════════════════════════════════════════════
    # SECTION 1 — SECTOR GRID
    # ════════════════════════════════════════════════════════

    st.markdown(
        '<div style="font-size:14px;font-weight:700;color:#8ab4d8;'
        'letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">'
        '📊 Sector Grid</div>',
        unsafe_allow_html=True,
    )

    _COLS_PER_ROW = 3
    _sec_chunks   = [
        _ss_sectors[i: i + _COLS_PER_ROW]
        for i in range(0, len(_ss_sectors), _COLS_PER_ROW)
    ]

    for _chunk in _sec_chunks:
        _gcols = st.columns(len(_chunk))
        for _ci, (_gcol, _sec) in enumerate(zip(_gcols, _chunk)):
            _cnt  = _ss_counts.get(_sec, 0)
            _desc = get_sector_description(_sec)
            _is_active = st.session_state.get("ss_active_sector") == _sec
            _card_border = "#00d4a8" if _is_active else "#1e3a5f"
            _card_bg     = "#0d1e16" if _is_active else "#0b1017"

            with _gcol:
                st.markdown(
                    f'<div style="background:{_card_bg};border:1.5px solid {_card_border};'
                    f'border-radius:12px;padding:14px 16px;margin-bottom:6px;">'
                    f'<div style="font-weight:800;font-size:14px;color:#ccd9e8;">{_sec}</div>'
                    f'<div style="font-size:10px;color:#4a6480;margin:3px 0 6px;">{_desc}</div>'
                    f'<div style="font-size:12px;color:#8ab4d8;">📦 {_cnt} stocks</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"🔍 Scan {_sec}",
                    key=f"ss_scan_{_sec}",
                    width="stretch",
                ):
                    st.session_state["ss_active_sector"] = _sec
                    st.session_state["ss_sector_result"]  = None
                    st.session_state["ss_scan_all_done"]  = False
                    st.session_state["ss_all_results"]    = []
                    st.rerun()

    # ════════════════════════════════════════════════════════
    # PER-SECTOR SCAN RESULT
    # ════════════════════════════════════════════════════════

    _active_sec = st.session_state.get("ss_active_sector")

    if _active_sec and st.session_state.get("ss_sector_result") is None:
        _ss_stk_count = _ss_counts.get(_active_sec, 0)
        with st.spinner(
            f"🔍 Scanning {_active_sec} — fetching {_ss_stk_count} stocks…"
        ):
            try:
                _raw_rows = build_sector_raw_rows(_active_sec, _ss_mode)
                if _raw_rows:
                    _sec_df = _run_full_pipeline(_raw_rows, _ss_mode)
                    _pred   = compute_sector_prediction(_active_sec, _sec_df, _ss_mode)
                    st.session_state["ss_sector_result"] = {
                        "pred": _pred, "df": _sec_df,
                    }
                else:
                    st.session_state["ss_sector_result"] = {"pred": None, "df": pd.DataFrame()}
            except Exception as _e:
                st.session_state["ss_sector_result"] = {"pred": None, "df": pd.DataFrame(), "err": str(_e)}

    if _active_sec and st.session_state.get("ss_sector_result") is not None:
        _res = st.session_state["ss_sector_result"]
        _pred  = _res.get("pred")
        _sec_df = _res.get("df", pd.DataFrame())
        _err   = _res.get("err")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:14px;font-weight:700;color:#8ab4d8;'
            f'letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">'
            f'📈 Sector: {_active_sec}</div>',
            unsafe_allow_html=True,
        )

        if _err:
            st.error(f"Scan error: {_err}")
        elif _pred is None or _sec_df.empty:
            st.warning(
                f"No stocks in {_active_sec} passed data validation. "
                "Markets may be closed or data unavailable."
            )
        else:
            _p       = _pred["prediction"]
            _prob    = _pred["probability_pct"]
            _conf    = _pred["confidence_pct"]
            _bull    = _pred["bullish_pct"]
            _avg_sc  = _pred["avg_score"]
            _vol_str = _pred["volume_strength"]
            _flag    = _pred["flag"]
            _n       = _pred["stock_count"]

            _p_color  = _pred_color(_p)
            _p_icon   = _pred_icon(_p)
            _p_prob_c = _prob_color(_prob)

            # ── MAIN PREDICTION CARD ──────────────────────────────────
            st.markdown(
                f'<div style="background:#0b1017;border:2px solid {_p_color};'
                f'border-radius:16px;padding:22px 26px;margin-bottom:16px;">'
                # Header row
                f'<div style="display:flex;align-items:center;'
                f'justify-content:space-between;flex-wrap:wrap;gap:12px;">'
                f'<div>'
                f'<div style="font-size:12px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:4px;">Tomorrow\'s Prediction</div>'
                f'<div style="font-family:\'Syne\',sans-serif;font-size:36px;'
                f'font-weight:800;color:{_p_color};line-height:1;">'
                f'{_p_icon} {_p}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:12px;color:#4a6480;margin-bottom:2px;">Probability</div>'
                f'<div style="font-size:32px;font-weight:800;color:{_p_prob_c};">'
                f'{_prob:.0f}%</div>'
                f'</div>'
                f'</div>'
                # Metrics row
                f'<div style="display:flex;gap:24px;margin-top:18px;flex-wrap:wrap;">'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;">Confidence</div>'
                f'<div style="font-size:20px;font-weight:700;color:#0094ff;">{_conf:.0f}%</div></div>'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;">Bullish %</div>'
                f'<div style="font-size:20px;font-weight:700;color:#00d4a8;">{_bull:.0f}%</div></div>'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;">Avg Score</div>'
                f'<div style="font-size:20px;font-weight:700;color:#ccd9e8;">{_avg_sc:.1f}</div></div>'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;">Vol Strength</div>'
                f'<div style="font-size:20px;font-weight:700;color:#b08cff;">{_vol_str:.0f}</div></div>'
                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                f'letter-spacing:1px;">Stocks</div>'
                f'<div style="font-size:20px;font-weight:700;color:#8ab4d8;">{_n}</div></div>'
                f'<div style="display:flex;align-items:flex-end;padding-bottom:4px;">'
                f'{_flag_badge(_flag)}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── TOP 5 STOCKS ──────────────────────────────────────────
            if _pred["top_stocks"]:
                st.markdown(
                    '<div style="font-size:13px;font-weight:700;color:#8ab4d8;'
                    'margin:14px 0 8px;">🏆 Top Stocks in Sector</div>',
                    unsafe_allow_html=True,
                )

                _ts_html = (
                    '<div style="overflow-x:auto;">'
                    '<table style="width:100%;border-collapse:collapse;font-size:13px;'
                    'color:#ccd9e8;">'
                    '<thead><tr style="border-bottom:1.5px solid #1e3a5f;">'
                    '<th style="text-align:left;padding:7px 10px;color:#4a6480;">Symbol</th>'
                    '<th style="text-align:right;padding:7px 10px;color:#4a6480;">Score</th>'
                    '<th style="text-align:center;padding:7px 10px;color:#4a6480;">Signal</th>'
                    '<th style="text-align:center;padding:7px 10px;color:#4a6480;">Grade</th>'
                    '<th style="text-align:center;padding:7px 10px;color:#4a6480;">Trap Risk</th>'
                    '</tr></thead><tbody>'
                )
                for _ri, _ts in enumerate(_pred["top_stocks"]):
                    _bg = "#0d1520" if _ri % 2 == 0 else "#0b1017"
                    _sc_c = _prob_color(_ts["score"])
                    _trap_c = {"HIGH": "#ff4d6d", "MEDIUM": "#f0b429", "LOW": "#00d4a8"}.get(
                        _ts.get("trap", "LOW"), "#8ab4d8"
                    )
                    _ts_html += (
                        f'<tr style="background:{_bg};border-bottom:1px solid #0f1e2f;">'
                        f'<td style="padding:8px 10px;font-weight:700;">{_ts["symbol"]}</td>'
                        f'<td style="padding:8px 10px;text-align:right;'
                        f'font-weight:800;color:{_sc_c};">{_ts["score"]:.1f}</td>'
                        f'<td style="padding:8px 10px;text-align:center;color:#f0b429;">'
                        f'{_ts["signal"]}</td>'
                        f'<td style="padding:8px 10px;text-align:center;">'
                        f'{_ts["grade"]}</td>'
                        f'<td style="padding:8px 10px;text-align:center;'
                        f'color:{_trap_c};font-weight:700;">{_ts["trap"]}</td>'
                        f'</tr>'
                    )
                _ts_html += "</tbody></table></div>"
                st.markdown(_ts_html, unsafe_allow_html=True)

            # ── FULL SCAN DATAFRAME (expandable) ──────────────────────
            if not _sec_df.empty:
                with st.expander(f"📋 Full scan data — {len(_sec_df)} stocks", expanded=False):
                    _disp_cols = [
                        c for c in [
                            "Symbol", "Price (₹)", "RSI", "Vol / Avg",
                            "Final Score", "Prediction Score", "Signal",
                            "Grade", "Trap Risk", "Volume Trend",
                            "5D Return (%)", "Δ vs EMA20 (%)",
                        ]
                        if c in _sec_df.columns
                    ]
                    st.dataframe(
                        _sec_df[_disp_cols] if _disp_cols else _sec_df,
                        width="stretch",
                        hide_index=True,
                    )

    # ════════════════════════════════════════════════════════
    # SECTION 2 — SCAN ALL SECTORS
    # ════════════════════════════════════════════════════════

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:14px;font-weight:700;color:#8ab4d8;'
        'letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">'
        '🔄 Full Market Scan</div>',
        unsafe_allow_html=True,
    )

    _scan_all_col1, _scan_all_col2 = st.columns([2, 5])
    with _scan_all_col1:
        _scan_all_btn = st.button(
            "🚀 Scan All Sectors",
            key="ss_scan_all_btn",
            width="stretch",
        )
    with _scan_all_col2:
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;padding-top:10px;">'
            f'Will scan all {len(_ss_sectors)} sectors · May take 1–3 minutes · '
            'Does not block existing scan results</div>',
            unsafe_allow_html=True,
        )

    if _scan_all_btn:
        st.session_state["ss_scan_all_done"]  = False
        st.session_state["ss_all_results"]    = []
        st.session_state["ss_active_sector"]  = None
        st.session_state["ss_sector_result"]  = None

        _all_predictions: list[dict] = []
        _progress_bar = st.progress(0, text="Starting sector scan…")

        for _si, _sec in enumerate(_ss_sectors):
            try:
                _pct_text = f"Scanning {_sec} ({_si + 1}/{len(_ss_sectors)})…"
                _progress_bar.progress(
                    int((_si / len(_ss_sectors)) * 100),
                    text=_pct_text,
                )
                _raw  = build_sector_raw_rows(_sec, _ss_mode)
                _df_s = _run_full_pipeline(_raw, _ss_mode) if _raw else pd.DataFrame()
                _pr   = compute_sector_prediction(_sec, _df_s, _ss_mode)
                _all_predictions.append(_pr)
            except Exception:
                _all_predictions.append({
                    "sector": _sec,
                    "prediction": "SIDEWAYS",
                    "probability_pct": 50.0,
                    "confidence_pct": 40.0,
                    "bullish_pct": 50.0,
                    "avg_score": 50.0,
                    "volume_strength": 50.0,
                    "top_stocks": [],
                    "flag": "",
                    "stock_count": 0,
                    "mode": _ss_mode,
                })

        _progress_bar.progress(100, text="✅ Scan complete!")
        st.session_state["ss_all_results"]  = _all_predictions
        st.session_state["ss_scan_all_done"] = True

    # ── DISPLAY ALL SECTOR RESULTS ────────────────────────────────────
    if st.session_state.get("ss_scan_all_done") and st.session_state.get("ss_all_results"):
        _all_preds: list[dict] = st.session_state["ss_all_results"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:13px;font-weight:700;color:#8ab4d8;'
            'margin-bottom:10px;">📊 All Sector Predictions</div>',
            unsafe_allow_html=True,
        )

        # ── Results table ─────────────────────────────────────────────
        _all_html = (
            '<div style="overflow-x:auto;">'
            '<table style="width:100%;border-collapse:collapse;font-size:13px;color:#ccd9e8;">'
            '<thead><tr style="border-bottom:1.5px solid #1e3a5f;">'
            '<th style="text-align:left;padding:8px 10px;color:#4a6480;">Sector</th>'
            '<th style="text-align:center;padding:8px 10px;color:#4a6480;">Prediction</th>'
            '<th style="text-align:right;padding:8px 10px;color:#4a6480;">Probability</th>'
            '<th style="text-align:right;padding:8px 10px;color:#4a6480;">Confidence</th>'
            '<th style="text-align:right;padding:8px 10px;color:#4a6480;">Bullish%</th>'
            '<th style="text-align:right;padding:8px 10px;color:#4a6480;">Avg Score</th>'
            '<th style="text-align:center;padding:8px 10px;color:#4a6480;">Flag</th>'
            '</tr></thead><tbody>'
        )

        _sorted_preds = sorted(
            _all_preds,
            key=lambda x: x.get("probability_pct", 50.0),
            reverse=True,
        )

        for _ri, _pr in enumerate(_sorted_preds):
            _bg  = "#0d1520" if _ri % 2 == 0 else "#0b1017"
            _p   = _pr.get("prediction", "SIDEWAYS")
            _pc  = _pred_color(_p)
            _pi  = _pred_icon(_p)
            _prb = _pr.get("probability_pct", 50.0)
            _cof = _pr.get("confidence_pct",  40.0)
            _bl  = _pr.get("bullish_pct",     50.0)
            _sc  = _pr.get("avg_score",        50.0)
            _fl  = _pr.get("flag", "")
            _sn  = _pr.get("sector", "")
            _desc = get_sector_description(_sn)

            _fl_badge = (
                "⚠️ FAKE BULLISH" if _fl == "FAKE_BULLISH"
                else "⚡ CAUTION"  if _fl == "CAUTION"
                else "✅"
            )
            _fl_col = (
                "#ff4d6d" if _fl == "FAKE_BULLISH"
                else "#f0b429" if _fl == "CAUTION"
                else "#00d4a8"
            )

            _all_html += (
                f'<tr style="background:{_bg};border-bottom:1px solid #0f1e2f;">'
                f'<td style="padding:9px 10px;">'
                f'<div style="font-weight:700;">{_sn}</div>'
                f'<div style="font-size:10px;color:#4a6480;">{_desc}</div></td>'
                f'<td style="padding:9px 10px;text-align:center;">'
                f'<span style="font-weight:800;color:{_pc};">{_pi} {_p}</span></td>'
                f'<td style="padding:9px 10px;text-align:right;'
                f'font-weight:800;color:{_prob_color(_prb)};">{_prb:.0f}%</td>'
                f'<td style="padding:9px 10px;text-align:right;color:#0094ff;">{_cof:.0f}%</td>'
                f'<td style="padding:9px 10px;text-align:right;color:#00d4a8;">{_bl:.0f}%</td>'
                f'<td style="padding:9px 10px;text-align:right;color:#ccd9e8;">{_sc:.1f}</td>'
                f'<td style="padding:9px 10px;text-align:center;'
                f'font-size:11px;color:{_fl_col};">{_fl_badge}</td>'
                f'</tr>'
            )

        _all_html += '</tbody></table></div>'
        st.markdown(_all_html, unsafe_allow_html=True)

        # ── FINAL SUMMARY PANEL ───────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        # Compute market direction
        _up_count   = sum(1 for p in _all_preds if p.get("prediction") == "UP")
        _down_count = sum(1 for p in _all_preds if p.get("prediction") == "DOWN")
        _total_secs = len(_all_preds) or 1

        if _up_count / _total_secs >= 0.50:
            _mkt_dir   = "🚀 BULLISH"
            _mkt_color = "#00d4a8"
        elif _down_count / _total_secs >= 0.50:
            _mkt_dir   = "🔻 BEARISH"
            _mkt_color = "#ff4d6d"
        else:
            _mkt_dir   = "➡️ MIXED"
            _mkt_color = "#f0b429"

        # Strongest and weakest
        _valid = [p for p in _all_preds if p.get("stock_count", 0) > 0]
        _strongest = max(_valid, key=lambda x: x.get("probability_pct", 0.0), default={})
        _weakest   = min(_valid, key=lambda x: x.get("probability_pct", 100.0), default={})

        _strong_name = _strongest.get("sector", "—")
        _strong_prob = _strongest.get("probability_pct", 0.0)
        _weak_name   = _weakest.get("sector", "—")
        _weak_prob   = _weakest.get("probability_pct", 0.0)

        st.markdown(
            f'<div style="background:#0b1017;border:2px solid {_mkt_color};'
            f'border-radius:16px;padding:22px 26px;">'
            # Market direction
            f'<div style="font-family:\'Syne\',sans-serif;font-size:22px;'
            f'font-weight:800;color:{_mkt_color};margin-bottom:16px;">'
            f'Market Direction: {_mkt_dir}</div>'
            # Stats row
            f'<div style="display:flex;gap:32px;flex-wrap:wrap;">'
            f'<div>'
            f'<div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
            f'letter-spacing:1px;">🔥 Strongest Sector</div>'
            f'<div style="font-size:18px;font-weight:800;color:#00d4a8;">'
            f'{_strong_name}</div>'
            f'<div style="font-size:12px;color:#4a6480;">{_strong_prob:.0f}% probability</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
            f'letter-spacing:1px;">⚠️ Weakest Sector</div>'
            f'<div style="font-size:18px;font-weight:800;color:#ff4d6d;">'
            f'{_weak_name}</div>'
            f'<div style="font-size:12px;color:#4a6480;">{_weak_prob:.0f}% probability</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
            f'letter-spacing:1px;">Sectors Bullish</div>'
            f'<div style="font-size:18px;font-weight:800;color:#ccd9e8;">'
            f'{_up_count}/{_total_secs}</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
            f'letter-spacing:1px;">Sectors Bearish</div>'
            f'<div style="font-size:18px;font-weight:800;color:#ccd9e8;">'
            f'{_down_count}/{_total_secs}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Export CSV
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📥 Export all-sector results as CSV", expanded=False):
            _export_rows = [
                {
                    "Sector":      p.get("sector", ""),
                    "Prediction":  p.get("prediction", ""),
                    "Probability": p.get("probability_pct", 0.0),
                    "Confidence":  p.get("confidence_pct", 0.0),
                    "Bullish%":    p.get("bullish_pct", 0.0),
                    "Avg Score":   p.get("avg_score", 0.0),
                    "Vol Strength":p.get("volume_strength", 0.0),
                    "Flag":        p.get("flag", ""),
                    "Stocks":      p.get("stock_count", 0),
                }
                for p in _sorted_preds
            ]
            _csv_str = pd.DataFrame(_export_rows).to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=_csv_str,
                file_name="sector_screener_results.csv",
                mime="text/csv",
                key="ss_csv_download",
            )