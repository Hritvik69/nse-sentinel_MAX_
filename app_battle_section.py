# ═══════════════════════════════════════════════════════════════════════
# ⚔️  MULTI-STOCK BATTLE MODE  (10-Box Individual Input UI)
# ═══════════════════════════════════════════════════════════════════════
# HOW TO ADD: paste this entire block into app.py just ABOVE the line:
#   st.markdown("<hr>", unsafe_allow_html=True)
# that appears near the very bottom of app.py (after the CSV scan section).
# Do NOT modify anything else.
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
from strategy_engines.nse_autocomplete import (
    configure_nse_stock_search,
    search_nse_stocks,
)

# Safe stubs for lint/static analysis (this file is not imported by app.py).
try:
    stored_mode = st.session_state.get("mode", 2)
except Exception:
    stored_mode = 2

def enhance_results(*args, **kwargs):
    return pd.DataFrame()

def apply_universal_grading(df, market_bias=None):
    return df

def apply_enhanced_logic(df):
    return df

def apply_phase4_logic(df, market_bias_dict=None):
    return df

def apply_phase42_logic(df):
    return df

try:
    from battle_mode_engine import run_battle_mode, compute_battle_scores
    _BATTLE_OK = True
except ImportError:
    _BATTLE_OK = False


def stock_search_widget(label: str, key_prefix: str, placeholder: str) -> str:
    """
    Returns bare symbol string e.g. "RELIANCE" or "" if nothing selected.
    """
    query = st.text_input(
        label,
        placeholder=placeholder,
        key=f"{key_prefix}_input",
    ).strip().upper()

    if not query:
        return ""

    matches = search_nse_stocks(query)
    if not matches:
        st.caption("No matches found.")
        return ""

    select_key = f"{key_prefix}_select"
    options = [""] + matches
    if st.session_state.get(select_key, "") not in options:
        st.session_state[select_key] = ""

    chosen = st.selectbox(
        "Select stock",
        options=options,
        key=select_key,
        label_visibility="collapsed",
    )

    if not chosen:
        return ""

    return chosen.split("—", 1)[0].strip()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<h2>⚔️ Multi-Stock Battle Mode</h2>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
    'Compare up to 10 stocks head-to-head · Full pipeline per ticker · Ranks by battle probability, quality and risk-adjusted strength</div>',
    unsafe_allow_html=True,
)

if not _BATTLE_OK:
    st.warning("⚠️ battle_mode_engine.py not found. Place it in the same folder as app.py and restart.")
else:
    # ── 10 individual input boxes arranged in two columns of 5 ────────
    st.markdown(
        '<div style="font-size:13px;color:#7a9ab8;margin-bottom:10px;">'
        'Enter up to 10 stock tickers (e.g. RELIANCE, TCS). Empty boxes are ignored.</div>',
        unsafe_allow_html=True,
    )
    configure_nse_stock_search(None)

    _col_a, _col_b = st.columns(2)

    with _col_a:
        _t1  = stock_search_widget("Stock 1",  "battle_s1",  "Type symbol or company name...")
        _t2  = stock_search_widget("Stock 2",  "battle_s2",  "Type symbol or company name...")
        _t3  = stock_search_widget("Stock 3",  "battle_s3",  "Type symbol or company name...")
        _t4  = stock_search_widget("Stock 4",  "battle_s4",  "Type symbol or company name...")
        _t5  = stock_search_widget("Stock 5",  "battle_s5",  "Type symbol or company name...")

    with _col_b:
        _t6  = stock_search_widget("Stock 6",  "battle_s6",  "Type symbol or company name...")
        _t7  = stock_search_widget("Stock 7",  "battle_s7",  "Type symbol or company name...")
        _t8  = stock_search_widget("Stock 8",  "battle_s8",  "Type symbol or company name...")
        _t9  = stock_search_widget("Stock 9",  "battle_s9",  "Type symbol or company name...")
        _t10 = stock_search_widget("Stock 10", "battle_s10", "Type symbol or company name...")

    _battle_clicked = st.button("⚔️ Run Battle Analysis", key="battle_btn", use_container_width=False)

    if _battle_clicked:
        # ── Collect all non-empty inputs from the 10 boxes ───────────
        _all_inputs = [_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10]
        raw_tickers = [t.strip() for t in _all_inputs if t and t.strip()]

        if not raw_tickers:
            st.warning("Please enter at least 1 stock.")
        else:
            with st.spinner(f"⚔️ Running full pipeline for {len(raw_tickers)} stock(s)…"):
                try:
                    # ── Step 1: build raw rows (battle_mode_engine, no mode filter) ──
                    _battle_raw = run_battle_mode(raw_tickers, stored_mode)

                    if not _battle_raw:
                        st.error("No valid data found for the tickers entered. Check symbols and try again.")
                    else:
                        # ── Step 2: enhance_results (existing function, untouched) ──
                        _battle_df = enhance_results(_battle_raw, stored_mode)

                        # ── Step 3: universal grading (existing function, untouched) ──
                        try:
                            _mb = st.session_state.get("market_bias_result", None)
                            _battle_df = apply_universal_grading(_battle_df, _mb)
                        except Exception:
                            pass

                        # ── Step 4: enhanced logic (existing function, untouched) ──
                        try:
                            _battle_df = apply_enhanced_logic(_battle_df)
                        except Exception:
                            pass

                        # ── Step 5: phase4 logic (existing functions, untouched) ──
                        try:
                            _mb2 = st.session_state.get("market_bias_result", None)
                            _battle_df = apply_phase4_logic(_battle_df, _mb2)
                            _battle_df = apply_phase42_logic(_battle_df)
                        except Exception:
                            pass

                        # ── Step 6: compute Battle Score (new columns only) ──────
                        _battle_df = compute_battle_scores(_battle_df)

                        if _battle_df.empty:
                            st.warning("Pipeline returned no results. Try different tickers.")
                        else:
                            # ── 🥇 Winner Card ────────────────────────────────────
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="section-lbl">🥇 Battle Winner</div>', unsafe_allow_html=True)

                            _winner = _battle_df.iloc[0]
                            _w_sym    = _winner.get("Symbol", "—")
                            _w_score  = _winner.get("Final Score", 0)
                            _w_conf   = _winner.get("Confidence", 50)
                            _w_signal = _winner.get("Signal", _winner.get("Final Signal", "—"))
                            _w_setup  = _winner.get("Setup Type", _winner.get("Volume Trend", "—"))
                            _w_bat    = _winner.get("Battle Score", 0)
                            _w_prob   = _winner.get("Battle Probability", _w_bat)
                            _w_bconf  = _winner.get("Battle Confidence", _w_conf)
                            _w_bqual  = _winner.get("Battle Quality", _w_score)
                            _w_verdict = _winner.get("Battle Verdict", "BETTER PICK")
                            _w_edge   = _winner.get("Battle Edge", 0)
                            _w_notes  = _winner.get("Battle Notes", "")
                            _w_grade  = _winner.get("Grade", "—")
                            _w_color  = "#00d4a8" if _w_bat >= 65 else ("#f0b429" if _w_bat >= 45 else "#ff4d6d")

                            st.markdown(
                                f'<div style="background:#0b1017;border:2px solid {_w_color};border-radius:16px;padding:24px 28px;">'
                                f'<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
                                f'<div style="font-family:\'Syne\',sans-serif;font-size:42px;font-weight:800;color:{_w_color};line-height:1;">🥇</div>'
                                f'<div>'
                                f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;color:#ccd9e8;">{_w_sym}</div>'
                                f'<div style="font-size:12px;color:#4a6480;margin-top:4px;">Battle Winner · Grade: <b style="color:{_w_color}">{_w_grade}</b></div>'
                                f'</div>'
                                f'<div style="margin-left:auto;text-align:right;">'
                                f'<div style="font-family:\'Syne\',sans-serif;font-size:32px;font-weight:800;color:{_w_color};">{_w_bat:.1f}</div>'
                                f'<div style="font-size:11px;color:#4a6480;">Battle Score</div>'
                                f'</div>'
                                f'</div>'
                                f'<div style="display:flex;gap:32px;margin-top:18px;flex-wrap:wrap;">'
                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Final Score</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_score:.1f}</div></div>'
                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Battle Probability</div>'
                                f'<div style="font-size:18px;font-weight:700;color:{_w_color};">{_w_prob:.0f}%</div></div>'
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
                                f'<div style="font-size:18px;font-weight:700;color:{_w_color};">{_w_verdict}</div></div>'
                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Lead Margin</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_edge:.1f}</div></div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if _w_notes:
                                st.caption(f"Winner notes: {_w_notes}")

                            # ── 📊 Comparison Table ───────────────────────────────
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="section-lbl">📊 Head-to-Head Comparison</div>', unsafe_allow_html=True)

                            _table_rows = []
                            for _, _br in _battle_df.iterrows():
                                _sym      = _br.get("Symbol", "—")
                                _rank     = int(_br.get("Battle Rank", 0))
                                _bat_sc   = _br.get("Battle Score", 0)
                                _sig      = _br.get("Signal", _br.get("Final Signal", "—"))
                                _risk_sc  = _br.get("Risk Score", 50)
                                _ml_pct   = _br.get("ML %", 50)
                                _trap_r   = str(_br.get("Trap Risk", "")).strip()
                                _trap_w   = str(_br.get("Trap", "")).strip()
                                _trap_flag = "⚠️ Potential Bull Trap" if (_trap_r == "HIGH" or "Bull Trap" in _trap_w) else "✅ Clean"
                                _grade    = _br.get("Grade", "—")
                                _table_rows.append({
                                    "Rank":          _rank,
                                    "Stock":         _sym,
                                    "Verdict":       _br.get("Battle Verdict", "WATCHLIST"),
                                    "Battle Score":  round(float(_bat_sc), 1),
                                    "Probability %": round(float(_br.get("Battle Probability", _bat_sc)), 1),
                                    "Compare Conf %": round(float(_br.get("Battle Confidence", _br.get("Confidence", 50))), 1),
                                    "Quality":       round(float(_br.get("Battle Quality", _br.get("Final Score", 0))), 1),
                                    "Signal":        _sig,
                                    "Grade":         _grade,
                                    "Risk Score":    round(float(_risk_sc), 1),
                                    "Edge":          round(float(_br.get("Battle Edge", 0)), 1),
                                    "⚠️ Trap Check": _trap_flag,
                                    "Notes":         _br.get("Battle Notes", ""),
                                })

                            _cmp_df = pd.DataFrame(_table_rows)

                            st.dataframe(
                                _cmp_df,
                                column_config={
                                    "Rank":          st.column_config.NumberColumn("Rank", format="%d"),
                                    "Stock":         st.column_config.TextColumn("Stock"),
                                    "Verdict":       st.column_config.TextColumn("Verdict"),
                                    "Battle Score":  st.column_config.NumberColumn("Battle Score", format="%.1f"),
                                    "Probability %": st.column_config.NumberColumn("Probability %", format="%.1f%%"),
                                    "Compare Conf %": st.column_config.NumberColumn("Compare Conf %", format="%.1f%%"),
                                    "Quality":       st.column_config.NumberColumn("Quality", format="%.1f"),
                                    "Signal":        st.column_config.TextColumn("Signal"),
                                    "Grade":         st.column_config.TextColumn("Grade"),
                                    "Risk Score":    st.column_config.NumberColumn("Risk Score", format="%.1f"),
                                    "Edge":          st.column_config.NumberColumn("Edge", format="%.1f"),
                                    "⚠️ Trap Check": st.column_config.TextColumn("⚠️ Trap Check"),
                                    "Notes":         st.column_config.TextColumn("Notes", width="large"),
                                },
                                use_container_width=True,
                                hide_index=True,
                            )
                            with st.expander("🧾 Full Battle Diagnostics", expanded=False):
                                st.dataframe(_battle_df, use_container_width=True, hide_index=True)

                            # ── ⚠️ Trap Warnings ─────────────────────────────────
                            _trap_stocks = [
                                str(_r.get("Symbol", "?"))
                                for _, _r in _battle_df.iterrows()
                                if (str(_r.get("Trap Risk", "")).strip() == "HIGH"
                                    or "Bull Trap" in str(_r.get("Trap", "")))
                            ]
                            if _trap_stocks:
                                st.warning(
                                    f"⚠️ **Potential Bull Trap detected** in: {', '.join(_trap_stocks)}  —  "
                                    "RSI overbought and/or volume declining. Proceed with caution."
                                )

                except Exception as _battle_err:
                    st.error(f"Battle Mode encountered an error: {_battle_err}. Please check your tickers and try again.")
