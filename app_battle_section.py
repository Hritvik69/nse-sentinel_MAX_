# ═══════════════════════════════════════════════════════════════════════
# ⚔️ MULTI-STOCK BATTLE MODE (10-Box Individual Input UI) — UPGRADED
# ═══════════════════════════════════════════════════════════════════════
# HOW TO ADD: paste this entire block into app.py just ABOVE the line:
#   st.markdown("<hr>", unsafe_allow_html=True)
# that appears near the very bottom of app.py (after the CSV scan section).
#
# ✅ FIX NOTES vs original:
#   1. Removed stub function definitions — they were overriding real
#      enhance_results / grading functions already in app.py causing the
#      pipeline to always return empty and show "no results".
#   2. stored_mode is now read inside the button handler so mode changes
#      are picked up correctly.
#   3. Added Plotly bar chart for visual side-by-side comparison.
#   4. Added Price + Day Change % column to comparison table.
#   5. Better winner card layout with progress bars.
#   6. Added CSV export for battle results.
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from strategy_engines.nse_autocomplete import (
    configure_nse_stock_search,
    render_nse_stock_input,
)

try:
    from battle_mode_engine import run_battle_mode, compute_battle_scores
    _BATTLE_OK = True
except ImportError:
    _BATTLE_OK = False

# ── Helper: fetch live price + day change via yfinance ─────────────────
def _fetch_price_change(symbols: list[str]) -> dict:
    """Returns {symbol: (price, pct_change)} for each symbol."""
    result = {}
    try:
        import yfinance as yf
        tickers_ns = [s.upper() + ".NS" for s in symbols]
        data = yf.download(tickers_ns, period="2d", interval="1d",
                           group_by="ticker", progress=False, auto_adjust=True)
        for sym, ns in zip(symbols, tickers_ns):
            try:
                if len(tickers_ns) == 1:
                    closes = data["Close"].dropna().values
                else:
                    closes = data[ns]["Close"].dropna().values
                if len(closes) >= 2:
                    pct = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2)
                    result[sym.upper()] = (round(float(closes[-1]), 2), pct)
                elif len(closes) == 1:
                    result[sym.upper()] = (round(float(closes[-1]), 2), 0.0)
            except Exception:
                result[sym.upper()] = (None, None)
    except Exception:
        pass
    return result


# ── Helper: color for a numeric value ──────────────────────────────────
def _score_color(val: float) -> str:
    if val >= 65:
        return "#00d4a8"
    elif val >= 45:
        return "#f0b429"
    return "#ff4d6d"


# ── Helper: simple HTML progress bar ───────────────────────────────────
def _prog_bar(val: float, max_val: float = 100, color: str = "#00d4a8") -> str:
    pct = min(max(val / max_val * 100, 0), 100)
    return (
        f'<div style="background:#1a2535;border-radius:4px;height:6px;width:100%;margin-top:4px;">'
        f'<div style="background:{color};width:{pct:.0f}%;height:6px;border-radius:4px;"></div>'
        f'</div>'
    )


# ── Helper: Build Plotly comparison bar chart ───────────────────────────
def _build_battle_chart(battle_df: pd.DataFrame) -> go.Figure:
    symbols   = battle_df["Symbol"].tolist()
    bat_scores = battle_df["Battle Score"].tolist()
    probs      = battle_df.get("Battle Probability", battle_df["Battle Score"]).tolist()
    qualities  = battle_df.get("Battle Quality", battle_df.get("Final Score", [0]*len(symbols))).tolist()

    colors = [_score_color(s) for s in bat_scores]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Battle Score",
        x=symbols, y=bat_scores,
        marker_color=colors,
        text=[f"{v:.1f}" for v in bat_scores],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Probability %",
        x=symbols, y=probs,
        marker_color="rgba(0,148,255,0.7)",
        text=[f"{v:.0f}%" for v in probs],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Quality",
        x=symbols, y=qualities,
        marker_color="rgba(140,240,140,0.7)",
        text=[f"{v:.1f}" for v in qualities],
        textposition="outside",
    ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="#0b1017",
        paper_bgcolor="#0b1017",
        font=dict(color="#ccd9e8", size=12),
        legend=dict(bgcolor="#0b1017", bordercolor="#1e2d3d", borderwidth=1),
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(gridcolor="#1e2d3d", range=[0, 105]),
        xaxis=dict(gridcolor="#1e2d3d"),
        height=360,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
# BATTLE MODE UI
# ════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<h2>⚔️ Multi-Stock Battle Mode</h2>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
    'Compare up to 10 stocks head-to-head · Full pipeline per ticker · '
    'Ranks by battle probability, quality and risk-adjusted strength</div>',
    unsafe_allow_html=True,
)

if not _BATTLE_OK:
    st.warning(
        "⚠️ battle_mode_engine.py not found. "
        "Place it in the same folder as app.py and restart."
    )
else:
    # ── 10 individual input boxes arranged in two columns of 5 ──────────
    st.markdown(
        '<div style="font-size:13px;color:#7a9ab8;margin-bottom:10px;">'
        'Enter up to 10 NSE tickers (e.g. RELIANCE, TCS). Empty boxes are ignored.</div>',
        unsafe_allow_html=True,
    )
    configure_nse_stock_search(None)

    _col_a, _col_b = st.columns(2)
    with _col_a:
        _t1  = render_nse_stock_input("Stock 1",  key="bm_t1",  placeholder="e.g. RELIANCE")
        _t2  = render_nse_stock_input("Stock 2",  key="bm_t2",  placeholder="e.g. TCS")
        _t3  = render_nse_stock_input("Stock 3",  key="bm_t3",  placeholder="e.g. INFY")
        _t4  = render_nse_stock_input("Stock 4",  key="bm_t4",  placeholder="e.g. HDFCBANK")
        _t5  = render_nse_stock_input("Stock 5",  key="bm_t5",  placeholder="e.g. SBIN")
    with _col_b:
        _t6  = render_nse_stock_input("Stock 6",  key="bm_t6",  placeholder="e.g. ICICIBANK")
        _t7  = render_nse_stock_input("Stock 7",  key="bm_t7",  placeholder="e.g. AXISBANK")
        _t8  = render_nse_stock_input("Stock 8",  key="bm_t8",  placeholder="e.g. BAJFINANCE")
        _t9  = render_nse_stock_input("Stock 9",  key="bm_t9",  placeholder="e.g. TATAMOTORS")
        _t10 = render_nse_stock_input("Stock 10", key="bm_t10", placeholder="e.g. MARUTI")

    _battle_clicked = st.button(
        "⚔️ Run Battle Analysis", key="battle_btn", width="content"
    )

    if _battle_clicked:
        # ── Read mode INSIDE the handler (fix: was read at render time) ──
        _stored_mode = st.session_state.get("mode", 2)

        _all_inputs = [_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10]
        raw_tickers = [t.strip().upper() for t in _all_inputs if t and t.strip()]

        if not raw_tickers:
            st.warning("Please enter at least 1 stock.")
        else:
            with st.spinner(f"⚔️ Running full pipeline for {len(raw_tickers)} stock(s)…"):
                try:
                    # ── Step 1: raw battle data ──────────────────────────
                    _battle_raw = run_battle_mode(raw_tickers, _stored_mode)

                    if not _battle_raw:
                        st.error(
                            "No valid data found for the entered tickers. "
                            "Check symbols and try again."
                        )
                    else:
                        # ── Steps 2-5: enhancement pipeline ─────────────
                        # NOTE: enhance_results, apply_universal_grading etc.
                        # are defined earlier in app.py and used directly here.
                        # Do NOT redefine them in this block.
                        _battle_df = enhance_results(_battle_raw, _stored_mode)

                        try:
                            _mb = st.session_state.get("market_bias_result", None)
                            _battle_df = apply_universal_grading(_battle_df, _mb)
                        except Exception:
                            pass

                        try:
                            _battle_df = apply_enhanced_logic(_battle_df)
                        except Exception:
                            pass

                        try:
                            _mb2 = st.session_state.get("market_bias_result", None)
                            _battle_df = apply_phase4_logic(_battle_df, _mb2)
                            _battle_df = apply_phase42_logic(_battle_df)
                        except Exception:
                            pass

                        # ── Step 6: Battle Score ─────────────────────────
                        _battle_df = compute_battle_scores(_battle_df)

                        if _battle_df.empty:
                            st.warning(
                                "Pipeline returned no results. "
                                "Try different tickers or check battle_mode_engine."
                            )
                        else:
                            # ── Fetch live price + % change ──────────────
                            _price_map = _fetch_price_change(raw_tickers)

                            # ════════════════════════════════════════════
                            # 🥇 WINNER CARD
                            # ════════════════════════════════════════════
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                '<div class="section-lbl">🥇 Battle Winner</div>',
                                unsafe_allow_html=True,
                            )

                            _winner   = _battle_df.iloc[0]
                            _w_sym    = _winner.get("Symbol", "—")
                            _w_score  = float(_winner.get("Final Score", 0))
                            _w_conf   = float(_winner.get("Confidence", 50))
                            _w_signal = _winner.get("Signal", _winner.get("Final Signal", "—"))
                            _w_setup  = _winner.get("Setup Type", _winner.get("Volume Trend", "—"))
                            _w_bat    = float(_winner.get("Battle Score", 0))
                            _w_prob   = float(_winner.get("Battle Probability", _w_bat))
                            _w_bconf  = float(_winner.get("Battle Confidence", _w_conf))
                            _w_bqual  = float(_winner.get("Battle Quality", _w_score))
                            _w_verdict= _winner.get("Battle Verdict", "BETTER PICK")
                            _w_edge   = float(_winner.get("Battle Edge", 0))
                            _w_notes  = _winner.get("Battle Notes", "")
                            _w_grade  = _winner.get("Grade", "—")
                            _w_color  = _score_color(_w_bat)

                            _w_price_info = _price_map.get(_w_sym.upper(), (None, None))
                            _w_price_str  = (
                                f"₹{_w_price_info[0]:,.2f}" if _w_price_info[0] else "—"
                            )
                            _w_chg        = _w_price_info[1]
                            _w_chg_color  = "#00d4a8" if (_w_chg or 0) >= 0 else "#ff4d6d"
                            _w_chg_str    = (
                                f"{'+' if _w_chg >= 0 else ''}{_w_chg:.2f}%"
                                if _w_chg is not None else "—"
                            )

                            st.markdown(
                                f'<div style="background:#0b1017;border:2px solid {_w_color};'
                                f'border-radius:16px;padding:24px 28px;">'

                                # Top row: trophy + name + price + score
                                f'<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
                                f'<div style="font-size:42px;line-height:1;">🥇</div>'
                                f'<div>'
                                f'<div style="font-size:26px;font-weight:800;color:#ccd9e8;">{_w_sym}</div>'
                                f'<div style="font-size:12px;color:#4a6480;margin-top:2px;">'
                                f'Battle Winner · Grade: <b style="color:{_w_color}">{_w_grade}</b></div>'
                                f'</div>'

                                # Price block
                                f'<div style="margin-left:16px;">'
                                f'<div style="font-size:22px;font-weight:700;color:#ccd9e8;">{_w_price_str}</div>'
                                f'<div style="font-size:14px;font-weight:600;color:{_w_chg_color};">{_w_chg_str}</div>'
                                f'</div>'

                                # Battle score (right-aligned)
                                f'<div style="margin-left:auto;text-align:right;">'
                                f'<div style="font-size:38px;font-weight:800;color:{_w_color};">{_w_bat:.1f}</div>'
                                f'<div style="font-size:11px;color:#4a6480;">Battle Score</div>'
                                f'{_prog_bar(_w_bat, 100, _w_color)}'
                                f'</div>'
                                f'</div>'

                                # Metrics row
                                f'<div style="display:flex;gap:24px;margin-top:20px;flex-wrap:wrap;">'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Final Score</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_score:.1f}</div>'
                                f'{_prog_bar(_w_score)}</div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Battle Probability</div>'
                                f'<div style="font-size:18px;font-weight:700;color:{_w_color};">{_w_prob:.0f}%</div>'
                                f'{_prog_bar(_w_prob, 100, _w_color)}</div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Confidence</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#0094ff;">{_w_conf:.0f}%</div>'
                                f'{_prog_bar(_w_conf, 100, "#0094ff")}</div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Cmp Confidence</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#7fd1ff;">{_w_bconf:.0f}%</div>'
                                f'{_prog_bar(_w_bconf, 100, "#7fd1ff")}</div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Battle Quality</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#8cf08c;">{_w_bqual:.1f}</div>'
                                f'{_prog_bar(_w_bqual, 100, "#8cf08c")}</div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Signal</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#f0b429;">{_w_signal}</div></div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Setup</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#b08cff;">{_w_setup}</div></div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Verdict</div>'
                                f'<div style="font-size:18px;font-weight:700;color:{_w_color};">{_w_verdict}</div></div>'

                                f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;'
                                f'letter-spacing:1px;">Lead Margin</div>'
                                f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_edge:.1f}</div></div>'

                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            if _w_notes:
                                st.caption(f"📝 Winner notes: {_w_notes}")

                            # ════════════════════════════════════════════
                            # 📊 VISUAL COMPARISON CHART (NEW)
                            # ════════════════════════════════════════════
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(
                                '<div class="section-lbl">📊 Visual Comparison</div>',
                                unsafe_allow_html=True,
                            )
                            _fig = _build_battle_chart(_battle_df)
                            st.plotly_chart(_fig, width="stretch")

                            # ════════════════════════════════════════════
                            # 📋 HEAD-TO-HEAD COMPARISON TABLE
                            # ════════════════════════════════════════════
                            st.markdown(
                                '<div class="section-lbl">📋 Head-to-Head Comparison</div>',
                                unsafe_allow_html=True,
                            )

                            _table_rows = []
                            for _, _br in _battle_df.iterrows():
                                _sym     = _br.get("Symbol", "—")
                                _rank    = int(_br.get("Battle Rank", 0))
                                _bat_sc  = float(_br.get("Battle Score", 0))
                                _sig     = _br.get("Signal", _br.get("Final Signal", "—"))
                                _risk_sc = float(_br.get("Risk Score", 50))
                                _ml_pct  = float(_br.get("ML %", 50))
                                _trap_r  = str(_br.get("Trap Risk", "")).strip()
                                _trap_w  = str(_br.get("Trap", "")).strip()
                                _trap_flag = (
                                    "⚠️ Potential Bull Trap"
                                    if (_trap_r == "HIGH" or "Bull Trap" in _trap_w)
                                    else "✅ Clean"
                                )
                                _grade   = _br.get("Grade", "—")

                                # Price + change
                                _pi = _price_map.get(_sym.upper(), (None, None))
                                _px_str  = f"₹{_pi[0]:,.2f}" if _pi[0] else "—"
                                _chg_str = (
                                    f"{'+' if _pi[1] >= 0 else ''}{_pi[1]:.2f}%"
                                    if _pi[1] is not None else "—"
                                )

                                _table_rows.append({
                                    "Rank":           _rank,
                                    "Stock":          _sym,
                                    "Price":          _px_str,
                                    "Day Chg %":      _chg_str,
                                    "Verdict":        _br.get("Battle Verdict", "WATCHLIST"),
                                    "Battle Score":   round(_bat_sc, 1),
                                    "Probability %":  round(float(_br.get("Battle Probability", _bat_sc)), 1),
                                    "Cmp Conf %":     round(float(_br.get("Battle Confidence", _br.get("Confidence", 50))), 1),
                                    "Quality":        round(float(_br.get("Battle Quality", _br.get("Final Score", 0))), 1),
                                    "Signal":         _sig,
                                    "Grade":          _grade,
                                    "Risk Score":     round(_risk_sc, 1),
                                    "ML %":           round(_ml_pct, 1),
                                    "Edge":           round(float(_br.get("Battle Edge", 0)), 1),
                                    "⚠️ Trap Check":  _trap_flag,
                                    "Notes":          _br.get("Battle Notes", ""),
                                })

                            _cmp_df = pd.DataFrame(_table_rows)

                            st.dataframe(
                                _cmp_df,
                                column_config={
                                    "Rank":          st.column_config.NumberColumn("Rank", format="%d"),
                                    "Stock":         st.column_config.TextColumn("Stock"),
                                    "Price":         st.column_config.TextColumn("Price (LTP)"),
                                    "Day Chg %":     st.column_config.TextColumn("Day Chg %"),
                                    "Verdict":       st.column_config.TextColumn("Verdict"),
                                    "Battle Score":  st.column_config.NumberColumn("Battle Score",  format="%.1f"),
                                    "Probability %": st.column_config.NumberColumn("Probability %", format="%.1f%%"),
                                    "Cmp Conf %":    st.column_config.NumberColumn("Cmp Conf %",    format="%.1f%%"),
                                    "Quality":       st.column_config.NumberColumn("Quality",        format="%.1f"),
                                    "Signal":        st.column_config.TextColumn("Signal"),
                                    "Grade":         st.column_config.TextColumn("Grade"),
                                    "Risk Score":    st.column_config.NumberColumn("Risk Score",     format="%.1f"),
                                    "ML %":          st.column_config.NumberColumn("ML %",           format="%.1f%%"),
                                    "Edge":          st.column_config.NumberColumn("Edge",           format="%.1f"),
                                    "⚠️ Trap Check": st.column_config.TextColumn("⚠️ Trap Check"),
                                    "Notes":         st.column_config.TextColumn("Notes", width="large"),
                                },
                                width="stretch",
                                hide_index=True,
                            )

                            # ── CSV Export ────────────────────────────────
                            _csv_bytes = _cmp_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Battle Results (CSV)",
                                data=_csv_bytes,
                                file_name="battle_results.csv",
                                mime="text/csv",
                                key="battle_csv_dl",
                            )

                            # ── Full diagnostics expander ─────────────────
                            with st.expander("🧾 Full Battle Diagnostics", expanded=False):
                                st.dataframe(_battle_df, width="stretch", hide_index=True)

                            # ════════════════════════════════════════════
                            # ⚠️ TRAP WARNINGS
                            # ════════════════════════════════════════════
                            _trap_stocks = [
                                str(_r.get("Symbol", "?"))
                                for _, _r in _battle_df.iterrows()
                                if (
                                    str(_r.get("Trap Risk", "")).strip() == "HIGH"
                                    or "Bull Trap" in str(_r.get("Trap", ""))
                                )
                            ]
                            if _trap_stocks:
                                st.warning(
                                    f"⚠️ **Potential Bull Trap detected** in: "
                                    f"{', '.join(_trap_stocks)} — "
                                    "RSI overbought and/or volume declining. "
                                    "Proceed with caution."
                                )

                except Exception as _battle_err:
                    st.error(
                        f"Battle Mode encountered an error: {_battle_err}. "
                        "Please check your tickers and try again."
                    )
