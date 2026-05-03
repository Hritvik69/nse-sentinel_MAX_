"""
app_sector_prediction_section.py
══════════════════════════════════
Streamlit UI for the Sector Prediction System.

Integration
───────────
Add to app.py after your scan results are ready:

    from app_sector_prediction_section import render_sector_prediction_section
    render_sector_prediction_section(
        scan_df  = st.session_state.get("last_scan_df"),
        all_data = ALL_DATA,            # from strategy_engines._engine_utils
    )

Two sections are rendered:
  1. Sector Prediction Detail  — select a sector, see chart + prediction
  2. 📊 Model Performance Dashboard — full evaluation metrics
"""

from __future__ import annotations

import math
import pandas as pd
import streamlit as st

# ── Engine imports (graceful) ─────────────────────────────────────────
try:
    from sector_prediction_engine import predict_sector, SectorPrediction
    _PE_OK = True
except ImportError as _e:
    _PE_OK = False
    _PE_ERR = str(_e)

try:
    from sector_chart_engine import build_sector_chart
    _CE_OK = True
except ImportError:
    _CE_OK = False

try:
    from sector_prediction_tracker import log_prediction, backfill_outcomes, recent_predictions
    _TR_OK = True
except ImportError:
    _TR_OK = False
    def log_prediction(p): return False
    def backfill_outcomes(d): return 0
    def recent_predictions(s, n=5): return pd.DataFrame()

try:
    from sector_evaluation_engine import compute_full_evaluation, compute_sector_report
    _EV_OK = True
except ImportError:
    _EV_OK = False

try:
    from sector_master import get_all_sectors, SECTOR_DESCRIPTIONS
    _SM_OK = True
except ImportError:
    _SM_OK = False
    def get_all_sectors(): return []
    SECTOR_DESCRIPTIONS = {}


# ══════════════════════════════════════════════════════════════════════
# STYLE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _dir_color(direction: str) -> str:
    return {"Bullish": "#00d4a8", "Bearish": "#ff4d6d", "Sideways": "#8ab4d8"}.get(direction, "#8ab4d8")


def _dir_icon(direction: str) -> str:
    return {"Bullish": "▲", "Bearish": "▼", "Sideways": "◆"}.get(direction, "●")


def _conf_bar_html(value: float, color: str = "#f0b429") -> str:
    pct = int(max(0, min(100, value)))
    return (
        f'<div style="background:#1a2035;border-radius:4px;height:8px;width:100%;">'
        f'<div style="background:{color};height:8px;border-radius:4px;width:{pct}%;"></div>'
        f'</div>'
    )


def _signal_row_html(label: str, val: float, max_val: float = 100) -> str:
    pct = int(max(0, min(100, val / max_val * 100)))
    if val >= 65:
        bar_color = "#00d4a8"
    elif val >= 45:
        bar_color = "#f0b429"
    else:
        bar_color = "#ff4d6d"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
        f'<span style="color:#8ab4d8;font-size:11px;width:140px;flex-shrink:0;">{label}</span>'
        f'<div style="background:#1a2035;border-radius:3px;height:6px;flex:1;">'
        f'<div style="background:{bar_color};height:6px;border-radius:3px;width:{pct}%;"></div>'
        f'</div>'
        f'<span style="color:{bar_color};font-size:11px;width:36px;text-align:right;">{val:.0f}</span>'
        f'</div>'
    )


def _metric_card_html(label: str, value: str, color: str = "#f0b429", sub: str = "") -> str:
    return (
        f'<div style="background:#0d1626;border:1px solid #1e3a5f;border-radius:8px;'
        f'padding:14px 16px;text-align:center;">'
        f'<div style="color:#8ab4d8;font-size:10px;text-transform:uppercase;letter-spacing:1px;">{label}</div>'
        f'<div style="color:{color};font-size:22px;font-weight:700;margin:4px 0;">{value}</div>'
        f'{"" if not sub else f"<div style=color:#6a8aad;font-size:10px;>{sub}</div>"}'
        f'</div>'
    )


def _pretty_name(value: str) -> str:
    return str(value or "").replace("_", " ").strip().title()


def _sector_label(sector: str) -> str:
    labels = {
        "OVERALL": "Overall",
        "NIFTY_50": "Nifty 50",
        "NIFTY_150": "Nifty 150",
        "NIFTY_300": "Nifty 300",
        "BANKING": "Bank",
        "NBFC_FINANCE": "NBFC",
        "IT": "IT",
        "AUTO": "Auto",
        "FMCG": "FMCG",
        "PHARMA": "Pharma",
        "INFRA": "Infra",
    }
    return labels.get(sector, _pretty_name(sector))


def _prediction_source_text(pred) -> str:
    source = str(getattr(pred, "ohlc_source", "") or "")
    symbol = str(getattr(pred, "ohlc_symbol", "") or "")
    bars = int(getattr(pred, "ohlc_bars", 0) or 0)
    used = len(getattr(pred, "stocks_used", []) or [])
    if source == "real_sector_index":
        base = f"Real sector index{'' if not symbol else f' ({symbol})'}"
    elif source == "weighted_sector_basket":
        base = f"Weighted basket · {used} stocks"
    elif source == "leader_stock_fallback":
        base = f"Leader fallback{'' if not symbol else f' · {symbol}'}"
    else:
        base = f"{used} stocks" if used else "Sector data"
    if bars > 0:
        base += f" · {bars} daily candles"
    return f"{base} · {str(getattr(pred, 'predicted_at', ''))[:10]}".strip(" ·")


def _render_sector_picker(sectors: list[str]) -> str:
    selected = st.session_state.get("sector_pred_selected", sectors[0])
    if selected not in sectors:
        selected = sectors[0]

    priority = [
        sector for sector in
        ["OVERALL", "NIFTY_50", "NIFTY_150", "NIFTY_300", "BANKING", "IT", "AUTO", "FMCG", "PHARMA", "INFRA"]
        if sector in sectors
    ]

    if priority:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">Quick Sector Grid</div>',
            unsafe_allow_html=True,
        )
        for start in range(0, len(priority), 5):
            row = priority[start:start + 5]
            cols = st.columns(len(row))
            for col, sector in zip(cols, row):
                with col:
                    if st.button(
                        _sector_label(sector),
                        key=f"sector_card_{sector}",
                        use_container_width=True,
                        type="primary" if selected == sector else "secondary",
                    ):
                        selected = sector
                        st.session_state["sector_pred_selected"] = sector
                    st.caption(SECTOR_DESCRIPTIONS.get(sector, sector))

    sector_labels = {s: f"{_sector_label(s)}  —  {SECTOR_DESCRIPTIONS.get(s, s)}" for s in sectors}
    current = st.session_state.get("sector_pred_selected", selected)
    if current not in sectors:
        current = sectors[0]
    label_options = list(sector_labels.values())
    current_label = sector_labels[current]
    chosen_label = st.selectbox(
        "Choose a sector to analyse",
        options=label_options,
        index=label_options.index(current_label),
        key="sector_pred_selector",
    )
    selected = next((s for s, label in sector_labels.items() if label == chosen_label), current)
    st.session_state["sector_pred_selected"] = selected
    return selected


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — SECTOR PREDICTION DETAIL
# ══════════════════════════════════════════════════════════════════════

def _render_prediction_detail(
    sector: str,
    scan_df: pd.DataFrame | None,
    all_data: dict,
) -> None:
    if not _PE_OK:
        st.error(f"sector_prediction_engine could not be loaded: {_PE_ERR}")
        return

    # ── Run prediction ────────────────────────────────────────────────
    with st.spinner(f"Analysing {sector}…"):
        pred = predict_sector(sector, scan_df, all_data)

    # ── Backfill any outstanding outcomes ─────────────────────────────
    if _TR_OK:
        backfill_outcomes(all_data)

    # ── Top prediction card ───────────────────────────────────────────
    dir_col = _dir_color(pred.direction)
    icon    = _dir_icon(pred.direction)
    source_text = _prediction_source_text(pred)

    st.markdown(
        f'<div style="background:#0d1626;border:2px solid {dir_col}40;border-radius:12px;'
        f'padding:20px 24px;margin-bottom:16px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<div style="color:#8ab4d8;font-size:11px;text-transform:uppercase;letter-spacing:2px;">Sector Prediction</div>'
        f'<div style="color:{dir_col};font-size:36px;font-weight:800;letter-spacing:-1px;">'
        f'{icon} {pred.direction}</div>'
        f'<div style="color:#6a8aad;font-size:12px;margin-top:2px;">'
        f'{source_text}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="color:#8ab4d8;font-size:11px;text-transform:uppercase;">Confidence</div>'
        f'<div style="color:{dir_col};font-size:48px;font-weight:900;">{pred.confidence:.0f}<span style="font-size:22px;">%</span></div>'
        f'{_conf_bar_html(pred.confidence, dir_col)}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Note (if any) ─────────────────────────────────────────────────
    if pred.note:
        st.warning(pred.note)
        return

    # ── CHART (main element) ──────────────────────────────────────────
    if _CE_OK:
        fig = build_sector_chart(pred)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart could not be built (insufficient OHLC data).")
    else:
        st.info("Install `plotly` to enable the candlestick chart.")

    # ── Two columns: signal breakdown + sector model accuracy ─────────
    if getattr(pred, "mtf_note", ""):
        st.caption(pred.mtf_note)
    if getattr(pred, "sideways_forced", False):
        st.info("Low signal agreement forced this call to Sideways.")

    meta_1, meta_2, meta_3, meta_4 = st.columns(4)
    with meta_1:
        st.markdown(
            _metric_card_html(
                "Regime",
                _pretty_name(getattr(pred, "regime", "Range Bound")),
                "#8ab4d8",
                f"Confidence {getattr(pred, 'regime_confidence', 50.0):.0f}%",
            ),
            unsafe_allow_html=True,
        )
    with meta_2:
        st.markdown(
            _metric_card_html(
                "MTF Score",
                f"{getattr(pred, 'mtf_score', 50.0):.0f}",
                "#00d4a8" if getattr(pred, "mtf_score", 50.0) >= 60 else "#f0b429",
                "Multi-timeframe alignment",
            ),
            unsafe_allow_html=True,
        )
    with meta_3:
        st.markdown(
            _metric_card_html(
                "Agreement",
                f"{getattr(pred, 'signal_agreement', 50.0):.0f}%",
                "#00d4a8" if getattr(pred, "signal_agreement", 50.0) >= 60 else "#f0b429",
                "Cross-signal consensus",
            ),
            unsafe_allow_html=True,
        )
    with meta_4:
        st.markdown(
            _metric_card_html(
                "Confidence Cap",
                f"{getattr(pred, 'confidence_cap', 95.0):.0f}%",
                "#ff4d6d" if getattr(pred, "confidence_cap", 95.0) < 70 else "#8ab4d8",
                "Regime-adjusted ceiling",
            ),
            unsafe_allow_html=True,
        )

    col_sig, col_acc = st.columns([3, 2])

    with col_sig:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">Signal Breakdown</div>',
            unsafe_allow_html=True,
        )
        sig = pred.signals
        signal_rows = [
            ("EMA Slope",       sig.ema_slope),
            ("Price vs EMA",    sig.price_vs_ema),
            ("Candle Direction",sig.candle_direction),
            ("Body Strength",   sig.body_strength),
            ("Consecutive",     sig.consecutive),
            ("Volume Confirm",  sig.volume_confirm),
            ("Volatility",      sig.volatility),
            ("Momentum",        sig.momentum),
            ("Sector Strength", sig.sector_strength),
            ("Bullish %",       sig.bullish_pct),
            ("Money Flow",      sig.money_flow),
            ("Participation",   sig.participation),
        ]
        html = '<div style="background:#080e1c;border-radius:8px;padding:12px;">'
        for lbl, val in signal_rows:
            html += _signal_row_html(lbl, val)
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with col_acc:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">Model Accuracy (This Sector)</div>',
            unsafe_allow_html=True,
        )
        if _EV_OK:
            sr = compute_sector_report(sector)
            if sr.total > 0:
                acc_col = "#00d4a8" if sr.accuracy_pct >= 60 else ("#f0b429" if sr.accuracy_pct >= 45 else "#ff4d6d")
                st.markdown(
                    f'<div style="background:#080e1c;border-radius:8px;padding:16px;">'
                    f'<div style="color:{acc_col};font-size:42px;font-weight:800;">{sr.accuracy_pct:.0f}%</div>'
                    f'<div style="color:#6a8aad;font-size:11px;">over {sr.total} predictions '
                    f'({sr.correct} correct)</div>'
                    f'<hr style="border-color:#1e3a5f;margin:12px 0;">'
                    f'<div style="color:#8ab4d8;font-size:11px;">Avg Return: '
                    f'<span style="color:#f0b429;">{sr.avg_return:+.2f}%</span></div>'
                    f'<div style="color:#8ab4d8;font-size:11px;">Best Trade: '
                    f'<span style="color:#00d4a8;">{sr.best_return:+.2f}%</span></div>'
                    f'<div style="color:#8ab4d8;font-size:11px;">Worst Trade: '
                    f'<span style="color:#ff4d6d;">{sr.worst_return:+.2f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#080e1c;border-radius:8px;padding:16px;'
                    'color:#6a8aad;font-size:12px;">No validated predictions yet for this '
                    'sector.<br><br>Run scans daily and check back tomorrow to see accuracy '
                    'build up.</div>',
                    unsafe_allow_html=True,
                )

    # ── Log this prediction ───────────────────────────────────────────
        weights = getattr(pred, "dynamic_weights", {}) or {}
        if weights:
            top_weights = pd.DataFrame(
                [
                    {"Signal": _pretty_name(name), "Weight %": round(weight * 100, 2)}
                    for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:6]
                ]
            )
            st.markdown(
                '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
                'letter-spacing:1px;margin:16px 0 8px;">Active Weighting</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(top_weights, hide_index=True, use_container_width=True)

    if _TR_OK:
        key = f"_pred_logged_{sector}_{pred.predicted_at[:10]}"
        if not st.session_state.get(key, False):
            if log_prediction(pred):
                st.session_state[key] = True

    # ── Recent 5 predictions for this sector ─────────────────────────
    if _TR_OK:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin:20px 0 8px;">Recent Predictions — Last 5</div>',
            unsafe_allow_html=True,
        )
        hist = recent_predictions(sector, n=5)
        if hist.empty:
            st.markdown(
                '<div style="color:#6a8aad;font-size:12px;">No prediction history yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            display_cols = [c for c in (
                "predicted_at", "direction", "confidence",
                "return_pct", "correct",
            ) if c in hist.columns]
            renamed = {
                "predicted_at": "Date",
                "direction":    "Direction",
                "confidence":   "Confidence",
                "return_pct":   "Return %",
                "correct":      "Correct",
            }

            def _style_row(row):
                styles = []
                for col in row.index:
                    val = str(row[col]).strip()
                    if col == "Direction":
                        color = _dir_color(val)
                        styles.append(f"color: {color};font-weight:600;")
                    elif col == "Correct":
                        color = "#00d4a8" if val == "True" else ("#ff4d6d" if val == "False" else "#6a8aad")
                        styles.append(f"color: {color};")
                    elif col == "Return %":
                        try:
                            fval = float(val)
                            color = "#00d4a8" if fval > 0 else "#ff4d6d"
                        except Exception:
                            color = "#6a8aad"
                        styles.append(f"color: {color};")
                    else:
                        styles.append("color: #8ab4d8;")
                return styles

            disp = hist[display_cols].rename(columns=renamed)
            if "Date" in disp.columns:
                disp["Date"] = disp["Date"].str[:10]
            try:
                st.dataframe(
                    disp.style.apply(_style_row, axis=1),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(disp, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def _render_performance_dashboard() -> None:
    st.markdown(
        '<h3 style="color:#f0b429;margin-bottom:4px;">📊 Sector Model Performance</h3>',
        unsafe_allow_html=True,
    )

    if not _EV_OK:
        st.error("sector_evaluation_engine could not be loaded.")
        return

    ev = compute_full_evaluation()

    if ev.total_predictions == 0:
        st.info(
            "No predictions logged yet. Run a scan and visit the Sector Prediction "
            "section to start building history."
        )
        return

    # ── Top metrics row ───────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    acc_col = "#00d4a8" if ev.accuracy_pct >= 60 else ("#f0b429" if ev.accuracy_pct >= 45 else "#ff4d6d")
    ret_col = "#00d4a8" if ev.avg_return_pct > 0 else "#ff4d6d"
    dd_col  = "#ff4d6d" if ev.max_drawdown < -5 else "#f0b429"
    sharpe_col = "#00d4a8" if ev.sharpe_approx > 0 else "#ff4d6d"
    stability_col = "#00d4a8" if ev.wf_stability_score >= 60 else ("#f0b429" if ev.wf_stability_score >= 45 else "#ff4d6d")

    with m1:
        st.markdown(_metric_card_html("Total Predictions", str(ev.total_predictions),
                                      "#8ab4d8", f"{ev.validated} validated"), unsafe_allow_html=True)
    with m2:
        st.markdown(_metric_card_html("Accuracy", f"{ev.accuracy_pct:.1f}%",
                                      acc_col, f"{ev.correct}/{ev.validated} correct"), unsafe_allow_html=True)
    with m3:
        st.markdown(_metric_card_html("Avg Return", f"{ev.avg_return_pct:+.2f}%",
                                      ret_col, f"W/L {ev.win_loss_ratio:.1f}"), unsafe_allow_html=True)
    with m4:
        st.markdown(_metric_card_html("Cum. Return", f"{ev.cumulative_return:+.1f}%",
                                      ret_col, f"Best {ev.best_trade:+.1f}%"), unsafe_allow_html=True)
    with m5:
        st.markdown(_metric_card_html("Max Drawdown", f"{ev.max_drawdown:.1f}%",
                                      dd_col, f"σ {ev.return_volatility:.2f}%"), unsafe_allow_html=True)

    with m6:
        st.markdown(_metric_card_html("Stability", f"{ev.wf_stability_score:.1f}",
                                      stability_col, ev.wf_note or "Walk-forward"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.markdown(_metric_card_html("Sharpe", f"{ev.sharpe_approx:.2f}",
                                      sharpe_col, f"Expectancy {ev.expectancy:+.2f}%"), unsafe_allow_html=True)
    with r2:
        st.markdown(_metric_card_html("Win Streak", str(ev.max_win_streak),
                                      "#00d4a8", f"Loss streak {ev.max_loss_streak}"), unsafe_allow_html=True)
    with r3:
        st.markdown(_metric_card_html("Calibration", f"{ev.calibration_score:.1f}",
                                      "#f0b429", "Lower is better"), unsafe_allow_html=True)
    with r4:
        st.markdown(_metric_card_html("W/F Accuracy", f"{ev.wf_overall_accuracy:.1f}%",
                                      stability_col, "Walk-forward"), unsafe_allow_html=True)
    with r5:
        st.markdown(_metric_card_html("Best Trade", f"{ev.best_trade:+.1f}%",
                                      ret_col, f"Worst {ev.worst_trade:+.1f}%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-sector table + Calibration side-by-side ──────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">Per-Sector Accuracy</div>',
            unsafe_allow_html=True,
        )
        if ev.per_sector:
            rows = []
            for sr in ev.per_sector:
                rows.append({
                    "Sector":   sr.sector,
                    "Preds":    sr.total,
                    "Accuracy": f"{sr.accuracy_pct:.1f}%",
                    "Avg Ret":  f"{sr.avg_return:+.2f}%",
                    "Best":     f"{sr.best_return:+.2f}%",
                    "Worst":    f"{sr.worst_return:+.2f}%",
                })
            sdf = pd.DataFrame(rows)

            def _sector_style(row):
                styles = []
                for col in row.index:
                    val = str(row[col]).strip()
                    if col == "Accuracy":
                        try:
                            fv = float(val.replace("%", ""))
                            c = "#00d4a8" if fv >= 60 else ("#f0b429" if fv >= 45 else "#ff4d6d")
                        except Exception:
                            c = "#8ab4d8"
                        styles.append(f"color:{c};font-weight:600;")
                    elif col in ("Avg Ret", "Best", "Worst"):
                        try:
                            fv = float(val.replace("%", ""))
                            c = "#00d4a8" if fv > 0 else "#ff4d6d"
                        except Exception:
                            c = "#6a8aad"
                        styles.append(f"color:{c};")
                    else:
                        styles.append("color:#8ab4d8;")
                return styles

            try:
                st.dataframe(
                    sdf.style.apply(_sector_style, axis=1),
                    hide_index=True,
                    use_container_width=True,
                    height=min(400, 36 * len(ev.per_sector) + 38),
                )
            except Exception:
                st.dataframe(sdf, hide_index=True, use_container_width=True)

            st.markdown(
                f'<div style="font-size:11px;color:#6a8aad;margin-top:4px;">'
                f'🏆 Best: <b>{ev.best_sector}</b> &nbsp;|&nbsp; '
                f'⚠️ Worst: <b>{ev.worst_sector}</b></div>',
                unsafe_allow_html=True,
            )

    with right:
        # ── Calibration chart ─────────────────────────────────────────
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">Calibration (Confidence vs Actual)</div>',
            unsafe_allow_html=True,
        )
        if ev.calibration_buckets:
            try:
                import plotly.graph_objects as go
                bkts = ev.calibration_buckets
                labels  = [b.label for b in bkts]
                confs   = [b.avg_confidence for b in bkts]
                actuals = [b.actual_accuracy for b in bkts]

                cal_fig = go.Figure()
                cal_fig.add_trace(go.Bar(
                    x=labels, y=actuals,
                    name="Actual Accuracy",
                    marker_color="#00d4a8",
                ))
                cal_fig.add_trace(go.Scatter(
                    x=labels, y=confs,
                    name="Avg Confidence",
                    mode="lines+markers",
                    line=dict(color="#f0b429", width=2, dash="dot"),
                    marker=dict(size=7),
                ))
                cal_fig.update_layout(
                    height=260,
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0d1117",
                    font=dict(color="#8ab4d8", size=10),
                    margin=dict(l=30, r=10, t=10, b=30),
                    legend=dict(
                        orientation="h", x=0, y=1.15,
                        font=dict(size=9),
                    ),
                    yaxis=dict(range=[0, 100], ticksuffix="%"),
                )
                st.plotly_chart(cal_fig, use_container_width=True)
                st.markdown(
                    f'<div style="font-size:11px;color:#6a8aad;">Calibration MAE: '
                    f'<b style="color:#f0b429;">{ev.calibration_score:.1f}%</b> '
                    f'(lower = better calibrated)</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.info(f"Calibration chart unavailable: {exc}")
        else:
            st.markdown(
                '<div style="color:#6a8aad;font-size:12px;">Calibration data available '
                'after 10+ validated predictions per confidence bucket.</div>',
                unsafe_allow_html=True,
            )

    # ── Last 10 trades ────────────────────────────────────────────────
    if ev.wf_rolling_accuracy or (ev.signal_perf_df is not None and not ev.signal_perf_df.empty):
        st.markdown("<br>", unsafe_allow_html=True)
        wf_col, sig_col = st.columns([3, 2])

        with wf_col:
            st.markdown(
                '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
                'letter-spacing:1px;margin-bottom:8px;">Walk-Forward Stability</div>',
                unsafe_allow_html=True,
            )
            if ev.wf_rolling_accuracy:
                try:
                    import plotly.graph_objects as go

                    wf_fig = go.Figure()
                    wf_fig.add_trace(go.Scatter(
                        x=ev.wf_fold_dates or list(range(1, len(ev.wf_rolling_accuracy) + 1)),
                        y=ev.wf_rolling_accuracy,
                        name="Accuracy",
                        mode="lines+markers",
                        line=dict(color="#00d4a8", width=2),
                    ))
                    if ev.wf_rolling_returns:
                        wf_fig.add_trace(go.Bar(
                            x=ev.wf_fold_dates or list(range(1, len(ev.wf_rolling_returns) + 1)),
                            y=ev.wf_rolling_returns,
                            name="Fold Return",
                            marker_color="#f0b429",
                            opacity=0.45,
                            yaxis="y2",
                        ))
                    wf_fig.update_layout(
                        height=280,
                        paper_bgcolor="#0a0e1a",
                        plot_bgcolor="#0d1117",
                        font=dict(color="#8ab4d8", size=10),
                        margin=dict(l=30, r=30, t=10, b=30),
                        legend=dict(orientation="h", x=0, y=1.15, font=dict(size=9)),
                        yaxis=dict(title="Accuracy %", range=[0, 100]),
                        yaxis2=dict(title="Return %", overlaying="y", side="right"),
                    )
                    st.plotly_chart(wf_fig, use_container_width=True)
                except Exception as exc:
                    st.info(f"Walk-forward chart unavailable: {exc}")
            else:
                st.caption(ev.wf_note or "Walk-forward data will appear after more validated predictions.")

        with sig_col:
            st.markdown(
                '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
                'letter-spacing:1px;margin-bottom:8px;">Signal Quality</div>',
                unsafe_allow_html=True,
            )
            if ev.signal_perf_df is not None and not ev.signal_perf_df.empty:
                sig_df = ev.signal_perf_df.copy()
                if "Signal" in sig_df.columns:
                    sig_df["Signal"] = sig_df["Signal"].apply(_pretty_name)
                sig_df.columns = ["Delta Weight" if "Weight" in str(col) and "Î" in str(col) else col for col in sig_df.columns]
                st.dataframe(sig_df, hide_index=True, use_container_width=True, height=280)
            else:
                st.caption("Signal reliability will populate after validated predictions accumulate.")

    if ev.regime_perf_df is not None and not ev.regime_perf_df.empty:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin:20px 0 8px;">Regime Split</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(ev.regime_perf_df, hide_index=True, use_container_width=True)

    if not ev.last_10.empty:
        st.markdown(
            '<div style="color:#8ab4d8;font-size:12px;text-transform:uppercase;'
            'letter-spacing:1px;margin:20px 0 8px;">Last 10 Predictions</div>',
            unsafe_allow_html=True,
        )
        show_cols = [c for c in (
            "predicted_at", "sector", "direction", "regime",
            "confidence", "return_pct", "correct",
        ) if c in ev.last_10.columns]
        d10 = ev.last_10[show_cols].rename(columns={
            "predicted_at": "Date", "sector": "Sector",
            "direction": "Direction", "regime": "Regime", "confidence": "Conf",
            "return_pct": "Return %", "correct": "Correct",
        })
        if "Date" in d10.columns:
            d10["Date"] = d10["Date"].str[:10]
        if "Regime" in d10.columns:
            d10["Regime"] = d10["Regime"].apply(_pretty_name)

        def _t10_style(row):
            styles = []
            for col in row.index:
                val = str(row[col]).strip()
                if col == "Direction":
                    styles.append(f"color:{_dir_color(val)};font-weight:600;")
                elif col == "Correct":
                    c = "#00d4a8" if val == "True" else ("#ff4d6d" if val == "False" else "#6a8aad")
                    styles.append(f"color:{c};")
                elif col == "Return %":
                    try:
                        c = "#00d4a8" if float(val) > 0 else "#ff4d6d"
                    except Exception:
                        c = "#6a8aad"
                    styles.append(f"color:{c};")
                else:
                    styles.append("color:#8ab4d8;")
            return styles

        try:
            st.dataframe(
                d10.style.apply(_t10_style, axis=1),
                hide_index=True,
                use_container_width=True,
            )
        except Exception:
            st.dataframe(d10, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def render_sector_prediction_section(
    scan_df:  pd.DataFrame | None = None,
    all_data: dict | None = None,
) -> None:
    """
    Render both the Sector Prediction Detail and the Performance Dashboard.

    Parameters
    ----------
    scan_df  : pd.DataFrame | None   Latest scan output.
    all_data : dict | None           ALL_DATA from strategy_engines._engine_utils.
    """
    if all_data is None:
        try:
            from strategy_engines._engine_utils import ALL_DATA
            all_data = ALL_DATA
        except ImportError:
            all_data = {}

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-bottom:4px;">🔮 Sector Prediction Engine</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:20px;">'
        'Chart-driven sector probability · Signal breakdown · Tracked accuracy</div>',
        unsafe_allow_html=True,
    )

    if not _SM_OK:
        st.warning("sector_master.py could not be loaded — no sectors available.")
        return

    sectors = get_all_sectors()
    if not sectors:
        st.warning("No sectors configured in sector_master.py.")
        return

    # ── Tabs: Prediction | Performance ───────────────────────────────
    tab_pred, tab_perf = st.tabs(["🔮 Sector Prediction", "📊 Model Performance"])

    with tab_pred:
        chosen = _render_sector_picker(sectors)
        _render_prediction_detail(chosen, scan_df, all_data)

    with tab_perf:
        _render_performance_dashboard()
