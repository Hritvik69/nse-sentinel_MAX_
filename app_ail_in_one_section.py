from __future__ import annotations

import html
from typing import Any, Callable

import pandas as pd
import streamlit as st

from ail_in_one_engine import AIL_CATEGORY_ORDER, AILPipelineResult, run_ail_pipeline


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default


def _plain_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _row_symbol(row: pd.Series | dict[str, Any]) -> str:
    for key in ("Symbol", "Ticker", "symbol", "ticker", "Stock"):
        try:
            symbol = _plain_symbol(row.get(key))
        except Exception:
            symbol = ""
        if symbol:
            return symbol
    return ""


def _existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [col for col in cols if col in df.columns]


def _metric_card(label: str, value: str, subtext: str = "", color: str = "#00d4a8") -> str:
    return (
        f'<div style="background:#0b1017;border:1px solid #1e3a5f;border-left:4px solid {color};'
        f'border-radius:10px;padding:14px 15px;min-height:92px;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1.1px;text-transform:uppercase;">{html.escape(label)}</div>'
        f'<div style="font-size:24px;font-weight:900;color:{color};margin-top:7px;">{html.escape(value)}</div>'
        f'<div style="font-size:11px;color:#8ab4d8;margin-top:5px;line-height:1.45;">{html.escape(subtext)}</div>'
        f'</div>'
    )


def _leader_card(label: str, item: dict[str, Any], color: str) -> str:
    symbol = str(item.get("symbol", "") or "-")
    score = _safe_float(item.get("score"), 0.0)
    metric = str(item.get("metric", "") or "")
    reason = str(item.get("reason", "") or "Existing scoring signal")
    return (
        f'<div style="background:#0b1017;border:1px solid #1e3a5f;border-left:4px solid {color};'
        f'border-radius:10px;padding:14px 15px;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1.1px;text-transform:uppercase;">{html.escape(label)}</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-top:7px;">'
        f'<div style="font-size:22px;font-weight:900;color:#ccd9e8;">{html.escape(symbol)}</div>'
        f'<div style="font-size:20px;font-weight:900;color:{color};">{score:.1f}</div>'
        f'</div>'
        f'<div style="font-size:10px;color:#4a6480;margin-top:2px;">{html.escape(metric)}</div>'
        f'<div style="font-size:12px;color:#8ab4d8;line-height:1.55;margin-top:8px;">{html.escape(reason)}</div>'
        f'</div>'
    )


def _render_dataframe(df: pd.DataFrame, cols: list[str], *, height: int | None = None) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No rows available for this section.")
        return
    view_cols = _existing_cols(df, cols)
    view = df[view_cols].copy() if view_cols else df.copy()
    st.dataframe(view, width="stretch", hide_index=True, height=height)


def _top_buy_tomorrow(aura_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.Series | None:
    if isinstance(aura_df, pd.DataFrame) and not aura_df.empty:
        text = (
            aura_df.get("AI Verdict", pd.Series("", index=aura_df.index)).fillna("").astype(str)
            + " "
            + aura_df.get("Final Verdict", pd.Series("", index=aura_df.index)).fillna("").astype(str)
        ).str.upper()
        preferred = aura_df.loc[text.str.contains("BUY TOMORROW|GOOD SWING|STRONG BUY", regex=True, na=False)]
        if not preferred.empty:
            return preferred.iloc[0]
        return aura_df.iloc[0]
    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        return final_df.iloc[0]
    return None


def _render_market_scan_summary(result: AILPipelineResult) -> None:
    st.subheader("1. Market Scan Summary")
    mode_df = pd.DataFrame(result.mode_summaries)
    raw_hits = int(mode_df["Raw Hits"].sum()) if "Raw Hits" in mode_df.columns and not mode_df.empty else 0
    enhanced = int(mode_df["Enhanced Candidates"].sum()) if "Enhanced Candidates" in mode_df.columns and not mode_df.empty else 0
    preload_ready = int(result.preload_stats.get("loaded", 0) or 0) + int(result.preload_stats.get("cache_hits", 0) or 0)
    cards = [
        _metric_card("Ticker Universe", f"{result.requested_tickers:,}", "Full selected NSE universe", "#00d4a8"),
        _metric_card("Modes Scanned", f"{len(result.modes_scanned)}", "Modes 1 through 7", "#0094ff"),
        _metric_card("Raw Hits", f"{raw_hits:,}", "Signals returned by existing scanners", "#f0b429"),
        _metric_card("Ranked Pool", f"{len(result.final_ranked_df):,}", f"Completed in {result.elapsed_sec:.1f}s", "#b08cff"),
    ]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:8px 0 14px 0;">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Preload ready rows: {preload_ready:,} | Snapshot loaded: {int(result.preload_stats.get('snapshot_loaded', 0) or 0):,} | "
        f"Enhanced candidates: {enhanced:,}"
    )
    _render_dataframe(mode_df, ["Mode", "Mode Name", "Raw Hits", "Enhanced Candidates", "Elapsed Sec", "Error"])


def _render_mode_top3(result: AILPipelineResult) -> None:
    st.subheader("2. Mode-wise Top 3")
    tabs = st.tabs(list(AIL_CATEGORY_ORDER))
    top_cols = [
        "AIL Category Rank",
        "Symbol",
        "Mode Name",
        "AIL Top3 Score",
        "AIL Top3 Confidence",
        "Prediction Score",
        "Final Score",
        "Confidence",
        "Conviction Tier",
        "Trap Risk",
        "Setup Quality",
        "Entry Timing",
        "Sector",
        "AIL Top3 Drivers",
        "AIL Top3 Penalties",
    ]
    for tab, category in zip(tabs, AIL_CATEGORY_ORDER):
        with tab:
            payload = result.category_top3.get(category, {})
            top_df = payload.get("top_df") if isinstance(payload, dict) else pd.DataFrame()
            if isinstance(top_df, pd.DataFrame) and not top_df.empty:
                st.caption(
                    f"Evaluated {int(payload.get('evaluated', len(result.categories.get(category, []))) or 0):,} | "
                    f"Scored {int(payload.get('scored', len(top_df)) or 0):,} | "
                    f"Eliminated {int(payload.get('eliminated', 0) or 0):,}"
                )
            _render_dataframe(top_df, top_cols)


def _render_ranked_leaders(result: AILPipelineResult) -> None:
    st.subheader("3. AI Ranked Leaders")
    df = result.final_ranked_df
    if df is None or df.empty:
        st.info("A-I-L did not find a cross-mode ranked candidate yet.")
        return
    cards = []
    colors = ["#00d4a8", "#0094ff", "#f0b429"]
    for idx, (_, row) in enumerate(df.head(3).iterrows()):
        symbol = _row_symbol(row)
        score = _safe_float(row.get("Smart Potential Score", row.get("Battle Score", row.get("Final Score", 0))))
        verdict = str(row.get("AI Verdict", row.get("Smart Verdict", row.get("Final Verdict", ""))) or "")
        cards.append(_metric_card(f"Rank #{idx + 1}", symbol, f"{verdict} | Score {score:.1f}", colors[idx % len(colors)]))
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:8px 0 14px 0;">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    cols = [
        "AIL Master Rank",
        "AIL Rank Label",
        "Symbol",
        "AIL Categories",
        "Mode Name",
        "Smart Potential Score",
        "Bullish Probability",
        "Smart Confidence",
        "Setup Cleanliness",
        "Momentum Quality",
        "Volume Quality",
        "Trap Risk Score",
        "Final Verdict",
        "AI Verdict",
        "Smart Notes",
    ]
    _render_dataframe(df, cols, height=360)


def _render_comparison_results(result: AILPipelineResult) -> None:
    st.subheader("4. Comparison Results")
    summary = result.comparison_summary or {}
    if not summary:
        st.info("No comparison summary was produced.")
        return
    order = [
        ("best_overall", "Best Overall", "#00d4a8"),
        ("safest_candidate", "Safest Candidate", "#8cf08c"),
        ("strongest_momentum", "Strongest Momentum", "#0094ff"),
        ("best_swing_setup", "Best Swing Setup", "#f0b429"),
        ("early_accumulation", "Early Accumulation", "#7fd1ff"),
        ("lowest_trap_risk", "Lowest Trap Risk", "#b08cff"),
        ("institutional_setup", "Institutional Setup", "#b08cff"),
        ("high_risk_high_reward", "High Risk High Reward", "#ff8c00"),
    ]
    cards = [_leader_card(label, summary.get(key, {}), color) for key, label, color in order]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:8px 0 14px 0;">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_aura_verdict(result: AILPipelineResult) -> pd.DataFrame:
    st.subheader("5. Final Aura Verdict")
    aura_df = pd.DataFrame(result.aura_verdicts)
    cols = [
        "Symbol",
        "Aura Score",
        "Final Verdict",
        "AI Verdict",
        "Entry Timing",
        "Entry Low",
        "Entry High",
        "ATR SL",
        "Risk %",
        "Target 1",
        "Target 2",
        "RR",
        "Timing Reason",
        "Warnings",
    ]
    _render_dataframe(aura_df, cols, height=340)
    return aura_df


def _render_best_buy_tomorrow(aura_df: pd.DataFrame, result: AILPipelineResult) -> None:
    st.subheader("6. Best Buy Tomorrow")
    row = _top_buy_tomorrow(aura_df, result.final_ranked_df)
    if row is None:
        st.info("No buy-tomorrow candidate is available.")
        return
    symbol = _row_symbol(row) or str(row.get("Symbol", "-"))
    verdict = str(row.get("AI Verdict", row.get("Final Verdict", "")) or "")
    aura_score = _safe_float(row.get("Aura Score", row.get("Smart Potential Score", 0.0)), 0.0)
    timing = str(row.get("Entry Timing", row.get("Timing Reason", "")) or "")
    risk = _safe_float(row.get("Risk %", 0.0), 0.0)
    target = _safe_float(row.get("Target 2", row.get("Target 1", 0.0)), 0.0)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0b1017 58%,rgba(0,212,168,0.12));'
        f'border:1.5px solid #00d4a8;border-radius:12px;padding:18px 20px;margin:8px 0 14px 0;">'
        f'<div style="font-size:10px;color:#4a6480;letter-spacing:1.4px;text-transform:uppercase;">Best Buy Tomorrow</div>'
        f'<div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;margin-top:7px;">'
        f'<div><div style="font-size:28px;font-weight:900;color:#00d4a8;">{html.escape(symbol)}</div>'
        f'<div style="font-size:13px;color:#8ab4d8;margin-top:5px;">{html.escape(verdict)} | {html.escape(timing)}</div></div>'
        f'<div style="text-align:right;"><div style="font-size:32px;font-weight:900;color:#f0b429;">{aura_score:.1f}</div>'
        f'<div style="font-size:11px;color:#4a6480;">Aura / Smart Score</div></div>'
        f'</div>'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:16px;font-size:12px;color:#ccd9e8;">'
        f'<div>Risk: <b style="color:#ff4d6d;">{risk:.1f}%</b></div>'
        f'<div>Target zone: <b style="color:#00d4a8;">{target:.2f}</b></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_risk_sector_confidence_learning(result: AILPipelineResult) -> None:
    st.subheader("7. Risk Warnings")
    if isinstance(result.risk_warnings, pd.DataFrame) and not result.risk_warnings.empty:
        _render_dataframe(result.risk_warnings, ["Rank", "Symbol", "Warnings", "Trap Risk Score", "RSI", "Vol / Avg", "EMA20 Distance %"])
    else:
        st.success("No high-priority trap or extension warnings in the final ranked set.")

    st.subheader("8. Sector Strength")
    _render_dataframe(result.sector_strength, ["Sector", "Candidates", "Best Stock", "Best Score", "Avg Smart Score", "Avg Sector Support"])

    st.subheader("9. Confidence Meter")
    meter = result.confidence_meter or {}
    score = _safe_float(meter.get("score"), 0.0)
    label = str(meter.get("label", "No candidates") or "No candidates")
    st.progress(min(max(score / 100.0, 0.0), 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Master Confidence", f"{score:.1f}%")
    c2.metric("Confidence Label", label)
    c3.metric("Ranked Candidates", int(meter.get("count", 0) or 0))

    st.subheader("10. Learning Insights")
    insights = result.learning_insights or {}
    status = insights.get("training_status", {}) if isinstance(insights.get("training_status"), dict) else {}
    feedback = insights.get("feedback_summary", {}) if isinstance(insights.get("feedback_summary"), dict) else {}
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Logged A-I-L Rows", int(insights.get("logged_predictions", 0) or 0))
    l2.metric("Learning Samples", int(status.get("samples", status.get("stock_samples", 0)) or 0))
    l3.metric("Feedback Rows", int(feedback.get("total_logged", 0) or 0))
    acc = feedback.get("accuracy_pct", status.get("validation_accuracy_pct", None))
    l4.metric("Recent Accuracy", f"{_safe_float(acc, 0.0):.1f}%" if acc is not None else "-")
    if insights.get("log_error"):
        st.warning(f"Prediction logging warning: {insights.get('log_error')}")
    weights = insights.get("dynamic_weights")
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        _render_dataframe(weights.head(8), ["Signal", "Observations", "Win Rate", "Static Weight", "Dynamic Weight", "Δ Weight"])


def _render_errors(result: AILPipelineResult) -> None:
    if not result.errors:
        return
    with st.expander("A-I-L fallback notes", expanded=False):
        for error in result.errors:
            st.caption(str(error))


def render_ail_in_one_panel(
    *,
    tickers: list[str],
    workers: int = 12,
    prepare_market_session_data_fn: Callable[..., dict[str, Any]] | None = None,
    preload_all_fn: Callable[..., dict[str, Any]] | None = None,
    run_scan_fn: Callable[..., tuple[list[dict[str, Any]], float]] | None = None,
    enhance_results_fn: Callable[[list[dict[str, Any]], int], pd.DataFrame] | None = None,
    apply_enhanced_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_universal_grading_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase4_logic_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase42_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_gate_to_scan_df_fn: Callable[..., pd.DataFrame] | None = None,
    compute_market_bias_fn: Callable[[], dict[str, Any]] | None = None,
    get_train_function_fn: Callable[[int], Callable] | None = None,
    compute_battle_scores_fn: Callable[..., pd.DataFrame] | None = None,
    run_aura_engine_fn: Callable[[pd.DataFrame, str, dict[str, Any] | None], Any] | None = None,
    compare_prediction_cache_fn: Callable[[], object] | None = None,
    log_scan_predictions_fn: Callable[[pd.DataFrame, int, dict[str, Any] | None], None] | None = None,
    all_data: dict[str, Any] | None = None,
    tt_module: Any = None,
) -> None:
    if not st.session_state.get("ail_in_one_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    hcol, bcol, ccol = st.columns([5, 1.4, 1])
    with hcol:
        st.markdown(
            '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:24px;color:#ccd9e8;margin-bottom:4px;">'
            '🧠 A-I-L IN ONE</h2>'
            '<div style="font-size:12px;color:#4a6480;margin-bottom:12px;">'
            'Master orchestration across NSE Sentinel scanners, comparison, Stock Aura, and learning feedback.</div>',
            unsafe_allow_html=True,
        )
    with bcol:
        st.write("")
        rerun_clicked = st.button("Run Again", key="ail_in_one_run_again_btn", width="stretch")
    with ccol:
        st.write("")
        if st.button("Close", key="ail_in_one_close_btn", width="stretch"):
            st.session_state["ail_in_one_show_panel"] = False
            st.session_state.pop("ail_in_one_run_requested", None)
            st.rerun()

    run_requested = bool(st.session_state.pop("ail_in_one_run_requested", False)) or rerun_clicked
    if run_requested:
        if st.session_state.get("_ail_in_one_running", False):
            st.warning("A-I-L orchestration is already running. Please wait for it to finish.")
        else:
            st.session_state["_ail_in_one_running"] = True
            status_box = st.empty()
            preload_bar = st.progress(0.0)

            def _preload_progress(done: int, total: int, loaded: int) -> None:
                pct = (done / total) if total else 1.0
                preload_bar.progress(min(max(pct, 0.0), 1.0))
                status_box.markdown(
                    f'<div class="status-line"><span class="sdot sdot-green"></span>'
                    f'&nbsp;Preloading shared ALL_DATA&nbsp;·&nbsp;{done:,}/{total:,}'
                    f'&nbsp;·&nbsp;Ready <b style="color:#00d4a8">{loaded:,}</b></div>',
                    unsafe_allow_html=True,
                )

            def _status(stage: str, payload: dict[str, Any]) -> None:
                label = {
                    "preload_start": "Preparing full market data",
                    "preload_done": "Preload complete",
                    "mode_start": f"Scanning Mode {payload.get('mode')} - {payload.get('mode_name', '')}",
                    "mode_done": f"Mode {payload.get('mode')} complete - {payload.get('enhanced', 0)} candidates",
                    "classify_start": "Classifying cross-mode candidates",
                    "compare_start": "Running comparison engine",
                    "aura_start": "Running Stock Aura verdicts",
                    "done": "A-I-L orchestration complete",
                }.get(stage, stage)
                status_box.markdown(
                    f'<div class="status-line"><span class="sdot sdot-green"></span>&nbsp;{html.escape(str(label))}</div>',
                    unsafe_allow_html=True,
                )

            tt_date = st.session_state.get("tt_date_val")
            tt_active = tt_date is not None and tt_module is not None and callable(getattr(tt_module, "activate", None))
            try:
                if tt_active:
                    try:
                        tt_module.activate(tt_date)
                        st.caption(f"Time Travel active for A-I-L: {tt_date}")
                    except Exception:
                        tt_active = False
                with st.spinner("Running A-I-L master orchestration across all scanner modes..."):
                    result = run_ail_pipeline(
                        tickers,
                        workers=workers,
                        prepare_market_session_data_fn=prepare_market_session_data_fn,
                        preload_all_fn=preload_all_fn,
                        run_scan_fn=run_scan_fn,
                        enhance_results_fn=enhance_results_fn,
                        apply_enhanced_logic_fn=apply_enhanced_logic_fn,
                        apply_universal_grading_fn=apply_universal_grading_fn,
                        apply_phase4_logic_fn=apply_phase4_logic_fn,
                        apply_phase42_logic_fn=apply_phase42_logic_fn,
                        apply_gate_to_scan_df_fn=apply_gate_to_scan_df_fn,
                        compute_market_bias_fn=compute_market_bias_fn,
                        get_train_function_fn=get_train_function_fn,
                        compute_battle_scores_fn=compute_battle_scores_fn,
                        run_aura_engine_fn=run_aura_engine_fn,
                        compare_prediction_cache_fn=compare_prediction_cache_fn,
                        log_scan_predictions_fn=log_scan_predictions_fn,
                        all_data=all_data,
                        status_callback=_status,
                        preload_progress_callback=_preload_progress,
                    )
                    st.session_state["ail_in_one_result"] = result
                preload_bar.progress(1.0)
                status_box.success("A-I-L orchestration finished.")
            except Exception as exc:
                status_box.error(f"A-I-L orchestration failed: {exc}")
            finally:
                if tt_active and callable(getattr(tt_module, "restore", None)):
                    try:
                        tt_module.restore()
                    except Exception:
                        pass
                st.session_state.pop("_ail_in_one_running", None)

    result = st.session_state.get("ail_in_one_result")
    if not isinstance(result, AILPipelineResult):
        st.info("A-I-L will run automatically when opened from the sidebar. Use Run Again to refresh the master ranking.")
        return

    _render_errors(result)
    _render_market_scan_summary(result)
    _render_mode_top3(result)
    _render_ranked_leaders(result)
    _render_comparison_results(result)
    aura_df = _render_aura_verdict(result)
    _render_best_buy_tomorrow(aura_df, result)
    _render_risk_sector_confidence_learning(result)
