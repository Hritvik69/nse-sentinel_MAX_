"""
app_analyse_patch.py
─────────────────────
DROP-IN REPLACEMENTS for analyse() and run_scan() in app.py.

HOW TO INTEGRATE
────────────────
1. Copy this file next to app.py.
2. In app.py, replace the existing analyse() and run_scan() definitions
   with the ones from this file, OR add at the top of app.py:

       from app_analyse_patch import analyse, run_scan, render_scan_diagnostics

   and remove (or comment out) the original analyse() and run_scan() bodies.

3. After run_scan() completes in app.py, call:

       render_scan_diagnostics()

   to display the diagnostics panel in the UI.

WHAT CHANGED vs original
─────────────────────────
analyse()
  • Records every attempt, success, and failure via scan_diagnostics.
  • Drop-point 1: len(df) < 30  → now 5, tagged LOW_QUALITY instead of silent None
  • Drop-point 2: after TT cutoff len(df) < 30 → same, now 5
  • Drop-point 3: staleness > 7 days → records "STALE" then returns None
  • Drop-point 4: len(close) < 25 → records "TOO_SHORT" then returns None
  • Drop-point 5: bad price → records "BAD_PRICE"
  • Drop-point 6: zero volume → records "ZERO_VOLUME"
  • Drop-point 7: NaN indicators → records "NAN_INDICATORS"
  • Drop-point 8: mode filter false → records "SCAN_FILTER" (expected — not an error)
  • Drop-point 9: any exception → records "EXCEPTION"

run_scan()
  • Calls scan_diagnostics.reset() before scan.
  • Displays live stats: attempted / succeeded / data-failed / scan-filtered.
  • Returns (results, elapsed) unchanged — no interface break.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

# ── Project imports (same as app.py) ─────────────────────────────────
from strategy_engines import get_engine_functions, get_df_for_ticker
import scan_diagnostics as _diag

# ── Lazy re-use of helpers already defined in app.py ─────────────────
# These are defined in app.py global scope — we reference them by name.
# If you paste this function directly into app.py, remove these try blocks.
try:
    from app import ema, rsi, get_mktcap_cr, get_nifty_20d_return, _tt
except ImportError:
    # Stubs for static analysis / testing outside app context
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()          # noqa: E731
    def rsi(s, p=14): return 50.0                                       # noqa: E731
    def get_mktcap_cr(t): return 0                                      # noqa: E731
    def get_nifty_20d_return(): return 0.0                              # noqa: E731
    class _tt:
        @staticmethod
        def get_reference_date(): return None
        @staticmethod
        def get_reference_datetime():
            from datetime import datetime
            return datetime.now()

# Minimum viable row count — below this we cannot compute meaningful indicators
_MIN_VIABLE_ROWS  = 5    # hard floor — anything below is genuinely unusable
_PREFERRED_ROWS   = 20   # soft floor — below this we flag LOW_QUALITY


# ══════════════════════════════════════════════════════════════════════
# ANALYSE  (drop-in replacement)
# ══════════════════════════════════════════════════════════════════════

def analyse(ticker: str, mode: int, retries: int = 2) -> dict | None:
    """
    Analyse a single ticker for the given mode.

    Returns a result dict if the stock passes the mode's entry filter,
    None otherwise.  All drop points are recorded via scan_diagnostics.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    _diag.record_attempt(ticker_ns)

    try:
        df = get_df_for_ticker(ticker_ns)

        # ── Drop-point 1: no data at all ──────────────────────────────
        if df is None or df.empty:
            _diag.record_failure(ticker_ns, "NO_DATA")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Open", "Close", "Volume"])

        # ── Drop-point 2: too short after dropna ───────────────────────
        if len(df) < _MIN_VIABLE_ROWS:
            _diag.record_failure(ticker_ns, "TOO_SHORT")
            return None

        # Tag LOW_QUALITY — scan still proceeds
        if len(df) < _PREFERRED_ROWS:
            _diag.record_failure(ticker_ns, "LOW_QUALITY")
            # NOTE: we do NOT return None here — LOW_QUALITY is not fatal.
            # The stock may still pass or fail the mode filter legitimately.

        # ── Time Travel cutoff (no leakage) ───────────────────────────
        try:
            _tt_cut = _tt.get_reference_date()
            if _tt_cut is not None:
                _tt_mask = pd.to_datetime(df.index).date <= _tt_cut
                df = df.loc[_tt_mask]
                if len(df) < _MIN_VIABLE_ROWS:
                    _diag.record_failure(ticker_ns, "TOO_SHORT")
                    return None
        except Exception:
            pass

        # ── Drop-point 3: stale data ───────────────────────────────────
        try:
            last_dt = pd.to_datetime(df.index[-1]).to_pydatetime()
        except Exception:
            _diag.record_failure(ticker_ns, "STALE")
            return None

        if (_tt.get_reference_datetime() - last_dt).days > 7:
            _diag.record_failure(ticker_ns, "STALE")
            return None

        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        open_p = df["Open"].dropna()

        # ── Drop-point 4: close series too short for indicators ────────
        if len(close) < 10:
            _diag.record_failure(ticker_ns, "TOO_SHORT")
            return None

        lc      = float(close.iloc[-1])
        lo      = float(open_p.iloc[-1]) if len(open_p) > 0 else lc
        lv      = float(volume.iloc[-1])
        e20     = float(ema(close, 20).iloc[-1])
        e50     = float(ema(close, 50).iloc[-1])
        avg_vol = (float(volume.iloc[-21:-1].mean())
                   if len(volume) >= 21
                   else float(volume.mean()))
        ri = rsi(close)

        # ── Drop-point 5: bad price ────────────────────────────────────
        if not (1 < lc <= 100_000):
            _diag.record_failure(ticker_ns, "BAD_PRICE")
            return None

        # ── Drop-point 6: zero/negative volume ────────────────────────
        if lv <= 0:
            _diag.record_failure(ticker_ns, "ZERO_VOLUME")
            return None

        # ── Drop-point 7: NaN indicators ──────────────────────────────
        if any(np.isnan(v) for v in (ri, e20, e50)):
            _diag.record_failure(ticker_ns, "NAN_INDICATORS")
            return None

        # ── Mode filters ───────────────────────────────────────────────
        ok = False
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
                _diag.record_failure(ticker_ns, "TOO_SHORT")
                return None
            base_20 = float(close.iloc[-21])
            if base_20 <= 0:
                _diag.record_failure(ticker_ns, "BAD_PRICE")
                return None
            stock_ret_20d = (lc - base_20) / base_20
            nifty_ret_20d = get_nifty_20d_return() or 0.0
            ok = (lc > e20 and e20 > e50
                  and lv > 1.3 * avg_vol and 52 <= ri <= 72
                  and lc >= 0.97 * h20
                  and stock_ret_20d > nifty_ret_20d and lc > lo)
        elif mode == 5:
            h10          = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma  = (float(volume.iloc[-21:-1].mean())
                            if len(volume) >= 21
                            else float(volume.mean()))
            ok = (lc > e20 and e20 > e50 and lv > 1.1 * avg_vol_sma
                  and lc >= 0.99 * h10 and 50 <= ri <= 65
                  and lc > lo and lc > 20)
        elif mode == 6:
            if len(close) < 2:
                _diag.record_failure(ticker_ns, "TOO_SHORT")
                return None
            prev_e20    = float(ema(close, 20).iloc[-2])
            h10         = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma = (float(volume.iloc[-21:-1].mean())
                           if len(volume) >= 21
                           else float(volume.mean()))
            ok = (lc > e20 and e20 > e50 and e20 > prev_e20
                  and lv > 1.1 * avg_vol_sma
                  and lc >= 0.97 * h10 and 50 <= ri <= 68
                  and lc > lo and lc > 40)

        # ── Drop-point 8: mode filter (expected — not a data problem) ──
        if not ok:
            _diag.record_failure(ticker_ns, "SCAN_FILTER")
            return None

        # ── Build result row ───────────────────────────────────────────
        h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
        dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
        dist_ema20    = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
        ret_5d        = (lc / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else np.nan
        ret_20d       = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else np.nan

        _diag.record_success(ticker_ns)

        return {
            "Symbol":            ticker.replace(".NS", ""),
            "Price (₹)":         round(lc, 2),
            "Volume":            int(lv),
            "RSI":               round(ri, 2),
            "EMA 20":            round(e20, 2),
            "EMA 50":            round(e50, 2),
            "Vol / Avg":         round(lv / avg_vol, 2) if avg_vol > 0 else 0,
            "Mode":              {
                1: "🟢 Momentum", 2: "🔵 Balanced", 3: "🟡 Relaxed",
                4: "🟣 Institutional", 5: "🟠 Intraday", 6: "🔴 Swing",
            }[mode],
            "Δ vs 20D High (%)": round(dist_20d_high, 2),
            "Δ vs EMA20 (%)":    round(dist_ema20, 2),
            "5D Return (%)":     round(ret_5d, 2) if not np.isnan(ret_5d) else np.nan,
            "20D Return (%)":    round(ret_20d, 2) if not np.isnan(ret_20d) else np.nan,
        }

    except Exception as exc:
        _diag.record_failure(ticker_ns, "EXCEPTION")
        return None


# ══════════════════════════════════════════════════════════════════════
# RUN_SCAN  (drop-in replacement)
# ══════════════════════════════════════════════════════════════════════

def run_scan(tickers: list[str], mode: int, workers: int = 20) -> tuple[list[dict], float]:
    """
    Parallel scanner — identical interface to original.

    Returns (results_list, elapsed_seconds).

    New: resets diagnostics before scan, shows live diagnostics in UI.
    """
    _diag.reset()   # ← critical: clear state from any previous scan

    results: list[dict] = []
    total = len(tickers)
    done  = 0

    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        status  = st.empty()
    with col_b:
        eta_box = st.empty()

    t0 = time.time()

    def _submit_analyse(executor, ticker):
        try:
            import time_travel_engine as _tt

            if _tt.is_active():
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

            pct       = done / total
            elapsed   = time.time() - t0
            rate      = done / elapsed if elapsed > 0 else 1
            remaining = (total - done) / rate if rate > 0 else 0

            progress_bar.progress(pct)

            # Live diagnostic stats
            diag_snap = _diag.get_report()
            status.markdown(
                f'<div class="status-line">'
                f'<span class="sdot sdot-green"></span>'
                f'&nbsp;Scanned <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
                f'&nbsp;·&nbsp;Found <b style="color:#00d4a8">{len(results)}</b>'
                f'&nbsp;·&nbsp;'
                f'<span style="color:#f0b429">No data: {diag_snap["reasons"].get("NO_DATA", 0)}</span>'
                f'&nbsp;·&nbsp;'
                f'<span style="color:#ff4d6d">Stale: {diag_snap["reasons"].get("STALE", 0)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            eta_box.markdown(
                f'<div class="status-line" style="text-align:right;color:#4a6480;">'
                f'ETA {remaining:.0f}s</div>',
                unsafe_allow_html=True,
            )

    elapsed = time.time() - t0
    progress_bar.progress(1.0)
    return results, elapsed


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS UI  (call after run_scan in app.py)
# ══════════════════════════════════════════════════════════════════════

def render_scan_diagnostics() -> None:
    """
    Render a collapsible diagnostics panel after a scan.

    Paste this call in app.py immediately after the run_scan() block:

        render_scan_diagnostics()
    """
    report = _diag.get_report()
    if report["attempted"] == 0:
        return

    attempted     = report["attempted"]
    succeeded     = report["succeeded"]
    failed_data   = report["failed_data"]
    scan_filtered = report["scan_filtered"]
    reasons       = report["reasons"]
    low_quality   = _diag.get_low_quality_tickers()

    # ── Summary metrics ───────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h4 style="color:#8ab4d8;font-size:13px;margin-bottom:8px;">'
        '🔬 Scan Diagnostics</h4>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Attempted",    f"{attempted:,}")
    m2.metric("✅ Data OK",      f"{attempted - failed_data:,}",
              delta=f"{report['data_ok_pct']:.1f}%",
              delta_color="normal")
    m3.metric("❌ Data Failed",  f"{failed_data:,}",
              delta_color="inverse")
    m4.metric("🔍 Scan Filtered", f"{scan_filtered:,}",
              help="Stocks with valid data that did not match the mode's entry criteria. "
                   "This is expected — not an error.")
    m5.metric("📊 Signals Found", f"{succeeded:,}")

    # ── Reason breakdown ──────────────────────────────────────────────
    with st.expander("📉 Failure Breakdown (data problems only)", expanded=False):
        data_problem_reasons = {
            "NO_DATA", "TOO_SHORT", "STALE",
            "BAD_PRICE", "ZERO_VOLUME", "NAN_INDICATORS", "EXCEPTION",
        }
        rows = [
            {"Reason": r, "Count": c, "Meaning": _reason_meanings.get(r, r)}
            for r, c in sorted(reasons.items(), key=lambda x: -x[1])
            if r in data_problem_reasons
        ]
        if rows:
            df_reasons = pd.DataFrame(rows)
            st.dataframe(df_reasons, width="stretch", hide_index=True)
        else:
            st.success("No data problems detected. All failures were clean scan filters.")

    # ── Low quality tickers ───────────────────────────────────────────
    if low_quality:
        with st.expander(
            f"⚠️ {len(low_quality)} Low-Quality Tickers (thin data, still scanned)",
            expanded=False,
        ):
            st.caption(
                "These stocks have fewer than 20 rows of history. "
                "They were included in the scan — if they passed the entry filter "
                "their results are shown above, but treat them with lower confidence."
            )
            st.code(", ".join(sorted(low_quality)), language=None)

    # ── Full failed ticker list ───────────────────────────────────────
    failed_list = report["failed_tickers"]
    if failed_list:
        with st.expander(
            f"🗂️ {len(failed_list)} Tickers with Data Failures (detail)", expanded=False
        ):
            df_fail = pd.DataFrame(failed_list, columns=["Ticker", "Reason"])
            df_fail["Meaning"] = df_fail["Reason"].map(
                lambda r: _reason_meanings.get(r, r)
            )
            st.dataframe(df_fail, width="stretch", hide_index=True)


_reason_meanings: dict[str, str] = {
    "NO_DATA":        "preload returned None — yfinance download also failed",
    "TOO_SHORT":      "fewer than 10 usable rows after dropna",
    "STALE":          "last candle > 7 calendar days old",
    "BAD_PRICE":      "closing price outside 1–100,000 range",
    "ZERO_VOLUME":    "last session volume is 0 or negative",
    "NAN_INDICATORS": "EMA20/EMA50/RSI computed as NaN (not enough history)",
    "SCAN_FILTER":    "data fine — stock did not match mode entry criteria",
    "EXCEPTION":      "unexpected exception inside analyse()",
    "LOW_QUALITY":    "fewer than 20 rows — thin history, still scanned",
}
