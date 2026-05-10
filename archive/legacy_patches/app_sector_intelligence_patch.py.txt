"""
app_sector_intelligence_patch.py
══════════════════════════════════════════════════════════════════════════════
Integration patch for app.py  —  Sector Intelligence UI v2

HOW TO APPLY
─────────────
This is NOT a standalone file. Copy the relevant snippet into app.py.

STEP 1 — Replace the old import
────────────────────────────────────────────────────────────────────────────
OLD (remove this):
    from app_sector_intelligence_section import render_sector_intelligence_section
    # or the old inline code that called the old section

NEW (add this, near the top of app.py with your other section imports):

    from app_sector_intelligence_section import render_sector_intelligence_section

(Same import name — the new file is a drop-in replacement.)

STEP 2 — Store enriched scan DataFrame after every scan completes
────────────────────────────────────────────────────────────────────────────
In app.py, find where your scan finishes building the enriched DataFrame
(usually after `enhance_results()`, `apply_universal_grading()`, etc.).

Add ONE line immediately after the final enriched df is ready:

    st.session_state["last_scan_df"] = df.copy()   # ← add this

This is what feeds the Sector Intelligence engine.

If you already have this line, no change needed.

STEP 3 — Remove the old section call, add the new one
────────────────────────────────────────────────────────────────────────────
OLD (remove):
    render_sector_intelligence_section()   # old version — removed

NEW (add in same position):
    render_sector_intelligence_section()   # new clean version

The function name is identical — nothing else in app.py changes.

STEP 4 — Session state init (only if you use explicit init blocks)
────────────────────────────────────────────────────────────────────────────
If app.py initialises session_state keys at startup, add:

    st.session_state.setdefault("_sie_selected_sector", None)
    st.session_state.setdefault("_sie_cache_intel",     None)
    st.session_state.setdefault("_sie_cache_sig",       None)
    st.session_state.setdefault("_sie_bias_cache",      None)

If you don't have explicit init blocks, skip this — the code handles it.

SUMMARY — What changed
════════════════════════════════════════════════════════════════════════════

REMOVED from UI:
  ✗  Huge sector ranking table
  ✗  Money Flow panel with duplicate inflow/outflow blocks
  ✗  Sector Leaders section (separate heavy section)
  ✗  Trade Decision Panel (duplicate of prediction)
  ✗  Export ranking expander
  ✗  All full-width metric rows (st.metric spam)
  ✗  Debug-style stock count / signal quality panels

KEPT (backend — unchanged):
  ✓  compute_sector_intelligence()   — all math intact
  ✓  get_sector_strength()           — intact
  ✓  filter_top_stocks()             — intact
  ✓  get_sector_leaders()            — intact
  ✓  detect_rotation()               — intact
  ✓  compute_market_bias()           — intact, now drives Tomorrow Prediction

NEW UI:
  ✓  Clickable sector grid cards (3-column, minimal, dark)
  ✓  Sector detail view with back button
  ✓  Tomorrow prediction card (Bullish / Bearish / Sideways + confidence %)
  ✓  Market condition card
  ✓  Leader stocks panel (medals)
  ✓  Prediction score bar chart (plotly or HTML fallback)
  ✓  Top stocks compact cards (first 4 visible, rest in expander)
  ✓  Graceful fallback if plotly/yfinance unavailable on Streamlit Cloud
"""