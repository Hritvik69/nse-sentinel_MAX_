"""
PATCH FILE — fetch_nse_tickers_fix.py
════════════════════════════════════════════════════════════════════════
Drop-in replacement for the fetch_nse_tickers() function in app.py.

HOW TO APPLY
────────────
Replace the existing @st.cache_data fetch_nse_tickers() function in
app.py with this version.  Everything else in app.py stays the same.

WHY THIS FIX WORKS
──────────────────
Streamlit Cloud hibernates after ~15 min of inactivity.  On wake-up:
  • st.cache_data  — cleared                      ❌
  • st.cache_resource — cleared                   ❌
  • /tmp/ files — cleared                         ❌
  • GitHub raw — 403 from Streamlit's IP          ❌
  • NSE direct — 403 from Streamlit's IP          ❌
  • nse_tickers.txt in repo — ALWAYS READABLE     ✅

The ONLY permanent fix is to make nse_tickers.txt hold the full list.
This is achieved two ways:

1. nse_ticker_universe.py now writes back to nse_tickers.txt whenever
   it gets ≥2500 tickers from GitHub (first successful wake after deploy).
   After that, every hibernation wake reads the full list from the file.

2. fetch_nse_tickers() below stores the list in st.session_state as a
   permanent within-session backup so that st.cache_data TTL expiry
   within the same session never causes a re-fetch.
"""

# ─────────────────────────────────────────────────────────────────────
# PASTE THIS INTO app.py, replacing the existing fetch_nse_tickers()
# ─────────────────────────────────────────────────────────────────────

# ── Remove the old @st.cache_data decorator on fetch_nse_tickers ─────
# ── and replace the entire function with this version             ─────

@st.cache_data(ttl=43200, show_spinner=False)   # 12-hour TTL (was 6h/24h)
def fetch_nse_tickers() -> list:
    """
    Load the scan universe from the shared ticker-universe module.

    FIX: result is stored in st.session_state as a permanent backup so
    that a TTL-expiry mid-session never drops the count.  The
    nse_ticker_universe module now also writes back to nse_tickers.txt
    on every successful GitHub fetch, making the file self-updating.
    """
    fallback = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","BAJFINANCE.NS",
        "HCLTECH.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","ONGC.NS",
        "NESTLEIND.NS","WIPRO.NS","POWERGRID.NS","NTPC.NS","TECHM.NS",
        "INDUSINDBK.NS","ADANIPORTS.NS","TATAMOTORS.NS","JSWSTEEL.NS",
        "BAJAJFINSV.NS","HINDALCO.NS","GRASIM.NS","DIVISLAB.NS","CIPLA.NS",
        "DRREDDY.NS","BPCL.NS","EICHERMOT.NS","APOLLOHOSP.NS",
        "TATACONSUM.NS","BRITANNIA.NS","COALINDIA.NS","HEROMOTOCO.NS",
        "SHREECEM.NS","SBILIFE.NS","HDFCLIFE.NS","ADANIENT.NS",
        "BAJAJ-AUTO.NS","TATASTEEL.NS","UPL.NS","M&M.NS",
    ]

    # ── FIX 1: session-state fast-path ───────────────────────────────
    # If we already loaded a large list this session, return it immediately
    # without any network or disk I/O — survives st.cache_data TTL expiry.
    _ss_tickers = st.session_state.get("_full_ticker_list", [])
    if len(_ss_tickers) >= 2000:
        return _ss_tickers

    # ── FIX 2: Try the universe module ────────────────────────────────
    try:
        from nse_ticker_universe import get_all_tickers as _get_all_tickers
        from nse_ticker_universe import invalidate_cache as _invalidate_ticker_cache
    except Exception:
        _get_all_tickers = None
        _invalidate_ticker_cache = None

    if _get_all_tickers is not None:
        try:
            tickers = _get_all_tickers(live=True)
            if len(tickers) >= 1000:
                # FIX 3: persist to session_state — survives cache TTL expiry
                if len(tickers) >= 2000:
                    st.session_state["_full_ticker_list"] = tickers
                return tickers
            # Got a small count — invalidate and retry with live=False
            if _invalidate_ticker_cache is not None:
                _invalidate_ticker_cache()
            tickers = _get_all_tickers(live=False)
            if tickers:
                # Still save if it's decent
                if len(tickers) >= 1000:
                    st.session_state.setdefault("_full_ticker_list", tickers)
                return tickers
        except Exception:
            pass

    # ── FIX 4: Read directly from nse_tickers.txt ─────────────────────
    # Last resort before fallback — always works since the file ships with
    # the repo.  After the first successful GitHub fetch, this file will
    # contain the full ~2985-ticker list written back by nse_ticker_universe.
    try:
        import pathlib
        ticker_file = pathlib.Path(__file__).with_name("nse_tickers.txt")
        if ticker_file.exists():
            symbols = {
                f"{line.strip().upper().replace('.NS', '')}.NS"
                for line in ticker_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            }
            if symbols:
                result = sorted(symbols)
                if len(result) >= 1000:
                    st.session_state.setdefault("_full_ticker_list", result)
                return result
    except Exception:
        pass

    return fallback
