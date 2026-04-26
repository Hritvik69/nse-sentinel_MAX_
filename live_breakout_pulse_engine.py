"""
live_breakout_pulse_engine.py
──────────────────────────────────────────────────────────────────────
*️⃣  LIVE BREAKOUT PULSE — NSE Sentinel  (Live-Data Edition)

Detects stocks showing REAL-TIME breakout strength using live yfinance
data. Deliberately faster and leaner than CSV Breakout Radar — aimed at
catching momentum AS IT IS HAPPENING, not pre-empting it.

Key design differences vs breakout_radar_engine
────────────────────────────────────────────────
• 100 % LIVE  — always calls yfinance (no CSV / ALL_DATA dependency).
• Short history — only last 60 calendar days downloaded per ticker.
• Broad universe — scans the full shared NSE live universe when available.
• Low concurrency — max 4 worker threads + 0.05 s delay each.
• Slightly STRICTER than CSV radar — vol ≥ 1.5 mandatory, RSI 78 = hard
  reject, 6 % EMA20 distance = hard reject.
• Time-Travel aware — when TT is active, data is truncated to cutoff
  before any indicator is computed (zero future leakage).

Scoring Model (0–100)
──────────────────────
Component           Weight  Signals
─────────────────── ──────  ───────────────────────────────────────────
Trend Strength        30 %  Price > EMA20 > EMA50 · EMA20 slope rising
Volume Strength       25 %  Vol / 20-day avg ≥ 1.5 (mandatory)
Breakout Proximity    20 %  Distance to rolling 20-day high
RSI Quality           15 %  Sweet spot 55–68 · Overbought penalty ≥ 72
Momentum              10 %  Green candle · Close near high · Higher-high

Trap Filter (HARD REJECTS)
───────────────────────────
    Vol/Avg < 1.2         → instant reject
    RSI > 78              → instant reject (exhaustion)
    Price > 6 % EMA20     → instant reject (overextended)
    Spike w/o structure   → penalised (5-day ATR spike guard)

Signal Labels
─────────────
    "LIVE BREAKOUT"    → Final Score ≥ 80
    "STRONG MOMENTUM"  → Final Score 65–79
    "WATCH"            → Final Score 50–64

Public API
──────────
    run_live_breakout_pulse(cutoff_date=None)  → pd.DataFrame
    pulse_summary(df)                          → dict
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from nse_ticker_universe import get_all_tickers as _get_all_tickers
    _NSE_UNIVERSE_OK = True
except Exception:
    _NSE_UNIVERSE_OK = False

    def _get_all_tickers(live: bool = True):  # type: ignore[misc]
        return []

# ── Optional Time-Travel integration ─────────────────────────────────
try:
    from time_travel_engine import (
        apply_time_travel_cutoff,
        get_reference_date,
        truncate_df,
        is_active as _tt_is_active,
    )
    _TT_OK = True
except ImportError:
    _TT_OK = False

    def apply_time_travel_cutoff(df):     return df          # type: ignore[misc]
    def get_reference_date():             return None        # type: ignore[misc]
    def truncate_df(df, cutoff):          return df          # type: ignore[misc]
    def _tt_is_active():                  return False       # type: ignore[misc]


def _emit_progress(progress_callback, done: int, total: int, found: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(done, total, found)
        return
    except TypeError:
        try:
            progress_callback(done, total)
        except Exception:
            pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# FALLBACK LIQUID UNIVERSE
# Used only if the shared NSE universe is unavailable.
# ─────────────────────────────────────────────────────────────────────
_LIQUID_500 = [
    # ── Nifty 50 core ────────────────────────────────────────────────
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
    "BAJFINANCE","HCLTECH","SUNPHARMA","TITAN","ULTRACEMCO","ONGC",
    "NESTLEIND","WIPRO","POWERGRID","NTPC","M&M","BAJAJFINSV","DIVISLAB",
    "TATAMOTORS","GRASIM","JSWSTEEL","TATASTEEL","HINDALCO","APOLLOHOSP",
    "TECHM","ADANIPORTS","ADANIENT","CIPLA","COALINDIA","DRREDDY",
    "EICHERMOT","HEROMOTOCO","INDUSINDBK","BRITANNIA","BPCL","SHREECEM",
    "TATACONSUM","UPL","SBILIFE","HDFCLIFE","BAJAJ-AUTO",
    # ── Nifty Next 50 / Nifty100 ─────────────────────────────────────
    "PIDILITIND","BERGEPAINT","HAVELLS","VOLTAS","POLYCAB","GODREJCP",
    "GODREJPROP","DLF","SIEMENS","ABB","TORNTPHARM","LUPIN","AUROPHARMA",
    "ALKEM","BIOCON","RECLTD","PFC","IDFCFIRSTB","BANDHANBNK","FEDERALBNK",
    "RBLBANK","AUBANK","PERSISTENT","COFORGE","LTTS","MPHASIS","TATAELXSI",
    "DIXON","ANGELONE","ANGELBRKG","NAUKRI","ZOMATO","PAYTM","IRCTC",
    "HAL","BEL","BHEL","SAIL","NMDC","NALCO","GAIL","PETRONET",
    "HPCL","IOC","MRPL","ONGC","OIL","ADANIGREEN","ADANIPOWER",
    "ADANITRANS","TATAPOWER","TORNTPOWER","CESC","NHPC","SJVN","IRFC",
    "IREDA","HUDCO","RVNL","RAILTEL","RITES","NBCC","COCHINSHIP",
    # ── Nifty Midcap 150 picks ────────────────────────────────────────
    "ASTRAL","SUPREMEIND","AAVAS","CANFINHOME","LICHSGFIN","HOMEFIRST",
    "PNBHOUSING","REPCO","CHOLAFIN","MUTHOOTFIN","MANAPPURAM","SBIN",
    "KPITTECH","ZENSARTECH","MASTEK","CYIENT","LTIM","OFSS","HEXAWARE",
    "TRENT","SHOPERSTOP","VMART","METRO","CAMPUS","REDTAPE","SAFARI",
    "RELAXO","BATAINDIA","PAGEIND","KEWAL","MANYAVAR","KALYANKJIL",
    "SENCO","THANGAMAYL","RAJESHEXPO","TITAN","ETHOS","DOMS",
    "JUBLFOOD","DEVYANI","SAPPHIRE","BARBEQUE","WESTLIFE","BURGERKING",
    "GODREJIND","MARICO","EMAMILTD","DABUR","JYOTHYLAB","COLPAL",
    "VBL","VARUNBEV","MCDOWELL-N","RADICO","ABINBEV","TILAKNAGAR",
    "IPCALAB","NATCOPHARM","LAURUSLABS","GRANULES","SUVEN","AJANTPHARM",
    "GLAXO","PFIZER","SANOFI","ASTRAZEN","NOVARTIS","ABBOTINDIA",
    "APOLLOPIPE","APLAPOLLO","SUPREMEIND","WELCORP","JINDALSTEL",
    "JINDALSAW","RATNAMANI","MAHSEAMLES","MTAR","GRSE","MAZAGON",
    "TITAGARH","IRCON","KNRCON","PNCINFRA","ASHOKA","HGINFRA",
    "KECL","KEC","KALPATPOWR","THERMAX","CUMMINSIND","GRINDWELL",
    "SCHAEFFLER","SKFINDIA","TIMKEN","KENNAMETAL","VESUVIUS",
    "CARBORUNIV","ELGIEQUIP","TDPOWERSYS","VOLTAMP","AMARARAJA",
    "EXIDEIND","HEG","GRAPHITE","MOIL","NATIONALUM","HINDZINC","VEDL",
    "PRAJIND","DEEPAKFERT","CHAMBAL","GNFC","GSFC","DEEPAKNTR",
    "PIDILITIND","VINATI","FINEORG","ALKYLAMINE","LAXMIMACH","AIAENG",
    "BHARAT","BHARATFORG","TIINDIA","SUNDRMFAST","SUPRAJIT","GABRIEL",
    "MOTHERSON","ENDURANCE","MINDA","UNOMINDA","SAMVARDHANA","BOSCHLTD",
    "SONA","CRAFTSMAN","JMAUTOLTD","MAHINDCIE","JTEKTINDIA","SCHAEFFLER",
    "ICICIGI","HDFCGI","STARHEALTH","NIACL","GICRE","SBIMF","UTIAMC",
    "HDFCAMC","NIPPONLIFE","ABSLAMC","MOTILALOFS","ANGELONE","ISEC",
    "IIFLSEC","5PAISA","BAJAJHFL","POONAWALLA","CHOLAHLDNG","EDELWEISS",
    "CREDITACC","FUSION","AROHAN","SPANDANA","UGROCAP","SURYODAY",
    "EQUITASBNK","ESAFSFB","UJJIVANSFB","UTKARSHBNK","CAPITALSFB","JANA",
    "DMART","AVENUE","NYKAA","MAPMYINDIA","INDIAMART","JUSTDIAL","ZAGGLE",
    "IXIGO","EASEMYTRIP","RATEGAIN","TANLA","INTELLECT","NEWGEN",
    "DATAMATICS","SAKSOFT","KSOLVES","LATENTVIEW","INFOBEAN","TATACOMM",
    "TTML","HFCL","STLTECH","CMSINFO","ECLERX","SUBEXLTD","TRIGYN",
    "ROUTE","FIRSTSOURCE","TEAMLEASE","QUESS","XCHANGING","CONCENTRIX",
    "ZEEL","TV18BRDCST","NETWORK18","SUNTV","NDTV","TVTODAY","DBCORP",
    "JAGRAN","HMVL","SAREGAMA","NAZARA","TIPFILMS","PVRINOX","INOX",
    "WONDERLA","DELTACORP","DELTAMET","BSEINDIA","CDSL","NSDL","MCX",
    "IIFL","MSTCLTD","CRISIL","CARERATING","ICRA","KFINTECH","CAMS",
    "PCJEWELLER","KALYAN","GOLDIAM","RAJESHEXPO","TRIBHOVAN","SENCO",
    "SPICEJET","INDIGO","INTERGLOBE","BLUEDART","DELHIVERY","GATI",
    "TCI","TCIEXPRESS","VRLLOG","MAHINDLOG","NAVKARCORP","CONTAINERCO",
    "CONCOR","ALLCARGO","GATEWAY","ESCORTS","JKIL","JBMA","PATELENG",
    "DILIPBUILDCON","NCC","MBLINFRA","CAPACITE","SADBHAV","MONTECARLO",
    "GPPL","GPIL","ADANIPORTS","KANDLAPORT","SEAMECLTD","GESHIP",
    "RPOWER","JPPOWER","JSWENERGY","GREENKO","WEBSOL","WAAREEENG",
    "SURYAROSNI","INOXWIND","SUZLON","ORIENTGREEN","SJVN","NHPC",
    "TATASP","ACMESOLAR","AZAD","GOLDENKA","NEWVISTA","PREMIER",
]

# De-duplicate, remove empties
_LIQUID_500 = sorted(set(t.strip() for t in _LIQUID_500 if t.strip()))

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
_BATCH_WORKERS  = 4          # batch-level concurrency
_BATCH_SIZE     = 60         # tickers per yfinance batch request
_MAX_WORKERS    = 4          # per-ticker fallback concurrency
_REQUEST_DELAY  = 0.01       # only used by per-ticker fallback
_MIN_ROWS       = 20         # minimum candles needed for indicators
_PERIOD         = "2mo"      # ~60 calendar days
_MIN_SCORE      = 50         # minimum score to appear in results
_MIN_VOL_RATIO  = 1.2        # hard reject below this
_MAX_RSI        = 78         # hard reject above this
_MAX_EMA_DIST   = 6.0        # % — hard reject above this (overextended)

# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Return the most-recent RSI value. Never raises."""
    try:
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        return float(rsi_series.dropna().iloc[-1])
    except Exception:
        return 50.0


def _sf(v: object, default: float = 0.0) -> float:
    """safe float cast."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _normalize_ns_symbol(symbol: object) -> str:
    raw = str(symbol or "").strip().upper().replace(".NS", "")
    if not raw:
        return ""
    return f"{raw}.NS"


def _clean_live_df(df: pd.DataFrame | None, cutoff: date | None) -> pd.DataFrame | None:
    """Normalize one ticker OHLCV frame and apply the optional TT cutoff."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    try:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out = out.dropna(subset=["Close", "Volume"])
        if cutoff is not None:
            out = truncate_df(out, cutoff)
        if out is None or not isinstance(out, pd.DataFrame) or len(out) < _MIN_ROWS:
            return None
        return out
    except Exception:
        return None


def _build_live_universe() -> list[str]:
    """
    Build the live scan universe.

    Priority:
      1. Shared NSE universe (live=True)
      2. Shared NSE universe (live=False) fallback
      3. Existing liquid-watchlist names
    """
    if _NSE_UNIVERSE_OK:
        try:
            live_universe = [
                _normalize_ns_symbol(t)
                for t in _get_all_tickers(live=True)
            ]
            live_universe = [t for t in live_universe if t]
            if live_universe:
                return sorted(set(live_universe))
        except Exception:
            pass

        try:
            static_universe = [
                _normalize_ns_symbol(t)
                for t in _get_all_tickers(live=False)
            ]
            static_universe = [t for t in static_universe if t]
            if static_universe:
                return sorted(set(static_universe))
        except Exception:
            pass

    return sorted({_normalize_ns_symbol(t) for t in _LIQUID_500 if _normalize_ns_symbol(t)})


def _download_live(ticker_ns: str, cutoff: date | None) -> pd.DataFrame | None:
    """
    Download OHLCV for ticker_ns.
    If cutoff is set (Time Travel) → truncate immediately after download.
    Returns None if data unusable (< _MIN_ROWS rows or download error).
    """
    try:
        time.sleep(_REQUEST_DELAY)
        df = yf.download(
            ticker_ns,
            period=_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < _MIN_ROWS:
            return None
        # ── Time-Travel truncation ────────────────────────────────────
        if cutoff is not None:
            df = truncate_df(df, cutoff)
            if df is None or len(df) < _MIN_ROWS:
                return None
        return df
    except Exception:
        return None


def _download_live_batch(tickers_ns: list[str], cutoff: date | None) -> dict[str, pd.DataFrame | None]:
    """Download many tickers in one yfinance call and split them back out."""
    if not tickers_ns:
        return {}

    normalized = [_normalize_ns_symbol(t) for t in tickers_ns if _normalize_ns_symbol(t)]
    if not normalized:
        return {}

    try:
        batch_df = yf.download(
            normalized,
            period=_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=20,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return {ticker_ns: None for ticker_ns in normalized}

    if batch_df is None or batch_df.empty:
        return {ticker_ns: None for ticker_ns in normalized}

    out: dict[str, pd.DataFrame | None] = {}

    if len(normalized) == 1 and not isinstance(batch_df.columns, pd.MultiIndex):
        out[normalized[0]] = _clean_live_df(batch_df, cutoff)
        return out

    if not isinstance(batch_df.columns, pd.MultiIndex):
        return {ticker_ns: None for ticker_ns in normalized}

    available = set(batch_df.columns.get_level_values(0))
    for ticker_ns in normalized:
        if ticker_ns not in available:
            out[ticker_ns] = None
            continue
        try:
            one_df = batch_df.xs(ticker_ns, axis=1, level=0)
        except Exception:
            out[ticker_ns] = None
            continue
        out[ticker_ns] = _clean_live_df(one_df, cutoff)

    return out


# ─────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────

def _score_ticker(
    ticker_ns: str,
    cutoff: date | None,
    df_override: pd.DataFrame | None = None,
) -> dict | None:
    """
    Download and score a single ticker.
    Returns a result dict or None if the ticker fails any mandatory gate.
    """
    sym = ticker_ns.replace(".NS", "")
    df = df_override if df_override is not None else _download_live(ticker_ns, cutoff)
    if df is None:
        return None

    try:
        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        high   = df["High"].astype(float)
        low    = df["Low"].astype(float)
        opn    = df["Open"].astype(float)

        if len(close) < _MIN_ROWS:
            return None

        # ── Indicators ─────────────────────────────────────────────
        e20  = _ema(close, 20)
        e50  = _ema(close, 50) if len(close) >= 50 else _ema(close, len(close) // 2)
        e20_prev = float(e20.iloc[-2]) if len(e20) >= 2 else float(e20.iloc[-1])

        price    = float(close.iloc[-1])
        e20_now  = float(e20.iloc[-1])
        e50_now  = float(e50.iloc[-1])
        rsi_val  = _rsi(close)

        avg_vol  = float(volume.rolling(20, min_periods=10).mean().shift(1).iloc[-1])
        vol_now  = float(volume.iloc[-1])
        vol_rat  = (vol_now / avg_vol) if avg_vol > 0 else 0.0

        high_20d = float(high.tail(20).max())
        dist_high = ((high_20d - price) / high_20d * 100.0) if high_20d > 0 else 99.0

        ema_dist = ((price / e20_now) - 1.0) * 100.0 if e20_now > 0 else 0.0
        ema_slope_rising = e20_now > e20_prev

        # ── HARD REJECT GATES ──────────────────────────────────────
        if vol_rat < _MIN_VOL_RATIO:
            return None
        if rsi_val > _MAX_RSI:
            return None
        if ema_dist > _MAX_EMA_DIST:
            return None
        # Trend gate: price must be above EMA20
        if price <= e20_now:
            return None

        # ── ATR spike-without-structure guard ──────────────────────
        atr_now  = float((high - low).tail(5).mean())
        atr_base = float((high - low).tail(20).mean())
        spike_ratio = (atr_now / atr_base) if atr_base > 0 else 1.0
        spike_penalty = max(0.0, (spike_ratio - 2.5) * 6.0)   # > 2.5× ATR = penalise

        # ── SCORE COMPONENTS ──────────────────────────────────────
        # 1. Trend Strength (30 %)
        trend_pts = 0.0
        if price > e20_now > 0:
            trend_pts += 40.0
        if e20_now > e50_now > 0:
            trend_pts += 40.0
        if ema_slope_rising:
            trend_pts += 20.0
        trend_score = min(trend_pts, 100.0) * 0.30

        # 2. Volume Strength (25 %)
        if vol_rat >= 3.0:
            vol_pts = 100.0
        elif vol_rat >= 2.0:
            vol_pts = 80.0
        elif vol_rat >= 1.5:
            vol_pts = 60.0
        else:
            vol_pts = max(0.0, (vol_rat - 1.2) / 0.3 * 40.0)
        vol_score = vol_pts * 0.25

        # 3. Breakout Proximity (20 %)
        if dist_high <= 1.0:
            prox_pts = 100.0
        elif dist_high <= 2.0:
            prox_pts = 80.0
        elif dist_high <= 3.0:
            prox_pts = 60.0
        elif dist_high <= 5.0:
            prox_pts = 35.0
        else:
            prox_pts = max(0.0, 20.0 - dist_high * 2.0)
        prox_score = prox_pts * 0.20

        # 4. RSI Quality (15 %)
        if 55.0 <= rsi_val <= 68.0:
            rsi_pts = 100.0
        elif 50.0 <= rsi_val < 55.0:
            rsi_pts = 70.0
        elif 68.0 < rsi_val <= 72.0:
            rsi_pts = 45.0
        elif rsi_val > 72.0:
            rsi_pts = max(0.0, 45.0 - (rsi_val - 72.0) * 8.0)
        else:
            rsi_pts = max(0.0, (rsi_val - 40.0) * 4.0)
        rsi_score = rsi_pts * 0.15

        # 5. Momentum (10 %)
        mom_pts = 0.0
        last_open  = float(opn.iloc[-1])
        last_close = float(close.iloc[-1])
        last_high  = float(high.iloc[-1])
        last_low   = float(low.iloc[-1])
        prev_high  = float(high.iloc[-2]) if len(high) >= 2 else last_high - 1.0

        candle_body = last_close - last_open
        candle_range = last_high - last_low if last_high > last_low else 0.01
        close_vs_range = (last_close - last_low) / candle_range  # 0–1 (1 = close at high)

        if candle_body > 0:                     mom_pts += 30.0   # green candle
        if close_vs_range >= 0.70:              mom_pts += 30.0   # close near high of day
        if last_high > prev_high:               mom_pts += 25.0   # higher high
        if vol_rat >= 1.5 and candle_body > 0:  mom_pts += 15.0   # vol-confirmed green
        mom_score = min(mom_pts, 100.0) * 0.10

        # ── AGGREGATE ─────────────────────────────────────────────
        raw_score  = trend_score + vol_score + prox_score + rsi_score + mom_score
        final_score = float(np.clip(raw_score - spike_penalty, 0.0, 100.0))

        if final_score < _MIN_SCORE:
            return None

        # ── SIGNAL LABEL ──────────────────────────────────────────
        if final_score >= 80.0:
            signal = "LIVE BREAKOUT"
        elif final_score >= 65.0:
            signal = "STRONG MOMENTUM"
        else:
            signal = "WATCH"

        # ── MOMENTUM STRENGTH label ───────────────────────────────
        if mom_pts >= 75.0:
            mom_label = "Strong"
        elif mom_pts >= 40.0:
            mom_label = "Moderate"
        else:
            mom_label = "Weak"

        # ── EMA20 Slope label ─────────────────────────────────────
        trend_label = "Rising" if ema_slope_rising else "Flat"

        return {
            "Symbol":            sym,
            "Price (₹)":         round(price, 2),
            "RSI":               round(rsi_val, 1),
            "Vol / Avg":         round(vol_rat, 2),
            "Dist to High (%)":  round(dist_high, 2),
            "Δ vs EMA20 (%)":    round(ema_dist, 2),
            "EMA20 Slope":       trend_label,
            "Momentum":          mom_label,
            "Final Score":       round(final_score, 1),
            "Signal":            signal,
            "Chart Link":        f"https://www.tradingview.com/chart/?symbol=NSE:{sym}",
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────

def run_live_breakout_pulse(
    cutoff_date: date | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run the Live Breakout Pulse scan.

    Parameters
    ----------
    cutoff_date : date | None
        When set (Time-Travel mode) all data is truncated to this date
        before indicator computation — zero future leakage.
    progress_callback : callable(done: int, total: int) | None
        Optional callback invoked after each ticker completes so the
        UI can update a progress bar.

    Returns
    -------
    pd.DataFrame
        Results sorted by Final Score descending.
        Empty DataFrame if no stocks passed the filter.
    """
    # ── Resolve cutoff ─────────────────────────────────────────────
    if cutoff_date is None and _TT_OK and _tt_is_active():
        cutoff_date = get_reference_date()

    # ── Build ticker list from the shared live NSE universe ──────────
    tickers_ns = _build_live_universe()

    total   = len(tickers_ns)
    results: list[dict] = []
    done    = 0
    lock    = threading.Lock()
    batches = [
        tickers_ns[i:i + _BATCH_SIZE]
        for i in range(0, total, _BATCH_SIZE)
    ]

    def _worker(batch: list[str]) -> tuple[list[dict], int]:
        batch_frames = _download_live_batch(batch, cutoff_date)
        batch_results: list[dict] = []
        for ticker_ns in batch:
            frame = batch_frames.get(ticker_ns)
            row = _score_ticker(ticker_ns, cutoff_date, df_override=frame)
            if row is not None:
                batch_results.append(row)
        return batch_results, len(batch)

    with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as ex:
        futs = {ex.submit(_worker, batch): batch for batch in batches}
        for fut in as_completed(futs):
            try:
                batch_rows, batch_done = fut.result()
                with lock:
                    if batch_rows:
                        results.extend(batch_rows)
            except Exception:
                batch_done = len(futs.get(fut, []))
            with lock:
                done += batch_done
                current_done = done
                current_found = len(results)
            _emit_progress(progress_callback, current_done, total, current_found)

    if not results:
        empty = pd.DataFrame()
        empty.attrs["universe_scanned"] = total
        return empty

    df = pd.DataFrame(results)
    df = df.sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.attrs["universe_scanned"] = total
    return df


def pulse_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for display in the UI.

    Keys: total, live_breakouts, strong_momentum, watch,
          avg_score, avg_rsi, avg_vol_ratio, scan_time
    """
    out = {
        "universe_scanned": 0,
        "total":           0,
        "live_breakouts":  0,
        "strong_momentum": 0,
        "watch":           0,
        "avg_score":       0.0,
        "avg_rsi":         0.0,
        "avg_vol_ratio":   0.0,
    }
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return out
        out["universe_scanned"] = int(df.attrs.get("universe_scanned", 0) or 0)
        out["total"]           = int(len(df))
        out["live_breakouts"]  = int((df["Signal"] == "LIVE BREAKOUT").sum())
        out["strong_momentum"] = int((df["Signal"] == "STRONG MOMENTUM").sum())
        out["watch"]           = int((df["Signal"] == "WATCH").sum())
        out["avg_score"]       = round(float(df["Final Score"].mean()), 1)
        if "RSI" in df.columns:
            out["avg_rsi"]     = round(float(df["RSI"].mean()), 1)
        if "Vol / Avg" in df.columns:
            out["avg_vol_ratio"] = round(float(df["Vol / Avg"].mean()), 2)
    except Exception:
        pass
    return out
