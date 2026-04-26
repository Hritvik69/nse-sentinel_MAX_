"""
multi_index_market_bias_engine.py
───────────────────────────────────
ADD-ON layer: Multi-Index Market Bias & Sector Prediction Engine for NSE Sentinel.

NEW FILE ONLY — does not modify any existing file or function.

Public API
──────────
    preload_all_sectors(workers=12)
        → None  Preloads all sector stock data in parallel via ALL_DATA.

    analyze_index(sector_name)
        → dict  Index trend, RSI, 5D/20D returns, strength score (0–100).

    build_sector_raw_rows(sector_name, mode=2)
        → list[dict]  Raw indicator rows ready for enhance_results().

    compute_sector_prediction(sector_name, processed_df, index_analysis)
        → dict  Sector direction (Bullish/Bearish/Sideways), confidence,
                bullish %, top stocks, tomorrow prediction.

    compute_overall_market(all_sector_results)
        → dict  Weighted market prediction, strongest/weakest sector.

Design rules
────────────
• Stock data : ZERO new API logic — uses ALL_DATA / get_df_for_ticker().
• Index data : Downloaded once per session via yfinance; cached in-module.
• Never modifies any existing column, engine, scoring, or ML logic.
• Never crashes — every function wrapped in try/except.
• Works alongside any scan mode (1–6).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from strategy_engines._engine_utils import (
        ema, rsi_vec, safe, get_df_for_ticker, preload_all, ALL_DATA,
    )
    _EU_OK = True
except Exception:
    _EU_OK = False
    # Provide stubs so the module still imports cleanly
    import numpy as _np
    import pandas as _pd
    ALL_DATA: dict = {}  # type: ignore[assignment]
    def ema(series, period):          return series  # type: ignore[misc]
    def rsi_vec(close, period=14):    return _pd.Series([50.0] * len(close), index=close.index)  # type: ignore[misc]
    def safe(v, default=0.0):         # type: ignore[misc]
        try: return float(v) if _np.isfinite(float(v)) else default  # type: ignore[arg-type]
        except Exception: return default
    def get_df_for_ticker(ticker):    return None  # type: ignore[misc]
    def preload_all(tickers, **kw):   pass  # type: ignore[misc]

# ── Time Travel integration ────────────────────────────────────────────
try:
    from time_travel_engine import (
        apply_time_travel_cutoff as _tt_cutoff,
        get_reference_date       as _tt_ref_date,
    )
    _TT_OK = True
except ImportError:
    _TT_OK = False
    def _tt_cutoff(df):    # type: ignore[misc]
        return df
    def _tt_ref_date():    # type: ignore[misc]
        return None


# ═══════════════════════════════════════════════════════════════════════
# ── CONSTANTS ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

# Yahoo Finance index tickers (NSE indices)
INDEX_TICKERS: dict[str, str] = {
    "Nifty 50":      "^NSEI",
    "Nifty Next 50": "^NSMIDCP",   # best available proxy via yfinance
    "Nifty Bank":    "^NSEBANK",
    "Nifty IT":      "^CNXIT",
    "Nifty Auto":    "^CNXAUTO",
    "Nifty Pharma":  "^CNXPHARMA",
    "Nifty FMCG":    "^CNXFMCG",
}

# Constituent stocks per index (static mapping — no API needed)
INDEX_STOCK_MAP: dict[str, list[str]] = {
    "Nifty 50": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "BAJFINANCE",
        "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "ONGC",
        "NTPC", "POWERGRID", "WIPRO", "M&M", "NESTLEIND",
        "BAJAJFINSV", "TATAMOTORS", "TECHM", "INDUSINDBK", "ADANIPORTS",
    ],
    "Nifty Next 50": [
        "ADANIGREEN", "HAVELLS", "BERGEPAINT", "SIEMENS", "CHOLAFIN",
        "DLF", "IRCTC", "NAUKRI", "TRENT", "SRF",
        "PIIND", "GRASIM", "VEDL", "BOSCHLTD", "MCDOWELL-N",
    ],
    "Nifty Bank": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
        "PNB", "BANKBARODA",
    ],
    "Nifty IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTTS", "PERSISTENT", "COFORGE", "MPHASIS", "TATAELXSI",
    ],
    "Nifty Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "TVSMOTOR", "ASHOKLEY", "MOTHERSON", "BALKRISIND",
    ],
    "Nifty Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
        "TORNTPHARM", "LUPIN", "ALKEM", "BIOCON", "GLENMARK",
    ],
    "Nifty FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "GODREJCP", "MARICO", "COLPAL", "TATACONSUM", "EMAMILTD",
    ],
}

# Weights for overall market bias computation (must sum to 1.0)
SECTOR_WEIGHTS: dict[str, float] = {
    "Nifty 50":      0.30,
    "Nifty Bank":    0.25,
    "Nifty IT":      0.15,
    "Nifty Auto":    0.10,
    "Nifty Pharma":  0.10,
    "Nifty FMCG":    0.10,
    "Nifty Next 50": 0.00,   # informational only — excluded from overall weight
}

# Mode label map (mirrors battle_mode_engine)
_MODE_LABELS: dict[int, str] = {
    1: "🟢 Momentum",
    2: "🔵 Balanced",
    3: "🟡 Relaxed",
    4: "🟣 Institutional",
    5: "🟠 Intraday",
    6: "🔴 Swing",
}

# ═══════════════════════════════════════════════════════════════════════
# ── MODULE-LEVEL CACHE for index data (one download per session) ───────
# ═══════════════════════════════════════════════════════════════════════

_INDEX_CACHE: dict[str, dict] = {}
_INDEX_CACHE_LOCK = threading.Lock()
_DASHBOARD_STOCK_ROW_CACHE: dict[tuple, dict | None] = {}
_DASHBOARD_STOCK_ROW_CACHE_LOCK = threading.Lock()
_SECTOR_MASTER_STOCKS_CACHE: dict[str, list[str]] | None = None
_SECTOR_MASTER_STOCKS_LOCK = threading.Lock()
_DASHBOARD_SECTOR_STOCKS_CACHE: dict[str, tuple[str, ...]] = {}
_DASHBOARD_SECTOR_STOCKS_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
# ── INTERNAL HELPERS ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _sf(v: object, default: float = 0.0) -> float:
    """Safe float — never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _download_index_ohlcv(ticker: str, period: str = "3mo") -> pd.DataFrame | None:
    """
    Download index OHLCV from yfinance.
    Separate from _engine_utils.download_history() to avoid .NS suffix logic.
    Returns None on failure or insufficient data (< 25 rows).
    """
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=True, progress=False, timeout=12, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # ── Time-Travel: truncate to historical cutoff ─────────────────
        df = _tt_cutoff(df)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df if len(df) >= 25 else None
    except Exception:
        return None


def _compute_index_strength(df: pd.DataFrame) -> dict:
    """
    Compute index strength metrics from OHLCV DataFrame.
    Returns a dict with trend, RSI, returns, and a 0–100 strength score.
    """
    try:
        close = df["Close"].dropna().astype(float)
        if len(close) < 25:
            return _index_fallback()

        e20_s = ema(close, 20)
        e50_s = ema(close, 50)
        rsi_s = rsi_vec(close)

        e20  = _sf(e20_s.iloc[-1])
        e50  = _sf(e50_s.iloc[-1])
        rsi_val = _sf(rsi_s.iloc[-1], 50.0)
        lc   = _sf(close.iloc[-1])

        ret_5d  = (_sf(close.iloc[-1]) / _sf(close.iloc[-6],  lc) - 1) * 100 if len(close) >= 6  else 0.0
        ret_20d = (_sf(close.iloc[-1]) / _sf(close.iloc[-21], lc) - 1) * 100 if len(close) >= 21 else 0.0

        # Volume trend (if available)
        vol_trend = "N/A"
        if "Volume" in df.columns:
            try:
                vol = df["Volume"].dropna().astype(float)
                if len(vol) >= 21:
                    avg20 = float(vol.iloc[-21:-1].mean())
                    last_vol = float(vol.iloc[-1])
                    ratio = last_vol / avg20 if avg20 > 0 else 1.0
                    if ratio > 1.4:
                        vol_trend = "STRONG"
                    elif ratio > 1.1:
                        vol_trend = "BUILDING"
                    elif ratio >= 0.8:
                        vol_trend = "NORMAL"
                    else:
                        vol_trend = "WEAK"
            except Exception:
                pass

        # ── Strength score (0–100) ────────────────────────────────────
        score = 50.0

        # EMA trend component (±15)
        if e20 > 0 and e50 > 0:
            if e20 > e50:
                score += 15.0
            else:
                score -= 12.0

        # RSI component (±15)
        if 50.0 <= rsi_val <= 65.0:
            score += 15.0
        elif 65.0 < rsi_val <= 72.0:
            score += 7.0          # still bullish but stretched
        elif rsi_val > 72.0:
            score += 2.0          # overbought → reduce confidence
        elif rsi_val < 40.0:
            score -= 15.0
        elif rsi_val < 50.0:
            score -= 5.0

        # 5D return component (±10)
        score += float(np.clip(ret_5d * 2.0, -10.0, 10.0))

        # 20D return component (±15)
        score += float(np.clip(ret_20d * 0.75, -15.0, 15.0))

        # Overbought RSI penalty on strength
        if rsi_val > 70.0:
            score -= 5.0

        strength = float(np.clip(score, 0.0, 100.0))

        # ── Trend label ───────────────────────────────────────────────
        if e20 > e50 and rsi_val >= 50.0 and ret_20d > 0:
            trend = "Bullish"
        elif e20 < e50 or (rsi_val < 45.0 and ret_20d < 0):
            trend = "Bearish"
        else:
            trend = "Sideways"

        return {
            "trend":        trend,
            "strength_score": round(strength, 1),
            "rsi":          round(rsi_val, 1),
            "ret_5d":       round(ret_5d, 2),
            "ret_20d":      round(ret_20d, 2),
            "ema_above":    e20 > e50,
            "vol_trend":    vol_trend,
            "available":    True,
        }

    except Exception:
        return _index_fallback()


def _index_fallback() -> dict:
    """Return a neutral/unavailable index analysis dict."""
    return {
        "trend":          "Sideways",
        "strength_score": 50.0,
        "rsi":            50.0,
        "ret_5d":         0.0,
        "ret_20d":        0.0,
        "ema_above":      False,
        "vol_trend":      "N/A",
        "available":      False,
    }


def _build_stock_row(ticker_ns: str, mode: int) -> dict | None:
    """
    Build a raw indicator row for one stock using preloaded ALL_DATA.
    Mirrors the logic in battle_mode_engine._build_battle_row().
    Returns None on any failure.
    """
    try:
        df = get_df_for_ticker(ticker_ns)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = [c for c in ["Close", "Volume"] if c in df.columns]
        if not needed:
            return None

        df = df.dropna(subset=needed)
        if len(df) < 25:
            return None

        close  = df["Close"].dropna().astype(float)
        volume = df["Volume"].dropna().astype(float)

        if len(close) < 25:
            return None

        lc  = float(close.iloc[-1])
        lv  = float(volume.iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])

        avg_vol = (
            float(volume.iloc[-21:-1].mean())
            if len(volume) >= 21
            else float(volume.mean())
        )

        rsi_s = rsi_vec(close)
        ri    = float(rsi_s.iloc[-1]) if not rsi_s.empty else float("nan")

        # Basic validity
        if not (1 < lc <= 1_00_000):
            return None
        if lv <= 0 or any(np.isnan(v) for v in (ri, e20, e50)):
            return None

        h20     = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
        d20h    = (lc / h20 - 1.0) * 100.0        if h20 > 0 else 0.0
        d_ema20 = (lc / e20 - 1.0) * 100.0        if e20 > 0 else 0.0
        ret_5d  = (lc / float(close.iloc[-6])  - 1.0) * 100.0 if len(close) >= 6  else float("nan")
        ret_20d = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else float("nan")

        sym = ticker_ns.replace(".NS", "")

        return {
            "Symbol":             sym,
            "Price (₹)":          round(lc, 2),
            "Volume":             int(lv),
            "RSI":                round(ri, 2),
            "EMA 20":             round(e20, 2),
            "EMA 50":             round(e50, 2),
            "Vol / Avg":          round(lv / avg_vol, 2) if avg_vol > 0 else 0.0,
            "Mode":               _MODE_LABELS.get(mode, "🔵 Balanced"),
            "Δ vs 20D High (%)":  round(d20h, 2),
            "Δ vs EMA20 (%)":     round(d_ema20, 2),
            "5D Return (%)":      round(ret_5d, 2)  if np.isfinite(ret_5d)  else 0.0,
            "20D Return (%)":     round(ret_20d, 2) if np.isfinite(ret_20d) else 0.0,
        }
    except Exception:
        return None


def _dashboard_df_signature(df: pd.DataFrame | None) -> tuple[int, str, float, float]:
    """Small hashable snapshot used to invalidate cached stock rows safely."""
    try:
        if df is None or df.empty:
            return (0, "", 0.0, 0.0)
        last_idx = str(pd.to_datetime(df.index[-1]))
        close_s = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
        vol_s = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)
        last_close = float(close_s.iloc[-1]) if not close_s.empty else 0.0
        last_volume = float(vol_s.iloc[-1]) if not vol_s.empty else 0.0
        return (len(df), last_idx, round(last_close, 6), round(last_volume, 2))
    except Exception:
        return (0, "", 0.0, 0.0)


def _build_stock_row_cached(ticker_ns: str, mode: int) -> dict | None:
    """Reuse stock-row computation when the same OHLCV snapshot is scanned again."""
    try:
        df = ALL_DATA.get(ticker_ns)
        if df is None:
            return None
        cache_key = (mode, ticker_ns, _dashboard_df_signature(df))
        with _DASHBOARD_STOCK_ROW_CACHE_LOCK:
            if cache_key in _DASHBOARD_STOCK_ROW_CACHE:
                cached = _DASHBOARD_STOCK_ROW_CACHE[cache_key]
                return dict(cached) if isinstance(cached, dict) else None

        row = _build_stock_row(ticker_ns, mode)
        with _DASHBOARD_STOCK_ROW_CACHE_LOCK:
            _DASHBOARD_STOCK_ROW_CACHE[cache_key] = dict(row) if isinstance(row, dict) else None
        return row
    except Exception:
        return None


def _is_bullish(row: "pd.Series | dict") -> bool:
    """Detect BUY / STRONG BUY signal from any pipeline output row."""
    def _get(k: str) -> str:
        v = row.get(k, "") if isinstance(row, dict) else row.get(k, "")
        return str(v).upper().strip()

    signal       = _get("Signal")
    final_signal = _get("Final Signal")

    if "BUY" in signal or "BUY" in final_signal:
        return True

    grade = _get("Grade")
    if grade in ("A+", "A"):
        return True

    # Fallback: score-based
    fs = _sf(row.get("Final Score", 0) if isinstance(row, dict) else row.get("Final Score", 0), 0)
    return fs >= 62.0


def _is_strong_bullish(row: "pd.Series | dict") -> bool:
    """Detect STRONG BUY specifically."""
    def _get(k: str) -> str:
        v = row.get(k, "") if isinstance(row, dict) else row.get(k, "")
        return str(v).upper().strip()

    if "STRONG BUY" in _get("Signal") or "STRONG BUY" in _get("Final Signal"):
        return True
    if _get("Grade") == "A+":
        return True
    fs = _sf(row.get("Final Score", 0) if isinstance(row, dict) else row.get("Final Score", 0), 0)
    return fs >= 75.0


def _has_high_trap(row: "pd.Series | dict") -> bool:
    """Detect HIGH trap risk from pipeline output."""
    def _get(k: str) -> str:
        v = row.get(k, "") if isinstance(row, dict) else row.get(k, "")
        return str(v).upper().strip()
    return "HIGH" in _get("Trap Risk") or "TRAP" in _get("Final Signal")


# ═══════════════════════════════════════════════════════════════════════
# ── PUBLIC: PRELOAD ALL SECTOR STOCKS ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def preload_all_sectors(workers: int = 12) -> None:
    """
    Preload OHLCV data for every stock across all sectors into ALL_DATA.
    Call this once before calling build_sector_raw_rows().
    Uses the existing preload_all() utility — zero new API logic.
    """
    try:
        all_tickers: list[str] = []
        seen: set[str] = set()
        for tickers in INDEX_STOCK_MAP.values():
            for t in tickers:
                t_ns = t if t.endswith(".NS") else f"{t}.NS"
                if t_ns not in seen:
                    seen.add(t_ns)
                    all_tickers.append(t_ns)
        preload_all(all_tickers, period="6mo", workers=max(1, workers))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# ── PUBLIC: ANALYZE INDEX ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def analyze_index(sector_name: str) -> dict:
    """
    Download (and cache) index OHLCV for the given sector, then compute
    trend / RSI / returns / strength score.

    Returns
    -------
    dict with keys:
        trend          : "Bullish" | "Bearish" | "Sideways"
        strength_score : float  0–100
        rsi            : float
        ret_5d         : float  (%)
        ret_20d        : float  (%)
        ema_above      : bool   (EMA20 > EMA50)
        vol_trend      : str    STRONG / BUILDING / NORMAL / WEAK / N/A
        available      : bool
    """
    with _INDEX_CACHE_LOCK:
        # Include active TT cutoff in cache key so date changes force re-download
        _cache_key = f"{sector_name}::{_tt_ref_date()}"
        if _cache_key in _INDEX_CACHE:
            return _INDEX_CACHE[_cache_key]

    try:
        ticker = INDEX_TICKERS.get(sector_name)
        if not ticker:
            result = _index_fallback()
        else:
            df = _download_index_ohlcv(ticker, period="3mo")
            result = _compute_index_strength(df) if df is not None else _index_fallback()

        result["sector"] = sector_name
        result["index_ticker"] = INDEX_TICKERS.get(sector_name, "N/A")

        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE[_cache_key] = result

        return result
    except Exception:
        fb = _index_fallback()
        fb["sector"] = sector_name
        fb["index_ticker"] = INDEX_TICKERS.get(sector_name, "N/A")
        return fb


def clear_index_cache() -> None:
    """Clear cached index analyses (call to force fresh download)."""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()


# ═══════════════════════════════════════════════════════════════════════
# ── PUBLIC: BUILD SECTOR RAW ROWS ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def build_sector_raw_rows(sector_name: str, mode: int = 2) -> list[dict]:
    """
    Build raw indicator rows for all stocks in the given sector.
    Output is ready to pass directly into enhance_results(rows, mode).

    Parameters
    ----------
    sector_name : str   One of the keys in INDEX_STOCK_MAP.
    mode        : int   Strategy mode (1–6) for the "Mode" label column.

    Returns
    -------
    list[dict]   May be empty if no stock data is available.
    """
    try:
        tickers = INDEX_STOCK_MAP.get(sector_name, [])
        if not tickers:
            return []

        rows: list[dict] = []
        for t in tickers:
            try:
                t_ns = t if t.endswith(".NS") else f"{t}.NS"
                row = _build_stock_row(t_ns, mode)
                if row is not None:
                    rows.append(row)
            except Exception:
                continue

        return rows
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
# ── PUBLIC: COMPUTE SECTOR PREDICTION ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def compute_sector_prediction(
    sector_name: str,
    processed_df: pd.DataFrame,
    index_analysis: dict,
) -> dict:
    """
    Aggregate stock-level pipeline output into a sector-level prediction.

    Parameters
    ----------
    sector_name    : str           Name of the sector.
    processed_df   : pd.DataFrame  DataFrame after the full pipeline
                                   (enhance → grading → enhanced_logic → phase4).
    index_analysis : dict          Output from analyze_index(sector_name).

    Returns
    -------
    dict with keys:
        sector              : str
        total_stocks        : int
        bullish_count       : int
        bullish_pct         : float   (0–100)
        strong_bullish_pct  : float   (0–100)
        avg_score           : float   (0–100)
        avg_pred_score      : float   (0–100)
        avg_vol_strength    : float   (vol/avg ratio)
        trap_high_pct       : float   (% with HIGH trap risk)
        sector_direction    : str     "Bullish" | "Bearish" | "Sideways"
        bullish_probability : float   (0–100)
        tomorrow_prediction : str     "UP" | "DOWN" | "SIDEWAYS"
        confidence          : float   (0–100)
        is_fake_bullish     : bool
        index_contradicts   : bool
        top_stocks          : list[dict]   top 5 by score
    """
    _FALLBACK = {
        "sector":               sector_name,
        "total_stocks":         0,
        "bullish_count":        0,
        "bullish_pct":          50.0,
        "strong_bullish_pct":   0.0,
        "avg_score":            50.0,
        "avg_pred_score":       50.0,
        "avg_vol_strength":     1.0,
        "trap_high_pct":        0.0,
        "sector_direction":     "Sideways",
        "bullish_probability":  50.0,
        "tomorrow_prediction":  "SIDEWAYS",
        "confidence":           40.0,
        "is_fake_bullish":      False,
        "index_contradicts":    False,
        "top_stocks":           [],
    }

    try:
        if processed_df is None or not isinstance(processed_df, pd.DataFrame) or processed_df.empty:
            return _FALLBACK

        df = processed_df.copy()
        n  = len(df)

        # ── Per-stock classification ──────────────────────────────────
        bullish_flags: list[bool]       = []
        strong_bull:   list[bool]       = []
        scores:        list[float]      = []
        pred_scores:   list[float]      = []
        vol_ratios:    list[float]      = []
        trap_flags:    list[bool]       = []

        for idx in df.index:
            row = df.loc[idx]
            b   = _is_bullish(row)
            sb  = _is_strong_bullish(row)
            bullish_flags.append(b)
            strong_bull.append(sb)
            scores.append(_sf(row.get("Final Score",      50), 50))
            pred_scores.append(_sf(row.get("Prediction Score", 50), 50))
            vol_ratios.append(_sf(row.get("Vol / Avg",    1.0), 1.0))
            trap_flags.append(_has_high_trap(row))

        bullish_count      = sum(bullish_flags)
        bullish_pct        = (bullish_count / n * 100) if n > 0 else 50.0
        strong_bull_pct    = (sum(strong_bull) / n * 100) if n > 0 else 0.0
        avg_score          = float(np.mean(scores))          if scores else 50.0
        avg_pred_score     = float(np.mean(pred_scores))     if pred_scores else 50.0
        avg_vol            = float(np.mean(vol_ratios))      if vol_ratios else 1.0
        trap_high_pct      = (sum(trap_flags) / n * 100)     if n > 0 else 0.0

        # ── Sector direction ──────────────────────────────────────────
        if bullish_pct > 60 and avg_score > 65:
            sector_dir = "Bullish"
        elif bullish_pct < 40 and avg_score < 50:
            sector_dir = "Bearish"
        else:
            sector_dir = "Sideways"

        # ── Bullish probability (0–100) ───────────────────────────────
        bp = float(np.clip(
            0.40 * bullish_pct
            + 0.35 * avg_score
            + 0.15 * min(avg_vol * 40, 40)     # vol contribution capped at 40
            + 0.10 * strong_bull_pct,
            0.0, 100.0,
        ))

        # ── Adjustments ───────────────────────────────────────────────
        idx_rsi        = _sf(index_analysis.get("rsi", 50), 50)
        idx_ret5d      = _sf(index_analysis.get("ret_5d", 0), 0)
        idx_trend      = str(index_analysis.get("trend", "Sideways"))
        idx_strength   = _sf(index_analysis.get("strength_score", 50), 50)
        idx_available  = bool(index_analysis.get("available", False))

        # Overbought index → reduce confidence
        if idx_rsi > 70.0:
            bp -= 5.0

        # Weak volume → reduce score
        if avg_vol < 0.9:
            bp -= 5.0

        # Fake bullish detection: majority HIGH trap + index not confirming
        is_fake_bullish = (trap_high_pct > 50.0 and sector_dir == "Bullish")

        # Index contradiction: sector says bullish but index is bearish
        index_contradicts = (
            idx_available
            and sector_dir == "Bullish"
            and idx_trend == "Bearish"
        )
        if index_contradicts:
            bp -= 10.0

        bp = float(np.clip(bp, 0.0, 100.0))

        # ── Tomorrow prediction ───────────────────────────────────────
        # Combine: sector strength + index trend + momentum + RSI caution
        raw_conf = (
            0.35 * bp
            + 0.25 * idx_strength
            + 0.20 * float(np.clip(50.0 + idx_ret5d * 5, 0, 100))
            + 0.20 * float(np.clip(100 - abs(idx_rsi - 58) * 2, 0, 100))
        )
        confidence = float(np.clip(raw_conf, 0.0, 100.0))

        # Fake bullish → mark as SIDEWAYS and reduce confidence
        if is_fake_bullish:
            confidence = max(confidence - 15.0, 20.0)

        if confidence >= 60.0 and bp >= 60.0:
            tomorrow = "UP"
        elif confidence >= 55.0 and bp < 40.0:
            tomorrow = "DOWN"
        else:
            tomorrow = "SIDEWAYS"

        # ── Top 5 stocks ──────────────────────────────────────────────
        df_sorted = df.copy()
        if "Final Score" in df_sorted.columns:
            df_sorted = df_sorted.sort_values("Final Score", ascending=False)

        top_stocks: list[dict] = []
        for _, r in df_sorted.head(5).iterrows():
            sym    = str(r.get("Symbol", r.get("Ticker", "?")))
            fscore = _sf(r.get("Final Score", 50), 50)
            sig    = str(r.get("Signal", r.get("Final Signal", "WATCH")))
            grade  = str(r.get("Grade", ""))
            conf   = _sf(r.get("Confidence", 50), 50)
            top_stocks.append({
                "symbol":  sym,
                "score":   round(fscore, 1),
                "signal":  sig,
                "grade":   grade,
                "conf":    round(conf, 1),
            })

        return {
            "sector":               sector_name,
            "total_stocks":         n,
            "bullish_count":        bullish_count,
            "bullish_pct":          round(bullish_pct, 1),
            "strong_bullish_pct":   round(strong_bull_pct, 1),
            "avg_score":            round(avg_score, 1),
            "avg_pred_score":       round(avg_pred_score, 1),
            "avg_vol_strength":     round(avg_vol, 2),
            "trap_high_pct":        round(trap_high_pct, 1),
            "sector_direction":     sector_dir,
            "bullish_probability":  round(bp, 1),
            "tomorrow_prediction":  tomorrow,
            "confidence":           round(confidence, 1),
            "is_fake_bullish":      is_fake_bullish,
            "index_contradicts":    index_contradicts,
            "top_stocks":           top_stocks,
        }

    except Exception:
        _FALLBACK["sector"] = sector_name
        return _FALLBACK


# ═══════════════════════════════════════════════════════════════════════
# ── PUBLIC: COMPUTE OVERALL MARKET ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def compute_overall_market(all_sector_results: dict[str, dict]) -> dict:
    """
    Combine sector predictions into an overall market bias using
    the weights defined in SECTOR_WEIGHTS.

    Parameters
    ----------
    all_sector_results : dict[str, dict]
        Mapping sector_name → result from compute_sector_prediction().

    Returns
    -------
    dict with keys:
        overall_prediction  : "BULLISH" | "BEARISH" | "SIDEWAYS"
        confidence          : float  (0–100)
        weighted_score      : float  (0–100)
        strongest_sector    : str
        weakest_sector      : str
        sectors             : dict[str, dict]
        top_sectors         : list[str]  (top 3 by bullish_probability)
        weak_sectors        : list[str]  (bottom 2 by bullish_probability)
    """
    _EMPTY = {
        "overall_prediction": "SIDEWAYS",
        "confidence":          50.0,
        "weighted_score":      50.0,
        "strongest_sector":    "N/A",
        "weakest_sector":      "N/A",
        "sectors":             {},
        "top_sectors":         [],
        "weak_sectors":        [],
    }

    try:
        if not all_sector_results:
            return _EMPTY

        weighted_bp    = 0.0
        weighted_conf  = 0.0
        total_weight   = 0.0

        scored: list[tuple[float, str]] = []   # (bullish_prob, sector_name)

        for sector, result in all_sector_results.items():
            w  = SECTOR_WEIGHTS.get(sector, 0.0)
            bp = _sf(result.get("bullish_probability", 50), 50)
            c  = _sf(result.get("confidence",          50), 50)
            weighted_bp   += w * bp
            weighted_conf += w * c
            total_weight  += w
            if w > 0:
                scored.append((bp, sector))

        if total_weight > 0:
            wbp  = weighted_bp   / total_weight
            wcon = weighted_conf / total_weight
        else:
            wbp  = 50.0
            wcon = 50.0

        # ── Overall prediction ────────────────────────────────────────
        if wbp >= 60.0 and wcon >= 55.0:
            overall = "BULLISH"
        elif wbp <= 40.0 and wcon >= 50.0:
            overall = "BEARISH"
        else:
            overall = "SIDEWAYS"

        # ── Strongest / weakest ───────────────────────────────────────
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

        strongest = scored_sorted[0][1]  if scored_sorted else "N/A"
        weakest   = scored_sorted[-1][1] if scored_sorted else "N/A"

        top_sectors  = [s for _, s in scored_sorted[:3]]
        weak_sectors = [s for _, s in scored_sorted[-2:]]

        return {
            "overall_prediction": overall,
            "confidence":          round(wcon, 1),
            "weighted_score":      round(wbp, 1),
            "strongest_sector":    strongest,
            "weakest_sector":      weakest,
            "sectors":             all_sector_results,
            "top_sectors":         top_sectors,
            "weak_sectors":        weak_sectors,
        }

    except Exception:
        return _EMPTY


# ═══════════════════════════════════════════════════════════════════════
# ▼▼▼  UPGRADE EXTENSIONS  —  ADDITIVE ONLY  —  DO NOT MODIFY ABOVE  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════
#
# UPGRADE 1 : FULL_INDEX_STOCK_MAP  (20–30 stocks per sector)
# UPGRADE 2 : Market-cap weighting  (static dict + log-normalisation)
# UPGRADE 3 : Sector dominance logic (Bank boost / Nifty contradiction)
# UPGRADE 4 : Signal quality filter  (WEAK BULLISH tier, RSI / vol adj.)
# UPGRADE 5 : Enhanced output fields (new keys appended to result dicts)
#
# All new functions are suffixed _enhanced / _full / _weighted so they
# never shadow the originals.  Original functions are called internally;
# their return values are only extended, never mutated.
# ═══════════════════════════════════════════════════════════════════════

import math   # for log weight — stdlib, no new dependency


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 1 — FULL_INDEX_STOCK_MAP  (expanded coverage, backward-compat)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Original INDEX_STOCK_MAP is UNCHANGED above.
# FULL_INDEX_STOCK_MAP is the new, expanded version used by the
# *_enhanced / *_full wrappers only.

FULL_INDEX_STOCK_MAP: dict[str, list[str]] = {
    "Nifty 50": [
        # Original 30 + 1 additional large-cap
        "RELIANCE", "HDFCBANK", "BHARTIARTL", "SBIN", "TCS",
        "ICICIBANK", "INFY", "BAJFINANCE", "LT", "HINDUNILVR",
        "SUNPHARMA", "MARUTI", "HCLTECH", "M&M", "AXISBANK",
        "ITC", "TITAN", "ONGC", "KOTAKBANK", "NTPC",
        "ADANIPORTS", "ULTRACEMCO", "BAJAJFINSV", "BAJAJ-AUTO", "TATASTEEL",
        "ADANIENT", "HINDALCO", "WIPRO", "EICHERMOT", "SBILIFE",
        "GRASIM",
    ],
    "Nifty Next 50": [
        "ADANIPOWER", "DMART", "VEDL", "HAL", "HINDZINC",
        "IOC", "TVSMOTOR", "DIVISLAB", "TATAMOTORS", "ADANIGREEN",
        "HDFCAMC", "VARUNBEV", "TORNTPHARM", "PFC", "UNIONBANK",
        "BRITANNIA", "ABB", "PIDILITIND", "DLF", "BANKBARODA",
        "CUMMINSIND", "MUTHOOTFIN", "LTIM", "TATAPOWER", "BPCL",
        "IRFC", "PNB", "SOLARINDS", "JSPL", "SIEMENS",
    ],
    "Nifty Bank": [
        "HDFCBANK", "SBIN", "ICICIBANK", "AXISBANK", "KOTAKBANK",
        "UNIONBANK", "BANKBARODA", "PNB", "CANBK", "FEDERALBNK",
        "AUBANK", "INDUSINDBK", "YESBANK", "IDFCFIRSTB",
    ],
    "Nifty IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "PERSISTENT", "OFSS", "MPHASIS", "COFORGE",
        "HEXAWARE", "TATAELXSI", "LTTS", "KPIT", "SONATSOFTW",
    ],
    "Nifty Auto": [
        "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR",
        "MOTHERSON", "HEROMOTOCO", "BOSCHLTD", "TATAMOTORS", "ASHOKLEY",
        "BHARATFORG", "UNOMINDA", "TIINDIA", "SONACOMS", "EXIDEIND",
        "MRF", "APOLLOTYRE", "BALKRISIND", "ENDURANCE", "ESCORTS",
    ],
    "Nifty Pharma": [
        "SUNPHARMA", "DIVISLAB", "TORNTPHARM", "LUPIN", "DRREDDY",
        "CIPLA", "ZYDUSLIFE", "MANKIND", "AUROPHARMA", "ALKEM",
        "GLENMARK", "BIOCON", "LAURUSLABS", "ABBOTINDIA", "IPCALAB",
        "AJANTPHARM", "JBCHEPHARM", "GLAND", "SYNGENE", "NATCOPHARM",
        "GRANULES", "ERIS", "WOCKPHARMA", "LALPATHLAB", "METROPOLIS",
    ],
    "Nifty FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "VARUNBEV", "BRITANNIA",
        "TATACONSUM", "GODREJCP", "MARICO", "USL", "DABUR",
        "PATANJALI", "COLPAL", "UBL", "RADICO", "EMAMILTD",
        "GILLETTE", "ADANIWILMAR", "HATSUN",
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 2 — APPROXIMATE MARKET CAPS (₹ Crores, static, ~Q1 2026)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source: NSE / Screener public data.  Used ONLY for log-weight
# normalisation — not displayed as exact values.
# Default for unknown tickers: 10_000 Cr (mid-small cap fallback).

APPROX_MKTCAP_CR: dict[str, float] = {
    # ── Nifty 50 heavyweights ─────────────────────────────────────────
    "RELIANCE":   1_827_560, "HDFCBANK":  1_155_888, "BHARTIARTL": 1_090_482,
    "SBIN":         940_046, "TCS":         886_685, "ICICIBANK":    870_526,
    "INFY":         527_551, "BAJFINANCE":  514_506, "LT":           497_030,
    "HINDUNILVR":   485_261, "SUNPHARMA":   406_351, "MARUTI":       397_122,
    "HCLTECH":      380_510, "M&M":         374_514, "AXISBANK":     372_328,
    "ITC":          366_925, "TITAN":       363_744, "ONGC":         361_306,
    "KOTAKBANK":    356_083, "NTPC":        348_741, "ADANIPORTS":   317_393,
    "ULTRACEMCO":   312_949, "BAJAJFINSV":  263_643, "BAJAJ-AUTO":   248_765,
    "TATASTEEL":    243_005, "ADANIENT":    237_870, "HINDALCO":     203_261,
    "WIPRO":        200_614, "EICHERMOT":   187_266, "SBILIFE":      179_555,
    "GRASIM":       176_608,
    # ── Nifty Next 50 ────────────────────────────────────────────────
    "ADANIPOWER":   308_497, "DMART":       283_876, "VEDL":         268_898,
    "HAL":          246_577, "HINDZINC":    217_921, "IOC":          189_408,
    "TVSMOTOR":     161_121, "DIVISLAB":    155_472, "TATAMOTORS":   142_911,
    "ADANIGREEN":   140_998, "HDFCAMC":     139_066, "VARUNBEV":     136_535,
    "TORNTPHARM":   134_945, "PFC":         132_747, "UNIONBANK":    131_290,
    "BRITANNIA":    131_081, "ABB":         130_154, "PIDILITIND":   129_623,
    "DLF":          129_273, "BANKBARODA":  129_057, "CUMMINSIND":   128_801,
    "MUTHOOTFIN":   127_619, "LTIM":        127_608, "TATAPOWER":    123_021,
    "BPCL":         118_658, "IRFC":        117_682, "PNB":          116_711,
    "SOLARINDS":    115_665, "JSPL":        114_107, "SIEMENS":      170_000,
    # ── Banking ───────────────────────────────────────────────────────
    "CANBK":         111_070, "INDUSINDBK":   60_669, "YESBANK":      56_076,
    "IDFCFIRSTB":    51_799, "FEDERALBNK":   65_439, "AUBANK":       64_968,
    # ── IT ────────────────────────────────────────────────────────────
    "TECHM":         141_290, "PERSISTENT":   82_504, "OFSS":         60_859,
    "MPHASIS":        42_156, "COFORGE":      40_750, "HEXAWARE":     27_114,
    "TATAELXSI":      19_500, "LTTS":         35_234, "KPIT":         19_102,
    "SONATSOFTW":     61_280,
    # ── Auto ──────────────────────────────────────────────────────────
    "HEROMOTOCO":   100_275, "BOSCHLTD":     94_778, "ASHOKLEY":     87_192,
    "BHARATFORG":    78_531, "MOTHERSON":   112_732, "MRF":          53_604,
    "APOLLOTYRE":    30_000, "BALKRISIND":   45_000, "UNOMINDA":     59_209,
    "TIINDIA":        49_691, "SONACOMS":    30_878, "EXIDEIND":     25_436,
    "ENDURANCE":      25_000, "ESCORTS":     35_000, "CEAT":         12_000,
    # ── Pharma ───────────────────────────────────────────────────────
    "DRREDDY":      101_603, "CIPLA":        96_320, "LUPIN":       103_985,
    "AUROPHARMA":    77_462, "ALKEM":        62_867, "BIOCON":       57_145,
    "GLENMARK":      59_031, "LAURUSLABS":   56_302, "ZYDUSLIFE":    86_964,
    "MANKIND":        82_504, "ABBOTINDIA":   55_856, "IPCALAB":     37_759,
    "AJANTPHARM":    34_872, "JBCHEPHARM":   31_488, "GLAND":        27_821,
    "SYNGENE":        15_678, "NATCOPHARM":   8_500, "GRANULES":      7_500,
    "ERIS":           12_800, "WOCKPHARMA":  20_661, "LALPATHLAB":   18_000,
    "METROPOLIS":     10_000,
    # ── FMCG ─────────────────────────────────────────────────────────
    "NESTLEIND":    228_100, "TATACONSUM":  101_826, "GODREJCP":    100_288,
    "MARICO":        96_693, "USL":          88_420, "DABUR":        72_349,
    "PATANJALI":     50_183, "COLPAL":       49_093, "UBL":          39_230,
    "RADICO":        35_320, "EMAMILTD":     17_189, "ADANIWILMAR":  66_000,
    "GILLETTE":      25_000, "HATSUN":       88_000,
    # ── Others (common large-caps) ────────────────────────────────────
    "POWERGRID":    200_000, "COALINDIA":   250_000, "JSWSTEEL":    200_000,
    "TATACHEM":      30_000, "GAIL":        130_000, "HINDPETRO":    60_000,
    "IRCTC":         80_000, "HAVELLS":     100_000, "POLYCAB":      70_000,
    "DIXON":          50_000, "BEL":        150_000, "BHARATDYNAM":  80_000,
    "ASIANPAINT":   230_000, "BERGEPAINT":   65_000, "CHOLAFIN":     80_000,
    "TRENT":          70_000, "NAUKRI":      60_000, "PIIND":        30_000,
}

_DEFAULT_MKTCAP_CR = 10_000.0   # fallback for any ticker not in dict


def get_mktcap_cr(ticker: str) -> float:
    """
    Return approximate market cap in ₹ Crores for the given NSE ticker.
    Uses the static APPROX_MKTCAP_CR dict — zero API calls.
    Returns _DEFAULT_MKTCAP_CR (10,000 Cr) for unknown tickers.

    Parameters
    ----------
    ticker : str   With or without .NS suffix.

    Returns
    -------
    float   Market cap in Crores.
    """
    sym = str(ticker).upper().strip().replace(".NS", "")
    return float(APPROX_MKTCAP_CR.get(sym, _DEFAULT_MKTCAP_CR))


def _log_weights(symbols: list[str]) -> list[float]:
    """
    Compute log-normalised market-cap weights for a list of symbols.
    weight_i = log(mktcap_i + 1)  →  normalised so sum = 1.
    Falls back to equal weight if all log values are zero.

    Returns
    -------
    list[float]  Same length as symbols.  Each value in [0, 1].
    """
    if not symbols:
        return []
    raw = [math.log(get_mktcap_cr(s) + 1.0) for s in symbols]
    total = sum(raw)
    if total <= 0:
        n = len(symbols)
        return [1.0 / n] * n
    return [r / total for r in raw]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 3+4+5 — ENHANCED SECTOR PREDICTION WRAPPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_sector_prediction_enhanced(
    sector_name: str,
    processed_df: "pd.DataFrame",
    index_analysis: dict,
) -> dict:
    """
    Wrapper around compute_sector_prediction() that adds:

    UPGRADE 2 — Market-cap weighted metrics
        weighted_bullish_pct      : float  (mktcap-weighted bullish %)
        market_cap_weighted_score : float  (mktcap-weighted avg Final Score)
        weighted_pred_score       : float  (mktcap-weighted Prediction Score)

    UPGRADE 4 — Signal quality filter
        signal_quality : "HIGH" | "MEDIUM" | "WEAK_BULLISH" | "LOW"
            HIGH        → <20% trap HIGH, RSI index ≤70, avg vol ≥1.0
            WEAK_BULLISH → >50% trap HIGH AND sector was Bullish
            MEDIUM      → mixed conditions
            LOW         → majority trap HIGH or index RSI >75

    UPGRADE 5 — Additional output fields
        dominance_adjustment : float  (applied by compute_overall_market_enhanced)

    All existing fields from compute_sector_prediction() are preserved
    verbatim.  New fields are appended to the returned dict.
    """
    # ── Step 1: call original (untouched) ────────────────────────────
    base = compute_sector_prediction(sector_name, processed_df, index_analysis)

    try:
        if processed_df is None or not isinstance(processed_df, pd.DataFrame) or processed_df.empty:
            base.update({
                "weighted_bullish_pct":       base.get("bullish_pct", 50.0),
                "market_cap_weighted_score":  base.get("avg_score", 50.0),
                "weighted_pred_score":        base.get("avg_pred_score", 50.0),
                "signal_quality":             "MEDIUM",
                "dominance_adjustment":       0.0,
            })
            return base

        df = processed_df

        # ── Step 2: collect symbols + weights ─────────────────────────
        symbols: list[str] = []
        for idx in df.index:
            row = df.loc[idx]
            sym = str(row.get("Symbol", row.get("Ticker", ""))).strip().upper()
            symbols.append(sym)

        weights = _log_weights(symbols)

        # ── Step 3: weighted aggregations ─────────────────────────────
        w_bull_sum  = 0.0
        w_score_sum = 0.0
        w_pred_sum  = 0.0

        for i, idx in enumerate(df.index):
            row = df.loc[idx]
            w   = weights[i] if i < len(weights) else 0.0
            b   = 1.0 if _is_bullish(row) else 0.0
            fs  = _sf(row.get("Final Score",      50), 50)
            ps  = _sf(row.get("Prediction Score", 50), 50)
            w_bull_sum  += w * b
            w_score_sum += w * fs
            w_pred_sum  += w * ps

        # Convert weighted bullish to percentage
        w_bull_pct    = round(float(np.clip(w_bull_sum * 100.0, 0.0, 100.0)), 1)
        w_score       = round(float(np.clip(w_score_sum, 0.0, 100.0)), 1)
        w_pred        = round(float(np.clip(w_pred_sum, 0.0, 100.0)), 1)

        # ── Step 4: signal quality ─────────────────────────────────────
        trap_pct   = _sf(base.get("trap_high_pct",   0.0), 0.0)
        idx_rsi    = _sf(index_analysis.get("rsi",  50.0), 50.0)
        avg_vol    = _sf(base.get("avg_vol_strength", 1.0), 1.0)
        sec_dir    = str(base.get("sector_direction", "Sideways"))

        if trap_pct > 50.0 and sec_dir == "Bullish":
            signal_quality = "WEAK_BULLISH"
        elif trap_pct > 50.0 or idx_rsi > 75.0:
            signal_quality = "LOW"
        elif trap_pct < 20.0 and idx_rsi <= 70.0 and avg_vol >= 1.0:
            signal_quality = "HIGH"
        else:
            signal_quality = "MEDIUM"

        # ── Step 5: append new fields ──────────────────────────────────
        base["weighted_bullish_pct"]       = w_bull_pct
        base["market_cap_weighted_score"]  = w_score
        base["weighted_pred_score"]        = w_pred
        base["signal_quality"]             = signal_quality
        base["dominance_adjustment"]       = 0.0   # set by overall market fn

        return base

    except Exception:
        # Fail-safe: return original with neutral new fields
        base.setdefault("weighted_bullish_pct",      base.get("bullish_pct", 50.0))
        base.setdefault("market_cap_weighted_score", base.get("avg_score", 50.0))
        base.setdefault("weighted_pred_score",       base.get("avg_pred_score", 50.0))
        base.setdefault("signal_quality",            "MEDIUM")
        base.setdefault("dominance_adjustment",      0.0)
        return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 3+5 — ENHANCED OVERALL MARKET WRAPPER  (dominance logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Priority weight multipliers for dominance boost/penalty
_DOMINANCE_PRIORITY: dict[str, float] = {
    "Nifty Bank":    2.0,   # VERY HIGH  — banking drives broader market
    "Nifty 50":      1.5,   # HIGH       — broad market benchmark
    "Nifty IT":      1.2,   # MEDIUM     — significant but sector-specific
}
_DEFAULT_PRIORITY  = 1.0   # NORMAL for all other sectors

# Bank influence thresholds
_BANK_STRONG_BULL_THRESHOLD  = 65.0   # bp >= this → add boost
_BANK_STRONG_BEAR_THRESHOLD  = 40.0   # bp <= this → add penalty
_BANK_DOMINANCE_MAGNITUDE    = 12.5   # max ±% shift on weighted score


def compute_overall_market_enhanced(
    all_sector_results: dict[str, dict],
) -> dict:
    """
    Wrapper around compute_overall_market() that adds:

    UPGRADE 3 — Sector dominance logic
        • Bank strongly bullish  → weighted_score += up to +12.5
        • Bank strongly bearish  → weighted_score -= up to -12.5
        • Nifty 50 contradicts majority sectors → confidence reduced

    UPGRADE 5 — Additional output fields
        bank_influence         : float  (signed adjustment from Bank sector)
        market_pressure        : str    "BULLISH_PRESSURE" | "BEARISH_PRESSURE" | "NEUTRAL"
        dominant_sector_score  : float  (highest weighted bullish_probability)

    All existing fields from compute_overall_market() are preserved.
    """
    # ── Step 1: call original (untouched) ────────────────────────────
    base = compute_overall_market(all_sector_results)

    try:
        if not all_sector_results:
            base.update({
                "bank_influence":        0.0,
                "market_pressure":       "NEUTRAL",
                "dominant_sector_score": 50.0,
            })
            return base

        # ── Step 2: Bank dominance adjustment ────────────────────────
        bank_result = all_sector_results.get("Nifty Bank", {})
        bank_bp     = _sf(bank_result.get(
            "weighted_bullish_pct",        # prefer weighted if enhanced was used
            bank_result.get("bullish_probability", 50.0)
        ), 50.0)

        bank_influence = 0.0
        if bank_bp >= _BANK_STRONG_BULL_THRESHOLD:
            # Scale boost proportional to how far above threshold
            excess = (bank_bp - _BANK_STRONG_BULL_THRESHOLD) / (100.0 - _BANK_STRONG_BULL_THRESHOLD)
            bank_influence = +float(np.clip(excess * _BANK_DOMINANCE_MAGNITUDE, 0.0, _BANK_DOMINANCE_MAGNITUDE))
        elif bank_bp <= _BANK_STRONG_BEAR_THRESHOLD:
            # Scale penalty proportional to how far below threshold
            deficit = (_BANK_STRONG_BEAR_THRESHOLD - bank_bp) / _BANK_STRONG_BEAR_THRESHOLD
            bank_influence = -float(np.clip(deficit * _BANK_DOMINANCE_MAGNITUDE, 0.0, _BANK_DOMINANCE_MAGNITUDE))

        # Apply bank influence to weighted score
        adjusted_score = float(np.clip(
            _sf(base.get("weighted_score", 50.0), 50.0) + bank_influence,
            0.0, 100.0,
        ))

        # ── Step 3: Nifty 50 vs majority contradiction check ──────────
        nifty50_result = all_sector_results.get("Nifty 50", {})
        nifty50_dir    = str(nifty50_result.get("sector_direction", "Sideways"))

        # Count how many sectors (excl. Nifty 50) agree on direction
        other_dirs = [
            str(r.get("sector_direction", "Sideways"))
            for s, r in all_sector_results.items()
            if s != "Nifty 50"
        ]
        majority_bullish = sum(1 for d in other_dirs if d == "Bullish") > len(other_dirs) / 2
        majority_bearish = sum(1 for d in other_dirs if d == "Bearish") > len(other_dirs) / 2

        nifty_contradicts = (
            (nifty50_dir == "Bearish" and majority_bullish) or
            (nifty50_dir == "Bullish" and majority_bearish)
        )

        # Reduce confidence if Nifty 50 contradicts sector majority
        adjusted_conf = _sf(base.get("confidence", 50.0), 50.0)
        if nifty_contradicts:
            adjusted_conf = float(np.clip(adjusted_conf - 8.0, 0.0, 100.0))

        # ── Step 4: Re-derive overall prediction from adjusted score ──
        if adjusted_score >= 60.0 and adjusted_conf >= 55.0:
            adjusted_prediction = "BULLISH"
        elif adjusted_score <= 40.0 and adjusted_conf >= 50.0:
            adjusted_prediction = "BEARISH"
        else:
            adjusted_prediction = "SIDEWAYS"

        # ── Step 5: Dominant sector score ────────────────────────────
        dominant_score = max(
            (
                _sf(r.get("weighted_bullish_pct",
                          r.get("bullish_probability", 50.0)), 50.0)
                * _DOMINANCE_PRIORITY.get(s, _DEFAULT_PRIORITY)
            )
            for s, r in all_sector_results.items()
        ) if all_sector_results else 50.0
        dominant_score = float(np.clip(dominant_score, 0.0, 100.0))

        # ── Step 6: Market pressure label ────────────────────────────
        if bank_influence > 5.0 and adjusted_score >= 58.0:
            market_pressure = "BULLISH_PRESSURE"
        elif bank_influence < -5.0 and adjusted_score <= 42.0:
            market_pressure = "BEARISH_PRESSURE"
        else:
            market_pressure = "NEUTRAL"

        # ── Step 7: Append new + updated fields ──────────────────────
        # Update the score/prediction/confidence with dominance-adjusted values
        base["weighted_score"]      = round(adjusted_score, 1)
        base["confidence"]          = round(adjusted_conf, 1)
        base["overall_prediction"]  = adjusted_prediction
        base["bank_influence"]      = round(bank_influence, 2)
        base["market_pressure"]     = market_pressure
        base["dominant_sector_score"] = round(dominant_score, 1)
        base["nifty_contradicts_majority"] = nifty_contradicts

        return base

    except Exception:
        base.setdefault("bank_influence",        0.0)
        base.setdefault("market_pressure",       "NEUTRAL")
        base.setdefault("dominant_sector_score", 50.0)
        base.setdefault("nifty_contradicts_majority", False)
        return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPGRADE 1 — FULL MAP BUILD HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_sector_raw_rows_full(sector_name: str, mode: int = 2) -> list[dict]:
    """
    Like build_sector_raw_rows() but uses FULL_INDEX_STOCK_MAP
    (20–30 stocks per sector) instead of the original 10–15 stock map.
    Backward-compatible: falls back to original map if sector not in FULL map.
    """
    try:
        tickers = FULL_INDEX_STOCK_MAP.get(sector_name) or INDEX_STOCK_MAP.get(sector_name, [])
        if not tickers:
            return []

        rows: list[dict] = []
        for t in tickers:
            try:
                t_ns = t if t.endswith(".NS") else f"{t}.NS"
                row  = _build_stock_row(t_ns, mode)
                if row is not None:
                    rows.append(row)
            except Exception:
                continue
        return rows
    except Exception:
        return []


def preload_all_sectors_full(workers: int = 12) -> None:
    """
    Like preload_all_sectors() but covers FULL_INDEX_STOCK_MAP stocks.
    Call this before build_sector_raw_rows_full() for zero-API runs.
    """
    try:
        all_tickers: list[str] = []
        seen: set[str] = set()
        for tickers in FULL_INDEX_STOCK_MAP.values():
            for t in tickers:
                t_ns = t if t.endswith(".NS") else f"{t}.NS"
                if t_ns not in seen:
                    seen.add(t_ns)
                    all_tickers.append(t_ns)
        preload_all(all_tickers, period="6mo", workers=max(1, workers))
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DASHBOARD-SPECIFIC BROAD SECTOR HELPERS
# Used by the interactive Sector Screener dashboard to scan wider baskets
# than the compact index-only maps above.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DASHBOARD_SECTOR_ORDER: list[str] = [
    "Nifty 50",
    "Bank",
    "IT",
    "Auto",
    "Pharma",
    "FMCG",
    "Overall",
]

DASHBOARD_TO_INDEX_SECTOR: dict[str, str] = {
    "Nifty 50": "Nifty 50",
    "Bank": "Nifty Bank",
    "IT": "Nifty IT",
    "Auto": "Nifty Auto",
    "Pharma": "Nifty Pharma",
    "FMCG": "Nifty FMCG",
    "Overall": "Nifty 50",
}

DASHBOARD_SECTOR_DESCRIPTIONS: dict[str, str] = {
    "Nifty 50": "Broad large-cap basket with full Nifty 50 coverage",
    "Bank": "Banks, NBFCs, insurers, AMCs and finance-heavy names",
    "IT": "IT services, software, platforms and digital tech names",
    "Auto": "Auto OEMs, EVs, tyres, batteries and auto ancillaries",
    "Pharma": "Pharma, diagnostics, hospitals and healthcare names",
    "FMCG": "Staples, foods, beverages, personal care and QSR names",
    "Overall": "Weighted scan across all sector baskets",
}

_DASHBOARD_SOURCE_BUCKETS: dict[str, tuple[str, ...]] = {
    "Nifty 50": ("Nifty 50",),
    "Bank": ("Nifty Bank", "BANKING", "NBFC_FINANCE"),
    "IT": ("Nifty IT", "IT"),
    "Auto": ("Nifty Auto", "AUTO"),
    "Pharma": ("Nifty Pharma", "PHARMA"),
    "FMCG": ("Nifty FMCG", "FMCG"),
}

_DASHBOARD_MANUAL_ADDONS: dict[str, list[str]] = {
    "Nifty 50": [
        "APOLLOHOSP", "ASIANPAINT", "BEL", "BRITANNIA", "CIPLA",
        "COALINDIA", "DRREDDY", "HDFCLIFE", "HEROMOTOCO", "INDUSINDBK",
        "JIOFIN", "JSWSTEEL", "NESTLEIND", "POWERGRID", "SHRIRAMFIN",
        "TATACONSUM", "TECHM", "TRENT",
    ],
    "Bank": [
        "360ONE", "ABSLAMC", "ANGELONE", "BSE", "CAMS",
        "CDSL", "CHOLAHLDNG", "CREDITACC", "CSBBANK", "EDELWEISS",
        "EQUITASBNK", "FIVESTAR", "HOMEFIRST", "IIFL", "INDIASHLTR",
        "JIOFIN", "JMFINANCIL", "KARURVYSYA", "KFINTECH", "MFSL",
        "MOTILALOFS", "NIVABUPA", "PAYTM", "POONAWALLA", "POLICYBZR",
        "SBICARD", "SOUTHBANK", "TMB", "UJJIVANSFB", "UTIAMC",
    ],
    "IT": [
        "BBOX", "CARTRADE", "EASEMYTRIP", "FIRSTSOURCE", "HAPPSTMNDS",
        "INDIAMART", "INTELLECT", "JUSTDIAL", "MAPMYINDIA", "MINDTECK",
        "NAUKRI", "NETWEB", "NYKAA", "PAYTM", "RAMCOSYS",
        "ROUTE", "SAGILITY", "SASKEN", "SUBEXLTD", "TANLA",
        "TATACOMM", "TEJASNET", "ZENSARTECH", "ZOMATO",
    ],
    "Auto": [
        "AUTOAXLES", "FIEMIND", "GNA", "JBMA", "JTEKTINDIA",
        "MUNJALAU", "MUNJALSHOW", "OLECTRA", "RICOAUTO", "SCHAEFFLER",
        "SHARDAMOTR", "SMLISUZU", "SUBROS", "SUNDRMBRAK", "SUNDRMFAST",
        "TALBROAUTO", "TIMKEN", "TUBEINVEST", "TVSHLTD", "TVSSCS",
        "VARROC", "VSTTILLERS", "WHEELS",
    ],
    "Pharma": [
        "AARTIDRUGS", "AARTIPHARM", "ALEMBICLTD", "APLLTD", "ASTRAZEN",
        "BLISSGVS", "CAPLIPOINT", "FDC", "GUFICBIO", "HIKAL",
        "INDOCO", "JUBLPHARMA", "MARKSANS", "MEDANTA", "MOREPENLAB",
        "NEULANDLAB", "ORCHPHARMA", "PANACEABIO", "PFIZER", "RAINBOW",
        "SANOFI", "SEQUENT", "SMSPHARMA", "SOLARA", "STRIDES",
        "SUVENPHAR", "THEMISMED", "THYROCARE", "UNICHEMLAB", "VIJAYA",
    ],
    "FMCG": [
        "AWL", "BBTC", "BECTORFOOD", "BIKAJI", "CCL",
        "DEVYANI", "DODLA", "GODFRYPHLP", "HERITGFOOD", "HONASA",
        "JUBLFOOD", "JYOTHYLAB", "KRBL", "LTFOODS", "PGHH",
        "RENUKA", "SAPPHIRE", "TASTYBITE", "VBL", "WESTLIFE",
    ],
}


def _dashboard_normalize_symbol(symbol: object) -> str:
    """Return a clean NSE symbol without the .NS suffix."""
    return str(symbol).upper().strip().replace(".NS", "")


def _dashboard_unique_symbols(symbols: list[object]) -> list[str]:
    """Preserve order while removing duplicates / empty values."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in symbols:
        sym = _dashboard_normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        ordered.append(sym)
    return ordered


def _load_sector_master_stocks() -> dict[str, list[str]]:
    """Load the static sector master map once per session."""
    global _SECTOR_MASTER_STOCKS_CACHE

    with _SECTOR_MASTER_STOCKS_LOCK:
        if _SECTOR_MASTER_STOCKS_CACHE is not None:
            return _SECTOR_MASTER_STOCKS_CACHE

        sector_stocks_raw: dict = {}
        try:
            from strategy_engines.sector_master import SECTOR_STOCKS as _sector_stocks  # type: ignore[import]
        except Exception:
            try:
                from sector_master import SECTOR_STOCKS as _sector_stocks  # type: ignore[import]
            except Exception:
                _sector_stocks = {}

        try:
            sector_stocks_raw = dict(_sector_stocks)
        except Exception:
            sector_stocks_raw = {}

        _SECTOR_MASTER_STOCKS_CACHE = {
            str(bucket): _dashboard_unique_symbols(list(symbols))
            for bucket, symbols in sector_stocks_raw.items()
        }
        return _SECTOR_MASTER_STOCKS_CACHE


def _compute_dashboard_sector_stocks(sector_name: str) -> tuple[str, ...]:
    """Build one dashboard basket once, then let callers reuse the cached tuple."""
    name = str(sector_name).strip()
    if not name:
        return ()

    if name == "Overall":
        combined: list[str] = []
        for label in get_dashboard_sector_labels(include_overall=False):
            combined.extend(get_dashboard_sector_stocks(label))
        return tuple(_dashboard_unique_symbols(combined))

    sector_stocks = _load_sector_master_stocks()
    symbols: list[str] = []
    for bucket in _DASHBOARD_SOURCE_BUCKETS.get(name, ()):
        symbols.extend(FULL_INDEX_STOCK_MAP.get(bucket, []))
        symbols.extend(INDEX_STOCK_MAP.get(bucket, []))
        symbols.extend(sector_stocks.get(bucket, []))

    symbols.extend(_DASHBOARD_MANUAL_ADDONS.get(name, []))
    return tuple(_dashboard_unique_symbols(symbols))


def preload_dashboard_sector_data(
    sector_name: str,
    workers: int = 12,
) -> list[str]:
    """
    Preload one dashboard basket once into ALL_DATA and return the deduplicated symbols.

    This keeps sector scans and the overall scan on the same shared dataset.
    """
    try:
        symbols = get_dashboard_sector_stocks(sector_name)
        if not symbols:
            return []

        tickers_ns = [sym if sym.endswith(".NS") else f"{sym}.NS" for sym in symbols]
        missing = [ticker_ns for ticker_ns in tickers_ns if ticker_ns not in ALL_DATA]
        if missing:
            preload_all(missing, period="6mo", workers=max(1, int(workers)))
        return symbols
    except Exception:
        return []


def get_dashboard_data_signature(tickers: list[str]) -> tuple[tuple[str, tuple[int, str, float, float]], ...]:
    """Return a compact data signature for a ticker universe already present in ALL_DATA."""
    symbols = _dashboard_unique_symbols(tickers)
    signature: list[tuple[str, tuple[int, str, float, float]]] = []
    for sym in symbols:
        ticker_ns = sym if sym.endswith(".NS") else f"{sym}.NS"
        signature.append((sym, _dashboard_df_signature(ALL_DATA.get(ticker_ns))))
    return tuple(signature)


def get_dashboard_sector_signature(sector_name: str) -> tuple[tuple[str, tuple[int, str, float, float]], ...]:
    """Convenience wrapper for one dashboard basket signature."""
    return get_dashboard_data_signature(get_dashboard_sector_stocks(sector_name))


def get_dashboard_sector_labels(include_overall: bool = True) -> list[str]:
    """Return dashboard sector labels in UI order."""
    if include_overall:
        return DASHBOARD_SECTOR_ORDER.copy()
    return [s for s in DASHBOARD_SECTOR_ORDER if s != "Overall"]


def get_dashboard_index_sector(sector_name: str) -> str:
    """Return the index sector used for macro/index confirmation."""
    return DASHBOARD_TO_INDEX_SECTOR.get(str(sector_name).strip(), "Nifty 50")


def get_dashboard_sector_description(sector_name: str) -> str:
    """Human-readable dashboard description."""
    return DASHBOARD_SECTOR_DESCRIPTIONS.get(str(sector_name).strip(), str(sector_name))


def get_dashboard_sector_stocks(sector_name: str) -> list[str]:
    """
    Return a broad, deduplicated stock basket for the dashboard.

    The dashboard intentionally scans wider sector universes than the compact
    index maps above so users can scan more names per sector.
    """
    try:
        name = str(sector_name).strip()
        if not name:
            return []

        with _DASHBOARD_SECTOR_STOCKS_LOCK:
            cached = _DASHBOARD_SECTOR_STOCKS_CACHE.get(name)

        if cached is None:
            cached = _compute_dashboard_sector_stocks(name)
            with _DASHBOARD_SECTOR_STOCKS_LOCK:
                _DASHBOARD_SECTOR_STOCKS_CACHE[name] = cached

        return list(cached)
    except Exception:
        return []


def get_dashboard_sector_count(sector_name: str) -> int:
    """Return dashboard basket size for one sector label."""
    return len(get_dashboard_sector_stocks(sector_name))


def build_raw_rows_for_tickers(
    tickers: list[str],
    mode: int = 2,
    preload_missing: bool = True,
    workers: int = 12,
) -> list[dict]:
    """
    Build raw rows for an explicit ticker list.

    Preloads only the missing tickers first so local CSV data can be reused
    and repeated dashboard scans stay fast.
    """
    try:
        symbols = _dashboard_unique_symbols(tickers)
        if not symbols:
            return []

        tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in symbols]
        if preload_missing:
            missing = [t for t in tickers_ns if t not in ALL_DATA]
            if missing:
                preload_all(missing, period="6mo", workers=max(1, int(workers)))
                try:
                    from time_travel_engine import (
                        is_active as _tt_brfr_active,
                        get_reference_date as _tt_brfr_date,
                        truncate_df as _tt_brfr_trunc,
                    )
                    if _tt_brfr_active():
                        _cutoff = _tt_brfr_date()
                        if _cutoff is not None:
                            for _t in missing:
                                _df_raw = ALL_DATA.get(_t)
                                if _df_raw is not None and not _df_raw.empty:
                                    ALL_DATA[_t] = _tt_brfr_trunc(_df_raw, _cutoff)
                except Exception:
                    pass

        max_workers = max(1, min(int(workers), len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = [
                row
                for row in ex.map(lambda sym: _build_stock_row_cached(f"{sym}.NS", mode), symbols)
                if row is not None
            ]
        return rows
    except Exception:
        return []


def build_dashboard_sector_raw_rows(
    sector_name: str,
    mode: int = 2,
    preload_missing: bool = True,
    workers: int = 12,
) -> list[dict]:
    """Build raw rows for one broad dashboard sector basket."""
    return build_raw_rows_for_tickers(
        get_dashboard_sector_stocks(sector_name),
        mode=mode,
        preload_missing=preload_missing,
        workers=workers,
    )
