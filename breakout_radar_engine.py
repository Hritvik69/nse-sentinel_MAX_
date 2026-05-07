"""
breakout_radar_engine.py
─────────────────────────────────────────────────────────────────────────
⚡ Next-Day Breakout Radar — NSE Sentinel Intelligence Layer

Identifies stocks in a PRE-BREAKOUT stage — tight consolidation near
resistance, volume building, trend intact — BEFORE the actual breakout
candle fires. Returns 10–80 candidates from the full NSE universe.

Architecture
────────────
• Zero API calls — reads exclusively from ALL_DATA (preloaded) or local CSV
  files via data_downloader.load_csv().
• Fully standalone — no imports from any strategy_engines mode.
• Never crashes — every public function is try/except wrapped.
• Does NOT filter/remove rows from existing scan results.
• Works as an independent scan invoked from app.py.

Scoring Model (0–100)
─────────────────────
Component        Weight  Key signals
─────────────    ──────  ──────────────────────────────────────────────────
Trend              25    Price > EMA20 > EMA50; EMA20 slope positive
Compression        25    ATR tightening (3-5d vs 20d); proximity to 20D high
Volume Build       20    Vol/Avg ratio; rising vol over last 2-3 sessions
RSI Zone           20    52–65 ideal; 65–72 late; >75 penalise; <45 weak
EMA20 Distance     10    0–4% sweet spot; >6% overextended penalty
─────────────    ──────
                  100

Trap Detection
──────────────
WEAK_VOLUME         → Vol/Avg < 1.1
OVERBOUGHT          → RSI > 75
EXTENDED            → Price > 6% above EMA20
FAKE_BREAKOUT_RISK  → Near 20D high but vol not supporting (< 1.1)

Trap count penalties:
  1 trap  → −5 pts (mild caution)
  2 traps → −15 pts (signal downgrade)
  3+ traps → mark TRAP regardless of score

Public entry point
──────────────────
    from breakout_radar_engine import run_breakout_radar
    df = run_breakout_radar()                      # full CSV universe scan
    df = run_breakout_radar(existing_df)           # enrich scan results
    df = run_breakout_radar(cutoff_date=some_date) # time-travel mode
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — graceful stubs when not available
# ──────────────────────────────────────────────────────────────────────

# ALL_DATA: the central preloaded OHLCV store (zero-API engine)
try:
    from strategy_engines._engine_utils import ALL_DATA  # type: ignore[import]
    _ALL_DATA_OK = True
except Exception:
    ALL_DATA: dict = {}
    _ALL_DATA_OK = False

# CSV loader (data_downloader) for standalone universe scans
try:
    from data_downloader import load_csv, DATA_DIR  # type: ignore[import]
    _DOWNLOADER_OK = True
except Exception:
    _DOWNLOADER_OK = False
    DATA_DIR = Path("data")

    def load_csv(ticker_ns: str) -> pd.DataFrame | None:  # type: ignore[misc]
        return None


# ──────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────

def _sf(v: object, default: float = 0.0) -> float:
    """Safe float — never raises, returns finite value or default."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _app_universe_tickers() -> list[str]:
    """
    Borrow the full app universe when running inside app.py so the focused
    scanners report against the same ticker set as Modes 1-6.
    """
    for module_name in ("app", "__main__"):
        mod = sys.modules.get(module_name)
        fetch_fn = getattr(mod, "fetch_nse_tickers", None) if mod else None
        if not callable(fetch_fn):
            continue
        try:
            tickers = fetch_fn()
        except Exception:
            continue
        out: list[str] = []
        seen: set[str] = set()
        for raw in tickers or []:
            ticker_ns = str(raw).strip().upper()
            if not ticker_ns:
                continue
            if not ticker_ns.endswith(".NS"):
                ticker_ns = f"{ticker_ns}.NS"
            if ticker_ns in seen:
                continue
            seen.add(ticker_ns)
            out.append(ticker_ns)
        if out:
            return out
    return []


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi_last(close: pd.Series, period: int = 14) -> float:
    """Return last RSI(14) value — fast scalar result."""
    try:
        if len(close) < period + 2:
            return 50.0
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rsi_s = 100.0 - (100.0 / (1.0 + g / l.replace(0, np.nan)))
        val = float(rsi_s.iloc[-1])
        return val if np.isfinite(val) else 50.0
    except Exception:
        return 50.0


def _chart_link(symbol: str) -> str:
    sym = symbol.replace(".NS", "").strip()
    return f"https://www.tradingview.com/chart/?symbol=NSE:{sym}"


# ──────────────────────────────────────────────────────────────────────
# COMPONENT SCORERS
# ──────────────────────────────────────────────────────────────────────

def _score_trend(
    price: float,
    ema20: float,
    ema50: float,
    ema20_prev: float,
    ema20_slope_pct: float,
) -> float:
    """
    Trend component (0–25 pts).

    Strong uptrend: price > EMA20 > EMA50, EMA20 rising.
    Partial credit for partial alignment.
    """
    pts = 0.0

    # Price vs EMA alignment
    above_ema20 = price > ema20
    ema20_above_ema50 = ema20 > ema50

    if above_ema20 and ema20_above_ema50:
        pts += 15.0   # Full trend alignment
    elif above_ema20:
        pts += 8.0    # Above EMA20 but EMA20 below EMA50 → mixed
    elif ema20_above_ema50:
        pts += 4.0    # EMA20 > EMA50 but price pulled back → caution

    # EMA20 slope bonus
    if ema20_slope_pct > 0.20:
        pts += 10.0   # Rising fast
    elif ema20_slope_pct > 0.08:
        pts += 7.0    # Rising
    elif ema20_slope_pct > 0.01:
        pts += 4.0    # Slightly rising
    elif ema20_slope_pct < -0.10:
        pts -= 5.0    # Falling EMA20 — bad signal

    return float(np.clip(pts, 0.0, 25.0))


def _score_compression(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> tuple[float, float]:
    """
    Compression component (0–25 pts) + distance from 20D high (%).

    Uses ATR ratio (recent 5D vs 20D ATR) and 20D-high proximity.
    Returns (component_score, dist_from_20d_high_pct).
    """
    try:
        if len(close) < 22:
            return 12.0, -5.0   # neutral fallback

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low,
             (high - prev_close).abs(),
             (low  - prev_close).abs()],
            axis=1
        ).max(axis=1)

        atr_20 = float(tr.rolling(20, min_periods=12).mean().iloc[-1])
        atr_5  = float(tr.rolling(5,  min_periods=3 ).mean().iloc[-1])

        if not np.isfinite(atr_20) or atr_20 <= 0:
            atr_ratio = 1.0
        else:
            atr_ratio = atr_5 / atr_20

        # Day range % tightening over last 5 sessions
        current_price = float(close.iloc[-1])
        if current_price <= 0:
            current_price = 1.0
        day_range_pct = ((high - low) / current_price * 100.0).tail(5)
        tightening = 0
        rng_vals = day_range_pct.values
        if len(rng_vals) >= 3:
            for i in range(1, len(rng_vals)):
                if np.isfinite(rng_vals[i]) and np.isfinite(rng_vals[i-1]):
                    if rng_vals[i] <= rng_vals[i-1]:
                        tightening += 1

        pts = 0.0

        # ATR compression ratio score (0–12 pts)
        if atr_ratio < 0.50:
            pts += 12.0   # Very tight — coiling spring
        elif atr_ratio < 0.65:
            pts += 9.0    # Good compression
        elif atr_ratio < 0.80:
            pts += 6.0    # Mild
        elif atr_ratio < 0.95:
            pts += 3.0    # Slightly below average
        elif atr_ratio > 1.20:
            pts -= 4.0    # Range expanding — bad

        # Tightening candle sequence bonus (0–5 pts)
        if tightening >= 4:
            pts += 5.0
        elif tightening == 3:
            pts += 3.0
        elif tightening == 2:
            pts += 1.5

        # Proximity to 20-day high (0–8 pts) — REQUIRED signal for breakout
        high_20d = float(high.tail(21).max())
        dist_20h = (current_price / high_20d - 1.0) * 100.0 if high_20d > 0 else -10.0

        if -1.5 <= dist_20h <= 0.5:
            pts += 8.0    # Right at breakout zone
        elif -3.5 <= dist_20h < -1.5:
            pts += 5.0    # Approaching
        elif -5.5 <= dist_20h < -3.5:
            pts += 2.0    # Some distance
        elif dist_20h < -5.5:
            pts -= 3.0    # Too far — unlikely to break next day

        return float(np.clip(pts, 0.0, 25.0)), float(dist_20h)

    except Exception:
        return 12.0, -5.0


def _score_volume(
    volume: pd.Series,
    vol_ratio: float,
) -> float:
    """
    Volume Build component (0–20 pts).

    vol_ratio  = today's vol / 20D avg vol (already computed).
    Bonus if volume is rising over last 2-3 sessions.
    """
    pts = 0.0

    # Core ratio score (0–14 pts)
    if vol_ratio > 2.5:
        pts += 14.0   # Institutional surge
    elif vol_ratio > 1.8:
        pts += 11.0   # Strong
    elif vol_ratio > 1.4:
        pts += 8.0    # Good build
    elif vol_ratio > 1.2:
        pts += 5.0    # Moderate
    elif vol_ratio > 1.0:
        pts += 2.0    # Barely above average
    elif vol_ratio < 1.0:
        pts -= 8.0    # Below average — penalise hard
    # Exhaustion haircut (vol spike without price follow-through)
    if vol_ratio > 3.5:
        pts -= 4.0

    # Rising volume over last 3 sessions bonus (0–6 pts)
    try:
        if len(volume) >= 4:
            last3 = volume.tail(3).values
            if last3[1] > last3[0] and last3[2] > last3[1]:
                pts += 6.0    # Consecutive rising volume — strong signal
            elif last3[2] > last3[0]:
                pts += 3.0    # Rising vs 2 days ago
    except Exception:
        pass

    return float(np.clip(pts, 0.0, 20.0))


def _score_rsi(rsi: float) -> float:
    """
    RSI Zone component (0–20 pts).

    52–65 → ideal momentum zone (sweet spot)
    65–72 → extended but still running
    > 75  → penalise — overbought
    < 45  → weak — no buying pressure
    """
    if 52.0 <= rsi <= 65.0:
        return 20.0   # Perfect zone — momentum without overbought
    elif 47.0 <= rsi < 52.0:
        return 14.0   # Just entering momentum zone — early, fine
    elif 65.0 < rsi <= 72.0:
        return 9.0    # Late but still trending
    elif 72.0 < rsi <= 75.0:
        return 3.0    # Near overbought — caution
    elif rsi > 75.0:
        return -10.0  # Overbought — strong penalty
    elif 40.0 <= rsi < 47.0:
        return 4.0    # Weak — not in momentum
    else:
        return -5.0   # RSI < 40 — distribution


def _score_ema_distance(delta_ema20_pct: float) -> float:
    """
    EMA20 Distance component (0–10 pts).

    0–4%  → ideal extension (price riding EMA upward)
    > 6%  → overextended — breakout may already be done
    < 0%  → price below EMA (pullback) — mixed signal
    """
    de = delta_ema20_pct
    if 0.0 <= de <= 4.0:
        return 10.0   # Healthy extension
    elif 4.0 < de <= 6.0:
        return 5.0    # Mildly extended
    elif de > 6.0:
        return -8.0   # Overextended — strong penalty
    elif -1.0 <= de < 0.0:
        return 7.0    # Slight pullback to EMA — setup building
    elif -3.0 <= de < -1.0:
        return 3.0    # Pulled back more — accumulation possible
    else:
        return -3.0   # Well below EMA — no edge


# ──────────────────────────────────────────────────────────────────────
# TRAP DETECTION
# ──────────────────────────────────────────────────────────────────────

def _detect_traps(
    rsi: float,
    vol_ratio: float,
    delta_ema20_pct: float,
    dist_20h: float,
) -> list[str]:
    """
    Detect bull-trap conditions. Each flag is an independent risk signal.

    Returns list of active trap flag names.
    """
    flags: list[str] = []

    # WEAK_VOLUME — no institutional support
    if vol_ratio < 1.1:
        flags.append("WEAK_VOLUME")

    # OVERBOUGHT — RSI stretched, likely to mean-revert
    if rsi > 75.0:
        flags.append("OVERBOUGHT")

    # EXTENDED — price too far from EMA20 for a clean entry
    if delta_ema20_pct > 6.0:
        flags.append("EXTENDED")

    # FAKE_BREAKOUT_RISK — price near 20D high but volume not supporting
    # (looks like breakout but no institutional follow-through)
    if dist_20h >= -2.5 and vol_ratio < 1.1:
        if "WEAK_VOLUME" not in flags:    # avoid double-counting label
            flags.append("FAKE_BREAKOUT_RISK")
        else:
            flags.append("FAKE_BREAKOUT_RISK")

    return flags


def _trap_penalty(trap_flags: list[str]) -> float:
    """Return score penalty based on trap count (always negative or zero)."""
    n = len(trap_flags)
    if n == 0:
        return 0.0
    if n == 1:
        return -5.0    # Mild — a single caution flag, not a blocker
    if n == 2:
        return -15.0   # Signal downgrade
    return -25.0       # 3+ = severe, score will be very low → TRAP signal


# ──────────────────────────────────────────────────────────────────────
# FINAL SIGNAL CLASSIFIERS
# ──────────────────────────────────────────────────────────────────────

def _final_signal(score: float, trap_flags: list[str]) -> str:
    """
    Convert numeric score + trap flags into the actionable signal label.

    80+   → HIGH PROBABILITY BREAKOUT (few/no traps only)
    65–79 → STRONG SETUP
    50–64 → WATCHLIST
    < 50  → AVOID
    TRAP  → if 3+ trap flags regardless of score
    """
    if len(trap_flags) >= 3:
        return "TRAP"
    if score >= 80.0:
        return "HIGH PROBABILITY BREAKOUT"
    if score >= 65.0:
        return "STRONG SETUP"
    if score >= 50.0:
        return "WATCHLIST"
    return "AVOID"


def _risk_score(trap_flags: list[str], rsi: float, vol_ratio: float, delta_ema20: float) -> float:
    """
    Risk score 0–100 (higher = more risk).
    Combines trap count + individual factor extremes.
    """
    risk = float(len(trap_flags)) * 20.0

    if rsi > 75:
        risk += 15.0
    elif rsi > 70:
        risk += 7.0

    if vol_ratio < 0.9:
        risk += 15.0
    elif vol_ratio < 1.1:
        risk += 5.0

    if delta_ema20 > 8.0:
        risk += 15.0
    elif delta_ema20 > 6.0:
        risk += 7.0

    return float(np.clip(risk, 0.0, 100.0))


# ──────────────────────────────────────────────────────────────────────
# OHLCV-LEVEL ANALYSIS  (core computation per stock)
# ──────────────────────────────────────────────────────────────────────

def _analyze_ohlcv(
    df_raw: pd.DataFrame,
    symbol: str,
    cutoff_date=None,
) -> dict | None:
    """
    Run full breakout-radar analysis on a single stock's OHLCV DataFrame.

    Parameters
    ----------
    df_raw : pd.DataFrame
        OHLCV data with at least Close/Volume columns.
    symbol : str
        Ticker symbol without .NS suffix.
    cutoff_date : datetime.date | None
        If set, slices the data to <= cutoff_date (time-travel safety).

    Returns
    -------
    dict  Single row dict with all output columns, or None if insufficient data.
    """
    try:
        if df_raw is None or len(df_raw) < 45:
            return None

        df = df_raw.copy()

        # ── Time-travel slice ─────────────────────────────────────────
        if cutoff_date is not None:
            try:
                mask = pd.to_datetime(df.index).date <= cutoff_date
                df = df.loc[mask]
            except Exception:
                pass
            if len(df) < 45:
                return None

        df = df.tail(250)  # cap to ~1 year for performance

        # ── Column normalisation ──────────────────────────────────────
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ("Close", "Volume"):
            if col not in df.columns:
                return None

        close  = df["Close"].astype(float).ffill()
        volume = df["Volume"].astype(float).ffill()
        high   = df["High"].astype(float).ffill()  if "High"  in df.columns else close.copy()
        low    = df["Low"].astype(float).ffill()   if "Low"   in df.columns else close.copy()
        open_s = df["Open"].astype(float).ffill()  if "Open"  in df.columns else close.shift(1).fillna(close)

        if len(close) < 45:
            return None

        current_price = float(close.iloc[-1])
        if not np.isfinite(current_price) or current_price <= 0:
            return None

        # ── Indicators ────────────────────────────────────────────────
        ema20      = _ema(close, 20)
        ema50      = _ema(close, 50)
        rsi        = _rsi_last(close, 14)
        avg_vol_20 = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio_s = volume / avg_vol_20.replace(0, np.nan)
        vol_ratio  = _sf(vol_ratio_s.iloc[-1], 1.0)

        ema20_now    = _sf(ema20.iloc[-1],   current_price)
        ema50_now    = _sf(ema50.iloc[-1],   current_price)
        ema20_prev   = _sf(ema20.iloc[-2],   ema20_now)
        ema20_slope  = ((ema20_now / ema20_prev) - 1.0) * 100.0 if ema20_prev > 0 else 0.0

        delta_ema20  = ((current_price / ema20_now) - 1.0) * 100.0 if ema20_now > 0 else 0.0

        # 5-day return
        ret5  = (current_price / _sf(close.iloc[-6], current_price) - 1.0) * 100.0 if len(close) >= 6 else 0.0

        # ── Component scores ──────────────────────────────────────────
        s_trend = _score_trend(
            price=current_price,
            ema20=ema20_now,
            ema50=ema50_now,
            ema20_prev=ema20_prev,
            ema20_slope_pct=ema20_slope,
        )

        s_compression, dist_20h = _score_compression(close, high, low)
        s_volume   = _score_volume(volume, vol_ratio)
        s_rsi      = _score_rsi(rsi)
        s_ema_dist = _score_ema_distance(delta_ema20)

        raw_score = s_trend + s_compression + s_volume + s_rsi + s_ema_dist

        # ── Trap detection + penalties ────────────────────────────────
        trap_flags  = _detect_traps(rsi, vol_ratio, delta_ema20, dist_20h)
        penalty     = _trap_penalty(trap_flags)
        final_score = float(np.clip(raw_score + penalty, 0.0, 100.0))

        # ── Labels ───────────────────────────────────────────────────
        signal    = _final_signal(final_score, trap_flags)
        risk_val  = _risk_score(trap_flags, rsi, vol_ratio, delta_ema20)
        trap_str  = ", ".join(trap_flags) if trap_flags else "None"

        return {
            "Symbol":               symbol.replace(".NS", "").strip(),
            "Price (₹)":            round(current_price, 2),
            "Volume Ratio":         round(vol_ratio, 2),
            "RSI":                  round(rsi, 1),
            "Δ EMA20 (%)":          round(delta_ema20, 2),
            "Δ 20D High (%)":       round(dist_20h, 2),
            "5D Return (%)":        round(ret5, 2),
            "Compression Score":    round(s_compression, 1),
            "Trend Score":          round(s_trend, 1),
            "Volume Score":         round(s_volume, 1),
            "RSI Score":            round(s_rsi, 1),
            "EMA Dist Score":       round(s_ema_dist, 1),
            "Risk Score":           round(risk_val, 1),
            "Trap Flags":           trap_str,
            "Trap Count":           len(trap_flags),
            "Final Score":          round(final_score, 1),
            "Signal":               signal,
            "Chart Link":           _chart_link(symbol),
        }

    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# DATA SOURCE: prefer ALL_DATA, fall back to CSV
# ──────────────────────────────────────────────────────────────────────

def _get_df(ticker_ns: str) -> pd.DataFrame | None:
    def _check_fresh(df):
        # Returns df if fresh enough, None if stale.
        # Always returns df unchanged if check is unavailable or TT is on.
        if df is None:
            return None
        try:
            from strategy_engines._engine_utils import (
                is_fresh_enough as _ife,
            )
        except ImportError:
            return df
        try:
            from time_travel_engine import is_active as _tt_on
            if _tt_on():
                return df
        except Exception:
            pass
        try:
            return df if _ife(df, strict=True) else None
        except Exception:
            return df

    # 1. ALL_DATA (zero API — fastest path)
    if _ALL_DATA_OK and ALL_DATA:
        df = ALL_DATA.get(ticker_ns)
        if df is not None and len(df) >= 45:
            fresh = _check_fresh(df)
            if fresh is not None:
                return fresh

    # 2. CSV fallback
    if _DOWNLOADER_OK:
        try:
            csv_df = load_csv(ticker_ns)
            if csv_df is not None:
                return _check_fresh(csv_df)
        except Exception:
            pass

    return None


def _get_all_tickers() -> list[str]:
    """
    Return all available tickers as ticker_ns strings (with .NS suffix).

    Priority order:
      1. Full app universe (same list shown in the main dashboard)
      2. ALL_DATA keys (already preloaded — fastest)
      3. DATA_DIR CSV files
    """
    ordered: list[str] = []
    seen: set[str] = set()
    app_universe = _app_universe_tickers()

    if len(app_universe) >= 2000:
        return app_universe

    def _add_many(items) -> None:
        for raw in items or []:
            ticker_ns = str(raw).strip().upper()
            if not ticker_ns:
                continue
            if not ticker_ns.endswith(".NS"):
                ticker_ns = f"{ticker_ns}.NS"
            if ticker_ns in seen:
                continue
            seen.add(ticker_ns)
            ordered.append(ticker_ns)

    _add_many(app_universe)

    if _ALL_DATA_OK and ALL_DATA:
        _add_many(
            k for k, v in ALL_DATA.items()
            if v is not None and len(v) >= 45
        )

    if _DOWNLOADER_OK:
        try:
            _add_many(p.stem for p in DATA_DIR.glob("*.csv"))
        except Exception:
            pass

    return ordered


# ──────────────────────────────────────────────────────────────────────
# ENRICHMENT MODE  (called with existing scan DataFrame)
# ──────────────────────────────────────────────────────────────────────

def _enrich_from_scan_df(df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
    """
    Enrich an existing scan results DataFrame with breakout-radar columns.
    Pulls OHLCV from ALL_DATA / CSV for each symbol; uses row indicators as
    fallback when OHLCV is unavailable.
    """
    rows: list[dict] = []

    for _, row in df.iterrows():
        sym_raw = str(row.get("Symbol") or row.get("Ticker") or "").strip()
        if not sym_raw:
            continue

        ticker_ns = sym_raw if sym_raw.endswith(".NS") else f"{sym_raw}.NS"
        df_h = _get_df(ticker_ns)

        if df_h is not None and len(df_h) >= 45:
            result = _analyze_ohlcv(df_h, sym_raw, cutoff_date=cutoff_date)
        else:
            # Fallback: reconstruct from scan-result row indicators
            rsi     = _sf(row.get("RSI", 55))
            vol_r   = _sf(row.get("Vol / Avg", 1.0))
            de20    = _sf(row.get("Δ vs EMA20 (%)", 0.0))
            d20h    = _sf(row.get("Δ vs 20D High (%)", -5.0))
            r5d     = _sf(row.get("5D Return (%)", 0.0))
            price   = _sf(row.get("Price (₹)", 0.0))

            trap_flags  = _detect_traps(rsi, vol_r, de20, d20h)
            penalty     = _trap_penalty(trap_flags)
            # Simplified score without OHLCV data
            s_trend  = 12.0 if de20 > 0 else 6.0       # rough proxy
            s_compr  = 15.0 if -3.5 <= d20h <= 0.5 else 8.0
            s_vol    = _score_volume(pd.Series([vol_r]), vol_r)
            s_rsi    = _score_rsi(rsi)
            s_emd    = _score_ema_distance(de20)
            raw      = s_trend + s_compr + s_vol + s_rsi + s_emd
            final    = float(np.clip(raw + penalty, 0.0, 100.0))
            signal   = _final_signal(final, trap_flags)
            risk_val = _risk_score(trap_flags, rsi, vol_r, de20)

            result = {
                "Symbol":            sym_raw,
                "Price (₹)":         round(price, 2),
                "Volume Ratio":      round(vol_r, 2),
                "RSI":               round(rsi, 1),
                "Δ EMA20 (%)":       round(de20, 2),
                "Δ 20D High (%)":    round(d20h, 2),
                "5D Return (%)":     round(r5d, 2),
                "Compression Score": round(s_compr, 1),
                "Trend Score":       round(s_trend, 1),
                "Volume Score":      round(s_vol, 1),
                "RSI Score":         round(s_rsi, 1),
                "EMA Dist Score":    round(s_emd, 1),
                "Risk Score":        round(risk_val, 1),
                "Trap Flags":        ", ".join(trap_flags) if trap_flags else "None",
                "Trap Count":        len(trap_flags),
                "Final Score":       round(final, 1),
                "Signal":            signal,
                "Chart Link":        _chart_link(sym_raw),
            }

        if result is not None:
            rows.append(result)

    return _build_output_df(rows)


# ──────────────────────────────────────────────────────────────────────
# UNIVERSE SCAN MODE  (called with no existing DataFrame)
# ──────────────────────────────────────────────────────────────────────

def _scan_universe(
    cutoff_date=None,
    max_workers: int = 16,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Full universe scan — iterates all available tickers in parallel.
    """
    tickers = _get_all_tickers()
    if not tickers:
        return pd.DataFrame()

    rows: list[dict] = []
    total = len(tickers)
    done = 0

    def _worker(ticker_ns: str) -> dict | None:
        df_h = _get_df(ticker_ns)
        if df_h is None:
            return None
        sym = ticker_ns.replace(".NS", "")
        return _analyze_ohlcv(df_h, sym, cutoff_date=cutoff_date)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            try:
                result = fut.result()
                if result is not None:
                    rows.append(result)
            except Exception:
                pass
            finally:
                done += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done, total, len(rows))
                    except Exception:
                        pass

    return _build_output_df(rows)


# ──────────────────────────────────────────────────────────────────────
# OUTPUT BUILDER + FILTER + SORT
# ──────────────────────────────────────────────────────────────────────

_SIGNAL_RANK = {
    "HIGH PROBABILITY BREAKOUT": 4,
    "STRONG SETUP":              3,
    "WATCHLIST":                 2,
    "AVOID":                     1,
    "TRAP":                      0,
}


def _build_output_df(rows: list[dict]) -> pd.DataFrame:
    """
    Convert row dicts → filtered, sorted DataFrame.

    Filter: Final Score >= 48 AND Trap Count < 3
    (Ensures 10–80 candidates from a typical NSE universe.)
    """
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # ── Filter ────────────────────────────────────────────────────────
    # Score threshold ≥ 48 ensures healthy candidate count (not 0–5)
    # Removing stocks with 3+ traps prevents blatant fakes
    if "Final Score" in out.columns and "Trap Count" in out.columns:
        out = out[
            (out["Final Score"] >= 48.0) &
            (out["Trap Count"]  <  3)
        ].copy()

    if out.empty:
        return out

    # ── Sort: signal tier → Final Score → Risk Score (asc) ───────────
    out["_sig_rank"] = out["Signal"].map(_SIGNAL_RANK).fillna(0)
    out = out.sort_values(
        ["_sig_rank", "Final Score", "Risk Score"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    out = out.drop(columns=["_sig_rank", "Trap Count"], errors="ignore")

    return out


# ──────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def run_breakout_radar(
    df: pd.DataFrame | None = None,
    cutoff_date=None,
    max_workers: int = 16,
    progress_callback=None,
) -> pd.DataFrame:
    """
    ⚡ Next-Day Breakout Radar — main entry point.

    Parameters
    ----------
    df : pd.DataFrame | None
        (a) Existing scan results from NSE Sentinel main scanner.
            When provided, enriches those results with breakout-radar columns.
        (b) None (default) → scans the full available universe from
            ALL_DATA / local CSVs independently.

    cutoff_date : datetime.date | None
        When set, slices OHLCV data to rows on or before this date.
        Used for Time Travel mode.  Prevents future data leakage.

    max_workers : int
        Thread-pool workers for parallel universe scan (default 16).

    progress_callback : callable(done: int, total: int, found: int) | None
        Optional UI callback invoked after each ticker completes during
        full-universe scanning.

    Returns
    -------
    pd.DataFrame
        Columns:
            Symbol              – ticker without .NS
            Price (₹)           – last close
            Volume Ratio        – vol / 20D avg
            RSI                 – RSI(14)
            Δ EMA20 (%)         – % distance from EMA20
            Δ 20D High (%)      – % below 20-day high (negative = below)
            5D Return (%)       – 5-day price return
            Compression Score   – 0–25 (ATR tightening + proximity)
            Trend Score         – 0–25 (EMA alignment + slope)
            Volume Score        – 0–20
            RSI Score           – 0–20
            EMA Dist Score      – 0–10
            Risk Score          – 0–100 (higher = more risky)
            Trap Flags          – comma-separated active trap names
            Final Score         – 0–100 composite
            Signal              – HIGH PROBABILITY BREAKOUT / STRONG SETUP /
                                  WATCHLIST / AVOID / TRAP
            Chart Link          – TradingView URL

        Sorted by Signal tier then Final Score descending.
        Filtered: Final Score >= 48, Trap Count < 3.
        Typically returns 10–80 candidates from full NSE universe.

    Never raises — all paths are try/except wrapped.
    """
    try:
        # ── Branch A: enrich existing scan results ─────────────────────
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty and (
            "RSI" in df.columns or "Symbol" in df.columns
        ):
            return _enrich_from_scan_df(df, cutoff_date=cutoff_date)

        # ── Branch B: full universe scan ───────────────────────────────
        return _scan_universe(
            cutoff_date=cutoff_date,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )

    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────
# SUMMARY HELPER  (called by app.py UI layer)
# ──────────────────────────────────────────────────────────────────────

def radar_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for the metric bar in the UI.

    Keys:
        total           int
        high_prob       int    ("HIGH PROBABILITY BREAKOUT" count)
        strong          int    ("STRONG SETUP" count)
        watchlist       int    ("WATCHLIST" count)
        avg_score       float
        avg_volume_ratio float
        top_symbol      str
    """
    empty = {
        "total": 0, "high_prob": 0, "strong": 0,
        "watchlist": 0, "avg_score": 0.0,
        "avg_volume_ratio": 0.0, "top_symbol": "-",
    }
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return empty
        sig = df.get("Signal", pd.Series(dtype=str))
        return {
            "total":            len(df),
            "high_prob":        int((sig == "HIGH PROBABILITY BREAKOUT").sum()),
            "strong":           int((sig == "STRONG SETUP").sum()),
            "watchlist":        int((sig == "WATCHLIST").sum()),
            "avg_score":        round(float(df["Final Score"].mean()), 1),
            "avg_volume_ratio": round(float(df["Volume Ratio"].mean()), 2),
            "top_symbol":       str(df["Symbol"].iloc[0]) if len(df) > 0 else "-",
        }
    except Exception:
        return empty
