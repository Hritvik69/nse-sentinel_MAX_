"""
app_prediction_chart_section.py  ·  NSE Sentinel  ·  v2 (Pro)
═══════════════════════════════════════════════════════════════════════════════
"📊 Prediction Chart Tomorrow"

UPGRADES FROM v1
────────────────
  PART 1  Candle visuals   whiskerwidth=0.8, body line_width=1.5, opacity=1.0
  PART 2  Multi-EMA        EMA20 (amber) + EMA50 (blue), alignment → confidence
  PART 3  Predicted candle solid blue body + polished halo projection
  PART 4  Regime detection TRENDING / SIDEWAYS / HIGH VOLATILITY
  PART 5  Confidence       signal agreement bonus/penalty, EMA alignment weight
  PART 6  Zoom+interaction range buttons 1M / All, toolbar cleaned
  PART 7  Chart structure  height 600, balanced margins, proper candle spacing
  PART 8  Label            "Bullish (72%) — Strong Momentum" contextual tags
  PART 9  Error UX         styled error cards, detailed messages
  PART 10 Performance      unchanged cache, avoids re-fetch
  PART 11 Unchanged        data engine, prediction logic structure, UI flow

COLOUR PALETTE (TradingView)
─────────────────────────────
  Background  #0a0e1a    Grid     rgba(255,255,255,0.06)
  Bull        #00ff88    Bear     #ff3b5c
  EMA20       #f5a623    EMA50    #2196f3
  Side        #8ab4d8    Tick     #4a6480
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    from prediction_feedback_store import read_feedback_log as _read_feedback_log
except Exception:
    def _read_feedback_log() -> pd.DataFrame:
        return pd.DataFrame()

try:
    from tomorrow_prediction_engine import get_tomorrow_prediction as _get_tomorrow_prediction
except Exception:
    def _get_tomorrow_prediction(ticker, all_data, mode):  # type: ignore[misc]
        return {}

try:
    from nse_learning_brain import get_cached_prediction as _get_cached_prediction
except Exception:
    def _get_cached_prediction(ticker):  # type: ignore[misc]
        return {}

try:
    from strategy_engines._engine_utils import get_market_data_signature as _get_market_data_signature
except Exception:
    def _get_market_data_signature(live_bucket_minutes: int = 5) -> str:
        return "fallback"

try:
    from feature_data_manager import (
        get_current_window as _get_feature_window,
        get_time_travel_date as _get_feature_tt_date,
        render_data_status_badge as _render_data_status_badge,
    )
except Exception:
    def _get_feature_window() -> str:
        return "CLOSED"

    def _get_feature_tt_date():
        return None

    def _render_data_status_badge(status, label: str = "") -> None:
        return None

try:
    from prediction_chart_engine import fetch_chart_data as _fetch_chart_data, get_chart_status as _get_chart_status
    _PREDICTION_CHART_ENGINE_OK = True
except Exception:
    _PREDICTION_CHART_ENGINE_OK = False

    def _fetch_chart_data(
        symbol: str,
        *,
        period: str = "2mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame | None:
        return None

    def _get_chart_status(symbol: str):
        return None

# ── yfinance ──────────────────────────────────────────────────────────────────
_YF_OK = False
try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    pass

# ── Plotly ────────────────────────────────────────────────────────────────────
_PLOTLY_OK = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BG         = "#0a0e1a"
GRID       = "rgba(255,255,255,0.06)"
GRID_MINOR = "rgba(255,255,255,0.03)"
BULL       = "#00ff88"
BEAR       = "#ff3b5c"
SIDE       = "#8ab4d8"
EMA20_COL  = "#f5a623"      # amber — short trend
EMA50_COL  = "#2196f3"      # blue  — mid trend
PRED_COL   = "#4da3ff"      # blue  — prediction candle
VOL_BULL   = "rgba(0,255,136,0.35)"
VOL_BEAR   = "rgba(255,59,92,0.35)"
TICK       = "#4a6480"
LABEL      = "#ccd9e8"
SUBTEXT    = "#8ab4d8"

# Predicted candle
PRED_GLOW_ALPHA = 0.14
PRED_BORDER_W   = 2.4
PRED_EDGE_COL   = "#dbeafe"
PRED_WICK_COL   = "#8ec5ff"
PRED_PATH_COL   = "rgba(77,163,255,0.78)"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sf(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


def _normalise_symbol(raw: str) -> str:
    s = raw.strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — DATA ENGINE  (unchanged from v1, cache preserved)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(
    symbol: str,
    timeframe: str = "1d",
    data_signature: str = "",
    time_context_key: str = "live",
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch chart data via the shared feature-data pipeline when available.

    data_signature and time_context_key are included so live/simulated panels
    can refresh when either the market bucket or the selected simulation date changes.

    Returns clean DataFrame[Open,High,Low,Close,Volume] or None on failure.
    Minimum 30 candles required.
    """
    ticker = _normalise_symbol(symbol)
    if _PREDICTION_CHART_ENGINE_OK:
        try:
            shared = _fetch_chart_data(
                ticker,
                period="3mo",
                interval=timeframe,
                force_refresh=force_refresh,
            )
            if shared is not None and not shared.empty:
                return shared
        except Exception:
            pass

    if not _YF_OK:
        return None
    tt_cutoff = _get_feature_tt_date()
    try:
        if tt_cutoff is not None:
            cutoff_ts = pd.Timestamp(tt_cutoff)
            start = cutoff_ts - pd.Timedelta(days=110)
            end = cutoff_ts + pd.Timedelta(days=1)
            raw = yf.download(
                ticker,
                start=start.date().isoformat(),
                end=end.date().isoformat(),
                interval=timeframe,
                auto_adjust=True,
                progress=False,
                timeout=15,
            )
        else:
            raw = yf.download(
                ticker, period="3mo", interval=timeframe,
                auto_adjust=True, progress=False, timeout=15,
            )
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(raw.columns):
        return None

    df = raw[list(required)].copy().dropna(subset=["Open","High","Low","Close"])
    df = df.sort_index()
    if tt_cutoff is not None:
        try:
            df = df.loc[pd.to_datetime(df.index).date <= tt_cutoff].copy()
        except Exception:
            pass
        if df.empty:
            return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    return df if len(df) >= 30 else None


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — REGIME DETECTION  (new)
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> dict:
    """
    Classify market regime from the last 20 sessions.

    Logic
    ─────
    ATR ratio  = current ATR(14) / mean ATR over last 30 bars
    EMA slope  = (EMA20[-1] - EMA20[-5]) / EMA20[-5] * 100

    Regime:
        HIGH VOLATILITY  → ATR ratio > 1.6
        TRENDING UP      → EMA slope > +0.8% AND EMA20 > EMA50
        TRENDING DOWN    → EMA slope < −0.8% AND EMA20 < EMA50
        SIDEWAYS         → all else

    Returns dict:
        regime       str
        emoji        str
        color        str   hex
        description  str
        ema_aligned  bool  EMA20 > EMA50
        atr_ratio    float
        ema_slope    float
    """
    _default = {
        "regime": "Sideways", "emoji": "↔", "color": SIDE,
        "description": "No clear trend", "ema_aligned": False,
        "atr_ratio": 1.0, "ema_slope": 0.0,
    }
    try:
        df  = df.sort_index().tail(60).copy()
        n   = len(df)
        if n < 20:
            return _default

        close = df["Close"].astype(float)
        e20   = _ema(close, 20)
        e50   = _ema(close, 50) if n >= 30 else e20

        atr_series = _atr(df, 14)
        atr_now    = float(atr_series.iloc[-1])
        atr_hist   = float(atr_series.tail(30).mean())
        atr_ratio  = atr_now / (atr_hist + 1e-9)

        e20v  = e20.values
        slope_pct = (e20v[-1] - e20v[-5]) / (abs(e20v[-5]) + 1e-9) * 100

        ema_aligned = float(e20.iloc[-1]) > float(e50.iloc[-1])

        if atr_ratio > 1.6:
            regime = "High Volatility"
            emoji  = "⚡"
            color  = "#f0b429"
            desc   = "Elevated volatility — widen stops"
        elif slope_pct > 0.8 and ema_aligned:
            regime = "Trending Up"
            emoji  = "↗"
            color  = BULL
            desc   = "Bullish structure — trend following valid"
        elif slope_pct < -0.8 and not ema_aligned:
            regime = "Trending Down"
            emoji  = "↘"
            color  = BEAR
            desc   = "Bearish structure — avoid longs"
        else:
            regime = "Sideways"
            emoji  = "↔"
            color  = SIDE
            desc   = "Range-bound — breakout watch"

        return {
            "regime":      regime,
            "emoji":       emoji,
            "color":       color,
            "description": desc,
            "ema_aligned": ema_aligned,
            "atr_ratio":   round(atr_ratio, 2),
            "ema_slope":   round(slope_pct, 3),
        }
    except Exception:
        return _default


# ─────────────────────────────────────────────────────────────────────────────
# PART 4+5 — PREDICTION ENGINE  (upgraded: EMA50, confidence refinement)
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction(df: pd.DataFrame) -> dict:
    """
    6-signal prediction with calibrated confidence.

    Signals (each −1 … +1):
        ema20_slope        EMA20 direction over 5 bars        wt 0.25
        ema_alignment      EMA20 vs EMA50 position             wt 0.20
        price_vs_ema20     % distance from EMA20               wt 0.15
        candle_direction   majority last 3 candles             wt 0.20
        volume_trend       last vol vs 10-day avg              wt 0.12
        volatility         ATR compression/expansion           wt 0.08

    Confidence calibration (Part 5):
        base: sigmoid of composite
        +5%  if all top-3 signals agree in direction
        −8%  if top-3 signals conflict (mix of +/-)
        ±3%  regime context bonus/penalty
    """
    _neutral = {
        "direction": "Sideways", "confidence": 48.0,
        "signals": {}, "atr": 0.0,
        "regime": {}, "label_tag": "Unclear Structure",
    }

    try:
        df  = df.sort_index().tail(60).copy()
        n   = len(df)
        if n < 14:
            return _neutral

        close  = df["Close"].astype(float)
        high   = df["High"].astype(float)
        low    = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        e20   = _ema(close, 20)
        e50   = _ema(close, 50) if n >= 30 else e20

        atr_series = _atr(df, 14)
        atr14      = float(atr_series.iloc[-1])
        atr_hist   = float(atr_series.tail(30).mean())
        atr_ratio  = atr14 / (atr_hist + 1e-9)

        # ── Signal 1: EMA20 slope ─────────────────────────────────────
        e20v = e20.values
        slope_pct  = (e20v[-1] - e20v[-5]) / (abs(e20v[-5]) + 1e-9) * 100 if n >= 6 else 0.0
        ema20_slope = _clamp(slope_pct / 1.5)

        # ── Signal 2: EMA alignment (EMA20 vs EMA50) ─────────────────
        e20_last = float(e20.iloc[-1])
        e50_last = float(e50.iloc[-1])
        dist50   = (e20_last - e50_last) / (abs(e50_last) + 1e-9) * 100
        ema_align = _clamp(dist50 / 3.0)          # ±3% gap → ±1

        # ── Signal 3: Price vs EMA20 ──────────────────────────────────
        last_close = float(close.iloc[-1])
        dist20     = (last_close - e20_last) / (e20_last + 1e-9) * 100
        price_ema  = _clamp(dist20 / 4.0)

        # ── Signal 4: Last-3 candle direction ────────────────────────
        cv  = close.values[-4:]
        ups = sum(1 for i in range(1, len(cv)) if cv[i] > cv[i-1])
        dns = sum(1 for i in range(1, len(cv)) if cv[i] < cv[i-1])
        cand_dir = (ups - dns) / 3.0

        # ── Signal 5: Volume trend ────────────────────────────────────
        vol_avg = float(volume.iloc[-11:-1].mean()) if n >= 11 else float(volume.mean())
        vol_r   = float(volume.iloc[-1]) / (vol_avg + 1e-9)
        is_up   = last_close >= float(df["Open"].astype(float).iloc[-1])
        raw_vol = _clamp((vol_r - 1.0) / 1.0)
        vol_sig = raw_vol if is_up else -abs(raw_vol)

        # ── Signal 6: Volatility condition ───────────────────────────
        if atr_ratio < 0.80:
            vol_cond = 0.30 * ema20_slope
        elif atr_ratio > 1.50:
            vol_cond = -0.20
        else:
            vol_cond = 0.0

        # ── Composite ────────────────────────────────────────────────
        W = {
            "ema20_slope":      0.25,
            "ema_alignment":    0.20,
            "price_vs_ema20":   0.15,
            "candle_direction": 0.20,
            "volume_trend":     0.12,
            "volatility":       0.08,
        }
        S = {
            "ema20_slope":      round(ema20_slope, 3),
            "ema_alignment":    round(ema_align,   3),
            "price_vs_ema20":   round(price_ema,   3),
            "candle_direction": round(cand_dir,    3),
            "volume_trend":     round(vol_sig,     3),
            "volatility":       round(vol_cond,    3),
        }
        composite = sum(S[k] * W[k] for k in W)

        # ── Direction ─────────────────────────────────────────────────
        if composite >= 0.11:
            direction = "Bullish"
        elif composite <= -0.11:
            direction = "Bearish"
        else:
            direction = "Sideways"

        # ── Base confidence (sigmoid) ─────────────────────────────────
        prob      = 100.0 / (1.0 + math.exp(-composite * 6.0))
        base_conf = round(50.0 + abs(prob - 50.0) * 1.05, 1)

        # ── PART 5: Confidence calibration ───────────────────────────
        # Check agreement between top-3 weighted signals
        top3 = ["ema20_slope", "ema_alignment", "candle_direction"]
        top3_vals = [S[k] for k in top3]
        signs     = [1 if v > 0.05 else -1 if v < -0.05 else 0 for v in top3_vals]
        n_pos     = signs.count(1)
        n_neg     = signs.count(-1)

        if n_pos == 3 or n_neg == 3:
            # All three agree
            conf_adj = +5.0
        elif n_pos >= 2 or n_neg >= 2:
            # Two agree
            conf_adj = +2.0
        else:
            # Conflict
            conf_adj = -8.0

        # EMA alignment bonus: if prediction matches EMA stack
        if direction == "Bullish" and ema_align > 0.2:
            conf_adj += 3.0
        elif direction == "Bearish" and ema_align < -0.2:
            conf_adj += 3.0
        elif (direction == "Bullish" and ema_align < -0.3) or \
             (direction == "Bearish" and ema_align > 0.3):
            conf_adj -= 5.0   # prediction fights EMA alignment

        # Regime context
        regime = detect_regime(df)
        if regime["regime"] == "High Volatility":
            conf_adj -= 4.0   # uncertainty in high-vol
        elif regime["regime"] in ("Trending Up", "Trending Down"):
            if (direction == "Bullish" and regime["regime"] == "Trending Up") or \
               (direction == "Bearish" and regime["regime"] == "Trending Down"):
                conf_adj += 3.0   # regime confirms direction

        confidence = round(min(92.0, max(44.0, base_conf + conf_adj)), 1)

        # ── Label tag (Part 8) ────────────────────────────────────────
        label_tag = _build_label_tag(direction, confidence, composite, regime)

        return {
            "direction":  direction,
            "confidence": confidence,
            "signals":    S,
            "atr":        round(atr14, 4),
            "regime":     regime,
            "label_tag":  label_tag,
            "composite":  round(composite, 4),
        }

    except Exception:
        return _neutral


def _build_label_tag(
    direction: str, confidence: float, composite: float, regime: dict
) -> str:
    """
    Return a short contextual tag string for the label annotation.
    Examples: "Strong Momentum", "Weak Structure", "Range Breakout Watch"
    """
    if direction == "Sideways":
        if regime.get("atr_ratio", 1.0) < 0.75:
            return "Volatility Compression"
        return "Range-Bound"

    strength = abs(composite)
    if direction == "Bullish":
        if strength > 0.35:   return "Strong Momentum"
        if strength > 0.18:   return "Moderate Upside"
        return "Weak Bullish"
    else:
        if strength > 0.35:   return "Strong Breakdown"
        if strength > 0.18:   return "Moderate Downside"
        return "Weak Bearish"


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — PREDICTED CANDLE BUILDER  (unchanged logic, same formulas)
# ─────────────────────────────────────────────────────────────────────────────

def build_predicted_candle(df: pd.DataFrame, prediction: dict) -> dict:
    """
    ATR-based OHLC projection for next business day.
    Bullish: O=last_close, C=+0.8×ATR, H=C+0.2×ATR, L=O-0.2×ATR
    Bearish: O=last_close, C=-0.8×ATR, H=O+0.2×ATR, L=C-0.2×ATR
    Sideways: small body, H/L = ±0.3×ATR
    """
    last_close = float(df["Close"].iloc[-1])
    atr        = prediction.get("atr") or float((df["High"] - df["Low"]).tail(14).mean())
    direction  = prediction.get("direction", "Sideways")

    next_date = pd.Timestamp(df.index[-1]) + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)

    if direction == "Bullish":
        o, c = last_close, last_close + atr * 0.80
        h, l = c + atr * 0.20, o - atr * 0.20
    elif direction == "Bearish":
        o, c = last_close, last_close - atr * 0.80
        h, l = o + atr * 0.20, c - atr * 0.20
    else:
        o, c = last_close, last_close + atr * 0.05
        h, l = o + atr * 0.30, o - atr * 0.30

    return {"date": next_date, "open": o, "high": h, "low": l, "close": c}


# ─────────────────────────────────────────────────────────────────────────────
# PART 3+7 — CHART ENGINE  (full visual upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(
    df:          pd.DataFrame,
    symbol:      str,
    prediction:  dict,
    pred_candle: dict,
) -> "go.Figure | None":
    """
    TradingView-style Plotly chart — pro-level v2.

    Upgrades vs v1
    ──────────────
    • whiskerwidth=0.8  (thicker wicks)
    • Candlestick line_width=1.5 (thicker bodies)
    • EMA50 trace added (blue)
    • Predicted candle: solid blue projected candle with halo
    • Label: "Bullish (72%) — Strong Momentum"
    • Range buttons: 1M / All
    • Height 600, balanced margins
    • Vertical separator to isolate prediction zone
    """
    if not _PLOTLY_OK:
        return None

    df     = df.sort_index().copy()
    dates  = list(df.index)
    opens  = df["Open"].astype(float).values
    highs  = df["High"].astype(float).values
    lows   = df["Low"].astype(float).values
    closes = df["Close"].astype(float).values
    vols   = df["Volume"].astype(float).values
    n      = len(dates)

    e20 = _ema(df["Close"].astype(float), 20).values
    e50 = _ema(df["Close"].astype(float), 50).values

    direction  = prediction.get("direction",  "Sideways")
    confidence = prediction.get("confidence", 50.0)
    label_tag  = prediction.get("label_tag",  "")
    regime     = prediction.get("regime",     {})

    # Per-candle colours
    vol_col = [VOL_BULL if c >= o else VOL_BEAR for c, o in zip(closes, opens)]

    # Predicted candle styling: always solid blue for a cleaner forecast zone
    pc_fill   = PRED_COL
    pc_glow   = f"rgba(77,163,255,{PRED_GLOW_ALPHA})"
    pc_border = PRED_EDGE_COL
    pc_wick   = PRED_WICK_COL

    pc_open  = pred_candle["open"]
    pc_close = pred_candle["close"]
    pc_high  = pred_candle["high"]
    pc_low   = pred_candle["low"]
    pc_date  = pred_candle["date"]

    # ── Candle half-width ─────────────────────────────────────────────
    try:
        gap      = (dates[-1] - dates[-2]).total_seconds() / 86400 if n >= 2 else 1.0
        half_day = pd.Timedelta(days=gap * 0.36)
        glow_day = pd.Timedelta(days=gap * 0.52)   # wider glow
    except Exception:
        half_day = pd.Timedelta(hours=9)
        glow_day = pd.Timedelta(hours=13)

    # ── Figure ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.025,
    )

    # ── Candlesticks (PART 1 upgrade: thicker bodies + wicks) ─────────
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens, high=highs, low=lows, close=closes,
            increasing=dict(
                line=dict(color=BULL, width=1.5),
                fillcolor=BULL,
            ),
            decreasing=dict(
                line=dict(color=BEAR, width=1.5),
                fillcolor=BEAR,
            ),
            name=symbol.replace(".NS", ""),
            showlegend=True,
            opacity=1.0,
        ),
        row=1, col=1,
    )

    # ── EMA 20  (amber, short-term) ───────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates, y=e20,
            mode="lines",
            line=dict(color=EMA20_COL, width=2.0),
            name="EMA 20",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── EMA 50  (blue, mid-term) — PART 2 new ─────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates, y=e50,
            mode="lines",
            line=dict(color=EMA50_COL, width=1.6, dash="dot"),
            name="EMA 50",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Volume bars ───────────────────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=dates, y=vols,
            marker_color=vol_col,
            marker_line_width=0,
            showlegend=False,
            name="Volume",
        ),
        row=2, col=1,
    )

    # ── Vertical separator (prediction zone boundary) ─────────────────
    sep_x = pd.Timestamp(pc_date - half_day * 1.5).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=sep_x,
        x1=sep_x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="rgba(142,197,255,0.28)", width=1.15, dash="dot"),
    )

    # ── Predicted candle — PART 3 upgrade ────────────────────────────
    body_y0 = min(pc_open, pc_close)
    body_y1 = max(pc_open, pc_close)
    if body_y1 <= body_y0:
        body_y1 = body_y0 + abs(pc_high - pc_low) * 0.05 + 1e-6

    # Layer 1: outer glow beneath the projected candle
    fig.add_shape(
        type="rect",
        x0=pc_date - glow_day, x1=pc_date + glow_day,
        y0=body_y0, y1=body_y1,
        fillcolor=pc_glow,
        line=dict(color=pc_glow, width=0),
        row=1, col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=[pc_date],
            open=[pc_open], high=[pc_high], low=[pc_low], close=[pc_close],
            increasing=dict(
                line=dict(color=pc_border, width=PRED_BORDER_W),
                fillcolor=pc_fill,
            ),
            decreasing=dict(
                line=dict(color=pc_border, width=PRED_BORDER_W),
                fillcolor=pc_fill,
            ),
            whiskerwidth=0.9,
            name="Projection",
            showlegend=True,
            opacity=1.0,
        ),
        row=1, col=1,
    )

    # Wick reinforcement keeps the projected bar crisp above the glow.
    fig.add_shape(
        type="line",
        x0=pc_date, x1=pc_date,
        y0=pc_low, y1=pc_high,
        line=dict(color=pc_wick, width=3.0),
        row=1, col=1,
    )

    # ── Clean projection path marker ──────────────────────────────────
    fig.add_shape(
        type="line",
        x0=dates[-1], x1=pc_date,
        y0=float(closes[-1]), y1=float(pc_close),
        line=dict(color=PRED_PATH_COL, width=2.0, dash="dot"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[pc_date],
            y=[pc_close],
            mode="markers",
            marker=dict(
                size=11,
                color=pc_fill,
                line=dict(color=pc_border, width=1.6),
                symbol="diamond",
            ),
            name="Projected Close",
            showlegend=False,
            hovertemplate=(
                "Projected Close: %{y:.2f}<br>"
                f"Projected Open: {pc_open:.2f}<br>"
                f"Projected High: {pc_high:.2f}<br>"
                f"Projected Low: {pc_low:.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # ── Layout (PART 6+7) ─────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG,
        height=600,
        margin=dict(l=4, r=60, t=52, b=4),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0, y=1.045,
            font=dict(color=LABEL, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        title=dict(
            text=(
                f"<b style='color:{LABEL};font-size:15px;'>"
                f"{symbol.replace('.NS','')}"
                f"</b>"
                f"<span style='color:{TICK};font-size:11px;'>"
                f"  ·  Daily  ·  {len(df)} sessions</span>"
            ),
            x=0.0, xanchor="left",
            font=dict(size=14),
        ),
        # Price y-axis
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1,
            zeroline=False, side="right",
            tickfont=dict(color=TICK, size=10),
            tickformat=",.0f",
        ),
        # Volume y-axis
        yaxis2=dict(
            showgrid=False, zeroline=False, side="right",
            tickfont=dict(color=TICK, size=9),
        ),
        # Date axis (main)
        xaxis=dict(
            showgrid=True, gridcolor=GRID_MINOR, gridwidth=1,
            zeroline=False,
            tickfont=dict(color=TICK, size=10),
            type="date",
            # PART 6: range selector buttons
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M",  step="month", stepmode="backward"),
                    dict(count=2, label="2M",  step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#0d1520",
                activecolor="#1e3a5f",
                bordercolor="#1e3a5f",
                font=dict(color=SUBTEXT, size=10),
                x=0, y=1.0,
                xanchor="left",
            ),
        ),
        # Date axis (volume)
        xaxis2=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(color=TICK, size=10),
            type="date",
        ),
        # Hover
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0d1520",
            bordercolor="#1e3a5f",
            font=dict(color=LABEL, size=11),
        ),
    )

    # PART 1: thick wicks
    fig.update_traces(
        selector=dict(type="candlestick"),
        whiskerwidth=0.8,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.pc-pred-box { border-radius:14px; padding:20px 24px; }
.pc-h3 {
    font-size:12px; font-weight:700; color:#4a6480;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:10px;
}
.pc-sig-outer {
    background:#0d1520; border-radius:4px; height:8px;
    overflow:hidden; margin-top:3px; position:relative;
}
.pc-regime-pill {
    display:inline-flex; align-items:center; gap:6px;
    border-radius:8px; padding:5px 12px;
    font-size:12px; font-weight:700; border-width:1.5px; border-style:solid;
}
.pc-stat {
    background:#0b1017; border:1px solid #1e3a5f; border-radius:10px;
    padding:11px 14px; text-align:center;
}
.pc-error {
    background:#1a0b10; border:1.5px solid #ff3b5c; border-radius:12px;
    padding:18px 22px;
}
.pc-warn {
    background:#1a1508; border:1.5px solid #f0b429; border-radius:12px;
    padding:18px 22px;
}
</style>
"""

def _css() -> None:
    if not st.session_state.get("_pc2_css"):
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["_pc2_css"] = True


def _normalize_prediction_chart_symbol(value: object) -> str:
    try:
        return str(value or "").strip().upper().replace(".NS", "")
    except Exception:
        return ""


def _normalize_prediction_chart_imports(values: object, limit: int = 20) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    try:
        raw_values = list(values or [])
    except Exception:
        raw_values = []
    for raw in raw_values:
        symbol = _normalize_prediction_chart_symbol(raw)
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
        if len(symbols) >= limit:
            break
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL BARS
# ─────────────────────────────────────────────────────────────────────────────

_SIG_LABELS = {
    "ema20_slope":      "EMA20 Slope",
    "ema_slope":        "EMA Slope",
    "ema_alignment":    "EMA20 vs EMA50",
    "price_vs_ema20":   "Price vs EMA20",
    "price_vs_ema":     "Price vs EMA20",
    "candle_direction": "Last 3 Candles",
    "body_strength":    "Body Strength",
    "volume_trend":     "Volume Trend",
    "volume_confirm":   "Volume Confirm",
    "volatility":       "Volatility",
    "momentum":         "Momentum",
    "sector_strength":  "MTF Strength",
    "bullish_pct":      "Bullish Breadth",
    "money_flow":       "Money Flow",
    "participation":    "Participation",
}
_SIG_WEIGHTS = {
    "ema20_slope": 0.25, "ema_alignment": 0.20, "price_vs_ema20": 0.15,
    "candle_direction": 0.20, "volume_trend": 0.12, "volatility": 0.08,
    "ema_slope": 0.10, "price_vs_ema": 0.08, "body_strength": 0.07,
    "volume_confirm": 0.10, "momentum": 0.04, "sector_strength": 0.12,
    "bullish_pct": 0.12, "money_flow": 0.08, "participation": 0.08,
}


def _center_signal_value(value: float) -> float:
    value_f = _sf(value, 0.0)
    if abs(value_f) <= 1.5:
        return _clamp(value_f)
    return _clamp((value_f - 50.0) / 50.0)


def _render_signal_bars(signals: dict, weights: dict | None = None) -> None:
    st.markdown("<div class='pc-h3'>Signal Breakdown</div>", unsafe_allow_html=True)
    for key, val in signals.items():
        label  = _SIG_LABELS.get(key, key)
        weight = (weights or _SIG_WEIGHTS).get(key, _SIG_WEIGHTS.get(key, 0.0))
        centered = _center_signal_value(_sf(val, 0.0))
        col    = BULL if centered > 0.08 else BEAR if centered < -0.08 else SIDE
        disp = f"{_sf(val, 0.0):.0f}/100" if abs(_sf(val, 0.0)) > 1.5 else f"{_sf(val, 0.0):+.2f}"

        if centered >= 0:
            bar_left = "50%"
            bar_w    = f"{min(centered * 50.0, 50.0):.1f}%"
        else:
            bp       = min(abs(centered) * 50.0, 50.0)
            bar_left = f"{50.0 - bp:.1f}%"
            bar_w    = f"{bp:.1f}%"

        st.markdown(
            f"""
            <div style="margin-bottom:9px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;color:#8ab4d8;">{label}</span>
                <span style="font-size:11px;color:{col};font-weight:700;">
                  {disp}
                  <span style="color:#2a5080;font-size:10px;"> ({weight*100:.0f}%)</span>
                </span>
              </div>
              <div class="pc-sig-outer">
                <div style="position:absolute;top:0;left:50%;width:1px;
                     height:100%;background:#1e3a5f;"></div>
                <div style="position:absolute;top:0;left:{bar_left};
                     width:{bar_w};height:100%;background:{col};
                     opacity:0.9;border-radius:4px;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# REGIME PILL WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def _render_regime_pill(regime: dict) -> None:
    if not regime:
        return
    r_name = regime.get("regime",      "—")
    r_emj  = regime.get("emoji",       "")
    r_col  = regime.get("color",       SIDE)
    r_desc = regime.get("description", "")
    r_bg   = r_col.replace("#", "") if r_col.startswith("#") else "4a6480"
    st.markdown(
        f"""
        <div style="margin-bottom:12px;">
          <span class="pc-regime-pill"
                style="background:rgba(0,0,0,0.3);
                       border-color:{r_col};color:{r_col};">
            {r_emj} {r_name}
          </span>
          <span style="font-size:11px;color:#4a6480;margin-left:8px;">{r_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_unified_regime_pill(prediction: dict) -> dict:
    try:
        regime_key = str(prediction.get("regime", "UNKNOWN") or "UNKNOWN").strip().upper()
        indicators = dict(prediction.get("indicators") or {})
        ema_aligned = _sf(indicators.get("EMA20"), 0.0) >= _sf(indicators.get("EMA50"), 0.0)
        mapping = {
            "TRENDING_UP": ("Trending Up", "Trend-following backdrop with higher bullish follow-through.", BULL, "▲"),
            "TRENDING_DOWN": ("Trending Down", "Risk-off backdrop. Bearish setups deserve more respect.", BEAR, "▼"),
            "RANGE_BOUND": ("Range Bound", "Mean reversion regime. Breakouts need extra confirmation.", SIDE, "◆"),
            "HIGH_VOLATILITY": ("High Volatility", "Wide ranges reduce forecast stability and widen risk.", "#f0b429", "⚡"),
        }
        name, desc, color, emoji = mapping.get(
            regime_key,
            ("Unknown", "Regime model could not classify the market backdrop.", SIDE, "•"),
        )
        return {
            "regime": name,
            "description": desc,
            "color": color,
            "emoji": emoji,
            "ema_aligned": ema_aligned,
        }
    except Exception:
        return {
            "regime": "Unknown",
            "description": "",
            "color": SIDE,
            "emoji": "•",
            "ema_aligned": False,
        }


def _load_ticker_feedback_history(symbol: str) -> pd.DataFrame:
    try:
        raw = _read_feedback_log()
        if raw is None or raw.empty:
            return pd.DataFrame()
        plain = str(symbol or "").strip().upper().replace(".NS", "")
        df = raw.copy()
        df["_symbol_norm"] = df.get("symbol", "").astype(str).str.upper().str.replace(".NS", "", regex=False)
        df = df[df["_symbol_norm"] == plain].copy()
        if df.empty:
            return pd.DataFrame()
        df["logged_at"] = pd.to_datetime(df.get("logged_at"), errors="coerce")
        df = df.sort_values("logged_at", ascending=False)
        df["_prediction_score"] = pd.to_numeric(df.get("prediction_score"), errors="coerce")
        df["_final_score"] = pd.to_numeric(df.get("final_score"), errors="coerce")
        df["_confidence"] = ((df["_prediction_score"].fillna(0) + df["_final_score"].fillna(df["_prediction_score"].fillna(0))) / 2.0).clip(0, 100)
        df["_return"] = pd.to_numeric(df.get("actual_next_return_pct"), errors="coerce")
        df["_direction"] = np.where(
            df.get("pred_bullish", "").astype(str).str.strip().isin(["1", "1.0", "true", "True"]),
            "Bullish",
            "Bearish",
        )
        return df
    except Exception:
        return pd.DataFrame()


def _render_feedback_overlay(symbol: str, latest_prediction: dict) -> None:
    try:
        history = _load_ticker_feedback_history(symbol)
        with st.expander("How did my past predictions do?", expanded=False):
            if history.empty:
                st.caption("No logged prediction history for this ticker yet.")
                return

            validated = history[history.get("correct", "").astype(str).isin(["True", "False"])].copy()
            win_rate = float((validated["correct"] == "True").mean() * 100.0) if not validated.empty else 0.0
            avg_hist_conf = float(validated["_confidence"].mean()) if not validated.empty else 0.0
            current_conf = _sf(latest_prediction.get("confidence"), 0.0)
            calibration_gap = abs(avg_hist_conf - win_rate) if not validated.empty else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Ticker Win Rate", f"{win_rate:.1f}%")
            c2.metric("Validated Calls", str(int(len(validated))))
            c3.metric("Current Confidence", f"{current_conf:.1f}%")

            recent = history.head(10).copy()
            recent["Date"] = recent["logged_at"].dt.strftime("%Y-%m-%d")
            recent["Confidence"] = recent["_confidence"].round(1)
            recent["Return %"] = recent["_return"].round(2)
            recent["Correct"] = recent.get("correct", "").replace("", "-")
            display = recent[["Date", "_direction", "Confidence", "Return %", "Correct"]].rename(
                columns={"_direction": "Direction"}
            )
            st.dataframe(display, hide_index=True, width="stretch")

            st.markdown(
                f"""
                <div style="background:#0b1017;border:1px solid #1e3a5f;border-radius:12px;padding:14px 16px;margin-top:10px;">
                  <div style="font-size:11px;color:#8ab4d8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;">
                    Model confidence vs actual accuracy
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div>
                      <div style="font-size:10px;color:#4a6480;margin-bottom:4px;">Average model confidence</div>
                      {_conf_bar_html(avg_hist_conf, "#4da3ff")}
                      <div style="font-size:11px;color:#8ab4d8;margin-top:4px;">{avg_hist_conf:.1f}%</div>
                    </div>
                    <div>
                      <div style="font-size:10px;color:#4a6480;margin-bottom:4px;">Actual hit rate</div>
                      {_conf_bar_html(win_rate, "#00d4a8")}
                      <div style="font-size:11px;color:#8ab4d8;margin-top:4px;">{win_rate:.1f}%</div>
                    </div>
                  </div>
                  <div style="font-size:10px;color:#4a6480;margin-top:10px;">Calibration gap: {calibration_gap:.1f} pts</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if win_rate > 65.0:
                st.success("✅ High-confidence ticker — historically accurate")
            elif validated.empty:
                st.info("This ticker has history, but not enough validated outcomes yet.")
            elif win_rate < 40.0:
                st.warning("⚠️ Low win rate on this ticker — reduce position size")
    except Exception:
        return


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_prediction_chart_section(ticker_list: list[str] | None = None) -> None:
    """
    Render the full "📊 Prediction Chart Tomorrow" panel.
    Call from app.py when  pred_chart_show_panel  is True.
    """
    _css()

    # ── Fallback ticker list ──────────────────────────────────────────
    if not ticker_list:
        ticker_list = [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR",
            "SBIN","BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK",
            "ASIANPAINT","MARUTI","BAJFINANCE","HCLTECH","SUNPHARMA",
            "TITAN","ULTRACEMCO","ONGC","NESTLEIND","WIPRO","POWERGRID",
            "NTPC","TECHM","INDUSINDBK","ADANIPORTS","TATAMOTORS",
            "JSWSTEEL","BAJAJFINSV","HINDALCO","GRASIM","DIVISLAB",
            "CIPLA","DRREDDY","BPCL","EICHERMOT","APOLLOHOSP",
            "TATACONSUM","BRITANNIA","COALINDIA","HEROMOTOCO","SHREECEM",
            "SBILIFE","HDFCLIFE","ADANIENT","BAJAJ-AUTO","TATASTEEL",
            "UPL","M&M",
        ]

    display_tickers = sorted(
        {
            _normalize_prediction_chart_symbol(t)
            for t in ticker_list
            if _normalize_prediction_chart_symbol(t)
        }
    )
    imported_symbols = _normalize_prediction_chart_imports(
        st.session_state.get("prediction_chart_imported_symbols", []),
        limit=20,
    )
    if imported_symbols:
        display_tickers = imported_symbols + [ticker for ticker in display_tickers if ticker not in imported_symbols]

    focus_symbol = _normalize_prediction_chart_symbol(
        st.session_state.pop("prediction_chart_focus_symbol", "")
    )
    if not focus_symbol:
        focus_symbol = _normalize_prediction_chart_symbol(st.session_state.get("pc_loaded_symbol", ""))

    if display_tickers:
        default_symbol = (
            focus_symbol
            if focus_symbol in display_tickers
            else imported_symbols[0]
            if imported_symbols
            else "RELIANCE"
            if "RELIANCE" in display_tickers
            else display_tickers[0]
        )

    # ── Header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
          <span style="font-size:30px;">📊</span>
          <div>
            <div style="font-size:22px;font-weight:900;color:#ccd9e8;
                 letter-spacing:-0.3px;">
              Prediction Chart Tomorrow
            </div>
            <div style="font-size:12px;color:#4a6480;">
              TradingView-style · EMA20+50 · Regime detection ·
              Chart-driven AI prediction
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1e3a5f;margin:12px 0;'>",
        unsafe_allow_html=True,
    )

    # ── PART 9: dependency guards (styled) ────────────────────────────
    if not _YF_OK:
        st.markdown(
            """<div class="pc-error">
              <b style="color:#ff3b5c;">⚠ yfinance not installed</b><br>
              <span style="color:#8ab4d8;font-size:12px;">
                Add <code>yfinance</code> to requirements.txt and redeploy.
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
        return
    if not _PLOTLY_OK:
        st.markdown(
            """<div class="pc-error">
              <b style="color:#ff3b5c;">⚠ plotly not installed</b><br>
              <span style="color:#8ab4d8;font-size:12px;">
                Add <code>plotly</code> to requirements.txt and redeploy.
              </span>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # ── Search row ────────────────────────────────────────────────────
    if imported_symbols:
        import_origin = str(st.session_state.get("prediction_chart_import_origin", "Mode scan") or "Mode scan")
        import_mode = st.session_state.get("prediction_chart_import_mode", "")
        import_mode_label = f" | Mode M{import_mode}" if str(import_mode).strip() else ""
        visible_imports = imported_symbols[:6]
        st.markdown(
            f"""
            <div style="background:#0b1017;border:1.5px solid #1e3a5f;border-radius:14px;padding:12px 14px;margin-bottom:14px;">
              <div style="font-size:10px;color:#4a6480;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">Imported AI Watchlist</div>
              <div style="font-size:13px;color:#8ab4d8;">
                Imported from <span style="color:#ccd9e8;font-weight:700;">{import_origin}</span>{import_mode_label}. Quick-load any imported symbol below.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        quick_cols = st.columns(len(visible_imports) + 1, gap="small")
        for idx, symbol in enumerate(visible_imports):
            if quick_cols[idx].button(symbol, key=f"pc_imported_quick_{symbol}", width="stretch"):
                st.session_state["pc_loaded_symbol"] = symbol
                st.session_state["prediction_chart_focus_symbol"] = symbol
                st.rerun()
        if quick_cols[-1].button("Clear Imported", key="pc_imported_clear_btn", width="stretch"):
            for key in (
                "prediction_chart_imported_symbols",
                "prediction_chart_import_origin",
                "prediction_chart_import_mode",
                "prediction_chart_focus_symbol",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        if len(imported_symbols) > len(visible_imports):
            st.caption(f"Showing {len(visible_imports)} of {len(imported_symbols)} imported symbols.")

    c_sel, c_btn = st.columns([4, 1])
    with c_sel:
        widget_key = "pc_stock_select"
        current_selected = _normalize_prediction_chart_symbol(st.session_state.get(widget_key, ""))
        initial_symbol = current_selected if current_selected in display_tickers else default_symbol
        needs_reset = False
        if current_selected not in display_tickers:
            needs_reset = True
        elif focus_symbol and focus_symbol in display_tickers and current_selected != focus_symbol:
            initial_symbol = focus_symbol
            needs_reset = True

        if needs_reset:
            st.session_state.pop(widget_key, None)

        selectbox_kwargs = {
            "label": "stock",
            "options": display_tickers,
            "key": widget_key,
            "label_visibility": "collapsed",
        }
        if widget_key not in st.session_state:
            selectbox_kwargs["index"] = display_tickers.index(initial_symbol)
        selected_bare = st.selectbox(**selectbox_kwargs)
    with c_btn:
        load_btn = st.button(
            "Load Chart",
            key="pc_load_btn",
            width="stretch",
            type="primary",
        )

    st.markdown(
        '<div style="font-size:11px;color:#2a5080;margin-bottom:14px;">'
        "Select any NSE stock · Load Chart fetches real data and generates prediction"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Symbol tracking ───────────────────────────────────────────────
    st.caption("Shared flow: ALL_DATA first, feature cache second, yfinance last.")
    trigger_symbol = st.session_state.get("pc_loaded_symbol")
    if load_btn:
        try:
            fetch_stock_data.clear()
        except Exception:
            pass
        st.session_state["pc_loaded_symbol"] = selected_bare
        trigger_symbol = selected_bare

    if not trigger_symbol:
        st.markdown(
            """
            <div style="background:#0b1017;border:1.5px solid #1e3a5f;
                 border-radius:14px;padding:56px;text-align:center;margin-top:8px;">
              <div style="font-size:44px;margin-bottom:14px;">📈</div>
              <div style="font-size:17px;font-weight:700;color:#8ab4d8;">
                Select a stock and click Load Chart
              </div>
              <div style="font-size:12px;color:#4a6480;margin-top:6px;">
                Uses shared market data with live/session-aware caching
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    symbol_ns = _normalise_symbol(trigger_symbol)
    tt_key = str(_get_feature_tt_date() or "live")
    data_signature = f"{_get_market_data_signature()}::{tt_key}"
    force_refresh = False
    if _get_feature_window() == "LIVE":
        refresh_col, note_col = st.columns([1.2, 3.0])
        with refresh_col:
            force_refresh = st.button("Refresh Data", key=f"pc_refresh_{symbol_ns}", width="stretch")
        with note_col:
            st.caption("Live window: refresh clears the chart cache and reloads the latest shared market data.")
        if force_refresh:
            try:
                fetch_stock_data.clear()
            except Exception:
                pass
    else:
        st.caption("Data locked until next market session.")

    # ── Fetch ──────────────────────────────────────────────────────────
    with st.spinner(f"Loading {trigger_symbol}…"):
        df = fetch_stock_data(
            symbol_ns,
            timeframe="1d",
            data_signature=data_signature,
            time_context_key=tt_key,
            force_refresh=force_refresh,
        )
    _render_data_status_badge(_get_chart_status(symbol_ns), label=trigger_symbol)

    # PART 9: styled error cards
    if df is None:
        st.markdown(
            f"""
            <div class="pc-error">
              <b style="color:#ff3b5c;">⚠ Could not load data for {trigger_symbol}</b><br>
              <span style="color:#8ab4d8;font-size:12px;">
                Possible causes: symbol not found on NSE · no internet connection ·
                yfinance API timeout.  Try a different stock or refresh the page.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if len(df) < 30:
        st.markdown(
            f"""
            <div class="pc-warn">
              <b style="color:#f0b429;">⚠ Insufficient data for {trigger_symbol}</b><br>
              <span style="color:#8ab4d8;font-size:12px;">
                Only {len(df)} candles returned — minimum 30 required.
                This stock may be newly listed or have low trading activity.
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Prediction + regime ────────────────────────────────────────────
    try:
        prediction = dict(_get_cached_prediction(trigger_symbol) or {})
    except Exception:
        prediction = {}

    if not prediction:
        try:
            prediction = dict(
                _get_tomorrow_prediction(
                    trigger_symbol,
                    {
                        trigger_symbol: df,
                        symbol_ns: df,
                        trigger_symbol.upper().replace(".NS", ""): df,
                    },
                    "prediction_chart",
                )
                or {}
            )
        except Exception:
            prediction = {}

    if not prediction:
        prediction = compute_prediction(df)
    else:
        prediction = {
            "direction": prediction.get("direction", "Sideways"),
            "confidence": _sf(prediction.get("confidence"), 48.0),
            "signals": dict(prediction.get("signals") or {}),
            "atr": _sf(prediction.get("atr"), float((df["High"] - df["Low"]).tail(14).mean())),
            "regime": dict(prediction.get("regime_snapshot") or _build_unified_regime_pill(prediction)),
            "label_tag": str(prediction.get("label_tag", "") or ""),
            "weights": dict(prediction.get("weights") or {}),
            "score": _sf(prediction.get("score"), 50.0),
            "key_signal": str(prediction.get("key_signal", "") or ""),
        }
    pred_candle = build_predicted_candle(df, prediction)
    regime      = prediction.get("regime", {})

    direction  = prediction["direction"]
    confidence = prediction["confidence"]
    label_tag  = prediction.get("label_tag", "")
    signals    = prediction["signals"]
    atr        = prediction["atr"]

    p_color = {"Bullish": BULL, "Bearish": BEAR}.get(direction, SIDE)
    p_bg    = {"Bullish": "#091a10", "Bearish": "#1a0b10"}.get(direction, "#0b1220")
    p_icon  = {"Bullish": "🟢", "Bearish": "🔴", "Sideways": "🟡"}[direction]

    # ── Chart ──────────────────────────────────────────────────────────
    with st.spinner("Building chart…"):
        fig = build_chart(df, symbol_ns, prediction, pred_candle)

    if fig is None:
        st.error("Chart failed to render — check plotly installation.")
        return

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "toImage", "select2d", "lasso2d", "autoScale2d",
            ],
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    # ── Prediction card + Signals ──────────────────────────────────────
    c_pred, c_sigs = st.columns([5, 4])

    with c_pred:
        last_close = float(df["Close"].iloc[-1])
        proj_close = pred_candle["close"]
        proj_ret   = (proj_close / last_close - 1.0) * 100.0
        ret_sign   = "+" if proj_ret >= 0 else ""

        # PART 8: upgraded label
        full_label = f"{direction} ({confidence:.0f}%) — {label_tag}"

        st.markdown(
            f"""
            <div class="pc-pred-box"
                 style="background:{p_bg};border:2px solid {p_color};">
              <div style="font-size:10px;color:#4a6480;text-transform:uppercase;
                   letter-spacing:.09em;margin-bottom:10px;">
                Tomorrow's Prediction
              </div>
              <div style="display:flex;align-items:center;gap:18px;margin-bottom:12px;">
                <span style="font-size:48px;line-height:1;">{p_icon}</span>
                <div>
                  <div style="font-size:28px;font-weight:900;
                       color:{p_color};line-height:1;">{direction}</div>
                  <div style="font-size:13px;color:#8ab4d8;margin-top:3px;">
                    <b style="color:{p_color};">{confidence:.0f}%</b> confidence
                    — {label_tag}
                  </div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
                <div>
                  <div style="font-size:10px;color:#4a6480;">Current</div>
                  <div style="font-size:15px;font-weight:700;
                       color:#ccd9e8;">₹{last_close:,.2f}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#4a6480;">Projected</div>
                  <div style="font-size:15px;font-weight:700;
                       color:{p_color};">₹{proj_close:,.2f}</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#4a6480;">Est. Move</div>
                  <div style="font-size:15px;font-weight:700;
                       color:{p_color};">{ret_sign}{proj_ret:.2f}%</div>
                </div>
              </div>
              <div style="margin-top:12px;padding-top:10px;
                   border-top:1px solid rgba(255,255,255,0.06);">
                <div style="display:flex;gap:16px;flex-wrap:wrap;">
                  <div>
                    <span style="font-size:10px;color:#4a6480;">ATR(14) </span>
                    <span style="font-size:12px;font-weight:700;
                          color:#8ab4d8;">₹{atr:,.2f}</span>
                  </div>
                  <div>
                    <span style="font-size:10px;color:#4a6480;">Candles </span>
                    <span style="font-size:12px;font-weight:700;
                          color:#8ab4d8;">{len(df)}</span>
                  </div>
                  <div>
                    <span style="font-size:10px;color:#4a6480;">Last date </span>
                    <span style="font-size:12px;font-weight:700;
                          color:#8ab4d8;">{str(df.index[-1])[:10]}</span>
                  </div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:10px;color:#2a3a50;">
                ⚠ Projected candle is a visual estimate — not a real forecast.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Regime pill below prediction card
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        _render_regime_pill(regime)

    with c_sigs:
        _render_signal_bars(signals, prediction.get("weights"))

        key_signal = str(prediction.get("key_signal", "") or "").replace("_", " ").strip().title()
        if key_signal:
            st.markdown(
                f"""
                <div style="background:#0b1017;border:1px solid #1e3a5f;
                     border-radius:10px;padding:10px 14px;margin-top:8px;">
                  <div style="font-size:10px;color:#4a6480;margin-bottom:4px;">
                    Strongest Driver Today</div>
                  <div style="font-size:12px;font-weight:700;color:#00d4ff;">
                    {key_signal}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # EMA alignment summary
        ema_ok  = regime.get("ema_aligned", False)
        ema_col = BULL if ema_ok else BEAR
        ema_txt = "EMA20 above EMA50 — Bullish structure" if ema_ok \
                  else "EMA20 below EMA50 — Bearish structure"
        st.markdown(
            f"""
            <div style="background:#0b1017;border:1px solid #1e3a5f;
                 border-radius:10px;padding:10px 14px;margin-top:8px;">
              <div style="font-size:10px;color:#4a6480;margin-bottom:4px;">
                EMA Alignment</div>
              <div style="font-size:12px;font-weight:700;color:{ema_col};">
                {ema_txt}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _render_feedback_overlay(trigger_symbol, prediction)

    st.markdown(
        "<hr style='border:none;border-top:1px solid #1e3a5f;margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # ── Disclaimer ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:11px;color:#2a5080;text-align:center;'
        'padding-bottom:8px;">'
        "Signal-based estimate only · Not financial advice · "
        "Data via yfinance · For educational and research use"
        "</div>",
        unsafe_allow_html=True,
    )
