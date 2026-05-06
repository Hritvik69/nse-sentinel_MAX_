"""
sector_regime_engine.py
════════════════════════
Module 1 — Market Regime Detection.

Classifies the current market regime daily from index OHLC already in
ALL_DATA (zero API calls when available; falls back to yfinance once).

Regimes
───────
  TRENDING_UP      EMA20 > EMA50, positive slope, low-medium volatility
  TRENDING_DOWN    EMA20 < EMA50, negative slope, low-medium volatility
  RANGE_BOUND      EMA20 ≈ EMA50, low ADX proxy, compressed range
  HIGH_VOLATILITY  Realized vol >> baseline; ADX unreliable

Public API
──────────
  detect_regime(all_data)  → RegimeState
  regime_weight_adjustments(regime) → dict[str, float]   # signal multipliers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ── Index universe (priority order) ───────────────────────────────────
_INDEX_TICKERS = ["^NSEI", "NIFTY_50.NS", "%5ENSEI"]
_STOCK_PROXIES  = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "INFY.NS", "TCS.NS", "LT.NS",
]
_MIN_ROWS = 50


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    regime:          str   = "RANGE_BOUND"   # canonical 4 labels
    confidence:      float = 50.0            # 0–100
    adx_proxy:       float = 20.0            # 0–100
    realized_vol:    float = 1.0             # % daily std (annualized / √252)
    ema_separation:  float = 0.0             # EMA20 − EMA50 as % of price
    ema_slope:       float = 0.0             # EMA20 slope (5-bar)
    vol_expansion:   float = 1.0             # short_vol / long_vol
    computed_at:     str   = ""


# ── Weight multipliers by regime ──────────────────────────────────────
# Values < 1 suppress a signal group; > 1 amplify it.
_REGIME_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "TRENDING_UP": {
        "ema_slope":        1.30,
        "price_vs_ema":     1.20,
        "candle_direction": 1.15,
        "body_strength":    1.10,
        "consecutive":      1.10,
        "volume_confirm":   1.10,
        "volatility":       0.80,
        "momentum":         1.20,
        "sector_strength":  1.15,
        "bullish_pct":      1.10,
        "money_flow":       1.15,
        "participation":    1.05,
        "_confidence_cap":  95.0,
        "_sideways_bias":   0.0,
    },
    "TRENDING_DOWN": {
        "ema_slope":        1.30,
        "price_vs_ema":     1.20,
        "candle_direction": 1.15,
        "body_strength":    1.10,
        "consecutive":      1.10,
        "volume_confirm":   1.05,
        "volatility":       0.80,
        "momentum":         1.20,
        "sector_strength":  0.90,
        "bullish_pct":      0.85,
        "money_flow":       0.90,
        "participation":    0.90,
        "_confidence_cap":  90.0,
        "_sideways_bias":   0.0,
    },
    "RANGE_BOUND": {
        # Reduce trend signals, boost mean-reversion proxies
        "ema_slope":        0.55,
        "price_vs_ema":     1.20,   # distance from EMA = reversion trigger
        "candle_direction": 0.70,
        "body_strength":    0.80,
        "consecutive":      0.60,
        "volume_confirm":   0.90,
        "volatility":       1.30,   # compression matters more
        "momentum":         0.65,
        "sector_strength":  1.00,
        "bullish_pct":      1.00,
        "money_flow":       1.10,
        "participation":    1.00,
        "_confidence_cap":  72.0,   # range is harder to predict
        "_sideways_bias":   12.0,   # add to composite → pushes toward Sideways
    },
    "HIGH_VOLATILITY": {
        # Everything uncertain; dampen all signals
        "ema_slope":        0.60,
        "price_vs_ema":     0.60,
        "candle_direction": 0.70,
        "body_strength":    0.80,
        "consecutive":      0.55,
        "volume_confirm":   0.90,
        "volatility":       0.55,
        "momentum":         0.65,
        "sector_strength":  0.80,
        "bullish_pct":      0.80,
        "money_flow":       0.85,
        "participation":    0.85,
        "_confidence_cap":  62.0,   # hard cap
        "_sideways_bias":   8.0,
    },
}


def regime_weight_adjustments(regime: str) -> dict[str, float]:
    """Return the signal multiplier dict for a given regime."""
    return dict(_REGIME_ADJUSTMENTS.get(regime, _REGIME_ADJUSTMENTS["RANGE_BOUND"]))


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False).mean()


def _adx_proxy(close: pd.Series, n: int = 14) -> float:
    """
    Lightweight ADX proxy using directional movement of closes only.
    Returns 0–100; values > 25 suggest trending.
    """
    if len(close) < n + 5:
        return 20.0
    diff = close.diff()
    up   = diff.clip(lower=0)
    dn   = (-diff).clip(lower=0)
    sm_u = up.ewm(com=n - 1, adjust=False).mean()
    sm_d = dn.ewm(com=n - 1, adjust=False).mean()
    dx   = ((sm_u - sm_d).abs() / (sm_u + sm_d + 1e-9) * 100).iloc[-1]
    return float(np.clip(dx, 0, 100))


def _realized_vol(close: pd.Series, n: int = 20) -> float:
    """Daily return std over last n bars, annualised (%)."""
    if len(close) < n + 1:
        return 1.0
    rets = np.log(close / close.shift(1)).dropna()
    return float(rets.tail(n).std() * math.sqrt(252) * 100)


def _get_index_df(all_data: dict) -> pd.DataFrame | None:
    """
    Try index tickers first, then fall back to a synthetic equal-weight
    close built from large-cap proxies already in ALL_DATA.
    """
    for tk in _INDEX_TICKERS:
        df = all_data.get(tk)
        if df is not None and len(df) >= _MIN_ROWS:
            return df

    # Synthetic from proxies
    frames = []
    min_proxy_rows = max(_MIN_ROWS, int(_MIN_ROWS * 1.2))
    for tk in _STOCK_PROXIES:
        df = all_data.get(tk)
        if df is not None and "Close" in df.columns and len(df) >= min_proxy_rows:
            frames.append(df[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index())

    if not frames:
        return None

    # Align prices on a broader calendar so short proxy histories do not erase
    # older valid data from longer-lived proxies.
    close_stack = pd.concat(
        [f["Close"].rename(str(i)) for i, f in enumerate(frames)], axis=1
    ).sort_index().ffill()
    min_active_proxies = max(2, min(len(frames), 3))
    close_stack = close_stack[close_stack.notna().sum(axis=1) >= min_active_proxies]
    if len(close_stack) < _MIN_ROWS:
        return None
    common = close_stack.index

    agg = {
        col: pd.concat(
            [f[col].reindex(common).sort_index().ffill() for f in frames], axis=1
        ).mean(axis=1, skipna=True)
        for col in ("Open", "High", "Low", "Close")
    }
    agg["Volume"] = pd.concat(
        [f["Volume"].reindex(common) for f in frames], axis=1
    ).sum(axis=1)

    return pd.DataFrame(agg, index=common).dropna()


# ══════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def _classify(
    ema_sep: float,
    ema_slope: float,
    adx: float,
    rv: float,
    vol_exp: float,
) -> tuple[str, float]:
    """
    Map features → (regime, confidence).

    Feature meanings
    ────────────────
    ema_sep   : (EMA20 − EMA50) / price × 100  — signed
    ema_slope : EMA20 5-bar change / price × 100 — signed
    adx       : 0–100 trend strength proxy
    rv        : annualised daily vol (%)
    vol_exp   : recent ATR / long ATR  (>1 = expanding)
    """
    # ── High volatility override ──────────────────────────────────────
    if rv > 28 or vol_exp > 1.8:
        conf = float(np.clip(40 + min(rv - 28, 20) * 1.5, 45, 90))
        return "HIGH_VOLATILITY", round(conf, 1)

    # ── Trending up ───────────────────────────────────────────────────
    if ema_sep > 0.4 and ema_slope > 0.1 and adx > 22:
        conf = float(np.clip(50 + ema_sep * 10 + adx * 0.5, 55, 92))
        return "TRENDING_UP", round(conf, 1)

    # ── Trending down ─────────────────────────────────────────────────
    if ema_sep < -0.4 and ema_slope < -0.1 and adx > 22:
        conf = float(np.clip(50 + abs(ema_sep) * 10 + adx * 0.5, 55, 92))
        return "TRENDING_DOWN", round(conf, 1)

    # ── Range bound ───────────────────────────────────────────────────
    # Low ADX, tight EMA separation, low vol
    range_score = (1 - min(abs(ema_sep) / 1.5, 1)) * 40 + (1 - min(adx / 40, 1)) * 40
    conf_range  = float(np.clip(40 + range_score, 42, 82))
    return "RANGE_BOUND", round(conf_range, 1)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def detect_regime(all_data: dict) -> RegimeState:
    """
    Detect the current market regime from index / proxy OHLC in ALL_DATA.

    Parameters
    ----------
    all_data : dict   From strategy_engines._engine_utils.ALL_DATA

    Returns
    -------
    RegimeState
    """
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    fallback = RegimeState(computed_at=ts)

    try:
        df = _get_index_df(all_data)
        if df is None or len(df) < _MIN_ROWS:
            return fallback

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        e20 = _ema(close, 20)
        e50 = _ema(close, 50)

        price      = float(close.iloc[-1])
        ema_sep    = float((e20.iloc[-1] - e50.iloc[-1]) / (price + 1e-9) * 100)
        ema_slope  = float((e20.iloc[-1] - e20.iloc[-5]) / (e20.iloc[-5] + 1e-9) * 100)
        adx        = _adx_proxy(close)
        rv         = _realized_vol(close)

        # Volatility expansion: 5-bar ATR / 20-bar ATR
        atr_s = float(_atr(high, low, close, 5).iloc[-1])
        atr_l = float(_atr(high, low, close, 20).iloc[-1])
        vol_exp = atr_s / (atr_l + 1e-9)

        regime, conf = _classify(ema_sep, ema_slope, adx, rv, vol_exp)

        return RegimeState(
            regime         = regime,
            confidence     = conf,
            adx_proxy      = round(adx, 1),
            realized_vol   = round(rv, 2),
            ema_separation = round(ema_sep, 3),
            ema_slope      = round(ema_slope, 3),
            vol_expansion  = round(vol_exp, 3),
            computed_at    = ts,
        )
    except Exception:
        return fallback
