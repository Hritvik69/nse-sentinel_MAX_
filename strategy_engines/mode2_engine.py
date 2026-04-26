"""
strategy_engines/mode2_engine.py
──────────────────────────────────
MODE 2 — BALANCED (Swing)

Philosophy: no single dominant signal. Equal weighting across EMA trend,
volume, RSI zone, and price action. Stable trend confirmation.
ML target: next day green (standard).
Backtest entry: vol > 1.3× AND ema20 > ema50 AND RSI 48-72.
Training universe: diversified NSE Nifty-100 mix.
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd

from strategy_engines._engine_utils import (
    safe, ema, rsi_vec, SKLEARN_OK, get_df_for_ticker,
)
if SKLEARN_OK:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

_MODEL:  "LogisticRegression | None" = None  # noqa: F821
_SCALER: "StandardScaler | None"    = None  # noqa: F821
_LOCK   = threading.Lock()
_BT_CACHE: dict[str, float] = {}
_BT_LOCK  = threading.Lock()

_TRAIN_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS","BAJFINANCE.NS","SUNPHARMA.NS","TITAN.NS",
    "ULTRACEMCO.NS","ONGC.NS","NESTLEIND.NS","WIPRO.NS","POWERGRID.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode2(row: dict) -> tuple[float, dict]:
    """
    Balanced scoring: equal weighting across all quality signals.
    No single factor dominates — stable trend confirmation preferred.
    """
    pts: dict[str, int] = {}
    ri    = safe(row.get("RSI",               50))
    vol_r = safe(row.get("Vol / Avg",          1))
    d20h  = safe(row.get("Δ vs 20D High (%)", -5))
    de20  = safe(row.get("Δ vs EMA20 (%)",     0))
    r5d   = safe(row.get("5D Return (%)",       0))
    r20d  = safe(row.get("20D Return (%)",      0))
    price = safe(row.get("Price (₹)",           0))
    e20   = safe(row.get("EMA 20",              0))
    e50   = safe(row.get("EMA 50",              0))

    # RSI — balanced zone
    if 52 <= ri <= 63:    pts["RSI Balanced Zone"] = 14
    elif 63 < ri <= 70:   pts["RSI Upper Balanced"] = 10

    # Volume — moderate weighting
    if   vol_r > 2.0:     pts["Vol >2×"]           = 20
    elif vol_r > 1.5:     pts["Vol >1.5×"]          = 15

    # Near 15D high
    if  -2.0 <= d20h <= 0.0:  pts["Near 15D High"]  = 14
    elif -4.0 <= d20h < -2.0: pts["Approaching High"] = 8

    # EMA structure — equal to volume
    if price > e20 > 0:   pts["Price > EMA20"]      = 14
    if e20 > e50 > 0:     pts["EMA20 > EMA50"]      = 14

    # Both short and long return confirm trend
    if 1.0 <= r5d <= 6.0: pts["5D Return OK"]       = 12
    if r20d > 2.0:        pts["20D Return Positive"] = 8

    # PENALTIES — moderate
    if ri > 70:           pts["RSI Overbought"]      = -15
    if de20 > 6.0:        pts["Overextended"]         = -12
    if r5d > 8.0:         pts["5D Overrun"]           = -10
    if vol_r < 1.2:       pts["Low Volume"]           = -12
    if r20d < -2.0:       pts["Weak 20D Trend"]       = -8

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


def check_bull_trap_mode2(row: dict) -> str:
    """Balanced trap: RSI overbought + weak volume + negative 20D trend."""
    ri    = safe(row.get("RSI",            50))
    vol_r = safe(row.get("Vol / Avg",       1))
    r20d  = safe(row.get("20D Return (%)", 0))
    hits  = sum([ri > 72, vol_r < 1.1, r20d < -1.0])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (Mode 2 entry: balanced trend + vol confirmation)
# ─────────────────────────────────────────────────────────────────────
def backtest_mode2(row: dict, ticker: str) -> float:
    """
    Mode 2 backtest: requires EMA trend + vol + RSI all aligned.
    No single dominant condition — all three must agree.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    with _BT_LOCK:
        if ticker_ns in _BT_CACHE:
            return _BT_CACHE[ticker_ns]

    result = 50.0
    try:
        # BUG FIX: Use get_df_for_ticker (respects TT patch) not download_history
        df = get_df_for_ticker(ticker_ns)
        if df is None or len(df) < 40:
            raise ValueError("insufficient data")

        close     = df["Close"].copy()
        volume    = df["Volume"].copy()
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        rsi_s     = rsi_vec(close)
        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        high_15d  = close.rolling(15, min_periods=7).max().shift(1)

        target_rsi  = safe(row.get("RSI",      60))
        target_volr = safe(row.get("Vol / Avg", 1.5))

        # Mode 2: balanced — all three aligned
        mask = (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 4) &
            (rsi_s    <= target_rsi  + 4) &
            (vol_ratio >= target_volr * 0.80) &
            (vol_ratio <= target_volr * 1.20) &
            (e20s     >  e50s) &                    # EMA trend required
            (high_15d.notna()) &
            (close    >= high_15d * 0.97)            # within 3% of 15D high
        )
        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]
        if len(idx) < 15:
            raise ValueError("too few")

        cv    = close.values
        green = int(sum(cv[i + 1] > cv[i] for i in idx))
        result = round((green / len(idx)) * 100, 1)
    except Exception:
        result = 50.0

    with _BT_LOCK:
        _BT_CACHE[ticker_ns] = result
    return result


# ─────────────────────────────────────────────────────────────────────
# PART 4 — ML (balanced feature set, next-day target)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode2(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 2 balanced features: RSI, vol, EMA, short + long returns.
    Target: next day green. No training filter — all conditions included.
    """
    try:
        if len(close) < 30:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        ema_dist  = (close / e20s.replace(0, np.nan) - 1.0) * 100
        rsi_s     = rsi_vec(close)
        ret5d     = close.pct_change(5) * 100
        ret20d    = close.pct_change(20) * 100
        target    = (close.shift(-1) > close).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_s,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ret_5d":    ret5d,
            "ret_20d":   ret20d,
            "ema_trend": (e20s > e50s).astype(int),
            "target":    target,
        }).dropna()
        # no training filter — balanced
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode2() -> bool:
    global _MODEL, _SCALER
    if not SKLEARN_OK:
        return False
    with _LOCK:
        if _MODEL is not None:
            return True

    all_rows: list[pd.DataFrame] = []
    for t in _TRAIN_TICKERS:
        # BUG FIX: Use get_df_for_ticker (TT-patched) instead of download_history
        # so ML training in Time Travel mode only sees historical data.
        df_h = get_df_for_ticker(t)
        try:
            from time_travel_engine import apply_time_travel_cutoff as _tt_cut_tr
            df_h = _tt_cut_tr(df_h)
        except Exception:
            pass
        if df_h is None:
            continue
        rows = _build_features_mode2(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 80:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ret_5d", "ret_20d", "ema_trend"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        mdl = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced",
                                  solver="lbfgs", random_state=2)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode2 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode2 ML] train failed: {e}")
        return False


def predict_ml_mode2(row: dict) -> float:
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode2()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",             50))
        vol_r = safe(row.get("Vol / Avg",        1))
        de20  = safe(row.get("Δ vs EMA20 (%)",   0))
        r5d   = safe(row.get("5D Return (%)",     0))
        r20d  = safe(row.get("20D Return (%)",    0))
        feat  = np.array([[ri, vol_r, de20, r5d, r20d, 1.0]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # balanced mode: no large adjustments — mild alignment bonus
        adj   = 0.02 if (vol_r > 1.4 and ri > 52) else 0.0
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0