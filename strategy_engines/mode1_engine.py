"""
strategy_engines/mode1_engine.py
──────────────────────────────────
MODE 1 — MOMENTUM (Breakout)

Philosophy: reward explosive volume + price clearing recent highs.
Penalise overextension aggressively.
ML target: price higher 3 days later (momentum holding period).
Backtest entry: vol > 1.5× AND price at/above 10D high.
Training universe: large-cap, high-beta NSE stocks.
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd

from strategy_engines._engine_utils import (
    safe, ema, rsi_vec, download_history, SKLEARN_OK, get_df_for_ticker,
)
if SKLEARN_OK:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

# ── per-mode model / backtest cache ──────────────────────────────────
_MODEL:  "LogisticRegression | None" = None  # noqa: F821
_SCALER: "StandardScaler | None"    = None  # noqa: F821
_LOCK   = threading.Lock()
_BT_CACHE: dict[str, float] = {}
_BT_LOCK  = threading.Lock()

_TRAIN_TICKERS = [
    "TATAMOTORS.NS","ADANIPORTS.NS","HINDALCO.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "BAJFINANCE.NS","ICICIBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS",
    "HCLTECH.NS","WIPRO.NS","TECHM.NS","INFY.NS","TCS.NS",
    "BHARTIARTL.NS","RELIANCE.NS","MARUTI.NS","M&M.NS","TATACONSUM.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode1(row: dict) -> tuple[float, dict]:
    """
    Momentum scoring: heavy volume + near-breakout weighting.
    Overextension (price far from EMA, RSI > 72) penalised most.
    """
    pts: dict[str, int] = {}
    ri    = safe(row.get("RSI",               50))
    vol_r = safe(row.get("Vol / Avg",          1))
    d20h  = safe(row.get("Δ vs 20D High (%)", -5))
    de20  = safe(row.get("Δ vs EMA20 (%)",     0))
    r5d   = safe(row.get("5D Return (%)",       0))
    price = safe(row.get("Price (₹)",           0))
    e20   = safe(row.get("EMA 20",              0))
    e50   = safe(row.get("EMA 50",              0))

    # ── RSI momentum zone (58-70 is ideal for breakout) ──────────────
    if 58 <= ri <= 65:    pts["RSI Momentum Zone"]   = 15
    elif 65 < ri <= 70:   pts["RSI Upper Zone"]       = 10

    # ── Volume — HIGHEST weight for momentum ─────────────────────────
    if   vol_r > 2.5:     pts["Vol >2.5× Explosive"] = 35
    elif vol_r > 2.0:     pts["Vol >2×"]              = 30
    elif vol_r > 1.7:     pts["Vol >1.7×"]            = 22

    # ── Breakout proximity — HIGHEST weight ──────────────────────────
    if  -1.0 <= d20h <= 0.0:  pts["At 20D High"]      = 28
    elif -3.0 <= d20h < -1.0: pts["Near 20D High"]     = 18

    # ── EMA structure ────────────────────────────────────────────────
    if price > e20 > 0:   pts["Price > EMA20"]         = 10
    if e20 > e50 > 0:     pts["EMA20 > EMA50"]         = 8

    # ── Short-term return confirms momentum ──────────────────────────
    if 1.0 <= r5d <= 7.0: pts["5D Return 1-7%"]        = 10

    # ── PENALTIES (aggressive for momentum) ──────────────────────────
    if ri > 72:           pts["RSI Overbought"]         = -25
    if de20 > 5.0:        pts["Overextended EMA"]       = -20
    if r5d > 10.0:        pts["5D >10% Exhaustion"]     = -18
    if vol_r < 1.5:       pts["Weak Volume"]            = -22

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


# ─────────────────────────────────────────────────────────────────────
# PART 2 — BULL TRAP (mode-specific)
# ─────────────────────────────────────────────────────────────────────
def check_bull_trap_mode1(row: dict) -> str:
    """Momentum trap: RSI overbought + vol drying up + overextended."""
    ri    = safe(row.get("RSI",            50))
    vol_r = safe(row.get("Vol / Avg",       1))
    de20  = safe(row.get("Δ vs EMA20 (%)", 0))
    hits  = sum([ri > 74, vol_r < 1.2, de20 > 7.0])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (simulates Mode 1 entry logic)
# Entry: vol_ratio > 1.5 AND price at/above 10D rolling high
# ─────────────────────────────────────────────────────────────────────
def backtest_mode1(row: dict, ticker: str) -> float:
    """
    Simulates Mode 1 breakout entry: high-volume price clearing 10D high.
    Returns % of historical matches where next day closed green.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    with _BT_LOCK:
        if ticker_ns in _BT_CACHE:
            return _BT_CACHE[ticker_ns]

    result = 50.0
    try:
        # BUG FIX: Use get_df_for_ticker (respects time-travel patch) instead
        # of download_history (bypasses it), preventing future-data leakage.
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
        high_10d  = close.rolling(10, min_periods=5).max().shift(1)

        target_rsi  = safe(row.get("RSI",      60))
        target_volr = safe(row.get("Vol / Avg", 1.7))

        # Mode 1 entry: breakout volume + price at/near 10D high + RSI range
        mask = (
            rsi_s.notna() &
            (rsi_s   >= target_rsi  - 3) &
            (rsi_s   <= target_rsi  + 3) &
            (vol_ratio >= max(target_volr * 0.85, 1.4)) &
            (e20s    >  e50s) &
            (high_10d.notna()) &
            (close   >= high_10d * 0.99)   # price at/above 10D high
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
# PART 4 — ML (separate model, separate features, 3-day target)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode1(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 1 features: breakout-centric.
    Target: close[+3] > close[today]  (3-day momentum hold)
    Training filter: only rows where vol_ratio > 1.3 (momentum conditions)
    """
    try:
        if len(close) < 35:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        high_10d  = close.rolling(10, min_periods=5).max().shift(1)
        rsi_s     = rsi_vec(close)
        ret1d     = close.pct_change(1) * 100
        ret3d     = close.pct_change(3) * 100
        near_high = ((close / high_10d.replace(0, np.nan)) - 1.0) * 100
        breakout  = vol_r * (1 + near_high.clip(lower=-10, upper=0) / 100)

        # target: price higher 3 days later (momentum holding period)
        target = (close.shift(-3) > close).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_s,
            "vol_ratio": vol_r,
            "near_high": near_high,
            "ret_1d":    ret1d,
            "ret_3d":    ret3d,
            "breakout":  breakout,
            "ema_trend": (e20s > e50s).astype(int),
            "target":    target,
        }).dropna()

        # training filter: only momentum conditions
        df = df[df["vol_ratio"] > 1.3]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode1() -> bool:
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
        rows = _build_features_mode1(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 80:
        return False

    FEAT = ["rsi", "vol_ratio", "near_high", "ret_1d", "ret_3d", "breakout", "ema_trend"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        mdl = LogisticRegression(max_iter=500, C=0.4, class_weight="balanced",
                                  solver="lbfgs", random_state=1)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode1 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode1 ML] train failed: {e}")
        return False


def predict_ml_mode1(row: dict) -> float:
    """Mode 1 ML: uses breakout-centric features."""
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode1()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",               60))
        vol_r = safe(row.get("Vol / Avg",          1.7))
        d20h  = safe(row.get("Δ vs 20D High (%)", -2))
        r5d   = safe(row.get("5D Return (%)",       0))
        r1d   = r5d / 5.0   # approx 1D return
        r3d   = r5d * 0.6   # approx 3D return
        # near_high was trained on 10D high proximity. Mode 1 scan filter already
        # guarantees price is within 1% of 10D high. Use d20h as a proxy but scale
        # conservatively — 10D high is always ≥ 20D high, so d20h is a lower bound.
        # For passing stocks (at/near breakout), d20h is close to 0 anyway.
        near_h = max(d20h, -5.0)   # clip pessimistic proxy to avoid over-penalising
        bt     = vol_r * (1 + max(near_h, -10) / 100)
        feat  = np.array([[ri, vol_r, near_h, r1d, r3d, bt, 1.0]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # boost for high vol + near-breakout (Mode 1 primary signals)
        adj   = 0.0
        if vol_r > 2.0:            adj += 0.06
        if -1.0 <= d20h <= 0.0:    adj += 0.05
        if ri > 72:                adj -= 0.08
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0