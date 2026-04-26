"""
strategy_engines/mode4_engine.py
──────────────────────────────────
MODE 4 — INSTITUTIONAL STRENGTH

Philosophy: relative strength vs market is the DOMINANT signal.
EMA trend quality and 20D return are weighted highest.
Stocks underperforming the market are heavily penalised.
ML target: close[+5] > close[today] (institutional 5-day hold).
Training filter: only rows where EMA20 > EMA50 AND 20D return > 0.
Training universe: Nifty 50 components (institutional grade).
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
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","BAJFINANCE.NS",
    "HCLTECH.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","ONGC.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode4(row: dict) -> tuple[float, dict]:
    """
    Institutional scoring: EMA trend + relative strength (20D return)
    are the two dominant factors. Volume is secondary. RSI matters less.
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

    # EMA structure — STRONGEST factor for institutional
    if e20 > e50 > 0:                pts["EMA Trend Aligned"]     = 28
    if price > e20 > 0:              pts["Price > EMA20"]          = 18

    # 20D return = relative strength proxy — SECOND strongest
    if r20d > 8.0:                   pts["Strong Rel Strength"]   = 25
    elif r20d > 4.0:                 pts["Good Rel Strength"]     = 18
    elif r20d > 1.0:                 pts["Positive 20D"]          = 10

    # Volume — secondary for institutional (they accumulate quietly)
    if   vol_r > 1.5:                pts["Vol Confirmation"]      = 14
    elif vol_r > 1.2:                pts["Vol Developing"]        = 8

    # Near 20D high (within 2%)
    if -2.0 <= d20h <= 0.0:          pts["Near 20D High"]         = 14

    # RSI — moderate zone preferred
    if 55 <= ri <= 68:               pts["RSI Institutional Zone"] = 10

    # PENALTIES — relative weakness punished hardest
    if r20d < 0:                     pts["Relative Weakness"]      = -22
    if e20 < e50:                    pts["EMA Break Down"]         = -30  # critical
    if ri > 70:                      pts["RSI Overbought"]         = -15
    if vol_r < 1.5:                  pts["Insufficient Vol"]       = -10
    if de20 > 6.0:                   pts["Overextended"]           = -12

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


def check_bull_trap_mode4(row: dict) -> str:
    """Institutional trap: EMA breakdown + relative weakness + volume drying."""
    e20   = safe(row.get("EMA 20", 1))
    e50   = safe(row.get("EMA 50", 0))
    vol_r = safe(row.get("Vol / Avg", 1))
    r20d  = safe(row.get("20D Return (%)", 0))
    hits  = sum([e20 < e50, vol_r < 1.0, r20d < -2.0])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (relative strength entry simulation)
# ─────────────────────────────────────────────────────────────────────
def backtest_mode4(row: dict, ticker: str) -> float:
    """
    Mode 4: matches on EMA trend + positive 20D return momentum.
    No near-high proximity requirement — institutional holding.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    with _BT_LOCK:
        if ticker_ns in _BT_CACHE:
            return _BT_CACHE[ticker_ns]

    result = 50.0
    try:
        # BUG FIX: Use get_df_for_ticker (respects TT patch) not download_history
        df = get_df_for_ticker(ticker_ns)
        if df is None or len(df) < 45:
            raise ValueError("insufficient data")

        close     = df["Close"].copy()
        volume    = df["Volume"].copy()
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        rsi_s     = rsi_vec(close)
        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        ret_20d   = close.pct_change(20) * 100
        high_20d  = close.rolling(20, min_periods=10).max().shift(1)

        target_rsi  = safe(row.get("RSI",      60))
        target_volr = safe(row.get("Vol / Avg", 1.5))

        # Mode 4 entry: EMA trend + positive relative strength + vol
        mask = (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 3) &
            (rsi_s    <= target_rsi  + 3) &
            (vol_ratio >= target_volr * 0.85) &
            (e20s     >  e50s) &                    # EMA trend required
            (ret_20d  >  0) &                        # positive 20D return (relative strength)
            (high_20d.notna()) &
            (close    >= high_20d * 0.98)            # within 2% of 20D high
        )
        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]
        if len(idx) < 12:
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
# PART 4 — ML (institutional features, 5-day target)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode4(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 4 features: EMA quality + relative strength proxy (20D return).
    Target: close[+5] > close[today] — institutional 5-day hold.
    Training filter: EMA20 > EMA50 AND 20D return > 0.
    """
    try:
        if len(close) < 35:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        ema_dist  = (close / e20s.replace(0, np.nan) - 1.0) * 100
        rsi_s     = rsi_vec(close)
        ret20d    = close.pct_change(20) * 100
        high_20d  = close.rolling(20, min_periods=10).max().shift(1)
        near_20h  = (close / high_20d.replace(0, np.nan) - 1.0) * 100

        # 5-day target (institutional holding period)
        target    = (close.shift(-5) > close).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_s,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ret_20d":   ret20d,
            "near_20h":  near_20h,
            "ema_trend": (e20s > e50s).astype(int),
            "target":    target,
        }).dropna()

        # filter: only institutional-condition rows
        df = df[(df["ema_trend"] == 1) & (df["ret_20d"] > 0)]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode4() -> bool:
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
        rows = _build_features_mode4(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 60:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ret_20d", "near_20h", "ema_trend"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=4, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        mdl = LogisticRegression(max_iter=500, C=0.4, class_weight="balanced",
                                  solver="lbfgs", random_state=4)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode4 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode4 ML] train failed: {e}")
        return False


def predict_ml_mode4(row: dict) -> float:
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode4()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",               60))
        vol_r = safe(row.get("Vol / Avg",          1))
        de20  = safe(row.get("Δ vs EMA20 (%)",     0))
        r20d  = safe(row.get("20D Return (%)",      0))
        d20h  = safe(row.get("Δ vs 20D High (%)", -2))
        feat  = np.array([[ri, vol_r, de20, r20d, d20h, 1.0]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # institutional adjustments
        adj = 0.0
        if r20d > 5.0 and ri > 57:   adj += 0.05   # strong relative strength
        if r20d < 0:                 adj -= 0.06   # relative weakness penalty
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0