"""
strategy_engines/mode5_engine.py
──────────────────────────────────
MODE 5 — INTRADAY (Short-term Momentum)

Philosophy: explosive volume spike + price above 10D high = primary signal.
IGNORE long-term returns (20D return excluded from scoring).
Strict RSI ceiling — RSI > 65 is a penalty (overheated intraday).
ML target: next day green (shortest holding period).
Training filter: high-vol rows only (vol_ratio > 1.2, RSI < 63).
Backtest: price at 5D high + vol spike matching.
Training universe: high-volume liquid NSE stocks.
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
    "SBIN.NS","ICICIBANK.NS","AXISBANK.NS","TATAMOTORS.NS","HINDALCO.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","COALINDIA.NS","ONGC.NS","BPCL.NS",
    "IOC.NS","HPCL.NS","VEDL.NS","SAIL.NS","NMDC.NS",
    "RELIANCE.NS","BHARTIARTL.NS","HDFCBANK.NS","INFY.NS","TCS.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode5(row: dict) -> tuple[float, dict]:
    """
    Intraday scoring: volume spike + near-10D-high are DOMINANT.
    Long-term signals (20D return) completely excluded.
    RSI > 65 is strictly penalised (overbought intraday).
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
    # NOTE: 20D return deliberately excluded from intraday scoring

    # RSI — intraday sweet spot (52-60 only)
    if 52 <= ri <= 58:    pts["RSI Intraday Zone"]   = 18
    elif 58 < ri <= 62:   pts["RSI Upper Intraday"]  = 10

    # Volume — HIGHEST weight (intraday needs confirmation)
    if   vol_r > 2.5:     pts["Vol >2.5× Explosive"] = 38
    elif vol_r > 2.0:     pts["Vol >2×"]             = 30
    elif vol_r > 1.5:     pts["Vol >1.5×"]           = 22
    elif vol_r > 1.2:     pts["Vol Minimal"]         = 10

    # Near 10D high — critical for intraday breakout
    if  -1.0 <= d20h <= 0.0:   pts["At 10D Breakout"]   = 22
    elif -2.0 <= d20h < -1.0:  pts["Near 10D High"]     = 14

    # EMA structure (short-term focus)
    if price > e20 > 0:   pts["Price > EMA20"]         = 10
    if e20 > e50 > 0:     pts["EMA Stack"]             = 8

    # Short-term return only (5D)
    if 0.5 <= r5d <= 4.0: pts["5D Momentum"]           = 10

    # PENALTIES — strict for intraday
    if ri > 63:           pts["RSI Too Hot (Intraday)"] = -22
    if ri > 65:           pts["RSI Overheated"]         = -8   # additional
    if vol_r < 1.3:       pts["Low Vol (Critical)"]    = -28
    if de20 > 4.0:        pts["Overextended (Short)"]  = -16
    if r5d > 7.0:         pts["Exhaustion Signal"]     = -14

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


def check_bull_trap_mode5(row: dict) -> str:
    """Intraday trap: RSI > 63 + low vol = dangerous (strict thresholds)."""
    ri    = safe(row.get("RSI",            50))
    vol_r = safe(row.get("Vol / Avg",       1))
    de20  = safe(row.get("Δ vs EMA20 (%)", 0))
    # very strict for intraday
    hits  = sum([ri > 63, vol_r < 1.0, de20 > 5.0])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (intraday: 5D high + vol spike entry)
# ─────────────────────────────────────────────────────────────────────
def backtest_mode5(row: dict, ticker: str) -> float:
    """
    Mode 5: short 5D rolling high + strong vol spike matching.
    RSI tolerance: ±2 (tighter than other modes — intraday precision).
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
        high_5d   = close.rolling(5, min_periods=3).max().shift(1)   # 5D high (short window)

        target_rsi  = safe(row.get("RSI",      55))
        target_volr = safe(row.get("Vol / Avg", 1.5))

        # Mode 5 entry: at/above 5D high + strong vol + tight RSI range
        mask = (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 2) &         # ±2 RSI (tightest tolerance)
            (rsi_s    <= target_rsi  + 2) &
            (vol_ratio >= max(target_volr * 0.88, 1.3)) &
            (e20s     >  e50s) &
            (high_5d.notna()) &
            (close    >= high_5d * 0.99) &           # at/above 5D high
            (vol_ratio <= 3.5)                       # exclude extreme spikes
        )
        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]
        if len(idx) < 10:
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
# PART 4 — ML (intraday features, next-day target, high-vol filter)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode5(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 5 features: intraday-only signals.
    No 20D return — excluded entirely.
    Training filter: vol_ratio > 1.2 AND rsi < 63.
    Target: next day green (shortest hold).
    """
    try:
        if len(close) < 30:
            return None
        e20s     = ema(close, 20)
        e50s     = ema(close, 50)
        avg_vol  = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r    = volume / avg_vol.replace(0, np.nan)
        rsi_s    = rsi_vec(close)
        high_5d  = close.rolling(5, min_periods=3).max().shift(1)
        near_5h  = (close / high_5d.replace(0, np.nan) - 1.0) * 100
        ret1d    = close.pct_change(1)  * 100
        ret5d    = close.pct_change(5)  * 100
        vol_spk  = (vol_r > 1.5).astype(int)     # binary vol spike signal
        # NOTE: no ret_20d — intraday ignores long-term
        target   = (close.shift(-1) > close).astype(int)

        df = pd.DataFrame({
            "rsi":      rsi_s,
            "vol_ratio": vol_r,
            "near_5h":  near_5h,
            "ret_1d":   ret1d,
            "ret_5d":   ret5d,
            "vol_spk":  vol_spk,
            "ema_trend": (e20s > e50s).astype(int),
            "target":   target,
        }).dropna()

        # training filter: intraday conditions only
        df = df[(df["vol_ratio"] > 1.2) & (df["rsi"] < 63)]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode5() -> bool:
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
        rows = _build_features_mode5(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 60:
        return False

    FEAT = ["rsi", "vol_ratio", "near_5h", "ret_1d", "ret_5d", "vol_spk", "ema_trend"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=5, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        # tighter regularisation for intraday (noisy data)
        mdl = LogisticRegression(max_iter=500, C=0.3, class_weight="balanced",
                                  solver="lbfgs", random_state=5)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode5 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode5 ML] train failed: {e}")
        return False


def predict_ml_mode5(row: dict) -> float:
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode5()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",               55))
        vol_r = safe(row.get("Vol / Avg",          1))
        d20h  = safe(row.get("Δ vs 20D High (%)", -2))
        r5d   = safe(row.get("5D Return (%)",       0))
        r1d   = r5d / 5.0
        # near_5h proxy: Mode 5 scan filter guarantees price ≥ 99% of 5D high,
        # so the true near_5h is close to 0. d20h is a conservative (more negative)
        # lower bound since 20D high ≥ 5D high. Use 0.5*d20h to avoid over-penalising.
        near_5h_proxy = max(d20h * 0.5, -3.0)
        vspk  = 1.0 if vol_r > 1.5 else 0.0
        feat  = np.array([[ri, vol_r, near_5h_proxy, r1d, r5d, vspk, 1.0]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # intraday adjustments
        adj = 0.0
        if vol_r > 2.0:   adj += 0.07   # strong vol spike = confidence booster
        if ri > 62:       adj -= 0.08   # strict RSI penalty
        if vol_r < 1.3:   adj -= 0.05
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0