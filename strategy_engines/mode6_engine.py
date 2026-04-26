"""
strategy_engines/mode6_engine.py
──────────────────────────────────
MODE 6 — SWING (EMA Slope + Controlled Momentum)

Philosophy: EMA20 must be rising (slope positive). RSI must be in a
controlled range 53-59. Vol spike is OK but SHARP spikes penalised
(short-squeeze risk). Avoid late-stage moves.
ML target: close[+3] > close[today] (3-day swing hold).
Training filter: EMA slope > 0 AND 50 < RSI < 63.
Backtest: EMA slope + controlled RSI + moderate vol.
Training universe: trend-following quality stocks.
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
    "PIDILITIND.NS","BERGEPAINT.NS","HAVELLS.NS","VOLTAS.NS","POLYCAB.NS",
    "TITAN.NS","ASIANPAINT.NS","BAJFINANCE.NS","KOTAKBANK.NS","HDFCBANK.NS",
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","LT.NS",
    "SUNPHARMA.NS","DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","APOLLOHOSP.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode6(row: dict) -> tuple[float, dict]:
    """
    Swing scoring: EMA slope (rising) + controlled RSI 53-59 are primary.
    Sharp vol spikes are penalised (short-squeeze avoidance).
    5D return 1-4% preferred — avoid late-stage exhaustion.
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

    # EMA slope — cannot compute directly from row, use EMA20 > EMA50 as proxy
    # + EMA20 > price * 0.995 signals rising EMA context
    if e20 > e50 > 0:     pts["EMA Slope (Rising)"]    = 22   # strongest for swing
    if price > e20 > 0:   pts["Price > EMA20"]          = 15

    # RSI 52-62 sweet spot — controlled swing zone
    if 52 <= ri <= 62:    pts["Swing RSI Sweet Spot"]   = 18   # strongest for swing
    elif 62 < ri <= 65:   pts["RSI Acceptable"]         = 10

    # Volume — moderate only (controlled vol for swing)
    if 1.1 <= vol_r <= 2.0:  pts["Controlled Vol"]      = 16
    elif vol_r > 2.0:        pts["Vol Rising"]           = 10  # lower than other modes

    # Near high (within 3%)
    if -2.0 <= d20h <= 0.0:  pts["Near Breakout"]       = 14
    elif -4.0 <= d20h < -2.0: pts["Approaching"]        = 8

    # Swing return zone (modest move preferred)
    if 1.0 <= r5d <= 4.0:   pts["5D Controlled Move"]   = 16
    if r20d > 1.0:           pts["20D Uptrend"]          = 8

    # PENALTIES — strict RSI ceiling, sharp vol penalised
    if ri > 65:              pts["RSI Above Swing Zone"] = -18
    if ri < 49:              pts["RSI Too Low"]          = -8
    if vol_r > 3.0:          pts["Sharp Vol Spike"]      = -12  # unique to swing
    if r5d > 6.0:            pts["Overrun (Swing)"]      = -16
    if de20 > 5.0:           pts["Overextended EMA"]     = -12
    if e20 < e50:            pts["EMA Bearish"]          = -18

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


def check_bull_trap_mode6(row: dict) -> str:
    """Swing trap: RSI above zone + EMA slope weakening + sharp spike."""
    ri    = safe(row.get("RSI",            50))
    vol_r = safe(row.get("Vol / Avg",       1))
    r5d   = safe(row.get("5D Return (%)",   0))
    e20   = safe(row.get("EMA 20",          1))
    e50   = safe(row.get("EMA 50",          0))
    hits  = sum([ri > 65, vol_r > 3.0, r5d > 8.0, e20 < e50])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (swing: EMA slope + controlled RSI + moderate vol)
# ─────────────────────────────────────────────────────────────────────
def backtest_mode6(row: dict, ticker: str) -> float:
    """
    Mode 6: EMA20 must be rising (slope > 0). RSI tightly controlled 50-63.
    Vol must be moderate — NOT extreme. Avoids sharp spike entries.
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
        ema_slope = e20s > e20s.shift(1)   # EMA20 rising

        target_rsi  = safe(row.get("RSI",      56))
        target_volr = safe(row.get("Vol / Avg", 1.4))

        # Mode 6 entry: rising EMA slope + controlled RSI + moderate vol
        mask = (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 3) &
            (rsi_s    <= target_rsi  + 3) &
            (vol_ratio >= target_volr * 0.85) &
            (vol_ratio <= 2.2) &                    # cap at 2.2× (no sharp spikes)
            (e20s     >  e50s) &
            ema_slope &                              # EMA20 slope MUST be positive
            (rsi_s    >= 50) &
            (rsi_s    <= 63)                         # RSI range enforcement
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
# PART 4 — ML (swing features, 3-day target, slope filter)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode6(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 6 features: EMA slope % + controlled RSI binary.
    Target: close[+3] > close[today] (3-day swing hold).
    Training filter: EMA slope positive AND 48 < RSI < 65.
    """
    try:
        if len(close) < 35:
            return None
        e20s       = ema(close, 20)
        e50s       = ema(close, 50)
        avg_vol    = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r      = volume / avg_vol.replace(0, np.nan)
        ema_dist   = (close / e20s.replace(0, np.nan) - 1.0) * 100
        # EMA slope as percentage change
        ema_slope  = (e20s / e20s.shift(1).replace(0, np.nan) - 1.0) * 100
        rsi_s      = rsi_vec(close)
        ret5d      = close.pct_change(5)  * 100
        rsi_ctrl   = ((rsi_s >= 50) & (rsi_s <= 62)).astype(int)   # controlled zone binary
        vol_ctrl   = (vol_r <= 2.2).astype(int)                    # no sharp spikes binary

        # 3-day swing target
        target     = (close.shift(-3) > close).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_s,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ema_slope": ema_slope,
            "ret_5d":    ret5d,
            "rsi_ctrl":  rsi_ctrl,
            "vol_ctrl":  vol_ctrl,
            "target":    target,
        }).dropna()

        # training filter: swing conditions only
        df = df[(df["ema_slope"] > 0) & (df["rsi"] >= 48) & (df["rsi"] <= 65)]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode6() -> bool:
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
        rows = _build_features_mode6(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 60:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ema_slope", "ret_5d", "rsi_ctrl", "vol_ctrl"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=6, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        mdl = LogisticRegression(max_iter=500, C=0.45, class_weight="balanced",
                                  solver="lbfgs", random_state=6)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode6 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode6 ML] train failed: {e}")
        return False


def predict_ml_mode6(row: dict) -> float:
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode6()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",               56))
        vol_r = safe(row.get("Vol / Avg",          1.3))
        de20  = safe(row.get("Δ vs EMA20 (%)",     0))
        r5d   = safe(row.get("5D Return (%)",       0))
        rctrl = 1.0 if (50 <= ri <= 62) else 0.0
        vctrl = 1.0 if (vol_r <= 2.2) else 0.0
        # ema_slope proxy: price above EMA20 (de20 > 0) strongly implies rising slope.
        # Training feature is ema_slope in %, typically ±0.1–0.3% for trending stocks.
        # Map de20 (% distance) → slope proxy: de20=+5% → eslope≈+0.125 (rising),
        # de20=-5% → eslope≈-0.125 (likely falling). Clamp to realistic slope range.
        eslope = float(np.clip(de20 * 0.025, -0.3, 0.5))
        feat  = np.array([[ri, vol_r, de20, eslope, r5d, rctrl, vctrl]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # swing adjustments
        adj = 0.0
        if 53 <= ri <= 59:   adj += 0.04   # sweet spot
        if vol_r > 2.3:      adj -= 0.05   # sharp spike penalty
        if ri > 63:          adj -= 0.07   # above swing zone
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0