"""
strategy_engines/mode3_engine.py
──────────────────────────────────
MODE 3 — RELAXED (Early Accumulation)

Philosophy: detect early-stage setups before they're obvious.
Low penalty system — don't punish developing trends.
RSI 50-58 is the TARGET zone (accumulation before breakout).
ML target: next day green. Training filter: RSI 45-65 rows only.
Backtest: loose matching — wider RSI/vol tolerance.
Training universe: mid-cap developing trends.
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
    "AUBANK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS","BANDHANBNK.NS","RBLBANK.NS",
    "LTTS.NS","PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","TATAELXSI.NS",
    "AUROPHARMA.NS","LUPIN.NS","TORNTPHARM.NS","ALKEM.NS","GRANULES.NS",
    "ASTRAL.NS","SUPREMEIND.NS","AAVAS.NS","CANFINHOME.NS","LICHSGFIN.NS",
]


# ─────────────────────────────────────────────────────────────────────
# PART 1 — SMART SCORE
# ─────────────────────────────────────────────────────────────────────
def compute_score_mode3(row: dict) -> tuple[float, dict]:
    """
    Relaxed scoring: early accumulation signals rewarded.
    RSI 50-58 is the sweet spot (not yet overbought).
    Penalties are HALVED vs other modes.
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

    # RSI 50-58 = early accumulation zone → highest bonus in Mode 3
    if 50 <= ri <= 58:    pts["Early Accumulation RSI"]  = 22
    elif 58 < ri <= 65:   pts["Mid RSI Zone"]            = 14
    elif 65 < ri <= 70:   pts["Upper RSI (caution)"]     = 7

    # Volume — moderate; even small pickup rewarded
    if   vol_r > 2.0:     pts["Vol >2× Strong"]         = 18
    elif vol_r > 1.5:     pts["Vol >1.5×"]              = 14
    elif vol_r > 1.2:     pts["Vol Developing"]         = 8

    # Within 5% of 20D high — relaxed proximity
    if  -2.0 <= d20h <= 0.0:  pts["Near 20D High"]      = 12
    elif -5.0 <= d20h < -2.0: pts["Approaching High"]   = 8

    # EMA structure
    if price > e20 > 0:   pts["Price > EMA20"]          = 12
    if e20 > e50 > 0:     pts["EMA20 > EMA50"]          = 10

    # Modest returns preferred (early stage)
    if 0.5 <= r5d <= 4.0: pts["5D Gentle Return"]       = 12
    if r20d > 0:          pts["20D Uptrend"]            = 8

    # PENALTIES — HALVED vs other modes (relaxed system)
    if ri > 72:           pts["RSI Too High"]            = -8    # halved
    if de20 > 8.0:        pts["Overextended (mild)"]    = -7
    if r5d > 9.0:         pts["5D Overrun"]             = -7
    if vol_r < 1.1:       pts["Low Vol (mild)"]         = -6

    score = float(np.clip(sum(pts.values()), 0, 100))
    return score, pts


def check_bull_trap_mode3(row: dict) -> str:
    """Relaxed trap: only trigger on extreme overbought (threshold raised)."""
    ri    = safe(row.get("RSI",            50))
    vol_r = safe(row.get("Vol / Avg",       1))
    de20  = safe(row.get("Δ vs EMA20 (%)", 0))
    # higher thresholds — mode 3 is already relaxed
    hits  = sum([ri > 78, vol_r < 0.9, de20 > 10.0])
    return "⚠️ Bull Trap" if hits >= 2 else ""


# ─────────────────────────────────────────────────────────────────────
# PART 3 — BACKTEST (loose matching — wide tolerances)
# ─────────────────────────────────────────────────────────────────────
def backtest_mode3(row: dict, ticker: str) -> float:
    """
    Mode 3: widest matching tolerances (±6 RSI, ±35% vol).
    Does NOT require near-high condition — early setups are OK far from high.
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

        target_rsi  = safe(row.get("RSI",      55))
        target_volr = safe(row.get("Vol / Avg", 1.3))

        # Mode 3: loose — wide RSI/vol tolerance, EMA trend preferred not required
        # No upper vol cap: early accumulation can have vol spikes; include them all.
        mask = (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 6) &        # ±6 RSI
            (rsi_s    <= target_rsi  + 6) &
            (vol_ratio >= target_volr * 0.65) &    # wide lower tolerance
            (close    >  e20s)                     # price above EMA20 (only condition)
        )
        # Note: NOT requiring e20s > e50s — early setups may not have EMA cross yet
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
# PART 4 — ML (early accumulation features, RSI 45-65 filter)
# ─────────────────────────────────────────────────────────────────────
def _build_features_mode3(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 3 features: early-stage signals.
    Training filter: RSI 45-65 rows only (accumulation phase).
    Target: next day green.
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
        high_20d  = close.rolling(20, min_periods=10).max().shift(1)
        dist_high = (close / high_20d.replace(0, np.nan) - 1.0) * 100
        ret5d     = close.pct_change(5) * 100
        # early_signal: 1 if RSI in accumulation zone + vol building
        early_sig = ((rsi_s >= 48) & (rsi_s <= 60) & (vol_r > 1.1)).astype(int)
        target    = (close.shift(-1) > close).astype(int)

        df = pd.DataFrame({
            "rsi":        rsi_s,
            "vol_ratio":  vol_r,
            "ema_dist":   ema_dist,
            "dist_high":  dist_high,
            "ret_5d":     ret5d,
            "early_sig":  early_sig,
            "ema_trend":  (e20s > e50s).astype(int),
            "target":     target,
        }).dropna()

        # training filter: only accumulation-phase rows
        df = df[(df["rsi"] >= 45) & (df["rsi"] <= 65)]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode3() -> bool:
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
        rows = _build_features_mode3(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 60:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "dist_high", "ret_5d", "early_sig", "ema_trend"]
    X, y = data[FEAT].values, data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=3, stratify=y)
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        # higher C = less regularisation (relaxed mode)
        mdl = LogisticRegression(max_iter=500, C=0.8, class_weight="balanced",
                                  solver="lbfgs", random_state=3)
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode3 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as e:
        print(f"[Mode3 ML] train failed: {e}")
        return False


def predict_ml_mode3(row: dict) -> float:
    if not SKLEARN_OK:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode3()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None:
            return 50.0
    try:
        ri    = safe(row.get("RSI",               55))
        vol_r = safe(row.get("Vol / Avg",          1))
        de20  = safe(row.get("Δ vs EMA20 (%)",     0))
        d20h  = safe(row.get("Δ vs 20D High (%)", -3))
        r5d   = safe(row.get("5D Return (%)",       0))
        esig  = 1.0 if (48 <= ri <= 60 and vol_r > 1.1) else 0.0
        feat  = np.array([[ri, vol_r, de20, d20h, r5d, esig, 1.0]])
        prob  = float(mdl.predict_proba(sc.transform(feat))[0][1])
        # small confidence haircut for relaxed mode
        adj   = -0.02
        if 50 <= ri <= 58:   adj += 0.04  # early zone bonus
        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100, 1)
    except Exception:
        return 50.0