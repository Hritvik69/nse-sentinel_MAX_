"""
strategy_engines/_df_extensions.py
────────────────────────────────────
DataFrame-accepting backtest & training wrappers for all 6 modes.

These replace the ticker-based download_history() calls inside each engine.
The scan loop passes pre-loaded DataFrames from ALL_DATA directly.

Design rules:
  • Zero API calls — all functions accept `df: pd.DataFrame`
  • No crashes — every function catches exceptions and returns 50
  • Formulas/logic identical to originals — only data source changes
  • Training functions accept `all_data: dict[str, df]` and iterate it
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd

from strategy_engines._engine_utils import (
    safe, ema, rsi_vec, SKLEARN_OK, ALL_DATA,
)
if SKLEARN_OK:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _run_backtest(df: pd.DataFrame, mask_fn, min_samples: int = 10) -> float:
    """
    Common backtest runner.
    mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio)
    → pd.Series[bool]

    BUG FIX: Docstring previously listed 8 parameters including a spurious
    `row` argument. The actual call (line below) passes only 7 — `row` was
    never passed and no inline mask_fn lambda declared it. Corrected to match
    the real signature.
    """
    try:
        if df is None or len(df) < 40:
            return 50.0
        close     = df["Close"].copy()
        volume    = df["Volume"].copy()
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        rsi_s     = rsi_vec(close)
        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)

        mask = mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio)
        idx  = np.where(mask.values)[0]
        idx  = idx[idx < len(close) - 1]
        if len(idx) < min_samples:
            return 50.0
        cv    = close.values
        green = int(sum(cv[i + 1] > cv[i] for i in idx))
        return round((green / len(idx)) * 100, 1)
    except Exception:
        return 50.0


def _train_from_data(
    build_fn,
    feat_cols: list[str],
    all_data: dict[str, pd.DataFrame | None],
    train_tickers: list[str],
    model_lock: threading.Lock,
    model_holder: list,      # [model, scaler]  — mutable container
    min_samples: int = 60,
    C: float = 0.5,
    random_state: int = 42,
    min_per_ticker: int = 10,
) -> bool:
    """
    Generic train-from-preloaded-data function.
    build_fn(close, volume) → pd.DataFrame | None  (feature builder)
    model_holder = [model_ref, scaler_ref]  — updated in-place
    """
    if not SKLEARN_OK:
        return False
    with model_lock:
        if model_holder[0] is not None:
            return True

    all_rows: list[pd.DataFrame] = []
    for t in train_tickers:
        tk = t if t.endswith(".NS") else f"{t}.NS"
        df_h = all_data.get(tk)
        if df_h is None or len(df_h) < 35:
            continue
        try:
            rows = build_fn(df_h["Close"], df_h["Volume"])
            if rows is not None and len(rows) >= min_per_ticker:
                all_rows.append(rows)
        except Exception:
            continue

    if not all_rows:
        return False
    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < min_samples:
        return False

    X = data[feat_cols].values
    y = data["target"].values
    try:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=random_state, stratify=y
            )
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        sc  = StandardScaler()
        X_s = sc.fit_transform(X_tr)
        mdl = LogisticRegression(
            max_iter=500, C=C, class_weight="balanced",
            solver="lbfgs", random_state=random_state,
        )
        mdl.fit(X_s, y_tr)
        acc = mdl.score(sc.transform(X_te), y_te)
        print(f"[ML df-train] C={C} rs={random_state} samples={len(data)} acc={acc:.3f}")
        with model_lock:
            model_holder[0] = mdl
            model_holder[1] = sc
        return True
    except Exception as e:
        print(f"[ML df-train] failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# MODE 1 — MOMENTUM
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode1_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 1: high-volume price clearing 10D high. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    target_rsi  = safe(row.get("RSI",      60))
    target_volr = safe(row.get("Vol / Avg", 1.7))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        high_10d = close.rolling(10, min_periods=5).max().shift(1)
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 3) &
            (rsi_s    <= target_rsi  + 3) &
            (vol_ratio >= max(target_volr * 0.85, 1.4)) &
            (e20s     >  e50s) &
            high_10d.notna() &
            (close    >= high_10d * 0.99)
        )
    return _run_backtest(df, mask_fn, min_samples=15)


# ═══════════════════════════════════════════════════════════════════════
# MODE 2 — BALANCED
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode2_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 2: balanced EMA+vol+RSI. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    target_rsi  = safe(row.get("RSI",      60))
    target_volr = safe(row.get("Vol / Avg", 1.5))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        high_15d = close.rolling(15, min_periods=7).max().shift(1)
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 4) &
            (rsi_s    <= target_rsi  + 4) &
            (vol_ratio >= target_volr * 0.80) &
            (vol_ratio <= target_volr * 1.20) &
            (e20s     >  e50s) &
            high_15d.notna() &
            (close    >= high_15d * 0.97)
        )
    return _run_backtest(df, mask_fn, min_samples=15)


# ═══════════════════════════════════════════════════════════════════════
# MODE 3 — RELAXED
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode3_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 3: wide tolerances. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    target_rsi  = safe(row.get("RSI",      55))
    target_volr = safe(row.get("Vol / Avg", 1.3))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 6) &
            (rsi_s    <= target_rsi  + 6) &
            (vol_ratio >= target_volr * 0.75) &
            (vol_ratio <= target_volr * 1.35) &
            (close    >  e20s)
        )
    return _run_backtest(df, mask_fn, min_samples=12)


# ═══════════════════════════════════════════════════════════════════════
# MODE 4 — INSTITUTIONAL
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode4_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 4: EMA trend + positive 20D return. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    if len(df) < 45:
        return 50.0
    target_rsi  = safe(row.get("RSI",      60))
    target_volr = safe(row.get("Vol / Avg", 1.5))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        ret_20d  = close.pct_change(20) * 100
        high_20d = close.rolling(20, min_periods=10).max().shift(1)
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 3) &
            (rsi_s    <= target_rsi  + 3) &
            (vol_ratio >= target_volr * 0.85) &
            (e20s     >  e50s) &
            (ret_20d  >  0) &
            high_20d.notna() &
            (close    >= high_20d * 0.98)
        )
    return _run_backtest(df, mask_fn, min_samples=12)


# ═══════════════════════════════════════════════════════════════════════
# MODE 5 — INTRADAY
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode5_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 5: 5D high + vol spike, tight RSI. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    target_rsi  = safe(row.get("RSI",      55))
    target_volr = safe(row.get("Vol / Avg", 1.5))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        high_5d = close.rolling(5, min_periods=3).max().shift(1)
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 2) &
            (rsi_s    <= target_rsi  + 2) &
            (vol_ratio >= max(target_volr * 0.88, 1.3)) &
            (e20s     >  e50s) &
            high_5d.notna() &
            (close    >= high_5d * 0.99) &
            (vol_ratio <= 3.5)
        )
    return _run_backtest(df, mask_fn, min_samples=10)


# ═══════════════════════════════════════════════════════════════════════
# MODE 6 — SWING
# ═══════════════════════════════════════════════════════════════════════

def backtest_mode6_df(row: dict, df: pd.DataFrame | None) -> float:
    """Mode 6: rising EMA slope + controlled RSI. Accepts pre-loaded df."""
    if df is None:
        return 50.0
    target_rsi  = safe(row.get("RSI",      56))
    target_volr = safe(row.get("Vol / Avg", 1.4))

    def mask_fn(close, volume, e20s, e50s, rsi_s, avg_vol, vol_ratio):
        ema_slope = e20s > e20s.shift(1)
        return (
            rsi_s.notna() &
            (rsi_s    >= target_rsi  - 3) &
            (rsi_s    <= target_rsi  + 3) &
            (vol_ratio >= target_volr * 0.85) &
            (vol_ratio <= 2.2) &
            (e20s     >  e50s) &
            ema_slope &
            (rsi_s    >= 50) &
            (rsi_s    <= 63)
        )
    return _run_backtest(df, mask_fn, min_samples=12)


# ═══════════════════════════════════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════════════════════════════════

_BACKTEST_FNS = {
    1: backtest_mode1_df,
    2: backtest_mode2_df,
    3: backtest_mode3_df,
    4: backtest_mode4_df,
    5: backtest_mode5_df,
    6: backtest_mode6_df,
}


def backtest_with_preloaded(mode: int, row: dict, ticker: str) -> float:
    """
    Public entry point called by enhance_results().
    Looks up ALL_DATA[ticker] and passes to the appropriate df-variant.
    Falls back to 50 if data not preloaded.
    NEVER calls download_history() — zero-API by design.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    df = ALL_DATA.get(ticker_ns)
    fn = _BACKTEST_FNS.get(mode)
    if fn is None:
        return 50.0
    try:
        return fn(row, df)
    except Exception:
        return 50.0