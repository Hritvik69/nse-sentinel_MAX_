"""
strategy_engines/mode7_engine.py
--------------------------------
MODE 7 - MOMENTUM (S&R)

Support + resistance momentum scanner.

Philosophy: reward clean market structure first, then confirm with volume,
controlled RSI, EMA alignment, and measured momentum. Penalise weak-volume
breakouts, fake breakouts, blow-off volume, and late overextended moves.

ML target: close[+3] > close[today] (3-day swing continuation).
Backtest: 3-5 day continuation setup with 2.5% target and -2% failure guard.
Training universe: liquid, high-quality NSE momentum stocks only.
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd

from strategy_engines.constants import (
    MODE7_BASE_LOOSE,
    MODE7_BASE_TIGHT_HIGH,
    MODE7_BASE_TIGHT_MEDIUM,
    MODE7_BREAKOUT_ZONE_MAX,
    MODE7_BREAKOUT_ZONE_MIN,
    MODE7_EMA_EXTENSION_HARD,
    MODE7_EMA_EXTENSION_WARN,
    MODE7_5D_SPIKE,
    MODE7_IDEAL_RSI_MAX,
    MODE7_IDEAL_RSI_MIN,
    MODE7_SR_SCORE_HIGH,
    MODE7_SR_SCORE_WEAK,
    MODE7_VOL_CONFIRM_MAX,
    MODE7_VOL_CONFIRM_MIN,
    MODE7_VOL_WEAK,
    debug_log,
)
from strategy_engines._engine_utils import (
    safe, ema, rsi_vec, SKLEARN_OK, get_df_for_ticker,
)

_SKLEARN_READY = False
if SKLEARN_OK:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        _SKLEARN_READY = True
    except Exception as exc:
        debug_log("Mode 7 sklearn import disabled: %s", exc, exc_info=True)
        _SKLEARN_READY = False


_MODEL: "LogisticRegression | None" = None  # noqa: F821
_SCALER: "StandardScaler | None" = None     # noqa: F821
_LOCK = threading.Lock()
_TRAINING: bool = False
_BT_CACHE: dict[str, float] = {}
_BT_CACHE_MAX = 3500
_BT_LOCK = threading.Lock()

_TRAIN_TICKERS = [
    "RELIANCE.NS", "TATAMOTORS.NS", "POLYCAB.NS", "TRENT.NS", "BEL.NS",
    "HAL.NS", "COFORGE.NS", "PERSISTENT.NS", "BSE.NS", "DIXON.NS",
    "KPITTECH.NS", "CGPOWER.NS", "SIEMENS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SUNPHARMA.NS", "BHARTIARTL.NS", "SBIN.NS", "TCS.NS", "INFY.NS",
]


def _get_num(row: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        try:
            if key in row and row.get(key) is not None:
                return safe(row.get(key), default)
        except Exception:
            continue
    return default


def _get_text(row: dict, key: str, default: str = "") -> str:
    try:
        value = row.get(key, default)
        if value is None:
            return default
        return str(value).strip().upper()
    except Exception:
        return default


def _ema_alignment(price: float, ema20: float, ema50: float) -> float:
    if price > ema20 > ema50 > 0:
        return 1.0
    if price > ema20 > 0 and ema50 > 0:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# PART 1 - SMART SCORE
# ---------------------------------------------------------------------------
def compute_score_mode7(row: dict) -> tuple[float, dict]:
    """
    Structure-first momentum score.

    Target distribution:
      - Perfect institutional-quality setup: 80-95
      - Good momentum setup: 68-80
      - Average: 55-67
      - Weak / avoid: below 50
    """
    pts: dict[str, int] = {}

    ri = _get_num(row, ("RSI",), 50.0)
    vol_r = _get_num(row, ("Vol / Avg", "Vol/Avg"), 1.0)
    d20h = _get_num(row, ("Δ vs 20D High (%)", "Near High (%)"), -5.0)
    de20 = _get_num(row, ("Δ vs EMA20 (%)", "EMA Distance (%)"), 0.0)
    r5d = _get_num(row, ("5D Return (%)",), 0.0)
    r20d = _get_num(row, ("20D Return (%)",), 0.0)
    price = _get_num(row, ("Price (₹)", "Price", "Close"), 0.0)
    e20 = _get_num(row, ("EMA 20", "EMA20"), 0.0)
    e50 = _get_num(row, ("EMA 50", "EMA50"), 0.0)
    atr_pct = _get_num(row, ("ATR %", "ATR%"), 0.0)
    support_touches = _get_num(row, ("Support Touches",), 0.0)
    resistance_touches = _get_num(row, ("Resistance Touches",), 0.0)
    resistance_rejections = _get_num(row, ("Resistance Rejections",), 0.0)
    base_tightness = _get_num(row, ("Base Tightness (%)",), 99.0)
    sr_score = _get_num(row, ("S&R Structure Score",), 50.0)

    volume_trend = _get_text(row, "Volume Trend", "NORMAL")
    setup_quality = _get_text(row, "Setup Quality", "MEDIUM")
    trap_risk = _get_text(row, "Trap Risk", "LOW")
    pivot_support_quality = _get_text(row, "Pivot Support Quality", "LOW")
    pivot_resistance_quality = _get_text(row, "Pivot Resistance Quality", "LOW")
    atr_contraction = _get_text(row, "ATR Contraction", "NO")
    breakout_retest = _get_text(row, "Breakout Retest", "NO")
    liquidity_sweep = _get_text(row, "Liquidity Sweep", "NO")
    wick_rejection = _get_text(row, "Wick Rejection", "MEDIUM")

    # 1. Support / resistance quality - largest block.
    if MODE7_BREAKOUT_ZONE_MIN <= d20h <= MODE7_BREAKOUT_ZONE_MAX:
        pts["S&R Ideal Breakout Zone"] = 28
    elif -5.0 <= d20h < -2.0:
        pts["Resistance Compression"] = 18
    elif 1.5 < d20h <= 3.0:
        pts["Early Breakout Extension"] = 10
    elif -8.0 <= d20h < -5.0:
        pts["Still Building Below Resistance"] = 6

    if -1.0 <= de20 <= 3.0:
        pts["Support Respect Near EMA20"] = 10
    elif 3.0 < de20 <= 5.0:
        pts["Controlled Pullback Structure"] = 6

    if d20h > 6.0:
        pts["Massive Breakout Extension"] = -24
    elif d20h > 3.0:
        pts["Too Far Past Resistance"] = -12
    if d20h < -8.0:
        pts["Too Far From Breakout Zone"] = -18

    # 2. Volume confirmation - confirm, do not chase manipulation.
    if MODE7_VOL_CONFIRM_MIN <= vol_r <= MODE7_VOL_CONFIRM_MAX:
        pts["Institutional Volume Confirmation"] = 20
    elif 1.15 <= vol_r < 1.4:
        pts["Building Volume"] = 11
    elif 2.8 < vol_r <= 4.0:
        pts["Strong But Hot Volume"] = 7

    if vol_r < MODE7_VOL_WEAK:
        pts["Weak Volume Breakout"] = -18
    if vol_r > 4.0:
        pts["Possible Blow-off Volume"] = -14

    # 3. RSI quality - controlled momentum only.
    if MODE7_IDEAL_RSI_MIN <= ri <= MODE7_IDEAL_RSI_MAX:
        pts["RSI Controlled Momentum"] = 15
    elif 52.0 <= ri < 55.0:
        pts["RSI Early Momentum"] = 9
    elif 67.0 < ri <= 70.0:
        pts["RSI Upper But Acceptable"] = 6

    if ri > 74.0:
        pts["RSI Exhaustion"] = -22
    elif ri > 70.0:
        pts["RSI Late Momentum"] = -8
    if ri < 48.0:
        pts["RSI Below Momentum Zone"] = -10

    # 4. EMA structure.
    if price > e20 > e50 > 0:
        pts["Price > EMA20 > EMA50"] = 20
    else:
        if price > e20 > 0:
            pts["Price > EMA20"] = 7
        if e20 > e50 > 0:
            pts["EMA20 > EMA50"] = 9

    if price > e20 > 0 and -1.0 <= de20 <= 5.0:
        pts["EMA20 Support Structure"] = 5
    if price < e20 and e20 > 0:
        pts["Price Below EMA20"] = -17
    if e20 < e50 and e50 > 0:
        pts["EMA Stack Bearish"] = -24

    # 5. Momentum quality.
    if 2.0 <= r5d <= 9.0:
        pts["5D Momentum Sweet Spot"] = 11
    elif 0.5 <= r5d < 2.0:
        pts["Early 5D Momentum"] = 5
    if 5.0 <= r20d <= 18.0:
        pts["20D Trend Continuation"] = 9
    elif 1.0 <= r20d < 5.0:
        pts["20D Trend Positive"] = 4

    if r5d < 0.0:
        pts["Negative 5D Momentum"] = -8
    if r20d < 0.0:
        pts["Negative 20D Momentum"] = -10
    if r5d > 14.0:
        pts["Vertical 5D Spike"] = -18
    if r20d > 28.0:
        pts["Late-stage 20D Move"] = -12

    # 6. Overextension filter.
    if 0.0 <= de20 <= 5.0:
        pts["Not Overextended"] = 6
    if de20 > MODE7_EMA_EXTENSION_WARN:
        pts["Overextended From EMA20"] = -20
    elif de20 > 5.0:
        pts["Moderately Extended From EMA20"] = -9
    if de20 < -3.0:
        pts["Below Support Zone"] = -8

    # 7. Optional structure intelligence from later pipeline stages.
    if volume_trend == "STRONG":
        pts["Volume Trend Strong"] = 6
    elif volume_trend == "BUILDING":
        pts["Volume Trend Building"] = 4
    elif volume_trend == "WEAK":
        pts["Volume Trend Weak"] = -7

    if setup_quality == "HIGH":
        pts["Setup Quality High"] = 8
    elif setup_quality == "MEDIUM":
        pts["Setup Quality Medium"] = 3
    elif setup_quality == "LOW":
        pts["Setup Quality Low"] = -9

    if trap_risk == "HIGH":
        pts["High Trap Risk"] = -28
    elif trap_risk == "MEDIUM":
        pts["Medium Trap Risk"] = -11

    # 8. Real S&R structure signals from cached OHLCV analysis.
    if support_touches >= 3:
        pts["Repeated Support Touches"] = 7
    elif support_touches >= 2:
        pts["Valid Support Touches"] = 4
    if resistance_touches >= 3 and base_tightness <= 8.0:
        pts["Resistance Compression Touches"] = 8
    elif resistance_touches >= 2:
        pts["Resistance Touch Count"] = 4
    if pivot_support_quality == "HIGH":
        pts["Pivot Support Quality High"] = 5
    elif pivot_support_quality == "MEDIUM":
        pts["Pivot Support Quality Medium"] = 2
    if pivot_resistance_quality == "HIGH":
        pts["Pivot Resistance Quality High"] = 5
    elif pivot_resistance_quality == "MEDIUM":
        pts["Pivot Resistance Quality Medium"] = 2
    if 0.0 < base_tightness <= MODE7_BASE_TIGHT_HIGH:
        pts["Tight Breakout Base"] = 7
    elif base_tightness <= MODE7_BASE_TIGHT_MEDIUM:
        pts["Controlled Base Tightness"] = 3
    elif base_tightness > MODE7_BASE_LOOSE:
        pts["Loose Volatile Base"] = -6
    if atr_contraction == "YES":
        pts["ATR Contraction Before Breakout"] = 5
    if breakout_retest == "YES":
        pts["Breakout Retest Hold"] = 7
    if liquidity_sweep == "YES":
        pts["Liquidity Sweep Reclaim"] = 4
    if wick_rejection == "HIGH":
        pts["Bullish Wick Rejection"] = 4
    elif wick_rejection == "LOW" and d20h > -1.0:
        pts["Upper Wick Rejection Near Resistance"] = -8
    if resistance_rejections >= 2 and vol_r < 1.25:
        pts["Repeated Resistance Rejection"] = -7
    if sr_score >= MODE7_SR_SCORE_HIGH:
        pts["S&R Structure Score High"] = 5
    elif sr_score < MODE7_SR_SCORE_WEAK:
        pts["S&R Structure Score Weak"] = -5

    if 1.0 <= atr_pct <= 6.0:
        pts["ATR Risk Controlled"] = 3
    elif atr_pct > 9.0:
        pts["ATR Too Hot"] = -6

    score = float(np.clip(sum(pts.values()), 0, 95))
    return score, pts


# ---------------------------------------------------------------------------
# PART 2 - BULL TRAP / TRAP PROBABILITY
# ---------------------------------------------------------------------------
def check_bull_trap_mode7(row: dict) -> str:
    """
    Return LOW / MEDIUM / HIGH trap probability.

    HIGH requires multiple independent risks so clean momentum names are not
    over-filtered.
    """
    ri = _get_num(row, ("RSI",), 50.0)
    vol_r = _get_num(row, ("Vol / Avg", "Vol/Avg"), 1.0)
    d20h = _get_num(row, ("Δ vs 20D High (%)", "Near High (%)"), -5.0)
    de20 = _get_num(row, ("Δ vs EMA20 (%)", "EMA Distance (%)"), 0.0)
    r5d = _get_num(row, ("5D Return (%)",), 0.0)
    resistance_rejections = _get_num(row, ("Resistance Rejections",), 0.0)
    base_tightness = _get_num(row, ("Base Tightness (%)",), 99.0)
    setup_quality = _get_text(row, "Setup Quality", "MEDIUM")
    volume_trend = _get_text(row, "Volume Trend", "NORMAL")
    wick_rejection = _get_text(row, "Wick Rejection", "MEDIUM")
    breakout_retest = _get_text(row, "Breakout Retest", "NO")

    risks = [
        ri > 76.0,
        vol_r < 1.0,
        de20 > MODE7_EMA_EXTENSION_HARD,
        r5d > MODE7_5D_SPIKE,
        setup_quality == "LOW",
        volume_trend == "WEAK",
        (d20h > 1.5 and vol_r < 1.2),
        (d20h > 4.0 and de20 > 6.0),
        (resistance_rejections >= 2 and vol_r < 1.25 and breakout_retest != "YES"),
        (wick_rejection == "LOW" and d20h > -1.0),
        (base_tightness > MODE7_BASE_LOOSE and d20h > -2.0),
    ]
    count = sum(bool(x) for x in risks)

    if count >= 3 or (ri > 80.0 and de20 > 8.0):
        return "HIGH"
    if count >= 1:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# PART 3 - BACKTEST
# ---------------------------------------------------------------------------
def backtest_mode7(row: dict, ticker: str) -> float:
    """
    Mode 7 3-5 day continuation backtest.

    Entry:
      - Vol / Avg > 1.3
      - price > EMA20 > EMA50
      - RSI 52-70
      - near breakout zone
      - not overextended

    Success: any close in days 3-5 is > +2.5%.
    Failure: any close in days 3-5 is < -2.0%.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    with _BT_LOCK:
        if ticker_ns in _BT_CACHE:
            return _BT_CACHE[ticker_ns]

    result = 50.0
    try:
        df = get_df_for_ticker(ticker_ns)
        if df is None or len(df) < 60:
            raise ValueError("insufficient data")

        close = df["Close"].copy()
        volume = df["Volume"].copy()
        e20s = ema(close, 20)
        e50s = ema(close, 50)
        rsi_s = rsi_vec(close)
        avg_vol = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        high_20d = close.rolling(20, min_periods=10).max().shift(1)
        dist_high = (close / high_20d.replace(0, np.nan) - 1.0) * 100.0
        ema_dist = (close / e20s.replace(0, np.nan) - 1.0) * 100.0
        ema_slope = e20s > e20s.shift(1)

        mask = (
            rsi_s.notna()
            & (vol_ratio > 1.3)
            & (close > e20s)
            & (e20s > e50s)
            & ema_slope
            & (rsi_s >= 52.0)
            & (rsi_s <= 70.0)
            & (dist_high >= -5.0)
            & (dist_high <= 2.5)
            & (ema_dist <= 7.0)
            & (ema_dist >= -2.0)
        )

        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 5]
        if len(idx) < 12:
            raise ValueError("too few")

        cv = close.values
        wins = 0
        losses = 0
        neutral = 0
        for i in idx:
            future = cv[i + 3 : i + 6]
            if len(future) == 0 or cv[i] <= 0:
                continue
            best_ret = (np.nanmax(future) / cv[i] - 1.0) * 100.0
            worst_ret = (np.nanmin(future) / cv[i] - 1.0) * 100.0
            if best_ret > 2.5:
                wins += 1
            elif worst_ret < -2.0:
                losses += 1
            else:
                neutral += 1

        resolved = wins + losses
        if resolved >= 8:
            result = round((wins / resolved) * 100.0, 1)
        elif (wins + losses + neutral) > 0:
            result = round(((wins + 0.5 * neutral) / (wins + losses + neutral)) * 100.0, 1)
    except Exception:
        result = 50.0

    with _BT_LOCK:
        if len(_BT_CACHE) >= _BT_CACHE_MAX:
            _BT_CACHE.clear()
        _BT_CACHE[ticker_ns] = result
    return result


# ---------------------------------------------------------------------------
# PART 4 - ML
# ---------------------------------------------------------------------------
def _build_features_mode7(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Mode 7 features:
      RSI, Vol ratio, EMA distance, distance from breakout, 5D return,
      20D return, EMA trend alignment.

    Target: future close after 3 days > current close.
    """
    try:
        if len(close) < 45:
            return None

        e20s = ema(close, 20)
        e50s = ema(close, 50)
        avg_vol = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r = volume / avg_vol.replace(0, np.nan)
        ema_dist = (close / e20s.replace(0, np.nan) - 1.0) * 100.0
        high_20d = close.rolling(20, min_periods=10).max().shift(1)
        dist_high = (close / high_20d.replace(0, np.nan) - 1.0) * 100.0
        rsi_s = rsi_vec(close)
        ret5d = close.pct_change(5) * 100.0
        ret20d = close.pct_change(20) * 100.0
        ema_trend = ((close > e20s) & (e20s > e50s)).astype(int)
        target = (close.shift(-3) > close).astype(int)

        df = pd.DataFrame({
            "rsi": rsi_s,
            "vol_ratio": vol_r,
            "ema_dist": ema_dist,
            "dist_high": dist_high,
            "ret_5d": ret5d,
            "ret_20d": ret20d,
            "ema_trend": ema_trend,
            "target": target,
        }).dropna()

        df = df[
            (df["vol_ratio"] > 1.05)
            & (df["rsi"] >= 48.0)
            & (df["rsi"] <= 74.0)
            & (df["ema_dist"] <= 9.0)
            & (df["dist_high"] >= -8.0)
            & (df["dist_high"] <= 4.0)
            & (df["ema_trend"] == 1)
        ]
        return df if len(df) >= 10 else None
    except Exception:
        return None


def train_model_mode7() -> bool:
    global _MODEL, _SCALER, _TRAINING
    if not _SKLEARN_READY:
        return False
    with _LOCK:
        if _MODEL is not None:
            return True
        if _TRAINING:
            return False
        _TRAINING = True

    try:
        all_rows: list[pd.DataFrame] = []
        for ticker in _TRAIN_TICKERS:
            df_h = get_df_for_ticker(ticker)
            try:
                from time_travel_engine import apply_time_travel_cutoff as _tt_cut_tr

                df_h = _tt_cut_tr(df_h)
            except Exception:
                pass
            if df_h is None:
                continue
            rows = _build_features_mode7(df_h["Close"], df_h["Volume"])
            if rows is not None:
                all_rows.append(rows)

        if not all_rows:
            return False
        data = pd.concat(all_rows, ignore_index=True)
        if len(data) < 70 or data["target"].nunique(dropna=True) < 2:
            return False

        feat_cols = ["rsi", "vol_ratio", "ema_dist", "dist_high", "ret_5d", "ret_20d", "ema_trend"]
        X = data[feat_cols].values
        y = data["target"].values
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=7, stratify=y
            )
        except Exception:
            split = int(len(X) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        mdl = LogisticRegression(
            max_iter=500,
            C=0.42,
            class_weight="balanced",
            solver="lbfgs",
            random_state=7,
        )
        mdl.fit(X_tr_sc, y_tr)
        acc = mdl.score(X_te_sc, y_te)
        print(f"[Mode7 ML] samples={len(data)} acc={acc:.3f}")
        with _LOCK:
            _MODEL, _SCALER = mdl, sc
        return True
    except Exception as exc:
        print(f"[Mode7 ML] train failed: {exc}")
        return False
    finally:
        with _LOCK:
            _TRAINING = False


def predict_ml_mode7(row: dict) -> float:
    if not _SKLEARN_READY:
        return 50.0
    with _LOCK:
        mdl, sc = _MODEL, _SCALER
    if mdl is None or sc is None:
        train_model_mode7()
        with _LOCK:
            mdl, sc = _MODEL, _SCALER
        if mdl is None or sc is None:
            return 50.0

    try:
        ri = _get_num(row, ("RSI",), 58.0)
        vol_r = _get_num(row, ("Vol / Avg", "Vol/Avg"), 1.3)
        de20 = _get_num(row, ("Δ vs EMA20 (%)", "EMA Distance (%)"), 0.0)
        d20h = _get_num(row, ("Δ vs 20D High (%)", "Near High (%)"), -3.0)
        r5d = _get_num(row, ("5D Return (%)",), 3.0)
        r20d = _get_num(row, ("20D Return (%)",), 7.0)
        price = _get_num(row, ("Price (₹)", "Price", "Close"), 0.0)
        e20 = _get_num(row, ("EMA 20", "EMA20"), 0.0)
        e50 = _get_num(row, ("EMA 50", "EMA50"), 0.0)
        ema_trend = _ema_alignment(price, e20, e50)

        feat = np.array([[ri, vol_r, de20, d20h, r5d, r20d, ema_trend]])
        prob = float(mdl.predict_proba(sc.transform(feat))[0][1])

        adj = 0.0
        if -2.0 <= d20h <= 1.5:
            adj += 0.05
        if 1.4 <= vol_r <= 2.8:
            adj += 0.05
        if 55.0 <= ri <= 67.0:
            adj += 0.03
        if 2.0 <= r5d <= 9.0 and 5.0 <= r20d <= 18.0:
            adj += 0.04
        if de20 <= 5.0:
            adj += 0.02

        if vol_r < 1.0:
            adj -= 0.08
        if vol_r > 4.0:
            adj -= 0.06
        if de20 > 7.0:
            adj -= 0.09
        if ri > 74.0:
            adj -= 0.10
        if d20h > 6.0 or d20h < -8.0:
            adj -= 0.07
        if r5d > 14.0:
            adj -= 0.07
        if check_bull_trap_mode7(row) == "HIGH":
            adj -= 0.10

        return round(float(np.clip(prob + adj, 0.01, 0.99)) * 100.0, 1)
    except Exception:
        return 50.0
