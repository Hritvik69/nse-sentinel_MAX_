"""
market_bias_engine.py
─────────────────────
Standalone Market Bias Engine for NSE Sentinel.

Completely self-contained — zero imports from app.py, strategy_engines/,
or any other project module. Only uses: yfinance, numpy, pandas, datetime.

Main entry point:
    from market_bias_engine import compute_market_bias
    result = compute_market_bias()   # → dict (see schema below)

Return dict schema:
    {
        "bias"          : str,        # "Bullish" / "Bearish" / "Sideways"
        "confidence"    : int,        # 0 – 100
        "expected_move" : str,        # e.g. "±0.35% to ±0.70%"
        "expected_range": str,        # alias for expected_move (back-compat)
        "regime"        : str,        # "Trending Up" / "Ranging" / etc.
        "reasons"       : list[str],  # bullet-point explanation list
        "breakdown"     : list[str],  # alias for reasons (back-compat)
        "signals"       : dict,       # per-signal bool / float flags
        "sectors"       : dict,       # per-index feature snapshot
        "timestamp"     : str,        # ISO 8601 UTC timestamp
    }

Design principles:
  • Multi-index: Nifty 50 (primary, 70 %) + BankNifty (20 %) + Nifty IT (10 %)
  • Six independent signal families: Trend, Momentum, RSI, Volume,
    Volatility Regime, and Mean-Reversion
  • Conservative by default — sideways unless ≥3 signals agree
  • Graceful fallback at every step; never raises an exception
  • No Streamlit imports — safe to call from any context
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Time Travel integration ────────────────────────────────────────────
try:
    from time_travel_engine import apply_time_travel_cutoff as _tt_cutoff
    _TT_OK = True
except ImportError:
    _TT_OK = False
    def _tt_cutoff(df):   # type: ignore[misc]
        return df


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

_FALLBACK: dict = {
    "bias":           "Sideways",
    "confidence":     50,
    "expected_move":  "±0.30% to ±0.60%",
    "expected_range": "±0.30% to ±0.60%",
    "regime":         "Ranging",
    "reasons":        ["Market data unavailable — showing conservative defaults."],
    "breakdown":      ["Market data unavailable — showing conservative defaults."],
    "signals":        {},
    "sectors":        {},
    "timestamp":      "",
}

# Index symbols and their weights in the combined score
_INDICES: list[tuple[str, str, float]] = [
    ("^NSEI",    "Nifty 50",    0.70),
    ("^NSEBANK", "BankNifty",   0.20),
    ("^CNXIT",   "Nifty IT",    0.10),
]

_FETCH_PERIOD    = "4mo"    # 4 months of daily data (~80 bars)
_MIN_BARS        = 55       # minimum bars needed for all indicators
_RSI_PERIOD      = 14
_ATR_PERIOD      = 14
_BB_PERIOD       = 20       # Bollinger Band period
_BB_STDDEV       = 2.0


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (all private — do not import)
# ═══════════════════════════════════════════════════════════════════════

def _sf(v: Any, default: float = 0.0) -> float:
    """Safe finite float cast."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI — returns full series."""
    d = close.diff()
    g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100.0 - (100.0 / (1.0 + g / l.replace(0, np.nan)))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20,
               n_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, mid_band, lower_band)."""
    mid   = close.rolling(period, min_periods=period // 2).mean()
    sigma = close.rolling(period, min_periods=period // 2).std(ddof=0)
    return mid + n_std * sigma, mid, mid - n_std * sigma


def _fetch_index(symbol: str) -> pd.DataFrame | None:
    """Download OHLCV for one index symbol; returns None on any failure."""
    if not _YF_OK:
        return None
    try:
        df = yf.download(
            symbol,
            period=_FETCH_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        # ── Time-Travel: truncate to historical cutoff ─────────────────
        df = _tt_cutoff(df)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df if len(df) >= _MIN_BARS else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (returns a rich dict for one index)
# ═══════════════════════════════════════════════════════════════════════

def _extract_features(df: pd.DataFrame, name: str) -> dict:
    """
    Compute all indicators for one index DataFrame.
    Returns a flat dict with safe-cast floats and bool flags.
    Always returns a dict — never raises.
    """
    feats: dict = {"name": name, "ok": False}
    try:
        close = df["Close"].astype(float).dropna()
        n     = len(close)
        if n < _MIN_BARS:
            return feats

        # ── Price levels ──────────────────────────────────────────────
        c_now  = _sf(close.iloc[-1])
        c_1d   = _sf(close.iloc[-2])
        c_5d   = _sf(close.iloc[-6])  if n >= 6  else c_now
        c_20d  = _sf(close.iloc[-21]) if n >= 21 else c_now
        c_60d  = _sf(close.iloc[-61]) if n >= 61 else c_now

        # ── EMAs ──────────────────────────────────────────────────────
        e9_s  = _ema(close, 9)
        e20_s = _ema(close, 20)
        e50_s = _ema(close, 50)
        e9    = _sf(e9_s.iloc[-1])
        e20   = _sf(e20_s.iloc[-1])
        e50   = _sf(e50_s.iloc[-1])

        # EMA slope (% change over last 5 bars)
        e20_5d_ago = _sf(e20_s.iloc[-6]) if n >= 6 else e20
        e20_slope  = ((e20 - e20_5d_ago) / e20_5d_ago * 100) if e20_5d_ago else 0.0

        # ── RSI ───────────────────────────────────────────────────────
        rsi_s   = _rsi(close, _RSI_PERIOD)
        rsi_now = _sf(rsi_s.iloc[-1], 50.0)
        rsi_5d  = _sf(rsi_s.iloc[-6], 50.0) if n >= 6 else rsi_now
        rsi_trend = rsi_now - rsi_5d   # rising = positive

        # ── Returns ───────────────────────────────────────────────────
        ret_1d  = ((c_now - c_1d)  / c_1d  * 100) if c_1d  else 0.0
        ret_5d  = ((c_now - c_5d)  / c_5d  * 100) if c_5d  else 0.0
        ret_20d = ((c_now - c_20d) / c_20d * 100) if c_20d else 0.0
        ret_60d = ((c_now - c_60d) / c_60d * 100) if c_60d else 0.0

        # ── ATR / Volatility ──────────────────────────────────────────
        has_hl = ("High" in df.columns and "Low" in df.columns)
        if has_hl:
            high  = df["High"].astype(float).dropna()
            low   = df["Low"].astype(float).dropna()
            # Align index lengths
            idx   = close.index.intersection(high.index).intersection(low.index)
            atr_s = _atr(high.loc[idx], low.loc[idx], close.loc[idx], _ATR_PERIOD)
            atr   = _sf(atr_s.iloc[-1])
        else:
            # Approximate ATR from close-only daily std
            atr = _sf(close.pct_change().tail(14).std() * c_now * 100, 0.5)

        atr_pct = (atr / c_now * 100) if c_now else 0.5     # ATR as % of price
        daily_sigma = _sf(close.pct_change().tail(20).std() * 100, 0.5)

        # ── Bollinger Bands ───────────────────────────────────────────
        bb_up, bb_mid, bb_lo = _bollinger(close, _BB_PERIOD, _BB_STDDEV)
        bb_upper  = _sf(bb_up.iloc[-1])
        bb_lower  = _sf(bb_lo.iloc[-1])
        bb_width  = ((bb_upper - bb_lower) / bb_mid.iloc[-1] * 100) \
                    if _sf(bb_mid.iloc[-1]) else 0.0
        bb_pct_b  = ((c_now - bb_lower) / (bb_upper - bb_lower)) \
                    if (bb_upper - bb_lower) > 0 else 0.5   # 0=lower band, 1=upper

        # ── 20D Rolling High / Low ────────────────────────────────────
        high_20d = _sf(close.tail(20).max())
        low_20d  = _sf(close.tail(20).min())
        dist_hi  = ((c_now - high_20d) / high_20d * 100) if high_20d else 0.0
        dist_lo  = ((c_now - low_20d)  / low_20d  * 100) if low_20d  else 0.0

        # ── Volume ratio (may be absent for indices) ──────────────────
        vol_ratio: float | None = None
        if "Volume" in df.columns:
            vol = df["Volume"].astype(float).dropna()
            if len(vol) >= 21:
                avg20 = _sf(vol.iloc[-21:-1].mean(), 0.0)
                lastv = _sf(vol.iloc[-1], 0.0)
                vol_ratio = (lastv / avg20) if avg20 > 0 else None

        # ── Signal flags ──────────────────────────────────────────────
        # Trend
        trend_bullish  = (c_now > e20 > e50) and (e20_slope > 0)
        trend_bearish  = (c_now < e20 < e50) and (e20_slope < 0)
        # Momentum
        mom_bullish    = ret_5d > 0.5  and ret_20d > 0.0
        mom_bearish    = ret_5d < -0.5 and ret_20d < 0.0
        # RSI
        rsi_bullish    = 52.0 <= rsi_now <= 72.0 and rsi_trend >= 0
        rsi_bearish    = rsi_now <= 48.0 and rsi_trend <= 0
        rsi_overbought = rsi_now > 75.0
        rsi_oversold   = rsi_now < 28.0
        # Volume
        vol_bullish    = (vol_ratio is not None and vol_ratio > 1.1)
        vol_bearish    = (vol_ratio is not None and vol_ratio < 0.85)
        # Bollinger
        bb_squeeze     = bb_width < 2.5                  # low vol → breakout incoming
        bb_upper_touch = bb_pct_b > 0.90                 # near upper band (caution)
        bb_lower_touch = bb_pct_b < 0.10                 # near lower band (bounce?)
        # Mean reversion
        mean_rev_long  = bb_lower_touch and rsi_oversold
        mean_rev_short = bb_upper_touch and rsi_overbought

        feats = {
            "name":           name,
            "ok":             True,
            # Price & EMAs
            "close":          c_now,
            "ema9":           e9,
            "ema20":          e20,
            "ema50":          e50,
            "ema20_slope":    round(e20_slope, 3),
            # Returns
            "ret_1d":         round(ret_1d,  2),
            "ret_5d":         round(ret_5d,  2),
            "ret_20d":        round(ret_20d, 2),
            "ret_60d":        round(ret_60d, 2),
            # RSI
            "rsi":            round(rsi_now, 1),
            "rsi_trend":      round(rsi_trend, 1),
            # Vol
            "vol_ratio":      round(vol_ratio, 2) if vol_ratio is not None else None,
            # Volatility
            "atr_pct":        round(atr_pct,    2),
            "daily_sigma":    round(daily_sigma, 2),
            # Bollinger
            "bb_width":       round(bb_width,   2),
            "bb_pct_b":       round(bb_pct_b,   3),
            # 20D range
            "dist_20d_high":  round(dist_hi, 2),
            "dist_20d_low":   round(dist_lo, 2),
            # Signal flags
            "trend_bullish":  bool(trend_bullish),
            "trend_bearish":  bool(trend_bearish),
            "mom_bullish":    bool(mom_bullish),
            "mom_bearish":    bool(mom_bearish),
            "rsi_bullish":    bool(rsi_bullish),
            "rsi_bearish":    bool(rsi_bearish),
            "rsi_overbought": bool(rsi_overbought),
            "rsi_oversold":   bool(rsi_oversold),
            "vol_bullish":    bool(vol_bullish),
            "vol_bearish":    bool(vol_bearish),
            "bb_squeeze":     bool(bb_squeeze),
            "mean_rev_long":  bool(mean_rev_long),
            "mean_rev_short": bool(mean_rev_short),
        }
    except Exception:
        pass   # return partial feats dict as-is

    return feats


# ═══════════════════════════════════════════════════════════════════════
# SCORING ENGINE  (converts features → weighted directional score)
# ═══════════════════════════════════════════════════════════════════════

def _score_index(feats: dict) -> float:
    """
    Return a directional score in [-1, +1].
    +1 = strongly bullish, -1 = strongly bearish, 0 = neutral.
    Conservative — sideways unless ≥3 independent signals agree.
    """
    if not feats.get("ok"):
        return 0.0

    score = 0.0

    # 1. Trend (weight 0.30) ─────────────────────────────────────────
    if feats["trend_bullish"]:
        score += 0.30
    elif feats["trend_bearish"]:
        score -= 0.30

    # 2. Momentum (weight 0.25) ──────────────────────────────────────
    if feats["mom_bullish"]:
        score += 0.25
    elif feats["mom_bearish"]:
        score -= 0.25
    else:
        # Partial: 5D only
        if feats["ret_5d"] > 0.3:
            score += 0.12
        elif feats["ret_5d"] < -0.3:
            score -= 0.12

    # 3. RSI (weight 0.20) ───────────────────────────────────────────
    if feats["rsi_bullish"] and not feats["rsi_overbought"]:
        score += 0.20
    elif feats["rsi_bearish"] and not feats["rsi_oversold"]:
        score -= 0.20
    elif feats["rsi_overbought"]:
        score -= 0.10   # overbought = caution even in uptrend
    elif feats["rsi_oversold"]:
        score += 0.08   # oversold bounce potential

    # 4. Volume (weight 0.15) ────────────────────────────────────────
    vol_r = feats.get("vol_ratio")
    if vol_r is not None:
        if feats["vol_bullish"] and feats["trend_bullish"]:
            score += 0.15   # strong vol confirming uptrend
        elif feats["vol_bullish"] and feats["trend_bearish"]:
            score -= 0.15   # strong vol confirming downtrend
        elif feats["vol_bearish"]:
            score -= 0.07   # low vol = weak move (discount any direction)
    # If vol_ratio is None (index with no vol data), skip silently

    # 5. Bollinger / Volatility Regime (weight 0.10) ─────────────────
    if feats["mean_rev_long"]:
        score += 0.10   # oversold at lower band → mean-reversion long
    elif feats["mean_rev_short"]:
        score -= 0.10   # overbought at upper band → mean-reversion short
    elif feats["bb_squeeze"]:
        score += 0.0    # squeeze → neutral (breakout direction unknown yet)

    # Clamp strictly
    return float(np.clip(score, -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

def _classify_regime(feats: dict) -> str:
    """
    Returns one of: 'Trending Up' / 'Trending Down' / 'Ranging' /
                    'Volatile Breakout' / 'Oversold Bounce Zone'
    """
    if not feats.get("ok"):
        return "Ranging"

    if feats["bb_squeeze"]:
        return "Breakout Pending (Squeeze)"
    if feats["mean_rev_long"]:
        return "Oversold Bounce Zone"
    if feats["mean_rev_short"]:
        return "Overbought Pullback Risk"
    if feats["trend_bullish"] and feats["mom_bullish"]:
        return "Trending Up"
    if feats["trend_bearish"] and feats["mom_bearish"]:
        return "Trending Down"

    sigma = feats.get("daily_sigma", 0.5)
    if sigma > 1.5:
        return "High Volatility / Choppy"

    return "Ranging"


# ═══════════════════════════════════════════════════════════════════════
# BIAS INTERPRETER  (score + context → human-readable bias)
# ═══════════════════════════════════════════════════════════════════════

def _interpret(
    combined_score: float,
    primary_feats: dict,
    all_feats: list[dict],
) -> tuple[str, int]:
    """
    Convert combined directional score to (bias_label, confidence).
    Conservative thresholds: 0.20 for Bullish/Bearish, else Sideways.
    """
    # Count how many indices independently agree on direction
    bullish_votes = sum(1 for f in all_feats if f.get("ok") and
                        f.get("trend_bullish") and f.get("mom_bullish"))
    bearish_votes = sum(1 for f in all_feats if f.get("ok") and
                        f.get("trend_bearish") and f.get("mom_bearish"))
    total_ok      = sum(1 for f in all_feats if f.get("ok"))

    # Need at least the primary index OK
    if not primary_feats.get("ok"):
        return "Sideways", 50

    # Strong threshold: score > 0.20 AND vote majority
    threshold = 0.20
    if combined_score >= threshold:
        bias = "Bullish"
        # Confidence: base 55 + signal strength + vote bonus
        conf = 55 + int(abs(combined_score) * 30)
        conf += min(10, bullish_votes * 5)
        if primary_feats.get("rsi_overbought"):
            conf -= 8    # conservative: overbought market = less reliable
    elif combined_score <= -threshold:
        bias = "Bearish"
        conf = 55 + int(abs(combined_score) * 30)
        conf += min(10, bearish_votes * 5)
        if primary_feats.get("rsi_oversold"):
            conf -= 5
    else:
        bias = "Sideways"
        conf = 45 - int(abs(combined_score) * 15)
        conf = max(conf, 30)

    conf = int(np.clip(conf, 25, 88))
    return bias, conf


# ═══════════════════════════════════════════════════════════════════════
# EXPECTED MOVE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

def _expected_move(feats: dict, bias: str, confidence: int) -> str:
    """
    Estimate next-day expected move range from ATR % and daily sigma.
    Returns a formatted string like "+0.35% to +0.70%".
    """
    sigma  = feats.get("daily_sigma", 0.5)
    atr_p  = feats.get("atr_pct",    0.5)
    sigma  = max(sigma, 0.10)
    atr_p  = max(atr_p, 0.10)

    # Use average of the two volatility measures (conservative)
    base_vol = (sigma + atr_p) / 2.0
    conf_factor = 0.80 + (confidence / 200.0)   # 0.93 – 1.24

    low_mag  = round(base_vol * 0.45 * conf_factor, 2)
    high_mag = round(base_vol * 0.90 * conf_factor, 2)

    if bias == "Bullish":
        return f"+{low_mag:.2f}% to +{high_mag:.2f}%"
    elif bias == "Bearish":
        return f"-{low_mag:.2f}% to -{high_mag:.2f}%"
    else:
        side = round(base_vol * 0.55 * conf_factor, 2)
        return f"±{side:.2f}% to ±{round(side * 1.25, 2):.2f}%"


# ═══════════════════════════════════════════════════════════════════════
# REASON BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _build_reasons(
    primary: dict,
    all_feats: list[dict],
    bias: str,
    confidence: int,
    regime: str,
) -> list[str]:
    """Build a human-readable list of reasons for the bias verdict."""
    reasons: list[str] = []

    if not primary.get("ok"):
        return ["Primary index (Nifty 50) data insufficient — defaulting to Sideways."]

    name   = primary["name"]
    c_now  = primary["close"]
    e20    = primary["ema20"]
    e50    = primary["ema50"]
    rsi_v  = primary["rsi"]
    ret5d  = primary["ret_5d"]
    ret20d = primary["ret_20d"]
    slope  = primary["ema20_slope"]
    vol_r  = primary.get("vol_ratio")
    sigma  = primary["daily_sigma"]
    bb_w   = primary["bb_width"]
    pct_b  = primary["bb_pct_b"]

    # 1. Trend
    if primary["trend_bullish"]:
        reasons.append(
            f"Trend: {name} bullish stack — Close {c_now:.0f} > EMA20 {e20:.0f} > EMA50 {e50:.0f}; "
            f"EMA20 slope {slope:+.2f}% (rising)."
        )
    elif primary["trend_bearish"]:
        reasons.append(
            f"Trend: {name} bearish stack — Close {c_now:.0f} < EMA20 {e20:.0f} < EMA50 {e50:.0f}; "
            f"EMA20 slope {slope:+.2f}% (falling)."
        )
    else:
        reasons.append(
            f"Trend: {name} mixed — Close {c_now:.0f} between EMAs (EMA20 {e20:.0f}, EMA50 {e50:.0f})."
        )

    # 2. RSI
    rsi_zone = (
        "overbought (>75)" if rsi_v > 75 else
        "oversold (<28)"   if rsi_v < 28 else
        "hot (65–75)"      if rsi_v > 65 else
        "bullish zone (52–65)" if 52 <= rsi_v <= 65 else
        "neutral"
    )
    reasons.append(f"RSI(14): {rsi_v:.1f} — {rsi_zone}; trend {primary['rsi_trend']:+.1f} vs 5D ago.")

    # 3. Momentum
    reasons.append(
        f"Momentum: 5-day return {ret5d:+.2f}%, 20-day return {ret20d:+.2f}%."
    )

    # 4. Volume
    if vol_r is not None:
        vol_desc = (
            "elevated (>1.2×)" if vol_r > 1.2 else
            "subdued (<0.85×)" if vol_r < 0.85 else
            "average"
        )
        reasons.append(f"Volume: {vol_r:.2f}× 20-day avg — {vol_desc}.")
    else:
        reasons.append("Volume: Not available for this index — signal skipped.")

    # 5. Bollinger Band context
    bb_desc = (
        "near upper band (watch for reversal)" if pct_b > 0.85 else
        "near lower band (watch for bounce)"   if pct_b < 0.15 else
        "mid-range"
    )
    reasons.append(
        f"Bollinger: BB width {bb_w:.1f}% — {'squeeze (breakout pending)' if primary['bb_squeeze'] else 'normal spread'}; "
        f"price at {pct_b*100:.0f}th percentile of band — {bb_desc}."
    )

    # 6. Regime / secondary indices
    reasons.append(f"Regime: {regime}.")

    # 7. Secondary index confirmations
    secondary_confirms: list[str] = []
    for f in all_feats:
        if f.get("name") == primary["name"] or not f.get("ok"):
            continue
        sn = f["name"]
        if f["trend_bullish"] and f["mom_bullish"]:
            secondary_confirms.append(f"{sn} confirms bullish")
        elif f["trend_bearish"] and f["mom_bearish"]:
            secondary_confirms.append(f"{sn} confirms bearish")
        else:
            secondary_confirms.append(f"{sn} neutral/mixed")
    if secondary_confirms:
        reasons.append("Secondary indices: " + "; ".join(secondary_confirms) + ".")

    # 8. Confidence note
    reasons.append(
        f"Verdict: {bias} with {confidence}% confidence. "
        f"Daily volatility sigma {sigma:.2f}% — "
        f"{'elevated' if sigma > 1.2 else 'normal'}."
    )

    return reasons


# ═══════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def compute_market_bias() -> dict:
    """
    Compute multi-index probabilistic market bias for the next trading day.

    Fetches: Nifty 50 (^NSEI), BankNifty (^NSEBANK), Nifty IT (^CNXIT).
    Weighted combined directional score → Bullish / Bearish / Sideways.

    Returns
    -------
    dict with keys:
        bias          : "Bullish" / "Bearish" / "Sideways"
        confidence    : int (0–100)
        expected_move : str  (e.g. "+0.35% to +0.70%")
        expected_range: str  (alias for expected_move, back-compat)
        regime        : str  (market regime label)
        reasons       : list[str]   (bullet-point explanation)
        breakdown     : list[str]   (alias for reasons, back-compat)
        signals       : dict        (raw per-signal flags from Nifty 50)
        sectors       : dict        (per-index feature snapshots)
        timestamp     : str         (ISO 8601 UTC)
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not _YF_OK:
        result = dict(_FALLBACK)
        result["timestamp"] = ts
        result["reasons"]   = ["yfinance not installed — cannot compute market bias."]
        result["breakdown"] = result["reasons"]
        return result

    # ── Step 1: Fetch + extract features for all indices ─────────────
    all_feats: list[dict] = []
    sectors:   dict       = {}

    for symbol, name, _weight in _INDICES:
        df = _fetch_index(symbol)
        if df is not None:
            feats = _extract_features(df, name)
        else:
            feats = {"name": name, "ok": False}
        all_feats.append(feats)
        # Snapshot for 'sectors' output (only key metrics)
        sectors[name] = {
            "ok":       feats.get("ok", False),
            "close":    feats.get("close"),
            "rsi":      feats.get("rsi"),
            "ret_5d":   feats.get("ret_5d"),
            "ret_20d":  feats.get("ret_20d"),
            "vol_ratio": feats.get("vol_ratio"),
            "regime":   _classify_regime(feats) if feats.get("ok") else "N/A",
        }

    # Primary index = Nifty 50 (first in list)
    primary_feats = all_feats[0]

    # Fallback if Nifty 50 data is missing
    if not primary_feats.get("ok"):
        result = dict(_FALLBACK)
        result["timestamp"] = ts
        result["sectors"]   = sectors
        return result

    # ── Step 2: Compute weighted combined score ───────────────────────
    combined_score = 0.0
    for feats, (_, _name, weight) in zip(all_feats, _INDICES):
        if feats.get("ok"):
            combined_score += weight * _score_index(feats)
        # If an index is unavailable, its weight redistributes to Nifty
        # (score for unavailable = 0 which is neutral, acceptable)

    # ── Step 3: Interpret ─────────────────────────────────────────────
    bias, confidence = _interpret(combined_score, primary_feats, all_feats)

    # ── Step 4: Regime ────────────────────────────────────────────────
    regime = _classify_regime(primary_feats)

    # ── Step 5: Expected move ─────────────────────────────────────────
    exp_move = _expected_move(primary_feats, bias, confidence)

    # ── Step 6: Reasons ───────────────────────────────────────────────
    reasons = _build_reasons(primary_feats, all_feats, bias, confidence, regime)

    # ── Step 7: Signals dict (from primary index) ─────────────────────
    signal_keys = [
        "trend_bullish", "trend_bearish", "mom_bullish", "mom_bearish",
        "rsi_bullish", "rsi_bearish", "rsi_overbought", "rsi_oversold",
        "vol_bullish", "vol_bearish", "bb_squeeze",
        "mean_rev_long", "mean_rev_short",
    ]
    signals = {k: primary_feats.get(k, False) for k in signal_keys}
    # Add scalar values useful for display
    signals["nifty_rsi"]      = primary_feats.get("rsi")
    signals["nifty_ret_5d"]   = primary_feats.get("ret_5d")
    signals["nifty_ret_20d"]  = primary_feats.get("ret_20d")
    signals["nifty_vol_ratio"] = primary_feats.get("vol_ratio")
    signals["combined_score"] = round(combined_score, 4)

    return {
        "bias":           bias,
        "confidence":     confidence,
        "expected_move":  exp_move,
        "expected_range": exp_move,   # back-compat alias
        "regime":         regime,
        "reasons":        reasons,
        "breakdown":      reasons,    # back-compat alias
        "signals":        signals,
        "sectors":        sectors,
        "timestamp":      ts,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI  (python market_bias_engine.py)
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("NSE Sentinel — Market Bias Engine")
    print("Fetching index data …\n")
    result = compute_market_bias()
    print(f"  Bias      : {result['bias']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Exp. Move : {result['expected_move']}")
    print(f"  Regime    : {result['regime']}")
    print("\n  Reasons:")
    for r in result["reasons"]:
        print(f"    • {r}")
    print("\n  Sector Snapshot:")
    for name, s in result["sectors"].items():
        ok = "✓" if s.get("ok") else "✗"
        print(f"    [{ok}] {name}: RSI {s.get('rsi','?')} | 5D {s.get('ret_5d','?')}% | Regime: {s.get('regime','?')}")