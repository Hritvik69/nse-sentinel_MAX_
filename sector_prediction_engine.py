"""
sector_prediction_engine.py
════════════════════════════
Layers 1, 2, 3 of the Sector Prediction System.

  Layer 1 — DATA        Sector OHLC from ALL_DATA (multi-stock aggregation)
  Layer 2 — SIGNAL      8 candle signals + 4 sector signals, all normalised
  Layer 3 — DECISION    Weighted composite model → direction + real probability

Public API
──────────
    predict_sector(sector_name, scan_df, all_data)
    → SectorPrediction (dataclass)

Design rules
────────────
• Zero API calls during prediction (uses ALL_DATA already in memory)
• Every signal is independently computed and normalised 0–100
• Confidence is derived from historical calibration stored in the tracker
• Never crashes — every path returns a neutral prediction on exception
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SignalBreakdown:
    """Normalised (0–100) scores for each independent signal."""
    # ── Candle-level signals ──────────────────────────────────────────
    ema_slope:        float = 50.0   # rising vs falling EMA20
    price_vs_ema:     float = 50.0   # above/below and distance from EMA20
    candle_direction: float = 50.0   # last 3–5 candles net direction
    body_strength:    float = 50.0   # large vs small candle bodies
    consecutive:      float = 50.0   # streak of same-colour candles
    volume_confirm:   float = 50.0   # volume expansion vs contraction
    volatility:       float = 50.0   # compression → expansion (bullish setup)
    momentum:         float = 50.0   # rate-of-change over 5 sessions
    # ── Sector-level signals ─────────────────────────────────────────
    sector_strength:  float = 50.0   # avg scan score of sector stocks
    bullish_pct:      float = 50.0   # % of stocks with bullish action
    money_flow:       float = 50.0   # relative volume trend (proxy for flow)
    participation:    float = 50.0   # how many stocks are confirming


@dataclass
class SectorPrediction:
    sector:          str
    direction:       str            # "Bullish" | "Bearish" | "Sideways"
    confidence:      float          # 0–100 (real, not synthetic)
    raw_score:       float          # 0–100 composite before calibration
    signals:         SignalBreakdown = field(default_factory=SignalBreakdown)
    ohlc_df:         Optional[pd.DataFrame] = None   # aggregated sector OHLC
    leader_ticker:   str = ""
    stocks_used:     list[str] = field(default_factory=list)
    predicted_at:    str = ""       # ISO timestamp
    entry_price:     float = 0.0    # reference price at prediction time
    note:            str = ""


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA
# ══════════════════════════════════════════════════════════════════════

_MIN_ROWS = 30   # minimum candles needed for signal computation


def _build_sector_ohlc(
    tickers: list[str],
    all_data: dict[str, pd.DataFrame | None],
    max_stocks: int = 5,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Build a synthetic sector OHLC DataFrame by equal-weight averaging the
    OHLCV of the top available stocks.

    Returns (ohlc_df, used_tickers).
    """
    frames: list[pd.DataFrame] = []
    used: list[str] = []

    for raw in tickers[:max_stocks * 3]:   # try up to 3× budget to find enough
        if len(used) >= max_stocks:
            break
        tk = raw if raw.endswith(".NS") else f"{raw}.NS"
        df = all_data.get(tk)
        if df is None:
            # try bare name
            df = all_data.get(raw)
        if df is None or df.empty:
            continue
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            continue
        if len(df) < _MIN_ROWS:
            continue
        frames.append(df[list(needed)].copy())
        used.append(raw)

    if not frames:
        return None, []

    # Align on common dates (inner join) then compute equal-weight OHLC
    closes = pd.concat([f["Close"].rename(u) for f, u in zip(frames, used)], axis=1).dropna()
    if closes.empty or len(closes) < _MIN_ROWS:
        return None, used

    common_idx = closes.index
    agg: dict[str, pd.Series] = {}
    for col in ("Open", "High", "Low", "Close"):
        parts = [f[col].reindex(common_idx) for f in frames]
        agg[col] = pd.concat(parts, axis=1).mean(axis=1)
    vol_parts = [f["Volume"].reindex(common_idx) for f in frames]
    agg["Volume"] = pd.concat(vol_parts, axis=1).sum(axis=1)

    out = pd.DataFrame(agg, index=common_idx).dropna()
    return out, used


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — SIGNAL
# ══════════════════════════════════════════════════════════════════════

def _safe(v: object, default: float = 50.0) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _norm(val: float, lo: float, hi: float) -> float:
    """Normalise val in [lo, hi] → 0–100."""
    if hi == lo:
        return 50.0
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))


# ── Candle signals ────────────────────────────────────────────────────

def _sig_ema_slope(close: pd.Series) -> float:
    """EMA20 slope over last 5 bars, normalised."""
    e = _ema(close, 20)
    if len(e) < 6:
        return 50.0
    slope = (e.iloc[-1] - e.iloc[-5]) / (e.iloc[-5] + 1e-9) * 100
    return _norm(slope, -3.0, 3.0)


def _sig_price_vs_ema(close: pd.Series) -> float:
    """Distance of price above/below EMA20 (%), normalised."""
    e = _ema(close, 20)
    if len(e) < 1:
        return 50.0
    dist = (close.iloc[-1] - e.iloc[-1]) / (e.iloc[-1] + 1e-9) * 100
    return _norm(dist, -8.0, 8.0)


def _sig_candle_direction(ohlc: pd.DataFrame, lookback: int = 5) -> float:
    """Net bullish fraction over last `lookback` candles (0 = all red, 100 = all green)."""
    if len(ohlc) < 2:
        return 50.0
    last = ohlc.tail(lookback)
    bulls = (last["Close"] >= last["Open"]).sum()
    return float(bulls / len(last) * 100)


def _sig_body_strength(ohlc: pd.DataFrame, lookback: int = 5) -> float:
    """
    Avg body-to-range ratio over last `lookback` candles.
    Large bodies → strong directional intent.
    """
    last = ohlc.tail(lookback)
    ranges = (last["High"] - last["Low"]).replace(0, np.nan)
    bodies = (last["Close"] - last["Open"]).abs()
    ratio = (bodies / ranges).dropna().mean()
    if math.isnan(ratio):
        return 50.0
    return _norm(ratio, 0.0, 0.85)


def _sig_consecutive(ohlc: pd.DataFrame) -> float:
    """
    Streak of same-colour candles ending at the most recent bar.
    Score > 50 → bullish streak; < 50 → bearish streak.
    """
    if len(ohlc) < 2:
        return 50.0
    colours = (ohlc["Close"] >= ohlc["Open"]).values[::-1]   # newest first
    streak, current = 0, colours[0]
    for c in colours:
        if c == current:
            streak += 1
        else:
            break
    # Bullish streak: score > 50; bearish: < 50
    signed = streak if current else -streak
    return _norm(signed, -6, 6)


def _sig_volume_confirm(ohlc: pd.DataFrame, lookback: int = 5) -> float:
    """
    Compare last candle's volume to avg of previous `lookback` bars.
    Expansion > 1.2× on a bullish candle → strong confirmation.
    """
    if len(ohlc) < lookback + 2:
        return 50.0
    avg_vol = ohlc["Volume"].iloc[-(lookback + 1):-1].mean()
    last_vol = ohlc["Volume"].iloc[-1]
    last_bull = ohlc["Close"].iloc[-1] >= ohlc["Open"].iloc[-1]
    ratio = last_vol / (avg_vol + 1e-9)
    # Volume expansion on bullish = good; on bearish = bad
    signed_ratio = ratio if last_bull else -ratio
    return _norm(signed_ratio, -3.0, 3.0)


def _sig_volatility(ohlc: pd.DataFrame, short: int = 5, long: int = 20) -> float:
    """
    Volatility compression then expansion = bullish setup.
    Ratio = short ATR / long ATR.  < 0.8 = compression (50+), > 1.2 = expansion.
    """
    if len(ohlc) < long + 2:
        return 50.0
    tr = (ohlc["High"] - ohlc["Low"])
    atr_short = tr.iloc[-short:].mean()
    atr_long  = tr.iloc[-long:].mean()
    ratio = atr_short / (atr_long + 1e-9)
    # Compression (ratio < 1) → score > 50 (coiled spring)
    return _norm(1.0 - ratio, -1.0, 1.0)


def _sig_momentum(close: pd.Series, n: int = 5) -> float:
    """5-bar rate-of-change, normalised."""
    if len(close) < n + 1:
        return 50.0
    roc = (close.iloc[-1] - close.iloc[-(n + 1)]) / (close.iloc[-(n + 1)] + 1e-9) * 100
    return _norm(roc, -6.0, 6.0)


# ── Sector signals ────────────────────────────────────────────────────

def _sig_sector_strength(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    """Average Final Score / Prediction Score of sector stocks in the scan."""
    syms = {s.upper().replace(".NS", "") for s in stocks}
    col = next((c for c in ("Final Score", "Prediction Score", "Score") if c in scan_df.columns), None)
    if col is None:
        return 50.0
    mask = scan_df.get("Symbol", scan_df.get("Ticker", pd.Series(dtype=str))).str.upper().str.replace(".NS", "", regex=False).isin(syms)
    sub = scan_df.loc[mask, col]
    if sub.empty:
        return 50.0
    return float(np.clip(sub.mean(), 0, 100))


def _sig_bullish_pct(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    """% of sector stocks with a bullish action signal."""
    syms = {s.upper().replace(".NS", "") for s in stocks}
    sym_col = next((c for c in ("Symbol", "Ticker") if c in scan_df.columns), None)
    if sym_col is None:
        return 50.0
    mask = scan_df[sym_col].str.upper().str.replace(".NS", "", regex=False).isin(syms)
    sub = scan_df.loc[mask]
    if sub.empty:
        return 50.0
    act_col = next((c for c in ("Action", "Signal") if c in sub.columns), None)
    if act_col is None:
        return float(min(len(sub) / max(len(stocks), 1) * 100, 100))
    bullish = sub[act_col].str.contains("Buy|Bullish|🟢", na=False, regex=True).sum()
    return float(bullish / len(sub) * 100)


def _sig_money_flow(ohlc: pd.DataFrame, lookback: int = 10) -> float:
    """
    Proxy for money flow: volume × (close - open) / (high - low + ε).
    Positive avg flow → inflow (bullish).
    """
    if len(ohlc) < lookback:
        return 50.0
    last = ohlc.tail(lookback)
    flow = last["Volume"] * (last["Close"] - last["Open"]) / (last["High"] - last["Low"] + 1e-9)
    mf = float(flow.mean())
    return _norm(mf, -1e7, 1e7)


def _sig_participation(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    """How many of the sector stocks appear in the scan results at all."""
    syms = {s.upper().replace(".NS", "") for s in stocks}
    sym_col = next((c for c in ("Symbol", "Ticker") if c in scan_df.columns), None)
    if sym_col is None:
        return 50.0
    found = scan_df[sym_col].str.upper().str.replace(".NS", "", regex=False).isin(syms).sum()
    return float(np.clip(found / max(len(stocks), 1) * 100, 0, 100))


# ── Compile all signals ───────────────────────────────────────────────

def _compute_signals(
    ohlc: pd.DataFrame,
    scan_df: pd.DataFrame,
    stocks: list[str],
) -> SignalBreakdown:
    close = ohlc["Close"]
    return SignalBreakdown(
        ema_slope        = _sig_ema_slope(close),
        price_vs_ema     = _sig_price_vs_ema(close),
        candle_direction = _sig_candle_direction(ohlc),
        body_strength    = _sig_body_strength(ohlc),
        consecutive      = _sig_consecutive(ohlc),
        volume_confirm   = _sig_volume_confirm(ohlc),
        volatility       = _sig_volatility(ohlc),
        momentum         = _sig_momentum(close),
        sector_strength  = _sig_sector_strength(scan_df, stocks),
        bullish_pct      = _sig_bullish_pct(scan_df, stocks),
        money_flow       = _sig_money_flow(ohlc),
        participation    = _sig_participation(scan_df, stocks),
    )


# ══════════════════════════════════════════════════════════════════════
# LAYER 3 — DECISION
# ══════════════════════════════════════════════════════════════════════

# Weights must sum to 1.0
_WEIGHTS = {
    # Chart / candle signals (60 % weight — these drive visual probability)
    "ema_slope":        0.10,
    "price_vs_ema":     0.08,
    "candle_direction": 0.10,
    "body_strength":    0.07,
    "consecutive":      0.07,
    "volume_confirm":   0.10,
    "volatility":       0.04,
    "momentum":         0.04,
    # Sector signals (40 % weight)
    "sector_strength":  0.12,
    "bullish_pct":      0.12,
    "money_flow":       0.08,
    "participation":    0.08,
}


def _composite_score(sig: SignalBreakdown) -> float:
    total = 0.0
    for attr, w in _WEIGHTS.items():
        total += getattr(sig, attr, 50.0) * w
    return float(np.clip(total, 0, 100))


def _direction_and_raw_confidence(score: float) -> tuple[str, float]:
    """Map composite score (0–100) → direction + raw confidence (0–100)."""
    if score >= 58:
        direction = "Bullish"
        conf = _norm(score, 58, 85)        # 58→0%, 85→100%
        conf = 50 + conf * 0.50            # range: 50–100%
    elif score <= 42:
        direction = "Bearish"
        conf = _norm(100 - score, 58, 85)
        conf = 50 + conf * 0.50
    else:
        direction = "Sideways"
        conf = _norm(abs(score - 50), 0, 8) * 0.5 + 40   # 40–55%
    return direction, float(np.clip(conf, 40, 95))


def _calibrated_confidence(
    direction: str,
    raw_conf: float,
    sector: str,
) -> float:
    """
    Adjust raw confidence using historical accuracy stored by the tracker.
    Falls back to raw_conf when no history is available.
    """
    try:
        from sector_prediction_tracker import get_calibration_factor
        factor = get_calibration_factor(sector, direction)
        return float(np.clip(raw_conf * factor, 35, 95))
    except Exception:
        return raw_conf


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def predict_sector(
    sector_name: str,
    scan_df: pd.DataFrame | None,
    all_data: dict[str, pd.DataFrame | None],
) -> SectorPrediction:
    """
    Full prediction pipeline for one sector.

    Parameters
    ----------
    sector_name : str
    scan_df     : pd.DataFrame   Latest scan output (may be None or empty)
    all_data    : dict           From strategy_engines._engine_utils.ALL_DATA

    Returns
    -------
    SectorPrediction
    """
    now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── 1. Get stock list for this sector ─────────────────────────────
    try:
        from sector_master import get_stocks_in_sector
        stocks = get_stocks_in_sector(sector_name)
    except Exception:
        stocks = []

    if not stocks:
        return SectorPrediction(
            sector=sector_name, direction="Sideways", confidence=50.0,
            raw_score=50.0, predicted_at=now_ts,
            note="No stocks found for this sector.",
        )

    # ── 2. Build sector OHLC ──────────────────────────────────────────
    ohlc, used_tickers = _build_sector_ohlc(stocks, all_data)

    if ohlc is None or len(ohlc) < _MIN_ROWS:
        return SectorPrediction(
            sector=sector_name, direction="Sideways", confidence=50.0,
            raw_score=50.0, stocks_used=used_tickers, predicted_at=now_ts,
            note="Insufficient OHLC data for this sector.",
        )

    # ── 3. Compute signals ────────────────────────────────────────────
    if scan_df is None or scan_df.empty:
        scan_df = pd.DataFrame()

    try:
        signals = _compute_signals(ohlc, scan_df, stocks)
    except Exception as exc:
        signals = SignalBreakdown()

    # ── 4. Composite score + direction ────────────────────────────────
    raw_score = _composite_score(signals)
    direction, raw_conf = _direction_and_raw_confidence(raw_score)
    confidence = _calibrated_confidence(direction, raw_conf, sector_name)

    # ── 5. Entry price (last close of synthetic index) ────────────────
    entry_price = float(ohlc["Close"].iloc[-1])

    return SectorPrediction(
        sector       = sector_name,
        direction    = direction,
        confidence   = round(confidence, 1),
        raw_score    = round(raw_score, 1),
        signals      = signals,
        ohlc_df      = ohlc,
        leader_ticker= used_tickers[0] if used_tickers else "",
        stocks_used  = used_tickers,
        predicted_at = now_ts,
        entry_price  = round(entry_price, 2),
    )