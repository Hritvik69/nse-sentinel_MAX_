"""
sector_prediction_engine.py   (v2 — Institutional Grade)
══════════════════════════════════════════════════════════
Upgraded decision model integrating all 5 new modules:

  Regime detection   → sector_regime_engine
  MTF alignment      → sector_mtf_engine
  Dynamic weights    → sector_dynamic_weights

Hard constraints
────────────────
  • HIGH_VOLATILITY → confidence capped (from regime engine)
  • Signal agreement < 35% → forced Sideways
  • MTF conflict → -8% confidence penalty
  • Participation < 20% → -10% confidence

Public API  (backwards compatible with v1)
──────────
  predict_sector(sector_name, scan_df, all_data, regime_state=None)
      → SectorPrediction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SignalBreakdown:
    ema_slope:        float = 50.0
    price_vs_ema:     float = 50.0
    candle_direction: float = 50.0
    body_strength:    float = 50.0
    consecutive:      float = 50.0
    volume_confirm:   float = 50.0
    volatility:       float = 50.0
    momentum:         float = 50.0
    sector_strength:  float = 50.0
    bullish_pct:      float = 50.0
    money_flow:       float = 50.0
    participation:    float = 50.0


@dataclass
class SectorPrediction:
    sector:            str
    direction:         str
    confidence:        float
    raw_score:         float
    signals:           SignalBreakdown = field(default_factory=SignalBreakdown)
    ohlc_df:           Optional[pd.DataFrame] = None
    leader_ticker:     str = ""
    stocks_used:       list[str] = field(default_factory=list)
    predicted_at:      str = ""
    entry_price:       float = 0.0
    note:              str = ""
    # v2 fields
    regime:            str = "RANGE_BOUND"
    regime_confidence: float = 50.0
    mtf_score:         float = 50.0
    mtf_note:          str = ""
    signal_agreement:  float = 50.0
    dynamic_weights:   dict[str, float] = field(default_factory=dict)
    sideways_forced:   bool = False
    confidence_cap:    float = 95.0


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA
# ══════════════════════════════════════════════════════════════════════

_MIN_ROWS = 30


def _build_sector_ohlc(
    tickers: list[str],
    all_data: dict,
    max_stocks: int = 5,
) -> tuple[pd.DataFrame | None, list[str]]:
    frames, used = [], []
    for raw in tickers[:max_stocks * 3]:
        if len(used) >= max_stocks:
            break
        tk_ns = raw if raw.endswith(".NS") else f"{raw}.NS"
        df = all_data.get(tk_ns)
        if df is None or (hasattr(df, "empty") and df.empty):
            df = all_data.get(raw)
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns) or len(df) < _MIN_ROWS:
            continue
        frames.append(df[list(needed)].copy())
        used.append(raw)

    if not frames:
        return None, []

    close_s = pd.concat([f["Close"].rename(str(i)) for i, f in enumerate(frames)], axis=1).dropna()
    if close_s.empty or len(close_s) < _MIN_ROWS:
        return None, used

    idx = close_s.index
    agg = {c: pd.concat([f[c].reindex(idx) for f in frames], axis=1).mean(axis=1)
           for c in ("Open", "High", "Low", "Close")}
    agg["Volume"] = pd.concat([f["Volume"].reindex(idx) for f in frames], axis=1).sum(axis=1)
    return pd.DataFrame(agg, index=idx).dropna(), used


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — SIGNALS
# ══════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _norm(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo + 1e-9) * 100, 0, 100))


def _sig_ema_slope(close: pd.Series) -> float:
    e = _ema(close, 20)
    return _norm((e.iloc[-1] - e.iloc[-5]) / (e.iloc[-5] + 1e-9) * 100, -3.0, 3.0) if len(e) >= 6 else 50.0


def _sig_price_vs_ema(close: pd.Series) -> float:
    e = _ema(close, 20)
    return _norm((close.iloc[-1] - e.iloc[-1]) / (e.iloc[-1] + 1e-9) * 100, -8.0, 8.0)


def _sig_candle_direction(ohlc: pd.DataFrame, lb: int = 5) -> float:
    t = ohlc.tail(lb)
    return float((t["Close"] >= t["Open"]).sum() / len(t) * 100)


def _sig_body_strength(ohlc: pd.DataFrame, lb: int = 5) -> float:
    t   = ohlc.tail(lb)
    rng = (t["High"] - t["Low"]).replace(0, np.nan)
    r   = ((t["Close"] - t["Open"]).abs() / rng).dropna().mean()
    return _norm(r if not math.isnan(r) else 0.5, 0.0, 0.85)


def _sig_consecutive(ohlc: pd.DataFrame) -> float:
    if len(ohlc) < 2:
        return 50.0
    c = (ohlc["Close"] >= ohlc["Open"]).values[::-1]
    streak, cur = 0, c[0]
    for x in c:
        if x == cur:
            streak += 1
        else:
            break
    return _norm(streak if cur else -streak, -6, 6)


def _sig_volume_confirm(ohlc: pd.DataFrame, lb: int = 5) -> float:
    if len(ohlc) < lb + 2:
        return 50.0
    avg  = ohlc["Volume"].iloc[-(lb + 1):-1].mean()
    lv   = ohlc["Volume"].iloc[-1]
    bull = ohlc["Close"].iloc[-1] >= ohlc["Open"].iloc[-1]
    r    = lv / (avg + 1e-9)
    return _norm(r if bull else -r, -3.0, 3.0)


def _sig_volatility(ohlc: pd.DataFrame) -> float:
    if len(ohlc) < 22:
        return 50.0
    tr = ohlc["High"] - ohlc["Low"]
    return _norm(1.0 - tr.iloc[-5:].mean() / (tr.iloc[-20:].mean() + 1e-9), -1.0, 1.0)


def _sig_momentum(close: pd.Series, n: int = 5) -> float:
    return _norm((close.iloc[-1] - close.iloc[-(n + 1)]) / (close.iloc[-(n + 1)] + 1e-9) * 100, -6.0, 6.0) if len(close) > n else 50.0


def _sig_sector_strength(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    if scan_df.empty:
        return 50.0
    syms = {s.upper().replace(".NS", "") for s in stocks}
    col  = next((c for c in ("Final Score", "Prediction Score") if c in scan_df.columns), None)
    scol = next((c for c in ("Symbol", "Ticker") if c in scan_df.columns), None)
    if col is None or scol is None:
        return 50.0
    mask = scan_df[scol].str.upper().str.replace(".NS", "", regex=False).isin(syms)
    v    = pd.to_numeric(scan_df.loc[mask, col], errors="coerce").dropna()
    return float(np.clip(v.mean(), 0, 100)) if not v.empty else 50.0


def _sig_bullish_pct(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    if scan_df.empty:
        return 50.0
    syms = {s.upper().replace(".NS", "") for s in stocks}
    scol = next((c for c in ("Symbol", "Ticker") if c in scan_df.columns), None)
    if scol is None:
        return 50.0
    mask = scan_df[scol].str.upper().str.replace(".NS", "", regex=False).isin(syms)
    sub  = scan_df.loc[mask]
    if sub.empty:
        return 50.0
    acol = next((c for c in ("Action", "Signal") if c in sub.columns), None)
    if acol is None:
        return float(min(len(sub) / max(len(stocks), 1) * 100, 100))
    return float(sub[acol].str.contains("Buy|Bullish|🟢", na=False, regex=True).sum() / len(sub) * 100)


def _sig_money_flow(ohlc: pd.DataFrame, lb: int = 10) -> float:
    if len(ohlc) < lb:
        return 50.0
    t  = ohlc.tail(lb)
    mf = (t["Volume"] * (t["Close"] - t["Open"]) / (t["High"] - t["Low"] + 1e-9)).mean()
    return _norm(float(mf), -1e7, 1e7)


def _sig_participation(scan_df: pd.DataFrame, stocks: list[str]) -> float:
    if scan_df.empty:
        return 50.0
    syms = {s.upper().replace(".NS", "") for s in stocks}
    scol = next((c for c in ("Symbol", "Ticker") if c in scan_df.columns), None)
    if scol is None:
        return 50.0
    found = scan_df[scol].str.upper().str.replace(".NS", "", regex=False).isin(syms).sum()
    return float(np.clip(found / max(len(stocks), 1) * 100, 0, 100))


def _compute_signals(ohlc, scan_df, stocks) -> SignalBreakdown:
    c = ohlc["Close"]
    return SignalBreakdown(
        ema_slope        = _sig_ema_slope(c),
        price_vs_ema     = _sig_price_vs_ema(c),
        candle_direction = _sig_candle_direction(ohlc),
        body_strength    = _sig_body_strength(ohlc),
        consecutive      = _sig_consecutive(ohlc),
        volume_confirm   = _sig_volume_confirm(ohlc),
        volatility       = _sig_volatility(ohlc),
        momentum         = _sig_momentum(c),
        sector_strength  = _sig_sector_strength(scan_df, stocks),
        bullish_pct      = _sig_bullish_pct(scan_df, stocks),
        money_flow       = _sig_money_flow(ohlc),
        participation    = _sig_participation(scan_df, stocks),
    )


# ══════════════════════════════════════════════════════════════════════
# LAYER 3 — DECISION
# ══════════════════════════════════════════════════════════════════════

def _composite_score(sig: SignalBreakdown, weights: dict[str, float]) -> float:
    return float(np.clip(sum(getattr(sig, k, 50.0) * w for k, w in weights.items()), 0, 100))


def _signal_agreement(sig: SignalBreakdown) -> float:
    vals  = [sig.ema_slope, sig.price_vs_ema, sig.candle_direction, sig.body_strength,
             sig.consecutive, sig.volume_confirm, sig.volatility, sig.momentum,
             sig.sector_strength, sig.bullish_pct, sig.money_flow, sig.participation]
    signs = [1 if v >= 50 else -1 for v in vals]
    return round(abs(sum(signs)) / len(signs) * 100, 1)


def _direction_and_raw_confidence(score: float, sideways_bias: float = 0.0) -> tuple[str, float]:
    eff = score + sideways_bias
    if eff >= 58:
        return "Bullish", float(50.0 + _norm(eff, 58, 85) * 0.50)
    if eff <= 42:
        return "Bearish", float(50.0 + _norm(100 - eff, 58, 85) * 0.50)
    return "Sideways", float(40.0 + _norm(abs(eff - 50), 0, 8) * 0.5)


def _calibrate(direction: str, raw_conf: float, sector: str) -> float:
    try:
        from sector_prediction_tracker import get_calibration_factor
        return float(np.clip(raw_conf * get_calibration_factor(sector, direction), 35, 95))
    except Exception:
        return raw_conf


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def predict_sector(
    sector_name:   str,
    scan_df:       pd.DataFrame | None,
    all_data:      dict,
    regime_state=None,
) -> SectorPrediction:

    now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    try:
        from sector_master import get_stocks_in_sector
        stocks = get_stocks_in_sector(sector_name)
    except Exception:
        stocks = []

    if not stocks:
        return SectorPrediction(sector=sector_name, direction="Sideways",
                                confidence=50.0, raw_score=50.0,
                                predicted_at=now_ts, note="No stocks found.")

    ohlc, used = _build_sector_ohlc(stocks, all_data)
    if ohlc is None or len(ohlc) < _MIN_ROWS:
        return SectorPrediction(sector=sector_name, direction="Sideways",
                                confidence=50.0, raw_score=50.0,
                                stocks_used=used, predicted_at=now_ts,
                                note="Insufficient OHLC data.")

    # ── Regime ───────────────────────────────────────────────────────
    if regime_state is None:
        try:
            from sector_regime_engine import detect_regime
            regime_state = detect_regime(all_data)
        except Exception:
            pass

    regime      = getattr(regime_state, "regime", "RANGE_BOUND")
    regime_conf = getattr(regime_state, "confidence", 50.0)

    try:
        from sector_regime_engine import regime_weight_adjustments
        reg_adj = regime_weight_adjustments(regime)
    except Exception:
        reg_adj = {}

    sideways_bias  = float(reg_adj.pop("_sideways_bias", 0.0))
    confidence_cap = float(reg_adj.pop("_confidence_cap", 95.0))

    # ── Dynamic weights ───────────────────────────────────────────────
    try:
        from sector_dynamic_weights import get_dynamic_weights
        weights = get_dynamic_weights(sector_name, regime, reg_adj)
    except Exception:
        weights = {
            "ema_slope": 0.10, "price_vs_ema": 0.08, "candle_direction": 0.10,
            "body_strength": 0.07, "consecutive": 0.07, "volume_confirm": 0.10,
            "volatility": 0.04, "momentum": 0.04, "sector_strength": 0.12,
            "bullish_pct": 0.12, "money_flow": 0.08, "participation": 0.08,
        }

    # ── Signals ───────────────────────────────────────────────────────
    if scan_df is None or scan_df.empty:
        scan_df = pd.DataFrame()
    try:
        signals = _compute_signals(ohlc, scan_df, stocks)
    except Exception:
        signals = SignalBreakdown()

    # ── MTF alignment ─────────────────────────────────────────────────
    mtf_score, mtf_note, mtf_agree = 50.0, "", True
    try:
        from sector_mtf_engine import compute_mtf_alignment
        mtf       = compute_mtf_alignment(ohlc)
        mtf_score = mtf.alignment_score
        mtf_note  = mtf.note
        mtf_agree = mtf.agreement
    except Exception:
        pass

    # ── Composite (signals 85% + MTF 15%) ────────────────────────────
    raw_score = _composite_score(signals, weights) * 0.85 + mtf_score * 0.15

    # ── Signal agreement constraint ───────────────────────────────────
    agreement      = _signal_agreement(signals)
    sideways_forced = agreement < 35.0

    # ── Direction + confidence ────────────────────────────────────────
    if sideways_forced:
        direction, raw_conf = "Sideways", float(45.0 + agreement * 0.10)
    else:
        direction, raw_conf = _direction_and_raw_confidence(raw_score, sideways_bias)

    # MTF adjustment
    raw_conf = float(np.clip(
        raw_conf + ((mtf_score - 50) * 0.08 if mtf_agree else -8.0),
        35, confidence_cap,
    ))

    # Calibrate from history then apply caps
    confidence = min(_calibrate(direction, raw_conf, sector_name), confidence_cap)

    # Participation penalty
    if signals.participation < 20.0 and direction != "Sideways":
        confidence = float(np.clip(confidence - 10.0, 35.0, confidence_cap))

    return SectorPrediction(
        sector            = sector_name,
        direction         = direction,
        confidence        = round(confidence, 1),
        raw_score         = round(raw_score, 1),
        signals           = signals,
        ohlc_df           = ohlc,
        leader_ticker     = used[0] if used else "",
        stocks_used       = used,
        predicted_at      = now_ts,
        entry_price       = round(float(ohlc["Close"].iloc[-1]), 2),
        regime            = regime,
        regime_confidence = round(regime_conf, 1),
        mtf_score         = round(mtf_score, 1),
        mtf_note          = mtf_note,
        signal_agreement  = round(agreement, 1),
        dynamic_weights   = {k: round(v, 4) for k, v in weights.items()},
        sideways_forced   = sideways_forced,
        confidence_cap    = confidence_cap,
    )