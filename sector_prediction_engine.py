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
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from feature_data_manager import feature_manager, get_current_window
from sector_ohlc_utils import build_weighted_synthetic_ohlc


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
    ohlc_source:       str = ""
    ohlc_symbol:       str = ""
    ohlc_bars:         int = 0


def _prediction_to_payload(pred: SectorPrediction) -> dict:
    return {
        "sector": pred.sector,
        "direction": pred.direction,
        "confidence": pred.confidence,
        "raw_score": pred.raw_score,
        "signals": dict(vars(pred.signals)),
        "leader_ticker": pred.leader_ticker,
        "stocks_used": list(pred.stocks_used),
        "predicted_at": pred.predicted_at,
        "entry_price": pred.entry_price,
        "note": pred.note,
        "regime": pred.regime,
        "regime_confidence": pred.regime_confidence,
        "mtf_score": pred.mtf_score,
        "mtf_note": pred.mtf_note,
        "signal_agreement": pred.signal_agreement,
        "dynamic_weights": dict(pred.dynamic_weights),
        "sideways_forced": pred.sideways_forced,
        "confidence_cap": pred.confidence_cap,
        "ohlc_source": pred.ohlc_source,
        "ohlc_symbol": pred.ohlc_symbol,
        "ohlc_bars": pred.ohlc_bars,
    }


def _prediction_from_payload(payload: dict, ohlc_df: pd.DataFrame | None) -> SectorPrediction:
    signal_payload = payload.get("signals", {}) if isinstance(payload.get("signals"), dict) else {}
    return SectorPrediction(
        sector=str(payload.get("sector", "") or ""),
        direction=str(payload.get("direction", "Sideways") or "Sideways"),
        confidence=float(payload.get("confidence", 50.0) or 50.0),
        raw_score=float(payload.get("raw_score", 50.0) or 50.0),
        signals=SignalBreakdown(**signal_payload),
        ohlc_df=ohlc_df,
        leader_ticker=str(payload.get("leader_ticker", "") or ""),
        stocks_used=list(payload.get("stocks_used", []) or []),
        predicted_at=str(payload.get("predicted_at", "") or ""),
        entry_price=float(payload.get("entry_price", 0.0) or 0.0),
        note=str(payload.get("note", "") or ""),
        regime=str(payload.get("regime", "RANGE_BOUND") or "RANGE_BOUND"),
        regime_confidence=float(payload.get("regime_confidence", 50.0) or 50.0),
        mtf_score=float(payload.get("mtf_score", 50.0) or 50.0),
        mtf_note=str(payload.get("mtf_note", "") or ""),
        signal_agreement=float(payload.get("signal_agreement", 50.0) or 50.0),
        dynamic_weights=dict(payload.get("dynamic_weights", {}) or {}),
        sideways_forced=bool(payload.get("sideways_forced", False)),
        confidence_cap=float(payload.get("confidence_cap", 95.0) or 95.0),
        ohlc_source=str(payload.get("ohlc_source", "") or ""),
        ohlc_symbol=str(payload.get("ohlc_symbol", "") or ""),
        ohlc_bars=int(payload.get("ohlc_bars", 0) or 0),
    )


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA
# ══════════════════════════════════════════════════════════════════════

_MIN_ROWS = 30
_TARGET_ROWS = 60
_MAX_AGGREGATION_STOCKS = 80
_MIN_AGGREGATION_STOCKS = 3
_WEIGHT_LOOKBACK = 20

_SECTOR_INDEX_CANDIDATES: dict[str, list[str]] = {
    "OVERALL": ["^NSEI"],
    "NIFTY_50": ["^NSEI"],
    "BANKING": ["^NSEBANK"],
    "IT": ["^CNXIT"],
    "AUTO": ["^CNXAUTO"],
    "FMCG": ["^CNXFMCG"],
    "PHARMA": ["^CNXPHARMA"],
    "METAL": ["^CNXMETAL"],
    "ENERGY": ["^CNXENERGY"],
    "REALTY": ["^CNXREALTY"],
    "INFRA": ["^CNXINFRA"],
}

_SYMBOL_ALIASES: dict[str, list[str]] = {
    "ZOMATO": ["ETERNAL"],
    "BIRLASOFT": ["BSOFT"],
    "BALKRISHIND": ["BALKRISIND"],
}


def _sector_key(value: str) -> str:
    return str(value or "").upper().strip().replace("&", "_").replace("-", "_").replace(" ", "_")


def _plain_symbol(value: str) -> str:
    return str(value or "").upper().strip().replace(".NS", "")


def _normalize_ohlc_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    try:
        if df is None or df.empty:
            return None
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out.columns = [str(col).strip().title() for col in out.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not set(needed).issubset(out.columns):
            return None
        out = out[needed].copy()
        dt_index = pd.to_datetime(out.index, errors="coerce")
        mask = ~dt_index.isna()
        out = out.loc[mask].copy()
        out.index = dt_index[mask]
        out = out[~out.index.duplicated(keep="last")].sort_index()
        for col in needed:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        out = out[(out["High"] >= out[["Open", "Close", "Low"]].max(axis=1))]
        out = out[(out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))]
        return out if not out.empty else None
    except Exception:
        return None


def _cache_sector_frame(all_data: dict, key: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    try:
        import time_travel_engine as _tt

        cutoff = _tt.get_reference_date()
        if cutoff is not None:
            _tt.cache_frame(key, df, cutoff, min_rows=5)
            return
    except Exception:
        pass
    try:
        all_data[key] = df
    except Exception:
        pass
    try:
        from strategy_engines._engine_utils import ALL_DATA as _ALL_DATA, _ALL_DATA_LOCK
        with _ALL_DATA_LOCK:
            _ALL_DATA[key] = df
    except Exception:
        pass


def _lookup_cached_frame(all_data: dict, *keys: str) -> pd.DataFrame | None:
    for key in keys:
        try:
            cached = _normalize_ohlc_frame(all_data.get(key))
            if cached is not None:
                return cached
        except Exception:
            continue
    try:
        from strategy_engines._engine_utils import ALL_DATA as _ALL_DATA, _ALL_DATA_LOCK
        with _ALL_DATA_LOCK:
            for key in keys:
                cached = _normalize_ohlc_frame(_ALL_DATA.get(key))
                if cached is not None:
                    return cached
    except Exception:
        pass
    return None


def _apply_time_travel_cutoff(df: pd.DataFrame | None) -> pd.DataFrame | None:
    try:
        from strategy_engines._engine_utils import _apply_time_travel_cutoff_if_needed
        return _normalize_ohlc_frame(_apply_time_travel_cutoff_if_needed(df))
    except Exception:
        return _normalize_ohlc_frame(df)


def _fetch_sector_index_frame(symbol: str, all_data: dict, force_refresh: bool = False) -> pd.DataFrame | None:
    key = str(symbol or "").strip().upper()
    cached = _lookup_cached_frame(all_data, key)
    if cached is not None and len(cached) >= _MIN_ROWS:
        return cached.tail(_TARGET_ROWS).copy()
    try:
        df = feature_manager.get_symbol_data(
            key,
            period="6mo",
            interval="1d",
            force_refresh=force_refresh,
            append_nse_suffix=False,
            min_rows=_MIN_ROWS,
            allow_snapshot=False,
        )
        normalized = _apply_time_travel_cutoff(df)
        if normalized is not None and len(normalized) >= _MIN_ROWS:
            normalized = normalized.tail(_TARGET_ROWS).copy()
            _cache_sector_frame(all_data, key, normalized)
            return normalized
    except Exception:
        pass
    try:
        df = yf.download(
            key,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
            threads=False,
        )
    except Exception:
        return None
    normalized = _apply_time_travel_cutoff(df)
    if normalized is None or len(normalized) < _MIN_ROWS:
        return None
    normalized = normalized.tail(_TARGET_ROWS).copy()
    _cache_sector_frame(all_data, key, normalized)
    return normalized


def _fetch_stock_frame(symbol: str, all_data: dict, force_refresh: bool = False) -> pd.DataFrame | None:
    raw = str(symbol or "").strip()
    plain = _plain_symbol(raw)
    aliases = [*_SYMBOL_ALIASES.get(plain, []), plain]
    cache_keys: list[str] = [raw]
    for alias in aliases:
        cache_keys.extend([alias, f"{alias}.NS"])
    cached = _lookup_cached_frame(all_data, *cache_keys)
    if cached is not None:
        return cached
    for alias in aliases:
        try:
            fetched = feature_manager.get_stock_data(
                alias,
                period="6mo",
                interval="1d",
                force_refresh=force_refresh,
            )
            fetched = _normalize_ohlc_frame(fetched)
        except Exception:
            fetched = None
        if fetched is not None:
            try:
                all_data[f"{alias}.NS"] = fetched
                all_data[plain] = fetched
            except Exception:
                pass
            return fetched
    return None


def _rank_sector_tickers(tickers: list[str], scan_df: pd.DataFrame | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in tickers:
        sym = _plain_symbol(item)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        ordered.append(sym)

    if scan_df is None or scan_df.empty:
        return ordered

    symbol_col = next((col for col in ("Symbol", "Ticker") if col in scan_df.columns), None)
    score_col = next((col for col in ("Final Score", "Prediction Score", "Rank Score") if col in scan_df.columns), None)
    if symbol_col is None or score_col is None:
        return ordered

    scores: dict[str, float] = {}
    for _, row in scan_df[[symbol_col, score_col]].iterrows():
        sym = _plain_symbol(row.get(symbol_col, ""))
        score = pd.to_numeric(row.get(score_col, np.nan), errors="coerce")
        if sym and pd.notna(score):
            scores[sym] = float(score)

    base_order = {sym: idx for idx, sym in enumerate(ordered)}
    return sorted(
        ordered,
        key=lambda sym: (-scores.get(sym, -1e9), base_order.get(sym, 10_000)),
    )


def _aggregate_weighted_sector_ohlc(
    ranked_tickers: list[str],
    all_data: dict,
    force_refresh: bool = False,
    max_components: int = _MAX_AGGREGATION_STOCKS,
) -> tuple[pd.DataFrame | None, list[str]]:
    panels: list[tuple[str, pd.DataFrame]] = []
    for symbol in ranked_tickers:
        frame = _fetch_stock_frame(symbol, all_data, force_refresh=force_refresh)
        if frame is None or len(frame) < _MIN_ROWS:
            continue
        panels.append((symbol, frame[["Open", "High", "Low", "Close", "Volume"]].copy()))

    if len(panels) < _MIN_AGGREGATION_STOCKS:
        return None, [sym for sym, _ in panels]

    common_index = panels[0][1].index
    for _, frame in panels[1:]:
        common_index = common_index.intersection(frame.index)
    common_index = common_index.sort_values()
    if len(common_index) < _MIN_ROWS:
        return None, [sym for sym, _ in panels]

    close_panel = pd.concat(
        [frame["Close"].reindex(common_index).rename(symbol) for symbol, frame in panels],
        axis=1,
    ).dropna(axis=1, how="any")
    if close_panel.shape[1] < _MIN_AGGREGATION_STOCKS or len(close_panel) < _MIN_ROWS:
        return None, [sym for sym, _ in panels]

    aligned_symbols = list(close_panel.columns)
    open_panel = pd.concat(
        [frame["Open"].reindex(common_index).rename(symbol) for symbol, frame in panels if symbol in aligned_symbols],
        axis=1,
    )[aligned_symbols]
    high_panel = pd.concat(
        [frame["High"].reindex(common_index).rename(symbol) for symbol, frame in panels if symbol in aligned_symbols],
        axis=1,
    )[aligned_symbols]
    low_panel = pd.concat(
        [frame["Low"].reindex(common_index).rename(symbol) for symbol, frame in panels if symbol in aligned_symbols],
        axis=1,
    )[aligned_symbols]
    volume_panel = pd.concat(
        [frame["Volume"].reindex(common_index).rename(symbol) for symbol, frame in panels if symbol in aligned_symbols],
        axis=1,
    )[aligned_symbols]

    turnover = (close_panel.tail(min(_WEIGHT_LOOKBACK, len(close_panel))) * volume_panel.tail(min(_WEIGHT_LOOKBACK, len(volume_panel)))).mean(axis=0)
    turnover = pd.to_numeric(turnover, errors="coerce").clip(lower=0).fillna(0.0)
    if turnover.sum() <= 0:
        turnover = pd.Series(1.0, index=aligned_symbols)

    if len(aligned_symbols) > max_components:
        keep = turnover.sort_values(ascending=False).head(max_components).index.tolist()
        close_panel = close_panel[keep]
        open_panel = open_panel[keep]
        high_panel = high_panel[keep]
        low_panel = low_panel[keep]
        volume_panel = volume_panel[keep]
        turnover = turnover[keep]
        aligned_symbols = keep

    weights = turnover / max(turnover.sum(), 1e-9)
    base_close = close_panel.iloc[0].replace(0, np.nan)
    valid_cols = [col for col in aligned_symbols if pd.notna(base_close.get(col))]
    if len(valid_cols) < _MIN_AGGREGATION_STOCKS:
        return None, aligned_symbols

    close_panel = close_panel[valid_cols]
    open_panel = open_panel[valid_cols]
    high_panel = high_panel[valid_cols]
    low_panel = low_panel[valid_cols]
    volume_panel = volume_panel[valid_cols]
    weights = weights[valid_cols]
    weights = weights / max(weights.sum(), 1e-9)

    agg = build_weighted_synthetic_ohlc(
        open_panel=open_panel,
        high_panel=high_panel,
        low_panel=low_panel,
        close_panel=close_panel,
        volume_panel=volume_panel,
        weights=weights,
        base_close=base_close[valid_cols],
    )
    agg = _normalize_ohlc_frame(agg)
    if agg is None or len(agg) < _MIN_ROWS:
        return None, valid_cols
    return agg.tail(_TARGET_ROWS).copy(), valid_cols


def _build_sector_ohlc(
    sector_name: str,
    tickers: list[str],
    scan_df: pd.DataFrame | None,
    all_data: dict,
    force_refresh: bool = False,
    max_components: int = _MAX_AGGREGATION_STOCKS,
) -> tuple[pd.DataFrame | None, list[str], str, str, str]:
    sector_key = _sector_key(sector_name)
    ranked_tickers = _rank_sector_tickers(tickers, scan_df)

    if not force_refresh and get_current_window() != "LIVE":
        cached_sector_df, _ = feature_manager.load_sector_ohlc_cache(sector_name)
        cached_sector_df = _normalize_ohlc_frame(cached_sector_df)
        if cached_sector_df is not None and len(cached_sector_df) >= _MIN_ROWS:
            return cached_sector_df.tail(_TARGET_ROWS).copy(), ranked_tickers, "feature_sector_cache", sector_key, ""

    for symbol in _SECTOR_INDEX_CANDIDATES.get(sector_key, []):
        index_frame = _fetch_sector_index_frame(symbol, all_data, force_refresh=force_refresh)
        if index_frame is not None and len(index_frame) >= _MIN_ROWS:
            return index_frame, [symbol], "real_sector_index", symbol, ""

    basket_frame, used = _aggregate_weighted_sector_ohlc(
        ranked_tickers,
        all_data,
        force_refresh=force_refresh,
        max_components=max_components,
    )
    if basket_frame is not None and len(basket_frame) >= _MIN_ROWS:
        return basket_frame, used, "weighted_sector_basket", used[0] if used else "", ""

    for symbol in ranked_tickers:
        leader_frame = _fetch_stock_frame(symbol, all_data, force_refresh=force_refresh)
        if leader_frame is not None and len(leader_frame) >= _MIN_ROWS:
            return leader_frame.tail(_TARGET_ROWS).copy(), [symbol], "leader_stock_fallback", symbol, ""

    if used:
        return None, used, "", "", (
            f"Only {len(used)} constituents had usable OHLC for {sector_name}. "
            f"Need at least {_MIN_ROWS} daily candles and {_MIN_AGGREGATION_STOCKS}+ stocks "
            "to draw the sector chart safely."
        )
    if ranked_tickers:
        return None, [], "", "", (
            f"No valid daily OHLC series were available for {sector_name}. "
            f"Need at least {_MIN_ROWS} daily candles to draw the sector chart safely."
        )
    return None, [], "", "", f"No stocks or index mapping found for {sector_name}."


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
    if streak <= 1:
        return 50.0
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
    t   = ohlc.tail(lb)
    hl  = (t["High"] - t["Low"]).replace(0, np.nan)
    typ = (t["High"] + t["Low"] + t["Close"]) / 3.0
    dv  = (t["Volume"] * typ).sum()
    if dv <= 0:
        return 50.0
    net = (t["Volume"] * typ * (t["Close"] - t["Open"]) / (hl + 1e-9)).sum()
    ratio = net / (dv + 1e-9)
    return _norm(float(ratio), -0.15, 0.15)


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


def _stable_json_hash(payload: object) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        encoded = str(payload)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _frame_signature(df: pd.DataFrame | None) -> tuple:
    try:
        if df is None or df.empty:
            return (0, "", 0.0, 0.0)
        close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        volume = pd.to_numeric(df.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
        return (
            int(len(df)),
            str(pd.to_datetime(df.index[-1])),
            round(float(close.iloc[-1]), 6) if not close.empty else 0.0,
            round(float(volume.iloc[-1]), 2) if not volume.empty else 0.0,
        )
    except Exception:
        return (0, "", 0.0, 0.0)


def _scan_input_signature(scan_df: pd.DataFrame | None) -> str:
    if scan_df is None or not isinstance(scan_df, pd.DataFrame) or scan_df.empty:
        return "empty"
    cols = [
        col for col in ("Symbol", "Ticker", "Final Score", "Prediction Score", "Signal", "Action")
        if col in scan_df.columns
    ]
    if not cols:
        return f"rows:{len(scan_df)}"
    sample = scan_df[cols].copy()
    symbol_col = "Symbol" if "Symbol" in sample.columns else ("Ticker" if "Ticker" in sample.columns else "")
    if symbol_col:
        sample[symbol_col] = sample[symbol_col].astype(str).str.upper().str.replace(".NS", "", regex=False)
        sample = sample.sort_values(symbol_col, kind="stable")
    return _stable_json_hash(sample.fillna("").astype(str).to_dict("records"))


def _all_data_input_signature(all_data: dict, symbols: list[str]) -> str:
    items: list[tuple[str, tuple]] = []
    for symbol in symbols[:_MAX_AGGREGATION_STOCKS]:
        plain = _plain_symbol(symbol)
        frame = None
        for key in (plain, f"{plain}.NS", symbol):
            try:
                candidate = all_data.get(key)
            except Exception:
                candidate = None
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                frame = candidate
                break
        items.append((plain, _frame_signature(frame)))
    return _stable_json_hash(items)


def _sector_cache_metadata(
    sector_name: str,
    scan_df: pd.DataFrame | None,
    all_data: dict,
    stocks: list[str],
    regime: object,
    weights: dict[str, float],
) -> dict[str, object]:
    try:
        market_date = feature_manager._cache_day().isoformat()
    except Exception:
        market_date = ""
    return {
        "market_date": market_date,
        "scan_input_signature": _scan_input_signature(scan_df),
        "stock_universe_signature": _stable_json_hash([_plain_symbol(s) for s in stocks]),
        "all_data_signature": _all_data_input_signature(all_data, stocks),
        "regime_key": str(regime or ""),
        "dynamic_weight_signature": _stable_json_hash({k: round(float(v), 6) for k, v in weights.items()}),
    }


def _cache_metadata_matches(payload: dict | None, expected: dict[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    meta = payload.get("cache_metadata")
    if not isinstance(meta, dict):
        return False
    for key, expected_value in expected.items():
        if str(meta.get(key, "")) != str(expected_value):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def predict_sector(
    sector_name:   str,
    scan_df:       pd.DataFrame | None,
    all_data:      dict,
    regime_state=None,
    force_refresh: bool = False,
) -> SectorPrediction:

    now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    try:
        from sector_master import get_stocks_in_sector
        stocks = get_stocks_in_sector(sector_name)
    except Exception:
        stocks = []

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

    expected_cache_meta = _sector_cache_metadata(sector_name, scan_df, all_data, stocks, regime, weights)
    if not force_refresh and get_current_window() != "LIVE":
        try:
            cached_payload = feature_manager.load_prediction_cache(sector_name)
            if _cache_metadata_matches(cached_payload, expected_cache_meta):
                cached_ohlc, _ = feature_manager.load_sector_ohlc_cache(sector_name)
                return _prediction_from_payload(cached_payload, _normalize_ohlc_frame(cached_ohlc))
        except Exception:
            pass

    ohlc, used, ohlc_source, ohlc_symbol, ohlc_note = _build_sector_ohlc(
        sector_name,
        stocks,
        scan_df,
        all_data,
        force_refresh=force_refresh,
    )
    if ohlc is None or len(ohlc) < _MIN_ROWS:
        return SectorPrediction(
            sector=sector_name,
            direction="Sideways",
            confidence=50.0,
            raw_score=50.0,
            stocks_used=used,
            predicted_at=now_ts,
            note=ohlc_note or "Insufficient OHLC data.",
            ohlc_source=ohlc_source,
            ohlc_symbol=ohlc_symbol,
            ohlc_bars=0 if ohlc is None else len(ohlc),
        )
    stock_universe = stocks or used
    try:
        signals = _compute_signals(ohlc, scan_df, stock_universe)
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

    prediction = SectorPrediction(
        sector            = sector_name,
        direction         = direction,
        confidence        = round(confidence, 1),
        raw_score         = round(raw_score, 1),
        signals           = signals,
        ohlc_df           = ohlc,
        leader_ticker     = ohlc_symbol or (used[0] if used else ""),
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
        ohlc_source       = ohlc_source,
        ohlc_symbol       = ohlc_symbol,
        ohlc_bars         = len(ohlc),
    )
    try:
        feature_manager.save_sector_ohlc_cache(sector_name, ohlc, top_n=len(used) if used else len(stocks))
        payload = _prediction_to_payload(prediction)
        payload["cache_metadata"] = {
            **expected_cache_meta,
            "ohlc_source": ohlc_source,
            "ohlc_bars": len(ohlc),
            "stocks_used_signature": _stable_json_hash([_plain_symbol(item) for item in used]),
        }
        feature_manager.save_prediction_cache(sector_name, payload)
    except Exception:
        pass
    return prediction
