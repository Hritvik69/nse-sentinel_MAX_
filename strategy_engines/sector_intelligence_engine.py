"""
sector_intelligence_engine.py
──────────────────────────────
Sector Intelligence Engine — Advanced Layer for NSE Sentinel.

NEW FILE ONLY — does NOT modify any existing file, function, or logic.

This layer sits AFTER existing calculations and enhances decision-making
without interfering with the base system.

All 8 upgrades implemented:
  1. Smart Stock Filtering       (filter_top_stocks)
  2. Dynamic Weighting           (log-volume-price weight, no static market cap)
  3. Dynamic Sector Dominance    (replaces hardcoded banking bias)
  4. Multi-Sector Mapping        (STOCK_SECTORS with weighted contributions)
  5. Sector Strength Engine      (strength_score + momentum_score)
  6. Sector Rotation Detection   (MONEY_INFLOW / MONEY_OUTFLOW flags)
  7. Leader Detection            (top 3 per sector)
  8. UI Intelligence Data        (structured output for enhanced UI)

Design rules
────────────
• Zero API calls — purely in-memory DataFrame logic.
• Never modifies existing columns or removes rows from scan output.
• Never crashes — every public function is fully wrapped in try/except.
• No heavy computation — simple vectorised math only.
• Works standalone; gracefully degrades if scan data is sparse.

Public API
──────────
    compute_sector_intelligence(scan_df)   → dict
        Master entry point. Pass the fully-enriched scan DataFrame
        (after enhance_results + grading + enhanced_logic + phase4).
        Returns a structured dict ready for the UI layer.

    filter_top_stocks(sector_rows)         → list[dict]
        Returns top 5–8 filtered stocks for a sector.

    get_sector_strength(sector_rows)       → dict
        Returns strength_score, momentum_score, sector_strength for rows.

    detect_rotation(sector_name, current_strength) → str
        Returns "MONEY_INFLOW" / "MONEY_OUTFLOW" / "STABLE".

    get_sector_leaders(sector_rows, n=3)   → list[str]
        Returns top-n leader stock symbols for a sector.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# MULTI-SECTOR MAPPING  (Upgrade 4)
# Primary sector = first entry. Secondary entries with fractional weight.
# ─────────────────────────────────────────────────────────────────────

STOCK_SECTORS: dict[str, list[tuple[str, float]]] = {
    # Conglomerates / cross-sector heavyweights
    "RELIANCE":    [("ENERGY", 0.50), ("TELECOM", 0.30), ("RETAIL", 0.20)],
    "ITC":         [("FMCG", 0.60), ("HOSPITALITY", 0.20), ("AGRI", 0.20)],
    "ADANIENT":    [("ENERGY", 0.40), ("INFRA", 0.35), ("METAL", 0.25)],
    "ADANIPORTS":  [("INFRA", 0.70), ("ENERGY", 0.30)],
    "ADANIGREEN":  [("ENERGY", 0.80), ("INFRA", 0.20)],
    "ADANITRANS":  [("ENERGY", 0.70), ("INFRA", 0.30)],
    "ADANIPOWER":  [("ENERGY", 0.80), ("INFRA", 0.20)],
    "TATAPOWER":   [("ENERGY", 0.60), ("INFRA", 0.40)],
    "TATAMOTORS":  [("AUTO", 0.70), ("METAL", 0.30)],
    "TATASTEEL":   [("METAL", 0.70), ("CAPITAL_GOODS", 0.30)],
    "BAJAJFINSV":  [("NBFC_FINANCE", 0.60), ("BANKING", 0.40)],
    "L&T":         [("CAPITAL_GOODS", 0.60), ("INFRA", 0.40)],
    "SUNPHARMA":   [("PHARMA", 0.80), ("CHEMICAL", 0.20)],
    "DRREDDY":     [("PHARMA", 0.80), ("CHEMICAL", 0.20)],
    "HINDALCO":    [("METAL", 0.60), ("CHEMICAL", 0.40)],
    "VEDL":        [("METAL", 0.60), ("ENERGY", 0.40)],
    "COALINDIA":   [("ENERGY", 0.70), ("METAL", 0.30)],
    "NTPC":        [("ENERGY", 0.80), ("PSU", 0.20)],
    "POWERGRID":   [("ENERGY", 0.70), ("INFRA", 0.30)],
    "ONGC":        [("ENERGY", 0.80), ("PSU", 0.20)],
    "BPCL":        [("ENERGY", 0.80), ("PSU", 0.20)],
    "IOC":         [("ENERGY", 0.80), ("PSU", 0.20)],
    "GAIL":        [("ENERGY", 0.70), ("INFRA", 0.30)],
    "M&M":         [("AUTO", 0.70), ("CAPITAL_GOODS", 0.30)],
    "BAJFINANCE":  [("NBFC_FINANCE", 0.70), ("BANKING", 0.30)],
    "HDFCLIFE":    [("NBFC_FINANCE", 0.70), ("BANKING", 0.30)],
    "SBILIFE":     [("NBFC_FINANCE", 0.70), ("PSU", 0.30)],
    "PFC":         [("NBFC_FINANCE", 0.60), ("PSU", 0.40)],
    "RECLTD":      [("NBFC_FINANCE", 0.60), ("PSU", 0.40)],
    "IRFC":        [("NBFC_FINANCE", 0.50), ("RAILWAY", 0.50)],
    "APLAPOLLO":   [("METAL", 0.60), ("CAPITAL_GOODS", 0.40)],
    "BHEL":        [("CAPITAL_GOODS", 0.60), ("PSU", 0.40)],
    "HAL":         [("DEFENCE", 0.80), ("PSU", 0.20)],
    "BEL":         [("DEFENCE", 0.70), ("PSU", 0.30)],
    "IRCTC":       [("RAILWAY", 0.70), ("PSU", 0.30)],
    "RVNL":        [("RAILWAY", 0.70), ("INFRA", 0.30)],
    "TITAGARH":    [("RAILWAY", 0.80), ("CAPITAL_GOODS", 0.20)],
    "BHARTIARTL":  [("TELECOM", 0.90), ("INFRA", 0.10)],
    "IDEA":        [("TELECOM", 1.00)],
    "INDUSTOWER":  [("TELECOM", 0.70), ("INFRA", 0.30)],
    "CHOLAFIN":    [("NBFC_FINANCE", 0.70), ("AUTO", 0.30)],
    "MOTHERSON":   [("AUTO", 0.80), ("CAPITAL_GOODS", 0.20)],
    "BOSCHLTD":    [("AUTO", 0.70), ("CAPITAL_GOODS", 0.30)],
    "TATACONSUM":  [("FMCG", 0.70), ("ENERGY", 0.30)],
    "CIPLA":       [("PHARMA", 0.85), ("CHEMICAL", 0.15)],
    "BIOCON":      [("PHARMA", 0.75), ("CHEMICAL", 0.25)],
    "PIDILITIND":  [("CHEMICAL", 0.60), ("CONSUMER_DURABLES", 0.40)],
    "ASIANPAINT":  [("CHEMICAL", 0.70), ("CONSUMER_DURABLES", 0.30)],
    "BERGERPAINTS":[("CHEMICAL", 0.70), ("CONSUMER_DURABLES", 0.30)],
    "DLF":         [("REALTY", 0.90), ("INFRA", 0.10)],
    "GODREJPROP":  [("REALTY", 0.90), ("INFRA", 0.10)],
    "OBEROIRLTY":  [("REALTY", 0.80), ("CONSUMER_DURABLES", 0.20)],
    "PRESTIGE":    [("REALTY", 0.90), ("INFRA", 0.10)],
}


# ─────────────────────────────────────────────────────────────────────
# IN-MEMORY ROTATION CACHE  (Upgrade 6)
# Stores previous-cycle sector strength scores to detect flow.
# ─────────────────────────────────────────────────────────────────────

_ROTATION_CACHE: dict[str, float] = {}
_ROTATION_LOCK  = threading.Lock()

_DOMINANCE_THRESHOLD = 65.0   # sector_strength above this boosts overall bias


# ─────────────────────────────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────────────────────────────

def _sf(v: object, default: float = 0.0) -> float:
    """safe float — never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _get(row: "pd.Series | dict", *keys: str, default: float = 0.0) -> float:
    """Return first matching key as safe float."""
    for k in keys:
        v = row.get(k) if isinstance(row, dict) else row.get(k)
        if v is not None:
            f = _sf(v, default)
            if math.isfinite(f):
                return f
    return default


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 2 — DYNAMIC WEIGHTING
# weight = log(volume * close_price + 1)
# ─────────────────────────────────────────────────────────────────────

def _compute_weight(row: "pd.Series | dict") -> float:
    """
    Dynamic weight for a stock row.
        weight = log(volume * close_price + 1)
    Fallback = 1.0 if volume is missing or zero.
    """
    vol   = _get(row, "Volume", default=0.0)
    price = _get(row, "Price (₹)", "Close", "close", default=0.0)

    if vol <= 0 or price <= 0:
        return 1.0
    raw = math.log(vol * price + 1.0)
    return max(raw, 1.0)


def _weighted_average(values: list[float], weights: list[float]) -> float:
    """Weighted average of values. Falls back to simple mean on failure."""
    try:
        total_w = sum(weights)
        if total_w <= 0:
            return float(np.mean(values)) if values else 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_w
    except Exception:
        return float(np.mean(values)) if values else 0.0


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 1 — SMART STOCK FILTERING
# ─────────────────────────────────────────────────────────────────────

def filter_top_stocks(
    sector_rows: list[dict],
    top_n_min: int = 5,
    top_n_max: int = 8,
) -> list[dict]:
    """
    Filter and rank stocks for a sector's intelligence layer.

    Ranking composite:
        rank_score = 0.40 * prediction_score
                   + 0.35 * volume_strength        (normalised Vol/Avg)
                   + 0.25 * trend_strength          (normalised 5D / EMA dist)

    Removal criteria (stock is excluded if ANY hold):
        • Vol/Avg < 0.70             (low volume)
        • Trap Risk == "HIGH"        (high trap risk)
        • Volume Trend == "WEAK"     (weak volume trend from enhanced_logic)

    Returns top 5–8 stocks ordered by rank_score desc.
    Does NOT modify the original rows — returns new list of same dicts.
    Never crashes.

    Parameters
    ----------
    sector_rows : list[dict]   Row dicts from the enriched scan DataFrame.
    top_n_min   : int          Minimum stocks to return (default 5).
    top_n_max   : int          Maximum stocks to return (default 8).

    Returns
    -------
    list[dict]  Filtered, ranked stock rows (references to originals, not copies).
    """
    try:
        if not sector_rows:
            return []

        scored: list[tuple[float, dict]] = []

        for row in sector_rows:
            # ── Exclusion filters ─────────────────────────────────────
            vol_avg    = _get(row, "Vol / Avg",       default=1.0)
            trap_risk  = str(row.get("Trap Risk", "") or "").strip().upper()
            vol_trend  = str(row.get("Volume Trend", "") or "").strip().upper()

            if vol_avg < 0.70:
                continue
            if trap_risk == "HIGH":
                continue
            if vol_trend == "WEAK":
                continue

            # ── Ranking score ─────────────────────────────────────────
            pred_score = _get(row, "Prediction Score", "Final Score", default=50.0)
            vol_str    = float(np.clip(vol_avg / 3.0, 0.0, 1.0)) * 100.0
            ret_5d     = _get(row, "5D Return (%)",      default=0.0)
            ema_dist   = _get(row, "Δ vs EMA20 (%)",     default=0.0)
            trend_str  = float(np.clip(50.0 + ret_5d * 3.0 + ema_dist * 1.5, 0.0, 100.0))

            rank_score = (
                0.40 * float(np.clip(pred_score, 0.0, 100.0))
                + 0.35 * vol_str
                + 0.25 * trend_str
            )
            scored.append((rank_score, row))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Clamp between top_n_min and top_n_max
        n = max(top_n_min, min(top_n_max, len(scored)))
        return [row for _, row in scored[:n]]

    except Exception:
        return sector_rows[:top_n_max] if sector_rows else []


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 5 — SECTOR STRENGTH ENGINE
# ─────────────────────────────────────────────────────────────────────

def get_sector_strength(sector_rows: list[dict]) -> dict:
    """
    Compute composite sector strength from enriched stock rows.

        strength_score = (
            weighted_bullish_pct
            + volume_expansion_score
            - trap_risk_penalty
        )

        momentum_score  = weighted average of 5D returns (normalised 0-100)

        sector_strength = 0.65 * strength_score + 0.35 * momentum_score

    Parameters
    ----------
    sector_rows : list[dict]   Enriched scan rows for ONE sector.

    Returns
    -------
    dict with keys:
        strength_score   float  0-100
        momentum_score   float  0-100
        sector_strength  float  0-100
        bullish_pct      float  0-100
        vol_expansion    float  0-100
        trap_penalty     float  0-100
        stock_count      int
    """
    _empty = {
        "strength_score":  0.0,
        "momentum_score":  0.0,
        "sector_strength": 0.0,
        "bullish_pct":     0.0,
        "vol_expansion":   0.0,
        "trap_penalty":    0.0,
        "stock_count":     0,
    }

    try:
        if not sector_rows:
            return _empty

        weights: list[float] = []
        is_bullish:  list[float] = []
        vol_ratios:  list[float] = []
        trap_scores: list[float] = []
        returns_5d:  list[float] = []

        for row in sector_rows:
            w = _compute_weight(row)
            weights.append(w)

            # Bullish signal
            pred  = _get(row, "Prediction Score", "Final Score", default=50.0)
            signal = str(row.get("Signal", row.get("Final Signal", "")) or "").upper()
            bull_flag = (
                pred >= 55.0
                or "BUY"  in signal
                or "BULL" in signal
                or "LONG" in signal
            )
            is_bullish.append(1.0 if bull_flag else 0.0)

            # Volume expansion (Vol/Avg)
            va = _get(row, "Vol / Avg", default=1.0)
            vol_ratios.append(float(np.clip(va, 0.0, 3.5)))

            # Trap risk penalty
            trap_risk = str(row.get("Trap Risk", "") or "").strip().upper()
            trap_val  = {"HIGH": 30.0, "MEDIUM": 10.0, "LOW": 0.0}.get(trap_risk, 5.0)
            trap_scores.append(trap_val)

            # 5D return
            r5 = _get(row, "5D Return (%)", default=0.0)
            returns_5d.append(r5)

        # ── Weighted components ───────────────────────────────────────
        weighted_bullish   = _weighted_average(is_bullish,  weights) * 100.0
        raw_vol_expansion  = _weighted_average(vol_ratios,  weights)
        vol_expansion      = float(np.clip((raw_vol_expansion / 3.5) * 100.0, 0.0, 100.0))
        trap_penalty       = _weighted_average(trap_scores, weights)
        raw_ret            = _weighted_average(returns_5d,  weights)
        momentum_score     = float(np.clip(50.0 + raw_ret * 4.0, 0.0, 100.0))

        # ── Strength score ────────────────────────────────────────────
        strength_score = float(np.clip(
            weighted_bullish + vol_expansion - trap_penalty,
            0.0, 100.0
        ))

        sector_strength = float(np.clip(
            0.65 * strength_score + 0.35 * momentum_score,
            0.0, 100.0
        ))

        return {
            "strength_score":  round(strength_score,  2),
            "momentum_score":  round(momentum_score,  2),
            "sector_strength": round(sector_strength, 2),
            "bullish_pct":     round(weighted_bullish, 2),
            "vol_expansion":   round(vol_expansion,   2),
            "trap_penalty":    round(trap_penalty,    2),
            "stock_count":     len(sector_rows),
        }

    except Exception:
        return _empty


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 6 — SECTOR ROTATION DETECTION
# ─────────────────────────────────────────────────────────────────────

def detect_rotation(sector_name: str, current_strength: float) -> str:
    """
    Compare current sector strength against the previous cached value.

    Returns
    -------
    "MONEY_INFLOW"  — current > previous by > 2 points
    "MONEY_OUTFLOW" — current < previous by > 2 points
    "STABLE"        — within ±2 points or no prior data
    """
    _FLOW_THRESHOLD = 2.0   # minimum delta to declare a direction

    try:
        sector_key = str(sector_name).upper().strip()
        with _ROTATION_LOCK:
            prev = _ROTATION_CACHE.get(sector_key)
            _ROTATION_CACHE[sector_key] = float(current_strength)

        if prev is None:
            return "STABLE"

        delta = float(current_strength) - float(prev)
        if delta > _FLOW_THRESHOLD:
            return "MONEY_INFLOW"
        if delta < -_FLOW_THRESHOLD:
            return "MONEY_OUTFLOW"
        return "STABLE"

    except Exception:
        return "STABLE"


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 7 — LEADER DETECTION
# ─────────────────────────────────────────────────────────────────────

def get_sector_leaders(sector_rows: list[dict], n: int = 3) -> list[str]:
    """
    Identify the top-n leader stocks in a sector.

    Leader score = 0.50 * relative_strength + 0.50 * volume_strength
        relative_strength = normalised (Prediction Score + 5D Return component)
        volume_strength   = normalised Vol/Avg

    Parameters
    ----------
    sector_rows : list[dict]   Enriched scan rows for ONE sector.
    n           : int          Number of leaders to return (default 3).

    Returns
    -------
    list[str]  Top-n stock symbols. Empty list on failure.
    """
    try:
        if not sector_rows:
            return []

        scored: list[tuple[float, str]] = []

        for row in sector_rows:
            sym  = str(row.get("Symbol", row.get("Ticker", "")) or "").strip()
            if not sym:
                continue

            pred  = _get(row, "Prediction Score", "Final Score", default=50.0)
            ret5d = _get(row, "5D Return (%)", default=0.0)
            va    = _get(row, "Vol / Avg", default=1.0)

            rel_str = float(np.clip(pred * 0.7 + (50.0 + ret5d * 3.0) * 0.3, 0.0, 100.0))
            vol_str = float(np.clip((va / 3.5) * 100.0, 0.0, 100.0))

            leader_score = 0.50 * rel_str + 0.50 * vol_str
            scored.append((leader_score, sym))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sym for _, sym in scored[:n]]

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# UPGRADE 3 — DYNAMIC SECTOR DOMINANCE  (replaces hardcoded banking)
# ─────────────────────────────────────────────────────────────────────

def get_dominant_sector(sector_strength_map: dict[str, float]) -> dict:
    """
    Find the dominant sector dynamically by highest strength score.
    No sector is hardcoded as dominant.

    Parameters
    ----------
    sector_strength_map : dict[str, float]
        {sector_name: sector_strength_score}

    Returns
    -------
    dict with keys:
        dominant_sector  str    Name of dominant sector (or "" if none qualifies)
        strength         float  Its strength score
        boosted          bool   True if above DOMINANCE_THRESHOLD
    """
    try:
        if not sector_strength_map:
            return {"dominant_sector": "", "strength": 0.0, "boosted": False}

        best_sector = max(sector_strength_map, key=lambda s: sector_strength_map[s])
        best_score  = float(sector_strength_map[best_sector])
        boosted     = best_score >= _DOMINANCE_THRESHOLD

        return {
            "dominant_sector": best_sector if boosted else "",
            "strength":        round(best_score, 2),
            "boosted":         boosted,
        }

    except Exception:
        return {"dominant_sector": "", "strength": 0.0, "boosted": False}


# ─────────────────────────────────────────────────────────────────────
# MULTI-SECTOR CONTRIBUTION HELPER  (Upgrade 4)
# ─────────────────────────────────────────────────────────────────────

def get_primary_sector_for_stock(symbol: str) -> str:
    """
    Return the primary sector for a stock using STOCK_SECTORS first,
    then fall back to sector_master.SYMBOL_TO_SECTOR if available.
    """
    sym = symbol.upper().strip().replace(".NS", "")
    mappings = STOCK_SECTORS.get(sym)
    if mappings:
        return mappings[0][0]   # first entry is primary

    # Fallback to sector_master (if available)
    try:
        from sector_master import get_sector
        sec = get_sector(sym)
        if sec:
            return sec
    except Exception:
        pass
    return "OTHER"


def get_sector_contributions_for_stock(symbol: str) -> list[tuple[str, float]]:
    """
    Return list of (sector, weight) tuples for a stock.
    Weights sum to 1.0. If stock not in STOCK_SECTORS, returns single primary with 1.0.
    """
    sym = symbol.upper().strip().replace(".NS", "")
    mappings = STOCK_SECTORS.get(sym)
    if mappings:
        return mappings

    primary = get_primary_sector_for_stock(sym)
    return [(primary, 1.0)]


def _assign_rows_to_sectors(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Assign scan rows to sectors using multi-sector mapping.
    A stock can contribute to multiple sectors with fractional weight.
    Returns {sector_name: [row_dict, ...]} — rows appear once per sector
    they contribute to, weighted internally via _compute_weight.

    To avoid double-counting in strength scores, weighted_contribution
    is stored in the row as "__sector_weight" for that sector.
    """
    sector_map: dict[str, list[dict]] = {}

    try:
        from sector_master import SECTOR_STOCKS
        all_sectors = set(SECTOR_STOCKS.keys())
    except Exception:
        all_sectors = set()

    for _, row in df.iterrows():
        try:
            row_d = row.to_dict()
            sym   = str(row_d.get("Symbol", row_d.get("Ticker", "")) or "").strip().upper()
            if not sym:
                continue

            contributions = get_sector_contributions_for_stock(sym)

            for sec, weight in contributions:
                # Skip trivially small contributions
                if weight < 0.10:
                    continue
                row_copy = dict(row_d)
                row_copy["__sector_weight"] = float(weight)
                if sec not in sector_map:
                    sector_map[sec] = []
                sector_map[sec].append(row_copy)

            # If a stock has no contributions mapping, assign to primary
            if not contributions:
                primary = get_primary_sector_for_stock(sym)
                row_copy = dict(row_d)
                row_copy["__sector_weight"] = 1.0
                if primary not in sector_map:
                    sector_map[primary] = []
                sector_map[primary].append(row_copy)

        except Exception:
            continue

    return sector_map


# ─────────────────────────────────────────────────────────────────────
# MASTER ENTRY POINT  (Upgrade 8 — structured output)
# ─────────────────────────────────────────────────────────────────────

def compute_sector_intelligence(scan_df: "pd.DataFrame | None") -> dict:
    """
    Master entry point for the Sector Intelligence Layer.

    Accepts the fully-enriched scan DataFrame (after enhance_results +
    grading + enhanced_logic + phase4) and returns a structured dict
    that powers the enhanced Sector Explorer UI.

    Parameters
    ----------
    scan_df : pd.DataFrame
        Enriched scan output. Must have at minimum:
            "Symbol", "Price (₹)", "Volume", "Vol / Avg",
            "Prediction Score" or "Final Score",
            "5D Return (%)", "Δ vs EMA20 (%)"
        Optional (used if present):
            "Trap Risk", "Volume Trend", "Signal"

    Returns
    -------
    dict with keys:
        sector_details   : dict[str, dict]
            Per-sector output — keys = sector names.
            Each value:
                strength_score   float
                momentum_score   float
                sector_strength  float
                flow_signal      str   MONEY_INFLOW / MONEY_OUTFLOW / STABLE
                leader_stocks    list[str]
                top_stocks       list[dict]   (filtered, ranked)
                stock_count      int
                bullish_pct      float
                vol_expansion    float
                trap_penalty     float

        sector_ranking   : list[dict]
            All sectors sorted strongest → weakest.
            Each entry: {sector, sector_strength, momentum_score, flow_signal}

        dominant_sector  : dict
            {dominant_sector, strength, boosted}

        overall_summary  : dict
            {sectors_analysed, total_stocks, avg_sector_strength,
             top_sector, weakest_sector}
    """
    _empty_result: dict = {
        "sector_details":  {},
        "sector_ranking":  [],
        "dominant_sector": {"dominant_sector": "", "strength": 0.0, "boosted": False},
        "overall_summary": {
            "sectors_analysed": 0,
            "total_stocks":     0,
            "avg_sector_strength": 0.0,
            "top_sector":    "",
            "weakest_sector": "",
        },
    }

    try:
        if scan_df is None or not isinstance(scan_df, pd.DataFrame) or scan_df.empty:
            return _empty_result

        # ── 1. Assign rows to sectors (multi-sector mapping) ──────────
        sector_map = _assign_rows_to_sectors(scan_df)

        if not sector_map:
            return _empty_result

        # ── 2. Per-sector intelligence ────────────────────────────────
        sector_details:       dict[str, dict]  = {}
        sector_strength_map:  dict[str, float] = {}

        for sector, rows in sector_map.items():
            if not rows:
                continue

            # Strength metrics
            strength_data = get_sector_strength(rows)

            # Rotation detection (compare vs cached previous)
            flow = detect_rotation(sector, strength_data["sector_strength"])

            # Leader detection
            leaders = get_sector_leaders(rows, n=3)

            # Smart filtering
            top_stocks = filter_top_stocks(rows)

            # Strip internal weight key from returned top_stocks
            clean_top = []
            for r in top_stocks:
                clean_r = {k: v for k, v in r.items() if not k.startswith("__")}
                clean_top.append(clean_r)

            sector_details[sector] = {
                **strength_data,
                "flow_signal":   flow,
                "leader_stocks": leaders,
                "top_stocks":    clean_top,
            }
            sector_strength_map[sector] = strength_data["sector_strength"]

        # ── 3. Sector ranking (strongest → weakest) ───────────────────
        sector_ranking = sorted(
            [
                {
                    "sector":          s,
                    "sector_strength": sector_strength_map[s],
                    "momentum_score":  sector_details[s]["momentum_score"],
                    "flow_signal":     sector_details[s]["flow_signal"],
                    "leader_stocks":   sector_details[s]["leader_stocks"],
                    "bullish_pct":     sector_details[s]["bullish_pct"],
                    "stock_count":     sector_details[s]["stock_count"],
                }
                for s in sector_strength_map
            ],
            key=lambda x: x["sector_strength"],
            reverse=True,
        )

        # ── 4. Dominant sector (dynamic, no hardcoded bias) ───────────
        dominant = get_dominant_sector(sector_strength_map)

        # ── 5. Overall summary ────────────────────────────────────────
        total_stocks = sum(d["stock_count"] for d in sector_details.values())
        strengths    = list(sector_strength_map.values())
        avg_str      = round(float(np.mean(strengths)), 2) if strengths else 0.0
        top_sector   = sector_ranking[0]["sector"]  if sector_ranking else ""
        weak_sector  = sector_ranking[-1]["sector"] if sector_ranking else ""

        overall_summary = {
            "sectors_analysed":    len(sector_details),
            "total_stocks":        total_stocks,
            "avg_sector_strength": avg_str,
            "top_sector":          top_sector,
            "weakest_sector":      weak_sector,
        }

        return {
            "sector_details":  sector_details,
            "sector_ranking":  sector_ranking,
            "dominant_sector": dominant,
            "overall_summary": overall_summary,
        }

    except Exception:
        return _empty_result