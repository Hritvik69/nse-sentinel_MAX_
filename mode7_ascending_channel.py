"""
mode7_ascending_channel.py
══════════════════════════════════════════════════════════════════════
Ascending channel detector for Mode 7.

Detects the pattern drawn by the user:
  • Two rising parallel trendlines (support and resistance)
  • Price makes higher highs and higher lows
  • Currently near the SUPPORT line = buyable zone
  • Hold 2–7 days for move toward resistance

Public API
──────────
  detect_ascending_channel(df)  → ChannelResult
  score_channel_entry(result)   → float  (0–100)
  apply_channel_filter(scan_df) → pd.DataFrame  (filtered, sorted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ChannelResult:
    detected:           bool  = False
    quality:            str   = "NONE"     # HIGH / MEDIUM / LOW / NONE
    entry_zone:         bool  = False      # True = price near support NOW
    support_slope:      float = 0.0        # rising = positive
    resistance_slope:   float = 0.0
    channel_width_pct:  float = 0.0        # resistance − support as % of price
    position_in_channel:float = 0.0        # 0 = at support, 1 = at resistance
    higher_lows:        int   = 0          # count of confirmed higher lows
    higher_highs:       int   = 0
    support_price:      float = 0.0        # current projected support
    resistance_price:   float = 0.0        # current projected resistance
    risk_reward:        float = 0.0        # reward (to resistance) / risk (1 ATR)
    atr_contraction:    bool  = False
    volume_declining_in_base: bool = False
    note:               str   = ""


# ─────────────────────────────────────────────────────────────────────
# TRENDLINE HELPERS
# ─────────────────────────────────────────────────────────────────────

def _pivot_lows(low: pd.Series, window: int = 3) -> pd.Series:
    """Return pivot lows — local minima confirmed on both sides."""
    shifted = {i: low.shift(i) for i in range(-window, window + 1) if i != 0}
    mask = pd.Series(True, index=low.index)
    for s in shifted.values():
        mask &= low <= s
    return low[mask]


def _pivot_highs(high: pd.Series, window: int = 3) -> pd.Series:
    shifted = {i: high.shift(i) for i in range(-window, window + 1) if i != 0}
    mask = pd.Series(True, index=high.index)
    for s in shifted.values():
        mask &= high >= s
    return high[mask]


def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Simple OLS slope + intercept."""
    if len(x) < 2:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    x = x.astype(float)
    y = y.astype(float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0, ym
    slope = ((x - xm) * (y - ym)).sum() / denom
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def _is_higher_sequence(values: np.ndarray, min_count: int = 2) -> tuple[bool, int]:
    """Return (is_higher, count_of_consecutive_highs/lows)."""
    if len(values) < min_count:
        return False, 0
    count = 0
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            count += 1
        else:
            break
    return count >= min_count - 1, count


# ─────────────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────────

def detect_ascending_channel(
    df: pd.DataFrame | None,
    *,
    lookback: int = 60,
    near_support_pct: float = 0.30,   # within bottom 30% of channel = entry zone
) -> ChannelResult:
    """
    Detect ascending channel from OHLCV data.

    Parameters
    ----------
    df              : OHLCV DataFrame (any length ≥ 30)
    lookback        : bars to analyse
    near_support_pct: if position_in_channel < this → entry zone

    Returns
    -------
    ChannelResult
    """
    result = ChannelResult()

    try:
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 30:
            result.note = "Insufficient data"
            return result

        work = df.copy()
        if isinstance(work.columns, pd.MultiIndex):
            work.columns = work.columns.get_level_values(0)
        for col in ("Open", "High", "Low", "Close"):
            if col not in work.columns:
                result.note = f"Missing column: {col}"
                return result

        work = work.dropna(subset=["High", "Low", "Close"]).tail(lookback)
        if len(work) < 20:
            result.note = "Too few rows after dropna"
            return result

        n = len(work)
        idx = np.arange(n, dtype=float)

        close  = pd.to_numeric(work["Close"],  errors="coerce").values
        high   = pd.to_numeric(work["High"],   errors="coerce").values
        low    = pd.to_numeric(work["Low"],    errors="coerce").values
        volume = pd.to_numeric(work.get("Volume", pd.Series(
            np.ones(n), index=work.index)), errors="coerce").fillna(1).values

        last_close = float(close[-1])
        if last_close <= 0:
            result.note = "Invalid close price"
            return result

        # ── ATR ───────────────────────────────────────────────────────
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low  - np.roll(close, 1)),
        ])
        tr[0] = high[0] - low[0]
        atr14 = pd.Series(tr).rolling(14, min_periods=7).mean().values
        atr_now   = float(atr14[-1]) if np.isfinite(atr14[-1]) else last_close * 0.02
        atr_base  = float(np.nanmean(atr14[-8:]))
        atr_prior = float(np.nanmean(atr14[-40:-16])) if len(atr14) >= 40 else atr_base
        atr_contraction = (atr_prior > 0) and (atr_base <= atr_prior * 0.88)

        # ── Pivot points ──────────────────────────────────────────────
        low_s  = pd.Series(low,  index=range(n))
        high_s = pd.Series(high, index=range(n))
        pl = _pivot_lows(low_s,  window=3).tail(6)
        ph = _pivot_highs(high_s, window=3).tail(6)

        if len(pl) < 2 or len(ph) < 2:
            result.note = "Not enough pivot points"
            return result

        # ── Support trendline (through pivot lows) ────────────────────
        pl_x = np.array(pl.index, dtype=float)
        pl_y = pl.values.astype(float)
        sup_slope, sup_int = _linreg_slope_intercept(pl_x, pl_y)

        # ── Resistance trendline (through pivot highs) ────────────────
        ph_x = np.array(ph.index, dtype=float)
        ph_y = ph.values.astype(float)
        res_slope, res_int = _linreg_slope_intercept(ph_x, ph_y)

        # ── Project to current bar ────────────────────────────────────
        cur_i = float(n - 1)
        sup_now = sup_slope * cur_i + sup_int
        res_now = res_slope * cur_i + res_int

        channel_width = res_now - sup_now
        if channel_width <= 0:
            result.note = "Inverted channel (resistance below support)"
            return result

        channel_width_pct = channel_width / last_close * 100
        position = (last_close - sup_now) / channel_width   # 0=at support, 1=at resistance

        # ── Higher lows and higher highs ─────────────────────────────
        _, hl_count = _is_higher_sequence(pl_y)
        _, hh_count = _is_higher_sequence(ph_y)

        # ── Volume declining in base ──────────────────────────────────
        vol_base  = float(np.mean(volume[-8:]))
        vol_prior = float(np.mean(volume[-30:-8])) if len(volume) >= 30 else vol_base
        vol_declining = vol_prior > 0 and vol_base < vol_prior * 0.90

        # ── Entry zone check ─────────────────────────────────────────
        entry_zone = (
            0.0 <= position <= near_support_pct          # near support
            and sup_slope > 0                             # support rising
            and res_slope > 0                             # resistance rising
            and last_close > sup_now - atr_now * 0.4     # not broken below support
        )

        # ── Risk / reward ─────────────────────────────────────────────
        reward = max(0.0, res_now - last_close)
        risk   = max(atr_now, last_close * 0.01)
        rr     = round(reward / risk, 2)

        # ── Quality rating ────────────────────────────────────────────
        detected = (
            sup_slope > 0
            and res_slope > 0
            and hl_count >= 1
            and hh_count >= 1
            and 3.0 <= channel_width_pct <= 20.0
        )

        if detected:
            score = 0
            score += min(hl_count, 3) * 20      # higher lows
            score += min(hh_count, 3) * 15      # higher highs
            score += 15 if atr_contraction else 0
            score += 10 if entry_zone else 0
            score += 10 if vol_declining else 0
            score += 10 if rr >= 1.5 else (5 if rr >= 1.0 else 0)
            score -= 10 if position > 0.70 else 0   # too near resistance

            if score >= 65:
                quality = "HIGH"
            elif score >= 40:
                quality = "MEDIUM"
            else:
                quality = "LOW"
        else:
            quality = "NONE"

        note_parts = []
        if sup_slope <= 0:
            note_parts.append("Support slope flat/falling")
        if res_slope <= 0:
            note_parts.append("Resistance slope flat/falling")
        if channel_width_pct > 20:
            note_parts.append(f"Channel too wide ({channel_width_pct:.1f}%)")
        if position > 0.70:
            note_parts.append("Price near resistance — wait for pullback")
        if not note_parts and detected:
            note_parts.append(
                f"HL×{hl_count} HH×{hh_count} "
                f"pos={position:.0%} in channel "
                f"RR={rr:.1f}"
            )

        result.detected            = detected
        result.quality             = quality
        result.entry_zone          = entry_zone
        result.support_slope       = round(sup_slope, 4)
        result.resistance_slope    = round(res_slope, 4)
        result.channel_width_pct   = round(channel_width_pct, 2)
        result.position_in_channel = round(float(np.clip(position, 0, 1)), 3)
        result.higher_lows         = int(hl_count)
        result.higher_highs        = int(hh_count)
        result.support_price       = round(float(sup_now), 2)
        result.resistance_price    = round(float(res_now), 2)
        result.risk_reward         = rr
        result.atr_contraction     = bool(atr_contraction)
        result.volume_declining_in_base = bool(vol_declining)
        result.note                = " | ".join(note_parts)

    except Exception as exc:
        result.note = f"Error: {exc}"

    return result


# ─────────────────────────────────────────────────────────────────────
# ENTRY SCORE  (0–100, feeds into Mode 7 Final Score)
# ─────────────────────────────────────────────────────────────────────

def score_channel_entry(result: ChannelResult) -> float:
    """
    Convert a ChannelResult into a 0–100 entry score.
    This is designed to BLEND with Mode 7's existing S&R Structure Score.
    """
    if not result.detected:
        return 0.0

    score = 40.0   # base for any detected channel

    # Quality
    if result.quality == "HIGH":
        score += 25
    elif result.quality == "MEDIUM":
        score += 12

    # Entry zone
    if result.entry_zone:
        score += 15

    # Higher lows / highs
    score += min(result.higher_lows,  3) * 5
    score += min(result.higher_highs, 3) * 3

    # Supporting signals
    if result.atr_contraction:
        score += 8
    if result.volume_declining_in_base:
        score += 5

    # Risk / reward
    if result.risk_reward >= 2.0:
        score += 8
    elif result.risk_reward >= 1.5:
        score += 5
    elif result.risk_reward >= 1.0:
        score += 2

    # Penalties
    if result.position_in_channel > 0.70:
        score -= 15   # too close to resistance
    if result.channel_width_pct > 18:
        score -= 8    # channel too wide = volatile
    if result.support_slope < 0:
        score -= 20   # support falling = not a channel

    return float(np.clip(round(score, 1), 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────
# SCAN FILTER  (apply to any Mode 7 result DataFrame)
# ─────────────────────────────────────────────────────────────────────

def apply_channel_filter(
    scan_df: pd.DataFrame,
    all_data: dict,           # ALL_DATA from _engine_utils
    *,
    min_quality: str = "MEDIUM",          # "HIGH" | "MEDIUM" | "LOW"
    require_entry_zone: bool = True,
    min_rr: float = 1.0,
    min_higher_lows: int = 1,
    add_score_column: bool = True,
) -> pd.DataFrame:
    """
    Filter a scan DataFrame to keep only ascending channel setups.

    Parameters
    ----------
    scan_df          : existing Mode 7 (or any mode) scan result
    all_data         : the preloaded OHLCV cache from _engine_utils.ALL_DATA
    min_quality      : minimum channel quality to keep ("HIGH" / "MEDIUM" / "LOW")
    require_entry_zone: if True, drop stocks not near support right now
    min_rr           : minimum risk/reward ratio to keep
    min_higher_lows  : minimum number of confirmed higher lows
    add_score_column : whether to append "Channel Score" column

    Returns
    -------
    pd.DataFrame     Filtered + sorted by Channel Score descending
    """
    if scan_df is None or scan_df.empty:
        return scan_df

    quality_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    min_q_rank   = quality_rank.get(min_quality.upper(), 2)

    sym_col = next(
        (c for c in ("Symbol", "Ticker", "ticker") if c in scan_df.columns),
        None
    )
    if sym_col is None:
        return scan_df

    results:      list[ChannelResult] = []
    keep_mask:    list[bool]           = []
    channel_scores: list[float]        = []

    for _, row in scan_df.iterrows():
        sym = str(row.get(sym_col, "")).strip()
        if not sym.endswith(".NS"):
            sym = sym + ".NS"

        df_ohlcv = all_data.get(sym)
        ch = detect_ascending_channel(df_ohlcv)

        q_rank   = quality_rank.get(ch.quality, 0)
        ch_score = score_channel_entry(ch)

        keep = (
            ch.detected
            and q_rank >= min_q_rank
            and ch.higher_lows >= min_higher_lows
            and ch.risk_reward >= min_rr
            and (not require_entry_zone or ch.entry_zone)
        )

        results.append(ch)
        keep_mask.append(keep)
        channel_scores.append(ch_score)

    out = scan_df.copy()

    if add_score_column:
        out["Channel Score"]       = channel_scores
        out["Channel Quality"]     = [r.quality          for r in results]
        out["Channel Entry Zone"]  = [r.entry_zone        for r in results]
        out["Support Price"]       = [r.support_price     for r in results]
        out["Resistance Price"]    = [r.resistance_price  for r in results]
        out["Higher Lows"]         = [r.higher_lows       for r in results]
        out["Higher Highs"]        = [r.higher_highs      for r in results]
        out["Channel Width %"]     = [r.channel_width_pct for r in results]
        out["Channel RR"]          = [r.risk_reward        for r in results]
        out["Channel Note"]        = [r.note              for r in results]

    out = out[keep_mask]

    if add_score_column and "Channel Score" in out.columns:
        out = out.sort_values("Channel Score", ascending=False).reset_index(drop=True)

    return out


__all__ = [
    "ChannelResult",
    "detect_ascending_channel",
    "score_channel_entry",
    "apply_channel_filter",
]
