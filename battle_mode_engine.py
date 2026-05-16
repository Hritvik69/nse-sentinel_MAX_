"""
battle_mode_engine.py
──────────────────────
Multi-Stock Battle Mode engine for NSE Sentinel.

NEW FILE ONLY — does not modify any existing file or function.

Public API
──────────
    run_battle_mode(tickers, mode)  →  list[dict]
        Build raw indicator rows (no mode filter) for up to 19 tickers.
        Return value is fed directly to app.py's enhance_results().

    compute_battle_scores(df)       →  pd.DataFrame
        Add battle comparison columns:
            "Battle Score"  (float 0-100)
            "Battle Rank"   (int 1-N)
            "Battle Probability"  (float 0-100)
            "Battle Confidence"   (float 0-100)
            "Battle Quality"      (float 0-100)
            "Battle Verdict"      (str)
            "Battle Notes"        (str)
            "Battle Edge"         (float)

Design rules
────────────
• Zero API calls — uses get_df_for_ticker() (reads ALL_DATA / CSV / fallback)
• Never crashes — every path wrapped in try/except; returns [] / df unchanged
• Never removes rows or modifies existing columns
• Never imports from app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Existing helpers — imported, never modified ───────────────────────
from strategy_engines._engine_utils import ema, rsi_vec
from strategy_engines.mode_registry import get_mode_label_map
from feature_data_manager import feature_manager

COMPARE_STOCK_LIMIT = 19


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: safe float helper
# ─────────────────────────────────────────────────────────────────────

def _sf(v: object, default: float = 0.0) -> float:
    """Return float(v) if finite, else default. Never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _get_text(row: pd.Series, *names: str, default: str = "") -> str:
    """Read the first matching string-like column from a row."""
    try:
        for name in names:
            if name in row.index:
                val = row.get(name, default)
                if val is not None:
                    return str(val).strip()
    except Exception:
        pass
    return str(default).strip()


def _get_value(
    row: pd.Series,
    *names: str,
    default: float = 0.0,
    contains: tuple[str, ...] = (),
) -> float:
    """Read the first matching numeric column from a row."""
    try:
        for name in names:
            if name in row.index:
                return _sf(row.get(name, default), default)
        if contains:
            for key in row.index:
                key_s = str(key).lower()
                if all(token.lower() in key_s for token in contains):
                    return _sf(row.get(key, default), default)
    except Exception:
        pass
    return default


def _score_lookup(label: str, mapping: dict[str, float], default: float = 0.0) -> float:
    return float(mapping.get(str(label).strip().upper(), default))


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(v, lo, hi))


def _plain_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _row_symbol(row: pd.Series | dict) -> str:
    try:
        getter = row.get  # type: ignore[attr-defined]
        return _plain_symbol(getter("Symbol", getter("Ticker", getter("ticker", ""))))
    except Exception:
        return ""


def _prediction_lookup(prediction_cache: object) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    try:
        if isinstance(prediction_cache, pd.DataFrame):
            iterable = [row.to_dict() for _, row in prediction_cache.iterrows()]
        elif isinstance(prediction_cache, dict):
            raw = prediction_cache.get("predictions", prediction_cache.get("records", []))
            if isinstance(raw, pd.DataFrame):
                iterable = [row.to_dict() for _, row in raw.iterrows()]
            elif isinstance(raw, dict):
                iterable = list(raw.values())
            elif isinstance(raw, (list, tuple)):
                iterable = list(raw)
            else:
                iterable = [prediction_cache]
        elif isinstance(prediction_cache, (list, tuple)):
            iterable = list(prediction_cache)
        else:
            iterable = []

        for item in iterable:
            if not isinstance(item, dict):
                continue
            symbol = _row_symbol(item)
            if symbol:
                lookup[symbol] = dict(item)
    except Exception:
        return {}
    return lookup


def _text_has_any(value: object, needles: tuple[str, ...]) -> bool:
    text = str(value or "").strip().lower()
    return any(needle in text for needle in needles)


def _quality_text_score(value: object, default: float = 55.0) -> float:
    text = str(value or "").strip().upper()
    if text in {"VERY HIGH", "EXCELLENT", "A+", "STRONG"}:
        return 88.0
    if text in {"HIGH", "A", "GOOD", "BUILDING", "IDEAL", "READY", "EARLY"}:
        return 76.0
    if text in {"MEDIUM", "B", "NORMAL", "NEUTRAL", "WATCH"}:
        return 58.0
    if text in {"LOW", "C", "WEAK", "LATE"}:
        return 38.0
    if text in {"D", "BAD", "AVOID", "TRAP", "HIGH RISK"}:
        return 22.0
    return default


def _market_alignment_score(market_bias: dict | None) -> float:
    if not isinstance(market_bias, dict):
        return 55.0
    bias = str(market_bias.get("bias", market_bias.get("regime", "")) or "").lower()
    conf = _clip(_sf(market_bias.get("confidence", 50.0), 50.0))
    if "bull" in bias or "up" in bias:
        return _clip(60.0 + conf * 0.25)
    if "bear" in bias or "down" in bias or "negative" in bias:
        return _clip(46.0 - conf * 0.12)
    if "sideways" in bias or "range" in bias or "no edge" in bias:
        return 54.0
    return 55.0


def _ratio_quality(vol_ratio: float) -> float:
    if vol_ratio >= 2.6:
        return 76.0
    if vol_ratio >= 1.45:
        return 88.0
    if vol_ratio >= 1.15:
        return 78.0
    if vol_ratio >= 0.95:
        return 62.0
    if vol_ratio >= 0.75:
        return 48.0
    return 32.0


def _rsi_quality(rsi: float) -> float:
    if 52.0 <= rsi <= 66.0:
        return 90.0
    if 46.0 <= rsi < 52.0:
        return 70.0
    if 66.0 < rsi <= 72.0:
        return 64.0
    if 40.0 <= rsi < 46.0:
        return 52.0
    if 72.0 < rsi <= 78.0:
        return 38.0
    return 28.0


def _breakout_quality(dist_20d_high: float) -> float:
    if -3.5 <= dist_20d_high <= 2.0:
        return 86.0
    if 2.0 < dist_20d_high <= 6.5:
        return 72.0
    if -7.5 <= dist_20d_high < -3.5:
        return 62.0
    if 6.5 < dist_20d_high <= 11.0:
        return 48.0
    if dist_20d_high < -12.0:
        return 36.0
    return 40.0


def _extension_quality(dist_ema20: float) -> float:
    if -1.0 <= dist_ema20 <= 4.0:
        return 88.0
    if 4.0 < dist_ema20 <= 7.0:
        return 66.0
    if -4.0 <= dist_ema20 < -1.0:
        return 58.0
    if 7.0 < dist_ema20 <= 10.0:
        return 38.0
    return 30.0


def _smart_verdict(score: float, bull_prob: float, trap_score: float) -> str:
    if trap_score >= 76.0:
        return "TRAP WARNING"
    if score >= 76.0 and bull_prob >= 66.0 and trap_score <= 45.0:
        return "BEST POTENTIAL"
    if score >= 66.0 and bull_prob >= 58.0:
        return "STRONG CANDIDATE"
    if score >= 56.0:
        return "WATCHLIST"
    return "LOW EDGE"


def _smart_notes(
    *,
    momentum: float,
    volume: float,
    setup: float,
    regime: float,
    trap: float,
    rsi: float,
    dist_ema20: float,
    vol_ratio: float,
    setup_type: str,
) -> str:
    strengths: list[str] = []
    cautions: list[str] = []
    if setup >= 72.0:
        strengths.append("clean setup")
    if momentum >= 72.0:
        strengths.append("strong momentum")
    if volume >= 70.0:
        strengths.append("volume support")
    if regime >= 64.0:
        strengths.append("regime/sector support")
    if setup_type:
        strengths.append(str(setup_type).strip().lower())

    if trap >= 70.0:
        cautions.append("elevated trap risk")
    elif trap >= 55.0:
        cautions.append("some trap risk")
    if rsi > 72.0:
        cautions.append("RSI exhaustion")
    if dist_ema20 > 7.0:
        cautions.append("extended above EMA20")
    if vol_ratio < 0.85:
        cautions.append("thin volume")

    left = ", ".join(strengths[:3]) if strengths else "mixed but comparable setup"
    if cautions:
        return f"{left} | caution: {', '.join(cautions[:2])}"
    return left


def _battle_verdict(
    battle_score: float,
    battle_prob: float,
    battle_conf: float,
    final_signal: str,
    trap_risk: str,
) -> str:
    signal_u = str(final_signal).strip().upper()
    trap_u = str(trap_risk).strip().upper()

    if signal_u == "TRAP" or trap_u == "HIGH":
        return "TRAP RISK"
    if signal_u == "AVOID" or battle_score < 48.0:
        return "AVOID"
    if battle_score >= 74.0 and battle_prob >= 68.0 and battle_conf >= 62.0:
        return "STRONG WINNER"
    if battle_score >= 64.0 and battle_prob >= 58.0 and battle_conf >= 55.0:
        return "BETTER PICK"
    if battle_score >= 54.0:
        return "WATCHLIST"
    return "WEAK SETUP"


def _battle_notes(
    final_signal: str,
    setup_quality: str,
    entry_timing: str,
    vol_trend: str,
    setup_type: str,
    trap_risk: str,
    advanced_trap: str,
    rsi: float,
    dist_ema20: float,
    ret_20d: float,
) -> str:
    strengths: list[str] = []
    cautions: list[str] = []

    sig_u = str(final_signal).strip().upper()
    sq_u = str(setup_quality).strip().upper()
    et_u = str(entry_timing).strip().upper()
    vt_u = str(vol_trend).strip().upper()
    trap_u = str(trap_risk).strip().upper()
    adv_u = str(advanced_trap).strip().upper()
    setup_label = str(setup_type).strip()

    if sig_u in {"STRONG BUY", "BUY"}:
        strengths.append(sig_u.title())
    if sq_u == "HIGH":
        strengths.append("high-quality setup")
    if et_u in {"EARLY", "IDEAL", "READY"}:
        strengths.append(f"{et_u.lower()} entry")
    if vt_u in {"STRONG", "BUILDING"}:
        strengths.append("volume confirmation")
    if setup_label:
        strengths.append(setup_label.lower())
    if ret_20d > 0 and dist_ema20 > -1.5:
        strengths.append("trend aligned")

    if trap_u == "HIGH":
        cautions.append("high trap risk")
    elif trap_u == "MEDIUM":
        cautions.append("medium trap risk")
    if adv_u and adv_u not in {"NONE", "N/A", "NA"}:
        cautions.append(adv_u.replace("_", " ").lower())
    if rsi >= 72.0:
        cautions.append("rsi overheated")
    elif rsi <= 40.0:
        cautions.append("weak rsi")
    if dist_ema20 > 6.0:
        cautions.append("stretched above ema20")
    elif dist_ema20 < -4.0:
        cautions.append("below ema20")

    left = ", ".join(strengths[:3]) if strengths else "mixed setup"
    if not cautions:
        return left
    return f"{left} | caution: {', '.join(cautions[:2])}"


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: build one raw indicator row (no mode filter applied)
# ─────────────────────────────────────────────────────────────────────

_MODE_LABELS = get_mode_label_map()


def _build_battle_row(ticker_ns: str, mode: int, df: pd.DataFrame | None = None) -> dict | None:
    """
    Build the same row structure that analyse() would return but WITHOUT
    applying any mode-specific filter conditions.

    Returns None only if:
        • Data cannot be loaded
        • Fewer than 25 rows after cleaning
        • Price / volume / EMA / RSI are invalid

    Never crashes.
    """
    try:
        df = df.copy() if isinstance(df, pd.DataFrame) else None
        if df is None or df.empty:
            return None

        # ── 🕰️ TIME TRAVEL: truncate to cutoff for tickers that arrived via
        # live fallback (not pre-snapshotted in ALL_DATA).  ALL_DATA entries
        # are already truncated by time_travel_engine.activate(), but any
        # ticker absent from the preload universe downloads fresh data — we
        # must slice it here to prevent future-data leakage.
        try:
            import time_travel_engine as _tt_be
            if _tt_be.is_active():
                _tt_cut = _tt_be.get_reference_date()
                if _tt_cut is not None:
                    _tt_mask = pd.to_datetime(df.index).date <= _tt_cut
                    df = df.loc[_tt_mask]
                    if df.empty or len(df) < 25:
                        return None
        except Exception:
            pass  # fail-safe: continue with whatever data we have

        # Normalise MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Drop rows missing critical columns
        needed = [c for c in ["Close", "Volume"] if c in df.columns]
        if not needed:
            return None
        df = df.dropna(subset=needed)
        if len(df) < 25:
            return None

        close  = df["Close"].dropna().astype(float)
        volume = df["Volume"].dropna().astype(float)

        if len(close) < 25:
            return None

        # ── Key indicators ────────────────────────────────────────────
        lc  = float(close.iloc[-1])
        lv  = float(volume.iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])

        avg_vol = (
            float(volume.iloc[-21:-1].mean())
            if len(volume) >= 21
            else float(volume.mean())
        )

        # Vectorised RSI (same as _engine_utils.rsi_vec)
        rsi_s   = rsi_vec(close)
        ri      = float(rsi_s.iloc[-1]) if not rsi_s.empty else float("nan")

        # Basic validity checks (same as analyse())
        if not (1 < lc <= 100_000):
            return None
        if lv <= 0:
            return None
        if any(np.isnan(v) for v in (ri, e20, e50)):
            return None

        # ── Derived fields ────────────────────────────────────────────
        h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
        dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
        dist_ema20    = (lc / e20 - 1.0) * 100.0        if e20    > 0 else 0.0
        ret_5d        = (lc / float(close.iloc[-6])  - 1.0) * 100.0 if len(close) >= 6  else float("nan")
        ret_20d       = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else float("nan")

        sym = ticker_ns.replace(".NS", "")

        return {
            "Symbol":             sym,
            "Price (₹)":          round(lc, 2),
            "Volume":             int(lv),
            "RSI":                round(ri, 2),
            "EMA 20":             round(e20, 2),
            "EMA 50":             round(e50, 2),
            "Vol / Avg":          round(lv / avg_vol, 2) if avg_vol > 0 else 0.0,
            "Mode ID":            int(mode),
            "Mode":               _MODE_LABELS.get(mode, "🔵 Balanced"),
            "Δ vs 20D High (%)":  round(dist_20d_high, 2),
            "Δ vs EMA20 (%)":     round(dist_ema20, 2),
            "5D Return (%)":      round(ret_5d, 2)  if not np.isnan(ret_5d)  else float("nan"),
            "20D Return (%)":     round(ret_20d, 2) if not np.isnan(ret_20d) else float("nan"),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# PUBLIC: run_battle_mode
# ─────────────────────────────────────────────────────────────────────

def run_battle_mode(tickers: list[str], mode: int) -> list[dict]:
    """
    Build raw indicator rows for up to 19 tickers.

    Parameters
    ----------
    tickers : list[str]
        User-supplied ticker symbols (with or without .NS suffix).
        Capped at 19 internally.
    mode : int
        Strategy mode. Used only to set the "Mode" label and to
        configure engine functions — no scan filter is applied.

    Returns
    -------
    list[dict]
        Ready to pass into app.py's enhance_results(rows, mode).
        Empty list on complete failure (never crashes).
    """
    try:
        if not tickers:
            return []

        # ── 1. Clean and cap tickers ──────────────────────────────────
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in tickers[:COMPARE_STOCK_LIMIT]:
            t = str(raw).strip().upper()
            if not t:
                continue
            t_ns = t if t.endswith(".NS") else f"{t}.NS"
            if t_ns not in seen:
                seen.add(t_ns)
                cleaned.append(t_ns)

        if not cleaned:
            return []

        # ── 2. Preload data using existing helper (zero new API logic) ─
        try:
            frames = feature_manager.get_multiple_stocks(
                cleaned,
                period="6mo",
                interval="1d",
                force_refresh=False,
            )
        except Exception:
            frames = {}

        # ── 3. Build rows ─────────────────────────────────────────────
        rows: list[dict] = []
        for t_ns in cleaned:
            try:
                row = _build_battle_row(t_ns, mode, frames.get(t_ns))
                if row is not None:
                    rows.append(row)
            except Exception:
                continue   # skip invalid ticker silently

        return rows

    except Exception:
        return []   # absolute fail-safe


# ─────────────────────────────────────────────────────────────────────
# PUBLIC: compute_battle_scores — ADDITIVE only, new columns only
# ─────────────────────────────────────────────────────────────────────

def compute_battle_scores(
    df: pd.DataFrame,
    market_bias: dict | None = None,
    prediction_cache: object = None,
    sector_context: dict | None = None,
) -> pd.DataFrame:
    """
    Add battle comparison columns to an already-enriched
    DataFrame (after enhance_results + grading + enhanced_logic + phase4).

    The ranking blends:
        • Final Score + Prediction Score
        • Confidence + ML %
        • Risk / trap penalties
        • Setup quality, timing, trend and volume quality

    Any missing column safely falls back to a neutral default.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        out = df.copy()
        prediction_map = _prediction_lookup(prediction_cache)
        market_alignment_base = _market_alignment_score(market_bias)

        battle_scores: list[float] = []
        battle_probs: list[float] = []
        battle_confidences: list[float] = []
        battle_qualities: list[float] = []
        battle_verdicts: list[str] = []
        battle_notes: list[str] = []
        smart_scores: list[float] = []
        bullish_probs: list[float] = []
        smart_confidences: list[float] = []
        momentum_scores: list[float] = []
        volume_scores: list[float] = []
        trap_scores: list[float] = []
        setup_scores: list[float] = []
        regime_scores: list[float] = []
        historical_scores: list[float] = []
        risk_reward_scores: list[float] = []
        sector_scores: list[float] = []
        smart_verdicts: list[str] = []
        smart_notes: list[str] = []
        trap_warnings: list[str] = []
        compare_tags: list[str] = []

        for idx in out.index:
            try:
                row = out.loc[idx]

                final_score = _get_value(row, "Final Score", "Score", default=0.0)
                pred_score = _get_value(row, "Prediction Score", default=final_score)
                confidence = _get_value(row, "Confidence", default=50.0)
                risk_score = _get_value(row, "Risk Score", default=50.0)
                ml_pct = _get_value(row, "ML %", "ML Prob", "ML Score", default=50.0)

                final_signal = _get_text(row, "Final Signal", "Signal", default="WATCH").upper()
                signal = _get_text(row, "Signal", "Final Signal", default=final_signal).upper()
                setup_quality = _get_text(row, "Setup Quality", default="MEDIUM").upper()
                entry_timing = _get_text(row, "Entry Timing", default="NEUTRAL").upper()
                volume_trend = _get_text(row, "Volume Trend", default="NORMAL").upper()
                setup_type = _get_text(row, "Setup Type", default="")
                trap_risk = _get_text(row, "Trap Risk", default="LOW").upper()
                advanced_trap = _get_text(row, "Advanced Trap", default="").upper()

                rsi = _get_value(row, "RSI", default=50.0)
                vol_avg = _get_value(row, "Vol / Avg", default=1.0)
                ret_5d = _get_value(row, "5D Return (%)", default=0.0)
                ret_20d = _get_value(row, "20D Return (%)", default=0.0)
                dist_ema20 = _get_value(
                    row,
                    "Δ vs EMA20 (%)",
                    default=0.0,
                    contains=("ema20",),
                )
                dist_20d_high = _get_value(
                    row,
                    "Δ vs 20D High (%)",
                    default=-5.0,
                    contains=("20d", "high"),
                )

                safety_score = _clip(100.0 - risk_score)
                core_prob = (
                    0.28 * final_score
                    + 0.24 * pred_score
                    + 0.18 * confidence
                    + 0.12 * ml_pct
                    + 0.18 * safety_score
                )

                signal_adj = _score_lookup(
                    final_signal or signal,
                    {
                        "STRONG BUY": 12.0,
                        "BUY": 7.0,
                        "WATCH": -2.0,
                        "AVOID": -12.0,
                        "TRAP": -18.0,
                    },
                )
                setup_adj = _score_lookup(
                    setup_quality,
                    {
                        "HIGH": 5.0,
                        "MEDIUM": 1.0,
                        "LOW": -4.0,
                    },
                )
                timing_adj = _score_lookup(
                    entry_timing,
                    {
                        "EARLY": 4.0,
                        "IDEAL": 4.0,
                        "READY": 3.0,
                        "GOOD": 2.0,
                        "NEUTRAL": 0.0,
                        "LATE": -3.0,
                    },
                )
                volume_adj = _score_lookup(
                    volume_trend,
                    {
                        "STRONG": 5.0,
                        "BUILDING": 3.0,
                        "NORMAL": 1.0,
                        "WEAK": -4.0,
                    },
                )

                rsi_adj = 0.0
                if 52.0 <= rsi <= 66.0:
                    rsi_adj = 4.0
                elif 45.0 <= rsi < 52.0:
                    rsi_adj = 1.0
                elif 66.0 < rsi <= 72.0:
                    rsi_adj = -1.5
                elif rsi > 72.0:
                    rsi_adj = -5.0
                elif rsi < 40.0:
                    rsi_adj = -6.0

                vol_ratio_adj = 0.0
                if vol_avg >= 1.5:
                    vol_ratio_adj = 4.0
                elif vol_avg >= 1.15:
                    vol_ratio_adj = 2.0
                elif vol_avg < 0.85:
                    vol_ratio_adj = -4.0

                trend_adj = float(np.clip(ret_5d * 0.9, -4.0, 4.0)) + float(np.clip(ret_20d * 0.35, -4.0, 5.0))

                stretch_adj = 0.0
                if -1.5 <= dist_ema20 <= 4.0:
                    stretch_adj += 3.0
                elif dist_ema20 > 6.0:
                    stretch_adj -= 4.0
                elif dist_ema20 < -4.0:
                    stretch_adj -= 4.0

                if -4.0 <= dist_20d_high <= 7.0:
                    # Stock is within or just above 20D high — healthy range including breakouts
                    stretch_adj += 2.0
                elif 7.0 < dist_20d_high <= 12.0:
                    # Extended above 20D high but not extreme — neutral
                    stretch_adj += 0.0
                elif dist_20d_high > 12.0:
                    # Very overextended above 20D high — likely parabolic, fade risk
                    stretch_adj -= 3.0
                elif dist_20d_high < -10.0:
                    stretch_adj -= 2.0

                if "BREAKOUT" in setup_type.upper():
                    stretch_adj += 1.5
                elif "PULLBACK" in setup_type.upper():
                    stretch_adj += 2.0
                elif "REVERSAL" in setup_type.upper():
                    stretch_adj += 0.5

                trap_penalty = _score_lookup(
                    trap_risk,
                    {
                        "HIGH": 14.0,
                        "MEDIUM": 6.0,
                        "LOW": 0.0,
                    },
                )
                adv_penalty = 0.0
                if advanced_trap and advanced_trap not in {"NONE", "N/A", "NA"}:
                    adv_penalty = 8.0
                    if "FAKE" in advanced_trap or "EXHAUST" in advanced_trap:
                        adv_penalty = 10.0

                quality_score = _clip(
                    50.0
                    + signal_adj
                    + setup_adj
                    + timing_adj
                    + volume_adj
                    + vol_ratio_adj
                    + rsi_adj
                    + trend_adj
                    + stretch_adj
                    - trap_penalty
                    - adv_penalty
                )

                agreement_gap = (
                    abs(final_score - pred_score)
                    + abs(final_score - ml_pct)
                    + abs(pred_score - ml_pct)
                ) / 3.0
                consistency_score = _clip(100.0 - agreement_gap * 1.4)
                agreement_scale = max(0.60, min(1.0, 0.55 + consistency_score / 220.0))

                battle_prob = 0.68 * core_prob + 0.32 * quality_score
                battle_prob = 50.0 + (battle_prob - 50.0) * agreement_scale

                battle_conf = (
                    0.42 * confidence
                    + 0.20 * safety_score
                    + 0.20 * consistency_score
                    + 0.18 * quality_score
                )

                if trap_risk == "HIGH":
                    battle_prob = 50.0 + (battle_prob - 50.0) * 0.72
                    battle_conf *= 0.78
                elif trap_risk == "MEDIUM":
                    battle_prob = 50.0 + (battle_prob - 50.0) * 0.86
                    battle_conf *= 0.90

                if advanced_trap and advanced_trap not in {"NONE", "N/A", "NA"}:
                    battle_prob = 50.0 + (battle_prob - 50.0) * 0.84
                    battle_conf *= 0.85

                if final_signal == "TRAP":
                    battle_prob = min(battle_prob, 40.0)
                    battle_conf *= 0.72
                elif final_signal == "AVOID":
                    battle_prob = min(battle_prob, 46.0)
                    battle_conf *= 0.82
                elif final_signal == "WATCH":
                    battle_prob = 50.0 + (battle_prob - 50.0) * 0.92

                battle_prob = _clip(battle_prob)
                battle_conf = _clip(battle_conf)

                bs = _clip(
                    0.50 * battle_prob
                    + 0.25 * battle_conf
                    + 0.25 * quality_score
                )

                if final_signal == "TRAP":
                    bs = min(bs, 40.0)
                elif final_signal == "AVOID":
                    bs = min(bs, 47.0)

                symbol = _row_symbol(row)
                pred_ctx = prediction_map.get(symbol, {})
                if isinstance(sector_context, dict):
                    try:
                        sector_by_symbol = sector_context.get("by_symbol", {})
                        if isinstance(sector_by_symbol, dict):
                            pred_ctx = {**dict(sector_by_symbol.get(symbol, {}) or {}), **pred_ctx}
                    except Exception:
                        pass

                price = _get_value(row, "Close", "Price", default=0.0, contains=("price",))
                ema20 = _get_value(row, "EMA 20", "EMA20", default=0.0, contains=("ema", "20"))
                ema50 = _get_value(row, "EMA 50", "EMA50", default=0.0, contains=("ema", "50"))
                ema_alignment = 50.0
                if price > 0 and ema20 > 0:
                    ema_alignment += 20.0 if price >= ema20 else -16.0
                if ema20 > 0 and ema50 > 0:
                    ema_alignment += 18.0 if ema20 >= ema50 else -14.0
                if -1.0 <= dist_ema20 <= 4.5:
                    ema_alignment += 10.0
                elif dist_ema20 > 8.0 or dist_ema20 < -5.0:
                    ema_alignment -= 10.0
                ema_alignment = _clip(ema_alignment)

                rsi_quality = _rsi_quality(rsi)
                trend_cleanliness = _clip(
                    50.0
                    + float(np.clip(ret_5d * 3.2, -18.0, 18.0))
                    + float(np.clip(ret_20d * 1.25, -18.0, 18.0))
                )
                if ret_5d > 0.0 and ret_20d > 0.0:
                    trend_cleanliness += 5.0
                elif ret_5d < 0.0 and ret_20d < 0.0:
                    trend_cleanliness -= 8.0
                trend_cleanliness = _clip(trend_cleanliness)

                breakout_quality = _breakout_quality(dist_20d_high)
                extension_quality = _extension_quality(dist_ema20)
                momentum_quality = _clip(
                    0.28 * rsi_quality
                    + 0.24 * ema_alignment
                    + 0.22 * trend_cleanliness
                    + 0.16 * breakout_quality
                    + 0.10 * extension_quality
                )

                vol_ratio_quality = _ratio_quality(vol_avg)
                volume_text_quality = _quality_text_score(volume_trend, default=58.0)
                volume_confirmation = _quality_text_score(
                    _get_text(row, "Volume Confirmation", "Volume Strength", default=volume_trend),
                    default=volume_text_quality,
                )
                volume_quality = _clip(
                    0.48 * vol_ratio_quality
                    + 0.30 * volume_text_quality
                    + 0.22 * volume_confirmation
                )
                suspicious_low_volume = vol_avg < 0.85 and (ret_5d > 1.5 or dist_20d_high >= -4.0)
                if suspicious_low_volume:
                    volume_quality = _clip(volume_quality - 14.0)

                trap_score = _score_lookup(
                    trap_risk,
                    {"HIGH": 76.0, "MEDIUM": 52.0, "LOW": 24.0},
                    default=38.0,
                )
                if advanced_trap and advanced_trap not in {"NONE", "N/A", "NA"}:
                    trap_score += 15.0
                if rsi > 72.0:
                    trap_score += 12.0
                if dist_ema20 > 7.0:
                    trap_score += 14.0
                if ret_20d > 18.0 and rsi > 66.0:
                    trap_score += 9.0
                if suspicious_low_volume:
                    trap_score += 12.0
                if final_signal == "TRAP":
                    trap_score += 20.0
                elif final_signal == "AVOID":
                    trap_score += 8.0
                trap_score = _clip(trap_score)

                grade_quality = _quality_text_score(_get_text(row, "Grade", default="B"), default=56.0)
                setup_text_quality = _quality_text_score(setup_quality, default=58.0)
                timing_quality = _quality_text_score(entry_timing, default=56.0)
                setup_cleanliness = _clip(
                    0.30 * setup_text_quality
                    + 0.20 * timing_quality
                    + 0.20 * grade_quality
                    + 0.18 * extension_quality
                    + 0.12 * volume_quality
                    - 0.34 * trap_score
                    + 28.0
                )

                tomorrow_score = _sf(
                    pred_ctx.get("score", pred_ctx.get("raw_score", pred_score)),
                    pred_score,
                )
                tomorrow_conf = _sf(pred_ctx.get("confidence", confidence), confidence)
                learned_prob = _sf(
                    pred_ctx.get("learned_probability", _get_value(row, "Learned Prob %", default=ml_pct)),
                    ml_pct,
                )
                pred_direction = str(pred_ctx.get("direction", "") or "").upper()

                sector_strength = _get_value(
                    row,
                    "Sector Strength",
                    "sector_strength",
                    default=_sf(pred_ctx.get("sector_accuracy", 55.0), 55.0),
                    contains=("sector", "strength"),
                )
                sector_momentum = _get_value(
                    row,
                    "Sector Momentum",
                    "momentum_score",
                    default=sector_strength,
                    contains=("sector", "momentum"),
                )
                sector_accuracy = _sf(pred_ctx.get("sector_accuracy", sector_strength), sector_strength)
                sector_support = _clip(0.48 * sector_strength + 0.30 * sector_momentum + 0.22 * sector_accuracy)
                regime_fit = _sf(pred_ctx.get("regime_fit", market_alignment_base), market_alignment_base)
                market_alignment = market_alignment_base
                if "BEAR" in pred_direction:
                    market_alignment = min(market_alignment, 48.0)
                elif "BULL" in pred_direction:
                    market_alignment = max(market_alignment, 58.0)
                regime_alignment = _clip(0.42 * market_alignment + 0.35 * sector_support + 0.23 * regime_fit)

                backtest = _get_value(row, "Backtest %", "Historical Win %", default=confidence)
                scanner_reliability = _get_value(row, "Scanner Reliability", "Mode Reliability", default=confidence)
                historical_reliability = _clip(
                    0.24 * backtest
                    + 0.22 * learned_prob
                    + 0.18 * ml_pct
                    + 0.16 * confidence
                    + 0.12 * scanner_reliability
                    + 0.08 * consistency_score
                )

                risk_reward = _clip(
                    0.32 * setup_cleanliness
                    + 0.24 * momentum_quality
                    + 0.18 * (100.0 - risk_score)
                    + 0.16 * breakout_quality
                    + 0.10 * extension_quality
                    - 0.22 * trap_score
                    + 18.0
                )

                bullish_probability = _clip(
                    0.45 * battle_prob
                    + 0.22 * tomorrow_score
                    + 0.12 * tomorrow_conf
                    + 0.09 * momentum_quality
                    + 0.07 * volume_quality
                    + 0.05 * regime_alignment
                    - 0.12 * trap_score
                )
                if "BEAR" in pred_direction:
                    bullish_probability = min(bullish_probability, 48.0)
                elif "BULL" in pred_direction:
                    bullish_probability = _clip(bullish_probability + 3.0)

                smart_score = _clip(
                    0.24 * bullish_probability
                    + 0.18 * setup_cleanliness
                    + 0.16 * momentum_quality
                    + 0.12 * volume_quality
                    + 0.12 * risk_reward
                    + 0.10 * regime_alignment
                    + 0.08 * historical_reliability
                    - 0.12 * trap_score
                    + 10.0
                )
                smart_confidence = _clip(
                    0.38 * battle_conf
                    + 0.24 * historical_reliability
                    + 0.18 * tomorrow_conf
                    + 0.12 * consistency_score
                    + 0.08 * (100.0 - trap_score)
                )

                trap_warning = ""
                if trap_score >= 72.0:
                    trap_warning = "High trap risk"
                elif trap_score >= 56.0:
                    trap_warning = "Watch trap risk"
                elif suspicious_low_volume:
                    trap_warning = "Low-volume move"
                else:
                    trap_warning = "Clean"

                tags: list[str] = []
                if bullish_probability >= 66.0:
                    tags.append("Probability")
                if setup_cleanliness >= 70.0:
                    tags.append("Clean")
                if momentum_quality >= 72.0:
                    tags.append("Momentum")
                if volume_quality >= 70.0:
                    tags.append("Volume")
                if regime_alignment >= 64.0:
                    tags.append("Regime")
                if trap_score >= 56.0:
                    tags.append("Trap Watch")
                if not tags:
                    tags.append("Mixed")

                smart_note = _smart_notes(
                    momentum=momentum_quality,
                    volume=volume_quality,
                    setup=setup_cleanliness,
                    regime=regime_alignment,
                    trap=trap_score,
                    rsi=rsi,
                    dist_ema20=dist_ema20,
                    vol_ratio=vol_avg,
                    setup_type=setup_type,
                )
                verdict = _battle_verdict(bs, battle_prob, battle_conf, final_signal, trap_risk)
                notes = _battle_notes(
                    final_signal=final_signal,
                    setup_quality=setup_quality,
                    entry_timing=entry_timing,
                    vol_trend=volume_trend,
                    setup_type=setup_type,
                    trap_risk=trap_risk,
                    advanced_trap=advanced_trap,
                    rsi=rsi,
                    dist_ema20=dist_ema20,
                    ret_20d=ret_20d,
                )

                battle_scores.append(round(bs, 2))
                battle_probs.append(round(battle_prob, 2))
                battle_confidences.append(round(battle_conf, 2))
                battle_qualities.append(round(quality_score, 2))
                battle_verdicts.append(verdict)
                battle_notes.append(notes)
                smart_scores.append(round(smart_score, 2))
                bullish_probs.append(round(bullish_probability, 2))
                smart_confidences.append(round(smart_confidence, 2))
                momentum_scores.append(round(momentum_quality, 2))
                volume_scores.append(round(volume_quality, 2))
                trap_scores.append(round(trap_score, 2))
                setup_scores.append(round(setup_cleanliness, 2))
                regime_scores.append(round(regime_alignment, 2))
                historical_scores.append(round(historical_reliability, 2))
                risk_reward_scores.append(round(risk_reward, 2))
                sector_scores.append(round(sector_support, 2))
                smart_verdicts.append(_smart_verdict(smart_score, bullish_probability, trap_score))
                smart_notes.append(smart_note)
                trap_warnings.append(trap_warning)
                compare_tags.append(", ".join(tags))
            except Exception:
                battle_scores.append(0.0)
                battle_probs.append(50.0)
                battle_confidences.append(40.0)
                battle_qualities.append(45.0)
                battle_verdicts.append("WATCHLIST")
                battle_notes.append("mixed setup")
                smart_scores.append(45.0)
                bullish_probs.append(50.0)
                smart_confidences.append(40.0)
                momentum_scores.append(45.0)
                volume_scores.append(45.0)
                trap_scores.append(55.0)
                setup_scores.append(45.0)
                regime_scores.append(55.0)
                historical_scores.append(50.0)
                risk_reward_scores.append(45.0)
                sector_scores.append(55.0)
                smart_verdicts.append("WATCHLIST")
                smart_notes.append("mixed setup")
                trap_warnings.append("Check manually")
                compare_tags.append("Mixed")

        out["Battle Score"] = battle_scores
        out["Battle Probability"] = battle_probs
        out["Battle Confidence"] = battle_confidences
        out["Battle Quality"] = battle_qualities
        out["Battle Verdict"] = battle_verdicts
        out["Battle Notes"] = battle_notes
        out["Smart Potential Score"] = smart_scores
        out["Bullish Probability"] = bullish_probs
        out["Smart Confidence"] = smart_confidences
        out["Momentum Quality"] = momentum_scores
        out["Volume Quality"] = volume_scores
        out["Trap Risk Score"] = trap_scores
        out["Setup Cleanliness"] = setup_scores
        out["Regime Alignment"] = regime_scores
        out["Historical Reliability"] = historical_scores
        out["Risk Reward Score"] = risk_reward_scores
        out["Sector Support"] = sector_scores
        out["Smart Verdict"] = smart_verdicts
        out["Smart Notes"] = smart_notes
        out["Trap Warning"] = trap_warnings
        out["Compare Tags"] = compare_tags

        # ── Within-group relative normalization ───────────────────────
        # When all stocks score similarly (e.g. 70–75), raw scores look identical.
        # Stretch scores within the group to a [45, 92] range so the best pick
        # and worst pick are always clearly separated in the comparison view.
        # Raw scores are preserved in "Battle Score Raw" for transparency.
        out["Battle Score Raw"] = out["Battle Score"]
        if len(out) > 1:
            raw_arr = np.array(out["Battle Score"].tolist(), dtype=float)
            lo, hi  = raw_arr.min(), raw_arr.max()
            _NORM_LO, _NORM_HI = 45.0, 92.0
            if hi - lo > 0.5:   # only normalize if there's actual spread
                normed = _NORM_LO + (raw_arr - lo) / (hi - lo) * (_NORM_HI - _NORM_LO)
                # Blend: 60% normalized (for clear ranking) + 40% raw (for accuracy)
                blended = 0.60 * normed + 0.40 * raw_arr
                out["Battle Score"] = [round(float(v), 2) for v in blended]
            else:
                # All identical — spread them evenly around the raw score
                step = 2.0
                base = float(lo)
                spread = [round(base + (len(out) - 1 - i) * step, 2) for i in range(len(out))]
                out["Battle Score"] = spread

        # Re-derive verdicts from normalized Battle Score for consistent labeling
        for idx2 in out.index:
            bs_n  = float(out.at[idx2, "Battle Score"])
            bp_n  = float(out.at[idx2, "Battle Probability"])
            bc_n  = float(out.at[idx2, "Battle Confidence"])
            fs_n  = str(out.at[idx2, "Battle Verdict"])   # keep TRAP/AVOID hard labels
            tr_n  = str(battle_verdicts[idx2] if idx2 < len(battle_verdicts) else "").upper()
            # Only re-derive non-TRAP/AVOID verdicts — preserve danger labels
            if "TRAP" not in fs_n.upper() and "AVOID" not in fs_n.upper():
                out.at[idx2, "Battle Verdict"] = _battle_verdict(bs_n, bp_n, bc_n, "WATCH", "LOW")

        # Sort by the smart selector score while preserving legacy Battle columns.
        sort_col = "Smart Potential Score" if "Smart Potential Score" in out.columns else "Battle Score"
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Assign rank (1-based)
        out["Battle Rank"] = range(1, len(out) + 1)
        out["Smart Rank"] = out["Battle Rank"]
        if len(out) > 1:
            edges = []
            scores = out[sort_col].tolist()
            for i, score in enumerate(scores):
                next_score = scores[i + 1] if i + 1 < len(scores) else score
                edges.append(round(float(score - next_score), 2))
            out["Battle Edge"] = edges
            out["Smart Edge"] = edges
        else:
            out["Battle Edge"] = [0.0]
            out["Smart Edge"] = [0.0]

        return out

    except Exception:
        # Absolute fail-safe — return unchanged df
        return df
