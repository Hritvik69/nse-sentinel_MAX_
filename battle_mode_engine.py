"""
battle_mode_engine.py
──────────────────────
Multi-Stock Battle Mode engine for NSE Sentinel.

NEW FILE ONLY — does not modify any existing file or function.

Public API
──────────
    run_battle_mode(tickers, mode)  →  list[dict]
        Build raw indicator rows (no mode filter) for up to 10 tickers.
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
from strategy_engines._engine_utils import ema, rsi_vec, preload_all
from strategy_engines import get_df_for_ticker


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

_MODE_LABELS = {
    1: "🟢 Momentum",
    2: "🔵 Balanced",
    3: "🟡 Relaxed",
    4: "🟣 Institutional",
    5: "🟠 Intraday",
    6: "🔴 Swing",
}


def _build_battle_row(ticker_ns: str, mode: int) -> dict | None:
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
        df = get_df_for_ticker(ticker_ns)
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
    Build raw indicator rows for up to 10 tickers.

    Parameters
    ----------
    tickers : list[str]
        User-supplied ticker symbols (with or without .NS suffix).
        Capped at 10 internally.
    mode : int
        Strategy mode (1-6). Used only to set the "Mode" label and to
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
        for raw in tickers[:10]:
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
            preload_all(cleaned, period="6mo", workers=min(len(cleaned), 10))
        except Exception:
            pass   # fall through — get_df_for_ticker has its own fallback

        # ── 3. Build rows ─────────────────────────────────────────────
        rows: list[dict] = []
        for t_ns in cleaned:
            try:
                row = _build_battle_row(t_ns, mode)
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

def compute_battle_scores(df: pd.DataFrame) -> pd.DataFrame:
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

        battle_scores: list[float] = []
        battle_probs: list[float] = []
        battle_confidences: list[float] = []
        battle_qualities: list[float] = []
        battle_verdicts: list[str] = []
        battle_notes: list[str] = []

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
            except Exception:
                battle_scores.append(0.0)
                battle_probs.append(50.0)
                battle_confidences.append(40.0)
                battle_qualities.append(45.0)
                battle_verdicts.append("WATCHLIST")
                battle_notes.append("mixed setup")

        out["Battle Score"] = battle_scores
        out["Battle Probability"] = battle_probs
        out["Battle Confidence"] = battle_confidences
        out["Battle Quality"] = battle_qualities
        out["Battle Verdict"] = battle_verdicts
        out["Battle Notes"] = battle_notes

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

        # Sort descending by Battle Score
        out = out.sort_values("Battle Score", ascending=False).reset_index(drop=True)

        # Assign rank (1-based)
        out["Battle Rank"] = range(1, len(out) + 1)
        if len(out) > 1:
            edges = []
            scores = out["Battle Score"].tolist()
            for i, score in enumerate(scores):
                next_score = scores[i + 1] if i + 1 < len(scores) else score
                edges.append(round(float(score - next_score), 2))
            out["Battle Edge"] = edges
        else:
            out["Battle Edge"] = [0.0]

        return out

    except Exception:
        # Absolute fail-safe — return unchanged df
        return df