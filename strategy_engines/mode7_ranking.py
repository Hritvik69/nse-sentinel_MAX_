"""
Dedicated Mode 7 ranking and signal suppression.

Mode 7 is structure-first: clean S&R, breakout quality, support strength,
volume confirmation, and low trap probability outrank raw prediction hype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_engines.constants import (
    MODE7_EMA_EXTENSION_HARD,
    MODE7_NEGATIVE_SETUPS,
    MODE7_POSITIVE_SETUPS,
    MODE7_RSI_EXHAUSTION,
    MODE7_VOL_WEAK,
    QUALITY_POINTS,
    TRAP_POINTS,
    debug_log,
)


_QUALITY_SCORE = QUALITY_POINTS
_TRAP_SCORE = TRAP_POINTS


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        f = float(value)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _text(row: pd.Series, key: str, default: str = "") -> str:
    try:
        return str(row.get(key, default) or default).strip().upper()
    except Exception:
        return default


def _q(row: pd.Series, key: str, default: str = "MEDIUM") -> float:
    return _QUALITY_SCORE.get(_text(row, key, default), _QUALITY_SCORE[default])


def _num_series(df: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    if key not in df.columns:
        if isinstance(default, pd.Series):
            return pd.to_numeric(default.reindex(df.index), errors="coerce").fillna(0.0).astype(float)
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[key], errors="coerce").fillna(default).astype(float)


def _text_series(df: pd.DataFrame, key: str, default: str = "") -> pd.Series:
    if key not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[key].where(df[key].notna(), default).astype(str).str.strip().str.upper()


def _quality_series(df: pd.DataFrame, key: str, default: str = "MEDIUM") -> pd.Series:
    return _text_series(df, key, default).map(_QUALITY_SCORE).fillna(_QUALITY_SCORE[default]).astype(float)


def _trap_series(df: pd.DataFrame, key: str = "Trap Probability") -> pd.Series:
    return _text_series(df, key, "LOW").map(_TRAP_SCORE).fillna(_TRAP_SCORE["LOW"]).astype(float)


def _regime_adjustment(market_bias: dict | None, row: pd.Series) -> float:
    try:
        if not isinstance(market_bias, dict):
            return 0.0
        bias = str(market_bias.get("bias", "") or "").lower()
        regime = str(market_bias.get("regime", "") or "").lower()
        vol = _num(row, "Vol / Avg", 1.0)
        setup = _text(row, "Setup Type", "")
        base_tight = _num(row, "Base Tightness (%)", 99.0)
        adj = 0.0
        if "bullish" in bias or "trending up" in regime:
            if setup in {"BREAKOUT READY", "EARLY BREAKOUT", "MOMENTUM CONTINUATION"}:
                adj += 3.0
        if "bearish" in bias or "trending down" in regime:
            adj -= 3.0
            if vol < 1.3:
                adj -= 2.0
        if "ranging" in regime or "sideways" in bias:
            if setup in {"RESISTANCE COMPRESSION", "SUPPORT BOUNCE"} and base_tight <= 8.0:
                adj += 3.0
            elif setup in {"EARLY BREAKOUT"} and vol < 1.4:
                adj -= 2.0
        return float(np.clip(adj, -5.0, 5.0))
    except Exception:
        return 0.0


def final_score_mode7(row: pd.Series, market_bias: dict | None = None) -> float:
    """Return a structure-first final score for one Mode 7 row."""
    try:
        final = _num(row, "Final Score", _num(row, "Score", 50.0))
        pred = _num(row, "Prediction Score", _num(row, "ML %", 50.0))
        score = _num(row, "Score", final)
        bt = _num(row, "Backtest %", 50.0)
        ml = _num(row, "ML %", 50.0)
        sr = _num(row, "S&R Structure Score", 50.0)
        channel_score = _num(row, "Channel Score", 0.0)
        channel_entry = _text(row, "Channel Entry Zone", "NO") == "YES"
        channel_detected = _text(row, "Ascending Channel", "NO") == "YES"

        structure = (
            0.22 * _q(row, "Structure Quality")
            + 0.18 * _q(row, "Breakout Quality")
            + 0.15 * _q(row, "Support Strength")
            + 0.14 * _q(row, "Volume Confirmation")
            + 0.10 * _q(row, "Momentum Continuation")
            + 0.08 * _q(row, "Resistance Distance")
            + 0.08 * _TRAP_SCORE.get(_text(row, "Trap Probability", "LOW"), 100.0)
            + 0.05 * sr
        )

        blended = (
            0.62 * structure
            + 0.15 * pred
            + 0.10 * final
            + 0.06 * score
            + 0.04 * bt
            + 0.03 * ml
        )
        blended += _regime_adjustment(market_bias, row)
        if channel_entry:
            blended += min(max(channel_score * 0.10, 4.0), 10.0)
        elif channel_detected:
            blended += min(channel_score * 0.04, 4.0)

        setup = _text(row, "Setup Type", "")
        trap_prob = _text(row, "Trap Probability", "LOW")
        trap_risk = _text(row, "Trap Risk", "LOW")
        advanced = _text(row, "Advanced Trap", "NONE")
        vol = _num(row, "Vol / Avg", 1.0)
        de20 = _num(row, "Δ vs EMA20 (%)", _num(row, "Delta vs EMA20 (%)", 0.0))
        rsi = _num(row, "RSI", 50.0)
        higher_lows = max(_num(row, "Higher Lows", 0.0), _num(row, "Channel Higher Lows", 0.0))

        if trap_prob == "HIGH" or trap_risk == "HIGH" or "FAKE" in advanced:
            blended -= 24.0
            blended = min(blended, 58.0)
        elif trap_prob == "MEDIUM" or trap_risk == "MEDIUM":
            blended -= 8.0
        if setup in MODE7_NEGATIVE_SETUPS:
            blended -= 18.0
            blended = min(blended, 62.0)
        if setup == "SUPPORT BOUNCE" and higher_lows < 2:
            blended -= 12.0
            blended = min(blended, 62.0)
        if vol < MODE7_VOL_WEAK:
            blended -= 9.0
        if de20 > MODE7_EMA_EXTENSION_HARD or rsi > MODE7_RSI_EXHAUSTION:
            blended -= 10.0

        return float(np.clip(round(blended, 2), 0.0, 100.0))
    except Exception:
        return float(np.clip(_num(row, "Final Score", 50.0), 0.0, 100.0))


def ranking_priority_mode7(row: pd.Series, market_bias: dict | None = None) -> tuple[float, float, float, float]:
    """Sort tuple: structure first, traps last, then prediction."""
    score = final_score_mode7(row, market_bias)
    trap = _TRAP_SCORE.get(_text(row, "Trap Probability", "LOW"), 100.0)
    structure = _q(row, "Structure Quality")
    breakout = _q(row, "Breakout Quality")
    return score, structure + 0.4 * breakout, trap, _num(row, "Prediction Score", 0.0)


def apply_mode7_ranking(df: pd.DataFrame, market_bias: dict | None = None) -> pd.DataFrame:
    """
    Apply Mode 7 final ranking and trap suppression to a scan dataframe.

    Existing columns are preserved.  Mode 7-specific score columns are added,
    and Final Score / Prediction Score are adjusted only for this Mode 7 pass.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        out = df.copy()

        final = _num_series(out, "Final Score", _num_series(out, "Score", 50.0))
        pred = _num_series(out, "Prediction Score", _num_series(out, "ML %", 50.0))
        score = _num_series(out, "Score", final)
        bt = _num_series(out, "Backtest %", 50.0)
        ml = _num_series(out, "ML %", 50.0)
        sr = _num_series(out, "S&R Structure Score", 50.0)

        structure_q = _quality_series(out, "Structure Quality")
        breakout_q = _quality_series(out, "Breakout Quality")
        support_q = _quality_series(out, "Support Strength")
        volume_q = _quality_series(out, "Volume Confirmation")
        momentum_q = _quality_series(out, "Momentum Continuation")
        resistance_q = _quality_series(out, "Resistance Distance")
        trap_q = _trap_series(out, "Trap Probability")
        channel_score = _num_series(out, "Channel Score", 0.0)
        channel_entry = _text_series(out, "Channel Entry Zone", "NO").eq("YES")
        channel_detected = _text_series(out, "Ascending Channel", "NO").eq("YES")

        structure = (
            0.22 * structure_q
            + 0.18 * breakout_q
            + 0.15 * support_q
            + 0.14 * volume_q
            + 0.10 * momentum_q
            + 0.08 * resistance_q
            + 0.08 * trap_q
            + 0.05 * sr
        )

        mode7_final = (
            0.62 * structure
            + 0.15 * pred
            + 0.10 * final
            + 0.06 * score
            + 0.04 * bt
            + 0.03 * ml
        )
        channel_bonus = pd.Series(0.0, index=out.index, dtype="float64")
        channel_bonus = channel_bonus.mask(
            channel_entry,
            (channel_score * 0.10).clip(lower=4.0, upper=10.0),
        )
        channel_bonus = channel_bonus.mask(
            ~channel_entry & channel_detected,
            (channel_score * 0.04).clip(lower=0.0, upper=4.0),
        )
        mode7_final += channel_bonus

        setup = _text_series(out, "Setup Type", "")
        trap_prob = _text_series(out, "Trap Probability", "LOW")
        trap_risk = _text_series(out, "Trap Risk", "LOW")
        advanced = _text_series(out, "Advanced Trap", "NONE")
        vol = _num_series(out, "Vol / Avg", 1.0)
        de20 = _num_series(out, "Δ vs EMA20 (%)", _num_series(out, "Delta vs EMA20 (%)", 0.0))
        rsi = _num_series(out, "RSI", 50.0)
        higher_lows = np.maximum(
            _num_series(out, "Higher Lows", 0.0),
            _num_series(out, "Channel Higher Lows", 0.0),
        )

        try:
            if isinstance(market_bias, dict):
                bias = str(market_bias.get("bias", "") or "").lower()
                regime = str(market_bias.get("regime", "") or "").lower()
                base_tight = _num_series(out, "Base Tightness (%)", 99.0)
                reg_adj = pd.Series(0.0, index=out.index, dtype="float64")
                # FIX BUG 2: scalar regime flags extracted first so
                # np.where receives a proper per-row boolean Series mask.
                # Old code used (("bullish" in bias) or ...) directly inside
                # np.where — a scalar True/False applied the same adj to ALL
                # rows regardless of their individual setup/vol values.
                is_bullish = bool("bullish" in bias or "trending up" in regime)
                is_bearish = bool("bearish" in bias or "trending down" in regime)
                is_ranging = bool("ranging" in regime or "sideways" in bias)

                if is_bullish:
                    reg_adj += np.where(
                        setup.isin(["BREAKOUT READY", "EARLY BREAKOUT", "MOMENTUM CONTINUATION"]),
                        3.0, 0.0,
                    )
                if is_bearish:
                    reg_adj -= 3.0
                    reg_adj += np.where(vol.lt(1.3), -2.0, 0.0)
                if is_ranging:
                    reg_adj += np.where(
                        setup.isin(["RESISTANCE COMPRESSION", "SUPPORT BOUNCE"]) & base_tight.le(8.0),
                        3.0, 0.0,
                    )
                    reg_adj += np.where(
                        setup.eq("EARLY BREAKOUT") & vol.lt(1.4),
                        -2.0, 0.0,
                    )
                mode7_final += reg_adj.clip(-5.0, 5.0)
        except (TypeError, ValueError):
            debug_log("Mode 7 regime adjustment failed", exc_info=True)

        high_trap = trap_prob.eq("HIGH") | trap_risk.eq("HIGH") | advanced.str.contains("FAKE", regex=False, na=False)
        med_trap = trap_prob.eq("MEDIUM") | trap_risk.eq("MEDIUM")
        negative_setup = setup.isin(MODE7_NEGATIVE_SETUPS)
        weak_support_bounce = setup.eq("SUPPORT BOUNCE") & higher_lows.lt(2.0)

        mode7_final = mode7_final.mask(high_trap, np.minimum(mode7_final - 24.0, 58.0))
        mode7_final = mode7_final.mask(~high_trap & med_trap, mode7_final - 8.0)
        mode7_final = mode7_final.mask(negative_setup, np.minimum(mode7_final - 18.0, 62.0))
        mode7_final = mode7_final.mask(weak_support_bounce, np.minimum(mode7_final - 12.0, 62.0))
        mode7_final = mode7_final.mask(vol.lt(MODE7_VOL_WEAK), mode7_final - 9.0)
        mode7_final = mode7_final.mask(de20.gt(MODE7_EMA_EXTENSION_HARD) | rsi.gt(MODE7_RSI_EXHAUSTION), mode7_final - 10.0)
        mode7_final = mode7_final.clip(0.0, 100.0).round(2)

        pred = pred.mask(high_trap, np.minimum(pred, 44.0))
        pred = pred.mask(~high_trap & med_trap, np.minimum(pred, 62.0))
        pred = pred.mask(negative_setup, np.minimum(pred, 50.0)).clip(0.0, 100.0).round(2)

        priority_structure = structure_q + 0.4 * breakout_q
        rank_score = (0.80 * mode7_final + 0.12 * priority_structure + 0.08 * trap_q).clip(0.0, 100.0).round(2)

        verdict = pd.Series("WATCHLIST CANDIDATE", index=out.index, dtype=object)
        verdict = verdict.mask(high_trap | setup.eq("FAKE BREAKOUT RISK"), "FAKE BREAKOUT RISK")
        verdict = verdict.mask(~high_trap & setup.eq("OVEREXTENDED"), "OVEREXTENDED")
        strong_mask = mode7_final.ge(78.0) & _text_series(out, "Structure Quality", "MEDIUM").eq("HIGH") & ~_text_series(out, "Volume Confirmation", "MEDIUM").eq("LOW")
        verdict = verdict.mask(strong_mask & setup.eq("SUPPORT BOUNCE") & ~weak_support_bounce, "CLEAN SUPPORT BOUNCE")
        verdict = verdict.mask(strong_mask & setup.eq("ASCENDING CHANNEL"), "ASCENDING CHANNEL BUY")
        verdict = verdict.mask(strong_mask & setup.isin(["BREAKOUT READY", "EARLY BREAKOUT"]), "STRONG BREAKOUT")
        verdict = verdict.mask(strong_mask & setup.eq("MOMENTUM CONTINUATION"), "MOMENTUM CONTINUATION")
        valid_mid = (
            ~high_trap
            & ~negative_setup
            & mode7_final.ge(66.0)
            & _text_series(out, "Structure Quality", "MEDIUM").isin(["HIGH", "MEDIUM"])
            & ~_text_series(out, "Breakout Quality", "MEDIUM").eq("LOW")
            & ~_text_series(out, "Support Strength", "MEDIUM").eq("LOW")
        )
        verdict = verdict.mask(valid_mid & setup.isin(MODE7_POSITIVE_SETUPS) & ~weak_support_bounce, setup)
        verdict = verdict.mask(
            ~high_trap & ~negative_setup & setup.eq("ASCENDING CHANNEL") & channel_entry,
            "ASCENDING CHANNEL",
        )
        verdict = verdict.mask(strong_mask & setup.eq("ASCENDING CHANNEL"), "ASCENDING CHANNEL BUY")
        verdict = verdict.mask(
            ~high_trap
            & ~negative_setup
            & ~setup.eq("ASCENDING CHANNEL")
            & _text_series(out, "Volume Confirmation", "MEDIUM").eq("LOW"),
            "WEAK VOLUME BREAKOUT",
        )
        verdict = verdict.mask(~high_trap & ~negative_setup & _text_series(out, "Structure Quality", "MEDIUM").eq("LOW"), "STRUCTURE FAILURE")

        out["Prediction Score"] = pred
        out["Mode7 Final Score"] = mode7_final
        out["Mode7 Rank Score"] = rank_score
        out["Mode7 Verdict"] = verdict
        out["Final Score"] = mode7_final
        out["rank_score"] = rank_score

        out.loc[high_trap, "Signal"] = "AVOID"
        out.loc[high_trap, "Final Signal"] = "TRAP"
        if "Adjusted Signal" in out.columns:
            out.loc[high_trap, "Adjusted Signal"] = "TRAP"
        if "Next-Day Signal" in out.columns:
            out.loc[high_trap, "Next-Day Signal"] = "FAKE BREAKOUT RISK"

        return out.sort_values("Mode7 Rank Score", ascending=False, kind="stable").reset_index(drop=True)
    except (KeyError, TypeError, ValueError) as exc:
        debug_log("Mode 7 ranking fallback activated: %s", exc, exc_info=True)
        return df


__all__ = ["final_score_mode7", "ranking_priority_mode7", "apply_mode7_ranking"]
