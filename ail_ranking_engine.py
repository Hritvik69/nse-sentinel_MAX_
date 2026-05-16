"""
Diversity-aware ranking and role selection for A-I-L IN ONE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ail_meta_score_engine import compute_ail_master_score


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "").replace(",", "")
            if cleaned.lower() in {"", "nan", "none", "null", "-", "n/a", "na"}:
                return default
            value = cleaned
        out = float(value)
        return float(out) if np.isfinite(out) else default
    except Exception:
        return default


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return float(np.clip(float(value), lo, hi))
    except Exception:
        return lo


def _get(row: pd.Series | dict[str, Any], *keys: str) -> Any:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in keys:
        value = getter(key, None)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null", "-", "n/a", "na"}:
            return value
    return None


def _numeric(row: pd.Series | dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = _safe_float(_get(row, key), np.nan)
        if np.isfinite(value):
            return _clip(value)
    return default


def _text(row: pd.Series | dict[str, Any], *keys: str) -> str:
    return str(_get(row, *keys) or "").strip()


def _symbol(row: pd.Series | dict[str, Any]) -> str:
    for key in ("Symbol", "Ticker", "symbol", "ticker", "Stock"):
        value = str(_get(row, key) or "").strip().upper()
        if value:
            return value[:-3] if value.endswith(".NS") else value
    return ""


def _categories(row: pd.Series | dict[str, Any]) -> set[str]:
    raw = _text(row, "AIL Categories", "AIL Category")
    return {part.strip().upper() for part in raw.replace("|", ",").split(",") if part.strip()}


def normalize_cross_mode_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in (
        "AIL Master Score",
        "Smart Potential Score",
        "Bullish Probability",
        "Prediction Score",
        "Final Score",
        "AIL Confidence",
        "Trap Risk Score",
        "Setup Cleanliness",
        "Momentum Quality",
        "Volume Quality",
        "Regime Alignment",
        "AIL Regime Alignment",
        "Risk Reward Score",
        "Sector Support",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    source = None
    for col in ("AIL Master Score", "Smart Potential Score", "Prediction Score", "Final Score"):
        if col in out.columns and out[col].notna().any():
            source = col
            break
    if source:
        values = out[source].astype(float)
        if len(out) > 1 and float(values.max() - values.min()) > 0.1:
            out["AIL Normalized Cross Score"] = 45.0 + (values - values.min()) * 47.0 / (values.max() - values.min())
        else:
            out["AIL Normalized Cross Score"] = values.fillna(0.0)
    else:
        out["AIL Normalized Cross Score"] = 0.0
    return out


def apply_diversity_penalty(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    work = df.copy()
    base = pd.to_numeric(work.get("AIL Master Score", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    work["AIL Ensemble Score Raw"] = base
    work = work.sort_values("AIL Ensemble Score Raw", ascending=False, kind="stable").reset_index(drop=True)

    sector_seen: dict[str, int] = {}
    category_seen: dict[str, int] = {}
    penalties: list[float] = []
    for _, row in work.iterrows():
        penalty = 0.0
        sector = _text(row, "Sector").upper()
        opportunity = _numeric(row, "AIL Opportunity Score", "AIL Speculative Score", default=0.0)
        elite = _numeric(row, "AIL Master Score", "Smart Potential Score", default=0.0) >= 78.0 or opportunity >= 48.0
        if sector:
            seen = sector_seen.get(sector, 0)
            sector_cap = 2.4 if elite else 4.0
            penalty += min(sector_cap, seen * (0.8 if elite else 1.2))
            sector_seen[sector] = seen + 1
        cats = sorted(_categories(row))
        primary = cats[0] if cats else ""
        if primary:
            seen = category_seen.get(primary, 0)
            category_cap = 1.8 if elite else 2.6
            penalty += min(category_cap, seen * (0.55 if elite else 0.85))
            category_seen[primary] = seen + 1
        penalties.append(round(penalty, 2))

    work["AIL Diversity Penalty"] = penalties
    work["AIL Master Score"] = (work["AIL Ensemble Score Raw"] - work["AIL Diversity Penalty"]).clip(lower=0.0, upper=100.0)
    return work.sort_values(
        ["AIL Master Score", "AIL Confidence", "AIL Risk Adjusted Score"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def build_master_rankings(
    df: pd.DataFrame,
    *,
    market_bias: dict[str, Any] | None = None,
    learning_profile: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    scored = df.copy()
    if "AIL Master Score" not in scored.columns:
        scored = compute_ail_master_score(scored, market_bias=market_bias, learning_profile=learning_profile)
    scored = normalize_cross_mode_scores(scored)
    scored = apply_diversity_penalty(scored)
    if not scored.empty:
        scored["AIL Master Rank"] = range(1, len(scored) + 1)
    return scored


def _role_score(row: pd.Series, role: str) -> float:
    master = _numeric(row, "AIL Master Score", default=0.0)
    confidence = _numeric(row, "AIL Confidence", "Smart Confidence", "Confidence", default=0.0)
    trap = _numeric(row, "Trap Risk Score", default=50.0)
    trap_quality = _clip(100.0 - trap)
    setup = _numeric(row, "Setup Cleanliness", "AIL Mode Philosophy Score", default=50.0)
    momentum = _numeric(row, "Momentum Quality", default=50.0)
    volume = _numeric(row, "Volume Quality", default=50.0)
    risk_reward = _numeric(row, "Risk Reward Score", "AIL Risk Adjusted Score", default=50.0)
    regime = _numeric(row, "AIL Regime Alignment", "Regime Alignment", default=50.0)
    sector = _numeric(row, "Sector Support", "Sector Strength", default=50.0)
    cats = _categories(row)
    rsi = _safe_float(_get(row, "RSI"), 50.0)
    early = 72.0 if 46.0 <= rsi <= 60.0 else 50.0
    setup_text = _text(row, "Setup Type", "Entry Timing").upper()
    if "EARLY" in setup_text or "ACCUM" in setup_text or "PULLBACK" in setup_text:
        early += 12.0

    if role == "best_overall":
        return 0.58 * master + 0.18 * confidence + 0.14 * risk_reward + 0.10 * regime
    if role == "safest_candidate":
        return 0.42 * trap_quality + 0.25 * setup + 0.18 * confidence + 0.15 * sector
    if role == "strongest_momentum":
        return 0.42 * momentum + 0.22 * volume + 0.18 * master + 0.10 * trap_quality + 0.08 * confidence
    if role == "best_swing_setup":
        boost = 8.0 if "SWING" in cats else 0.0
        return 0.38 * setup + 0.24 * risk_reward + 0.18 * trap_quality + 0.12 * master + 0.08 * confidence + boost
    if role == "early_accumulation":
        boost = 8.0 if "RELAXED" in cats else 0.0
        return 0.34 * early + 0.23 * setup + 0.18 * trap_quality + 0.13 * sector + 0.12 * master + boost
    if role == "lowest_trap_risk":
        return 0.62 * trap_quality + 0.18 * setup + 0.12 * confidence + 0.08 * sector
    if role == "institutional_setup":
        boost = 9.0 if "INSTITUTIONAL" in cats else 0.0
        return 0.36 * regime + 0.26 * sector + 0.18 * confidence + 0.12 * setup + 0.08 * master + boost
    if role == "high_risk_high_reward":
        moderate_risk = 12.0 if 45.0 <= trap <= 68.0 else (-10.0 if trap > 75.0 else 0.0)
        return 0.36 * risk_reward + 0.24 * momentum + 0.15 * volume + 0.13 * master + 0.12 * confidence + moderate_risk
    return master


def _summary(row: pd.Series | None, role_label: str, role_metric: str, score: float = 0.0) -> dict[str, Any]:
    if row is None:
        return {"label": role_label, "symbol": "", "score": 0.0, "metric": role_metric, "reason": "No candidate available."}
    reason = _text(row, "AIL Reasoning", "Smart Notes", "Battle Notes", "AIL Top3 Drivers") or "Ranked from real orchestration metrics"
    return {
        "label": role_label,
        "symbol": _symbol(row),
        "score": round(_clip(score), 2),
        "metric": role_metric,
        "reason": reason,
        "row": row.to_dict(),
    }


def _select_role(
    df: pd.DataFrame,
    role: str,
    role_label: str,
    used_symbols: dict[str, int],
    *,
    repeat_margin: float = 8.0,
) -> dict[str, Any]:
    if df is None or df.empty:
        return _summary(None, role_label, role)
    scored = df.copy()
    scored["_AIL_ROLE_SCORE"] = scored.apply(lambda row: _role_score(row, role), axis=1)
    scored = scored.sort_values("_AIL_ROLE_SCORE", ascending=False, kind="stable").reset_index(drop=True)
    top = scored.iloc[0]
    top_symbol = _symbol(top)
    chosen = top
    if used_symbols.get(top_symbol, 0) > 0 and len(scored) > 1:
        unused = scored[~scored.apply(lambda row: _symbol(row) in used_symbols, axis=1)]
        if not unused.empty:
            best_unused = unused.iloc[0]
            if float(top["_AIL_ROLE_SCORE"]) - float(best_unused["_AIL_ROLE_SCORE"]) <= repeat_margin:
                chosen = best_unused
    chosen_symbol = _symbol(chosen)
    if chosen_symbol:
        used_symbols[chosen_symbol] = used_symbols.get(chosen_symbol, 0) + 1
    return _summary(chosen, role_label, "AIL Role Score", float(chosen["_AIL_ROLE_SCORE"]))


def select_category_leaders(ranked_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
        return {}
    used: dict[str, int] = {}
    order = [
        ("best_overall", "Best Overall"),
        ("safest_candidate", "Safest Candidate"),
        ("strongest_momentum", "Strongest Momentum"),
        ("best_swing_setup", "Best Swing Setup"),
        ("early_accumulation", "Early Accumulation Leader"),
        ("lowest_trap_risk", "Lowest Trap Risk"),
        ("institutional_setup", "Strongest Institutional Setup"),
        ("high_risk_high_reward", "High Risk High Reward"),
    ]
    return {
        key: _select_role(ranked_df, key, label, used)
        for key, label in order
    }


__all__ = [
    "build_master_rankings",
    "apply_diversity_penalty",
    "select_category_leaders",
    "normalize_cross_mode_scores",
]
