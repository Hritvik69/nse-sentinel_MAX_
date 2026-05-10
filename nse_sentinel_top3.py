from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
import math
import re
from typing import Any

import pandas as pd


PROMPT_FILE = Path(__file__).with_name("nse_sentinel_top3_prompt.txt")

_KEY_RE = re.compile(r"[^a-z0-9]+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_top3_prompt_text() -> str:
    try:
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return (
            "NSE SENTINEL - MASTER STOCK SELECTION PROMPT\n\n"
            "Evaluate only the supplied NSE stock table. Apply hard "
            "disqualifiers, weighted composite scoring, quality bonuses, "
            "risk penalties, and regime overlays. Return only Top 3 picks."
        )


def _norm_key(value: object) -> str:
    return _KEY_RE.sub("", str(value or "").lower())


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    if text.lower() in {"nan", "none", "null", "?", "--"}:
        return True
    if not text.strip("- "):
        return True
    return not _norm_key(text) and not _NUM_RE.search(text)


def _text(value: object, default: str = "") -> str:
    if _is_blank(value):
        return default
    return str(value).strip()


def _raw_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _row_dict(row: object) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    try:
        if hasattr(row, "to_dict"):
            raw = row.to_dict()
            return dict(raw) if isinstance(raw, dict) else {}
    except Exception:
        return {}
    return {}


def _records_from_rows(rows: object) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return [_row_dict(row) for _, row in rows.iterrows()]
    if isinstance(rows, (list, tuple)):
        return [_row_dict(row) for row in rows]
    return []


def _get(row: dict[str, Any], aliases: list[str], default: object = None) -> object:
    lookup = {_norm_key(key): key for key in row.keys()}
    for alias in aliases:
        norm_alias = _norm_key(alias)
        if norm_alias in lookup:
            return row.get(lookup[norm_alias], default)
    for alias in aliases:
        norm_alias = _norm_key(alias)
        if not norm_alias:
            continue
        for norm_key, original_key in lookup.items():
            if norm_alias == norm_key or norm_alias in norm_key or norm_key in norm_alias:
                return row.get(original_key, default)
    return default


def _num(value: object, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            return numeric if math.isfinite(numeric) else default
        text = str(value).replace(",", "").strip()
        if _is_blank(text):
            return default
        match = _NUM_RE.search(text)
        if not match:
            return default
        numeric = float(match.group(0))
        return numeric if math.isfinite(numeric) else default
    except Exception:
        return default


def _level_text(value: object) -> str:
    text = _text(value).upper()
    if not text:
        return "LOW"
    if "NO TRAP" in text or "CLEAN" in text or "NONE" in text or "SAFE" in text:
        return "LOW"
    if "LOW" in text:
        return "LOW"
    if "HIGH" in text:
        return "HIGH"
    if "TRAP" in text or "RISKY" in text or "YES" in text:
        return "HIGH"
    if "MEDIUM" in text or "CAUTION" in text or "WARN" in text:
        return "MEDIUM"
    if "WEAK" in text:
        return "MEDIUM"
    return "LOW"


def _contains_any(value: object, needles: tuple[str, ...]) -> bool:
    raw = str(value or "")
    raw_lower = raw.lower()
    for needle in needles:
        if needle in raw or needle.lower() in raw_lower:
            return True
    return False


def _regime_key(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    compact = _norm_key(text)
    if not text:
        return "RANGE_BOUND"
    if ("HIGH" in text and "VOL" in text) or "highvolatility" in compact or "choppy" in compact:
        return "HIGH_VOLATILITY"
    if "TRENDING_UP" in text or "uptrend" in compact or ("BULL" in text and "BEAR" not in text):
        return "TRENDING_UP"
    if "TRENDING_DOWN" in text or "downtrend" in compact or "BEAR" in text:
        return "TRENDING_DOWN"
    if "RANGE" in text or "sideways" in compact:
        return "RANGE_BOUND"
    return "RANGE_BOUND"


def _pretty_regime(key: str) -> str:
    return {
        "TRENDING_UP": "TRENDING_UP",
        "TRENDING_DOWN": "TRENDING_DOWN",
        "RANGE_BOUND": "RANGE_BOUND",
        "HIGH_VOLATILITY": "HIGH_VOLATILITY",
    }.get(key, key or "RANGE_BOUND")


def _market_bias(market_context: object, fallback_regime: str) -> str:
    if isinstance(market_context, dict):
        for key in ("bias", "direction", "market_bias", "Market Bias"):
            value = _text(market_context.get(key))
            if value:
                lower = value.lower()
                if "bull" in lower or "up" in lower:
                    return "Bullish"
                if "bear" in lower or "down" in lower:
                    return "Bearish"
                return "Sideways"
    if fallback_regime == "TRENDING_UP":
        return "Bullish"
    if fallback_regime == "TRENDING_DOWN":
        return "Bearish"
    return "Sideways"


def _score_record(row: dict[str, Any]) -> dict[str, Any]:
    ticker = _text(_get(row, ["Ticker", "Symbol"], ""), "UNKNOWN").upper()

    pred = float(
        _num(
            _get(row, ["Prediction Score", "Pred Score", "Next Day Prob", "Tomorrow Pick Score", "Final Score"], 50),
            50,
        )
        or 50
    )
    conf = float(_num(_get(row, ["Calibrated Confidence", "Confidence", "AI Confidence"], pred), pred) or pred)
    meta = float(
        _num(_get(row, ["Meta Prob", "meta_model_output", "Learned Prob %", "AI Confidence"], pred), pred) or pred
    )
    regime_fit = float(_num(_get(row, ["Regime Fit", "regime_fit"], 55), 55) or 55)
    accuracy = float(
        _num(_get(row, ["Accuracy History", "Historical Win %", "sector_accuracy", "System Accuracy"], 55), 55) or 55
    )

    rsi = float(_num(_get(row, ["RSI"], 50), 50) or 50)
    vol = float(_num(_get(row, ["Vol / Avg", "Volume Ratio", "Vol Avg"], 1), 1) or 1)
    delta_ema20 = float(
        _num(_get(row, ["Delta vs EMA20 (%)", "Δ vs EMA20 (%)", "vs EMA20", "Delta EMA20", "delta_ema20"], 0), 0)
        or 0
    )
    ret_5d = float(_num(_get(row, ["5D Return (%)", "5D Return"], 0), 0) or 0)
    ret_20d = float(_num(_get(row, ["20D Return (%)", "20D Return"], 0), 0) or 0)

    ema20 = _num(_get(row, ["EMA 20", "EMA20"], None), None)
    ema50 = _num(_get(row, ["EMA 50", "EMA50"], None), None)
    price = _num(_get(row, ["Price", "Price Rs", "Close", "Last Price"], None), None)
    sector_strength = _num(_get(row, ["Sector Strength"], None), None)
    decision_gate = _num(_get(row, ["Decision Score"], None), None)

    setup_quality = _text(_get(row, ["Setup Quality"], "")).upper()
    entry_timing = _text(_get(row, ["Entry Timing"], "")).upper()
    volume_trend = _text(_get(row, ["Volume Trend"], "")).upper()
    trap_risk = _level_text(_get(row, ["Trap Risk", "Trap Check", "Trap", "Bull Trap"], ""))
    action = _raw_str(_get(row, ["Action"], ""))
    ai_action = _raw_str(_get(row, ["AI Action"], ""))

    hold_days = _text(_get(row, ["Hold Days", "AI Hold", "Hold"], ""), "1-3 days")
    if hold_days in {"-", "?", "--"}:
        hold_days = "1-3 days"

    raw_regime = _get(row, ["Regime", "AI Regime", "Market Regime", "Market Bias"], "")
    regime = _regime_key(raw_regime)

    composite = (0.26 * pred) + (0.24 * conf) + (0.18 * meta) + (0.16 * regime_fit) + (0.16 * accuracy)

    bonuses: list[tuple[int, str]] = []
    penalties: list[tuple[int, str]] = []
    overlay_penalties: list[tuple[int, str]] = []
    eliminated: list[str] = []

    if trap_risk == "HIGH":
        eliminated.append("Trap Risk HIGH")
    if vol < 1.0:
        eliminated.append("Vol/Avg below 1.0")
    if decision_gate is not None and decision_gate < 45:
        eliminated.append("Decision Score below 45")
    if rsi > 75:
        eliminated.append("RSI above 75")
    if _contains_any(action, ("avoid", "🔴")) or _contains_any(ai_action, ("avoid", "🔴")):
        eliminated.append("Action Avoid")

    if 52 <= rsi <= 65:
        bonuses.append((5, "RSI 52-65 sweet spot"))
    if vol >= 1.8 and regime_fit >= 60:
        bonuses.append((4, f"Vol/Avg {vol:.2f}x with regime fit {regime_fit:.0f}%"))
    if ema20 is not None and ema50 is not None and price is not None:
        if ema20 > ema50 and price > ema20:
            bonuses.append((4, "EMA20 > EMA50 and price above EMA20"))
    elif delta_ema20 > 0:
        bonuses.append((2, f"Price above EMA20 proxy (delta {delta_ema20:.1f}%)"))
    if setup_quality == "HIGH":
        bonuses.append((3, "HIGH setup quality"))
    if entry_timing == "EARLY":
        bonuses.append((3, "EARLY entry timing"))
    if volume_trend == "STRONG":
        bonuses.append((2, "STRONG volume trend"))
    if sector_strength is not None and sector_strength >= 65:
        bonuses.append((2, f"Sector strength {sector_strength:.0f}"))
    if ret_20d > 3:
        bonuses.append((2, f"20D return {ret_20d:.1f}% above 3%"))
    if accuracy >= 60:
        bonuses.append((2, f"Accuracy history {accuracy:.1f}%"))
    if _contains_any(action, ("buy tomorrow", "🟢")) or _contains_any(ai_action, ("buy tomorrow", "🟢")):
        bonuses.append((1, "Buy Tomorrow signal"))

    if trap_risk == "HIGH":
        penalties.append((8, "Trap Risk HIGH failsafe"))
    if trap_risk == "MEDIUM":
        penalties.append((5, "Trap Risk MEDIUM"))
    if rsi > 70:
        penalties.append((5, f"RSI {rsi:.1f} above 70"))
    if vol < 1.2:
        penalties.append((4, f"Vol/Avg {vol:.2f} below 1.2"))
    if delta_ema20 > 6:
        penalties.append((4, f"Delta EMA20 {delta_ema20:.1f}% overextended"))
    if entry_timing == "LATE":
        penalties.append((4, "LATE entry timing"))
    if ret_5d > 9:
        penalties.append((3, f"5D return {ret_5d:.1f}% possible exhaustion"))
    if setup_quality == "LOW":
        penalties.append((3, "LOW setup quality"))
    if regime == "HIGH_VOLATILITY":
        penalties.append((2, "HIGH_VOLATILITY regime"))
    if ret_20d < 0:
        penalties.append((2, f"20D return {ret_20d:.1f}% negative"))

    score_cap = 100.0
    if regime == "TRENDING_UP":
        score_cap = 95.0
    elif regime == "TRENDING_DOWN":
        score_cap = 90.0
        if vol < 1.4:
            overlay_penalties.append((10, f"TRENDING_DOWN Vol/Avg {vol:.2f} below 1.4"))
    elif regime == "RANGE_BOUND":
        score_cap = 72.0
        if delta_ema20 >= 3:
            overlay_penalties.append((5, f"RANGE_BOUND Delta EMA20 {delta_ema20:.1f}% >= 3%"))
    elif regime == "HIGH_VOLATILITY":
        score_cap = 62.0
        overlay_penalties.append((5, "HIGH_VOLATILITY base overlay"))
        if composite < 68:
            overlay_penalties.append((8, f"HIGH_VOLATILITY composite {composite:.1f} below 68"))

    total_bonus = sum(points for points, _ in bonuses)
    total_penalty = sum(points for points, _ in penalties) + sum(points for points, _ in overlay_penalties)
    final_score = max(0.0, min(score_cap, composite + total_bonus - total_penalty))

    return {
        "ticker": ticker,
        "row": row,
        "pred": pred,
        "conf": conf,
        "meta": meta,
        "regime_fit": regime_fit,
        "accuracy": accuracy,
        "rsi": rsi,
        "vol": vol,
        "delta_ema20": delta_ema20,
        "ret_20d": ret_20d,
        "sector_strength": sector_strength if sector_strength is not None else 0.0,
        "composite": composite,
        "final_score": final_score,
        "score_cap": score_cap,
        "bonuses": bonuses,
        "penalties": penalties,
        "overlay_penalties": overlay_penalties,
        "eliminated": eliminated,
        "regime": regime,
        "hold_days": hold_days,
        "action": action or ai_action,
        "trap_risk": trap_risk,
    }


def _conviction_factors(candidate: dict[str, Any]) -> str:
    top_bonuses = sorted(candidate.get("bonuses", []), key=lambda item: item[0], reverse=True)
    factors = [label for _, label in top_bonuses[:3]]
    if not factors:
        factors = [
            f"Pred {candidate['pred']:.1f}%",
            f"Vol/Avg {candidate['vol']:.2f}x",
            f"RSI {candidate['rsi']:.1f}",
        ]
    return "; ".join(factors[:3])


def _risk_flags(candidate: dict[str, Any]) -> str:
    flags = [label for _, label in candidate.get("penalties", [])]
    flags.extend(label for _, label in candidate.get("overlay_penalties", []))
    return "; ".join(flags[:5]) if flags else "NONE"


def _points_list(items: list[tuple[int, str]], sign: str = "+") -> str:
    if not items:
        return "NONE"
    return " | ".join(f"{sign}{points} {label}" for points, label in items)


def _overlay_text(candidate: dict[str, Any]) -> str:
    overlay_penalties = list(candidate.get("overlay_penalties", []) or [])
    cap = float(candidate.get("score_cap", 100) or 100)
    if not overlay_penalties:
        return f"{candidate.get('regime', 'RANGE_BOUND')} - no overlay penalty; ceiling {cap:.0f}"
    return (
        f"{candidate.get('regime', 'RANGE_BOUND')} - "
        + " | ".join(f"-{points} {label}" for points, label in overlay_penalties)
        + f"; ceiling {cap:.0f}"
    )


def _why_this(rank: int, candidate: dict[str, Any], others_below: list[dict[str, Any]]) -> str:
    if candidate.get("bonuses"):
        _points, top_signal = max(candidate["bonuses"], key=lambda item: item[0])
    else:
        top_signal = "composite strength"

    if others_below:
        next_candidate = others_below[0]
        margin = round(candidate["final_score"] - next_candidate["final_score"], 1)
        verb = "Leads" if margin >= 0 else "Selected over"
        return (
            f"{verb} #{rank + 1} {next_candidate['ticker']} by {abs(margin):.1f} pts; "
            f"{top_signal.lower()} is the decisive edge after risk adjustments."
        )
    return f"Clears remaining pool; {top_signal.lower()} keeps it inside the Top 3."


def _sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        -float(candidate.get("final_score", 0.0)),
        -float(candidate.get("vol", 0.0)),
        abs(float(candidate.get("rsi", 50.0)) - 55.0),
        -float(candidate.get("accuracy", 0.0)),
        0.0 if candidate.get("regime") == "TRENDING_UP" else 1.0,
        -float(candidate.get("sector_strength", 0.0)),
    )


def rank_top3_from_rows(
    rows: object,
    *,
    as_of: date | None = None,
    market_context: object = None,
) -> dict[str, Any]:
    source_rows = _records_from_rows(rows)
    scored = [_score_record(row) for row in source_rows if _text(_get(row, ["Ticker", "Symbol"], ""))]
    hard_eliminated = [candidate for candidate in scored if candidate["eliminated"]]
    score_pool = [candidate for candidate in scored if not candidate["eliminated"]]
    ranked = sorted(score_pool, key=_sort_key)

    regime_counts = Counter(candidate.get("regime", "RANGE_BOUND") for candidate in score_pool if candidate.get("regime"))
    current_regime = regime_counts.most_common(1)[0][0] if regime_counts else "RANGE_BOUND"

    payload: dict[str, Any] = {
        "date": (as_of or date.today()).isoformat(),
        "evaluated": len(scored),
        "eliminated": len(hard_eliminated),
        "scored": len(score_pool),
        "top": ranked[:3],
        "all_scored": scored,
        "eliminated_rows": hard_eliminated,
        "regime": _pretty_regime(current_regime),
        "bias": _market_bias(market_context, current_regime),
    }
    payload["text"] = format_top3_output(payload)
    return payload


def format_top3_output(result: dict[str, Any]) -> str:
    separator = "-" * 51
    lines: list[str] = [
        f"TOP 3 PICKS - {result.get('date', date.today().isoformat())}",
        "",
    ]
    top = list(result.get("top", []) or [])

    if not top:
        lines.extend(
            [
                "No valid candidates after hard disqualifiers.",
                "",
                "MARKET CONTEXT",
                f"Regime: {result.get('regime', 'RANGE_BOUND')} | Bias: {result.get('bias', 'Sideways')}",
                f"Evaluated: {result.get('evaluated', 0)} | Eliminated: {result.get('eliminated', 0)} | Scored: {result.get('scored', 0)}",
            ]
        )
        return "\n".join(lines)

    for rank, candidate in enumerate(top, start=1):
        others_below = top[rank:]
        score_cap = float(candidate.get("score_cap", 100) or 100)
        lines.extend(
            [
                separator,
                f"#{rank} {candidate['ticker']:<28} Score: {candidate['final_score']:.1f} / {score_cap:.0f}",
                separator,
                (
                    f"Breakdown: Pred {candidate['pred']:.1f}% | Conf {candidate['conf']:.1f}% | "
                    f"Meta {candidate['meta']:.1f}% | Regime fit {candidate['regime_fit']:.1f}% | "
                    f"Acc {candidate['accuracy']:.1f}%"
                ),
                f"Composite (before adj): {candidate['composite']:.1f}",
                f"Bonuses: {_points_list(candidate.get('bonuses', []), '+')}",
                f"Penalties: {_points_list(candidate.get('penalties', []), '-')}",
                f"Overlay: {_overlay_text(candidate)}",
                f"Risk flags: {_risk_flags(candidate)}",
                f"Conviction factors: {_conviction_factors(candidate)}",
                f"Suggested hold: {candidate.get('hold_days', '1-3 days')}",
                f"Why ranked here: {_why_this(rank, candidate, others_below)}",
                "",
            ]
        )

    eliminated_rows = list(result.get("eliminated_rows", []) or [])
    if eliminated_rows:
        eliminated_text = "; ".join(
            f"{candidate['ticker']} ({', '.join(candidate.get('eliminated', []))})"
            for candidate in eliminated_rows[:12]
        )
        if len(eliminated_rows) > 12:
            eliminated_text += f"; +{len(eliminated_rows) - 12} more"
    else:
        eliminated_text = "NONE"

    lines.extend(
        [
            separator,
            "MARKET CONTEXT",
            f"Regime: {result.get('regime', 'RANGE_BOUND')} | Bias: {result.get('bias', 'Sideways')}",
            f"Evaluated: {result.get('evaluated', 0)} | Eliminated: {result.get('eliminated', 0)} | Scored: {result.get('scored', 0)}",
            f"Eliminated: {eliminated_text}",
            "",
            "WARNING: Quantitative model output only. Not financial advice.",
            "Verify with live market data before taking any position.",
        ]
    )
    return "\n".join(lines).strip()
