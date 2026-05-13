from __future__ import annotations

from collections import Counter
from datetime import date
from functools import cmp_to_key
from pathlib import Path
import math
import re
from typing import Any

import pandas as pd


PROMPT_FILE = Path(__file__).with_name("nse_tomorrow_accuracy_prompt.txt")
LEGACY_PROMPT_FILE = Path(__file__).with_name("nse_sentinel_top3_prompt.txt")

_KEY_RE = re.compile(r"[^a-z0-9]+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

_BLANK_TOKENS = {
    "",
    "-",
    "--",
    "\u2014",
    "\u2013",
    "\u2011",
    "\u20b9",
    "%",
    "\u00b7",
    "\u2022",
    "nan",
    "none",
    "null",
    "?",
}

_TICKER_ALIASES = ["Ticker", "Symbol"]
_BASE_SCORE_ALIASES = ["Final Score", "Prediction Score", "Tomorrow Pick Score", "Breakout Score", "Pulse Score"]
_RSI_ALIASES = ["RSI"]
_VOL_ALIASES = ["Vol / Avg", "Vol/Avg", "Volume Ratio", "Vol Avg", "Gate Vol Ratio"]
_RET_5D_ALIASES = ["5D Return (%)", "5D Return", "5D Return %"]
_RET_20D_ALIASES = ["20D Return (%)", "20D Return", "Gate 20D Return %"]
_DELTA_EMA20_ALIASES = [
    "\u0394 EMA20 (%)",
    "\u0394 vs EMA20 (%)",
    "Delta vs EMA20 (%)",
    "\u00ce\u201d vs EMA20 (%)",
    "Delta EMA20",
    "delta_ema20",
]
_DELTA_20D_HIGH_ALIASES = [
    "\u0394 vs 20D High (%)",
    "\u0394 20D High (%)",
    "Delta vs 20D High (%)",
    "Near High (%)",
]
_BREAKOUT_ZONE_ALIASES = ["Breakout Zone"]
_SECTOR_ALIASES = ["Sector Strength", "Sector Score"]
_REGIME_ALIASES = ["Regime", "AI Regime", "Market Regime", "Market Bias"]
_TRAP_ALIASES = ["Trap Risk", "Trap Check", "Trap", "Bull Trap", "Trap Flags"]
_ACTION_ALIASES = ["Action", "AI Action"]
_CLOSING_ALIASES = ["Closing Strength", "Close vs High", "Candle Quality"]
_HOLD_ALIASES = ["Hold Days", "AI Hold", "Hold"]
_DECISION_ALIASES = ["Decision Score"]
_ENTRY_TIMING_ALIASES = ["Entry Timing"]

_REGIME_RANK = {
    "TRENDING_UP": 0,
    "RANGE_BOUND": 1,
    "HIGH_VOLATILITY": 2,
    "TRENDING_DOWN": 3,
}


def load_top3_prompt_text() -> str:
    for path in (PROMPT_FILE, LEGACY_PROMPT_FILE):
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return (
        "NSE SENTINEL - TOMORROW ACCURACY FILTER PROMPT\n\n"
        "Evaluate only the supplied NSE stock table. Apply hard disqualifiers, "
        "the six next-day readiness checks, the bridged base score, and the "
        "specified post-score penalties. Return exactly Top 3."
    )


def _norm_key(value: object) -> str:
    return _KEY_RE.sub("", str(value or "").lower())


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    if text.lower() in _BLANK_TOKENS:
        return True
    if all(char in "-\u2014\u2013\u2011\u20b9%\u00b7\u2022 " for char in text) and not _NUM_RE.search(text):
        return True
    return not _norm_key(text) and not _NUM_RE.search(text)


def _text(value: object, default: str = "") -> str:
    if _is_blank(value):
        return default
    return str(value).strip()


def _raw_str(value: object) -> str:
    return "" if value is None else str(value).strip()


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


def _find_key(row: dict[str, Any], aliases: list[str]) -> str | None:
    lookup = {_norm_key(key): key for key in row.keys()}
    for alias in aliases:
        norm_alias = _norm_key(alias)
        if norm_alias in lookup:
            return lookup[norm_alias]
    for alias in aliases:
        norm_alias = _norm_key(alias)
        if not norm_alias:
            continue
        for norm_key, original_key in lookup.items():
            if norm_alias == norm_key or norm_alias in norm_key or norm_key in norm_alias:
                return original_key
    return None


def _get(row: dict[str, Any], aliases: list[str], default: object = None) -> object:
    key = _find_key(row, aliases)
    if key is None:
        return default
    return row.get(key, default)


def _has_alias(row: dict[str, Any], aliases: list[str]) -> bool:
    return _find_key(row, aliases) is not None


def _has_value(row: dict[str, Any], aliases: list[str]) -> bool:
    key = _find_key(row, aliases)
    if key is None:
        return False
    return not _is_blank(row.get(key))


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


def _fmt_num(value: float | None, decimals: int = 1, suffix: str = "", missing: str = "N/A") -> str:
    if value is None:
        return missing
    try:
        numeric = float(value)
    except Exception:
        return missing
    if not math.isfinite(numeric):
        return missing
    return f"{numeric:.{decimals}f}{suffix}"


def _risk_level(value: object) -> str:
    text = _text(value).upper()
    if not text:
        return "LOW"
    if any(token in text for token in ("NO TRAP", "NOT HIGH", "CLEAN", "NONE", "SAFE")):
        return "LOW"
    if "HIGH" in text:
        return "HIGH"
    if any(token in text for token in ("MEDIUM", "CAUTION", "WARN", "WEAK")):
        return "MEDIUM"
    if "LOW" in text:
        return "LOW"
    if any(token in text for token in ("TRAP", "RISKY", "YES")):
        return "HIGH"
    return "LOW"


def _action_key(value: object) -> str:
    raw = _raw_str(value)
    if raw == "\U0001F534":
        return "AVOID"
    if raw == "\U0001F7E2":
        return "BUY_TOMORROW"
    text = raw.upper()
    if "AVOID" in text:
        return "AVOID"
    if "BUY TOMORROW" in text:
        return "BUY_TOMORROW"
    return text


def _regime_key(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    compact = _norm_key(text)
    if not text:
        return "RANGE_BOUND"
    if ("HIGH" in text and "VOL" in text) or "highvolatility" in compact or "choppy" in compact:
        return "HIGH_VOLATILITY"
    if "TRENDING_UP" in text or "UPTREND" in compact or ("BULL" in text and "BEAR" not in text):
        return "TRENDING_UP"
    if "TRENDING_DOWN" in text or "DOWNTREND" in compact or "BEAR" in text:
        return "TRENDING_DOWN"
    if "RANGE" in text or "SIDEWAYS" in compact:
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


def _market_regime(market_context: object, fallback_regime: str) -> str:
    if isinstance(market_context, dict):
        for key in ("regime", "market_regime", "Market Regime"):
            value = market_context.get(key)
            if value is not None:
                return _regime_key(value)
    return fallback_regime


def _classify_closing_strength(
    row: dict[str, Any],
    ret_5d: float | None,
    vol: float,
    delta_ema20: float | None,
) -> tuple[str, str]:
    key = _find_key(row, _CLOSING_ALIASES)
    if key is None or _is_blank(row.get(key)):
        strong_proxy = (
            delta_ema20 is not None
            and delta_ema20 > 0
            and ret_5d is not None
            and ret_5d > 0
            and vol >= 1.3
        )
        if strong_proxy:
            return "STRONG", "Proxy used: Delta EMA20 > 0, 5D Return > 0, and Vol/Avg >= 1.3"
        return "NEUTRAL", "Proxy used: closing column absent, so defaulted to NEUTRAL"

    raw_value = row.get(key)
    numeric = _num(raw_value, None)
    text = _text(raw_value).upper()
    norm_key = _norm_key(key)

    if numeric is not None and "closevshigh" in norm_key:
        if 0.0 <= numeric <= 1.0:
            if numeric >= 0.7:
                return "STRONG", f"{key}: {_fmt_num(numeric, 2)} ratio near day high"
            if numeric >= 0.4:
                return "NEUTRAL", f"{key}: {_fmt_num(numeric, 2)} mid-range close"
            return "WEAK", f"{key}: {_fmt_num(numeric, 2)} ratio near day low"
        if -5.0 <= numeric <= 0.0:
            if numeric >= -0.5:
                return "STRONG", f"{key}: {_fmt_num(numeric, 2)}% from day high"
            if numeric >= -1.5:
                return "NEUTRAL", f"{key}: {_fmt_num(numeric, 2)}% from day high"
            return "WEAK", f"{key}: {_fmt_num(numeric, 2)}% from day high"
        if 0.0 <= numeric <= 100.0:
            if numeric >= 75.0:
                return "STRONG", f"{key}: {_fmt_num(numeric, 1)} suggests a strong close"
            if numeric >= 40.0:
                return "NEUTRAL", f"{key}: {_fmt_num(numeric, 1)} suggests a mid-range close"
            return "WEAK", f"{key}: {_fmt_num(numeric, 1)} suggests a weak close"

    if any(token in text for token in ("STRONG", "NEAR HIGH", "NEAR DAY HIGH", "GREEN", "BULL")):
        return "STRONG", f"{key}: {text}"
    if any(token in text for token in ("WEAK", "NEAR LOW", "UPPER WICK", "DISTRIBUTION", "RED")):
        return "WEAK", f"{key}: {text}"
    if any(token in text for token in ("NEUTRAL", "MID", "AVERAGE")):
        return "NEUTRAL", f"{key}: {text}"
    return "NEUTRAL", f"{key}: {text or 'unclassified'}"


def _evaluate_freshness(ret_5d: float | None, ret_5d_present: bool, vol: float) -> dict[str, Any]:
    if not ret_5d_present or ret_5d is None:
        return {"status": "PARTIAL", "points": 6, "detail": f"5D Return N/A | Vol/Avg {_fmt_num(vol, 2)}"}
    if 0.0 <= ret_5d <= 3.0 and vol >= 1.3:
        return {
            "status": "YES",
            "points": 12,
            "detail": f"5D Return {_fmt_num(ret_5d, 2, '%')} | Vol/Avg {_fmt_num(vol, 2)}",
        }
    if 3.0 < ret_5d <= 6.0 and vol >= 1.5:
        return {
            "status": "PARTIAL",
            "points": 6,
            "detail": f"5D Return {_fmt_num(ret_5d, 2, '%')} | Vol/Avg {_fmt_num(vol, 2)}",
        }
    return {"status": "NO", "points": 0, "detail": f"5D Return {_fmt_num(ret_5d, 2, '%')} | Vol/Avg {_fmt_num(vol, 2)}"}


def _evaluate_rsi(rsi: float) -> dict[str, Any]:
    if 52.0 <= rsi <= 60.0:
        return {"status": "YES", "points": 12, "comment": "Sweet spot: momentum confirmed with room left"}
    if 60.0 < rsi <= 65.0:
        return {"status": "YES", "points": 8, "comment": "Healthy momentum and still acceptable headroom"}
    if 65.0 < rsi <= 68.0:
        return {"status": "PARTIAL", "points": 3, "comment": "Momentum is working, but it is getting stretched"}
    if 68.0 < rsi <= 72.0:
        return {"status": "PARTIAL", "points": 0, "comment": "Caution zone: upside room is getting limited"}
    return {"status": "NO", "points": -3, "comment": "Momentum is not confirmed yet"}


def _evaluate_volume(vol: float) -> dict[str, Any]:
    if vol >= 2.5:
        return {"status": "YES", "points": 14}
    if vol >= 1.8:
        return {"status": "YES", "points": 10}
    if vol >= 1.3:
        return {"status": "PARTIAL", "points": 5}
    return {"status": "PARTIAL", "points": 1}


def _evaluate_proximity(row: dict[str, Any], prox_20d: float | None, prox_present: bool) -> dict[str, Any]:
    if prox_present and prox_20d is not None:
        if -2.0 <= prox_20d <= 1.0:
            return {"status": "YES", "points": 12, "detail": f"Delta 20D High {_fmt_num(prox_20d, 2, '%')}"}
        if -4.0 <= prox_20d < -2.0:
            return {"status": "PARTIAL", "points": 6, "detail": f"Delta 20D High {_fmt_num(prox_20d, 2, '%')}"}
        if prox_20d > 1.0:
            return {"status": "PARTIAL", "points": 4, "detail": f"Delta 20D High {_fmt_num(prox_20d, 2, '%')}"}
        return {"status": "NO", "points": 0, "detail": f"Delta 20D High {_fmt_num(prox_20d, 2, '%')}"}

    breakout_zone = _text(_get(row, _BREAKOUT_ZONE_ALIASES, ""))
    if breakout_zone:
        zone = breakout_zone.upper()
        if any(token in zone for token in ("TRIGGER", "AT HIGH", "AT BREAKOUT", "IDEAL")):
            return {"status": "YES", "points": 12, "detail": f"Breakout Zone {breakout_zone}"}
        if any(token in zone for token in ("APPROACH", "NEAR", "SETUP")):
            return {"status": "PARTIAL", "points": 6, "detail": f"Breakout Zone {breakout_zone}"}
        if any(token in zone for token in ("ABOVE", "CONFIRMED", "BREAKOUT")):
            return {"status": "PARTIAL", "points": 4, "detail": f"Breakout Zone {breakout_zone}"}
        return {"status": "NO", "points": 0, "detail": f"Breakout Zone {breakout_zone}"}

    return {"status": "PARTIAL", "points": 6, "detail": "Delta 20D High N/A"}


def _evaluate_sector(
    sector_strength: float | None,
    sector_present: bool,
    regime: str,
    regime_present: bool,
) -> dict[str, Any]:
    if not sector_present and not regime_present:
        return {"status": "PARTIAL", "points": 3}
    if sector_strength is not None and sector_strength >= 65.0 and regime == "TRENDING_UP":
        return {"status": "YES", "points": 10}
    if (sector_strength is not None and sector_strength >= 55.0) or regime == "TRENDING_UP":
        return {"status": "PARTIAL", "points": 5}
    if regime == "RANGE_BOUND":
        return {"status": "PARTIAL", "points": 2}
    if regime == "HIGH_VOLATILITY":
        return {"status": "NO", "points": -5}
    if regime == "TRENDING_DOWN":
        return {"status": "NO", "points": -8}
    return {"status": "NO", "points": 0}


def _confidence_label(score: float) -> str:
    if score >= 80.0:
        return "HIGH"
    if score >= 68.0:
        return "MEDIUM"
    if score >= 55.0:
        return "LOW"
    return "SKIP"


def _tie_value_5d(candidate: dict[str, Any]) -> float:
    value = candidate.get("ret_5d")
    return float(value) if value is not None else 9999.0


def _tie_value_rsi(candidate: dict[str, Any]) -> float:
    value = candidate.get("rsi")
    return abs(float(value) - 56.0) if value is not None else 9999.0


def _candidate_compare(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_score = float(left.get("tomorrow_score", 0.0))
    right_score = float(right.get("tomorrow_score", 0.0))
    if abs(left_score - right_score) > 3.0:
        return -1 if left_score > right_score else 1

    left_5d = _tie_value_5d(left)
    right_5d = _tie_value_5d(right)
    if left_5d != right_5d:
        return -1 if left_5d < right_5d else 1

    left_rsi = _tie_value_rsi(left)
    right_rsi = _tie_value_rsi(right)
    if left_rsi != right_rsi:
        return -1 if left_rsi < right_rsi else 1

    left_vol = float(left.get("vol", 0.0))
    right_vol = float(right.get("vol", 0.0))
    if left_vol != right_vol:
        return -1 if left_vol > right_vol else 1

    left_sector = float(left.get("sector_strength") or -9999.0)
    right_sector = float(right.get("sector_strength") or -9999.0)
    if left_sector != right_sector:
        return -1 if left_sector > right_sector else 1

    left_regime = _REGIME_RANK.get(str(left.get("regime", "RANGE_BOUND")), 9)
    right_regime = _REGIME_RANK.get(str(right.get("regime", "RANGE_BOUND")), 9)
    if left_regime != right_regime:
        return -1 if left_regime < right_regime else 1

    if left_score != right_score:
        return -1 if left_score > right_score else 1
    return 0


def _score_record(row: dict[str, Any]) -> dict[str, Any]:
    ticker = _text(_get(row, _TICKER_ALIASES, ""), "UNKNOWN").upper()

    base_score = float(_num(_get(row, _BASE_SCORE_ALIASES, 55), 55) or 55.0)
    bridged_base = min(base_score * 0.35, 35.0)

    rsi = float(_num(_get(row, _RSI_ALIASES, 50), 50) or 50.0)
    vol = float(_num(_get(row, _VOL_ALIASES, 1), 1) or 1.0)

    ret_5d_present = _has_value(row, _RET_5D_ALIASES)
    ret_5d = _num(_get(row, _RET_5D_ALIASES, None), None)

    ret_20d_present = _has_value(row, _RET_20D_ALIASES)
    ret_20d = _num(_get(row, _RET_20D_ALIASES, None), None)

    delta_ema20_present = _has_value(row, _DELTA_EMA20_ALIASES)
    delta_ema20 = _num(_get(row, _DELTA_EMA20_ALIASES, None), None)

    prox_present = _has_value(row, _DELTA_20D_HIGH_ALIASES)
    prox_20d = _num(_get(row, _DELTA_20D_HIGH_ALIASES, None), None)

    sector_present = _has_value(row, _SECTOR_ALIASES)
    sector_strength = _num(_get(row, _SECTOR_ALIASES, None), None)

    decision_present = _has_value(row, _DECISION_ALIASES)
    decision_score = _num(_get(row, _DECISION_ALIASES, None), None)

    entry_timing = _text(_get(row, _ENTRY_TIMING_ALIASES, "")).upper()
    trap_risk = _risk_level(_get(row, _TRAP_ALIASES, ""))
    action_raw = _raw_str(_get(row, _ACTION_ALIASES, ""))
    action_key = _action_key(action_raw)
    buy_tomorrow_signal = action_key == "BUY_TOMORROW"

    hold_days = _text(_get(row, _HOLD_ALIASES, ""), "1-2 days")
    if hold_days in {"-", "?", "--"}:
        hold_days = "1-2 days"

    regime_present = _has_value(row, _REGIME_ALIASES)
    regime = _regime_key(_get(row, _REGIME_ALIASES, ""))

    eliminated: list[str] = []
    if rsi > 72.0:
        eliminated.append(f"RSI {_fmt_num(rsi, 1)} > 72")
    if vol < 1.0:
        eliminated.append(f"Vol/Avg {_fmt_num(vol, 2)} < 1.0")
    if delta_ema20_present and delta_ema20 is not None and delta_ema20 > 6.0:
        eliminated.append(f"Delta EMA20 {_fmt_num(delta_ema20, 2, '%')} > 6%")
    if ret_5d_present and ret_5d is not None and ret_5d > 11.0:
        eliminated.append(f"5D Return {_fmt_num(ret_5d, 2, '%')} > 11%")
    if trap_risk == "HIGH":
        eliminated.append("Trap Risk HIGH")
    if action_key == "AVOID":
        eliminated.append("Action = Avoid")
    if decision_present and decision_score is not None and decision_score < 45.0:
        eliminated.append(f"Decision Score {_fmt_num(decision_score, 1)} < 45")

    freshness = _evaluate_freshness(ret_5d, ret_5d_present, vol)
    closing_value, closing_reason = _classify_closing_strength(row, ret_5d, vol, delta_ema20)
    closing_status = "YES" if closing_value == "STRONG" else ("NO" if closing_value == "WEAK" else "PARTIAL")
    closing_points = 10 if closing_status == "YES" else (4 if closing_status == "PARTIAL" else -6)
    closing = {"status": closing_status, "points": closing_points, "reason": closing_reason}
    rsi_check = _evaluate_rsi(rsi)
    volume_check = _evaluate_volume(vol)
    proximity = _evaluate_proximity(row, prox_20d, prox_present)
    sector = _evaluate_sector(sector_strength, sector_present, regime, regime_present)

    subtotal = (
        bridged_base
        + freshness["points"]
        + closing["points"]
        + rsi_check["points"]
        + volume_check["points"]
        + proximity["points"]
        + sector["points"]
    )
    capped_subtotal = min(subtotal, 100.0)

    penalties: list[tuple[int, str]] = []
    if ret_5d_present and ret_5d is not None and ret_5d > 9.0:
        penalties.append((10, f"5D Return {_fmt_num(ret_5d, 2, '%')} > 9%"))
    if trap_risk == "MEDIUM":
        penalties.append((8, "Trap Risk = MEDIUM"))
    if rsi > 68.0:
        penalties.append((6, f"RSI {_fmt_num(rsi, 1)} > 68"))
    if delta_ema20_present and delta_ema20 is not None and delta_ema20 > 4.5:
        penalties.append((5, f"Delta EMA20 {_fmt_num(delta_ema20, 2, '%')} > 4.5%"))
    if ret_20d_present and ret_20d is not None and ret_20d < 0.0:
        penalties.append((4, f"20D Return {_fmt_num(ret_20d, 2, '%')} < 0"))
    if entry_timing == "LATE":
        penalties.append((4, 'Entry Timing = "LATE"'))
    if vol < 1.2:
        penalties.append((3, f"Vol/Avg {_fmt_num(vol, 2)} < 1.2"))
    if regime == "HIGH_VOLATILITY":
        penalties.append((2, "Regime = HIGH_VOLATILITY"))

    penalty_total = sum(points for points, _ in penalties)
    tomorrow_score = max(0.0, capped_subtotal - penalty_total)

    return {
        "ticker": ticker,
        "row": row,
        "base_score": base_score,
        "bridged_base": bridged_base,
        "rsi": rsi,
        "vol": vol,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "delta_ema20": delta_ema20,
        "delta_20d_high": prox_20d,
        "sector_strength": sector_strength,
        "regime": regime,
        "hold_days": hold_days,
        "trap_risk": trap_risk,
        "entry_timing": entry_timing,
        "action": action_raw,
        "buy_tomorrow_signal": buy_tomorrow_signal,
        "checks": {
            "freshness": freshness,
            "closing": closing,
            "rsi": rsi_check,
            "volume": volume_check,
            "proximity": proximity,
            "sector": sector,
        },
        "subtotal": subtotal,
        "capped_subtotal": capped_subtotal,
        "penalties": penalties,
        "penalty_total": penalty_total,
        "tomorrow_score": tomorrow_score,
        "confidence": _confidence_label(tomorrow_score),
        "qualified": tomorrow_score >= 55.0,
        "eliminated": eliminated,
    }


def _driver_signals(candidate: dict[str, Any]) -> list[str]:
    signals: list[tuple[float, str]] = []
    checks = candidate.get("checks", {})

    volume = checks.get("volume", {})
    if volume.get("points", 0) > 0:
        signals.append((float(volume["points"]), f"volume is active at {_fmt_num(candidate.get('vol'), 2)}x"))

    proximity = checks.get("proximity", {})
    if proximity.get("points", 0) > 0:
        prox = candidate.get("delta_20d_high")
        if prox is not None:
            signals.append((float(proximity["points"]), f"price is {_fmt_num(prox, 2, '%')} from the 20D high trigger"))
        else:
            signals.append((float(proximity["points"]), "the setup is already sitting in the breakout zone"))

    freshness = checks.get("freshness", {})
    if freshness.get("points", 0) > 0:
        ret_5d = candidate.get("ret_5d")
        if ret_5d is not None:
            signals.append((float(freshness["points"]), f"the move is still fresh with only {_fmt_num(ret_5d, 2, '%')} over 5D"))

    rsi_check = checks.get("rsi", {})
    if rsi_check.get("points", 0) >= 8:
        signals.append((float(rsi_check["points"]), f"RSI {_fmt_num(candidate.get('rsi'), 1)} still has room to run"))

    closing = checks.get("closing", {})
    if closing.get("points", 0) > 0:
        signals.append((float(closing["points"]), "buyers held the close instead of distributing into strength"))

    sector = checks.get("sector", {})
    if sector.get("points", 0) > 0 and candidate.get("regime") == "TRENDING_UP":
        signals.append((float(sector["points"]), "the broader tape is giving it a sector tailwind"))

    if candidate.get("buy_tomorrow_signal"):
        signals.append((1.0, "the scan itself still flags it as a buy-tomorrow candidate"))

    signals.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, text in signals]


def _fuel_comment(candidate: dict[str, Any]) -> str:
    ret_5d = candidate.get("ret_5d")
    rsi = candidate.get("rsi")
    delta_ema20 = candidate.get("delta_ema20")
    if ret_5d is not None and ret_5d <= 3.0:
        return "the move is still early rather than exhausted"
    if rsi is not None and 52.0 <= rsi <= 65.0:
        return "RSI still has enough headroom for one more follow-through day"
    if delta_ema20 is not None and delta_ema20 <= 4.5:
        return "the stock is not yet stretched too far above EMA20"
    return "there is still enough room for continuation if the market stays supportive"


def _why_tomorrow(candidate: dict[str, Any]) -> str:
    signals = _driver_signals(candidate)
    first = signals[0] if signals else "the base setup remains constructive"
    second = signals[1] if len(signals) > 1 else "buyers did not fully exhaust the move today"
    return f"{first.capitalize()} and {second}, while {_fuel_comment(candidate)}."


def _penalty_text(candidate: dict[str, Any]) -> str:
    penalties = list(candidate.get("penalties", []))
    if not penalties:
        return "NONE"
    return " | ".join(f"-{points} {reason}" for points, reason in penalties)


def _selected_top3(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qualified = [candidate for candidate in ranked if candidate.get("qualified")]
    if len(qualified) >= 3:
        return qualified[:3]
    extras = [candidate for candidate in ranked if not candidate.get("qualified")]
    return (qualified + extras)[:3]


def rank_top3_from_rows(
    rows: object,
    *,
    as_of: date | None = None,
    market_context: object = None,
) -> dict[str, Any]:
    source_rows = _records_from_rows(rows)
    scored = [_score_record(row) for row in source_rows if _text(_get(row, _TICKER_ALIASES, ""))]
    hard_eliminated = [candidate for candidate in scored if candidate["eliminated"]]
    score_pool = [candidate for candidate in scored if not candidate["eliminated"]]
    ranked = sorted(score_pool, key=cmp_to_key(_candidate_compare))
    selected = _selected_top3(ranked)

    regime_counts = Counter(candidate.get("regime", "RANGE_BOUND") for candidate in score_pool if candidate.get("regime"))
    fallback_regime = regime_counts.most_common(1)[0][0] if regime_counts else "RANGE_BOUND"
    current_regime = _market_regime(market_context, fallback_regime)

    payload: dict[str, Any] = {
        "date": (as_of or date.today()).isoformat(),
        "evaluated": len(scored),
        "eliminated": len(hard_eliminated),
        "scored": len(score_pool),
        "skipped": sum(1 for candidate in score_pool if not candidate.get("qualified")),
        "top": selected,
        "ranked": ranked,
        "all_scored": scored,
        "eliminated_rows": hard_eliminated,
        "regime": _pretty_regime(current_regime),
        "bias": _market_bias(market_context, current_regime),
    }
    payload["text"] = format_top3_output(payload)
    return payload


def format_top3_output(result: dict[str, Any]) -> str:
    separator = "-" * 55
    lines: list[str] = [
        separator,
        f"TOMORROW'S TOP 3 - {result.get('date', date.today().isoformat())}",
        "Target: Stocks most likely to close HIGHER tomorrow",
        separator,
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
                f"Skipped (score < 55): {result.get('skipped', 0)}",
                "",
                "DISCLAIMER: Quantitative model output only. Not financial advice.",
                "Always verify with live market data before taking any position.",
            ]
        )
        return "\n".join(lines).strip()

    for rank, candidate in enumerate(top, start=1):
        checks = candidate.get("checks", {})
        lines.extend(
            [
                separator,
                f"#{rank}  {candidate['ticker']:<18} Tomorrow Score: {candidate['tomorrow_score']:.1f} / 100",
                f"    Confidence: {candidate.get('confidence', 'SKIP')}",
                separator,
                f"Base (bridged):   {candidate['base_score']:.1f} x 0.35 = {candidate['bridged_base']:.1f}",
                f"Check 1 Freshness:     {checks['freshness']['status']} -> {checks['freshness']['points']:+d} pts",
                f"  {checks['freshness']['detail']}",
                f"Check 2 Closing:       {checks['closing']['status']} -> {checks['closing']['points']:+d} pts",
                f"  {checks['closing']['reason']}",
                f"Check 3 RSI:           {checks['rsi']['status']} -> {checks['rsi']['points']:+d} pts",
                f"  RSI {_fmt_num(candidate.get('rsi'), 1)} -> {checks['rsi']['comment']}",
                f"Check 4 Volume:        {checks['volume']['status']} -> {checks['volume']['points']:+d} pts",
                f"  Vol/Avg {_fmt_num(candidate.get('vol'), 2)}",
                f"Check 5 Proximity:     {checks['proximity']['status']} -> {checks['proximity']['points']:+d} pts",
                f"  {checks['proximity']['detail']}",
                f"Check 6 Sector:        {checks['sector']['status']} -> {checks['sector']['points']:+d} pts",
                f"  Sector {_fmt_num(candidate.get('sector_strength'), 1)} | Regime {candidate.get('regime', 'RANGE_BOUND')}",
                f"Subtotal (before penalties): {candidate['subtotal']:.1f}",
                f"Penalties: {_penalty_text(candidate)}",
                f"TOMORROW SCORE: {candidate['tomorrow_score']:.1f}",
                "",
                "Why this stock goes up tomorrow (ONE specific sentence):",
                f"  {_why_tomorrow(candidate)}",
                "",
                f"Suggested hold: {candidate.get('hold_days', '1-2 days')}",
                "",
            ]
        )

    eliminated_rows = list(result.get("eliminated_rows", []) or [])
    if eliminated_rows:
        eliminated_text = " | ".join(
            f"{candidate['ticker']} ({', '.join(candidate.get('eliminated', []))})"
            for candidate in eliminated_rows[:12]
        )
        if len(eliminated_rows) > 12:
            eliminated_text += f" | +{len(eliminated_rows) - 12} more"
    else:
        eliminated_text = "NONE"

    warnings: list[str] = []
    top_only = list(result.get("top", []) or [])
    late_entries = [candidate for candidate in top_only if candidate.get("ret_5d") is not None and candidate["ret_5d"] > 7.0]
    if late_entries:
        warnings.append(
            "WARNING: "
            + ", ".join(f"{candidate['ticker']} {_fmt_num(candidate['ret_5d'], 2, '%')}" for candidate in late_entries)
            + " - late entry risk"
        )
    stretched_rsi = [candidate for candidate in top_only if candidate.get("rsi") is not None and candidate["rsi"] > 67.0]
    if stretched_rsi:
        warnings.append(
            "WARNING: "
            + ", ".join(f"{candidate['ticker']} RSI {_fmt_num(candidate['rsi'], 1)}" for candidate in stretched_rsi)
            + " - limited upside room"
        )
    if any(candidate.get("regime") == "HIGH_VOLATILITY" for candidate in top_only):
        warnings.append("WARNING: HIGH_VOLATILITY regime - lower overall reliability")

    lines.extend(
        [
            separator,
            "MARKET CONTEXT",
            separator,
            f"Regime: {result.get('regime', 'RANGE_BOUND')}  |  Bias: {result.get('bias', 'Sideways')}",
            f"Evaluated: {result.get('evaluated', 0)}  |  Eliminated: {result.get('eliminated', 0)}  |  Scored: {result.get('scored', 0)}",
            f"Skipped (score < 55): {result.get('skipped', 0)}",
            f"Eliminated: {eliminated_text}",
            "",
            "ACCURACY WARNINGS:",
        ]
    )
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("NONE")
    lines.extend(
        [
            "",
            "DISCLAIMER: Quantitative model output only. Not financial advice.",
            "Always verify with live market data before taking any position.",
        ]
    )
    return "\n".join(lines).strip()
