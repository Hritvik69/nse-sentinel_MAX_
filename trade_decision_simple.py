from __future__ import annotations

import re

_STRIP_RE = re.compile(r"[%xX×,]")

def _safe_float(value, default):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none"} or text in {"-", "—"}:
                return default
            value = _STRIP_RE.sub("", text)
        return float(value)
    except Exception:
        return default


def _first_present(row, candidates, default=None):
    for column in candidates:
        try:
            value = row.get(column, None)
        except Exception:
            value = None
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _is_high_trap(trap_value):
    trap_text = str(trap_value or "").strip().upper()
    if not trap_text:
        return False
    if trap_text in {"NO", "NONE", "LOW", "SAFE", "CLEAN"}:
        return False
    if "NO TRAP" in trap_text or "LOW" in trap_text:
        return False
    return any(flag in trap_text for flag in ("HIGH", "TRAP", "RISKY", "YES"))


def _decision_payload(row):
    rsi = _safe_float(_first_present(row, ["RSI"], 50), 50)
    vol = _safe_float(_first_present(row, ["Vol / Avg", "Volume Ratio"], 1), 1)
    pred = _safe_float(
        _first_present(
            row,
            ["Prediction Score", "Next Day Prob", "Tomorrow Pick Score", "Final Score"],
            50,
        ),
        50,
    )
    ema = _safe_float(
        _first_present(
            row,
            ["Δ vs EMA20 (%)", "Δ EMA20 (%)", "delta_ema20"],
            0,
        ),
        0,
    )
    trap = _first_present(row, ["Trap Check", "Trap Risk", "Trap Flags", "Bull Trap", "Trap"], "")
    meta_prob = _safe_float(_first_present(row, ["Meta Prob", "meta_model_output"], pred), pred)
    calibrated_conf = _safe_float(
        _first_present(row, ["Calibrated Confidence", "Confidence", "Final Score"], pred),
        pred,
    )
    regime_fit = _safe_float(_first_present(row, ["Regime Fit", "regime_fit"], 55), 55)
    accuracy_hist = _safe_float(
        _first_present(row, ["Accuracy History", "sector_accuracy", "Historical Win %"], 55),
        55,
    )
    decision_score = max(
        0.0,
        min(
            100.0,
            (
                0.26 * pred
                + 0.24 * calibrated_conf
                + 0.18 * meta_prob
                + 0.16 * regime_fit
                + 0.16 * accuracy_hist
            ),
        ),
    )
    why = (
        f"Meta {meta_prob:.1f}% | Calibrated {calibrated_conf:.1f}% | "
        f"Regime fit {regime_fit:.1f}% | Accuracy history {accuracy_hist:.1f}%"
    )
    return {
        "rsi": rsi,
        "vol": vol,
        "pred": pred,
        "ema": ema,
        "trap": trap,
        "meta_prob": meta_prob,
        "calibrated_conf": calibrated_conf,
        "regime_fit": regime_fit,
        "accuracy_hist": accuracy_hist,
        "decision_score": decision_score,
        "why": why,
    }


def _decide_action(payload):
    rsi = payload["rsi"]
    vol = payload["vol"]
    ema = payload["ema"]
    trap = payload["trap"]
    meta_prob = payload["meta_prob"]
    calibrated_conf = payload["calibrated_conf"]
    regime_fit = payload["regime_fit"]
    accuracy_hist = payload["accuracy_hist"]
    decision_score = payload["decision_score"]

    action = "🟡 Watch"
    hold = "—"
    if vol < 1.0 or _is_high_trap(trap) or decision_score < 45:
        action = "🔴 Avoid"
    elif rsi > 72 or ema > 5.5 or calibrated_conf < 50 or meta_prob < 48:
        action = "🔵 Wait"
    elif 50 <= rsi <= 68 and vol >= 1.2 and decision_score >= 60 and calibrated_conf >= 56 and meta_prob >= 54:
        action = "🟢 Buy Tomorrow"
        if vol >= 1.8 and regime_fit >= 60 and accuracy_hist >= 58:
            hold = "3–5 Days"
        else:
            hold = "2–4 Days"
    elif decision_score < 52:
        action = "🔵 Wait"
    return action, hold


def apply_trade_decision_simple(df):
    if df is None or df.empty:
        return df

    actions = []
    holds = []

    for _, row in df.iterrows():
        payload = _decision_payload(row)
        action, hold = _decide_action(payload)
        actions.append(action)
        holds.append(hold)

    df["Action"] = actions
    df["Hold Days"] = holds
    return df


def apply_trade_decision_simple_any(df):
    if df is None or df.empty:
        return df

    actions = []
    holds = []
    decision_scores = []
    decision_reasons = []

    for _, row in df.iterrows():
        payload = _decision_payload(row)
        action, hold = _decide_action(payload)
        actions.append(action)
        holds.append(hold)
        decision_scores.append(round(payload["decision_score"], 1))
        decision_reasons.append(payload["why"])

    df["Action"] = actions
    df["Hold Days"] = holds
    df["Decision Score"] = decision_scores
    df["Why This Stock?"] = decision_reasons
    return df
