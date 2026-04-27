from __future__ import annotations


def apply_trade_decision_simple(df):

    if df is None or df.empty:
        return df

    actions = []
    holds = []

    for _, row in df.iterrows():

        rsi = float(row.get("RSI", 50))
        vol = float(row.get("Vol / Avg", 1))
        pred = float(row.get("Prediction Score", 50))
        ema = float(row.get("Δ vs EMA20 (%)", 0))
        trap = str(row.get("Trap Risk", "")).upper()

        action = "🟡 Watch"
        hold = "—"

        if vol < 1.0 or "HIGH" in trap or pred < 45:
            action = "🔴 Avoid"

        elif rsi > 70 or ema > 5:
            action = "🔵 Wait"

        elif 52 <= rsi <= 68 and vol >= 1.3 and pred >= 55:
            action = "🟢 Buy Tomorrow"

            if vol >= 1.8:
                hold = "3–5 Days"
            else:
                hold = "2–4 Days"

        actions.append(action)
        holds.append(hold)

    df["Action"] = actions
    df["Hold Days"] = holds

    return df


def _safe_float(value, default):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"nan", "none"} or text in {"-", "—"}:
                return default
            for token in ("%", "x", "X", "×", ","):
                text = text.replace(token, "")
            value = text
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


def apply_trade_decision_simple_any(df):

    if df is None or df.empty:
        return df

    actions = []
    holds = []

    for _, row in df.iterrows():

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
                ["Δ vs EMA20 (%)", "Î” vs EMA20 (%)", "Δ EMA20 (%)", "Î” EMA20 (%)"],
                0,
            ),
            0,
        )
        trap = _first_present(row, ["Trap Risk", "Trap Flags", "Bull Trap", "Trap"], "")

        action = "🟡 Watch"
        hold = "—"

        if vol < 1.0 or _is_high_trap(trap) or pred < 45:
            action = "🔴 Avoid"

        elif rsi > 70 or ema > 5:
            action = "🔵 Wait"

        elif 52 <= rsi <= 68 and vol >= 1.3 and pred >= 55:
            action = "🟢 Buy Tomorrow"

            if vol >= 1.8:
                hold = "3–5 Days"
            else:
                hold = "2–4 Days"

        actions.append(action)
        holds.append(hold)

    df["Action"] = actions
    df["Hold Days"] = holds

    return df
