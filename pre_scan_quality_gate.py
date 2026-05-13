"""
Pre-scan quality gate for NSE Sentinel.

The gate is intentionally separate from the calibrated scoring engines. It
blocks structurally bearish buy candidates upstream and caps inflated tomorrow
scores when trend or AI confidence disagrees with the rule-based score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCORE_MIN = 0.0
SCORE_MAX = 100.0
HARD_EMA_STACK_MODES = {1, 5, 6}
HARD_EMA_SLOPE_MODES = {1, 5, 6}


@dataclass
class GateMetrics:
    ema20: float = np.nan
    ema50: float = np.nan
    ema20_slope_5d: float = np.nan
    ret_20d: float = np.nan
    ret_60d: float = np.nan
    vol_ratio: float = np.nan
    rsi: float = np.nan
    data_available: bool = False


@dataclass
class GateDecision:
    blocked: bool = False
    penalty: float = 0.0
    cap: float = SCORE_MAX
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: GateMetrics = field(default_factory=GateMetrics)
    checklist_passes: int = 0
    buy_valid: bool = False
    dead_cat: bool = False

    def add_penalty(self, points: float, reason: str, cap: float | None = None) -> None:
        self.penalty += max(float(points or 0.0), 0.0)
        if reason:
            self.reasons.append(reason)
        if cap is not None:
            self.cap = min(self.cap, float(cap))

    def add_cap(self, cap: float, reason: str) -> None:
        self.cap = min(self.cap, float(cap))
        if reason:
            self.reasons.append(reason)

    def block(self, reason: str) -> None:
        self.blocked = True
        self.cap = min(self.cap, SCORE_MIN)
        if reason:
            self.reasons.append(reason)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "-"}:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _clip_score(value: Any) -> float:
    return float(np.clip(_safe_float(value, 0.0), SCORE_MIN, SCORE_MAX))


def _as_mode_int(mode: Any, fallback: int = 0) -> int:
    try:
        return int(mode)
    except Exception:
        return fallback


def _norm_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _symbol_candidates(value: Any) -> list[str]:
    base = _norm_symbol(value)
    if not base:
        return []
    raw = str(value or "").strip().upper()
    candidates = [raw, base, f"{base}.NS"]
    return list(dict.fromkeys([c for c in candidates if c]))


def _find_symbol(row: pd.Series | dict[str, Any]) -> str:
    for key in ("Symbol", "Ticker", "ticker", "symbol"):
        try:
            value = row.get(key)
        except Exception:
            value = None
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _history_for_symbol(all_data: dict[str, Any] | None, symbol: str) -> pd.DataFrame | None:
    if not isinstance(all_data, dict) or not symbol:
        return None
    for key in _symbol_candidates(symbol):
        hist = all_data.get(key)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            return hist
    return None


def _flatten_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def _numeric_col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        col = lower_map.get(name.lower())
        if col is not None:
            raw = df[col]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            series = pd.to_numeric(raw, errors="coerce").dropna()
            if not series.empty:
                return series
    return None


def _compute_history_metrics(df: pd.DataFrame | None) -> GateMetrics:
    metrics = GateMetrics()
    work = _flatten_ohlcv(df)
    if work is None:
        return metrics

    close = _numeric_col(work, "Close", "Adj Close")
    volume = _numeric_col(work, "Volume")
    if close is None or len(close) < 20:
        return metrics

    close = close.tail(120)
    if volume is not None:
        volume = volume.reindex(close.index).dropna().tail(120)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    metrics.ema20 = _safe_float(ema20.iloc[-1])
    metrics.ema50 = _safe_float(ema50.iloc[-1])
    if len(ema20) >= 6 and _safe_float(ema20.iloc[-6], 0.0) > 0:
        metrics.ema20_slope_5d = ((float(ema20.iloc[-1]) / float(ema20.iloc[-6])) - 1.0) * 100.0
    if len(close) >= 21 and _safe_float(close.iloc[-21], 0.0) > 0:
        metrics.ret_20d = ((float(close.iloc[-1]) / float(close.iloc[-21])) - 1.0) * 100.0
    if len(close) >= 61 and _safe_float(close.iloc[-61], 0.0) > 0:
        metrics.ret_60d = ((float(close.iloc[-1]) / float(close.iloc[-61])) - 1.0) * 100.0
    if volume is not None and len(volume) >= 2:
        avg20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.iloc[:-1].mean())
        if avg20 > 0:
            metrics.vol_ratio = float(volume.iloc[-1]) / avg20

    metrics.data_available = True
    return metrics


def _first_present(row: pd.Series | dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        try:
            value = row.get(name)
        except Exception:
            value = None
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return value
    return default


def _row_metrics(row: pd.Series | dict[str, Any], history_metrics: GateMetrics | None = None) -> GateMetrics:
    metrics = history_metrics or GateMetrics()

    if not np.isfinite(metrics.ema20):
        metrics.ema20 = _safe_float(_first_present(row, ("Gate EMA20", "EMA 20", "EMA20")))
    if not np.isfinite(metrics.ema50):
        metrics.ema50 = _safe_float(_first_present(row, ("Gate EMA50", "EMA 50", "EMA50")))
    if not np.isfinite(metrics.ret_20d):
        metrics.ret_20d = _safe_float(
            _first_present(row, ("Gate 20D Return %", "20D Return (%)", "20D Return", "Ret 20D"))
        )
    if not np.isfinite(metrics.ret_60d):
        metrics.ret_60d = _safe_float(
            _first_present(row, ("Gate 60D Return %", "60D Return (%)", "60D Return", "Ret 60D"))
        )
    if not np.isfinite(metrics.vol_ratio):
        metrics.vol_ratio = _safe_float(
            _first_present(row, ("Gate Vol Ratio", "Vol / Avg", "Vol/Avg", "Volume Ratio", "Vol Avg"))
        )
    if not np.isfinite(metrics.rsi):
        metrics.rsi = _safe_float(_first_present(row, ("RSI",)), np.nan)
    if not np.isfinite(metrics.ema20_slope_5d):
        metrics.ema20_slope_5d = _safe_float(
            _first_present(
                row,
                (
                    "Gate EMA20 Slope 5D %",
                    "EMA20 Slope 5D (%)",
                    "EMA20 Slope (%)",
                    "EMA Slope 5D",
                ),
            )
        )

    return metrics


def _ai_confidence(row: pd.Series | dict[str, Any], preferred_col: str | None = None) -> float:
    names = []
    if preferred_col:
        names.append(preferred_col)
    names.extend(("AI Conf %", "AI Confidence", "Calibrated Confidence", "Confidence"))
    return _safe_float(_first_present(row, names), np.nan)


def _text(row: pd.Series | dict[str, Any], *names: str) -> str:
    value = _first_present(row, names, "")
    return str(value or "").strip()


def _is_buyish(row: pd.Series | dict[str, Any]) -> bool:
    parts = [
        _text(row, name)
        for name in ("AI Action", "Action", "Adjusted Signal", "Final Signal", "Signal", "Next-Day Signal")
    ]
    blob = " ".join(" ".join(parts).split()).lower()
    return any(token in blob for token in ("buy", "bullish", "strong green", "possible up", "green"))


def _regime_key(row: pd.Series | dict[str, Any]) -> str:
    raw = _text(row, "Market Regime", "Regime", "AI Regime", "Market Bias").lower().replace("-", " ")
    if "high" in raw and "vol" in raw:
        return "HIGH_VOLATILITY"
    if "trending" in raw and "down" in raw:
        return "TRENDING_DOWN"
    if "bear" in raw:
        return "TRENDING_DOWN"
    return raw.upper().replace(" ", "_")


def _checklist(metrics: GateMetrics) -> tuple[int, bool]:
    ema_stack_ok = np.isfinite(metrics.ema20) and np.isfinite(metrics.ema50) and metrics.ema20 > metrics.ema50
    slope_ok = np.isfinite(metrics.ema20_slope_5d) and metrics.ema20_slope_5d >= 0.0
    ret60_ok = np.isfinite(metrics.ret_60d) and metrics.ret_60d > -5.0
    vol_ok = np.isfinite(metrics.vol_ratio) and metrics.vol_ratio >= 1.2
    rsi_ok = np.isfinite(metrics.rsi) and 50.0 <= metrics.rsi <= 70.0
    ret20_ok = np.isfinite(metrics.ret_20d) and metrics.ret_20d >= 0.0
    count = int(sum([ema_stack_ok, slope_ok, ret60_ok, vol_ok, rsi_ok, ret20_ok]))
    return count, bool(ema_stack_ok and slope_ok and count >= 4)


def evaluate_quality_gate(
    row: pd.Series | dict[str, Any],
    *,
    history: pd.DataFrame | None = None,
    mode: int = 0,
    ai_conf_col: str | None = None,
) -> GateDecision:
    decision = GateDecision()
    mode_int = _as_mode_int(_first_present(row, ("Mode ID", "Mode"), mode), mode)
    metrics = _row_metrics(row, _compute_history_metrics(history))
    decision.metrics = metrics

    ema_stack_known = np.isfinite(metrics.ema20) and np.isfinite(metrics.ema50)
    ema_stack_ok = bool(ema_stack_known and metrics.ema20 > metrics.ema50)
    slope_known = np.isfinite(metrics.ema20_slope_5d)
    slope_ok = bool(slope_known and metrics.ema20_slope_5d >= 0.0)

    if ema_stack_known and not ema_stack_ok:
        if mode_int in HARD_EMA_STACK_MODES:
            decision.block("G1 EMA20 below EMA50")
        elif mode_int in {2, 3, 4}:
            decision.add_penalty(20.0, "G1 EMA stack bearish", cap=72.0)
    elif mode_int == 4 and ema_stack_ok and metrics.ema50 > 0:
        ema_gap_pct = ((metrics.ema20 / metrics.ema50) - 1.0) * 100.0
        if ema_gap_pct <= 0.3:
            decision.block("Mode 4 EMA20/EMA50 gap <= 0.3%")

    if slope_known:
        mode6_flat_or_down = mode_int == 6 and metrics.ema20_slope_5d <= 0.0
        if metrics.ema20_slope_5d < 0.0 or mode6_flat_or_down:
            if mode_int in HARD_EMA_SLOPE_MODES:
                decision.block("G2 EMA20 slope not rising")
            else:
                decision.add_penalty(10.0, "G2 EMA20 slope falling", cap=78.0)

    if np.isfinite(metrics.ret_60d):
        if mode_int == 1 and metrics.ret_60d <= -10.0:
            decision.block("Mode 1 60D return <= -10%")
        elif mode_int == 5 and metrics.ret_60d <= -15.0:
            decision.block("Mode 5 60D return <= -15%")
        elif metrics.ret_60d <= -20.0:
            decision.block("G3 60D return <= -20%")
        elif metrics.ret_60d < -10.0:
            decision.add_penalty(15.0, "G3 60D return below -10%", cap=68.0)

    if (
        np.isfinite(metrics.ret_60d)
        and np.isfinite(metrics.ret_20d)
        and np.isfinite(metrics.vol_ratio)
        and metrics.ret_60d < -8.0
        and metrics.ret_20d > 0.0
        and metrics.vol_ratio < 1.3
    ):
        decision.dead_cat = True
        decision.add_penalty(12.0, "G4 dead-cat bounce: weak volume confirmation", cap=65.0)

    ai_conf = _ai_confidence(row, ai_conf_col)
    if np.isfinite(ai_conf):
        if ai_conf < 35.0:
            decision.add_cap(58.0, "G5 AI confidence below 35%")
        elif ai_conf < 40.0:
            decision.add_cap(65.0, "G5 AI confidence below 40%")
        elif ai_conf < 50.0:
            decision.add_cap(80.0, "G5 AI confidence below 50%")
        elif ai_conf < 65.0:
            decision.add_cap(90.0, "S1 AI confidence 50-65%")

    if np.isfinite(metrics.ret_20d):
        if mode_int == 2 and metrics.ret_20d < 0.0:
            decision.add_penalty(15.0, "Mode 2 20D return below 0%", cap=68.0)
        if metrics.ret_20d < -20.0:
            decision.block("G6 20D return below -20%")
        elif metrics.ret_20d < -10.0:
            decision.add_penalty(15.0, "G6 20D return below -10%", cap=70.0)
        elif metrics.ret_20d < -5.0:
            decision.add_penalty(8.0, "G6 20D return below -5%")
        elif metrics.ret_20d < 0.0:
            decision.add_penalty(5.0, "S3 20D return negative", cap=82.0)

    if _is_buyish(row):
        regime = _regime_key(row)
        if regime == "TRENDING_DOWN":
            decision.add_penalty(15.0, "S4 bullish pick in TRENDING_DOWN regime")
        if regime == "HIGH_VOLATILITY":
            decision.add_penalty(10.0, "S4 buy pick in HIGH_VOLATILITY regime", cap=65.0)

    sector_strength = _safe_float(_first_present(row, ("Sector Strength", "Sector Score")), np.nan)
    if np.isfinite(sector_strength) and sector_strength < 50.0:
        decision.add_penalty(8.0, "S5 sector strength below 50")

    checklist_count, buy_valid = _checklist(metrics)
    decision.checklist_passes = checklist_count
    decision.buy_valid = buy_valid
    if _is_buyish(row) and not buy_valid:
        decision.add_penalty(6.0, "S2 signal agreement below buy minimum", cap=75.0)

    return decision


def _score_columns(
    df: pd.DataFrame,
    score_col: str = "Final Score",
    tomorrow_col: str = "Tomorrow Score",
) -> list[str]:
    candidates = [
        score_col,
        tomorrow_col,
        "Tomorrow Pick Score",
        "Tomorrow Score",
        "Final Score",
    ]
    return list(dict.fromkeys([col for col in candidates if col and col in df.columns]))


def _apply_decisions(
    df: pd.DataFrame,
    decisions: list[GateDecision],
    *,
    score_col: str,
    tomorrow_col: str,
    drop_blocked: bool,
) -> pd.DataFrame:
    out = df.copy()
    score_cols = _score_columns(out, score_col=score_col, tomorrow_col=tomorrow_col)

    out["Gate Passed"] = [not d.blocked for d in decisions]
    out["Gate Blocked"] = [d.blocked for d in decisions]
    out["Gate Reasons"] = ["; ".join(dict.fromkeys(d.reasons)) for d in decisions]
    out["Gate Penalty"] = [round(float(d.penalty), 2) for d in decisions]
    out["Gate Score Cap"] = [round(float(d.cap), 2) for d in decisions]
    out["Gate Checklist Passes"] = [int(d.checklist_passes) for d in decisions]
    out["Gate Buy Valid"] = [bool(d.buy_valid) for d in decisions]
    out["Dead Cat Bounce"] = ["YES" if d.dead_cat else "NO" for d in decisions]
    out["Gate EMA20"] = [round(float(d.metrics.ema20), 4) if np.isfinite(d.metrics.ema20) else np.nan for d in decisions]
    out["Gate EMA50"] = [round(float(d.metrics.ema50), 4) if np.isfinite(d.metrics.ema50) else np.nan for d in decisions]
    out["Gate EMA20 Slope 5D %"] = [
        round(float(d.metrics.ema20_slope_5d), 4) if np.isfinite(d.metrics.ema20_slope_5d) else np.nan
        for d in decisions
    ]
    out["Gate 20D Return %"] = [
        round(float(d.metrics.ret_20d), 4) if np.isfinite(d.metrics.ret_20d) else np.nan for d in decisions
    ]
    out["Gate 60D Return %"] = [
        round(float(d.metrics.ret_60d), 4) if np.isfinite(d.metrics.ret_60d) else np.nan for d in decisions
    ]
    out["Gate Vol Ratio"] = [
        round(float(d.metrics.vol_ratio), 4) if np.isfinite(d.metrics.vol_ratio) else np.nan for d in decisions
    ]

    for col in score_cols:
        adjusted = []
        for value, decision in zip(out[col].tolist(), decisions):
            if decision.blocked:
                adjusted.append(SCORE_MIN)
                continue
            score = _clip_score(value)
            score = max(SCORE_MIN, score - decision.penalty)
            score = min(score, decision.cap)
            adjusted.append(round(float(np.clip(score, SCORE_MIN, SCORE_MAX)), 2))
        out[col] = adjusted

    blocked_count = int(sum(d.blocked for d in decisions))
    out.attrs["quality_gate"] = {
        "before": int(len(df)),
        "blocked": blocked_count,
        "after": int(len(df) - blocked_count) if drop_blocked else int(len(df)),
    }

    if drop_blocked:
        out = out.loc[~out["Gate Blocked"].astype(bool)].copy()

    sort_col = tomorrow_col if tomorrow_col in out.columns else score_col
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False, kind="stable").reset_index(drop=True)

    return out


def apply_gate_to_scan_df(
    results_df: pd.DataFrame,
    all_data: dict[str, Any] | None,
    *,
    mode: int = 0,
    score_col: str = "Final Score",
    tomorrow_col: str = "Tomorrow Score",
    ai_conf_col: str = "AI Conf %",
    drop_blocked: bool = True,
) -> pd.DataFrame:
    """
    Apply the OHLCV-aware G1-G6 quality gate to a scan result DataFrame.
    """
    if results_df is None or not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return results_df

    decisions: list[GateDecision] = []
    for _, row in results_df.iterrows():
        symbol = _find_symbol(row)
        hist = _history_for_symbol(all_data, symbol)
        row_mode = _as_mode_int(_first_present(row, ("Mode ID",), mode), mode)
        decisions.append(evaluate_quality_gate(row, history=hist, mode=row_mode, ai_conf_col=ai_conf_col))

    return _apply_decisions(
        results_df,
        decisions,
        score_col=score_col,
        tomorrow_col=tomorrow_col,
        drop_blocked=drop_blocked,
    )


def patch_tomorrow_score(
    results_df: pd.DataFrame,
    *,
    mode: int = 0,
    score_col: str = "Final Score",
    tomorrow_col: str = "Tomorrow Pick Score",
    ai_conf_col: str = "AI Confidence",
    drop_blocked: bool = False,
) -> pd.DataFrame:
    """
    Lightweight no-OHLCV patch for score inflation.

    It uses existing scan columns, so it is safe to run after tomorrow AI
    preview data is merged into a Top 3 table.
    """
    if results_df is None or not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return results_df

    decisions: list[GateDecision] = []
    for _, row in results_df.iterrows():
        row_mode = _as_mode_int(_first_present(row, ("Mode ID",), mode), mode)
        decisions.append(evaluate_quality_gate(row, history=None, mode=row_mode, ai_conf_col=ai_conf_col))

    out = _apply_decisions(
        results_df,
        decisions,
        score_col=score_col,
        tomorrow_col=tomorrow_col,
        drop_blocked=drop_blocked,
    )

    signal_col = next((col for col in ("AI Key Signal", "Key Signal") if col in out.columns), None)
    if signal_col and len(out) >= 3 and tomorrow_col in out.columns:
        top3 = out.head(3)
        signals = [str(v or "").strip().lower() for v in top3[signal_col].tolist()]
        signals = [s for s in signals if s and s not in {"-", "nan", "none"}]
        if len(signals) == 3 and len(set(signals)) == 1:
            idx = top3.index
            out.loc[idx, tomorrow_col] = pd.to_numeric(out.loc[idx, tomorrow_col], errors="coerce").fillna(0).clip(upper=75.0)
            out.loc[idx, "Gate Warning"] = "Top 3 share one AI key signal; capped at 75"
            out.attrs["quality_gate_signal_warning"] = "Top 3 share one AI key signal"

    if tomorrow_col in out.columns:
        out = out.sort_values(tomorrow_col, ascending=False, kind="stable").reset_index(drop=True)
    return out


def validate_tomorrow_picks(
    picks_df: pd.DataFrame,
    *,
    min_score: float = 60.0,
    score_col: str = "Tomorrow Pick Score",
    ai_conf_col: str = "AI Confidence",
    max_picks: int | None = 3,
) -> pd.DataFrame:
    """
    Final display validation for Tomorrow's Picks.
    """
    if picks_df is None or not isinstance(picks_df, pd.DataFrame) or picks_df.empty:
        return picks_df

    out = picks_df.copy()
    score = pd.to_numeric(out.get(score_col, pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    ai_conf = pd.to_numeric(out.get(ai_conf_col, pd.Series(np.nan, index=out.index)), errors="coerce")
    ai_action = out.get("AI Action", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    trap = out.get("Trap Check", out.get("Trap Risk", pd.Series("", index=out.index))).fillna("").astype(str).str.upper()
    has_ai_guidance = bool(ai_conf.notna().any() or ai_action.str.strip().ne("").any())

    mask = score.ge(float(min_score))
    if "Gate Blocked" in out.columns:
        mask &= ~out["Gate Blocked"].fillna(False).astype(bool)
    if "Gate Buy Valid" in out.columns:
        known_gate = out["Gate Buy Valid"].notna()
        mask &= (~known_gate) | out["Gate Buy Valid"].astype(bool)
    if has_ai_guidance:
        mask &= ai_action.str.contains("BUY TOMORROW", regex=False, na=False) | ai_conf.ge(55.0)
    mask &= (
        trap.str.contains("CLEAN", regex=False, na=False)
        | trap.str.contains("LOW", regex=False, na=False)
        | trap.eq("")
    )
    mask &= ~trap.str.contains("TRAP|HIGH", regex=True, na=False)

    out = out.loc[mask].copy()
    if score_col in out.columns:
        out = out.sort_values(score_col, ascending=False, kind="stable")
    if max_picks is not None:
        out = out.head(max(1, int(max_picks)))
    return out.reset_index(drop=True)


__all__ = [
    "GateDecision",
    "GateMetrics",
    "apply_gate_to_scan_df",
    "evaluate_quality_gate",
    "patch_tomorrow_score",
    "validate_tomorrow_picks",
]
