"""
A-I-L IN ONE orchestration engine.

This module is intentionally UI-light and dependency-injected.  The Streamlit
app passes its existing scan/enrichment functions in, so this layer coordinates
the current NSE Sentinel engines instead of duplicating scanner logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from ail_calibration_engine import apply_confidence_calibration, compute_confidence_calibration
from ail_confidence_health import preserve_high_conviction, preserve_speculative_conviction
from ail_confidence_engine import compute_smart_confidence
from ail_conflict_engine import apply_conflict_penalties
from ail_health_engine import compute_orchestration_health
from ail_learning_engine import build_ail_learning_profile, learning_profile_table
from ail_market_state_engine import apply_market_state_adjustments, detect_market_state
from ail_opportunity_engine import preserve_high_upside_candidates
from ail_penalty_guard import cap_total_penalty, prevent_confidence_collapse
from ail_philosophy_guard import preserve_mode_identity
from ail_pipeline_cache import cache_orchestration_features
from ail_ranking_engine import build_master_rankings, select_category_leaders
from ail_reasoning_engine import apply_orchestration_reasoning
from ail_regime_orchestrator import apply_regime_preference, compute_regime_strategy_bias
from strategy_engines.mode_helpers import resolve_mode_id
from strategy_engines.mode_registry import get_mode_label, get_mode_metadata, get_mode_name


AIL_MODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
AIL_CATEGORY_ORDER: tuple[str, ...] = (
    "Relaxed",
    "Intraday",
    "Momentum",
    "Swing",
    "Breakout",
    "Institutional",
    "Multi-Mode Leaders",
)
AIL_MODE_CATEGORY_MAP: dict[int, tuple[str, ...]] = {
    1: ("Momentum",),
    3: ("Relaxed",),
    4: ("Institutional",),
    5: ("Intraday",),
    6: ("Swing",),
    7: ("Momentum",),
}


@dataclass
class AILPipelineResult:
    started_at: str
    elapsed_sec: float = 0.0
    requested_tickers: int = 0
    modes_scanned: list[int] = field(default_factory=list)
    preload_stats: dict[str, Any] = field(default_factory=dict)
    market_bias: dict[str, Any] = field(default_factory=dict)
    market_state: dict[str, Any] = field(default_factory=dict)
    mode_summaries: list[dict[str, Any]] = field(default_factory=list)
    mode_frames: dict[int, pd.DataFrame] = field(default_factory=dict)
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    categories: dict[str, pd.DataFrame] = field(default_factory=dict)
    category_top3: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_ranked_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    aura_verdicts: list[dict[str, Any]] = field(default_factory=list)
    sector_strength: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_warnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence_meter: dict[str, Any] = field(default_factory=dict)
    learning_insights: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "-"}:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _plain_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def _row_symbol(row: pd.Series | dict[str, Any]) -> str:
    for key in ("Symbol", "Ticker", "ticker", "symbol", "Stock", "stock"):
        try:
            symbol = _plain_symbol(row.get(key))
        except Exception:
            symbol = ""
        if symbol:
            return symbol
    return ""


def _first_text(row: pd.Series | dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        try:
            value = row.get(key)
        except Exception:
            value = None
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return str(value).strip()
    return default


def _find_numeric(row: pd.Series | dict[str, Any], *keys: str, default: float = 0.0, contains: tuple[str, ...] = ()) -> float:
    for key in keys:
        try:
            value = row.get(key)
        except Exception:
            value = None
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}:
            return _safe_float(value, default)

    if contains:
        try:
            items = row.items()
        except Exception:
            items = []
        tokens = tuple(token.lower() for token in contains)
        for col, value in items:
            name = str(col).lower()
            if all(token in name for token in tokens):
                return _safe_float(value, default)
    return default


def _best_sort_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "AIL Master Score",
        "Smart Potential Score",
        "AIL Top3 Consensus Score",
        "AIL Top3 Score",
        "AIL Calibrated Confidence",
        "AIL Confidence",
        "AIL Opportunity Score",
        "AIL Philosophy Score",
        "Prediction Score",
        "Final Score",
        "Confidence",
        "ML %",
        "Backtest %",
        "Score",
    ]
    return [col for col in preferred if col in df.columns]


def _sort_existing_scores(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    cols = _best_sort_columns(out)
    if not cols:
        return out.reset_index(drop=True)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(cols, ascending=[ascending] * len(cols), kind="stable").reset_index(drop=True)


def _dedupe_best_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["_AIL_SYMBOL_KEY"] = work.apply(_row_symbol, axis=1)
    work = work.loc[work["_AIL_SYMBOL_KEY"].astype(str).str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame()
    work = _sort_existing_scores(work)
    return work.drop_duplicates("_AIL_SYMBOL_KEY", keep="first").drop(columns=["_AIL_SYMBOL_KEY"], errors="ignore").reset_index(drop=True)


def _normalize_tickers(tickers: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in tickers or []:
        symbol = str(raw or "").strip().upper()
        if not symbol:
            continue
        if symbol.endswith(".NS"):
            symbol = symbol[:-3]
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _mode_categories(mode_id: int) -> list[str]:
    mode_id = int(mode_id or 0)
    return list(AIL_MODE_CATEGORY_MAP.get(mode_id, ()))


def _is_breakout_row(row: pd.Series | dict[str, Any]) -> bool:
    setup_text = " ".join(
        [
            _first_text(row, ("Setup Type", "Mode7 Verdict", "Setup", "Compare Tags")),
            _first_text(row, ("Channel Entry Zone", "Breakout Quality", "Volume Confirmation")),
        ]
    ).upper()
    if "BREAKOUT" in setup_text or "CHANNEL ENTRY" in setup_text:
        return True
    dist_20d = _find_numeric(row, "Delta vs 20D High (%)", "Δ vs 20D High (%)", default=-99.0, contains=("20d", "high"))
    vol_ratio = _find_numeric(row, "Vol / Avg", default=1.0, contains=("vol", "avg"))
    rsi = _find_numeric(row, "RSI", default=50.0)
    return -3.5 <= dist_20d <= 2.5 and vol_ratio >= 1.2 and 48.0 <= rsi <= 72.0


def _add_mode_and_sector_columns(df: pd.DataFrame, mode: int) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    mode_values = out["Mode ID"] if "Mode ID" in out.columns else pd.Series(mode, index=out.index)
    out["Mode ID"] = pd.to_numeric(mode_values, errors="coerce").fillna(mode).astype(int)
    out["Mode Name"] = get_mode_name(mode)
    out["Mode Label"] = get_mode_label(mode)
    try:
        from sector_master import get_sector

        out["Sector"] = [
            str(get_sector(_row_symbol(row)) or "UNMAPPED")
            for _, row in out.iterrows()
        ]
    except Exception:
        if "Sector" not in out.columns:
            out["Sector"] = "UNMAPPED"
    return out


def _apply_pipeline_enrichment(
    raw_results: list[dict[str, Any]],
    mode: int,
    *,
    enhance_results_fn: Callable[[list[dict[str, Any]], int], pd.DataFrame] | None,
    apply_enhanced_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_universal_grading_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase4_logic_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase42_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_gate_to_scan_df_fn: Callable[..., pd.DataFrame] | None = None,
    market_bias: dict[str, Any] | None = None,
    all_data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if not raw_results:
        return pd.DataFrame()

    if not callable(enhance_results_fn):
        raise ValueError("A-I-L requires the existing enhance_results function.")

    df = enhance_results_fn(raw_results, mode)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    for fn, args in (
        (apply_enhanced_logic_fn, ()),
        (apply_universal_grading_fn, (market_bias,)),
        (apply_phase4_logic_fn, (market_bias,)),
    ):
        if callable(fn):
            try:
                df = fn(df, *args) if args else fn(df)
            except Exception:
                pass

    try:
        from trade_decision_simple import apply_trade_decision_simple

        df = apply_trade_decision_simple(df)
    except Exception:
        pass

    try:
        from learning_engine import batch_predict_success

        df["Learned Prob %"] = batch_predict_success(df)
    except Exception:
        pass

    if callable(apply_phase42_logic_fn):
        try:
            df = apply_phase42_logic_fn(df)
        except Exception:
            pass

    if int(mode) == 7:
        try:
            from strategy_engines.mode7_ranking import apply_mode7_ranking

            df = apply_mode7_ranking(df, market_bias)
        except Exception:
            pass

    if callable(apply_gate_to_scan_df_fn):
        try:
            df = apply_gate_to_scan_df_fn(
                df,
                all_data or {},
                mode=mode,
                score_col="Final Score",
                tomorrow_col="Prediction Score" if "Prediction Score" in df.columns else "Final Score",
                ai_conf_col="AI Confidence",
                drop_blocked=False,
            )
        except Exception:
            pass

    return _add_mode_and_sector_columns(df, mode)


def classify_scan_results(scan_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, list[dict[str, Any]]] = {category: [] for category in AIL_CATEGORY_ORDER}
    if scan_df is None or not isinstance(scan_df, pd.DataFrame) or scan_df.empty:
        return {category: pd.DataFrame() for category in AIL_CATEGORY_ORDER}

    work = scan_df.copy()
    work["_AIL_SYMBOL_KEY"] = work.apply(_row_symbol, axis=1)
    symbol_modes = (
        work.loc[work["_AIL_SYMBOL_KEY"].astype(str).str.len() > 0]
        .groupby("_AIL_SYMBOL_KEY")["Mode ID"]
        .agg(lambda values: sorted({int(v) for v in pd.to_numeric(values, errors="coerce").dropna()}))
        .to_dict()
    )

    for _, row in work.iterrows():
        symbol = _row_symbol(row)
        if not symbol:
            continue
        mode_id = resolve_mode_id(row.get("Mode ID"), None)
        mode_id = int(mode_id or 0)
        row_dict = row.drop(labels=["_AIL_SYMBOL_KEY"], errors="ignore").to_dict()
        row_dict["AIL Modes Matched"] = ", ".join(get_mode_name(m) for m in symbol_modes.get(symbol, []))
        row_dict["AIL Mode Count"] = len(symbol_modes.get(symbol, []))
        for category in _mode_categories(mode_id):
            groups.setdefault(category, []).append({**row_dict, "AIL Category": category})
        if _is_breakout_row(row):
            groups.setdefault("Breakout", []).append({**row_dict, "AIL Category": "Breakout"})
        if len(symbol_modes.get(symbol, [])) >= 2:
            groups.setdefault("Multi-Mode Leaders", []).append({**row_dict, "AIL Category": "Multi-Mode Leaders"})

    out: dict[str, pd.DataFrame] = {}
    for category in AIL_CATEGORY_ORDER:
        frame = pd.DataFrame(groups.get(category, []))
        out[category] = _dedupe_best_by_symbol(frame) if not frame.empty else pd.DataFrame()
    return out


def extract_top_candidates(
    categories: dict[str, pd.DataFrame],
    *,
    market_bias: dict[str, Any] | None = None,
    top_n: int = 3,
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    try:
        from nse_sentinel_top3 import rank_top3_from_rows
    except Exception:
        rank_top3_from_rows = None  # type: ignore[assignment]

    for category, frame in categories.items():
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            outputs[category] = {"top": [], "ranked": [], "evaluated": 0, "scored": 0, "eliminated": 0}
            continue

        source = _dedupe_best_by_symbol(frame)
        if callable(rank_top3_from_rows):
            try:
                payload = rank_top3_from_rows(source, market_context=market_bias)
            except Exception:
                payload = {}
        else:
            payload = {}

        normal_ranked = _normal_mode_top3_ranked(source)
        top_rows = _select_consensus_top3_rows(
            source,
            category=category,
            payload=payload,
            normal_ranked=normal_ranked,
            top_n=top_n,
            market_bias=market_bias,
        )

        payload = dict(payload)
        payload["top_rows"] = top_rows
        payload["top_df"] = pd.DataFrame(top_rows)
        payload["normal_scored"] = int(len(normal_ranked)) if isinstance(normal_ranked, pd.DataFrame) else 0
        payload["selection_method"] = "AIL consensus top3: normal screener + tomorrow accuracy + scanner conviction"
        outputs[category] = payload
    return outputs


def _candidate_symbol(candidate: dict[str, Any]) -> str:
    symbol = str(candidate.get("ticker", "") or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    if symbol:
        return symbol
    row = candidate.get("row", {})
    return _row_symbol(row) if isinstance(row, dict) else ""


def _candidate_maps(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    ranks: dict[str, int] = {}
    if not isinstance(payload, dict):
        return by_symbol, ranks
    for rank, candidate in enumerate(list(payload.get("ranked", []) or []), start=1):
        if not isinstance(candidate, dict):
            continue
        symbol = _candidate_symbol(candidate)
        if not symbol:
            continue
        ranks.setdefault(symbol, rank)
        by_symbol.setdefault(symbol, candidate)
    for candidate in list(payload.get("all_scored", []) or []) + list(payload.get("eliminated_rows", []) or []):
        if not isinstance(candidate, dict):
            continue
        symbol = _candidate_symbol(candidate)
        if symbol:
            by_symbol.setdefault(symbol, candidate)
    return by_symbol, ranks


def _normal_mode_top3_ranked(source: pd.DataFrame) -> pd.DataFrame:
    if source is None or not isinstance(source, pd.DataFrame) or source.empty:
        return pd.DataFrame()
    try:
        from strategy_engines._engine_utils import get_tomorrow_top_picks

        limit = max(3, int(len(source)))
        ranked = get_tomorrow_top_picks(source.copy(), source="main", top_n=limit)
    except Exception:
        return pd.DataFrame()
    return ranked if isinstance(ranked, pd.DataFrame) else pd.DataFrame()


def _normal_rank_maps(normal_ranked: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows: dict[str, dict[str, Any]] = {}
    ranks: dict[str, int] = {}
    if normal_ranked is None or not isinstance(normal_ranked, pd.DataFrame) or normal_ranked.empty:
        return rows, ranks
    for rank, (_, row) in enumerate(normal_ranked.iterrows(), start=1):
        symbol = _row_symbol(row)
        if not symbol:
            continue
        rows.setdefault(symbol, row.to_dict())
        ranks.setdefault(symbol, rank)
    return rows, ranks


def _existing_scanner_conviction(row: pd.Series | dict[str, Any]) -> float:
    weights = (
        ("Prediction Score", 0.34),
        ("Final Score", 0.30),
        ("Backtest %", 0.12),
        ("ML %", 0.12),
        ("Confidence", 0.12),
    )
    values: list[tuple[float, float]] = []
    for col, weight in weights:
        value = _find_numeric(row, col, default=np.nan)
        if np.isfinite(value):
            values.append((_safe_float(value, 0.0), weight))
    if not values:
        return 50.0
    return float(np.clip(sum(value * weight for value, weight in values) / sum(weight for _, weight in values), 0.0, 100.0))


def _blend_top3_scores(
    *,
    normal_score: float,
    tomorrow_score: float,
    scanner_score: float,
    prompt_candidate: dict[str, Any] | None,
    row: pd.Series | dict[str, Any],
) -> float:
    parts: list[tuple[float, float]] = []
    if np.isfinite(normal_score):
        parts.append((normal_score, 0.46))
    if np.isfinite(tomorrow_score):
        parts.append((tomorrow_score, 0.34))
    parts.append((scanner_score, 0.20))
    score = sum(value * weight for value, weight in parts) / sum(weight for _, weight in parts)

    eliminated = list((prompt_candidate or {}).get("eliminated", []) or [])
    if eliminated:
        score -= min(12.0, 6.0 + len(eliminated) * 2.0)
    elif prompt_candidate and not bool(prompt_candidate.get("qualified", False)):
        score -= 3.0

    mode_count = _find_numeric(row, "AIL Mode Count", default=1.0)
    if mode_count > 1:
        score += min(4.0, (mode_count - 1.0) * 1.4)
    return float(np.clip(score, 0.0, 100.0))


def _top3_source_label(normal_rank: int | None, tomorrow_rank: int | None, top_n: int) -> str:
    normal_hit = normal_rank is not None and normal_rank <= top_n
    tomorrow_hit = tomorrow_rank is not None and tomorrow_rank <= top_n
    if normal_hit and tomorrow_hit:
        return "Consensus"
    if normal_hit:
        return "Normal mode screener"
    if tomorrow_hit:
        return "Tomorrow accuracy"
    return "AIL score bridge"


def _top3_selection_notes(
    *,
    normal_rank: int | None,
    tomorrow_rank: int | None,
    normal_score: float,
    tomorrow_score: float,
    prompt_candidate: dict[str, Any] | None,
) -> str:
    notes: list[str] = []
    if normal_rank is not None:
        notes.append(f"normal screener rank {normal_rank}")
    if tomorrow_rank is not None:
        notes.append(f"tomorrow accuracy rank {tomorrow_rank}")
    if np.isfinite(normal_score):
        notes.append(f"normal score {normal_score:.1f}")
    if np.isfinite(tomorrow_score):
        notes.append(f"tomorrow score {tomorrow_score:.1f}")
    eliminated = list((prompt_candidate or {}).get("eliminated", []) or [])
    if eliminated:
        notes.append("tomorrow caution: " + ", ".join(str(item) for item in eliminated[:2]))
    return "; ".join(notes[:5])


def _select_consensus_top3_rows(
    source: pd.DataFrame,
    *,
    category: str,
    payload: dict[str, Any],
    normal_ranked: pd.DataFrame,
    top_n: int,
    market_bias: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    prompt_map, prompt_ranks = _candidate_maps(payload)
    normal_map, normal_ranks = _normal_rank_maps(normal_ranked)
    candidates: list[dict[str, Any]] = []

    for _, row in source.iterrows():
        symbol = _row_symbol(row)
        if not symbol:
            continue
        row_dict = row.to_dict()
        prompt_candidate = prompt_map.get(symbol, {})
        normal_row = normal_map.get(symbol, {})
        normal_score = _find_numeric(normal_row, "Tomorrow Pick Score", "Prediction Score", "Final Score", default=np.nan)
        tomorrow_score = _safe_float(prompt_candidate.get("tomorrow_score"), np.nan) if prompt_candidate else np.nan
        scanner_score = _existing_scanner_conviction(row)
        consensus_score = _blend_top3_scores(
            normal_score=normal_score,
            tomorrow_score=tomorrow_score,
            scanner_score=scanner_score,
            prompt_candidate=prompt_candidate,
            row=row,
        )
        normal_rank = normal_ranks.get(symbol)
        tomorrow_rank = prompt_ranks.get(symbol)
        confidence = compute_smart_confidence(row_dict, {"market_bias": market_bias or {}})
        candidate_conf = str(prompt_candidate.get("confidence", "") or "").strip()
        if not candidate_conf or candidate_conf.lower() == "fallback":
            candidate_conf = str(confidence.get("label", "Insufficient evidence") or "Insufficient evidence")
        penalties = list(prompt_candidate.get("penalties", []) or []) if prompt_candidate else []
        drivers = _candidate_driver_text(prompt_candidate) if prompt_candidate else []
        normal_reason = str(normal_row.get("Tomorrow Pick Reason", "") or "").strip()
        if normal_reason:
            drivers.insert(0, "normal: " + normal_reason)

        row_dict["AIL Category"] = category
        row_dict["AIL Top3 Score"] = round(consensus_score, 2)
        row_dict["AIL Top3 Consensus Score"] = round(consensus_score, 2)
        row_dict["AIL Top3 Normal Score"] = round(normal_score, 2) if np.isfinite(normal_score) else np.nan
        row_dict["AIL Top3 Tomorrow Score"] = round(tomorrow_score, 2) if np.isfinite(tomorrow_score) else np.nan
        row_dict["AIL Top3 Scanner Score"] = round(scanner_score, 2)
        row_dict["AIL Top3 Source"] = _top3_source_label(normal_rank, tomorrow_rank, top_n)
        row_dict["AIL Top3 Normal Rank"] = normal_rank if normal_rank is not None else np.nan
        row_dict["AIL Top3 Tomorrow Rank"] = tomorrow_rank if tomorrow_rank is not None else np.nan
        row_dict["AIL Top3 Qualified"] = bool(prompt_candidate.get("qualified", consensus_score >= 55.0)) and not bool(prompt_candidate.get("eliminated", []))
        row_dict["AIL Top3 Prompt Eliminated"] = bool(prompt_candidate.get("eliminated", []))
        row_dict["AIL Top3 Penalties"] = "; ".join(str(reason) for _, reason in penalties)
        row_dict["AIL Top3 Drivers"] = " | ".join(drivers[:5]) or str(confidence.get("drivers", "")) or "Ranked by normal screener and existing score columns"
        row_dict["AIL Top3 Selection Notes"] = _top3_selection_notes(
            normal_rank=normal_rank,
            tomorrow_rank=tomorrow_rank,
            normal_score=normal_score,
            tomorrow_score=tomorrow_score,
            prompt_candidate=prompt_candidate,
        )
        row_dict["AIL Top3 Confidence"] = candidate_conf
        row_dict["AIL Confidence"] = confidence.get("score", 0.0)
        row_dict["AIL Confidence Label"] = confidence.get("label", "")
        row_dict["AIL Confidence Drivers"] = confidence.get("drivers", "")
        row_dict["_AIL_CONSENSUS_SCORE"] = consensus_score
        candidates.append(row_dict)

    ranked = pd.DataFrame(candidates)
    if ranked.empty:
        return []
    ranked = ranked.sort_values(
        ["_AIL_CONSENSUS_SCORE", "AIL Top3 Normal Score", "AIL Top3 Tomorrow Score", "AIL Top3 Scanner Score"],
        ascending=False,
        kind="stable",
    ).head(max(1, int(top_n))).reset_index(drop=True)
    ranked["AIL Category Rank"] = range(1, len(ranked) + 1)
    return ranked.drop(columns=["_AIL_CONSENSUS_SCORE"], errors="ignore").to_dict("records")


def _candidate_driver_text(candidate: dict[str, Any]) -> list[str]:
    drivers: list[str] = []
    checks = candidate.get("checks", {}) if isinstance(candidate.get("checks"), dict) else {}
    for key in ("freshness", "closing", "rsi", "volume", "proximity", "sector"):
        item = checks.get(key, {})
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "").strip()
        reason = str(item.get("reason", "") or "").strip()
        if status and reason:
            drivers.append(f"{key}: {status} ({reason})")
        elif status:
            drivers.append(f"{key}: {status}")
    if not drivers and candidate.get("penalties"):
        drivers.append("penalties: " + "; ".join(str(reason) for _, reason in candidate.get("penalties", [])))
    return drivers[:4]


def _candidate_pool_from_top3(category_top3: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in category_top3.values():
        for row in list(payload.get("top_rows", []) or []):
            rows.append(dict(row))
    if not rows:
        return pd.DataFrame()
    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    grouped_categories: dict[str, list[str]] = {}
    for _, row in pool.iterrows():
        symbol = _row_symbol(row)
        category = str(row.get("AIL Category", "") or "").strip()
        if symbol and category:
            grouped_categories.setdefault(symbol, [])
            if category not in grouped_categories[symbol]:
                grouped_categories[symbol].append(category)

    pool = _dedupe_best_by_symbol(pool)
    if not pool.empty:
        pool["AIL Categories"] = [
            ", ".join(grouped_categories.get(_row_symbol(row), []))
            for _, row in pool.iterrows()
        ]
    return pool


def _prediction_cache_default() -> pd.DataFrame:
    try:
        from pathlib import Path

        path = Path(__file__).resolve().parent / "data" / "tomorrow_master_predictions.csv"
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _build_sector_context(df: pd.DataFrame) -> dict[str, Any]:
    by_symbol: dict[str, dict[str, Any]] = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"by_symbol": by_symbol}
    for _, row in df.iterrows():
        symbol = _row_symbol(row)
        if not symbol:
            continue
        by_symbol[symbol] = {
            "sector_accuracy": _find_numeric(row, "Sector Support", "Sector Strength", default=55.0, contains=("sector",)),
            "regime_fit": _find_numeric(row, "Regime Alignment", default=55.0),
        }
    return {"by_symbol": by_symbol}


def rank_cross_mode_leaders(
    candidate_pool: pd.DataFrame,
    *,
    compute_battle_scores_fn: Callable[..., pd.DataFrame] | None = None,
    market_bias: dict[str, Any] | None = None,
    prediction_cache: object = None,
    learning_profile: dict[str, Any] | None = None,
    market_state: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if candidate_pool is None or not isinstance(candidate_pool, pd.DataFrame) or candidate_pool.empty:
        return pd.DataFrame(), {}

    pool = _dedupe_best_by_symbol(candidate_pool)
    if pool.empty:
        return pd.DataFrame(), {}

    ranked = pool.copy()
    if callable(compute_battle_scores_fn):
        try:
            ranked = compute_battle_scores_fn(
                ranked,
                market_bias=market_bias,
                prediction_cache=prediction_cache if prediction_cache is not None else _prediction_cache_default(),
                sector_context=_build_sector_context(ranked),
            )
        except Exception:
            ranked = pool.copy()

    try:
        ranked = apply_market_state_adjustments(ranked, market_state or {})
        regime_bias = compute_regime_strategy_bias(market_bias, market_state)
        ranked = apply_regime_preference(ranked, regime_bias)
        ranked = preserve_mode_identity(ranked)
        ranked = preserve_high_upside_candidates(ranked)
        ranked = apply_conflict_penalties(ranked)
        ranked = apply_confidence_calibration(ranked, calibration)
        ranked = preserve_high_conviction(ranked)
        ranked = preserve_speculative_conviction(ranked)
        ranked = prevent_confidence_collapse(ranked)
        feature_cache = cache_orchestration_features(
            ranked,
            market_state=market_state,
            calibration=calibration,
            learning_profile=learning_profile,
        )
        ranked.attrs["ail_feature_cache_built_at"] = feature_cache.built_at
        ranked.attrs["ail_feature_cache_rows"] = feature_cache.row_count
        ranked = build_master_rankings(ranked, market_bias=market_bias, learning_profile=learning_profile)
        ranked = apply_confidence_calibration(ranked, calibration)
        ranked = prevent_confidence_collapse(ranked)
        ranked = cap_total_penalty(ranked)
        ranked = apply_orchestration_reasoning(ranked)
    except Exception:
        ranked = _sort_existing_scores(ranked)
        if not ranked.empty:
            if "AIL Master Score" not in ranked.columns:
                score_col = _best_sort_columns(ranked)[0] if _best_sort_columns(ranked) else None
                ranked["AIL Master Score"] = pd.to_numeric(ranked[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
            ranked["AIL Master Rank"] = range(1, len(ranked) + 1)

    return ranked, build_comparison_summary(ranked)


def _row_to_summary(row: pd.Series | None, label: str, metric_col: str | None = None) -> dict[str, Any]:
    if row is None:
        return {"label": label, "symbol": "", "score": 0.0, "reason": "No candidate available."}
    score_col = metric_col or "Smart Potential Score"
    reason = _first_text(row, ("Smart Notes", "Battle Notes", "AIL Top3 Drivers", "Compare Tags"), "Existing ranking signal")
    return {
        "label": label,
        "symbol": _row_symbol(row),
        "score": _safe_float(row.get(score_col), 0.0),
        "metric": score_col,
        "reason": reason,
        "row": row.to_dict(),
    }


def _max_row(df: pd.DataFrame, column: str, mask: pd.Series | None = None) -> pd.Series | None:
    if df is None or df.empty or column not in df.columns:
        return None
    work = df.loc[mask].copy() if mask is not None else df.copy()
    if work.empty:
        return None
    values = pd.to_numeric(work[column], errors="coerce")
    if values.dropna().empty:
        return None
    return work.loc[values.idxmax()]


def _min_row(df: pd.DataFrame, column: str, mask: pd.Series | None = None) -> pd.Series | None:
    if df is None or df.empty or column not in df.columns:
        return None
    work = df.loc[mask].copy() if mask is not None else df.copy()
    if work.empty:
        return None
    values = pd.to_numeric(work[column], errors="coerce")
    if values.dropna().empty:
        return None
    return work.loc[values.idxmin()]


def _first_available_row(*rows: pd.Series | None) -> pd.Series | None:
    for row in rows:
        if isinstance(row, pd.Series):
            return row
    return None


def build_comparison_summary(ranked_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
        return {}
    try:
        summary = select_category_leaders(ranked_df)
        if summary:
            return summary
    except Exception:
        pass

    top = ranked_df.iloc[0] if len(ranked_df) else None
    categories = ranked_df.get("AIL Categories", pd.Series("", index=ranked_df.index)).fillna("").astype(str)
    swing_mask = categories.str.contains("Swing", case=False, regex=False)
    inst_mask = categories.str.contains("Institutional", case=False, regex=False)
    trap_values = pd.to_numeric(ranked_df.get("Trap Risk Score", pd.Series(50.0, index=ranked_df.index)), errors="coerce").fillna(50.0)
    high_risk_mask = trap_values.ge(52.0)

    early_mask = pd.Series(False, index=ranked_df.index)
    try:
        rsi = pd.to_numeric(ranked_df.get("RSI", pd.Series(50.0, index=ranked_df.index)), errors="coerce")
        setup = ranked_df.get("Setup Type", pd.Series("", index=ranked_df.index)).fillna("").astype(str)
        timing = ranked_df.get("Entry Timing", pd.Series("", index=ranked_df.index)).fillna("").astype(str)
        early_mask = rsi.between(46, 58, inclusive="both") | setup.str.contains("PULLBACK|ACCUM", case=False, regex=True) | timing.str.contains("EARLY|FORMING", case=False, regex=True)
    except Exception:
        pass

    return {
        "best_overall": _row_to_summary(top, "Best Overall", "Smart Potential Score"),
        "safest_candidate": _row_to_summary(
            _first_available_row(_min_row(ranked_df, "Trap Risk Score"), _max_row(ranked_df, "Setup Cleanliness")),
            "Safest Candidate",
            "Trap Risk Score",
        ),
        "strongest_momentum": _row_to_summary(_max_row(ranked_df, "Momentum Quality"), "Strongest Momentum", "Momentum Quality"),
        "best_swing_setup": _row_to_summary(
            _first_available_row(_max_row(ranked_df, "Setup Cleanliness", swing_mask), _max_row(ranked_df, "Setup Cleanliness")),
            "Best Swing Setup",
            "Setup Cleanliness",
        ),
        "early_accumulation": _row_to_summary(
            _first_available_row(_max_row(ranked_df, "Smart Potential Score", early_mask), _max_row(ranked_df, "Setup Cleanliness")),
            "Early Accumulation Leader",
            "Smart Potential Score",
        ),
        "lowest_trap_risk": _row_to_summary(_min_row(ranked_df, "Trap Risk Score"), "Lowest Trap Risk", "Trap Risk Score"),
        "institutional_setup": _row_to_summary(
            _first_available_row(_max_row(ranked_df, "Regime Alignment", inst_mask), _max_row(ranked_df, "Regime Alignment")),
            "Strongest Institutional Setup",
            "Regime Alignment",
        ),
        "high_risk_high_reward": _row_to_summary(
            _first_available_row(_max_row(ranked_df, "Risk Reward Score", high_risk_mask), _max_row(ranked_df, "Risk Reward Score")),
            "High Risk High Reward",
            "Risk Reward Score",
        ),
    }


def _get_all_data_default() -> dict[str, Any]:
    try:
        from strategy_engines._engine_utils import ALL_DATA

        return ALL_DATA
    except Exception:
        return {}


def _history_for_symbol(symbol: str, all_data: dict[str, Any] | None) -> pd.DataFrame | None:
    data = all_data if isinstance(all_data, dict) else _get_all_data_default()
    base = _plain_symbol(symbol)
    for key in (base, f"{base}.NS"):
        try:
            frame = data.get(key)
        except Exception:
            frame = None
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None


def _default_aura_engine(df: pd.DataFrame, symbol: str, market_bias: dict[str, Any] | None):
    try:
        from app_stock_aura_section import _run_aura_engine

        return _run_aura_engine(df, symbol, market_bias)
    except Exception:
        return None


def _aura_to_dict(aura: Any, row: pd.Series | dict[str, Any] | None = None) -> dict[str, Any]:
    row_obj = row if row is not None else {}
    symbol = _plain_symbol(getattr(aura, "symbol", "") or _row_symbol(row_obj))
    verdict = str(getattr(aura, "verdict", "") or "")
    timing = str(getattr(aura, "timing", "") or "")
    aura_score = _safe_float(getattr(aura, "aura_score", 0.0), 0.0)
    trap_score = _find_numeric(row_obj, "Trap Risk Score", default=50.0)
    smart_score = _find_numeric(row_obj, "Smart Potential Score", "Battle Score", default=0.0)

    if "BUY TOMORROW" in verdict.upper() and smart_score >= 65 and trap_score < 56:
        ai_verdict = "Strong Buy Tomorrow"
    elif "BUY TOMORROW" in verdict.upper():
        ai_verdict = "Good Swing Candidate"
    elif "BUY TODAY" in verdict.upper() and trap_score < 56:
        ai_verdict = "Buy Today, but confirm entry discipline"
    elif "WATCH" in verdict.upper() and _find_numeric(row_obj, "Momentum Quality", default=0.0) >= 70:
        ai_verdict = "Momentum Strong but Wait for Entry"
    elif "WATCH" in verdict.upper():
        ai_verdict = "Watch for Breakout Confirmation"
    elif trap_score >= 70:
        ai_verdict = "Avoid - High Trap Risk"
    else:
        ai_verdict = "Avoid - Weak Confirmation"

    return {
        "Symbol": symbol,
        "Aura Score": round(aura_score, 2),
        "Final Verdict": verdict,
        "AI Verdict": ai_verdict,
        "Entry Timing": timing,
        "Timing Reason": str(getattr(aura, "timing_reason", "") or ""),
        "Entry Low": _safe_float(getattr(aura, "entry_low", 0.0), 0.0),
        "Entry High": _safe_float(getattr(aura, "entry_high", 0.0), 0.0),
        "ATR SL": _safe_float(getattr(aura, "sl_price", 0.0), 0.0),
        "Risk %": _safe_float(getattr(aura, "sl_pct", 0.0), 0.0),
        "Target 1": _safe_float(getattr(aura, "target1", 0.0), 0.0),
        "Target 2": _safe_float(getattr(aura, "target2", 0.0), 0.0),
        "RR": _safe_float(getattr(aura, "rr_ratio", 0.0), 0.0),
        "Market Note": str(getattr(aura, "market_note", "") or ""),
        "Positive Reasons": " | ".join(list(getattr(aura, "reasons_positive", []) or [])[:4]),
        "Warnings": " | ".join((list(getattr(aura, "reasons_warning", []) or []) + list(getattr(aura, "reasons_reject", []) or []))[:4]),
        "Smart Potential Score": smart_score,
        "Bullish Probability": _find_numeric(row_obj, "Bullish Probability", default=0.0),
        "Trap Risk Score": trap_score,
        "AIL Master Rank": int(_find_numeric(row_obj, "AIL Master Rank", default=0.0)),
    }


def run_final_aura_verdict(
    ranked_df: pd.DataFrame,
    *,
    all_data: dict[str, Any] | None = None,
    market_bias: dict[str, Any] | None = None,
    run_aura_engine_fn: Callable[[pd.DataFrame, str, dict[str, Any] | None], Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
        return []

    engine = run_aura_engine_fn if callable(run_aura_engine_fn) else _default_aura_engine
    verdicts: list[dict[str, Any]] = []
    for _, row in ranked_df.head(max(1, int(limit))).iterrows():
        symbol = _row_symbol(row)
        if not symbol:
            continue
        hist = _history_for_symbol(symbol, all_data)
        if hist is None or hist.empty:
            continue
        try:
            aura = engine(hist, symbol, market_bias)
            if aura is not None:
                verdicts.append(_aura_to_dict(aura, row))
        except Exception:
            continue
    return verdicts


def build_final_ranked_frame(ranked_df: pd.DataFrame, aura_verdicts: list[dict[str, Any]]) -> pd.DataFrame:
    if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
        return pd.DataFrame()
    out = ranked_df.copy()
    aura_df = pd.DataFrame(aura_verdicts)
    if not aura_df.empty and "Symbol" in aura_df.columns:
        aura_cols = [
            "Symbol",
            "Aura Score",
            "Final Verdict",
            "AI Verdict",
            "Entry Timing",
            "ATR SL",
            "Risk %",
            "Target 1",
            "Target 2",
            "RR",
            "Warnings",
        ]
        aura_df = aura_df[[col for col in aura_cols if col in aura_df.columns]].copy()
        out["_AIL_SYMBOL_KEY"] = out.apply(_row_symbol, axis=1)
        out = out.merge(aura_df, how="left", left_on="_AIL_SYMBOL_KEY", right_on="Symbol", suffixes=("", " Aura"))
        out = out.drop(columns=["_AIL_SYMBOL_KEY", "Symbol Aura"], errors="ignore")
    labels = [
        "Best Overall",
        "Strong Alternative",
        "Safe Setup",
        "High Risk High Reward",
        "Momentum Alternative",
        "Swing Watch",
        "Watchlist",
    ]
    out["AIL Rank Label"] = [
        labels[i] if i < len(labels) else "Watchlist"
        for i in range(len(out))
    ]
    return out.reset_index(drop=True)


def build_sector_strength(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df is None or not isinstance(ranked_df, pd.DataFrame) or ranked_df.empty:
        return pd.DataFrame()
    work = ranked_df.copy()
    if "Sector" not in work.columns:
        try:
            from sector_master import get_sector

            work["Sector"] = [str(get_sector(_row_symbol(row)) or "UNMAPPED") for _, row in work.iterrows()]
        except Exception:
            work["Sector"] = "UNMAPPED"
    for col in ("AIL Master Score", "Smart Potential Score", "Sector Support", "Momentum Quality", "Bullish Probability"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    score_col = "AIL Master Score" if "AIL Master Score" in work.columns else "Smart Potential Score" if "Smart Potential Score" in work.columns else _best_sort_columns(work)[0] if _best_sort_columns(work) else None
    rows: list[dict[str, Any]] = []
    for sector, grp in work.groupby("Sector", dropna=False):
        if grp.empty:
            continue
        best_row = _sort_existing_scores(grp).iloc[0]
        rows.append(
            {
                "Sector": str(sector or "UNMAPPED"),
                "Candidates": int(len(grp)),
                "Best Stock": _row_symbol(best_row),
                "Avg Smart Score": round(float(grp.get("Smart Potential Score", pd.Series(np.nan)).mean()), 2)
                if "Smart Potential Score" in grp.columns
                else np.nan,
                "Avg AIL Score": round(float(grp.get("AIL Master Score", pd.Series(np.nan)).mean()), 2)
                if "AIL Master Score" in grp.columns
                else np.nan,
                "Avg Sector Support": round(float(grp.get("Sector Support", pd.Series(np.nan)).mean()), 2)
                if "Sector Support" in grp.columns
                else np.nan,
                "Best Score": round(_safe_float(best_row.get(score_col), 0.0), 2) if score_col else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["Best Score", "Avg AIL Score", "Avg Smart Score"], ascending=False, kind="stable").reset_index(drop=True)


def build_risk_warnings(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df is None or not isinstance(final_df, pd.DataFrame) or final_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in final_df.iterrows():
        trap_score = _find_numeric(row, "Trap Risk Score", default=0.0)
        rsi = _find_numeric(row, "RSI", default=50.0)
        dist_ema20 = _find_numeric(row, "Delta vs EMA20 (%)", "Δ vs EMA20 (%)", default=0.0, contains=("ema20",))
        vol_ratio = _find_numeric(row, "Vol / Avg", default=1.0, contains=("vol", "avg"))
        warnings: list[str] = []
        if trap_score >= 70:
            warnings.append("High trap risk")
        elif trap_score >= 56:
            warnings.append("Trap risk watch")
        if rsi > 72:
            warnings.append("RSI overbought")
        if dist_ema20 > 6.5:
            warnings.append("Extended above EMA20")
        if vol_ratio < 1.0:
            warnings.append("Weak volume confirmation")
        final_signal = _first_text(row, ("Final Signal", "Adjusted Signal", "Signal"), "")
        if str(final_signal).upper() in {"AVOID", "TRAP"}:
            warnings.append(f"Final signal {final_signal}")
        ail_conf = _find_numeric(row, "AIL Confidence", default=0.0)
        ail_coverage = _find_numeric(row, "AIL Confidence Coverage", default=100.0)
        market_fit = _find_numeric(row, "AIL Market Compatibility", default=55.0)
        if ail_conf > 0 and ail_conf < 45:
            warnings.append("Low A-I-L confidence")
        if ail_coverage < 45:
            warnings.append("Thin confidence evidence")
        if market_fit < 45:
            warnings.append("Weak market compatibility")
        conflict = _find_numeric(row, "AIL Conflict Score", default=0.0)
        if conflict >= 45:
            warnings.append("Cross-mode conflict")
        temporal = _find_numeric(row, "AIL Temporal Fit", default=55.0)
        if temporal < 45:
            warnings.append("Weak market-state timing")
        suppression = _find_numeric(row, "AIL Suppression Index", default=0.0)
        if suppression > 12:
            warnings.append("Penalty stacking watch")
        if warnings:
            rows.append(
                {
                    "Symbol": _row_symbol(row),
                    "Rank": int(_find_numeric(row, "AIL Master Rank", default=0.0)),
                    "Warnings": "; ".join(dict.fromkeys(warnings)),
                    "Trap Risk Score": round(trap_score, 2),
                    "RSI": round(rsi, 2),
                    "Vol / Avg": round(vol_ratio, 2),
                    "EMA20 Distance %": round(dist_ema20, 2),
                    "AIL Reasoning": _first_text(row, ("AIL Reasoning",), ""),
                }
            )
    return pd.DataFrame(rows)


def build_confidence_meter(final_df: pd.DataFrame) -> dict[str, Any]:
    if final_df is None or not isinstance(final_df, pd.DataFrame) or final_df.empty:
        return {"score": 0.0, "label": "No candidates", "count": 0}
    confidence_cols = [col for col in ("AIL Calibrated Confidence", "AIL Confidence", "Smart Confidence", "Confidence", "Bullish Probability") if col in final_df.columns]
    if confidence_cols:
        values = []
        for col in confidence_cols:
            values.extend(pd.to_numeric(final_df[col], errors="coerce").dropna().tolist())
        score = float(np.mean(values)) if values else 0.0
    else:
        score = 0.0
    if score >= 70:
        label = "High"
    elif score >= 55:
        label = "Medium"
    elif score > 0:
        label = "Low"
    else:
        label = "No candidates"
    return {
        "score": round(score, 2),
        "label": label,
        "count": int(len(final_df)),
        "top_symbol": _row_symbol(final_df.iloc[0]) if len(final_df) else "",
    }


def collect_learning_insights(
    *,
    logged_predictions: int = 0,
    log_error: str = "",
    learning_profile: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibration_buckets = (
        calibration.get("buckets", pd.DataFrame())
        if isinstance(calibration, dict)
        else pd.DataFrame()
    )
    insights: dict[str, Any] = {
        "logged_predictions": int(logged_predictions or 0),
        "log_error": log_error,
        "training_status": {},
        "feedback_summary": {},
        "dynamic_weights": pd.DataFrame(),
        "ail_learning_profile": learning_profile or {},
        "ail_learning_table": learning_profile_table(learning_profile),
        "confidence_calibration": calibration_buckets,
        "calibration_drift": calibration.get("drift", {}) if isinstance(calibration, dict) else {},
    }
    try:
        from learning_engine import get_training_status

        status = get_training_status()
        if isinstance(status, dict):
            insights["training_status"] = status
    except Exception:
        pass
    try:
        from prediction_feedback_store import feedback_summary

        summary = feedback_summary()
        if isinstance(summary, dict):
            insights["feedback_summary"] = summary
    except Exception:
        pass
    try:
        from sector_dynamic_weights import get_signal_performance_report

        report = get_signal_performance_report()
        if isinstance(report, pd.DataFrame):
            insights["dynamic_weights"] = report
    except Exception:
        pass
    return insights


def build_health_summary(result: AILPipelineResult, *, logged_predictions: int = 0) -> dict[str, Any]:
    mode_df = pd.DataFrame(result.mode_summaries)
    raw_hits = 0
    enhanced = 0
    failed_modes = 0
    if not mode_df.empty:
        if "Raw Hits" in mode_df.columns:
            raw_hits = int(pd.to_numeric(mode_df["Raw Hits"], errors="coerce").fillna(0).sum())
        if "Enhanced Candidates" in mode_df.columns:
            enhanced = int(pd.to_numeric(mode_df["Enhanced Candidates"], errors="coerce").fillna(0).sum())
        if "Error" in mode_df.columns:
            failed_modes = int(mode_df["Error"].fillna("").astype(str).str.strip().ne("").sum())
    return {
        "modes_scanned": int(len(result.modes_scanned)),
        "raw_hits": raw_hits,
        "enhanced_candidates": enhanced,
        "ranked_candidates": int(len(result.final_ranked_df)) if isinstance(result.final_ranked_df, pd.DataFrame) else 0,
        "aura_verdicts": int(len(result.aura_verdicts or [])),
        "logged_predictions": int(logged_predictions or 0),
        "failed_modes": failed_modes,
    }


def log_ail_predictions(
    final_df: pd.DataFrame,
    *,
    market_bias: dict[str, Any] | None = None,
    log_scan_predictions_fn: Callable[[pd.DataFrame, int, dict[str, Any] | None], None] | None = None,
) -> tuple[int, str]:
    if final_df is None or not isinstance(final_df, pd.DataFrame) or final_df.empty:
        return 0, ""
    log_df = final_df.copy()
    log_df["Import Source"] = "A-I-L IN ONE"
    log_df["Import Category"] = "Master Ranking"
    log_df["Logged At"] = datetime.now().isoformat(timespec="seconds")
    if "Mode ID" in log_df.columns:
        log_df["Import Mode"] = log_df["Mode ID"]
    if "Mode" not in log_df.columns:
        log_df["Mode"] = log_df.get("Mode ID", 0)
    try:
        fn = log_scan_predictions_fn
        read_feedback_fn = None
        if not callable(fn):
            from prediction_feedback_store import log_scan_predictions as fn  # type: ignore[no-redef]
            try:
                from prediction_feedback_store import read_feedback_log as read_feedback_fn  # type: ignore[no-redef]
            except Exception:
                read_feedback_fn = None
        before_count = None
        if callable(read_feedback_fn):
            try:
                before_count = len(read_feedback_fn())
            except Exception:
                before_count = None
        fn(log_df, 0, market_bias)
        if callable(read_feedback_fn) and before_count is not None:
            try:
                after_count = len(read_feedback_fn())
                return max(0, int(after_count) - int(before_count)), ""
            except Exception:
                pass
        return int(len(log_df)), ""
    except Exception as exc:
        return 0, str(exc)


def run_ail_pipeline(
    tickers: list[str],
    *,
    workers: int = 12,
    modes: tuple[int, ...] | list[int] = AIL_MODES,
    prepare_market_session_data_fn: Callable[..., dict[str, Any]] | None = None,
    preload_all_fn: Callable[..., dict[str, Any]] | None = None,
    run_scan_fn: Callable[..., tuple[list[dict[str, Any]], float]] | None = None,
    enhance_results_fn: Callable[[list[dict[str, Any]], int], pd.DataFrame] | None = None,
    apply_enhanced_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_universal_grading_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase4_logic_fn: Callable[[pd.DataFrame, dict[str, Any] | None], pd.DataFrame] | None = None,
    apply_phase42_logic_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    apply_gate_to_scan_df_fn: Callable[..., pd.DataFrame] | None = None,
    compute_market_bias_fn: Callable[[], dict[str, Any]] | None = None,
    get_train_function_fn: Callable[[int], Callable] | None = None,
    compute_battle_scores_fn: Callable[..., pd.DataFrame] | None = None,
    run_aura_engine_fn: Callable[[pd.DataFrame, str, dict[str, Any] | None], Any] | None = None,
    compare_prediction_cache_fn: Callable[[], object] | None = None,
    log_scan_predictions_fn: Callable[[pd.DataFrame, int, dict[str, Any] | None], None] | None = None,
    all_data: dict[str, Any] | None = None,
    status_callback: Callable[[str, dict[str, Any]], None] | None = None,
    preload_progress_callback: Callable[[int, int, int], None] | None = None,
) -> AILPipelineResult:
    started = time.time()
    result = AILPipelineResult(started_at=datetime.now().isoformat(timespec="seconds"))
    tickers_clean = _normalize_tickers(tickers)
    result.requested_tickers = len(tickers_clean)
    result.modes_scanned = [int(m) for m in modes]
    data_store = all_data if isinstance(all_data, dict) else _get_all_data_default()

    def notify(stage: str, **payload: Any) -> None:
        if callable(status_callback):
            try:
                status_callback(stage, payload)
            except Exception:
                pass

    if not tickers_clean:
        result.errors.append("No tickers supplied.")
        return result
    if not callable(run_scan_fn):
        result.errors.append("run_scan function was not supplied.")
        return result

    notify("preload_start", total=len(tickers_clean))
    try:
        if callable(prepare_market_session_data_fn):
            result.preload_stats = prepare_market_session_data_fn(
                tickers_clean,
                period="6mo",
                workers=min(max(int(workers or 1), 1), 12),
                progress_callback=preload_progress_callback,
            )
        elif callable(preload_all_fn):
            result.preload_stats = preload_all_fn(
                tickers_clean,
                period="6mo",
                workers=min(max(int(workers or 1), 1), 12),
                progress_callback=preload_progress_callback,
            )
        else:
            result.preload_stats = {}
    except Exception as exc:
        result.errors.append(f"Preload failed: {exc}")
        result.preload_stats = {}
    try:
        plan = result.preload_stats.get("plan") if isinstance(result.preload_stats, dict) else {}
        result.market_state = detect_market_state(plan if isinstance(plan, dict) else {}, result.preload_stats)
    except Exception as exc:
        result.errors.append(f"Market state unavailable: {exc}")
        result.market_state = {}
    notify("preload_done", stats=result.preload_stats)

    try:
        result.market_bias = compute_market_bias_fn() if callable(compute_market_bias_fn) else {}
    except Exception as exc:
        result.errors.append(f"Market bias unavailable: {exc}")
        result.market_bias = {}

    for mode in result.modes_scanned:
        meta = get_mode_metadata(mode, copy=False)
        mode_name = str(meta.get("name", f"Mode {mode}"))
        notify("mode_start", mode=mode, mode_name=mode_name)
        raw_results: list[dict[str, Any]] = []
        elapsed = 0.0
        mode_error = ""

        try:
            if callable(get_train_function_fn):
                try:
                    get_train_function_fn(mode)()
                except Exception:
                    pass
            scan_output = run_scan_fn(tickers_clean, mode, workers=min(max(int(workers or 1), 1), 12))
            if isinstance(scan_output, tuple):
                raw_results, elapsed = scan_output
            else:
                raw_results, elapsed = list(scan_output or []), 0.0
        except Exception as exc:
            mode_error = str(exc)
            result.errors.append(f"Mode {mode} scan failed: {exc}")
            raw_results = []

        frame = pd.DataFrame()
        if raw_results:
            try:
                frame = _apply_pipeline_enrichment(
                    raw_results,
                    mode,
                    enhance_results_fn=enhance_results_fn,
                    apply_enhanced_logic_fn=apply_enhanced_logic_fn,
                    apply_universal_grading_fn=apply_universal_grading_fn,
                    apply_phase4_logic_fn=apply_phase4_logic_fn,
                    apply_phase42_logic_fn=apply_phase42_logic_fn,
                    apply_gate_to_scan_df_fn=apply_gate_to_scan_df_fn,
                    market_bias=result.market_bias,
                    all_data=data_store,
                )
            except Exception as exc:
                mode_error = str(exc)
                result.errors.append(f"Mode {mode} enrichment failed: {exc}")
                frame = pd.DataFrame(raw_results)
                frame = _add_mode_and_sector_columns(frame, mode)

        result.mode_frames[mode] = frame
        result.mode_summaries.append(
            {
                "Mode": mode,
                "Mode Name": mode_name,
                "Raw Hits": int(len(raw_results)),
                "Enhanced Candidates": int(len(frame)) if isinstance(frame, pd.DataFrame) else 0,
                "Elapsed Sec": round(float(elapsed or 0.0), 2),
                "Error": mode_error,
            }
        )
        notify("mode_done", mode=mode, mode_name=mode_name, raw=len(raw_results), enhanced=len(frame), elapsed=elapsed)

    frames = [frame for frame in result.mode_frames.values() if isinstance(frame, pd.DataFrame) and not frame.empty]
    result.combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    notify("classify_start", total=len(result.combined_df))

    result.categories = classify_scan_results(result.combined_df)
    result.category_top3 = extract_top_candidates(result.categories, market_bias=result.market_bias, top_n=3)
    result.candidate_pool = _candidate_pool_from_top3(result.category_top3)
    notify("compare_start", total=len(result.candidate_pool))

    try:
        learning_profile = build_ail_learning_profile()
    except Exception:
        learning_profile = {}
    try:
        calibration = compute_confidence_calibration()
    except Exception:
        calibration = {}
    try:
        candidate_cache = cache_orchestration_features(
            result.candidate_pool,
            market_state=result.market_state,
            calibration=calibration,
            learning_profile=learning_profile,
        )
        result.candidate_pool.attrs["ail_feature_cache_built_at"] = candidate_cache.built_at
    except Exception:
        pass

    prediction_cache = None
    if callable(compare_prediction_cache_fn):
        try:
            prediction_cache = compare_prediction_cache_fn()
        except Exception:
            prediction_cache = None

    result.comparison_df, result.comparison_summary = rank_cross_mode_leaders(
        result.candidate_pool,
        compute_battle_scores_fn=compute_battle_scores_fn,
        market_bias=result.market_bias,
        prediction_cache=prediction_cache,
        learning_profile=learning_profile,
        market_state=result.market_state,
        calibration=calibration,
    )
    notify("aura_start", total=len(result.comparison_df))

    result.aura_verdicts = run_final_aura_verdict(
        result.comparison_df,
        all_data=data_store,
        market_bias=result.market_bias,
        run_aura_engine_fn=run_aura_engine_fn,
        limit=10,
    )
    result.final_ranked_df = build_final_ranked_frame(result.comparison_df, result.aura_verdicts)
    result.sector_strength = build_sector_strength(result.final_ranked_df)
    result.risk_warnings = build_risk_warnings(result.final_ranked_df)
    result.confidence_meter = build_confidence_meter(result.final_ranked_df)

    logged, log_error = log_ail_predictions(
        result.final_ranked_df,
        market_bias=result.market_bias,
        log_scan_predictions_fn=log_scan_predictions_fn,
    )
    result.health = build_health_summary(result, logged_predictions=logged)
    result.learning_insights = collect_learning_insights(
        logged_predictions=logged,
        log_error=log_error,
        learning_profile=learning_profile,
        calibration=calibration,
    )

    result.elapsed_sec = round(time.time() - started, 2)
    try:
        result.health = compute_orchestration_health(
            result,
            state=result.market_state,
            calibration=calibration,
            conflict_df=result.final_ranked_df,
        )
    except Exception:
        pass
    notify("done", elapsed=result.elapsed_sec, ranked=len(result.final_ranked_df))
    return result


__all__ = [
    "AIL_MODES",
    "AIL_CATEGORY_ORDER",
    "AIL_MODE_CATEGORY_MAP",
    "AILPipelineResult",
    "run_ail_pipeline",
    "classify_scan_results",
    "extract_top_candidates",
    "rank_cross_mode_leaders",
    "run_final_aura_verdict",
    "build_sector_strength",
    "build_risk_warnings",
    "build_confidence_meter",
    "build_health_summary",
]
