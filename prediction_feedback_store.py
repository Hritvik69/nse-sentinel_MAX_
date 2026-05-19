"""
Append-only local log of scan predictions for accuracy tracking.
Safe: never raises to Streamlit callers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from atomic_io import atomic_write_csv_df, locked_path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

try:
    from persistent_store import push_file as _push_file
except Exception:
    def _push_file(*a, **kw):  # type: ignore[misc]
        pass

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
LOG_PATH = DATA_DIR / "prediction_feedback_log.csv"
_IMPORTED_AI_STORE_PATH = DATA_DIR / "imported_ai_learning_store.json"
_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))
_LOG_LOCK = threading.RLock()
_LOG = logging.getLogger(__name__)
_TARGET_POLICY_VERSION = "stock_next_session_v2"

_FIELDNAMES = [
    "logged_at",
    "market_date",
    "prediction_id",
    "symbol",
    "sector",
    "mode",
    "import_source",
    "import_category",
    "strategy_strip",
    "prediction_direction",
    "target_policy_version",
    "prediction_score",
    "final_score",
    "signal",
    "conviction_tier",
    "market_bias",
    "regime",
    "rsi",
    "vol_avg_ratio",
    "delta_ema20_pct",
    "trap_risk",
    "pred_bullish",
    "actual_next_return_pct",
    "correct",
    "outcome_label",
    "outcome_quality",
]

_LOG_CACHE_SIG: tuple[int, int] | None = None
_LOG_CACHE_DF: pd.DataFrame | None = None
_IMPORTED_META_CACHE_SIG: tuple[int, int] | None = None
_IMPORTED_META_CACHE: dict[str, dict[str, str]] = {}


def _ensure_data_dir() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _now_ist() -> datetime:
    try:
        return datetime.now(_IST_TZ)
    except Exception:
        return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _coerce_market_date(value: object, fallback: object = None) -> str:
    for candidate in (value, fallback):
        try:
            if candidate is None or str(candidate).strip() == "":
                continue
            parsed = pd.to_datetime(candidate, errors="coerce")
            if pd.isnull(parsed):
                continue
            if getattr(parsed, "tzinfo", None) is not None:
                parsed = parsed.tz_convert(_IST_TZ).tz_localize(None)
            return parsed.date().isoformat()
        except Exception:
            continue
    return _now_ist().date().isoformat()


def _normalize_symbol_key(value: object) -> str:
    return str(value or "").strip().upper().replace(".NS", "")


def _direction_from_score(score: object, fallback: object = "") -> str:
    text = str(fallback or "").strip().lower()
    if "bear" in text or "sell" in text or "avoid" in text:
        return "Bearish"
    if "bull" in text or "buy" in text or "green" in text:
        return "Bullish"
    try:
        score_f = float(score)
        return "Bullish" if np.isfinite(score_f) and score_f >= 55.0 else "Bearish"
    except Exception:
        return "Bearish"


def _stable_prediction_id(
    symbol: object,
    mode: object,
    market_date: object,
    import_source: object,
    direction: object,
) -> str:
    raw = "|".join(
        [
            _normalize_symbol_key(symbol),
            str(mode or "").strip(),
            str(market_date or "").strip(),
            str(import_source or "").strip().lower(),
            str(direction or "").strip().lower(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _dedupe_key(row: pd.Series | dict) -> tuple[str, str, str, str, str]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    return (
        _normalize_symbol_key(getter("symbol", "")),
        str(getter("mode", "") or "").strip(),
        str(getter("market_date", "") or "").strip(),
        str(getter("import_source", "") or "").strip().lower(),
        str(getter("prediction_direction", "") or "").strip().lower(),
    )


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        if not path.exists():
            return None
        stat = path.stat()
        return int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))), int(stat.st_size)
    except Exception:
        return None


def _set_cached_log(df: pd.DataFrame | None) -> pd.DataFrame:
    global _LOG_CACHE_SIG, _LOG_CACHE_DF
    cached = _coerce_schema(df)
    _LOG_CACHE_SIG = _file_signature(LOG_PATH)
    _LOG_CACHE_DF = cached.copy()
    return cached.copy()


def _invalidate_cache() -> None:
    global _LOG_CACHE_SIG, _LOG_CACHE_DF
    _LOG_CACHE_SIG = None
    _LOG_CACHE_DF = None


def _is_blank(value: object) -> bool:
    try:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return str(value).strip().lower() in ("", "nan", "none", "null")
    except Exception:
        return True


def _is_missing_label(value: object) -> bool:
    try:
        if _is_blank(value):
            return True
        return str(value).strip().lower() in {"-", "unknown", "n/a", "na"}
    except Exception:
        return True


def _clean_label(value: object, default: str = "") -> str:
    if _is_missing_label(value):
        return default
    return str(value).strip()


def _metadata_text_list(value: object) -> list[str]:
    raw_items: list[object]
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_items:
        text = _clean_label(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _join_metadata_values(value: object) -> str:
    return " | ".join(_metadata_text_list(value))


def _to_float(value: object) -> float | None:
    try:
        if _is_blank(value):
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _pred_is_bullish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "bullish", "yes", "y"}


def _first_present(row: pd.Series, keys: list[str], default: object = "") -> object:
    try:
        for key in keys:
            value = row.get(key, None)
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            return value
    except Exception:
        return default
    return default


def _correct_from_return(pred_bullish: object, next_return_pct: float | None) -> str:
    try:
        if next_return_pct is None or not np.isfinite(float(next_return_pct)):
            return ""
        is_correct = (
            (_pred_is_bullish(pred_bullish) and float(next_return_pct) > 0.0)
            or ((not _pred_is_bullish(pred_bullish)) and float(next_return_pct) <= 0.0)
        )
        return "True" if is_correct else "False"
    except Exception:
        return ""


def _outcome_from_correct(correct: object) -> str:
    text = str(correct).strip()
    if text == "True":
        return "correct"
    if text == "False":
        return "incorrect"
    return ""


def _outcome_quality_from_return(next_return_pct: object) -> str:
    try:
        value = _to_float(next_return_pct)
        if value is None:
            return ""
        if value >= 2.0:
            return "BIG_WIN"
        if value <= -2.0:
            return "BIG_LOSS"
        if -0.25 <= value <= 0.25:
            return "FLAT"
        if value > 0:
            return "WIN"
        return "LOSS"
    except Exception:
        return ""


def _strategy_strip_from_text(*values: object) -> str:
    text = " ".join(str(value or "") for value in values).strip().lower()
    if not text:
        return ""
    for label in ("relax", "swing", "intraday", "momentum", "breakout"):
        if label in text:
            return label.title()
    if "pulse" in text or "radar" in text:
        return "Breakout"
    return ""


def _mode_import_defaults(mode: object) -> dict[str, str]:
    text = _clean_label(mode).upper().replace("MODE", "").replace("M", "").strip()
    try:
        mode_int = int(float(text))
    except Exception:
        mode_int = -1
    if mode_int == 7:
        return {
            "import_category": "Momentum",
            "import_source": "Mode 7 / Tomorrow Picks",
            "strategy_strip": "Momentum",
        }
    return {}


def _snapshot_value(snapshot: dict, *keys: str) -> str:
    for key in keys:
        value = _clean_label(snapshot.get(key, ""))
        if value:
            return value
    return ""


def _load_imported_ai_metadata_map() -> dict[str, dict[str, str]]:
    global _IMPORTED_META_CACHE_SIG, _IMPORTED_META_CACHE
    try:
        sig = _file_signature(_IMPORTED_AI_STORE_PATH)
        if sig is not None and sig == _IMPORTED_META_CACHE_SIG:
            return {key: dict(value) for key, value in _IMPORTED_META_CACHE.items()}
        meta: dict[str, dict[str, str]] = {}
        if sig is None or not _IMPORTED_AI_STORE_PATH.exists():
            _IMPORTED_META_CACHE_SIG = sig
            _IMPORTED_META_CACHE = {}
            return {}

        payload = json.loads(_IMPORTED_AI_STORE_PATH.read_text(encoding="utf-8"))
        records = payload.get("records", []) if isinstance(payload, dict) else []
        try:
            from sector_master import get_sector
        except Exception:
            def get_sector(symbol: str) -> str | None:  # type: ignore[misc]
                return None

        for item in records if isinstance(records, list) else []:
            if not isinstance(item, dict):
                continue
            symbol = _normalize_symbol_key(item.get("ticker") or item.get("symbol"))
            if not symbol:
                continue
            snapshot = item.get("snapshot", {})
            snapshot = snapshot if isinstance(snapshot, dict) else {}
            categories = _metadata_text_list(item.get("categories", item.get("category", [])))
            sources = _metadata_text_list(item.get("sources", item.get("source", [])))
            modes = _metadata_text_list(item.get("modes", item.get("mode", [])))
            mode_defaults = _mode_import_defaults(modes[0] if modes else "")
            category_text = " | ".join(categories) or mode_defaults.get("import_category", "")
            source_text = " | ".join(sources) or mode_defaults.get("import_source", "")
            strip_text = (
                _strategy_strip_from_text(category_text, source_text, _snapshot_value(snapshot, "Setup Type", "Mode"))
                or mode_defaults.get("strategy_strip", "")
            )
            sector_text = _snapshot_value(snapshot, "Sector", "sector") or (get_sector(symbol) or "")
            trap_text = _snapshot_value(snapshot, "Trap Risk", "Trap", "trap_risk")
            meta[symbol] = {
                "import_category": category_text,
                "import_source": source_text,
                "strategy_strip": strip_text,
                "sector": sector_text,
                "trap_risk": trap_text,
                "mode": " | ".join([f"M{int(float(mode))}" if str(mode).replace('.', '', 1).isdigit() else str(mode) for mode in modes]),
            }

        _IMPORTED_META_CACHE_SIG = sig
        _IMPORTED_META_CACHE = {key: dict(value) for key, value in meta.items()}
        return meta
    except Exception:
        return {}


def enrich_imported_ai_feedback_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        out = df.copy()
        for column in ("import_category", "import_source", "strategy_strip", "sector", "trap_risk", "mode"):
            if column not in out.columns:
                out[column] = ""
        meta_map = _load_imported_ai_metadata_map()
        for idx, row in out.iterrows():
            symbol = _normalize_symbol_key(row.get("symbol", ""))
            meta = dict(meta_map.get(symbol, {}))
            has_meta = bool(meta)
            has_import_hint = any(
                not _is_missing_label(row.get(column, ""))
                for column in ("import_category", "import_source", "strategy_strip")
            )
            mode_defaults = _mode_import_defaults(row.get("mode", "")) if has_meta or has_import_hint else {}
            meta = {**mode_defaults, **{key: value for key, value in meta.items() if _clean_label(value)}}
            for column in ("import_category", "import_source", "strategy_strip", "sector", "trap_risk"):
                if _is_missing_label(row.get(column, "")) and _clean_label(meta.get(column, "")):
                    out.at[idx, column] = meta[column]
            if _is_missing_label(out.at[idx, "strategy_strip"]):
                strip = _strategy_strip_from_text(
                    out.at[idx, "import_category"],
                    out.at[idx, "import_source"],
                )
                if strip:
                    out.at[idx, "strategy_strip"] = strip
        for column in ("import_category", "import_source", "strategy_strip", "sector", "trap_risk"):
            out[column] = out[column].map(lambda value: _clean_label(value, ""))
        return out
    except Exception:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _coerce_logged_at(value: object, fallback: str) -> str:
    try:
        text = str(value or "").strip()
        if not text:
            return fallback
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isnull(parsed):
            return fallback
        if getattr(parsed, "tzinfo", None) is not None:
            parsed = parsed.tz_convert(_IST_TZ)
        else:
            parsed = parsed.tz_localize(_IST_TZ)
        return parsed.isoformat(timespec="seconds")
    except Exception:
        return fallback


def _coerce_schema(df: pd.DataFrame | None) -> pd.DataFrame:
    try:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for col in _FIELDNAMES:
            if col not in out.columns:
                out[col] = ""

        try:
            from sector_master import get_sector
        except Exception:
            def get_sector(symbol: str) -> str | None:  # type: ignore[misc]
                return None

        if "sector" in out.columns and "symbol" in out.columns:
            derived_sector = out["symbol"].astype(str).map(lambda value: get_sector(str(value).strip()) or "")
            out["sector"] = np.where(out["sector"].apply(_is_blank), derived_sector, out["sector"])

        fallback_logged = out["logged_at"] if "logged_at" in out.columns else ""
        out["market_date"] = [
            _coerce_market_date(market, logged)
            for market, logged in zip(out.get("market_date", ""), fallback_logged)
        ]
        out["prediction_direction"] = [
            direction if not _is_blank(direction) else _direction_from_score(score, signal)
            for direction, score, signal in zip(
                out.get("prediction_direction", ""),
                out.get("prediction_score", ""),
                out.get("signal", ""),
            )
        ]
        out["target_policy_version"] = np.where(
            out["target_policy_version"].apply(_is_blank),
            _TARGET_POLICY_VERSION,
            out["target_policy_version"],
        )
        out["prediction_id"] = [
            pid if not _is_blank(pid) else _stable_prediction_id(sym, mode, market, src, direction)
            for pid, sym, mode, market, src, direction in zip(
                out.get("prediction_id", ""),
                out.get("symbol", ""),
                out.get("mode", ""),
                out.get("market_date", ""),
                out.get("import_source", ""),
                out.get("prediction_direction", ""),
            )
        ]

        derived_correct = (
            out.get("outcome_label", "")
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"correct": "True", "incorrect": "False", "true": "True", "false": "False"})
            .fillna("")
        )
        out["correct"] = np.where(out["correct"].apply(_is_blank), derived_correct, out["correct"])

        derived_outcome = out["correct"].apply(_outcome_from_correct)
        out["outcome_label"] = np.where(
            out["outcome_label"].apply(_is_blank),
            derived_outcome,
            out["outcome_label"],
        )
        derived_strip = [
            _strategy_strip_from_text(strip, category, source)
            for strip, category, source in zip(
                out.get("strategy_strip", ""),
                out.get("import_category", ""),
                out.get("import_source", ""),
            )
        ]
        out["strategy_strip"] = np.where(
            out["strategy_strip"].apply(_is_blank),
            derived_strip,
            out["strategy_strip"],
        )
        derived_quality = out["actual_next_return_pct"].apply(_outcome_quality_from_return)
        out["outcome_quality"] = np.where(
            out["outcome_quality"].apply(_is_blank),
            derived_quality,
            out["outcome_quality"],
        )
        return out[_FIELDNAMES]
    except Exception:
        return pd.DataFrame(columns=_FIELDNAMES)


def _ensure_schema() -> None:
    try:
        _ensure_data_dir()
        if not LOG_PATH.exists():
            _invalidate_cache()
            return
        with locked_path(LOG_PATH):
            current = pd.read_csv(LOG_PATH, dtype=str)
            upgraded = _coerce_schema(current)
            if list(current.columns) != _FIELDNAMES:
                atomic_write_csv_df(LOG_PATH, upgraded, index=False)
        _set_cached_log(upgraded)
    except Exception:
        pass


def read_feedback_log() -> pd.DataFrame:
    global _LOG_CACHE_SIG, _LOG_CACHE_DF
    try:
        _ensure_schema()
        if not LOG_PATH.exists():
            _invalidate_cache()
            return pd.DataFrame(columns=_FIELDNAMES)
        current_sig = _file_signature(LOG_PATH)
        if current_sig is not None and _LOG_CACHE_SIG == current_sig and isinstance(_LOG_CACHE_DF, pd.DataFrame):
            return _LOG_CACHE_DF.copy()
        with locked_path(LOG_PATH):
            fresh = _coerce_schema(pd.read_csv(LOG_PATH, dtype=str))
        return _set_cached_log(fresh)
    except Exception:
        return pd.DataFrame(columns=_FIELDNAMES)


def _normalize_history_frame(df_hist: pd.DataFrame | None) -> pd.DataFrame | None:
    try:
        if df_hist is None or df_hist.empty:
            return None
        hist = df_hist.copy()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if "Close" not in hist.columns:
            return None
        hist.index = pd.to_datetime(hist.index, errors="coerce")
        hist = hist[~hist.index.isna()].sort_index()
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        hist = hist[~hist.index.duplicated(keep="last")].copy()
        hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
        hist = hist.dropna(subset=["Close"])
        return hist if len(hist) >= 2 else None
    except Exception:
        return None


def _resolve_history(all_data: dict | None, symbol: str) -> pd.DataFrame | None:
    try:
        if not isinstance(all_data, dict) or not all_data:
            return None
        raw = str(symbol or "").strip().upper()
        if not raw:
            return None
        plain = raw.replace(".NS", "")
        for key in (raw, plain, f"{plain}.NS"):
            hist = _normalize_history_frame(all_data.get(key))
            if hist is not None:
                return hist
        return None
    except Exception:
        return None


def _legacy_log_scan_predictions_unused_old(
    df: pd.DataFrame,
    mode: int,
    market_bias: dict | None,
) -> None:
    """Compatibility shim for old imports; delegates to the maintained logger."""
    return log_scan_predictions(df, mode, market_bias)

_log_scan_predictions_legacy = _legacy_log_scan_predictions_unused_old


def log_scan_predictions(
    df: pd.DataFrame,
    mode: int,
    market_bias: dict | None,
) -> None:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        mb = market_bias if isinstance(market_bias, dict) else {}
        bias_s = str(mb.get("bias", ""))[:160]
        regime_s = str(mb.get("regime", ""))[:80]
        try:
            from sector_master import get_sector
        except Exception:
            def get_sector(symbol: str) -> str | None:  # type: ignore[misc]
                return None

        ts = _now_ist().isoformat(timespec="seconds")
        rows: list[dict[str, object]] = []
        for _, row in df.iterrows():
            try:
                sym = str(row.get("Symbol") or row.get("Ticker") or "").strip()
                if not sym:
                    continue
                row_mode_raw = row.get("Import Mode", row.get("Mode", mode))
                try:
                    row_mode = int(row_mode_raw) if row_mode_raw is not None and pd.notna(row_mode_raw) else int(mode)
                except Exception:
                    row_mode = int(mode) if mode is not None else 0
                import_source = str(row.get("Import Source", "") or "")[:160]
                logged_at = _coerce_logged_at(
                    row.get("Logged At", row.get("logged_at", row.get("Imported At", ""))),
                    ts,
                )
                market_date = _coerce_market_date(row.get("Market Date", row.get("market_date", "")), logged_at)
                ps = row.get("Prediction Score", np.nan)
                signal = str(row.get("Signal", "") or "")[:40]
                direction = _direction_from_score(ps, signal)
                try:
                    ps_f = float(ps) if ps is not None and pd.notna(ps) else float("nan")
                except Exception:
                    ps_f = float("nan")
                fs = row.get("Final Score", np.nan)
                rsi = _to_float(_first_present(row, ["RSI"], ""))
                vol_avg_ratio = _to_float(_first_present(row, ["Vol / Avg", "Volume Ratio"], ""))
                delta_ema20_pct = _to_float(
                    _first_present(
                        row,
                        ["Delta vs EMA20 (%)", "EMA Distance (%)", "Î” vs EMA20 (%)", "ÃŽâ€ vs EMA20 (%)"],
                        "",
                    )
                )
                rows.append(
                    {
                        "logged_at": logged_at,
                        "market_date": market_date,
                        "prediction_id": _stable_prediction_id(sym, row_mode, market_date, import_source, direction),
                        "symbol": sym,
                        "sector": str(row.get("Sector") or get_sector(sym) or "").strip(),
                        "mode": row_mode,
                        "import_source": import_source,
                        "import_category": str(row.get("Import Category", "") or "")[:80],
                        "strategy_strip": str(
                            _first_present(
                                row,
                                ["Strategy Strip", "Tomorrow Strip", "Strip", "strategy_strip"],
                                "",
                            )
                            or _strategy_strip_from_text(row.get("Import Category", ""), import_source)
                        )[:80],
                        "prediction_direction": direction,
                        "target_policy_version": _TARGET_POLICY_VERSION,
                        "prediction_score": f"{ps_f:.4f}" if np.isfinite(ps_f) else "",
                        "final_score": f"{float(fs):.4f}" if fs is not None and pd.notna(fs) else "",
                        "signal": signal,
                        "conviction_tier": str(row.get("Conviction Tier", "") or "")[:20],
                        "market_bias": bias_s,
                        "regime": regime_s,
                        "rsi": f"{rsi:.4f}" if rsi is not None else "",
                        "vol_avg_ratio": f"{vol_avg_ratio:.4f}" if vol_avg_ratio is not None else "",
                        "delta_ema20_pct": f"{delta_ema20_pct:.4f}" if delta_ema20_pct is not None else "",
                        "trap_risk": str(
                            _first_present(row, ["Trap Risk", "Trap Check", "Trap Flags", "Bull Trap", "Trap"], "")
                            or ""
                        )[:40],
                        "pred_bullish": "1" if direction == "Bullish" else "0",
                        "actual_next_return_pct": "",
                        "correct": "",
                        "outcome_label": "",
                        "outcome_quality": "",
                    }
                )
            except Exception:
                continue
        if not rows:
            return

        with _LOG_LOCK:
            _ensure_data_dir()
            if LOG_PATH.exists():
                with locked_path(LOG_PATH):
                    existing = _coerce_schema(pd.read_csv(LOG_PATH, dtype=str))
            else:
                existing = pd.DataFrame(columns=_FIELDNAMES)
            seen_keys = {_dedupe_key(row) for _, row in existing.iterrows()}
            filtered = [row for row in rows if _dedupe_key(row) not in seen_keys]
            if not filtered:
                _set_cached_log(existing)
                return
            combined = _coerce_schema(pd.concat([existing, pd.DataFrame(filtered)], ignore_index=True))
            atomic_write_csv_df(LOG_PATH, combined, index=False)
            _set_cached_log(combined)

        if not _push_file(LOG_PATH):
            _LOG.error("prediction_feedback_store: queueing feedback log sync failed")
    except Exception:
        _LOG.exception("prediction_feedback_store: log_scan_predictions failed")


def feedback_summary() -> dict:
    out: dict = {
        "total_logged": 0,
        "rows_with_outcome": 0,
        "accuracy_pct": None,
        "bullish_precision_pct": None,
        "bearish_precision_pct": None,
        "false_bullish_pct": None,
        "false_bearish_pct": None,
    }
    try:
        df = read_feedback_log()
        out["total_logged"] = int(len(df))
        if df.empty:
            return out

        sub = df.copy()
        sub["_act"] = sub["actual_next_return_pct"].map(_to_float)
        sub = sub[sub["_act"].notna()].copy()
        out["rows_with_outcome"] = int(len(sub))
        if sub.empty:
            return out

        sub["_bull_pred"] = sub["pred_bullish"].map(_pred_is_bullish)
        sub["_act_pos"] = sub["_act"] > 0

        bull_rows = sub[sub["_bull_pred"]]
        bear_rows = sub[~sub["_bull_pred"]]

        if len(bull_rows) > 0:
            bull_ok = int((bull_rows["_act_pos"]).sum())
            out["bullish_precision_pct"] = round(100.0 * float(bull_ok) / float(len(bull_rows)), 2)
            out["false_bullish_pct"] = round(
                100.0 * float((~bull_rows["_act_pos"]).sum()) / float(len(bull_rows)),
                2,
            )

        if len(bear_rows) > 0:
            bear_ok = int(((~bear_rows["_act_pos"]) | (bear_rows["_act"] == 0)).sum())
            out["bearish_precision_pct"] = round(100.0 * float(bear_ok) / float(len(bear_rows)), 2)
            out["false_bearish_pct"] = round(
                100.0 * float((bear_rows["_act_pos"]).sum()) / float(len(bear_rows)),
                2,
            )

        sub["_correct"] = sub.apply(
            lambda row: _correct_from_return(row.get("pred_bullish"), row.get("_act")),
            axis=1,
        )
        hits = int((sub["_correct"] == "True").sum())
        out["accuracy_pct"] = round(100.0 * float(hits) / float(len(sub)), 2)
        return out
    except Exception:
        return out


def backfill_actual_returns(all_data: dict) -> int:
    """
    Auto-fill actual_next_return_pct and correctness for logged predictions.

    For every row where the outcome is still missing, look up the logged symbol
    inside ALL_DATA, align the logged date to the last available session on or
    before that date, then compute the next-session return from the next close.

    Returns the number of rows validated during this call.
    """
    try:
        with _LOG_LOCK:
            df = read_feedback_log()
            if df.empty:
                return 0

            validated = 0
            changed = False

            for idx in df.index:
                try:
                    existing_ret = _to_float(df.at[idx, "actual_next_return_pct"])
                    existing_correct = str(df.at[idx, "correct"]).strip()
                    if existing_ret is not None and existing_correct in ("True", "False"):
                        continue

                    ret_pct = existing_ret
                    if ret_pct is None:
                        sym = str(df.at[idx, "symbol"]).strip()
                        hist = _resolve_history(all_data, sym)
                        if hist is None:
                            continue

                        market_date = _coerce_market_date(
                            df.at[idx, "market_date"] if "market_date" in df.columns else "",
                            df.at[idx, "logged_at"],
                        )
                        logged_date = pd.to_datetime(market_date, errors="coerce")
                        if pd.isnull(logged_date):
                            continue
                        logged_date = logged_date.date()

                        hist_dates = np.array(pd.to_datetime(hist.index).date)
                        base_locs = np.where(hist_dates <= logged_date)[0]
                        if len(base_locs) == 0:
                            continue
                        base_idx = int(base_locs[-1])
                        next_idx = base_idx + 1
                        if next_idx >= len(hist):
                            continue

                        close_today = _to_float(hist["Close"].iloc[base_idx])
                        close_next = _to_float(hist["Close"].iloc[next_idx])
                        if close_today is None or close_next is None or close_today <= 0:
                            continue

                        ret_pct = round((close_next / close_today - 1.0) * 100.0, 4)
                        df.at[idx, "actual_next_return_pct"] = f"{ret_pct:.4f}"
                        changed = True

                    correct = _correct_from_return(df.at[idx, "pred_bullish"], ret_pct)
                    if correct in ("True", "False"):
                        if str(df.at[idx, "correct"]).strip() != correct:
                            df.at[idx, "correct"] = correct
                            changed = True
                        outcome = _outcome_from_correct(correct)
                        if str(df.at[idx, "outcome_label"]).strip().lower() != outcome:
                            df.at[idx, "outcome_label"] = outcome
                            changed = True
                        quality = _outcome_quality_from_return(ret_pct)
                        if "outcome_quality" in df.columns and str(df.at[idx, "outcome_quality"]).strip() != quality:
                            df.at[idx, "outcome_quality"] = quality
                            changed = True
                        if "target_policy_version" in df.columns and _is_blank(df.at[idx, "target_policy_version"]):
                            df.at[idx, "target_policy_version"] = _TARGET_POLICY_VERSION
                            changed = True
                        validated += 1
                except Exception:
                    continue

            if changed:
                df = _coerce_schema(df)
                atomic_write_csv_df(LOG_PATH, df, index=False)
                if not _push_file(LOG_PATH):
                    _LOG.error("prediction_feedback_store: queueing feedback backfill sync failed")
                _set_cached_log(df)
            return validated
    except Exception:
        _LOG.exception("prediction_feedback_store: backfill_actual_returns failed")
        return 0


def _is_imported_ai_row(row: pd.Series | dict) -> bool:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    source = _clean_label(getter("import_source", ""))
    category = _clean_label(getter("import_category", ""))
    strip = _clean_label(getter("strategy_strip", ""))
    if not any((source, category, strip)):
        return False
    if source.lower() in {"scan", "scanner", "main scan"} and not category and not strip:
        return False
    return True


def _bucket_performance(df: pd.DataFrame, column: str, *, min_rows: int = 1) -> list[dict[str, object]]:
    if df.empty or column not in df.columns:
        return []
    rows: list[dict[str, object]] = []
    work = df.copy()
    work[column] = work[column].map(lambda value: _clean_label(value, "UNKNOWN"))
    for bucket, group in work.groupby(column, dropna=False):
        total = int(len(group))
        if total < min_rows:
            continue
        correct = int(group["_correct_bool"].sum())
        avg_return = float(group["_actual_return"].mean()) if total else 0.0
        false_bull = 0.0
        bull_group = group[group["_bullish_bool"]]
        if not bull_group.empty:
            false_bull = round(100.0 * float((~bull_group["_correct_bool"]).sum()) / float(len(bull_group)), 2)
        rows.append(
            {
                "bucket": str(bucket or "UNKNOWN"),
                "rows": total,
                "correct": correct,
                "accuracy_pct": round(100.0 * float(correct) / float(total), 2),
                "avg_return_pct": round(avg_return, 4),
                "false_bullish_pct": false_bull,
            }
        )
    return sorted(rows, key=lambda item: (float(item["accuracy_pct"]), float(item["avg_return_pct"]), int(item["rows"])), reverse=True)


def summarize_imported_ai_performance(*, recent_limit: int = 20, min_bucket_rows: int = 1) -> dict[str, object]:
    default: dict[str, object] = {
        "total_logged": 0,
        "validated": 0,
        "accuracy_pct": None,
        "avg_return_pct": None,
        "false_bullish_pct": None,
        "best_category": {},
        "worst_category": {},
        "best_sector": {},
        "worst_sector": {},
        "by_import_category": [],
        "by_import_source": [],
        "by_mode": [],
        "by_sector": [],
        "by_trap_risk": [],
        "by_strategy_strip": [],
        "recent": pd.DataFrame(),
    }
    try:
        df = read_feedback_log()
        if df.empty:
            return default
        import_mask = df.apply(_is_imported_ai_row, axis=1)
        work = enrich_imported_ai_feedback_frame(df)
        work = work[import_mask].copy()
        default["total_logged"] = int(len(work))
        if work.empty:
            return default

        work["_actual_return"] = pd.to_numeric(work.get("actual_next_return_pct", ""), errors="coerce")
        work = work[work["_actual_return"].notna()].copy()
        work["_correct_text"] = work.get("correct", "").astype(str).str.strip()
        work = work[work["_correct_text"].isin(["True", "False"])].copy()
        default["validated"] = int(len(work))
        if work.empty:
            return default

        work["_correct_bool"] = work["_correct_text"].eq("True")
        work["_bullish_bool"] = work.get("pred_bullish", "").astype(str).str.strip().str.lower().isin(
            ["1", "1.0", "true", "bullish", "yes", "y"]
        )
        work["strategy_strip"] = [
            _clean_label(strip) or _strategy_strip_from_text(category, source) or "UNKNOWN"
            for strip, category, source in zip(
                work.get("strategy_strip", ""),
                work.get("import_category", ""),
                work.get("import_source", ""),
            )
        ]
        work["outcome_quality"] = [
            _clean_label(quality) or _outcome_quality_from_return(ret)
            for quality, ret in zip(work.get("outcome_quality", ""), work["_actual_return"])
        ]
        for column in ("import_category", "import_source", "sector", "trap_risk"):
            if column in work.columns:
                work[column] = work[column].map(lambda value: _clean_label(value, "UNKNOWN"))

        default["accuracy_pct"] = round(100.0 * float(work["_correct_bool"].sum()) / float(len(work)), 2)
        default["avg_return_pct"] = round(float(work["_actual_return"].mean()), 4)
        bull_rows = work[work["_bullish_bool"]]
        if not bull_rows.empty:
            default["false_bullish_pct"] = round(
                100.0 * float((~bull_rows["_correct_bool"]).sum()) / float(len(bull_rows)),
                2,
            )

        bucket_columns = {
            "by_import_category": "import_category",
            "by_import_source": "import_source",
            "by_mode": "mode",
            "by_sector": "sector",
            "by_trap_risk": "trap_risk",
            "by_strategy_strip": "strategy_strip",
        }
        for key, column in bucket_columns.items():
            default[key] = _bucket_performance(work, column, min_rows=min_bucket_rows)

        categories = list(default.get("by_import_category", []) or [])
        sectors = list(default.get("by_sector", []) or [])
        if categories:
            default["best_category"] = categories[0]
            default["worst_category"] = sorted(
                categories,
                key=lambda item: (float(item["accuracy_pct"]), float(item["avg_return_pct"]), -int(item["rows"])),
            )[0]
        if sectors:
            default["best_sector"] = sectors[0]
            default["worst_sector"] = sorted(
                sectors,
                key=lambda item: (float(item["accuracy_pct"]), float(item["avg_return_pct"]), -int(item["rows"])),
            )[0]

        if "logged_at" in work.columns:
            work["_logged_dt"] = pd.to_datetime(work["logged_at"], errors="coerce")
            work = work.sort_values("_logged_dt", ascending=False, na_position="last")
        recent_cols = [
            "logged_at",
            "symbol",
            "mode",
            "import_category",
            "import_source",
            "strategy_strip",
            "sector",
            "trap_risk",
            "prediction_score",
            "final_score",
            "actual_next_return_pct",
            "outcome_quality",
            "correct",
        ]
        default["recent"] = work[[col for col in recent_cols if col in work.columns]].head(recent_limit).reset_index(drop=True)
        return default
    except Exception:
        return default
