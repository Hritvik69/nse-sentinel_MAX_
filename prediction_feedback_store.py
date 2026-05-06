"""
Append-only local log of scan predictions for accuracy tracking.
Safe: never raises to Streamlit callers.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
LOG_PATH = DATA_DIR / "prediction_feedback_log.csv"

_FIELDNAMES = [
    "logged_at",
    "symbol",
    "sector",
    "mode",
    "import_source",
    "import_category",
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
]

_LOG_CACHE_SIG: tuple[int, int] | None = None
_LOG_CACHE_DF: pd.DataFrame | None = None


def _ensure_data_dir() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


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
        return str(value).strip() in ("", "nan", "None")
    except Exception:
        return True


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
        return out[_FIELDNAMES]
    except Exception:
        return pd.DataFrame(columns=_FIELDNAMES)


def _ensure_schema() -> None:
    try:
        _ensure_data_dir()
        if not LOG_PATH.exists():
            _invalidate_cache()
            return
        upgraded = _coerce_schema(pd.read_csv(LOG_PATH, dtype=str))
        upgraded.to_csv(LOG_PATH, index=False)
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


def log_scan_predictions(
    df: pd.DataFrame,
    mode: int,
    market_bias: dict | None,
) -> None:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        _invalidate_cache()
        _ensure_schema()
        mb = market_bias if isinstance(market_bias, dict) else {}
        bias_s = str(mb.get("bias", ""))[:160]
        regime_s = str(mb.get("regime", ""))[:80]
        try:
            from sector_master import get_sector
        except Exception:
            def get_sector(symbol: str) -> str | None:  # type: ignore[misc]
                return None
        ts = datetime.now().isoformat(timespec="seconds")
        file_exists = LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
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
                    sector = str(row.get("Sector") or get_sector(sym) or "").strip()
                    import_source = str(row.get("Import Source", "") or "")[:160]
                    import_category = str(row.get("Import Category", "") or "")[:80]
                    ps = row.get("Prediction Score", np.nan)
                    fs = row.get("Final Score", np.nan)
                    sig = str(row.get("Signal", "") or "")[:40]
                    ct = str(row.get("Conviction Tier", "") or "")[:20]
                    rsi = _to_float(_first_present(row, ["RSI"], ""))
                    vol_avg_ratio = _to_float(_first_present(row, ["Vol / Avg", "Volume Ratio"], ""))
                    delta_ema20_pct = _to_float(
                        _first_present(
                            row,
                            ["Δ vs EMA20 (%)", "Î” vs EMA20 (%)", "Delta vs EMA20 (%)", "EMA Distance (%)"],
                            "",
                        )
                    )
                    trap_risk = str(
                        _first_present(row, ["Trap Risk", "Trap Check", "Trap Flags", "Bull Trap", "Trap"], "")
                        or ""
                    )[:40]
                    try:
                        ps_f = float(ps) if ps is not None and pd.notna(ps) else float("nan")
                    except Exception:
                        ps_f = float("nan")
                    writer.writerow(
                        {
                            "logged_at": ts,
                            "symbol": sym,
                            "sector": sector,
                            "mode": row_mode,
                            "import_source": import_source,
                            "import_category": import_category,
                            "prediction_score": f"{ps_f:.4f}" if np.isfinite(ps_f) else "",
                            "final_score": f"{float(fs):.4f}" if fs is not None and pd.notna(fs) else "",
                            "signal": sig,
                            "conviction_tier": ct,
                            "market_bias": bias_s,
                            "regime": regime_s,
                            "rsi": f"{rsi:.4f}" if rsi is not None else "",
                            "vol_avg_ratio": f"{vol_avg_ratio:.4f}" if vol_avg_ratio is not None else "",
                            "delta_ema20_pct": f"{delta_ema20_pct:.4f}" if delta_ema20_pct is not None else "",
                            "trap_risk": trap_risk,
                            "pred_bullish": "1" if (np.isfinite(ps_f) and ps_f >= 55.0) else "0",
                            "actual_next_return_pct": "",
                            "correct": "",
                            "outcome_label": "",
                        }
                    )
                except Exception:
                    continue
        try:
            read_feedback_log()
        except Exception:
            pass
    except Exception:
        return


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

                    logged_dt = pd.to_datetime(str(df.at[idx, "logged_at"]).strip(), errors="coerce")
                    if pd.isnull(logged_dt):
                        continue
                    if getattr(logged_dt, "tzinfo", None) is not None:
                        logged_dt = logged_dt.tz_localize(None)
                    logged_date = logged_dt.date()

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
                    validated += 1
            except Exception:
                continue

        if changed:
            df.to_csv(LOG_PATH, index=False)
            _set_cached_log(df)
        return validated
    except Exception:
        return 0
