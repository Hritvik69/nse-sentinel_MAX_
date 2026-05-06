"""
sector_prediction_tracker.py
══════════════════════════════
Layer 4 — Execution Tracking & Feedback Loop.

Responsibilities
────────────────
• Log every prediction to a persistent CSV.
• Backfill actual outcomes (next-day return) from ALL_DATA.
• Compute calibration factors (historical accuracy by sector + direction).
• Never raises — every public function is fully exception-safe.

Storage
───────
data/sector_predictions.csv

Schema
──────
predicted_at      ISO-8601 UTC timestamp
sector            sector name
direction         Bullish | Bearish | Sideways
confidence        float 0–100
raw_score         float 0–100
entry_price       float (last synthetic sector close)
exit_price        float (next-session close, filled retroactively)
return_pct        float (exit/entry − 1) × 100
correct           True | False | ""  (blank = not yet validated)
leader_ticker     str   (first stock used in aggregation)
signal_ema_slope  float
signal_momentum   float
signal_volume     float
signal_sector_str float
signal_bullish_pct float
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from persistent_store import push_file as _push_file
except Exception:
    def _push_file(*a, **kw):  # type: ignore[misc]
        pass

# ── Storage location ──────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_LOG_PATH = _DATA_DIR / "sector_predictions.csv"

_FIELDNAMES = [
    "predicted_at", "sector", "direction", "confidence", "raw_score",
    "entry_price", "exit_price", "return_pct", "correct",
    "leader_ticker",
    "regime", "regime_confidence", "mtf_score", "mtf_note",
    "signal_agreement", "sideways_forced", "confidence_cap",
    "dynamic_weights_json", "ohlc_source", "ohlc_symbol", "stocks_used_json",
    "signal_ema_slope", "signal_price_vs_ema", "signal_candle_direction",
    "signal_body_strength", "signal_consecutive", "signal_volume_confirm",
    "signal_volatility", "signal_momentum", "signal_sector_strength",
    "signal_bullish_pct", "signal_money_flow", "signal_participation",
    # Legacy aliases kept for backward compatibility with older readers.
    "signal_volume", "signal_sector_str",
]

# ── Calibration in-memory cache (rebuilt on demand) ──────────────────
_calibration_cache: dict[str, dict[str, float]] = {}   # sector → dir → factor
_cache_built_at: str = ""
_LOG_CACHE_SIG: tuple[int, int] | None = None
_LOG_CACHE_DF: pd.DataFrame | None = None


def _ensure_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
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


def _invalidate_log_cache() -> None:
    global _LOG_CACHE_SIG, _LOG_CACHE_DF
    _LOG_CACHE_SIG = None
    _LOG_CACHE_DF = None


def _set_cached_log(df: pd.DataFrame | None) -> pd.DataFrame:
    global _LOG_CACHE_SIG, _LOG_CACHE_DF
    cached = _coerce_schema(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    _LOG_CACHE_SIG = _file_signature(_LOG_PATH)
    _LOG_CACHE_DF = cached.copy()
    return cached.copy()


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any missing columns and preserve older logs when the schema evolves.
    """
    out = df.copy()
    for col in _FIELDNAMES:
        if col not in out.columns:
            out[col] = ""
    return out[_FIELDNAMES]


def _ensure_schema() -> None:
    """
    Upgrade the CSV header in place when new columns are added.
    """
    try:
        _ensure_dir()
        if not _LOG_PATH.exists():
            _invalidate_log_cache()
            return
        df = pd.read_csv(_LOG_PATH, dtype=str)
        upgraded = _coerce_schema(df)
        if list(upgraded.columns) != list(df.columns) or len(upgraded.columns) != len(df.columns):
            upgraded.to_csv(_LOG_PATH, index=False)
        _set_cached_log(upgraded)
    except Exception:
        pass


def _plain_symbol(value: str) -> str:
    return str(value or "").upper().strip().replace(".NS", "")


def _normalize_hist(df: pd.DataFrame | None) -> pd.DataFrame | None:
    try:
        if df is None or df.empty:
            return None
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not set(needed).issubset(out.columns):
            return None
        out = out[needed].copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].sort_index()
        for col in needed:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return out if not out.empty else None
    except Exception:
        return None


def _rebuild_logged_weighted_basket(symbols: list[str], all_data: dict[str, "pd.DataFrame | None"]) -> pd.DataFrame | None:
    panels: list[tuple[str, pd.DataFrame]] = []
    for symbol in symbols:
        plain = _plain_symbol(symbol)
        hist = _normalize_hist(all_data.get(plain))
        if hist is None:
            hist = _normalize_hist(all_data.get(f"{plain}.NS"))
        if hist is None or len(hist) < 30:
            continue
        panels.append((plain, hist))

    if len(panels) < 3:
        return None

    common_index = panels[0][1].index
    for _, hist in panels[1:]:
        common_index = common_index.intersection(hist.index)
    common_index = common_index.sort_values()
    if len(common_index) < 30:
        return None

    close_panel = pd.concat(
        [hist["Close"].reindex(common_index).rename(symbol) for symbol, hist in panels],
        axis=1,
    ).dropna(axis=1, how="any")
    if close_panel.shape[1] < 3:
        return None

    symbols = list(close_panel.columns)
    open_panel = pd.concat(
        [hist["Open"].reindex(common_index).rename(symbol) for symbol, hist in panels if symbol in symbols],
        axis=1,
    )[symbols]
    high_panel = pd.concat(
        [hist["High"].reindex(common_index).rename(symbol) for symbol, hist in panels if symbol in symbols],
        axis=1,
    )[symbols]
    low_panel = pd.concat(
        [hist["Low"].reindex(common_index).rename(symbol) for symbol, hist in panels if symbol in symbols],
        axis=1,
    )[symbols]
    volume_panel = pd.concat(
        [hist["Volume"].reindex(common_index).rename(symbol) for symbol, hist in panels if symbol in symbols],
        axis=1,
    )[symbols]

    turnover = (close_panel.tail(min(20, len(close_panel))) * volume_panel.tail(min(20, len(volume_panel)))).mean(axis=0)
    turnover = pd.to_numeric(turnover, errors="coerce").clip(lower=0).fillna(0.0)
    if turnover.sum() <= 0:
        turnover = pd.Series(1.0, index=symbols)
    weights = turnover / max(turnover.sum(), 1e-9)

    base_close = close_panel.iloc[0].replace(0, np.nan)
    valid_cols = [col for col in symbols if pd.notna(base_close.get(col))]
    if len(valid_cols) < 3:
        return None

    close_panel = close_panel[valid_cols]
    open_panel = open_panel[valid_cols]
    high_panel = high_panel[valid_cols]
    low_panel = low_panel[valid_cols]
    volume_panel = volume_panel[valid_cols]
    weights = weights[valid_cols]
    weights = weights / max(weights.sum(), 1e-9)

    scale = 100.0 / base_close[valid_cols]
    agg = pd.DataFrame(
        {
            "Open": open_panel.mul(scale, axis=1).mul(weights, axis=1).sum(axis=1),
            "High": high_panel.mul(scale, axis=1).max(axis=1),
            "Low": low_panel.mul(scale, axis=1).min(axis=1),
            "Close": close_panel.mul(scale, axis=1).mul(weights, axis=1).sum(axis=1),
            "Volume": volume_panel.sum(axis=1),
        },
        index=common_index,
    )
    return _normalize_hist(agg)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC: LOG A PREDICTION
# ══════════════════════════════════════════════════════════════════════

def log_prediction(prediction) -> bool:  # prediction: SectorPrediction
    """
    Append one prediction to the CSV.  Returns True on success.

    Parameters
    ----------
    prediction : SectorPrediction  (from sector_prediction_engine)
    """
    try:
        _ensure_dir()
        _invalidate_log_cache()
        _ensure_schema()
        file_exists = _LOG_PATH.exists() and _LOG_PATH.stat().st_size > 0
        sig = prediction.signals

        row = {
            "predicted_at":      prediction.predicted_at,
            "sector":            prediction.sector,
            "direction":         prediction.direction,
            "confidence":        f"{prediction.confidence:.2f}",
            "raw_score":         f"{prediction.raw_score:.2f}",
            "entry_price":       f"{prediction.entry_price:.4f}",
            "exit_price":        "",
            "return_pct":        "",
            "correct":           "",
            "leader_ticker":     prediction.leader_ticker,
            "regime":            getattr(prediction, "regime", ""),
            "regime_confidence": f"{float(getattr(prediction, 'regime_confidence', 0.0)):.2f}",
            "mtf_score":         f"{float(getattr(prediction, 'mtf_score', 0.0)):.2f}",
            "mtf_note":          str(getattr(prediction, "mtf_note", "") or ""),
            "signal_agreement":  f"{float(getattr(prediction, 'signal_agreement', 0.0)):.2f}",
            "sideways_forced":   str(bool(getattr(prediction, "sideways_forced", False))),
            "confidence_cap":    f"{float(getattr(prediction, 'confidence_cap', 95.0)):.2f}",
            "dynamic_weights_json": json.dumps(
                getattr(prediction, "dynamic_weights", {}) or {},
                sort_keys=True,
            ),
            "ohlc_source":       str(getattr(prediction, "ohlc_source", "") or ""),
            "ohlc_symbol":       str(getattr(prediction, "ohlc_symbol", "") or ""),
            "stocks_used_json":  json.dumps(getattr(prediction, "stocks_used", []) or []),
            "signal_ema_slope":  f"{sig.ema_slope:.2f}",
            "signal_price_vs_ema": f"{sig.price_vs_ema:.2f}",
            "signal_candle_direction": f"{sig.candle_direction:.2f}",
            "signal_body_strength": f"{sig.body_strength:.2f}",
            "signal_consecutive": f"{sig.consecutive:.2f}",
            "signal_volume_confirm": f"{sig.volume_confirm:.2f}",
            "signal_volatility": f"{sig.volatility:.2f}",
            "signal_momentum":   f"{sig.momentum:.2f}",
            "signal_sector_strength": f"{sig.sector_strength:.2f}",
            "signal_bullish_pct":f"{sig.bullish_pct:.2f}",
            "signal_money_flow": f"{sig.money_flow:.2f}",
            "signal_participation": f"{sig.participation:.2f}",
            "signal_volume":     f"{sig.volume_confirm:.2f}",
            "signal_sector_str": f"{sig.sector_strength:.2f}",
        }

        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        _push_file(_LOG_PATH)
        try:
            read_log()
        except Exception:
            pass
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# PUBLIC: BACKFILL OUTCOMES
# ══════════════════════════════════════════════════════════════════════

def backfill_outcomes(all_data: dict[str, "pd.DataFrame | None"]) -> int:
    """
    Fill exit_price / return_pct / correct for rows where these are blank.

    Uses ALL_DATA so zero API calls.
    Returns number of rows filled.
    """
    try:
        _ensure_schema()
        if not _LOG_PATH.exists():
            _invalidate_log_cache()
            return 0
        df = read_log()
        if df.empty:
            return 0

        needs = df["exit_price"].apply(lambda x: str(x).strip() == "")
        if not needs.any():
            return 0

        filled = 0
        for idx in df.index[needs]:
            try:
                source = str(df.at[idx, "ohlc_source"]).strip()
                ohlc_symbol = str(df.at[idx, "ohlc_symbol"]).strip()
                ticker = ohlc_symbol or str(df.at[idx, "leader_ticker"]).strip()

                if source == "weighted_sector_basket":
                    try:
                        members = json.loads(str(df.at[idx, "stocks_used_json"]).strip() or "[]")
                    except Exception:
                        members = []
                    if not isinstance(members, list):
                        members = []
                    hist = _rebuild_logged_weighted_basket([str(item) for item in members], all_data)
                else:
                    if not ticker:
                        continue
                    lookup_keys = [ticker]
                    if not ticker.startswith("^") and not ticker.endswith(".NS"):
                        lookup_keys.append(f"{ticker}.NS")
                    hist = next(
                        (_normalize_hist(all_data.get(key)) for key in lookup_keys if all_data.get(key) is not None),
                        None,
                    )
                if hist is None or "Close" not in hist.columns or len(hist) < 2:
                    continue

                pred_str = str(df.at[idx, "predicted_at"]).strip()
                pred_dt  = pd.to_datetime(pred_str, errors="coerce", utc=True)
                if pd.isnull(pred_dt):
                    continue
                pred_date = pred_dt.date()

                dates = pd.to_datetime(hist.index).date
                arr   = np.array(dates)
                locs  = np.where(arr <= pred_date)[0]
                if len(locs) == 0:
                    continue
                day_i = int(locs[-1])
                if day_i + 1 >= len(hist):
                    continue

                entry = pd.to_numeric(df.at[idx, "entry_price"], errors="coerce")
                entry = float(entry) if pd.notna(entry) else float(hist["Close"].iloc[day_i])
                exit_ = float(hist["Close"].iloc[day_i + 1])
                if entry <= 0:
                    continue

                ret = round((exit_ / entry - 1.0) * 100, 4)
                direction = str(df.at[idx, "direction"]).strip()

                if direction == "Bullish":
                    correct = ret > 0.5
                elif direction == "Bearish":
                    correct = ret < -0.5
                else:  # Sideways
                    correct = abs(ret) <= 0.5

                df.at[idx, "exit_price"] = f"{exit_:.4f}"
                df.at[idx, "return_pct"] = f"{ret:.4f}"
                df.at[idx, "correct"]    = str(correct)
                filled += 1
            except Exception:
                continue

        if filled > 0:
            df.to_csv(_LOG_PATH, index=False)
            _push_file(_LOG_PATH)
            _set_cached_log(df)
            _rebuild_calibration_cache(df)
        return filled
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════
# CALIBRATION FACTOR
# ══════════════════════════════════════════════════════════════════════

def _rebuild_calibration_cache(df: pd.DataFrame) -> None:
    """
    Build sector × direction → accuracy-based adjustment factor.

    factor = actual_accuracy_rate / 0.65
    (0.65 is assumed prior accuracy for an uncalibrated model)

    Clipped to [0.6, 1.4] so calibration can't swing too wildly.
    Only computed for (sector, direction) pairs with ≥ 10 outcomes.
    """
    global _calibration_cache, _cache_built_at
    cache: dict[str, dict[str, float]] = {}

    try:
        sub = df[df["correct"].isin(["True", "False"])].copy()
        sub["_ok"] = sub["correct"] == "True"
        for (sector, direction), grp in sub.groupby(["sector", "direction"]):
            if len(grp) < 10:
                continue
            acc = float(grp["_ok"].mean())
            factor = float(np.clip(acc / 0.65, 0.6, 1.4))
            cache.setdefault(sector, {})[direction] = factor
    except Exception:
        pass

    _calibration_cache = cache
    _cache_built_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def get_calibration_factor(sector: str, direction: str) -> float:
    """
    Return the calibration factor for a sector+direction pair.
    1.0 = no adjustment; > 1 = model was over-confident; < 1 = under-confident.
    """
    if not _calibration_cache:
        try:
            if _LOG_PATH.exists():
                df = pd.read_csv(_LOG_PATH, dtype=str)
                _rebuild_calibration_cache(df)
        except Exception:
            pass
    return _calibration_cache.get(sector, {}).get(direction, 1.0)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC: READ LOG
# ══════════════════════════════════════════════════════════════════════

def read_log(sector: str | None = None) -> pd.DataFrame:
    """
    Return the full prediction log as a DataFrame.
    If sector is given, filter to that sector only.
    """
    try:
        _ensure_schema()
        if not _LOG_PATH.exists():
            _invalidate_log_cache()
            return pd.DataFrame(columns=_FIELDNAMES)
        current_sig = _file_signature(_LOG_PATH)
        if current_sig is not None and _LOG_CACHE_SIG == current_sig and isinstance(_LOG_CACHE_DF, pd.DataFrame):
            df = _LOG_CACHE_DF.copy()
        else:
            df = _set_cached_log(pd.read_csv(_LOG_PATH, dtype=str))
        if sector:
            df = df[df["sector"] == sector].copy()
        return df
    except Exception:
        return pd.DataFrame(columns=_FIELDNAMES)


def recent_predictions(sector: str, n: int = 5) -> pd.DataFrame:
    """Return the last n predictions for a given sector."""
    df = read_log(sector)
    if df.empty:
        return df
    # Sort descending by timestamp
    df = df.sort_values("predicted_at", ascending=False).head(n)
    return df.reset_index(drop=True)
