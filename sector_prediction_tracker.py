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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Storage location ──────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_LOG_PATH = _DATA_DIR / "sector_predictions.csv"

_FIELDNAMES = [
    "predicted_at", "sector", "direction", "confidence", "raw_score",
    "entry_price", "exit_price", "return_pct", "correct",
    "leader_ticker",
    "signal_ema_slope", "signal_momentum", "signal_volume",
    "signal_sector_str", "signal_bullish_pct",
]

# ── Calibration in-memory cache (rebuilt on demand) ──────────────────
_calibration_cache: dict[str, dict[str, float]] = {}   # sector → dir → factor
_cache_built_at: str = ""


def _ensure_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


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
        file_exists = _LOG_PATH.exists()
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
            "signal_ema_slope":  f"{sig.ema_slope:.2f}",
            "signal_momentum":   f"{sig.momentum:.2f}",
            "signal_volume":     f"{sig.volume_confirm:.2f}",
            "signal_sector_str": f"{sig.sector_strength:.2f}",
            "signal_bullish_pct":f"{sig.bullish_pct:.2f}",
        }

        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
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
        if not _LOG_PATH.exists():
            return 0
        df = pd.read_csv(_LOG_PATH, dtype=str)
        if df.empty:
            return 0

        needs = df["exit_price"].apply(lambda x: str(x).strip() == "")
        if not needs.any():
            return 0

        filled = 0
        for idx in df.index[needs]:
            try:
                ticker = str(df.at[idx, "leader_ticker"]).strip()
                if not ticker:
                    continue
                tk_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
                hist = all_data.get(tk_ns)
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

                entry = float(hist["Close"].iloc[day_i])
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
        if not _LOG_PATH.exists():
            return pd.DataFrame(columns=_FIELDNAMES)
        df = pd.read_csv(_LOG_PATH, dtype=str)
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