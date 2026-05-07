"""
sector_walkforward.py
══════════════════════
Module 4 — Walk-Forward Validation Engine.

Validates the prediction model without look-ahead bias by rolling
a train window forward and evaluating on a held-out window.

  Train window : 60 trading days
  Eval window  : 15 trading days
  Step size    : 15 trading days

Outputs
───────
  per-fold accuracy, returns, stability score, rolling curves

Public API
──────────
  run_walk_forward(log_df)        → WalkForwardReport
  get_stability_score(log_df)     → float (0–100)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardFold:
    fold:           int
    train_start:    str
    train_end:      str
    eval_start:     str
    eval_end:       str
    eval_n:         int
    accuracy_pct:   float
    avg_return_pct: float
    cum_return_pct: float


@dataclass
class WalkForwardReport:
    folds:             list[WalkForwardFold] = field(default_factory=list)
    overall_accuracy:  float = 0.0
    overall_return:    float = 0.0
    raw_stability_score: float = 0.0
    stability_score:   float = 0.0      # 0–100; higher = more consistent
    max_fold_dd:       float = 0.0      # worst single fold
    accuracy_std:      float = 0.0      # consistency of accuracy across folds
    rolling_accuracy:  list[float] = field(default_factory=list)
    rolling_returns:   list[float] = field(default_factory=list)
    fold_dates:        list[str]   = field(default_factory=list)
    note:              str = ""

    @property
    def n_folds(self) -> int:
        return len(self.folds)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _to_float(x: object) -> float | None:
    try:
        s = str(x).strip()
        if s in ("", "nan", "None"):
            return None
        f = float(s)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _parse_log(log_df: pd.DataFrame) -> pd.DataFrame | None:
    """Clean and type-cast the prediction log for walk-forward use."""
    try:
        df = log_df.copy()
        df["_dt"] = pd.to_datetime(df["predicted_at"], errors="coerce", utc=True)
        df["_ok"] = df["correct"].apply(lambda x: True if str(x).strip() == "True" else
                                         (False if str(x).strip() == "False" else None))
        df["_ret"] = df["return_pct"].apply(_to_float)
        df = df[df["_dt"].notna() & df["_ok"].notna() & df["_ret"].notna()]
        if df.empty:
            return None
        df = df.sort_values("_dt").reset_index(drop=True)
        return df
    except Exception:
        return None


def _fold_stats(fold_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (accuracy_pct, avg_return_pct, cum_return_pct) for a fold."""
    if fold_df.empty:
        return 0.0, 0.0, 0.0
    acc  = float(fold_df["_ok"].mean() * 100)
    rets = fold_df["_ret"].tolist()
    avg  = float(np.mean(rets))
    cum  = float((np.prod([1 + r / 100 for r in rets]) - 1.0) * 100) if rets else 0.0
    return round(acc, 2), round(avg, 3), round(cum, 2)


def _raw_stability_score(accuracies: list[float], returns: list[float]) -> float:
    """
    Raw stability score before clipping to the 0–100 display range.

    Penalises:
      • high variance in accuracy across folds
      • large negative folds (drawdown events)
    """
    if len(accuracies) < 2:
        return 50.0

    acc_std    = float(np.std(accuracies))
    acc_mean   = float(np.mean(accuracies))
    neg_folds  = sum(1 for r in returns if r < 0)
    neg_pct    = neg_folds / len(returns) * 100

    # Penalty = acc variability + losing fold frequency
    penalty    = acc_std * 1.5 + neg_pct * 0.5
    return float(acc_mean - penalty)


def _stability(accuracies: list[float], returns: list[float]) -> float:
    """
    Stability score (0–100).

    A return value of 0.0 means the raw stability score was <= 0 before
    clipping, not that model stability is exactly zero.
    """
    raw_score = _raw_stability_score(accuracies, returns)
    return round(float(np.clip(raw_score, 0, 100)), 1)


# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════════════

def run_walk_forward(
    log_df: pd.DataFrame,
    train_window: int = 60,
    eval_window:  int = 15,
    step:         int = 15,
) -> WalkForwardReport:
    """
    Execute walk-forward validation on the prediction log.

    Parameters
    ----------
    log_df       : pd.DataFrame   Full prediction log from tracker.
    train_window : int            Training window in trading days.
    eval_window  : int            Evaluation window in trading days.
    step         : int            Roll-forward step in trading days.

    Returns
    -------
    WalkForwardReport
    """
    report = WalkForwardReport()

    try:
        df = _parse_log(log_df)
        if df is None or len(df) < train_window + eval_window:
            report.note = (
                f"Insufficient validated predictions for walk-forward "
                f"(need ≥ {train_window + eval_window}, have {len(log_df)})."
            )
            return report

        folds:      list[WalkForwardFold]  = []
        acc_list:   list[float] = []
        ret_list:   list[float] = []
        date_list:  list[str]   = []

        n   = len(df)
        i   = 0
        fold_num = 0

        while i + train_window + eval_window <= n:
            train_df = df.iloc[i : i + train_window]
            eval_df  = df.iloc[i + train_window : i + train_window + eval_window]

            if eval_df.empty:
                break

            fold_num += 1
            acc, avg_ret, cum_ret = _fold_stats(eval_df)

            fold = WalkForwardFold(
                fold           = fold_num,
                train_start    = str(train_df["_dt"].iloc[0])[:10],
                train_end      = str(train_df["_dt"].iloc[-1])[:10],
                eval_start     = str(eval_df["_dt"].iloc[0])[:10],
                eval_end       = str(eval_df["_dt"].iloc[-1])[:10],
                eval_n         = len(eval_df),
                accuracy_pct   = acc,
                avg_return_pct = avg_ret,
                cum_return_pct = cum_ret,
            )
            folds.append(fold)
            acc_list.append(acc)
            ret_list.append(cum_ret)
            date_list.append(fold.eval_start)

            i += step

        if not folds:
            report.note = "Walk-forward produced no folds — not enough data."
            return report

        raw_stability = _raw_stability_score(acc_list, ret_list)
        report.folds            = folds
        report.overall_accuracy = round(float(np.mean(acc_list)), 2)
        report.overall_return   = round(float(np.mean(ret_list)), 2)
        report.raw_stability_score = round(float(raw_stability), 2)
        report.stability_score  = _stability(acc_list, ret_list)
        report.max_fold_dd      = round(float(min(ret_list)), 2)
        report.accuracy_std     = round(float(np.std(acc_list)), 2)
        report.rolling_accuracy = [round(a, 2) for a in acc_list]
        report.rolling_returns  = [round(r, 2) for r in ret_list]
        report.fold_dates       = date_list
        note_parts = [
            f"{fold_num} folds · train={train_window} eval={eval_window} step={step} bars"
        ]
        if raw_stability < 0:
            note_parts.append(
                f"Stability was clipped to 0.0 (raw score: {raw_stability:.1f}). "
                "Model is unstable across folds."
            )
        elif raw_stability < 20:
            note_parts.append(f"Low stability (raw: {raw_stability:.1f}).")
        report.note = " ".join(note_parts)

    except Exception as exc:
        report.note = f"Walk-forward error: {exc}"

    return report


def get_stability_score(log_df: pd.DataFrame) -> float:
    """
    Convenience wrapper — returns only the stability score (0–100).
    Returns 50 when insufficient data.
    """
    try:
        r = run_walk_forward(log_df)
        return r.stability_score if r.n_folds >= 2 else 50.0
    except Exception:
        return 50.0
