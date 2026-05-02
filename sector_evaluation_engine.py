"""
sector_evaluation_engine.py
═════════════════════════════
Layer 5 — Continuous Model Evaluation.

Reads the tracker CSV and produces:

  A. Accuracy        total, % correct, per-sector
  B. Returns         avg return per trade, cumulative, win/loss
  C. Risk            max drawdown, volatility
  D. Calibration     does confidence ≈ actual accuracy?

Public API
──────────
    compute_full_evaluation()     → EvaluationReport
    compute_sector_report(sector) → SectorReport
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sector_prediction_tracker import read_log
except ImportError:
    def read_log(sector=None):
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SectorReport:
    sector:        str
    total:         int   = 0
    correct:       int   = 0
    accuracy_pct:  float = 0.0
    avg_return:    float = 0.0
    best_return:   float = 0.0
    worst_return:  float = 0.0


@dataclass
class CalibrationBucket:
    label:          str    # e.g. "60–70%"
    count:          int
    avg_confidence: float
    actual_accuracy:float


@dataclass
class EvaluationReport:
    # ── A. Accuracy ───────────────────────────────────────────────────
    total_predictions:   int   = 0
    validated:           int   = 0      # rows with an outcome
    correct:             int   = 0
    accuracy_pct:        float = 0.0
    per_sector:          list[SectorReport] = field(default_factory=list)

    # ── B. Returns ────────────────────────────────────────────────────
    avg_return_pct:      float = 0.0
    cumulative_return:   float = 0.0
    win_count:           int   = 0
    loss_count:          int   = 0
    win_loss_ratio:      float = 0.0
    best_trade:          float = 0.0
    worst_trade:         float = 0.0

    # ── C. Risk ───────────────────────────────────────────────────────
    max_drawdown:        float = 0.0
    return_volatility:   float = 0.0

    # ── D. Calibration ───────────────────────────────────────────────
    calibration_buckets: list[CalibrationBucket] = field(default_factory=list)
    calibration_score:   float = 0.0   # 0 = perfectly calibrated, higher = worse

    # ── Meta ─────────────────────────────────────────────────────────
    last_10:             pd.DataFrame = field(default_factory=pd.DataFrame)
    best_sector:         str = ""
    worst_sector:        str = ""


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


def _max_drawdown(returns: list[float]) -> float:
    """Maximum peak-to-trough drawdown from a list of per-trade returns (%)."""
    if not returns:
        return 0.0
    equity = np.cumprod([1 + r / 100 for r in returns])
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9) * 100
    return float(dd.min())   # negative number → max loss


def _calibration(df: pd.DataFrame) -> tuple[list[CalibrationBucket], float]:
    """
    Group predictions by confidence decile, compare avg confidence to
    actual accuracy within each bucket.

    Returns (buckets, calibration_score).
    calibration_score = mean absolute error between confidence and accuracy.
    """
    buckets: list[CalibrationBucket] = []
    if df.empty:
        return buckets, 0.0

    df = df.copy()
    df["_conf"] = df["confidence"].apply(_to_float)
    df["_ok"]   = df["correct"].apply(lambda x: True if str(x).strip() == "True" else
                                      (False if str(x).strip() == "False" else None))
    sub = df[df["_conf"].notna() & df["_ok"].notna()].copy()
    if sub.empty:
        return buckets, 0.0

    edges = [0, 50, 60, 70, 80, 90, 100]
    labels_bkt = ["<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
    sub["_bucket"] = pd.cut(sub["_conf"], bins=edges, labels=labels_bkt, right=True)

    mae_vals: list[float] = []
    for lbl in labels_bkt:
        grp = sub[sub["_bucket"] == lbl]
        if grp.empty:
            continue
        avg_conf = float(grp["_conf"].mean())
        actual   = float(grp["_ok"].mean() * 100)
        buckets.append(CalibrationBucket(
            label=lbl,
            count=len(grp),
            avg_confidence=round(avg_conf, 1),
            actual_accuracy=round(actual, 1),
        ))
        mae_vals.append(abs(avg_conf - actual))

    cal_score = float(np.mean(mae_vals)) if mae_vals else 0.0
    return buckets, round(cal_score, 2)


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def compute_full_evaluation() -> EvaluationReport:
    """
    Read the full prediction log and compute all evaluation metrics.
    Never raises — returns an empty EvaluationReport on any error.
    """
    report = EvaluationReport()
    try:
        df = read_log()
        if df.empty:
            return report

        report.total_predictions = len(df)

        # ── Filter to validated rows ──────────────────────────────────
        validated = df[df["correct"].isin(["True", "False"])].copy()
        report.validated = len(validated)
        if validated.empty:
            report.last_10 = df.sort_values("predicted_at", ascending=False).head(10)
            return report

        validated["_ok"]  = validated["correct"] == "True"
        validated["_ret"] = validated["return_pct"].apply(_to_float)

        # ── A. Accuracy ───────────────────────────────────────────────
        report.correct       = int(validated["_ok"].sum())
        report.accuracy_pct  = round(float(validated["_ok"].mean() * 100), 2)

        # Per-sector
        sector_reports: list[SectorReport] = []
        for sector, grp in validated.groupby("sector"):
            ret_vals = grp["_ret"].dropna().tolist()
            sr = SectorReport(
                sector       = str(sector),
                total        = len(grp),
                correct      = int(grp["_ok"].sum()),
                accuracy_pct = round(float(grp["_ok"].mean() * 100), 2),
                avg_return   = round(float(np.mean(ret_vals)), 3) if ret_vals else 0.0,
                best_return  = round(float(np.max(ret_vals)), 3) if ret_vals else 0.0,
                worst_return = round(float(np.min(ret_vals)), 3) if ret_vals else 0.0,
            )
            sector_reports.append(sr)
        sector_reports.sort(key=lambda x: x.accuracy_pct, reverse=True)
        report.per_sector  = sector_reports
        report.best_sector  = sector_reports[0].sector  if sector_reports else ""
        report.worst_sector = sector_reports[-1].sector if sector_reports else ""

        # ── B. Returns ────────────────────────────────────────────────
        rets = validated["_ret"].dropna().tolist()
        if rets:
            report.avg_return_pct    = round(float(np.mean(rets)), 3)
            report.cumulative_return = round(float(
                (np.prod([1 + r / 100 for r in rets]) - 1.0) * 100
            ), 2)
            report.win_count         = int(sum(1 for r in rets if r > 0))
            report.loss_count        = int(sum(1 for r in rets if r <= 0))
            report.win_loss_ratio    = round(
                report.win_count / max(report.loss_count, 1), 2
            )
            report.best_trade        = round(float(max(rets)), 3)
            report.worst_trade       = round(float(min(rets)), 3)

        # ── C. Risk ───────────────────────────────────────────────────
        if rets:
            report.max_drawdown      = round(_max_drawdown(rets), 2)
            report.return_volatility = round(float(np.std(rets)), 3)

        # ── D. Calibration ────────────────────────────────────────────
        report.calibration_buckets, report.calibration_score = _calibration(validated)

        # ── Last 10 ───────────────────────────────────────────────────
        report.last_10 = df.sort_values("predicted_at", ascending=False).head(10)

    except Exception:
        pass

    return report


def compute_sector_report(sector: str) -> SectorReport:
    """Compute a SectorReport for a single sector."""
    try:
        df = read_log(sector)
        if df.empty:
            return SectorReport(sector=sector)
        validated = df[df["correct"].isin(["True", "False"])].copy()
        if validated.empty:
            return SectorReport(sector=sector, total=len(df))
        validated["_ok"]  = validated["correct"] == "True"
        validated["_ret"] = validated["return_pct"].apply(_to_float)
        rets = validated["_ret"].dropna().tolist()
        return SectorReport(
            sector       = sector,
            total        = len(df),
            correct      = int(validated["_ok"].sum()),
            accuracy_pct = round(float(validated["_ok"].mean() * 100), 2),
            avg_return   = round(float(np.mean(rets)), 3) if rets else 0.0,
            best_return  = round(float(max(rets)), 3) if rets else 0.0,
            worst_return = round(float(min(rets)), 3) if rets else 0.0,
        )
    except Exception:
        return SectorReport(sector=sector)