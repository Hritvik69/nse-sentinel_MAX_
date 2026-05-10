"""
sector_evaluation_engine.py   (v2 — Institutional Grade)
══════════════════════════════════════════════════════════
Layer 5 — Full Performance Evaluation.

Produces:
  A. Accuracy        per-sector, overall, regime-split
  B. Returns         avg, cumulative, expectancy, W/L
  C. Risk            Sharpe (approx), max drawdown, streaks, equity curve
  D. Calibration     confidence vs actual accuracy buckets
  E. Stability       walk-forward rolling accuracy & return curves
  F. Signal quality  dynamic weight report from tracker

Public API
──────────
  compute_full_evaluation()        → EvaluationReport
  compute_sector_report(sector)    → SectorReport
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
class SectorReport:
    sector:        str
    total:         int   = 0
    correct:       int   = 0
    accuracy_pct:  float = 0.0
    avg_return:    float = 0.0
    best_return:   float = 0.0
    worst_return:  float = 0.0
    sharpe:        float = 0.0
    max_drawdown:  float = 0.0
    expectancy:    float = 0.0


@dataclass
class CalibrationBucket:
    label:          str
    count:          int
    avg_confidence: float
    actual_accuracy:float


@dataclass
class EvaluationReport:
    # A. Accuracy
    total_predictions:   int   = 0
    validated:           int   = 0
    correct:             int   = 0
    accuracy_pct:        float = 0.0
    per_sector:          list[SectorReport] = field(default_factory=list)
    best_sector:         str = ""
    worst_sector:        str = ""

    # B. Returns
    avg_return_pct:      float = 0.0
    cumulative_return:   float = 0.0
    win_count:           int   = 0
    loss_count:          int   = 0
    win_loss_ratio:      float = 0.0
    best_trade:          float = 0.0
    worst_trade:         float = 0.0
    expectancy:          float = 0.0      # avg R-multiple

    # C. Risk
    sharpe_approx:       float = 0.0
    max_drawdown:        float = 0.0
    return_volatility:   float = 0.0
    max_win_streak:      int   = 0
    max_loss_streak:     int   = 0
    equity_curve:        list[float] = field(default_factory=list)
    equity_dates:        list[str]   = field(default_factory=list)

    # D. Calibration
    calibration_buckets: list[CalibrationBucket] = field(default_factory=list)
    calibration_score:   float = 0.0

    # E. Stability (walk-forward)
    wf_overall_accuracy:  float = 0.0
    wf_stability_score:   float = 0.0
    wf_rolling_accuracy:  list[float] = field(default_factory=list)
    wf_rolling_returns:   list[float] = field(default_factory=list)
    wf_fold_dates:        list[str]   = field(default_factory=list)
    wf_note:              str = ""

    # F. Signal quality
    signal_perf_df:       pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_perf_df:       pd.DataFrame = field(default_factory=pd.DataFrame)

    # Meta
    last_10:              pd.DataFrame = field(default_factory=pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _f(x: object) -> float | None:
    try:
        s = str(x).strip()
        if s in ("", "nan", "None"):
            return None
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _max_dd(returns: list[float]) -> float:
    if not returns:
        return 0.0
    eq   = np.cumprod([1 + r / 100 for r in returns])
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / (peak + 1e-9) * 100
    return float(dd.min())


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return round(float(np.mean(returns) / (np.std(returns) + 1e-9) * math.sqrt(252)), 2)


def _equity_curve(returns: list[float]) -> list[float]:
    eq = [100.0]
    for r in returns:
        eq.append(round(eq[-1] * (1 + r / 100), 4))
    return eq


def _calibration(df: pd.DataFrame) -> tuple[list[CalibrationBucket], float]:
    buckets: list[CalibrationBucket] = []
    if df.empty or "confidence" not in df.columns:
        return buckets, 0.0
    df = df.copy()
    df["_c"] = df["confidence"].apply(_f)
    df["_o"] = df["correct"].apply(lambda x: True if str(x).strip() == "True" else
                                    (False if str(x).strip() == "False" else None))
    sub = df[df["_c"].notna() & df["_o"].notna()].copy()
    if sub.empty:
        return buckets, 0.0
    edges  = [0, 50, 60, 70, 80, 90, 100]
    labels = ["<50%", "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
    sub["_bkt"] = pd.cut(sub["_c"], bins=edges, labels=labels, right=True)
    maes: list[float] = []
    for lbl in labels:
        g = sub[sub["_bkt"] == lbl]
        if g.empty:
            continue
        avg_c  = float(g["_c"].mean())
        actual = float(g["_o"].mean() * 100)
        buckets.append(CalibrationBucket(lbl, len(g), round(avg_c, 1), round(actual, 1)))
        maes.append(abs(avg_c - actual))
    return buckets, round(float(np.mean(maes)), 2) if maes else 0.0


def _streaks(outcomes: list[int]) -> tuple[int, int]:
    mw = ml = cw = cl = 0
    for o in outcomes:
        if o:
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0
        mw = max(mw, cw); ml = max(ml, cl)
    return mw, ml


def _regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise validated outcomes by market regime when that field is present.
    """
    if df.empty or "regime" not in df.columns:
        return pd.DataFrame()

    val = df[df["correct"].isin(["True", "False"])].copy()
    if val.empty:
        return pd.DataFrame()

    val["_ok"] = val["correct"] == "True"
    val["_ret"] = val["return_pct"].apply(_f)

    rows = []
    for regime, grp in val.groupby("regime"):
        rets = [r for r in grp["_ret"].tolist() if r is not None]
        rows.append(
            {
                "Regime": str(regime),
                "Preds": int(len(grp)),
                "Accuracy": round(float(grp["_ok"].mean() * 100), 2),
                "Avg Ret": round(float(np.mean(rets)), 3) if rets else 0.0,
                "Best": round(float(max(rets)), 3) if rets else 0.0,
                "Worst": round(float(min(rets)), 3) if rets else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# SECTOR REPORT
# ══════════════════════════════════════════════════════════════════════

def _sector_report(sector: str, sdf: pd.DataFrame) -> SectorReport:
    val = sdf[sdf["correct"].isin(["True", "False"])].copy()
    if val.empty:
        return SectorReport(sector=sector, total=len(sdf))
    val["_ok"]  = val["correct"] == "True"
    val["_ret"] = val["return_pct"].apply(_f)
    rets = [r for r in val["_ret"].tolist() if r is not None]
    return SectorReport(
        sector       = sector,
        total        = len(sdf),
        correct      = int(val["_ok"].sum()),
        accuracy_pct = round(float(val["_ok"].mean() * 100), 2),
        avg_return   = round(float(np.mean(rets)), 3) if rets else 0.0,
        best_return  = round(float(max(rets)), 3) if rets else 0.0,
        worst_return = round(float(min(rets)), 3) if rets else 0.0,
        sharpe       = _sharpe(rets),
        max_drawdown = round(_max_dd(rets), 2),
        expectancy   = round(float(np.mean(rets)), 3) if rets else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def compute_full_evaluation() -> EvaluationReport:
    report = EvaluationReport()
    try:
        from sector_prediction_tracker import read_log
        df = read_log()
        if df.empty:
            return report

        # ── Update signal performance from latest log ──────────────────
        try:
            from sector_dynamic_weights import update_signal_performance
            update_signal_performance(df)
        except Exception:
            pass

        report.total_predictions = len(df)
        val = df[df["correct"].isin(["True", "False"])].copy()
        report.validated = len(val)

        if val.empty:
            report.last_10 = df.sort_values("predicted_at", ascending=False).head(10)
            return report

        val["_ok"]  = val["correct"] == "True"
        val["_ret"] = val["return_pct"].apply(_f)
        val["_dt"]  = pd.to_datetime(val["predicted_at"], errors="coerce", utc=True)
        val = val.sort_values("_dt").reset_index(drop=True)

        # ── A. Accuracy ───────────────────────────────────────────────
        report.correct      = int(val["_ok"].sum())
        report.accuracy_pct = round(float(val["_ok"].mean() * 100), 2)

        sec_reports = []
        for sec, grp in df.groupby("sector"):
            sec_reports.append(_sector_report(str(sec), grp))
        sec_reports.sort(key=lambda s: s.accuracy_pct, reverse=True)
        report.per_sector   = sec_reports
        report.best_sector  = sec_reports[0].sector  if sec_reports else ""
        report.worst_sector = sec_reports[-1].sector if sec_reports else ""

        # ── B. Returns ────────────────────────────────────────────────
        rets = [r for r in val["_ret"].tolist() if r is not None]
        if rets:
            report.avg_return_pct    = round(float(np.mean(rets)), 3)
            report.cumulative_return = round(float((np.prod([1+r/100 for r in rets]) - 1) * 100), 2)
            report.win_count         = int(sum(1 for r in rets if r > 0))
            report.loss_count        = int(sum(1 for r in rets if r <= 0))
            report.win_loss_ratio    = round(report.win_count / max(report.loss_count, 1), 2)
            report.best_trade        = round(float(max(rets)), 3)
            report.worst_trade       = round(float(min(rets)), 3)
            report.expectancy        = round(float(np.mean(rets)), 3)

        # ── C. Risk ───────────────────────────────────────────────────
        if rets:
            report.sharpe_approx    = _sharpe(rets)
            report.max_drawdown     = round(_max_dd(rets), 2)
            report.return_volatility= round(float(np.std(rets)), 3)
            mw, ml = _streaks([1 if r > 0 else 0 for r in rets])
            report.max_win_streak   = mw
            report.max_loss_streak  = ml
            report.equity_curve     = _equity_curve(rets)
            report.equity_dates     = [str(d)[:10] for d in val["_dt"].tolist() if d is not pd.NaT]

        # ── D. Calibration ────────────────────────────────────────────
        report.calibration_buckets, report.calibration_score = _calibration(val)

        # ── E. Walk-forward stability ─────────────────────────────────
        try:
            from sector_walkforward import run_walk_forward
            wf = run_walk_forward(df)
            report.wf_overall_accuracy = wf.overall_accuracy
            report.wf_stability_score  = wf.stability_score
            report.wf_rolling_accuracy = wf.rolling_accuracy
            report.wf_rolling_returns  = wf.rolling_returns
            report.wf_fold_dates       = wf.fold_dates
            report.wf_note             = wf.note
        except Exception as e:
            report.wf_note = f"Walk-forward unavailable: {e}"

        # ── F. Signal quality ─────────────────────────────────────────
        try:
            from sector_dynamic_weights import get_signal_performance_report
            report.signal_perf_df = get_signal_performance_report()
        except Exception:
            pass

        # ── Last 10 ───────────────────────────────────────────────────
        report.regime_perf_df = _regime_summary(df)
        report.last_10 = df.sort_values("predicted_at", ascending=False).head(10)

    except Exception:
        pass
    return report


def compute_sector_report(sector: str) -> SectorReport:
    try:
        from sector_prediction_tracker import read_log
        return _sector_report(sector, read_log(sector))
    except Exception:
        return SectorReport(sector=sector)
