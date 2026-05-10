"""
sector_risk_engine.py
══════════════════════
Module 5 — Risk Management Simulator.

Simulates trade outcomes with explicit stop-loss and take-profit.
Tracks R-multiples, drawdown, loss streaks, and enforces portfolio-level
position limits and sector correlation constraints.

Public API
──────────
  simulate_trade(entry, exit_next, direction, stop_pct, tp_pct)
      → TradeResult

  portfolio_risk_check(active_signals)
      → PortfolioCheck

  compute_risk_metrics(log_df, stop_pct, tp_pct)
      → RiskReport
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# DEFAULTS
# ══════════════════════════════════════════════════════════════════════

DEFAULT_STOP_PCT = 1.0    # -1.0% stop loss
DEFAULT_TP_PCT   = 2.0    # +2.0% take profit
MAX_SIMULTANEOUS = 4      # max open sector positions
MAX_PORTFOLIO_DD = -8.0   # halt if portfolio drawdown > 8%

# Correlated sector groups — avoid doubling up
_CORR_GROUPS: list[set[str]] = [
    {"BANKING", "NBFC_FINANCE"},
    {"IT", "TELECOM"},
    {"METAL", "ENERGY"},
    {"PHARMA", "CHEMICAL"},
    {"CAPITAL_GOODS", "INFRA", "DEFENCE", "RAILWAY"},
    {"AUTO", "CONSUMER_DURABLES"},
    {"FMCG", "REALTY"},
]


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    entry_price:   float
    exit_price:    float     # actual exit (stop / tp / close)
    raw_return:    float     # (exit/entry − 1) × 100
    capped_return: float     # after stop/tp caps
    hit_stop:      bool
    hit_tp:        bool
    r_multiple:    float     # (actual_profit) / (stop_distance)
    direction:     str


@dataclass
class PortfolioCheck:
    allowed:           bool
    reason:            str
    active_count:      int
    correlated_sector: str | None = None


@dataclass
class RiskReport:
    n_trades:         int   = 0
    stop_hit_pct:     float = 0.0
    tp_hit_pct:       float = 0.0
    avg_r_multiple:   float = 0.0
    win_r_pct:        float = 0.0    # % trades with R > 0
    max_drawdown_pct: float = 0.0
    expectancy:       float = 0.0    # avg P&L per trade (using R-multiples)
    sharpe_approx:    float = 0.0
    max_loss_streak:  int   = 0
    max_win_streak:   int   = 0
    avg_win_r:        float = 0.0
    avg_loss_r:       float = 0.0
    # Simulated equity curve
    equity_curve:     list[float] = field(default_factory=list)
    trade_dates:      list[str]   = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════════

def simulate_trade(
    entry_price: float,
    exit_close:  float,      # next-session close (as currently tracked)
    direction:   str,
    stop_pct:    float = DEFAULT_STOP_PCT,
    tp_pct:      float = DEFAULT_TP_PCT,
) -> TradeResult:
    """
    Simulate one trade with stop-loss and take-profit.

    We don't have intrabar data so we approximate:
    - If the session moved against us by ≥ stop_pct → assume stop triggered
      at entry × (1 − stop/100) for Bullish (or + for Bearish).
    - If the session moved in our favour by ≥ tp_pct → assume TP triggered.
    - Otherwise exit at close.
    """
    if entry_price <= 0:
        return TradeResult(entry_price, exit_close, 0, 0, False, False, 0, direction)

    raw_ret = (exit_close / entry_price - 1.0) * 100
    if direction == "Bearish":
        raw_ret = -raw_ret   # short trade: profit when price falls

    stop_dist = stop_pct    # risk = stop_pct %
    tp_dist   = tp_pct

    hit_stop = raw_ret <= -stop_dist
    hit_tp   = raw_ret >= tp_dist

    if hit_stop:
        capped = -stop_dist
    elif hit_tp:
        capped = tp_dist
    else:
        capped = raw_ret

    r_multiple = round(capped / (stop_dist + 1e-9), 3)

    return TradeResult(
        entry_price   = entry_price,
        exit_price    = exit_close,
        raw_return    = round(raw_ret, 4),
        capped_return = round(capped, 4),
        hit_stop      = hit_stop,
        hit_tp        = hit_tp,
        r_multiple    = r_multiple,
        direction     = direction,
    )


# ══════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK CHECK
# ══════════════════════════════════════════════════════════════════════

def portfolio_risk_check(
    candidate_sector: str,
    active_sectors:   list[str],
    portfolio_dd:     float = 0.0,    # current portfolio drawdown (negative)
) -> PortfolioCheck:
    """
    Decide whether a new sector prediction should be acted on given:
      • Max simultaneous position cap
      • Correlated sector exposure
      • Portfolio drawdown halt

    Parameters
    ----------
    candidate_sector : str    The sector being considered.
    active_sectors   : list   Currently open sector positions.
    portfolio_dd     : float  Current portfolio drawdown % (0 or negative).

    Returns
    -------
    PortfolioCheck
    """
    # ── Hard drawdown halt ────────────────────────────────────────────
    if portfolio_dd <= MAX_PORTFOLIO_DD:
        return PortfolioCheck(
            allowed      = False,
            reason       = f"Portfolio drawdown {portfolio_dd:.1f}% exceeds limit {MAX_PORTFOLIO_DD}%.",
            active_count = len(active_sectors),
        )

    # ── Position cap ─────────────────────────────────────────────────
    if len(active_sectors) >= MAX_SIMULTANEOUS:
        return PortfolioCheck(
            allowed      = False,
            reason       = f"Max simultaneous positions ({MAX_SIMULTANEOUS}) reached.",
            active_count = len(active_sectors),
        )

    # ── Correlation check ─────────────────────────────────────────────
    for group in _CORR_GROUPS:
        if candidate_sector in group:
            overlap = group.intersection(set(active_sectors))
            if overlap:
                corr_name = next(iter(overlap))
                return PortfolioCheck(
                    allowed            = False,
                    reason             = (
                        f"{candidate_sector} is correlated with active position {corr_name}. "
                        "Avoid same-theme exposure."
                    ),
                    active_count       = len(active_sectors),
                    correlated_sector  = corr_name,
                )

    return PortfolioCheck(
        allowed      = True,
        reason       = "Position approved.",
        active_count = len(active_sectors),
    )


# ══════════════════════════════════════════════════════════════════════
# RISK REPORT FROM LOG
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


def compute_risk_metrics(
    log_df:   pd.DataFrame,
    stop_pct: float = DEFAULT_STOP_PCT,
    tp_pct:   float = DEFAULT_TP_PCT,
) -> RiskReport:
    """
    Replay the prediction log through the stop/TP simulator.

    Parameters
    ----------
    log_df   : pd.DataFrame   Full prediction log.
    stop_pct : float          Stop-loss % (default 1.0).
    tp_pct   : float          Take-profit % (default 2.0).

    Returns
    -------
    RiskReport
    """
    report = RiskReport()
    try:
        if log_df is None or log_df.empty:
            return report

        validated = log_df[log_df["correct"].isin(["True", "False"])].copy()
        if validated.empty:
            return report

        validated["_entry"] = validated["entry_price"].apply(_to_float)
        validated["_exit"]  = validated["exit_price"].apply(_to_float)
        validated["_dir"]   = validated["direction"].fillna("Bullish")
        validated["_dt"]    = pd.to_datetime(validated["predicted_at"], errors="coerce", utc=True)
        validated = validated.sort_values("_dt").reset_index(drop=True)

        trades: list[TradeResult] = []
        dates:  list[str] = []

        for _, row in validated.iterrows():
            entry = row["_entry"]
            exit_ = row["_exit"]
            if entry is None or exit_ is None:
                continue
            tr = simulate_trade(entry, exit_, str(row["_dir"]), stop_pct, tp_pct)
            trades.append(tr)
            dates.append(str(row["_dt"])[:10])

        if not trades:
            return report

        rs         = [t.r_multiple for t in trades]
        caps       = [t.capped_return for t in trades]
        stops      = sum(1 for t in trades if t.hit_stop)
        tps        = sum(1 for t in trades if t.hit_tp)
        win_r      = sum(1 for r in rs if r > 0)
        avg_win_r  = float(np.mean([r for r in rs if r > 0])) if win_r else 0.0
        avg_los_r  = float(np.mean([r for r in rs if r <= 0])) if win_r < len(rs) else 0.0
        expectancy = float(np.mean(rs))

        # Equity curve
        equity = [100.0]
        for c in caps:
            equity.append(equity[-1] * (1 + c / 100))
        peaks = np.maximum.accumulate(equity)
        dd    = (np.array(equity) - peaks) / (peaks + 1e-9) * 100
        max_dd = float(dd.min())

        # Sharpe ≈ mean(R) / std(R) × √252 (daily, single position sizing)
        sharpe = (float(np.mean(rs)) / (float(np.std(rs)) + 1e-9)) * math.sqrt(252) if len(rs) > 1 else 0.0

        # Streaks
        outcomes = [1 if r > 0 else 0 for r in rs]
        max_win_streak = max_loss_streak = cur_win = cur_loss = 0
        for o in outcomes:
            if o:
                cur_win += 1; cur_loss = 0
            else:
                cur_loss += 1; cur_win = 0
            max_win_streak  = max(max_win_streak, cur_win)
            max_loss_streak = max(max_loss_streak, cur_loss)

        n = len(trades)
        report.n_trades         = n
        report.stop_hit_pct     = round(stops / n * 100, 1)
        report.tp_hit_pct       = round(tps  / n * 100, 1)
        report.avg_r_multiple   = round(expectancy, 3)
        report.win_r_pct        = round(win_r / n * 100, 1)
        report.max_drawdown_pct = round(max_dd, 2)
        report.expectancy       = round(expectancy, 3)
        report.sharpe_approx    = round(sharpe, 2)
        report.max_loss_streak  = max_loss_streak
        report.max_win_streak   = max_win_streak
        report.avg_win_r        = round(avg_win_r, 3)
        report.avg_loss_r       = round(avg_los_r, 3)
        report.equity_curve     = [round(e, 4) for e in equity]
        report.trade_dates      = dates

    except Exception:
        pass

    return report