"""
sector_dynamic_weights.py
══════════════════════════
Module 3 — Feature Reliability Tracking & Dynamic Signal Weighting.

Tracks per-signal win rates from historical predictions.
Dynamically reweights signals so that high-accuracy signals get more
influence and low-accuracy signals are suppressed.

Static weights are the PRIOR (starting point).
Dynamic weights are continuously updated from outcomes.

Storage
───────
data/sector_signal_performance.csv

Public API
──────────
  get_dynamic_weights(sector, regime)   → dict[str, float]
  update_signal_performance(log_df)     → int   (rows processed)
  get_signal_performance_report()       → pd.DataFrame
"""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# STATIC PRIORS (used when no history exists)
# ══════════════════════════════════════════════════════════════════════
# These are normalised to sum = 1 and serve as Bayesian priors.

_STATIC_WEIGHTS: dict[str, float] = {
    "ema_slope":        0.10,
    "price_vs_ema":     0.08,
    "candle_direction": 0.10,
    "body_strength":    0.07,
    "consecutive":      0.07,
    "volume_confirm":   0.10,
    "volatility":       0.04,
    "momentum":         0.04,
    "sector_strength":  0.12,
    "bullish_pct":      0.12,
    "money_flow":       0.08,
    "participation":    0.08,
}

# Bounds: prevent any single signal from dominating
_MIN_WEIGHT = 0.02
_MAX_WEIGHT = 0.22

# Minimum observations before trusting a learned weight
_MIN_OBS = 20

# ── Storage ────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_PERF_PATH = _DATA_DIR / "sector_signal_performance.csv"

_PERF_FIELDS = [
    "signal_name", "observations", "wins", "win_rate",
    "last_updated", "dynamic_weight",
]

# In-memory cache
_perf_cache: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════

def _ensure_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _load_perf() -> dict[str, dict]:
    """Load signal performance from CSV into a dict keyed by signal name."""
    try:
        if not _PERF_PATH.exists():
            return {}
        df = pd.read_csv(_PERF_PATH, dtype=str)
        out: dict[str, dict] = {}
        for _, row in df.iterrows():
            sig = str(row.get("signal_name", "")).strip()
            if not sig:
                continue
            out[sig] = {
                "observations":   int(float(row.get("observations", 0) or 0)),
                "wins":           int(float(row.get("wins", 0) or 0)),
                "win_rate":       float(row.get("win_rate", 0.5) or 0.5),
                "dynamic_weight": float(row.get("dynamic_weight", _STATIC_WEIGHTS.get(sig, 0.08)) or 0.08),
            }
        return out
    except Exception:
        return {}


def _save_perf(perf: dict[str, dict]) -> None:
    try:
        _ensure_dir()
        rows = []
        ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        for sig, data in perf.items():
            rows.append({
                "signal_name":    sig,
                "observations":   data.get("observations", 0),
                "wins":           data.get("wins", 0),
                "win_rate":       round(data.get("win_rate", 0.5), 4),
                "last_updated":   ts,
                "dynamic_weight": round(data.get("dynamic_weight", _STATIC_WEIGHTS.get(sig, 0.08)), 6),
            })
        df = pd.DataFrame(rows, columns=_PERF_FIELDS)
        df.to_csv(_PERF_PATH, index=False)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# WEIGHT COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def _compute_dynamic_weights(perf: dict[str, dict]) -> dict[str, float]:
    """
    Compute dynamic weights from signal win rates.

    Method: Bayesian shrinkage toward the static prior.
    weight_raw = static_prior × (1 + α × (win_rate − 0.5))
    where α scales with sample size (0 when n < _MIN_OBS, 1 when n ≥ 200).

    Weights are then normalised to sum = 1 and clipped.
    """
    raw: dict[str, float] = {}
    for sig, static_w in _STATIC_WEIGHTS.items():
        data = perf.get(sig, {})
        n    = int(data.get("observations", 0))
        wr   = float(data.get("win_rate", 0.5))

        # Trust factor: linearly ramps from 0 (n=0) to 1 (n≥200)
        alpha = float(np.clip(n / 200.0, 0.0, 1.0))

        # Excess win rate above chance; centre on 0.5
        excess = wr - 0.50
        # raw weight = prior × (1 + 2 × alpha × excess)
        raw[sig] = static_w * (1.0 + 2.0 * alpha * excess)
        raw[sig] = float(np.clip(raw[sig], _MIN_WEIGHT, _MAX_WEIGHT))

    total = sum(raw.values())
    if total <= 0:
        return dict(_STATIC_WEIGHTS)
    normalised = {k: round(v / total, 6) for k, v in raw.items()}
    return normalised


# ══════════════════════════════════════════════════════════════════════
# UPDATE FROM PREDICTION LOG
# ══════════════════════════════════════════════════════════════════════

# Signal columns that are stored in the prediction CSV
_SIGNAL_LOG_ALIASES = {
    "ema_slope": ["signal_ema_slope"],
    "price_vs_ema": ["signal_price_vs_ema"],
    "candle_direction": ["signal_candle_direction"],
    "body_strength": ["signal_body_strength"],
    "consecutive": ["signal_consecutive"],
    "volume_confirm": ["signal_volume_confirm", "signal_volume"],
    "volatility": ["signal_volatility"],
    "momentum": ["signal_momentum"],
    "sector_strength": ["signal_sector_strength", "signal_sector_str"],
    "bullish_pct": ["signal_bullish_pct"],
    "money_flow": ["signal_money_flow"],
    "participation": ["signal_participation"],
}

# Threshold: signal value > 60 means "signal fired bullish"
_BULLISH_THRESHOLD = 60.0
_BEARISH_THRESHOLD = 40.0


def update_signal_performance(log_df: pd.DataFrame) -> int:
    """
    Read validated prediction rows (with 'correct' and signal columns)
    and update per-signal win rate tracking.

    Parameters
    ----------
    log_df : pd.DataFrame  Full prediction log with 'correct' column filled.

    Returns
    -------
    int  Number of rows processed.
    """
    try:
        if log_df is None or log_df.empty:
            return 0

        validated = log_df[log_df["correct"].isin(["True", "False"])].copy()
        if validated.empty:
            return 0

        # Rebuild from the log every time so the function is idempotent.
        perf: dict[str, dict] = {}

        processed = 0
        for _, row in validated.iterrows():
            correct_str = str(row.get("correct", "")).strip()
            outcome     = correct_str == "True"
            direction   = str(row.get("direction", "")).strip()

            for sig_name, aliases in _SIGNAL_LOG_ALIASES.items():
                val_str = ""
                for log_col in aliases:
                    candidate = str(row.get(log_col, "")).strip()
                    if candidate not in ("", "nan", "None"):
                        val_str = candidate
                        break
                if val_str in ("", "nan", "None"):
                    continue
                try:
                    sig_val = float(val_str)
                except Exception:
                    continue

                # Was this signal "active" for this prediction?
                if direction == "Bullish":
                    signal_fired = sig_val >= _BULLISH_THRESHOLD
                elif direction == "Bearish":
                    signal_fired = sig_val <= _BEARISH_THRESHOLD
                else:
                    signal_fired = abs(sig_val - 50) >= 10

                if not signal_fired:
                    continue

                if sig_name not in perf:
                    perf[sig_name] = {"observations": 0, "wins": 0,
                                      "win_rate": 0.5,
                                      "dynamic_weight": _STATIC_WEIGHTS.get(sig_name, 0.08)}

                perf[sig_name]["observations"] += 1
                if outcome:
                    perf[sig_name]["wins"] += 1

            processed += 1

        for sig_name, data in perf.items():
            n = int(data.get("observations", 0))
            wins = int(data.get("wins", 0))
            data["win_rate"] = round(wins / n, 4) if n > 0 else 0.5

        # Recompute dynamic weights
        dw = _compute_dynamic_weights(perf)
        for sig, w in dw.items():
            perf.setdefault(sig, {"observations": 0, "wins": 0,
                                  "win_rate": 0.5,
                                  "dynamic_weight": _STATIC_WEIGHTS.get(sig, 0.08)})
            perf[sig]["dynamic_weight"] = w

        _save_perf(perf)
        global _perf_cache
        _perf_cache = perf
        return processed
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def get_dynamic_weights(
    sector: str | None = None,
    regime: str | None = None,
    regime_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Return final signal weights combining:
      1. Learned dynamic weights from history
      2. Regime-based multipliers

    Parameters
    ----------
    sector             : str (unused — sector-specific weights are future work)
    regime             : str  current regime label
    regime_adjustments : dict from sector_regime_engine.regime_weight_adjustments()

    Returns
    -------
    dict[str, float]  normalised weights summing to 1.0
    """
    global _perf_cache
    if not _perf_cache:
        _perf_cache = _load_perf()

    base = _compute_dynamic_weights(_perf_cache)

    if not regime_adjustments:
        return base

    # Apply regime multipliers
    adjusted: dict[str, float] = {}
    for sig, w in base.items():
        mult = regime_adjustments.get(sig, 1.0)
        adjusted[sig] = float(np.clip(w * mult, _MIN_WEIGHT, _MAX_WEIGHT))

    total = sum(adjusted.values())
    if total <= 0:
        return base
    return {k: round(v / total, 6) for k, v in adjusted.items()}


def get_signal_performance_report() -> pd.DataFrame:
    """
    Return a DataFrame showing per-signal win rates and current weights.
    """
    global _perf_cache
    if not _perf_cache:
        _perf_cache = _load_perf()
    dw = _compute_dynamic_weights(_perf_cache)

    rows = []
    for sig, static_w in _STATIC_WEIGHTS.items():
        data = _perf_cache.get(sig, {})
        rows.append({
            "Signal":         sig,
            "Observations":   data.get("observations", 0),
            "Win Rate":       round(data.get("win_rate", 0.5) * 100, 1),
            "Static Weight":  round(static_w * 100, 2),
            "Dynamic Weight": round(dw.get(sig, static_w) * 100, 2),
            "Δ Weight":       round((dw.get(sig, static_w) - static_w) * 100, 2),
        })
    return pd.DataFrame(rows)
