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
    "mode",
    "prediction_score",
    "final_score",
    "signal",
    "conviction_tier",
    "market_bias",
    "regime",
    "pred_bullish",
    "actual_next_return_pct",
    "outcome_label",
]


def _ensure_data_dir() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def log_scan_predictions(
    df: pd.DataFrame,
    mode: int,
    market_bias: dict | None,
) -> None:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        _ensure_data_dir()
        mb = market_bias if isinstance(market_bias, dict) else {}
        bias_s = str(mb.get("bias", ""))[:160]
        regime_s = str(mb.get("regime", ""))[:80]
        ts = datetime.now().isoformat(timespec="seconds")
        file_exists = LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            for _, row in df.iterrows():
                try:
                    sym = str(row.get("Symbol") or row.get("Ticker") or "").strip()
                    if not sym:
                        continue
                    ps = row.get("Prediction Score", np.nan)
                    fs = row.get("Final Score", np.nan)
                    sig = str(row.get("Signal", "") or "")[:40]
                    ct = str(row.get("Conviction Tier", "") or "")[:20]
                    try:
                        ps_f = float(ps) if ps is not None and pd.notna(ps) else float("nan")
                    except Exception:
                        ps_f = float("nan")
                    pred_bull = "1" if (np.isfinite(ps_f) and ps_f >= 55.0) else "0"
                    w.writerow(
                        {
                            "logged_at": ts,
                            "symbol": sym,
                            "mode": int(mode) if mode is not None else 0,
                            "prediction_score": f"{ps_f:.4f}" if np.isfinite(ps_f) else "",
                            "final_score": f"{float(fs):.4f}" if fs is not None and pd.notna(fs) else "",
                            "signal": sig,
                            "conviction_tier": ct,
                            "market_bias": bias_s,
                            "regime": regime_s,
                            "pred_bullish": pred_bull,
                            "actual_next_return_pct": "",
                            "outcome_label": "",
                        }
                    )
                except Exception:
                    continue
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
        if not LOG_PATH.exists():
            return out
        df = pd.read_csv(LOG_PATH)
        out["total_logged"] = int(len(df))
        if df.empty or "actual_next_return_pct" not in df.columns:
            return out

        def _to_float(x: object) -> float | None:
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                s = str(x).strip()
                if s == "":
                    return None
                f = float(s)
                return f if np.isfinite(f) else None
            except Exception:
                return None

        df = df.copy()
        df["_act"] = df["actual_next_return_pct"].map(_to_float)
        sub = df[df["_act"].notna()].copy()
        out["rows_with_outcome"] = int(len(sub))
        if sub.empty:
            return out

        sub["_bull_pred"] = sub["pred_bullish"].astype(str).str.strip().isin(("1", "1.0", "True", "true"))
        sub["_act_pos"] = sub["_act"] > 0
        sub["_act_neg"] = sub["_act"] < 0

        bull_rows = sub[sub["_bull_pred"]]
        bear_rows = sub[~sub["_bull_pred"]]

        if len(bull_rows) > 0:
            bull_ok = (bull_rows["_act_pos"]).sum()
            out["bullish_precision_pct"] = round(100.0 * float(bull_ok) / float(len(bull_rows)), 2)
            out["false_bullish_pct"] = round(100.0 * float((~bull_rows["_act_pos"]).sum()) / float(len(bull_rows)), 2)

        if len(bear_rows) > 0:
            bear_ok = ((bear_rows["_act_neg"]) | (bear_rows["_act"] == 0)).sum()
            out["bearish_precision_pct"] = round(100.0 * float(bear_ok) / float(len(bear_rows)), 2)
            out["false_bearish_pct"] = round(100.0 * float((bear_rows["_act_pos"]).sum()) / float(len(bear_rows)), 2)

        hits = 0
        for _, r in sub.iterrows():
            bp = bool(r["_bull_pred"])
            ap = bool(r["_act_pos"])
            if bp and ap:
                hits += 1
            elif not bp and not ap:
                hits += 1
        out["accuracy_pct"] = round(100.0 * float(hits) / float(len(sub)), 2)
        return out
    except Exception:
        return out


def backfill_actual_returns(all_data: dict) -> int:
    """
    FIX 6 — Auto-fill actual_next_return_pct for logged predictions.

    Reads prediction_feedback_log.csv. For every row where
    actual_next_return_pct is blank AND the symbol exists in all_data,
    looks up the historical close on logged_at date, finds the next
    available close, computes the 1-day return (%) and writes it back.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Preloaded price history keyed by "SYMBOL.NS" — from ALL_DATA.

    Returns
    -------
    int  Number of rows filled (0 if nothing to do or any error).

    Never raises — fully wrapped in try/except.
    """
    try:
        if not LOG_PATH.exists():
            return 0
        df = pd.read_csv(LOG_PATH)
        if df.empty or "actual_next_return_pct" not in df.columns:
            return 0

        # Rows still needing an outcome
        needs_fill = df["actual_next_return_pct"].apply(
            lambda x: str(x).strip() == "" or (isinstance(x, float) and np.isnan(x))
        )
        if not needs_fill.any():
            return 0

        filled = 0
        for idx in df.index[needs_fill]:
            try:
                sym = str(df.at[idx, "symbol"]).strip()
                if not sym:
                    continue
                ticker_ns = sym if sym.endswith(".NS") else sym + ".NS"
                df_hist = all_data.get(ticker_ns)
                if df_hist is None or "Close" not in df_hist.columns or len(df_hist) < 2:
                    continue

                # Parse the logged date
                logged_str = str(df.at[idx, "logged_at"]).strip()
                logged_dt  = pd.to_datetime(logged_str, errors="coerce")
                if pd.isnull(logged_dt):
                    continue
                logged_date = logged_dt.date()

                # Align to historical index (date-only comparison)
                hist_dates = pd.to_datetime(df_hist.index).date
                date_arr   = np.array(hist_dates)
                match_idxs = np.where(date_arr == logged_date)[0]
                if len(match_idxs) == 0:
                    # Try the closest date on or before logged_date
                    before = np.where(date_arr <= logged_date)[0]
                    if len(before) == 0:
                        continue
                    match_idxs = [before[-1]]

                day_i = int(match_idxs[0])
                if day_i + 1 >= len(df_hist):
                    continue  # no next-day data yet

                close_today = float(df_hist["Close"].iloc[day_i])
                close_next  = float(df_hist["Close"].iloc[day_i + 1])
                if close_today <= 0:
                    continue

                ret_pct = round((close_next / close_today - 1.0) * 100, 4)
                df.at[idx, "actual_next_return_pct"] = ret_pct
                outcome = "correct" if (
                    (str(df.at[idx, "pred_bullish"]).strip() in ("1", "1.0") and ret_pct > 0)
                    or (str(df.at[idx, "pred_bullish"]).strip() not in ("1", "1.0") and ret_pct <= 0)
                ) else "incorrect"
                df.at[idx, "outcome_label"] = outcome
                filled += 1
            except Exception:
                continue

        if filled > 0:
            df.to_csv(LOG_PATH, index=False)
        return filled

    except Exception:
        return 0