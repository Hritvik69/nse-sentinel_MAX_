"""
csv_next_day_engine.py

Standalone local-cache scanner for the "CSV Next-Day Potential" panel.

The original file is missing from this workspace, so this replacement keeps
the same public entry point:

    run_csv_next_day(df=None, cutoff_date=None) -> pd.DataFrame

It scans the existing CSV cache under data/ and returns only "buy ready"
setups with the columns expected by the Streamlit UI.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_MAX_WORKERS = 12

_RESULT_COLUMNS = [
    "Symbol",
    "Price (INR)",
    "Price (₹)",
    "Next Day Prob",
    "Confidence",
    "Grade",
    "Buy Readiness",
    "Signal",
    "Setup",
    "Historical Win %",
    "Downside Risk %",
    "Analog Count",
    "Analog Avg Ret %",
    "Setup Quality",
    "Trigger Quality",
    "RSI",
    "Vol / Avg",
    "Volume Strength",
    "Bull Trap",
    "Risk Notes",
    "Chart Link",
]


CSV_NEXT_DAY_RESULT_COLUMNS = list(_RESULT_COLUMNS)


def _empty_result(
    reason: str = "",
    cache_status: dict | None = None,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=_RESULT_COLUMNS)
    empty.attrs["empty_reason"] = str(reason or "")
    empty.attrs["cache_status"] = dict(cache_status or {})
    return empty


def _app_universe_tickers() -> list[str]:
    for module_name in ("app", "__main__"):
        mod = sys.modules.get(module_name)
        fetch_fn = getattr(mod, "fetch_nse_tickers", None) if mod else None
        if not callable(fetch_fn):
            continue
        try:
            tickers = fetch_fn()
        except Exception:
            continue
        out: list[str] = []
        seen: set[str] = set()
        for raw in tickers or []:
            bare = str(raw).replace(".NS", "").strip().upper()
            if not bare or bare in seen:
                continue
            seen.add(bare)
            out.append(bare)
        if out:
            return out
    return []


def _path_symbol(path: Path) -> str:
    stem = path.stem.upper()
    return stem[:-3] if stem.endswith(".NS") else stem


def _all_cached_paths() -> list[Path]:
    if not _DATA_DIR.exists():
        return []
    return sorted(_DATA_DIR.glob("*.csv"))


def _requested_symbols_from_input(df: pd.DataFrame | None) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    if "Symbol" in df.columns:
        raw_symbols = df["Symbol"].dropna().tolist()
    elif "Ticker" in df.columns:
        raw_symbols = df["Ticker"].dropna().tolist()
    else:
        raw_symbols = []

    requested: list[str] = []
    seen: set[str] = set()
    for raw_symbol in raw_symbols:
        bare = str(raw_symbol).replace(".NS", "").strip().upper()
        if not bare or bare in seen:
            continue
        seen.add(bare)
        requested.append(bare)
    return requested


def get_csv_next_day_cache_status(df: pd.DataFrame | None = None) -> dict:
    cached_paths = _all_cached_paths()
    requested_symbols = _requested_symbols_from_input(df)
    cached_by_symbol = {_path_symbol(path): path for path in cached_paths}
    matching_paths = [cached_by_symbol[symbol] for symbol in requested_symbols if symbol in cached_by_symbol]
    missing_symbols = [symbol for symbol in requested_symbols if symbol not in cached_by_symbol]

    if not _DATA_DIR.exists() or not cached_paths:
        status = "empty_cache"
    elif requested_symbols and not matching_paths:
        status = "no_matching_csvs"
    else:
        status = "ready"

    return {
        "status": status,
        "data_dir": str(_DATA_DIR),
        "data_dir_exists": bool(_DATA_DIR.exists()),
        "file_count": len(cached_paths),
        "requested_count": len(requested_symbols),
        "matching_count": len(matching_paths),
        "missing_count": len(missing_symbols),
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    return (atr / close.replace(0.0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan)


def _chart_link(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=NSE:{symbol}"


def _volume_strength(vol_ratio: float) -> str:
    if vol_ratio >= 1.8:
        return "EXPLOSIVE"
    if vol_ratio >= 1.35:
        return "STRONG"
    if vol_ratio >= 0.95:
        return "HEALTHY"
    if vol_ratio >= 0.75:
        return "SOFT"
    return "WEAK"


def _grade(probability: float, confidence: float) -> str:
    score = 0.7 * probability + 0.3 * confidence
    if score >= 79.0:
        return "A"
    if score >= 69.0:
        return "B"
    if score >= 60.0:
        return "C"
    return "D"


def _signal(probability: float, confidence: float) -> str:
    if probability >= 74.0 and confidence >= 60.0:
        return "HIGH PROBABILITY BREAKOUT"
    if probability >= 66.0:
        return "STRONG SETUP"
    if probability >= 58.0:
        return "WATCHLIST"
    return "AVOID"


def _setup_label(dist_high: float, rsi_value: float, vol_ratio: float, ret_5d: float) -> str:
    if dist_high >= -1.0 and vol_ratio >= 1.2:
        return "Breakout Pressure"
    if dist_high >= -2.0 and 53.0 <= rsi_value <= 68.0:
        return "Tight Coil"
    if ret_5d >= 2.0 and vol_ratio >= 0.9:
        return "Momentum Continuation"
    if dist_high >= -4.0 and rsi_value >= 50.0:
        return "Base Build"
    return "Early Watch"


def _risk_note_list(
    rsi_value: float,
    vol_ratio: float,
    dist_high: float,
    ret_5d: float,
    atr_pct: float,
    analog_count: int,
    analog_avg_ret: float,
    data_quality: str,
) -> tuple[list[str], str]:
    notes: list[str] = []

    if vol_ratio < 0.9:
        notes.append("volume below normal")
    if rsi_value >= 72.0:
        notes.append("short-term overbought")
    if dist_high < -3.5:
        notes.append("still below breakout zone")
    if ret_5d >= 7.5:
        notes.append("extended after sharp 5D run")
    if atr_pct >= 4.8:
        notes.append("volatile structure")
    if analog_count < 5:
        notes.append("limited analog history")
    if analog_avg_ret < 0.0:
        notes.append("analogs have weak average follow-through")
    if data_quality == "LOW":
        notes.append("thin local history")

    if notes:
        return notes, " | ".join(notes)
    return notes, "clean structure"


def _load_csv(path: Path, cutoff_date=None) -> tuple[pd.DataFrame | None, str]:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None, "CRITICAL"

    if df is None or df.empty:
        return None, "CRITICAL"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(df.columns)):
        return None, "CRITICAL"

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close", "Volume"])
    df = df.sort_index()

    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date)
        df = df[df.index <= cutoff_ts]

    if df.empty:
        return None, "CRITICAL"

    row_count = len(df)
    quality = "OK" if row_count >= 60 else "LOW" if row_count >= 30 else "CRITICAL"
    if quality == "CRITICAL":
        return None, quality
    return df, quality


def _analog_stats(indicators: pd.DataFrame) -> tuple[int, float, float, float]:
    hist = indicators.iloc[:-1].copy()
    if hist.empty:
        return 0, 50.0, 0.0, 5.0

    last = indicators.iloc[-1]
    bands = [
        (2.0, 6.0, 0.65),
        (3.0, 8.0, 0.95),
        (4.0, 10.0, 1.20),
    ]

    mask = pd.Series(False, index=hist.index)
    for dist_tol, rsi_tol, vol_tol in bands:
        mask = (
            hist["next_ret"].notna()
            & hist["dist_high"].between(last["dist_high"] - dist_tol, last["dist_high"] + dist_tol)
            & hist["rsi"].between(last["rsi"] - rsi_tol, last["rsi"] + rsi_tol)
            & hist["vol_ratio"].between(
                max(0.0, last["vol_ratio"] - vol_tol),
                last["vol_ratio"] + vol_tol,
            )
            & (hist["close_above_ema20"] == last["close_above_ema20"])
        )
        if int(mask.sum()) >= 6:
            break

    selected = hist.loc[mask]
    if selected.empty:
        return 0, 50.0, 0.0, 5.0

    next_ret = selected["next_ret"].astype(float)
    win_rate = float((next_ret > 0.0).mean() * 100.0)
    avg_ret = float(next_ret.mean())
    losers = next_ret[next_ret < 0.0]
    downside = float(abs(losers.mean())) if not losers.empty else 1.0
    return int(len(selected)), win_rate, avg_ret, downside


def _scan_one(path: Path, cutoff_date=None) -> dict | None:
    df, data_quality = _load_csv(path, cutoff_date=cutoff_date)
    if df is None or df.empty:
        return None

    if len(df) < 35:
        return None

    work = df.copy()
    work["ema20"] = _ema(work["Close"], 20)
    work["ema50"] = _ema(work["Close"], 50)
    work["rsi"] = _rsi(work["Close"], 14)
    work["avg_vol20"] = work["Volume"].rolling(20).mean()
    work["vol_ratio"] = work["Volume"] / work["avg_vol20"].replace(0.0, np.nan)
    work["turnover"] = work["Close"] * work["Volume"]
    work["avg_turnover20"] = work["turnover"].rolling(20).mean()
    work["prior_high20"] = work["High"].shift(1).rolling(20).max()
    work["dist_high"] = (
        (work["Close"] / work["prior_high20"].replace(0.0, np.nan)) - 1.0
    ) * 100.0
    work["ret_1d"] = work["Close"].pct_change(1) * 100.0
    work["ret_5d"] = work["Close"].pct_change(5) * 100.0
    work["atr_pct"] = _atr_pct(work, 14)
    work["next_ret"] = work["Close"].shift(-1) / work["Close"] - 1.0
    work["next_ret"] = work["next_ret"] * 100.0
    work["close_above_ema20"] = work["Close"] > work["ema20"]

    last = work.iloc[-1]
    needed = [
        last.get("Close"),
        last.get("ema20"),
        last.get("ema50"),
        last.get("rsi"),
        last.get("vol_ratio"),
        last.get("dist_high"),
        last.get("ret_5d"),
        last.get("atr_pct"),
    ]
    if any(pd.isna(v) for v in needed):
        return None

    close_price = _safe_float(last["Close"])
    ema20 = _safe_float(last["ema20"], close_price)
    ema50 = _safe_float(last["ema50"], close_price)
    rsi_value = _safe_float(last["rsi"], 50.0)
    vol_ratio = _safe_float(last["vol_ratio"], 1.0)
    avg_turnover20 = _safe_float(last.get("avg_turnover20"), 0.0)
    dist_high = _safe_float(last["dist_high"], -10.0)
    ret_1d = _safe_float(last["ret_1d"], 0.0)
    ret_5d = _safe_float(last["ret_5d"], 0.0)
    atr_pct = _safe_float(last["atr_pct"], 5.0)

    if (
        close_price < ema20
        or rsi_value < 45.0
        or vol_ratio < 0.65
        or dist_high < -6.0
        or ret_5d < -4.5
        or avg_turnover20 < 15_000_000.0
    ):
        return None

    analog_count, hist_win, analog_avg_ret, downside_risk = _analog_stats(work)

    trend_score = 0.0
    trend_score += 18.0 if close_price > ema20 else 0.0
    trend_score += 12.0 if ema20 >= ema50 else 0.0
    trend_score += _clip((ret_5d + 4.0) * 2.2, 0.0, 12.0)
    trend_score += _clip((58.0 - abs(rsi_value - 58.0)) * 0.55, 0.0, 14.0)

    proximity_score = _clip(18.0 - abs(dist_high + 0.8) * 3.2, 0.0, 18.0)
    compression_score = _clip((4.8 - atr_pct) * 5.0, 0.0, 20.0)
    setup_quality = _clip(trend_score + proximity_score + compression_score, 0.0, 100.0)

    trigger_quality = 0.0
    trigger_quality += _clip((vol_ratio - 0.7) * 35.0, 0.0, 35.0)
    trigger_quality += _clip(14.0 - abs(dist_high + 0.2) * 4.0, 0.0, 14.0)
    trigger_quality += _clip((ret_1d + 2.0) * 5.0, 0.0, 14.0)
    trigger_quality += 10.0 if close_price >= ema20 else 0.0
    trigger_quality += 9.0 if rsi_value >= 52.0 else 0.0
    trigger_quality = _clip(trigger_quality, 0.0, 100.0)

    notes, risk_notes = _risk_note_list(
        rsi_value=rsi_value,
        vol_ratio=vol_ratio,
        dist_high=dist_high,
        ret_5d=ret_5d,
        atr_pct=atr_pct,
        analog_count=analog_count,
        analog_avg_ret=analog_avg_ret,
        data_quality=data_quality,
    )

    bull_trap = "LOW"
    if (
        (dist_high > 0.8 and vol_ratio < 0.95)
        or rsi_value >= 75.0
        or (ret_5d >= 8.5 and atr_pct >= 4.0)
    ):
        bull_trap = "HIGH"
    elif (
        vol_ratio < 1.0
        or rsi_value >= 69.0
        or analog_avg_ret < 0.0
        or "volatile structure" in risk_notes
    ):
        bull_trap = "MEDIUM"

    structural_prob = 34.0
    structural_prob += setup_quality * 0.28
    structural_prob += trigger_quality * 0.22
    structural_prob += _clip((vol_ratio - 1.0) * 10.0, -4.0, 8.0)
    structural_prob -= 7.0 if bull_trap == "HIGH" else 3.0 if bull_trap == "MEDIUM" else 0.0
    structural_prob = _clip(structural_prob, 25.0, 88.0)

    if analog_count >= 8:
        next_day_prob = 0.55 * hist_win + 0.45 * structural_prob
    elif analog_count >= 4:
        next_day_prob = 0.35 * hist_win + 0.65 * structural_prob
    else:
        next_day_prob = structural_prob

    next_day_prob += _clip(analog_avg_ret * 2.0, -4.0, 5.0)
    next_day_prob = _clip(next_day_prob, 20.0, 95.0)

    confidence = 34.0
    confidence += min(analog_count * 4.5, 28.0)
    confidence += 18.0 if data_quality == "OK" else 10.0
    confidence += _clip(18.0 - abs(hist_win - structural_prob) * 0.35, 0.0, 18.0)
    confidence += _clip(16.0 - atr_pct * 2.0, 0.0, 16.0)
    confidence -= 8.0 if bull_trap == "HIGH" else 3.0 if bull_trap == "MEDIUM" else 0.0
    confidence = _clip(confidence, 35.0, 92.0)

    grade = _grade(next_day_prob, confidence)
    signal = _signal(next_day_prob, confidence)
    setup = _setup_label(dist_high, rsi_value, vol_ratio, ret_5d)
    volume_strength = _volume_strength(vol_ratio)

    buy_ready = (
        next_day_prob >= 61.0
        and confidence >= 54.0
        and setup_quality >= 56.0
        and trigger_quality >= 52.0
        and close_price >= ema20
        and dist_high >= -4.5
        and bull_trap != "HIGH"
    )
    if not buy_ready:
        return None

    symbol = path.stem.replace(".NS", "")
    return {
        "Symbol": symbol,
        "Price (INR)": round(close_price, 2),
        "Price (₹)": round(close_price, 2),
        "Next Day Prob": round(next_day_prob, 1),
        "Confidence": round(confidence, 1),
        "Grade": grade,
        "Buy Readiness": "BUY READY",
        "Signal": signal,
        "Setup": setup,
        "Historical Win %": round(hist_win, 1),
        "Downside Risk %": round(_clip(downside_risk, 0.5, 15.0), 2),
        "Analog Count": int(analog_count),
        "Analog Avg Ret %": round(analog_avg_ret, 2),
        "Setup Quality": round(setup_quality, 1),
        "Trigger Quality": round(trigger_quality, 1),
        "RSI": round(rsi_value, 1),
        "Vol / Avg": round(vol_ratio, 2),
        "Volume Strength": volume_strength,
        "Bull Trap": bull_trap,
        "Risk Notes": risk_notes,
        "Chart Link": _chart_link(symbol),
    }


def _paths_from_input(df: pd.DataFrame | None) -> list[Path]:
    cached_paths = _all_cached_paths()
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return cached_paths

    requested_symbols = _requested_symbols_from_input(df)
    if not requested_symbols:
        return []

    cached_by_symbol = {_path_symbol(path): path for path in cached_paths}
    return [cached_by_symbol[symbol] for symbol in requested_symbols if symbol in cached_by_symbol]


def run_csv_next_day(
    df: pd.DataFrame | None = None,
    cutoff_date=None,
    max_workers: int = _MAX_WORKERS,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scan the local CSV cache and return buy-ready next-day candidates.

    Parameters
    ----------
    df : pd.DataFrame | None
        Optional result frame with Symbol/Ticker column. When provided,
        the scan is restricted to those symbols if corresponding CSVs exist.

    cutoff_date : datetime/date/str | None
        Optional time-travel cutoff; rows after this date are excluded.

    max_workers : int
        Parallel workers for the local CSV scan.

    progress_callback : callable(done: int, total: int, found: int) | None
        Optional UI callback invoked after each local CSV finishes.
    """
    try:
        cache_status = get_csv_next_day_cache_status(df)
        if cache_status["status"] == "empty_cache":
            return _empty_result("NO_LOCAL_CACHE", cache_status)

        paths = _paths_from_input(df)
        if not paths:
            return _empty_result("NO_MATCHING_CSVS", cache_status)

        rows: list[dict] = []
        worker_count = max(1, min(int(max_workers), _MAX_WORKERS, len(paths)))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_scan_one, path, cutoff_date): path for path in paths}
            total = len(futures)
            done = 0
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    row = None
                if row:
                    rows.append(row)
                done += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done, total, len(rows))
                    except Exception:
                        pass

        if not rows:
            return _empty_result("NO_SETUPS", cache_status)

        out = pd.DataFrame(rows)
        out = out.sort_values(
            ["Next Day Prob", "Confidence", "Setup Quality", "Analog Count"],
            ascending=[False, False, False, False],
            kind="stable",
        ).reset_index(drop=True)

        ordered = [col for col in _RESULT_COLUMNS if col in out.columns]
        out = out[ordered]
        out.attrs["empty_reason"] = ""
        out.attrs["cache_status"] = cache_status
        return out

    except Exception:
        return _empty_result("EXCEPTION", get_csv_next_day_cache_status(df))
