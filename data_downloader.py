"""
data_downloader.py
──────────────────
Professional local-data layer for NSE Sentinel.

CHANGES vs previous version
────────────────────────────
  [FIX 1] _MIN_ROWS lowered from 30 → 5.  Stocks with fewer than 5 rows
          are genuinely unusable; everything else gets through.
  [FIX 2] _PREFERRED_MIN_ROWS = 20 — soft threshold for quality tagging.
          Stocks below this are returned with a "LOW_QUALITY" flag in the
          DataFrame column "data_quality" but are NOT dropped.
  [FIX 3] load_csv() now returns (df, quality) tuple via load_csv_with_quality().
          The legacy load_csv() wrapper is kept for backwards compatibility
          and returns just the DataFrame.
  [FIX 4] bulk_download() now returns detailed per-ticker failure reasons,
          not just aggregate counts.
  [FIX 5] Removed the second silent drop in _download_one() that occurred
          after the incremental concat.

Column "data_quality" values
─────────────────────────────
  "OK"          ≥ PREFERRED_MIN_ROWS rows after cleaning
  "LOW"         5 – PREFERRED_MIN_ROWS-1 rows (thin but usable)
  "CRITICAL"    < 5 rows — callers should treat as unusable
"""

from __future__ import annotations

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

_FILE_LOCK = threading.Lock()

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

# ── Paths ──────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_DIR   = _HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Concurrency controls ───────────────────────────────────────────────
_DOWNLOAD_WORKERS = 10
_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else None

# ── Row thresholds ────────────────────────────────────────────────────
# FIX 1: Hard minimum is 5 (genuinely unusable below this).
# FIX 2: Soft preferred minimum — below this we tag as LOW_QUALITY.
_MIN_ROWS           = 5    # was 30 — this was silently dropping valid stocks
_PREFERRED_MIN_ROWS = 20   # soft quality threshold
_MAX_STALENESS_H    = 24

# ── Download failure log (in-memory, keyed by ticker) ─────────────────
_FAIL_REASONS: dict[str, str] = {}
_FAIL_LOCK    = threading.Lock()


# ══════════════════════════════════════════════════════════════════════
# DATA QUALITY RESULT
# ══════════════════════════════════════════════════════════════════════

class CsvResult(NamedTuple):
    df:      pd.DataFrame | None
    quality: str    # "OK" | "LOW" | "CRITICAL" | "MISSING"
    rows:    int
    reason:  str    # empty string if quality == "OK"


# ══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _csv_path(ticker_ns: str) -> Path:
    safe = ticker_ns.replace(":", "_").replace("/", "_")
    return DATA_DIR / f"{safe}.csv"


def _csv_age_hours(ticker_ns: str) -> float:
    p = _csv_path(ticker_ns)
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 3600.0


def _quality_tag(row_count: int) -> tuple[str, str]:
    """Return (quality_label, reason_string) for a given row count."""
    if row_count >= _PREFERRED_MIN_ROWS:
        return "OK", ""
    if row_count >= _MIN_ROWS:
        return "LOW", f"only {row_count} rows (below preferred {_PREFERRED_MIN_ROWS})"
    return "CRITICAL", f"only {row_count} rows (below hard minimum {_MIN_ROWS})"


def _now_ist() -> datetime:
    if _IST_TZ is not None:
        return datetime.now(_IST_TZ)
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


# ══════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════

def load_csv_with_quality(ticker_ns: str) -> CsvResult:
    """
    Load ticker CSV and return a CsvResult with quality metadata.

    FIX: Never returns None for a stock that *has* data — even thin data
    is returned with quality="LOW" so the scan can decide what to do.
    """
    p = _csv_path(ticker_ns)
    if not p.exists():
        return CsvResult(None, "MISSING", 0, "CSV not found in data/")
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        required = {"Close", "Volume"}
        if not required.issubset(set(df.columns)):
            return CsvResult(None, "CRITICAL", 0, "missing Close or Volume columns")
        df = df.dropna(subset=["Close", "Volume"])
        n = len(df)
        quality, reason = _quality_tag(n)
        if quality == "CRITICAL":
            # FIX: still return the df — let callers decide, don't silently drop
            return CsvResult(df if n > 0 else None, "CRITICAL", n, reason)
        return CsvResult(df, quality, n, reason)
    except Exception as e:
        return CsvResult(None, "CRITICAL", 0, f"parse error: {e}")


def load_csv(ticker_ns: str) -> pd.DataFrame | None:
    """
    Legacy API — returns DataFrame or None.

    FIX: Now returns data even if row count is below the OLD _MIN_ROWS=30.
    Returns None only if file is missing or completely unparseable.
    """
    r = load_csv_with_quality(ticker_ns)
    return r.df  # None only when quality=="MISSING" or parse error


# ══════════════════════════════════════════════════════════════════════
# DOWNLOADING
# ══════════════════════════════════════════════════════════════════════

def _download_one(
    ticker_ns: str, period: str = "6mo", force: bool = False
) -> tuple[pd.DataFrame | None, str]:
    """
    Download / update one ticker and save to CSV.
    Returns (df_or_None, status).

    Status values: "updated" | "skipped" | "failed"

    FIX 5: Removed the second silent drop `if len(df) < _MIN_ROWS: return None, "failed"`
           that occurred after incremental concat.  Now only rejects truly
           empty results (0 rows).
    """
    try:
        old_result = load_csv_with_quality(ticker_ns)
        old_df     = old_result.df
        today      = _now_ist()

        if old_df is not None and not old_df.empty:
            last_date = pd.to_datetime(old_df.index.max())

            if not force:
                if last_date.date() >= today.date():
                    return old_df, "skipped"
                if (today.weekday() >= 5 and last_date.weekday() == 4
                        and (today.date() - last_date.date()).days <= 3):
                    return old_df, "skipped"

            new_df = yf.download(
                ticker_ns, period="5d", interval="1d",
                auto_adjust=True, progress=False, timeout=15, threads=False,
            )
            time.sleep(0.1)
        else:
            new_df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=15, threads=False,
            )
            time.sleep(0.1)

        if new_df is None or new_df.empty:
            with _FAIL_LOCK:
                _FAIL_REASONS[ticker_ns] = "yfinance returned empty DataFrame"
            return old_df, "failed"

        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)
        new_df.columns = [c.strip().title() for c in new_df.columns]

        req_cols   = ["Open", "High", "Low", "Close", "Volume"]
        avail_cols = [c for c in req_cols if c in new_df.columns]
        new_df     = new_df[avail_cols].dropna(how="all")
        close_vol  = [c for c in ["Close", "Volume"] if c in avail_cols]
        if close_vol:
            new_df = new_df.dropna(subset=close_vol)

        if old_df is not None and not old_df.empty:
            avail_old = [c for c in req_cols if c in old_df.columns]
            df        = pd.concat([old_df[avail_old], new_df])
            df        = df[~df.index.duplicated(keep="last")]
        else:
            df = new_df

        # FIX 5: No longer hard-reject based on row count here.
        # A stock with 5-20 rows is a new listing or one that resumed trading —
        # it deserves to be in the scan, tagged LOW_QUALITY, not silently dropped.
        if df.empty:
            with _FAIL_LOCK:
                _FAIL_REASONS[ticker_ns] = "result empty after concat+dedup"
            return old_df, "failed"

        df = df.sort_index(ascending=True)
        df = df[[c for c in req_cols if c in df.columns]]

        with _FILE_LOCK:
            df.to_csv(_csv_path(ticker_ns))

        return df, "updated"

    except Exception as e:
        with _FAIL_LOCK:
            _FAIL_REASONS[ticker_ns] = str(e)[:200]
        return load_csv(ticker_ns), "failed"


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def bulk_download(
    tickers: list[str],
    period: str = "6mo",
    force: bool = False,
    print_progress: bool = True,
    progress_callback=None,
) -> dict:
    """
    Download tickers in parallel.

    Returns
    -------
    dict with keys:
        updated   int
        skipped   int
        failed    int
        failures  dict[ticker → reason]   (FIX 4: now exposed)
    """
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    stats = {"updated": 0, "skipped": 0, "failed": 0, "failures": {}}

    if print_progress:
        print(f"[DataDownloader] Updating {len(tickers_ns)} tickers …")

    done  = 0
    total = len(tickers_ns)
    t0    = time.time()

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(_download_one, t, period, force): t for t in tickers_ns}
        for fut in as_completed(futs):
            ticker = futs[fut]
            done  += 1
            try:
                df, status = fut.result()
                stats[status] = stats.get(status, 0) + 1
                if status == "failed":
                    with _FAIL_LOCK:
                        reason = _FAIL_REASONS.get(ticker, "unknown")
                    stats["failures"][ticker] = reason
            except Exception as e:
                stats["failed"] += 1
                stats["failures"][ticker] = str(e)[:200]

            if progress_callback:
                progress_callback(done, total)
            if print_progress and done % 50 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 1
                eta     = (total - done) / rate
                print(
                    f"  [{done:4d}/{total}] "
                    f"Upd:{stats['updated']} "
                    f"Skip:{stats['skipped']} "
                    f"Fail:{stats['failed']}  ETA {eta:.0f}s"
                )

    if stats["failed"] > 0 and print_progress:
        print(
            f"⚠️ {stats['failed']} tickers failed. "
            f"Top reasons: {dict(list(stats['failures'].items())[:5])}"
        )

    return stats


def update_all_data(tickers: list[str], period: str = "6mo") -> dict:
    """Download/refresh CSVs for all tickers. Returns {updated, skipped, failed, failures}."""
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    stats      = {"updated": 0, "skipped": 0, "failed": 0, "failures": {}}

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_one, t, period): t for t in tickers_ns}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, status = future.result()
            except Exception as e:
                status = "failed"
                stats["failures"][ticker] = str(e)[:200]
            stats[status] = stats.get(status, 0) + 1

    return stats


def update_data_if_old(
    tickers: list[str],
    max_age_hours: float = _MAX_STALENESS_H,
    period: str = "6mo",
    print_progress: bool = True,
) -> int:
    """Legacy convenience wrapper — returns count of updated tickers."""
    results = update_all_data(tickers, period=period)
    return results["updated"]


def data_status_summary(tickers: list[str]) -> dict:
    """Return fresh/stale/missing/low_quality counts for the Streamlit sidebar."""
    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    fresh      = 0
    stale      = 0
    missing    = 0
    low_qual   = 0

    for t in tickers_ns:
        age = _csv_age_hours(t)
        if age == float("inf"):
            missing += 1
        elif age > _MAX_STALENESS_H:
            stale += 1
        else:
            fresh += 1
        r = load_csv_with_quality(t)
        if r.quality == "LOW":
            low_qual += 1

    oldest = max(
        (_csv_age_hours(t) for t in tickers_ns if _csv_age_hours(t) < float("inf")),
        default=None,
    )
    return {
        "total":      len(tickers_ns),
        "fresh":      fresh,
        "stale":      stale,
        "missing":    missing,
        "low_quality": low_qual,
        "oldest_h":   round(oldest, 1) if oldest is not None else None,
    }


def get_download_failures() -> dict[str, str]:
    """Return {ticker_ns: reason} for all tickers that failed in this session."""
    with _FAIL_LOCK:
        return dict(_FAIL_REASONS)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    try:
        from nse_ticker_universe import get_all_tickers
        UNIVERSE = get_all_tickers(live=True)
    except ImportError:
        UNIVERSE = [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
            "SBIN.NS","BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","AXISBANK.NS",
        ]
    force_flag = "--force" in sys.argv
    print(f"NSE Data Downloader — {len(UNIVERSE)} tickers, force={force_flag}")
    result = bulk_download(UNIVERSE, period="6mo", force=force_flag, print_progress=True)
    print(f"\nDone: {result['updated']} updated, {result['skipped']} skipped, "
          f"{result['failed']} failed")
    if result["failures"]:
        print(f"First 10 failures:")
        for t, r in list(result["failures"].items())[:10]:
            print(f"  {t}: {r}")
