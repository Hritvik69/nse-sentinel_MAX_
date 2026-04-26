"""
scan_diagnostics.py
───────────────────
Thread-safe registry that records WHY each ticker was dropped during a scan.

Usage
─────
    # In analyse():
    from scan_diagnostics import record_failure, record_success, get_report

    record_failure("RELIANCE.NS", "NO_DATA")
    record_success("TCS.NS")

    # After run_scan():
    report = get_report()
    # report = {
    #   "attempted": 2100, "succeeded": 312, "failed": 1788,
    #   "reasons": {"NO_DATA": 950, "STALE": 201, "TOO_SHORT": 37,
    #               "SCAN_FILTER": 580, "EXCEPTION": 20},
    #   "failed_tickers": [...],
    # }

Failure reason codes (canonical)
──────────────────────────────────
    NO_DATA         preload returned None — ticker not in ALL_DATA and
                    live download also failed
    TOO_SHORT       DataFrame has < MIN_VIABLE_ROWS rows after dropna
    STALE           Last candle is > 7 calendar days old
    BAD_PRICE       Price outside sanity range (≤ 1 or > 100 000)
    ZERO_VOLUME     Last volume is 0 or negative
    NAN_INDICATORS  EMA/RSI computed as NaN (not enough history)
    SCAN_FILTER     Data fine, but stock did not pass the mode's entry filter
                    (this is NOT an error — it is expected behaviour)
    EXCEPTION       Unexpected exception inside analyse()
    LOW_QUALITY     Data exists but row count is below PREFERRED_MIN_ROWS
                    (scan still proceeds — flagged for UI display)

Only NO_DATA, TOO_SHORT, STALE, BAD_PRICE, ZERO_VOLUME, NAN_INDICATORS,
and EXCEPTION represent actual data problems.  SCAN_FILTER is a normal
outcome (stock just didn't match the strategy criteria).
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from typing import Literal

# ── Reason code type ──────────────────────────────────────────────────
FailReason = Literal[
    "NO_DATA", "TOO_SHORT", "STALE", "BAD_PRICE",
    "ZERO_VOLUME", "NAN_INDICATORS", "SCAN_FILTER",
    "EXCEPTION", "LOW_QUALITY",
]

# ── Module-level state ────────────────────────────────────────────────
_LOCK              = threading.Lock()
_attempted:  set[str]             = set()
_succeeded:  set[str]             = set()
_failed:     dict[str, FailReason] = {}   # ticker → reason
_reason_count: Counter            = Counter()


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def reset() -> None:
    """Clear all state — call this at the START of every scan."""
    global _attempted, _succeeded, _failed, _reason_count
    with _LOCK:
        _attempted     = set()
        _succeeded     = set()
        _failed        = {}
        _reason_count  = Counter()


def record_attempt(ticker: str) -> None:
    """Mark a ticker as attempted (call before any early-returns in analyse)."""
    with _LOCK:
        _attempted.add(ticker)


def record_success(ticker: str) -> None:
    """Mark a ticker as producing a valid scan result."""
    with _LOCK:
        _attempted.add(ticker)
        _succeeded.add(ticker)
        _failed.pop(ticker, None)  # un-fail if previously marked


def record_failure(ticker: str, reason: FailReason) -> None:
    """
    Record a failure reason for a ticker.
    SCAN_FILTER failures are counted separately so they don't inflate
    the 'data problem' count shown in the UI.
    """
    with _LOCK:
        _attempted.add(ticker)
        if ticker not in _succeeded:        # don't overwrite a success
            _failed[ticker] = reason
        _reason_count[reason] += 1


def get_report() -> dict:
    """
    Return a summary dict for display in the Streamlit UI.

    Keys
    ----
    attempted          int   — total tickers passed to analyse()
    succeeded          int   — tickers that produced a scan row
    failed_data        int   — tickers with a real data problem (not SCAN_FILTER)
    scan_filtered      int   — tickers rejected by entry filter (expected)
    reasons            dict  — {reason_code: count} for ALL failures
    failed_tickers     list  — [(ticker, reason), ...] for data problems only
    success_rate_pct   float — succeeded / attempted * 100
    data_ok_pct        float — (attempted - failed_data) / attempted * 100
    """
    with _LOCK:
        att  = len(_attempted)
        succ = len(_succeeded)
        reasons_snapshot = dict(_reason_count)
        failed_snapshot  = dict(_failed)

    data_problem_reasons = {
        "NO_DATA", "TOO_SHORT", "STALE", "BAD_PRICE",
        "ZERO_VOLUME", "NAN_INDICATORS", "EXCEPTION",
    }
    failed_data     = [(t, r) for t, r in failed_snapshot.items() if r in data_problem_reasons]
    scan_filtered   = reasons_snapshot.get("SCAN_FILTER", 0)
    failed_data_cnt = len(failed_data)

    return {
        "attempted":        att,
        "succeeded":        succ,
        "failed_data":      failed_data_cnt,
        "scan_filtered":    scan_filtered,
        "reasons":          reasons_snapshot,
        "failed_tickers":   sorted(failed_data, key=lambda x: x[0]),
        "success_rate_pct": round(100.0 * succ / att, 1) if att else 0.0,
        "data_ok_pct":      round(100.0 * (att - failed_data_cnt) / att, 1) if att else 0.0,
    }


def get_low_quality_tickers() -> list[str]:
    """Return tickers flagged as LOW_QUALITY (data exists but thin)."""
    with _LOCK:
        return [t for t, r in _failed.items() if r == "LOW_QUALITY"]