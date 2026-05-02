from __future__ import annotations

import json
import shutil
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)
_ROOT = Path(__file__).resolve().parent
_SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"
_FRAME_SOURCE_ATTR = "_nse_data_source"
_FRAME_DATE_ATTR = "_nse_market_date"
_FRAME_WINDOW_ATTR = "_nse_window"
_FRAME_CAPTURED_AT_ATTR = "_nse_captured_at"


def _snapshot_meta_path(market_date) -> Path:
    return get_snapshot_path(market_date) / "_meta.json"


def _now_ist() -> datetime:
    try:
        return datetime.now(_IST_TZ)
    except Exception:
        return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _time_travel_active() -> bool:
    try:
        from time_travel_engine import is_active as _tt

        return bool(_tt())
    except Exception:
        return False


def _coerce_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _previous_weekday(day: date) -> date:
    cur = day
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def _recent_market_days(anchor: date, count: int = 3) -> list[date]:
    days: list[date] = []
    cur = anchor
    limit = max(1, int(count))
    while len(days) < limit:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return days


def _available_snapshot_dates() -> list[date]:
    dates: list[date] = []
    try:
        if not _SNAPSHOT_ROOT.exists():
            return dates
        for child in _SNAPSHOT_ROOT.iterdir():
            if not child.is_dir():
                continue
            try:
                snap_day = date.fromisoformat(child.name)
            except Exception:
                continue
            if snapshot_exists(snap_day):
                dates.append(snap_day)
    except Exception:
        return []
    return sorted(dates)


def _latest_snapshot_on_or_before(day: date) -> date | None:
    latest: date | None = None
    try:
        for snap_day in _available_snapshot_dates():
            if snap_day <= day:
                latest = snap_day
    except Exception:
        return None
    return latest


def get_current_window() -> str:
    now_ist = _now_ist()
    today = now_ist.date()
    current_time = now_ist.time()

    if today.weekday() >= 5:
        return "WEEKEND"
    if _MARKET_OPEN <= current_time <= _MARKET_CLOSE:
        return "LIVE"
    if current_time > _MARKET_CLOSE:
        return "CLOSED"
    return "PRE_MARKET"


def get_market_hours_label() -> str:
    return f"{_MARKET_OPEN.strftime('%I:%M %p')} - {_MARKET_CLOSE.strftime('%I:%M %p')} IST"


def get_expected_data_date() -> date:
    now_ist = _now_ist()
    today = now_ist.date()
    window = get_current_window()

    if window in ("LIVE", "CLOSED"):
        return today

    if window == "PRE_MARKET":
        baseline = _previous_weekday(today - timedelta(days=1))
    else:
        baseline = _previous_weekday(today)

    fallback = _latest_snapshot_on_or_before(baseline)
    return fallback or baseline


def get_acceptable_data_dates() -> list[date]:
    """
    Return a short list of valid market dates for daily OHLCV freshness checks.

    Yahoo daily candles can lag the calendar on exchange holidays or shortly
    after the close, so we allow the most recent few trading sessions instead
    of rejecting otherwise valid last-available market data.
    """
    now_ist = _now_ist()
    today = now_ist.date()
    window = get_current_window()

    if window in ("LIVE", "CLOSED"):
        primary = _recent_market_days(today, count=3)
    elif window == "PRE_MARKET":
        primary = _recent_market_days(_previous_weekday(today - timedelta(days=1)), count=3)
    else:
        primary = _recent_market_days(_previous_weekday(today), count=3)

    snapshot_fallback = _latest_snapshot_on_or_before(today)
    snapshot_days = []
    if snapshot_fallback is not None and (today - snapshot_fallback).days <= 7:
        snapshot_days = _recent_market_days(snapshot_fallback, count=2)

    ordered: list[date] = []
    seen: set[date] = set()
    for day in primary + snapshot_days:
        if day in seen:
            continue
        seen.add(day)
        ordered.append(day)
    return ordered


def stamp_frame_metadata(
    df: pd.DataFrame | None,
    *,
    source: str,
    market_date: date | None = None,
    window: str | None = None,
    captured_at: str | None = None,
) -> pd.DataFrame | None:
    if df is None:
        return None
    try:
        out = df.copy()
        out.attrs[_FRAME_SOURCE_ATTR] = str(source or "").strip()
        if market_date is None:
            try:
                market_date = pd.to_datetime(out.index[-1]).date()
            except Exception:
                market_date = None
        if market_date is not None:
            out.attrs[_FRAME_DATE_ATTR] = market_date.isoformat()
        if window:
            out.attrs[_FRAME_WINDOW_ATTR] = str(window).strip()
        if captured_at:
            out.attrs[_FRAME_CAPTURED_AT_ATTR] = str(captured_at).strip()
        return out
    except Exception:
        return df


def get_frame_source(df: pd.DataFrame | None) -> str:
    try:
        if df is None:
            return ""
        return str(df.attrs.get(_FRAME_SOURCE_ATTR, "") or "").strip()
    except Exception:
        return ""


def read_snapshot_metadata(market_date) -> dict[str, object]:
    try:
        meta_path = _snapshot_meta_path(market_date)
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_scan_data_plan() -> dict[str, object]:
    """
    Return a fully-resolved data-flow plan for the current IST moment.

    Decision matrix
    ───────────────
    LIVE        09:30–16:00 Mon–Fri  → always fetch live; never snapshot
    CLOSED      after 16:00 Mon–Fri  → snapshot if it exists, else fetch+save
    PRE_MARKET  before 09:30 Mon–Fri → previous trading day snapshot only
    WEEKEND     Sat / Sun             → last trading day snapshot only

    The returned dict contains **exactly** these keys (plus diagnostics):
        use_snapshot            bool
        force_live_refresh      bool
        save_snapshot_after_scan bool
        window                  str   LIVE | CLOSED | PRE_MARKET | WEEKEND
        expected_date           date  the market date the scan data should represent
        snapshot_exists         bool
        snapshot_path           Path
        live_window_label       str
        source_label            str   short human-readable tag
        summary                 str   one-line UI description
    """
    window        = get_current_window()
    live_window   = get_market_hours_label()

    # ── Determine the target market date ──────────────────────────────
    now_ist = _now_ist()
    today   = now_ist.date()

    if window == "LIVE":
        # Active session — data should be today's
        expected_date = today

    elif window == "CLOSED":
        # Same calendar day; snapshot will be for today
        expected_date = today

    elif window == "PRE_MARKET":
        # Before today's open — use the last completed trading day
        expected_date = _previous_weekday(today - timedelta(days=1))

    else:  # WEEKEND
        # Roll back to the last weekday (Friday or earlier)
        expected_date = _previous_weekday(today)

    # ── Check whether a valid snapshot already exists ─────────────────
    has_snapshot = snapshot_exists(expected_date)
    snap_path    = get_snapshot_path(expected_date)

    # ── Apply the decision matrix ─────────────────────────────────────
    if window == "LIVE":
        # Always live — ignore any snapshot, do not build one mid-session
        use_snapshot             = False
        force_live_refresh       = True
        save_snapshot_after_scan = False
        source_label = "LIVE MARKET DATA (Refreshing)"
        summary = (
            f"Market is open ({live_window}). "
            "Scanner fetches fresh data on every run. "
            "No snapshot is used or saved during live hours."
        )

    elif window == "CLOSED":
        if has_snapshot:
            # Snapshot already saved for today — serve it exclusively for
            # the rest of the evening (consistent, stable results)
            use_snapshot             = True
            force_live_refresh       = False
            save_snapshot_after_scan = False
            source_label = "CLOSED MARKET (Snapshot Loaded)"
            summary = (
                f"Market closed. Today's snapshot ({expected_date.isoformat()}) "
                "is loaded. Results are stable for the remainder of the evening."
            )
        else:
            # First scan after close — fetch live data and save a snapshot
            use_snapshot             = False
            force_live_refresh       = True
            save_snapshot_after_scan = True
            source_label = "CLOSED MARKET (Building Snapshot)"
            summary = (
                f"Market closed. No snapshot found for {expected_date.isoformat()}. "
                "Fetching live data and saving a snapshot for the evening."
            )

    elif window == "PRE_MARKET":
        # Strict rule: NEVER touch live feeds before market open.
        # Pre-market API data (yfinance etc.) is incomplete or stale and
        # will produce incorrect scan results.  Snapshot is the ONLY valid
        # source regardless of whether one exists.
        use_snapshot             = True
        force_live_refresh       = False
        save_snapshot_after_scan = False
        source_label = "PRE-MARKET (Using Previous Day Data)"

        if has_snapshot:
            # Primary path — today's expected snapshot is present.
            summary = (
                f"Pre-market session. Using stable snapshot from "
                f"{expected_date.isoformat()}. Live refresh is disabled."
            )
        else:
            # Snapshot for expected_date is missing.
            # Step 1: look for the nearest older snapshot before falling back.
            fallback_date = _latest_snapshot_on_or_before(expected_date)

            if fallback_date is not None:
                # Step 2: a usable fallback snapshot was found — use it.
                expected_date = fallback_date
                has_snapshot  = True
                snap_path     = get_snapshot_path(expected_date)
                summary = (
                    f"Pre-market session. Expected snapshot was missing. "
                    f"Using nearest available snapshot ({expected_date.isoformat()}). "
                    "Live refresh is disabled."
                )
            else:
                # Step 3: no snapshot of any kind exists — warn the user.
                # Do NOT enable live fetch; pre-market data is unreliable.
                summary = (
                    f"Pre-market session. Snapshot for {expected_date.isoformat()} "
                    "is missing and no fallback snapshot was found. "
                    "Live data is intentionally disabled to prevent stale or "
                    "incorrect results. Please run the scanner after market "
                    "close to generate a snapshot."
                )

    else:  # WEEKEND
        use_snapshot             = True
        force_live_refresh       = False
        save_snapshot_after_scan = False
        source_label = "WEEKEND (Using Last Trading Day Data)"

        if has_snapshot:
            summary = (
                f"Weekend session. Using last trading day snapshot "
                f"({expected_date.isoformat()}). Live refresh is disabled."
            )
        else:
            # Best-effort: try to find the nearest available snapshot
            fallback_date = _latest_snapshot_on_or_before(expected_date)
            if fallback_date is not None:
                expected_date = fallback_date
                has_snapshot  = True
                snap_path     = get_snapshot_path(expected_date)
                summary = (
                    f"Weekend session. No snapshot found for the expected date. "
                    f"Using nearest available snapshot ({expected_date.isoformat()})."
                )
            else:
                # Absolute last resort — fetch live and save
                use_snapshot             = False
                force_live_refresh       = True
                save_snapshot_after_scan = True
                summary = (
                    "Weekend session. No snapshot available. "
                    "Performing a one-time live fetch to build a snapshot."
                )

    return {
        "window":                   window,
        "expected_date":            expected_date,
        "snapshot_exists":          has_snapshot,
        "snapshot_path":            snap_path,
        "live_window_label":        live_window,
        "use_snapshot":             use_snapshot,
        "force_live_refresh":       force_live_refresh,
        "save_snapshot_after_scan": save_snapshot_after_scan,
        "source_label":             source_label,
        "summary":                  summary,
    }


def is_data_fresh(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if _time_travel_active():
        return True

    try:
        last_seen = pd.to_datetime(df.index[-1]).date()
    except Exception:
        return False

    try:
        return last_seen in set(get_acceptable_data_dates())
    except Exception:
        return False


def get_snapshot_path(market_date) -> Path:
    try:
        snap_day = _coerce_date(market_date)
    except Exception:
        snap_day = _now_ist().date()
    return _SNAPSHOT_ROOT / snap_day.isoformat()


def snapshot_exists(market_date) -> bool:
    try:
        snap_dir = get_snapshot_path(market_date)
        if not snap_dir.exists():
            return False
        return len(list(snap_dir.glob("*.csv"))) >= 100
    except Exception:
        return False


def _cleanup_old_snapshots(reference_date: date) -> None:
    try:
        if not _SNAPSHOT_ROOT.exists():
            return
        cutoff = reference_date - timedelta(days=7)
        for child in _SNAPSHOT_ROOT.iterdir():
            if not child.is_dir():
                continue
            try:
                snap_day = date.fromisoformat(child.name)
            except Exception:
                continue
            if snap_day < cutoff:
                shutil.rmtree(child, ignore_errors=True)
    except Exception:
        pass


def save_closing_snapshot(ALL_DATA: dict, market_date, require_live_source: bool = False) -> int:
    try:
        snap_day = _coerce_date(market_date)
        if get_current_window() != "CLOSED":
            return 0
        if snapshot_exists(snap_day):
            return 0
        if not isinstance(ALL_DATA, dict) or not ALL_DATA:
            return 0

        snap_dir = get_snapshot_path(snap_day)
        snap_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for ticker, df in ALL_DATA.items():
            try:
                if df is None or df.empty or not isinstance(df, pd.DataFrame):
                    continue
                if require_live_source and not get_frame_source(df).startswith("live"):
                    continue
                last_seen = pd.to_datetime(df.index[-1]).date()
                if last_seen != snap_day:
                    continue
                safe_name = str(ticker).replace(":", "_").replace("/", "_")
                out_path = snap_dir / f"{safe_name}.csv"
                df.sort_index().to_csv(out_path)
                saved += 1
            except Exception:
                continue

        if saved > 0:
            meta = {
                "market_date": snap_day.isoformat(),
                "captured_at": _now_ist().isoformat(),
                "window": get_current_window(),
                "saved": saved,
            }
            _snapshot_meta_path(snap_day).write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

        _cleanup_old_snapshots(snap_day)
        return saved
    except Exception:
        return 0


def load_snapshot_into_ALL_DATA(market_date) -> int:
    try:
        snap_dir = get_snapshot_path(market_date)
        if not snap_dir.exists():
            return 0

        from strategy_engines._engine_utils import (
            ALL_DATA,
            _ALL_DATA_LOCK,
            _NO_DATA_LOCK,
            _coerce_no_data_tickers,
        )

        loaded_frames: dict[str, pd.DataFrame] = {}
        snapshot_meta = read_snapshot_metadata(market_date)
        captured_at = str(snapshot_meta.get("captured_at", "") or "").strip() or None
        for csv_path in snap_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if df is None or df.empty:
                    continue
                df = df.sort_index()
                df = stamp_frame_metadata(
                    df,
                    source="snapshot",
                    market_date=_coerce_date(market_date),
                    window="CLOSED",
                    captured_at=captured_at,
                )
                loaded_frames[csv_path.stem] = df
            except Exception:
                continue

        with _ALL_DATA_LOCK:
            ALL_DATA.clear()
            for ticker, df in loaded_frames.items():
                ALL_DATA[ticker] = df
        with _NO_DATA_LOCK:
            _coerce_no_data_tickers().clear()

        return len(loaded_frames)
    except Exception:
        return 0


def get_data_status_label() -> str:
    """
    Return a single emoji-prefixed status string for the Streamlit UI banner.

    Maps directly to the four window states produced by get_scan_data_plan():
        LIVE        → 🟢  LIVE MARKET DATA (Refreshing)
        CLOSED      → 🔵  CLOSED MARKET (Snapshot Loaded | Building Snapshot)
        PRE_MARKET  → 🟡  PRE-MARKET (Using Previous Day Data)
        WEEKEND     → ⚪  WEEKEND (Using Last Trading Day Data)
    """
    plan     = get_scan_data_plan()
    snap_day = plan.get("expected_date")
    day_text = snap_day.isoformat() if isinstance(snap_day, date) else str(snap_day)
    window   = str(plan.get("window", "") or "").upper()

    if window == "LIVE":
        return f"🟢 LIVE MARKET DATA (Refreshing) — {day_text}"

    if window == "CLOSED":
        if plan.get("use_snapshot"):
            return f"🔵 CLOSED MARKET (Snapshot Loaded) — {day_text}"
        return f"🔵 CLOSED MARKET (Building Snapshot) — {day_text}"

    if window == "PRE_MARKET":
        return f"🟡 PRE-MARKET (Using Previous Day Data) — {day_text}"

    # WEEKEND
    return f"⚪ WEEKEND (Using Last Trading Day Data) — {day_text}"