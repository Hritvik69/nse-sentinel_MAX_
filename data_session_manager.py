from __future__ import annotations

import shutil
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)
_ROOT = Path(__file__).resolve().parent
_SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"


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
        return last_seen == get_expected_data_date()
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


def save_closing_snapshot(ALL_DATA: dict, market_date) -> int:
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
                last_seen = pd.to_datetime(df.index[-1]).date()
                if last_seen != snap_day:
                    continue
                safe_name = str(ticker).replace(":", "_").replace("/", "_")
                out_path = snap_dir / f"{safe_name}.csv"
                df.sort_index().to_csv(out_path)
                saved += 1
            except Exception:
                continue

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
        for csv_path in snap_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if df is None or df.empty:
                    continue
                df = df.sort_index()
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
    window = get_current_window()
    snap_day = get_expected_data_date()
    day_text = snap_day.isoformat()

    if window == "LIVE":
        return f"🟢 Live Market Data — {day_text}"
    if window == "CLOSED":
        return f"🔵 Closing Snapshot — {day_text} 3:30 PM"
    if window == "PRE_MARKET":
        return f"🟡 Previous Close — {day_text}"
    return f"🟡 Last Close — {day_text}"
