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
    window = get_current_window()
    expected_date = get_expected_data_date()
    snap_path = get_snapshot_path(expected_date)
    has_snapshot = snapshot_exists(expected_date)
    live_window = get_market_hours_label()

    plan: dict[str, object] = {
        "window": window,
        "expected_date": expected_date,
        "snapshot_exists": has_snapshot,
        "snapshot_path": snap_path,
        "live_window_label": live_window,
        "use_snapshot": False,
        "force_live_refresh": False,
        "save_snapshot_after_scan": False,
        "source_label": "",
        "summary": "",
    }
    plan.update(
        {
            "use_snapshot": False,
            "force_live_refresh": True,
            "save_snapshot_after_scan": False,
            "source_label": "Always live refresh",
            "summary": (
                f"Main scanner now refreshes live data on every run and ignores saved snapshots at startup. "
                f"Market window reference: {live_window}."
            ),
        }
    )
    return plan


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
    plan = get_scan_data_plan()
    snap_day = plan.get("expected_date")
    day_text = snap_day.isoformat() if isinstance(snap_day, date) else str(snap_day)
    window = str(plan.get("window", "") or "").upper()
    has_snapshot = bool(plan.get("snapshot_exists", False))

    if window == "LIVE":
        return f"🟢 Live Market Refresh — {day_text}"
    if window == "CLOSED":
        return f"🟢 Always Live Refresh — {day_text}"
    if window == "PRE_MARKET":
        return f"🟢 Always Live Refresh — {day_text}"
    return f"🟢 Always Live Refresh — {day_text}"
