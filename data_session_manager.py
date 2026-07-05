from __future__ import annotations

import json
import hashlib
import logging
import shutil
import tempfile
import threading
import random
import uuid
import zipfile
from functools import lru_cache
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pandas as pd
from atomic_io import atomic_write_bytes, atomic_write_csv_df, atomic_write_json
from safe_paths import safe_filename, safe_join

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)
_ROOT = Path(__file__).resolve().parent
_SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"
_SNAPSHOT_ARCHIVE = _ROOT / "data" / "market_snapshot_latest.zip"
_FRAME_SOURCE_ATTR = "_nse_data_source"
_FRAME_DATE_ATTR = "_nse_market_date"
_FRAME_WINDOW_ATTR = "_nse_window"
_FRAME_CAPTURED_AT_ATTR = "_nse_captured_at"
_LOG = logging.getLogger(__name__)
_SNAPSHOT_SAVE_LOCK = threading.RLock()
_SNAPSHOT_RESTORE_LOCK = threading.Lock()
_SNAPSHOT_RESTORE_SIG: tuple[int, int] | None = None


def _invalidate_snapshot_caches() -> None:
    try:
        _snapshot_exists_cached.cache_clear()
        _available_snapshot_dates_cached.cache_clear()
        _latest_snapshot_on_or_before_cached.cache_clear()
        _acceptable_data_dates_cached.cache_clear()
    except Exception:
        pass


def _restore_snapshot_archive_if_available() -> None:
    global _SNAPSHOT_RESTORE_SIG
    try:
        if not _SNAPSHOT_ARCHIVE.exists():
            return
        archive_stat = _SNAPSHOT_ARCHIVE.stat()
        archive_sig = (int(archive_stat.st_mtime_ns), int(archive_stat.st_size))
        if _SNAPSHOT_RESTORE_SIG == archive_sig:
            return
        try:
            if _SNAPSHOT_ROOT.exists() and any(child.is_dir() for child in _SNAPSHOT_ROOT.iterdir()):
                _SNAPSHOT_RESTORE_SIG = archive_sig
                return
        except Exception:
            pass
        with _SNAPSHOT_RESTORE_LOCK:
            if _SNAPSHOT_RESTORE_SIG == archive_sig:
                return
            _restore_snapshot_archive_unlocked()
            _SNAPSHOT_RESTORE_SIG = archive_sig
    except Exception:
        pass


def _restore_snapshot_archive_unlocked() -> None:
    try:
        _SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
        root_resolved = _SNAPSHOT_ROOT.resolve()
        with zipfile.ZipFile(_SNAPSHOT_ARCHIVE, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename.replace("\\", "/").lstrip("/")
                if not name or name.startswith("../") or "/../" in name:
                    continue
                if not (name.endswith(".csv") or name.endswith("_meta.json")):
                    continue
                target = (_SNAPSHOT_ROOT / name).resolve()
                try:
                    target.relative_to(root_resolved)
                except Exception:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as src:
                    atomic_write_bytes(target, src.read())
    except Exception:
        pass


def _archive_latest_snapshot(snap_dir: Path) -> None:
    tmp_path: Path | None = None
    try:
        if not snap_dir.exists() or not snap_dir.is_dir():
            return
        with _SNAPSHOT_SAVE_LOCK:
            _SNAPSHOT_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{_SNAPSHOT_ARCHIVE.stem}.",
                suffix=".zip.tmp",
                dir=str(_SNAPSHOT_ARCHIVE.parent),
                delete=False,
            )
            tmp_path = Path(handle.name)
            handle.close()
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in snap_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    if not (path.name.endswith(".csv") or path.name == "_meta.json"):
                        continue
                    archive.write(path, arcname=str(Path(snap_dir.name) / path.relative_to(snap_dir)))
            tmp_path.replace(_SNAPSHOT_ARCHIVE)
            tmp_path = None
            try:
                from persistent_store import push_file

                if not push_file(_SNAPSHOT_ARCHIVE):
                    _LOG.error("snapshot archive sync queueing failed")
            except Exception:
                _LOG.exception("snapshot archive sync failed")
    except Exception:
        _LOG.exception("snapshot archive creation failed")
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _snapshot_meta_path(market_date) -> Path:
    return get_snapshot_path(market_date) / "_meta.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_csv_name(ticker: object) -> str:
    return safe_filename(ticker, ".csv")


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


def _csv_data_newer_than_snapshot(snapshot_date: date | None) -> bool:
    """
    Check if individual CSV files in the data/ folder have data more recent
    than the given snapshot date. This ensures we prefer fresh CSV data over
    old snapshots on weekends.
    """
    if snapshot_date is None:
        return False
    try:
        from data_downloader import _csv_read_path
        _DATA_DIR = _ROOT / "data"
        if not _DATA_DIR.exists():
            return False

        newest_date: date | None = None
        csv_files = list(_DATA_DIR.glob("*.csv"))
        if not csv_files:
            return False

        sample_size = min(20, len(csv_files))
        import random
        sampled = random.sample(csv_files, sample_size)

        for csv_path in sampled:
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if df is not None and not df.empty:
                    last_row_date = pd.to_datetime(df.index[-1]).date()
                    if newest_date is None or last_row_date > newest_date:
                        newest_date = last_row_date
            except Exception:
                continue

        if newest_date is not None and newest_date > snapshot_date:
            return True
    except Exception:
        pass
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


@lru_cache(maxsize=1)
def _available_snapshot_dates_cached() -> tuple[date, ...]:
    dates: list[date] = []
    try:
        _restore_snapshot_archive_if_available()
        if not _SNAPSHOT_ROOT.exists():
            return tuple()
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
        return tuple()
    return tuple(sorted(dates))


def _available_snapshot_dates() -> list[date]:
    return list(_available_snapshot_dates_cached())


@lru_cache(maxsize=64)
def _latest_snapshot_on_or_before_cached(day_iso: str) -> date | None:
    day = date.fromisoformat(day_iso)
    latest: date | None = None
    try:
        for snap_day in _available_snapshot_dates_cached():
            if snap_day <= day:
                latest = snap_day
            else:
                break
    except Exception:
        return None
    return latest


def _latest_snapshot_on_or_before(day: date) -> date | None:
    try:
        return _latest_snapshot_on_or_before_cached(day.isoformat())
    except Exception:
        return None


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


@lru_cache(maxsize=32)
def _acceptable_data_dates_cached(
    today_iso: str,
    window: str,
    snapshot_fallback_iso: str,
) -> tuple[date, ...]:
    today = date.fromisoformat(today_iso)
    window = str(window or "").upper()
    snapshot_fallback = date.fromisoformat(snapshot_fallback_iso) if snapshot_fallback_iso else None

    if window in ("LIVE", "CLOSED"):
        primary = _recent_market_days(today, count=3)
    elif window == "PRE_MARKET":
        primary = _recent_market_days(_previous_weekday(today - timedelta(days=1)), count=3)
    else:
        primary = _recent_market_days(_previous_weekday(today), count=3)

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
    return tuple(ordered)


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

    snapshot_fallback = _latest_snapshot_on_or_before(today)
    return list(
        _acceptable_data_dates_cached(
            today.isoformat(),
            window,
            snapshot_fallback.isoformat() if snapshot_fallback is not None else "",
        )
    )


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
        if has_snapshot:
            source_label = "PRE-MARKET (Snapshot Loaded)"
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
                source_label  = "PRE-MARKET (Fallback Snapshot Loaded)"
                summary = (
                    f"Pre-market session. Expected snapshot was missing. "
                    f"Using nearest available snapshot ({expected_date.isoformat()}). "
                    "Live refresh is disabled."
                )
            else:
                # Step 3: no snapshot of any kind exists — warn the user.
                # Do NOT enable live fetch; pre-market data is unreliable.
                source_label = "PRE-MARKET (Snapshot Missing)"
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

        if has_snapshot:
            # Check if individual CSV files have fresher data than this snapshot
            if _csv_data_newer_than_snapshot(expected_date):
                # CSV data is more recent - skip the old snapshot
                use_snapshot             = False
                force_live_refresh       = False
                source_label             = "WEEKEND (CSV Data Loaded)"
                summary = (
                    f"Weekend session. CSV data ({expected_date.isoformat()}+) "
                    "is more recent than snapshot. Using CSV files instead."
                )
            else:
                source_label = "WEEKEND (Snapshot Loaded)"
                summary = (
                    f"Weekend session. Using last trading day snapshot "
                    f"({expected_date.isoformat()}). Live refresh is disabled."
                )
        else:
            # Best-effort: try to find the nearest available snapshot
            fallback_date = _latest_snapshot_on_or_before(expected_date)
            if fallback_date is not None:
                # Check if CSV data is fresher than the fallback snapshot
                if _csv_data_newer_than_snapshot(fallback_date):
                    # CSV data is more recent - don't use the old snapshot
                    expected_date = fallback_date
                    has_snapshot  = False
                    use_snapshot   = False
                    force_live_refresh = False
                    snap_path = get_snapshot_path(expected_date)
                    source_label   = "WEEKEND (CSV Data Loaded)"
                    summary = (
                        f"Weekend session. CSV data is more recent than "
                        f"nearest snapshot ({fallback_date.isoformat()}). "
                        "Using CSV files instead."
                    )
                else:
                    expected_date = fallback_date
                    has_snapshot  = True
                    snap_path     = get_snapshot_path(expected_date)
                    source_label  = "WEEKEND (Fallback Snapshot Loaded)"
                    summary = (
                        f"Weekend session. No snapshot found for the expected date. "
                        f"Using nearest available snapshot ({expected_date.isoformat()})."
                    )
            else:
                # Absolute last resort — fetch live and save
                use_snapshot             = False
                force_live_refresh       = True
                save_snapshot_after_scan = True
                source_label             = "WEEKEND (Building Snapshot)"
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
        tagged_day = df.attrs.get(_FRAME_DATE_ATTR) if hasattr(df, "attrs") else None
        if tagged_day:
            last_seen = _coerce_date(tagged_day)
        else:
            last_seen = pd.to_datetime(df.index[-1]).date()
    except Exception:
        return False

    try:
        expected_day = get_expected_data_date()
        if last_seen == expected_day:
            return True
    except Exception:
        pass

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
        snap_day = _coerce_date(market_date)
        return _snapshot_exists_cached(snap_day.isoformat())
    except Exception:
        return False


@lru_cache(maxsize=64)
def _snapshot_exists_cached(day_iso: str) -> bool:
    try:
        _restore_snapshot_archive_if_available()
        snap_dir = _SNAPSHOT_ROOT / day_iso
        if not snap_dir.exists():
            return False
        meta_path = snap_dir / "_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not bool(meta.get("complete", False)):
                return False
            saved = int(meta.get("saved", meta.get("count", 0)) or 0)
            if saved <= 0:
                return False
            checksums = meta.get("checksums", {})
            if not isinstance(checksums, dict) or len(checksums) < saved:
                return False
            for name, expected_hash in checksums.items():
                filename = str(name or "").strip()
                expected = str(expected_hash or "").strip().lower()
                if not filename or not expected:
                    return False
                csv_path = (snap_dir / filename).resolve(strict=False)
                try:
                    csv_path.relative_to(snap_dir.resolve(strict=False))
                except Exception:
                    return False
                if not csv_path.exists() or not csv_path.is_file():
                    return False
                if _file_sha256(csv_path).lower() != expected:
                    return False
            return True
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
        _invalidate_snapshot_caches()
    except Exception:
        pass


def save_closing_snapshot(ALL_DATA: dict, market_date, require_live_source: bool = False) -> int:
    temp_dir: Path | None = None
    old_dir: Path | None = None
    try:
        snap_day = _coerce_date(market_date)
        current_window = get_current_window()
        if current_window not in {"CLOSED", "WEEKEND"}:
            return 0
        if not isinstance(ALL_DATA, dict) or not ALL_DATA:
            return 0

        with _SNAPSHOT_SAVE_LOCK:
            if snapshot_exists(snap_day):
                return 0

            snap_dir = get_snapshot_path(snap_day)
            snap_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{snap_day.isoformat()}.",
                    suffix=".tmp",
                    dir=str(snap_dir.parent),
                )
            )

            saved = 0
            checksums: dict[str, str] = {}
            symbols: dict[str, str] = {}
            for ticker, df in ALL_DATA.items():
                try:
                    if df is None or df.empty or not isinstance(df, pd.DataFrame):
                        continue
                    if require_live_source and not get_frame_source(df).startswith("live"):
                        continue
                    last_seen = pd.to_datetime(df.index[-1]).date()
                    if last_seen != snap_day:
                        continue
                    file_name = _snapshot_csv_name(ticker)
                    out_path = safe_join(temp_dir, file_name)
                    atomic_write_csv_df(out_path, df.sort_index())
                    checksums[out_path.name] = _file_sha256(out_path)
                    symbols[out_path.name] = str(ticker)
                    saved += 1
                except Exception:
                    continue

            if saved <= 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir = None
                _cleanup_old_snapshots(snap_day)
                return 0

            meta = {
                "market_date": snap_day.isoformat(),
                "captured_at": _now_ist().isoformat(),
                "window": get_current_window(),
                "saved": saved,
                "count": saved,
                "complete": True,
                "checksums": checksums,
                "symbols": symbols,
            }
            atomic_write_json(temp_dir / "_meta.json", meta, indent=2)
            if snap_dir.exists():
                old_dir = snap_dir.with_name(f".{snap_dir.name}.partial-{int(datetime.now().timestamp())}-{uuid.uuid4().hex[:8]}")
                try:
                    snap_dir.replace(old_dir)
                except Exception:
                    shutil.rmtree(snap_dir, ignore_errors=True)
            temp_dir.replace(snap_dir)
            temp_dir = None
            _archive_latest_snapshot(snap_dir)
            _invalidate_snapshot_caches()
            if old_dir is not None:
                shutil.rmtree(old_dir, ignore_errors=True)
                old_dir = None

            _cleanup_old_snapshots(snap_day)
            return saved
    except Exception:
        return 0
    finally:
        if temp_dir is not None:
            try:
                parent = temp_dir.parent.resolve()
                expected_parent = get_snapshot_path(market_date).parent.resolve()
                if parent == expected_parent and temp_dir.name.startswith(f".{_coerce_date(market_date).isoformat()}."):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        if old_dir is not None:
            try:
                snap_dir = get_snapshot_path(market_date)
                parent = old_dir.parent.resolve()
                expected_parent = snap_dir.parent.resolve()
                if parent == expected_parent and old_dir.name.startswith(f".{_coerce_date(market_date).isoformat()}.partial-"):
                    if snap_dir.exists():
                        shutil.rmtree(old_dir, ignore_errors=True)
                    elif old_dir.exists():
                        old_dir.replace(snap_dir)
            except Exception:
                pass


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
        checksums = snapshot_meta.get("checksums", {})
        symbols = snapshot_meta.get("symbols", {})
        csv_paths = list(snap_dir.glob("*.csv"))
        if isinstance(checksums, dict) and checksums:
            csv_paths = []
            snap_root = snap_dir.resolve(strict=False)
            for name in checksums:
                try:
                    candidate = (snap_dir / str(name)).resolve(strict=False)
                    candidate.relative_to(snap_root)
                except Exception:
                    continue
                if candidate.exists():
                    csv_paths.append(candidate)
        for csv_path in csv_paths:
            try:
                if isinstance(checksums, dict) and checksums:
                    expected_hash = str(checksums.get(csv_path.name, "") or "")
                    if not expected_hash or _file_sha256(csv_path) != expected_hash:
                        continue
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
                ticker_name = ""
                if isinstance(symbols, dict):
                    ticker_name = str(symbols.get(csv_path.name, "") or "").strip()
                loaded_frames[ticker_name or csv_path.stem] = df
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
        if plan.get("use_snapshot") and plan.get("snapshot_exists"):
            source_text = str(plan.get("source_label") or "PRE-MARKET (Snapshot Loaded)")
            return f"🟡 {source_text} — {day_text}"
        return f"🟡 PRE-MARKET (Snapshot Missing) — {day_text}"

    # WEEKEND
    if plan.get("use_snapshot") and plan.get("snapshot_exists"):
        source_text = str(plan.get("source_label") or "WEEKEND (Snapshot Loaded)")
        return f"⚪ {source_text} — {day_text}"
    if plan.get("save_snapshot_after_scan"):
        return f"⚪ WEEKEND (Building Snapshot) — {day_text}"
    return f"⚪ WEEKEND (Snapshot Missing) — {day_text}"
