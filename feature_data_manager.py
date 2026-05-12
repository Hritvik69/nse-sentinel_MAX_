from __future__ import annotations

import json
import hashlib
import threading
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from atomic_io import atomic_write_json

try:
    import streamlit as st
except Exception:
    st = None  # type: ignore[assignment]

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore[assignment]

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

try:
    import data_session_manager as _dsm
except Exception:
    _dsm = None  # type: ignore[assignment]

try:
    import time_travel_engine as _tt
except Exception:
    _tt = None  # type: ignore[assignment]

try:
    from strategy_engines._engine_utils import ALL_DATA, _ALL_DATA_LOCK
except Exception:
    ALL_DATA = {}  # type: ignore[assignment]
    _ALL_DATA_LOCK = threading.Lock()


_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else timezone(timedelta(hours=5, minutes=30))
_ROOT = Path(__file__).resolve().parent
_FEATURE_CACHE_ROOT = _ROOT / "data" / "feature_cache"
_SCANNER_SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"
_PREDICTION_CACHE_TTL_MINUTES = 90
_SAVED_TODAY: set[str] = set()
_SAVED_TODAY_DAY: date | None = None
_SAVED_TODAY_LOCK = threading.Lock()


def _now_ist() -> datetime:
    try:
        return datetime.now(_IST_TZ)
    except Exception:
        return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _should_save_stock_cache_once(symbol: str, cache_day: date) -> bool:
    global _SAVED_TODAY_DAY
    cache_key = f"{symbol}:{cache_day.isoformat()}"
    with _SAVED_TODAY_LOCK:
        if _SAVED_TODAY_DAY != cache_day:
            _SAVED_TODAY.clear()
            _SAVED_TODAY_DAY = cache_day
        if cache_key in _SAVED_TODAY:
            return False
        _SAVED_TODAY.add(cache_key)
        return True


def _get_real_current_window() -> str:
    fn = getattr(_dsm, "get_current_window", None)
    if callable(fn):
        try:
            return str(fn() or "CLOSED").upper()
        except Exception:
            pass
    now_ist = _now_ist()
    current_time = now_ist.time()
    if now_ist.weekday() >= 5:
        return "WEEKEND"
    if current_time >= datetime.strptime("09:30", "%H:%M").time() and current_time <= datetime.strptime("16:00", "%H:%M").time():
        return "LIVE"
    if current_time > datetime.strptime("16:00", "%H:%M").time():
        return "CLOSED"
    return "PRE_MARKET"


def _get_real_expected_data_date() -> date:
    fn = getattr(_dsm, "get_expected_data_date", None)
    if callable(fn):
        try:
            value = fn()
            return pd.to_datetime(value).date()
        except Exception:
            pass
    return _now_ist().date()


def _coerce_date_value(value: object) -> date | None:
    if value in (None, "", "None"):
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _get_time_travel_date() -> date | None:
    if _tt is not None:
        try:
            parsed = _coerce_date_value(getattr(_tt, "get_reference_date", lambda: None)())
            if parsed is not None:
                return parsed
        except Exception:
            pass
    return None


def get_current_window() -> str:
    return "SIMULATED" if _get_time_travel_date() is not None else _get_real_current_window()


def get_expected_data_date() -> date:
    return _get_time_travel_date() or _get_real_expected_data_date()


def get_time_travel_date() -> date | None:
    return _get_time_travel_date()


def _get_frame_source(df: pd.DataFrame | None) -> str:
    fn = getattr(_dsm, "get_frame_source", None)
    if callable(fn):
        try:
            return str(fn(df) or "").strip()
        except Exception:
            pass
    try:
        if df is None:
            return ""
        return str(df.attrs.get("_nse_data_source", "") or "").strip()
    except Exception:
        return ""


def _status_icon(kind: str) -> str:
    icons = {
        "LIVE": "🟢",
        "CACHED": "📂",
        "SIMULATED": "🕰️",
        "SNAPSHOT": "⚪",
        "PRE_MARKET": "🟡",
        "WEEKEND": "⚪",
        "MISSING": "🔴",
    }
    return icons.get(kind, "📂")


def _source_kind_for_frame(source: str, window: str) -> str:
    src = str(source or "").strip().lower()
    current_window = str(window or "").strip().upper()
    if current_window == "SIMULATED" or src.startswith("simulated"):
        return "SIMULATED"
    if src.startswith("live"):
        return "LIVE"
    if src.startswith("snapshot"):
        return "PRE_MARKET" if current_window == "PRE_MARKET" else "SNAPSHOT"
    if src.startswith("feature_cache") or src.startswith("cache"):
        if current_window == "PRE_MARKET":
            return "PRE_MARKET"
        if current_window == "WEEKEND":
            return "WEEKEND"
        return "CACHED"
    if src.startswith("csv"):
        return "SNAPSHOT" if current_window in {"PRE_MARKET", "WEEKEND"} else "CACHED"
    if current_window == "PRE_MARKET":
        return "PRE_MARKET"
    if current_window == "WEEKEND":
        return "WEEKEND"
    return "CACHED"


def _format_as_of(value: object) -> str:
    if value in (None, "", "None"):
        return ""
    try:
        dt = pd.to_datetime(value)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.tz_localize(_IST_TZ)
        else:
            dt = dt.tz_convert(_IST_TZ)
        return dt.strftime("%d %b %Y %H:%M IST")
    except Exception:
        return str(value)


def _frame_market_date(df: pd.DataFrame | None) -> date | None:
    if df is None or df.empty:
        return None
    try:
        attr_value = str(df.attrs.get("_nse_market_date", "") or "").strip()
        if attr_value:
            return pd.to_datetime(attr_value).date()
    except Exception:
        pass
    try:
        return pd.to_datetime(df.index[-1]).date()
    except Exception:
        return None


def _acceptable_market_dates() -> list[date]:
    fn = getattr(_dsm, "get_acceptable_data_dates", None)
    if callable(fn):
        try:
            values = list(fn())
            return [pd.to_datetime(value).date() for value in values]
        except Exception:
            pass
    return [_get_real_expected_data_date()]


def _format_simulation_note(cutoff: date | None) -> str:
    if cutoff is None:
        return ""
    try:
        return f"Time Simulation: {cutoff.strftime('%d %b %Y')}"
    except Exception:
        return f"Time Simulation: {cutoff}"


def render_data_status_badge(status: dict[str, Any] | None, label: str = "") -> None:
    if st is None or not status:
        return
    source_kind = str(status.get("source_kind", "CACHED") or "CACHED").upper()
    icon = _status_icon(source_kind)
    title = {
        "LIVE": "LIVE - Refreshing every scan",
        "CACHED": "CACHED",
        "SIMULATED": "TIME SIMULATION",
        "SNAPSHOT": "SNAPSHOT",
        "PRE_MARKET": "PRE-MARKET",
        "WEEKEND": "WEEKEND",
        "MISSING": "DATA UNAVAILABLE",
    }.get(source_kind, source_kind)
    as_of = str(status.get("as_of", "") or "").strip()
    note = str(status.get("note", "") or "").strip()
    parts = [f"{icon} {title}"]
    if as_of:
        parts.append(f"Data as of: {as_of}")
    if label:
        parts.append(label)
    if note:
        parts.append(note)
    st.caption(" | ".join(parts))


class FeatureDataManager:
    """
    Shared data manager for all non-scan features.

    Priority order:
    1. ALL_DATA
    2. feature_cache JSON
    3. existing scanner snapshot CSV
    4. legacy data/*.csv cache
    5. yfinance (LIVE / CLOSED, plus simulated historical fetches)
    """

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = Path(cache_root) if cache_root is not None else _FEATURE_CACHE_ROOT
        self._status_lock = threading.Lock()
        self._last_status: dict[str, dict[str, Any]] = {}

    def _normalize_symbol(self, symbol: str, append_nse_suffix: bool = True) -> str:
        raw = str(symbol or "").strip().upper()
        if not raw:
            return ""
        if append_nse_suffix and not raw.startswith("^") and not raw.endswith(".NS"):
            return f"{raw}.NS"
        return raw

    def _normalize_sector(self, sector_name: str) -> str:
        return str(sector_name or "").strip().upper().replace(" ", "_")

    def _cache_day(self) -> date:
        return get_expected_data_date()

    def _time_travel_cutoff(self) -> date | None:
        return get_time_travel_date()

    def _day_dir(self, cache_day: date) -> Path:
        return self.cache_root / cache_day.isoformat()

    def _meta_path(self, cache_day: date) -> Path:
        return self._day_dir(cache_day) / "meta.json"

    def _stock_dir(self, cache_day: date) -> Path:
        return self._day_dir(cache_day) / "stocks"

    def _sector_dir(self, cache_day: date) -> Path:
        return self._day_dir(cache_day) / "sectors"

    def _prediction_dir(self, cache_day: date) -> Path:
        return self._day_dir(cache_day) / "predictions"

    def _compare_dir(self, cache_day: date) -> Path:
        return self._day_dir(cache_day) / "compare"

    def _stock_path(self, cache_day: date, symbol: str) -> Path:
        safe = self._normalize_symbol(symbol, append_nse_suffix=False)
        return self._stock_dir(cache_day) / f"{safe}.json"

    def _sector_path(self, cache_day: date, sector_name: str) -> Path:
        return self._sector_dir(cache_day) / f"{self._normalize_sector(sector_name)}.json"

    def _prediction_path(self, cache_day: date, sector_name: str) -> Path:
        return self._prediction_dir(cache_day) / f"{self._normalize_sector(sector_name)}.json"

    def _compare_path(self, cache_day: date, symbols: list[str]) -> Path:
        key = hashlib.sha256("|".join(self._normalize_compare_symbols(symbols)).encode("utf-8")).hexdigest()[:24]
        return self._compare_dir(cache_day) / f"compare_{key}.json"

    def _legacy_compare_path(self, cache_day: date, symbols: list[str]) -> Path:
        joined = "_".join(sorted(self._normalize_symbol(sym, append_nse_suffix=False) for sym in symbols if str(sym).strip()))
        safe = joined.replace("/", "_").replace(":", "_")
        return self._compare_dir(cache_day) / f"{safe}.json"

    def _normalize_compare_symbols(self, symbols: list[str]) -> list[str]:
        return sorted(
            {
                self._normalize_symbol(sym, append_nse_suffix=False)
                for sym in symbols
                if str(sym).strip()
            }
        )

    def _status_note(self, note: str = "", *, tt_cutoff: date | None = None) -> str:
        parts: list[str] = []
        tt_note = _format_simulation_note(tt_cutoff)
        if tt_note:
            parts.append(tt_note)
        plain_note = str(note or "").strip()
        if plain_note:
            parts.append(plain_note)
        return " / ".join(parts)

    def _period_lookback_days(self, period: str) -> int:
        raw = str(period or "").strip().lower()
        if not raw:
            return 120
        if raw == "max":
            return 3650
        digits = "".join(ch for ch in raw if ch.isdigit())
        value = int(digits) if digits else 0
        if value <= 0:
            return 120
        if raw.endswith("mo"):
            return max(31, value * 31)
        if raw.endswith("wk"):
            return max(14, value * 7)
        if raw.endswith("y"):
            return max(366, value * 366)
        if raw.endswith("d"):
            return max(14, value)
        return 120

    def _cache_covers_period(
        self,
        df: pd.DataFrame | None,
        *,
        period: str,
        cache_day: date,
        min_rows: int,
    ) -> bool:
        try:
            if df is None or df.empty or len(df) < max(1, int(min_rows)):
                return False
            if str(period or "").strip().lower() in {"", "1d", "5d"}:
                return True
            required_days = self._period_lookback_days(period)
            if required_days <= 14:
                return True
            idx = pd.to_datetime(df.index, errors="coerce")
            if pd.isna(idx).any():
                return False
            first_day = idx.min().date()
            last_day = idx.max().date()
            if last_day > cache_day:
                return False
            span_days = max(0, (last_day - first_day).days)
            tolerance = max(5, min(30, int(required_days * 0.15)))
            return span_days >= max(0, required_days - tolerance)
        except Exception:
            return False

    def _ensure_day_dirs(self, cache_day: date) -> None:
        day_dir = self._day_dir(cache_day)
        for path in (
            day_dir,
            self._stock_dir(cache_day),
            self._sector_dir(cache_day),
            self._prediction_dir(cache_day),
            self._compare_dir(cache_day),
        ):
            path.mkdir(parents=True, exist_ok=True)
        meta = {
            "date": cache_day.isoformat(),
            "window": get_current_window(),
            "saved_at": _now_ist().isoformat(),
        }
        atomic_write_json(self._meta_path(cache_day), meta, indent=2)

    def _coerce_frame(self, df: pd.DataFrame | None, min_rows: int = 5) -> pd.DataFrame | None:
        try:
            if df is None or df.empty:
                return None
            src_attrs = dict(getattr(df, "attrs", {}) or {})
            out = df.copy()
            if isinstance(out.columns, pd.MultiIndex):
                out.columns = out.columns.get_level_values(0)
            out.columns = [str(col).strip().title() for col in out.columns]
            needed = ["Open", "High", "Low", "Close"]
            if not set(needed).issubset(out.columns):
                return None
            if "Volume" not in out.columns:
                out["Volume"] = 0.0
            cols = needed + ["Volume"]
            out = out[cols].copy()
            idx = pd.to_datetime(out.index, errors="coerce")
            valid = ~idx.isna()
            out = out.loc[valid].copy()
            out.index = idx[valid]
            out = out[~out.index.duplicated(keep="last")].sort_index()
            for col in cols:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=needed)
            out["Volume"] = out["Volume"].fillna(0.0)
            out.attrs.update(src_attrs)
            return out if len(out) >= max(1, int(min_rows)) else None
        except Exception:
            return None

    def _apply_time_travel_cutoff(
        self,
        df: pd.DataFrame | None,
        *,
        cutoff: date | None,
        min_rows: int = 5,
    ) -> pd.DataFrame | None:
        normalized = self._coerce_frame(df, min_rows=1)
        if normalized is None:
            return None
        attrs = dict(getattr(normalized, "attrs", {}) or {})
        if cutoff is not None:
            try:
                if _tt is not None and hasattr(_tt, "truncate_df"):
                    normalized = _tt.truncate_df(normalized, cutoff, min_rows=min_rows)
                else:
                    return None
            except Exception:
                return None
            if normalized is None:
                return None
        normalized = self._coerce_frame(normalized, min_rows=min_rows)
        if normalized is None:
            return None
        normalized.attrs.update(attrs)
        if cutoff is not None:
            normalized.attrs["_nse_tt_cutoff"] = cutoff.isoformat()
        return normalized

    def _frame_payload(
        self,
        df: pd.DataFrame,
        *,
        symbol: str = "",
        sector_name: str = "",
        period: str = "",
        interval: str = "",
        source: str = "",
        market_date: date | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "sector_name": sector_name,
            "period": period,
            "interval": interval,
            "source": source,
            "window": get_current_window(),
            "market_date": (market_date or self._cache_day()).isoformat(),
            "saved_at": _now_ist().isoformat(),
            "frame": df.to_json(orient="split", date_format="iso"),
        }

    def _load_frame_payload(self, path: Path, min_rows: int = 5) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame_json = str(payload.get("frame", "") or "")
            if not frame_json:
                return None, payload
            df = pd.read_json(StringIO(frame_json), orient="split")
            df = self._coerce_frame(df, min_rows=min_rows)
            if df is None:
                return None, payload
            df.attrs["_nse_data_source"] = str(payload.get("source", "feature_cache") or "feature_cache")
            df.attrs["_nse_market_date"] = str(payload.get("market_date", "") or "")
            df.attrs["_nse_window"] = str(payload.get("window", "") or "")
            df.attrs["_nse_captured_at"] = str(payload.get("saved_at", "") or "")
            return df, payload
        except Exception:
            return None, {}

    def _record_status(self, key: str, status: dict[str, Any]) -> None:
        with self._status_lock:
            self._last_status[str(key)] = dict(status)

    def get_last_status(self, key: str) -> dict[str, Any] | None:
        with self._status_lock:
            value = self._last_status.get(str(key))
            return dict(value) if isinstance(value, dict) else None

    def _make_status(
        self,
        *,
        key: str,
        source_kind: str,
        source: str,
        market_date: date | None,
        saved_at: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        tt_cutoff = self._time_travel_cutoff()
        status = {
            "key": key,
            "source_kind": source_kind,
            "source": source,
            "window": get_current_window(),
            "market_date": market_date.isoformat() if market_date is not None else "",
            "as_of": _format_as_of(saved_at) if saved_at else "",
            "note": self._status_note(note, tt_cutoff=tt_cutoff),
        }
        self._record_status(key, status)
        return status

    def _is_acceptable_frame(
        self,
        df: pd.DataFrame | None,
        *,
        window: str,
        cache_day: date,
    ) -> bool:
        market_date = _frame_market_date(df)
        if market_date is None:
            return df is not None and not df.empty
        if market_date == cache_day:
            return True
        if window in {"SIMULATED", "PRE_MARKET", "WEEKEND"}:
            return market_date <= cache_day

        recent_days: list[date] = []
        cur = cache_day
        while len(recent_days) < 3:
            if cur.weekday() < 5:
                recent_days.append(cur)
            cur -= timedelta(days=1)
        return market_date in set(recent_days)

    def _lookup_all_data(
        self,
        symbol: str,
        min_rows: int = 5,
        *,
        window: str,
        cache_day: date,
    ) -> pd.DataFrame | None:
        tt_cutoff = cache_day if window == "SIMULATED" else None
        keys = [symbol]
        plain = symbol.replace(".NS", "")
        if plain not in keys:
            keys.append(plain)
        ns_key = self._normalize_symbol(plain, append_nse_suffix=True)
        if ns_key not in keys:
            keys.append(ns_key)
        try:
            with _ALL_DATA_LOCK:
                for key in keys:
                    df = self._apply_time_travel_cutoff(
                        ALL_DATA.get(key),
                        cutoff=tt_cutoff,
                        min_rows=min_rows,
                    )
                    if df is not None and self._is_acceptable_frame(df, window=window, cache_day=cache_day):
                        return df
        except Exception:
            return None
        return None

    def _write_all_data(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            tt_cutoff = self._time_travel_cutoff()
            if tt_cutoff is not None:
                try:
                    import time_travel_engine as _tt_cache

                    _tt_cache.cache_frame(symbol, df, tt_cutoff, min_rows=5)
                except Exception:
                    pass
                return
            with _ALL_DATA_LOCK:
                ALL_DATA[symbol] = df
                plain = symbol.replace(".NS", "")
                ALL_DATA[plain] = df
        except Exception:
            pass

    def _resolve_snapshot_path(self, symbol: str, cache_day: date) -> tuple[Path | None, date | None]:
        candidates = [cache_day] + [cache_day - timedelta(days=offset) for offset in range(1, 8)]
        for day in candidates:
            csv_path = _SCANNER_SNAPSHOT_ROOT / day.isoformat() / f"{symbol}.csv"
            if csv_path.exists():
                return csv_path, day
        return None, None

    def _snapshot_file_valid(self, csv_path: Path, snapshot_day: date) -> tuple[bool, str]:
        meta_path = _SCANNER_SNAPSHOT_ROOT / snapshot_day.isoformat() / "_meta.json"
        if not meta_path.exists():
            return True, ""
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not bool(meta.get("complete", False)):
                return False, ""
            checksums = meta.get("checksums", {})
            saved_at = str(meta.get("captured_at", "") or "")
            if isinstance(checksums, dict) and checksums:
                expected = str(checksums.get(csv_path.name, "") or "")
                if not expected:
                    return False, saved_at
                digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
                if digest != expected:
                    return False, saved_at
            return True, saved_at
        except Exception:
            return False, ""

    def _load_scanner_snapshot(self, symbol: str, cache_day: date, min_rows: int = 5) -> tuple[pd.DataFrame | None, str]:
        csv_path, snapshot_day = self._resolve_snapshot_path(symbol, cache_day)
        if csv_path is None or snapshot_day is None:
            return None, ""
        valid_snapshot, saved_at = self._snapshot_file_valid(csv_path, snapshot_day)
        if not valid_snapshot:
            return None, ""
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception:
            return None, ""
        df = self._coerce_frame(df, min_rows=min_rows)
        if df is None:
            return None, ""
        df.attrs["_nse_data_source"] = "snapshot"
        df.attrs["_nse_market_date"] = snapshot_day.isoformat()
        if saved_at:
            df.attrs["_nse_captured_at"] = saved_at
        return df, saved_at

    def _load_legacy_csv_cache(self, symbol: str, min_rows: int = 5) -> tuple[pd.DataFrame | None, str]:
        try:
            import data_downloader

            df = data_downloader.load_csv(symbol)
            df = self._coerce_frame(df, min_rows=min_rows)
            if df is None:
                return None, ""

            saved_at = ""
            try:
                safe = symbol.replace(":", "_").replace("/", "_")
                csv_path = data_downloader.DATA_DIR / f"{safe}.csv"
                if csv_path.exists():
                    saved_at = datetime.fromtimestamp(csv_path.stat().st_mtime, _IST_TZ).isoformat()
            except Exception:
                saved_at = ""

            market_date = _frame_market_date(df)
            df.attrs["_nse_data_source"] = "csv_cache"
            df.attrs["_nse_market_date"] = (market_date or self._cache_day()).isoformat()
            df.attrs["_nse_window"] = get_current_window()
            if saved_at:
                df.attrs["_nse_captured_at"] = saved_at
            return df, saved_at
        except Exception:
            return None, ""

    def _save_stock_cache(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        period: str,
        interval: str,
        source: str,
        cache_day: date,
    ) -> None:
        self._ensure_day_dirs(cache_day)
        payload = self._frame_payload(
            df,
            symbol=symbol,
            period=period,
            interval=interval,
            source=source,
            market_date=_frame_market_date(df) or cache_day,
        )
        atomic_write_json(self._stock_path(cache_day, symbol), payload, indent=2)
        self._evict_old_stock_cache(cache_day)

    def _load_stock_cache(
        self,
        symbol: str,
        *,
        period: str,
        interval: str,
        cache_day: date,
        min_rows: int = 5,
    ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        path = self._stock_path(cache_day, symbol)
        if not path.exists():
            return None, {}
        df, payload = self._load_frame_payload(path, min_rows=min_rows)
        if df is None:
            return None, payload
        if str(payload.get("interval", interval) or interval) != interval:
            return None, payload
        if not self._cache_covers_period(
            df,
            period=period,
            cache_day=cache_day,
            min_rows=min_rows,
        ):
            return None, payload
        return df, payload

    def _evict_old_stock_cache(self, cache_day: date, max_symbols: int = 200) -> None:
        stock_dir = self._stock_dir(cache_day)
        try:
            files = sorted(
                [path for path in stock_dir.glob("*.json") if path.is_file()],
                key=lambda item: item.stat().st_mtime,
            )
        except Exception:
            return
        overflow = max(0, len(files) - max_symbols)
        for path in files[:overflow]:
            try:
                path.unlink()
            except Exception:
                continue

    def _fetch_yfinance(
        self,
        symbol: str,
        period: str,
        interval: str,
        min_rows: int = 5,
        *,
        cutoff: date | None = None,
    ) -> pd.DataFrame | None:
        if yf is None:
            return None
        try:
            if cutoff is not None:
                lookback_days = max(self._period_lookback_days(period) + 14, min_rows * 3)
                start = cutoff - timedelta(days=lookback_days)
                end = cutoff + timedelta(days=1)
                df = yf.download(
                    symbol,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                    threads=False,
                )
            else:
                df = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    timeout=15,
                    threads=False,
                )
        except Exception:
            return None
        return self._apply_time_travel_cutoff(df, cutoff=cutoff, min_rows=min_rows)

    def get_symbol_data(
        self,
        symbol: str,
        period: str = "2mo",
        interval: str = "1d",
        force_refresh: bool = False,
        *,
        append_nse_suffix: bool = True,
        min_rows: int = 5,
        allow_snapshot: bool = True,
    ) -> pd.DataFrame | None:
        window = get_current_window()
        cache_day = self._cache_day()
        tt_cutoff = cache_day if window == "SIMULATED" else None
        normalized = self._normalize_symbol(symbol, append_nse_suffix=append_nse_suffix)
        if not normalized:
            self._make_status(
                key=str(symbol),
                source_kind="MISSING",
                source="missing",
                market_date=cache_day,
                note="Blank symbol.",
            )
            return None

        status_key = normalized
        cached_fallback: tuple[pd.DataFrame | None, dict[str, Any]] = (None, {})

        if not force_refresh:
            df_all = self._lookup_all_data(
                normalized,
                min_rows=min_rows,
                window=window,
                cache_day=cache_day,
            )
            if df_all is not None:
                source = _get_frame_source(df_all) or "all_data"
                source_kind = _source_kind_for_frame(source, window)
                self._write_all_data(normalized, df_all)
                if append_nse_suffix and _should_save_stock_cache_once(normalized, cache_day):
                    self._save_stock_cache(
                        normalized,
                        df_all,
                        period=period,
                        interval=interval,
                        source=source,
                        cache_day=cache_day,
                    )
                self._make_status(
                    key=status_key,
                    source_kind=source_kind,
                    source=source,
                    market_date=_frame_market_date(df_all) or cache_day,
                    saved_at=str(df_all.attrs.get("_nse_captured_at", "") or ""),
                )
                return df_all

            df_cache, payload = self._load_stock_cache(
                normalized,
                period=period,
                interval=interval,
                cache_day=cache_day,
                min_rows=min_rows,
            )
            df_cache = self._apply_time_travel_cutoff(df_cache, cutoff=tt_cutoff, min_rows=min_rows)
            if df_cache is not None and self._is_acceptable_frame(df_cache, window=window, cache_day=cache_day):
                cached_fallback = (df_cache, payload)
                source = str(payload.get("source", "feature_cache") or "feature_cache")
                self._make_status(
                    key=status_key,
                    source_kind=_source_kind_for_frame(source, window),
                    source=source,
                    market_date=_frame_market_date(df_cache) or _coerce_date_value(payload.get("market_date")) or cache_day,
                    saved_at=str(payload.get("saved_at", "") or ""),
                )
                return df_cache

            if allow_snapshot:
                df_snapshot, saved_at = self._load_scanner_snapshot(normalized, cache_day, min_rows=min_rows)
                df_snapshot = self._apply_time_travel_cutoff(df_snapshot, cutoff=tt_cutoff, min_rows=min_rows)
                if df_snapshot is not None and self._is_acceptable_frame(df_snapshot, window=window, cache_day=cache_day):
                    self._write_all_data(normalized, df_snapshot)
                    self._save_stock_cache(
                        normalized,
                        df_snapshot,
                        period=period,
                        interval=interval,
                        source="snapshot",
                        cache_day=cache_day,
                    )
                    self._make_status(
                        key=status_key,
                        source_kind=_source_kind_for_frame("snapshot", window),
                        source="snapshot",
                        market_date=_frame_market_date(df_snapshot) or cache_day,
                        saved_at=saved_at,
                    )
                    return df_snapshot

            df_csv, csv_saved_at = self._load_legacy_csv_cache(normalized, min_rows=min_rows)
            df_csv = self._apply_time_travel_cutoff(df_csv, cutoff=tt_cutoff, min_rows=min_rows)
            if df_csv is not None and self._is_acceptable_frame(df_csv, window=window, cache_day=cache_day):
                self._write_all_data(normalized, df_csv)
                self._save_stock_cache(
                    normalized,
                    df_csv,
                    period=period,
                    interval=interval,
                    source="csv_cache",
                    cache_day=cache_day,
                )
                self._make_status(
                    key=status_key,
                    source_kind=_source_kind_for_frame("csv_cache", window),
                    source="csv_cache",
                    market_date=_frame_market_date(df_csv) or cache_day,
                    saved_at=csv_saved_at,
                )
                return df_csv

        snapshot_dir = _SCANNER_SNAPSHOT_ROOT / cache_day.isoformat()
        missing_snapshot_note = ""
        if window == "CLOSED" and snapshot_dir.exists():
            missing_snapshot_note = (
                "Selected stock is not available in the locked market snapshot; "
                "using the final chart-data fallback."
            )

        if window in {"PRE_MARKET", "WEEKEND"}:
            df_cache, payload = cached_fallback
            if df_cache is not None:
                return df_cache
            self._make_status(
                key=status_key,
                source_kind=window,
                source="locked",
                market_date=cache_day,
                note="Live fetch disabled in this market window.",
            )
            return None

        df_live = self._fetch_yfinance(
            normalized,
            period=period,
            interval=interval,
            min_rows=min_rows,
            cutoff=tt_cutoff,
        )
        if df_live is not None:
            source = "simulated_feature" if tt_cutoff is not None else "live_feature"
            market_date = _frame_market_date(df_live) or cache_day
            df_live.attrs["_nse_data_source"] = source
            df_live.attrs["_nse_market_date"] = market_date.isoformat()
            df_live.attrs["_nse_window"] = window
            df_live.attrs["_nse_captured_at"] = _now_ist().isoformat()
            self._write_all_data(normalized, df_live)
            self._save_stock_cache(
                normalized,
                df_live,
                period=period,
                interval=interval,
                source=source,
                cache_day=cache_day,
            )
            self._make_status(
                key=status_key,
                source_kind=_source_kind_for_frame(source, window),
                source=source,
                market_date=market_date,
                saved_at=str(df_live.attrs.get("_nse_captured_at", "") or ""),
                note=missing_snapshot_note,
            )
            return df_live

        df_cache, payload = cached_fallback
        if df_cache is not None:
            failure_note = "Historical fetch failed. Reused cached simulated data." if tt_cutoff is not None else "Live fetch failed. Reused cached feature data."
            self._make_status(
                key=status_key,
                source_kind=_source_kind_for_frame(str(payload.get("source", "feature_cache") or "feature_cache"), window),
                source=str(payload.get("source", "feature_cache") or "feature_cache"),
                market_date=_frame_market_date(df_cache) or _coerce_date_value(payload.get("market_date")) or cache_day,
                saved_at=str(payload.get("saved_at", "") or ""),
                note=failure_note,
            )
            return df_cache

        self._make_status(
            key=status_key,
            source_kind="MISSING",
            source="missing",
            market_date=cache_day,
            note=(
                f"{missing_snapshot_note} No data available from ALL_DATA, feature cache, "
                "snapshot, legacy CSV, or yfinance."
            ).strip(),
        )
        return None

    def get_stock_data(
        self,
        symbol: str,
        period: str = "2mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame | None:
        return self.get_symbol_data(
            symbol,
            period=period,
            interval=interval,
            force_refresh=force_refresh,
            append_nse_suffix=True,
            min_rows=5,
            allow_snapshot=True,
        )

    def get_multiple_stocks(
        self,
        symbols: list[str],
        period: str = "2mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        seen: set[str] = set()
        for raw in symbols:
            symbol = self._normalize_symbol(raw, append_nse_suffix=True)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            df = self.get_stock_data(
                symbol,
                period=period,
                interval=interval,
                force_refresh=force_refresh,
            )
            if df is not None:
                out[symbol] = df
        return out

    def _load_sector_stocks(self, sector_name: str) -> list[str]:
        try:
            from sector_master import get_stocks_in_sector

            stocks = list(get_stocks_in_sector(sector_name))
            if stocks:
                return stocks
        except Exception:
            pass
        try:
            from strategy_engines.multi_index_market_bias_engine import get_dashboard_sector_stocks

            stocks = list(get_dashboard_sector_stocks(sector_name))
            if stocks:
                return stocks
        except Exception:
            pass
        return []

    def _aggregate_sector_ohlc(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        if not frames:
            return None
        aligned: list[pd.DataFrame] = []
        for df in frames.values():
            norm = self._coerce_frame(df, min_rows=5)
            if norm is not None:
                aligned.append(norm[["Open", "High", "Low", "Close", "Volume"]].copy())
        if not aligned:
            return None
        common_index = aligned[0].index
        for frame in aligned[1:]:
            common_index = common_index.intersection(frame.index)
        if len(common_index) < 5:
            return None
        trimmed = [frame.reindex(common_index) for frame in aligned]
        out = pd.DataFrame(index=common_index)
        out["Open"] = pd.concat([frame["Open"] for frame in trimmed], axis=1).mean(axis=1)
        out["High"] = pd.concat([frame["High"] for frame in trimmed], axis=1).max(axis=1)
        out["Low"] = pd.concat([frame["Low"] for frame in trimmed], axis=1).min(axis=1)
        out["Close"] = pd.concat([frame["Close"] for frame in trimmed], axis=1).mean(axis=1)
        out["Volume"] = pd.concat([frame["Volume"] for frame in trimmed], axis=1).sum(axis=1)
        return self._coerce_frame(out, min_rows=5)

    def save_sector_ohlc_cache(self, sector_name: str, df: pd.DataFrame, top_n: int) -> None:
        cache_day = self._cache_day()
        self._ensure_day_dirs(cache_day)
        payload = self._frame_payload(
            df,
            sector_name=sector_name,
            period=f"top_n={top_n}",
            interval="1d",
            source="feature_cache_sector",
            market_date=_frame_market_date(df) or cache_day,
        )
        atomic_write_json(self._sector_path(cache_day, sector_name), payload, indent=2)
        self._make_status(
            key=f"sector:{self._normalize_sector(sector_name)}",
            source_kind=_source_kind_for_frame("feature_cache_sector", get_current_window()),
            source="feature_cache_sector",
            market_date=_frame_market_date(df) or cache_day,
            saved_at=str(payload.get("saved_at", "") or ""),
        )

    def load_sector_ohlc_cache(self, sector_name: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
        cache_day = self._cache_day()
        path = self._sector_path(cache_day, sector_name)
        if not path.exists():
            return None, {}
        df, payload = self._load_frame_payload(path, min_rows=5)
        df = self._apply_time_travel_cutoff(
            df,
            cutoff=cache_day if get_current_window() == "SIMULATED" else None,
            min_rows=5,
        )
        if df is not None:
            source = str(payload.get("source", "feature_cache_sector") or "feature_cache_sector")
            self._make_status(
                key=f"sector:{self._normalize_sector(sector_name)}",
                source_kind=_source_kind_for_frame(source, get_current_window()),
                source=source,
                market_date=_frame_market_date(df) or _coerce_date_value(payload.get("market_date")) or cache_day,
                saved_at=str(payload.get("saved_at", "") or ""),
            )
        return df, payload

    def get_sector_stocks_data(
        self,
        sector_name: str,
        top_n: int = 5,
        period: str = "2mo",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        stocks = self._load_sector_stocks(sector_name)
        if top_n > 0:
            stocks = stocks[:top_n]
        frames = self.get_multiple_stocks(
            stocks,
            period=period,
            interval=interval,
            force_refresh=force_refresh,
        )
        sector_ohlc = self._aggregate_sector_ohlc(frames)
        if sector_ohlc is not None:
            self.save_sector_ohlc_cache(sector_name, sector_ohlc, top_n=top_n)
            self._make_status(
                key=f"sector:{self._normalize_sector(sector_name)}",
                source_kind=_source_kind_for_frame("feature_cache_sector", get_current_window()),
                source="feature_cache_sector",
                market_date=_frame_market_date(sector_ohlc) or self._cache_day(),
                saved_at=_now_ist().isoformat(),
            )
        return frames

    def load_prediction_cache(self, sector_name: str) -> dict[str, Any] | None:
        cache_day = self._cache_day()
        path = self._prediction_path(cache_day, sector_name)
        try:
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            saved_at_str = str(payload.get("saved_at", "") or "")
            if saved_at_str:
                try:
                    saved_dt = datetime.fromisoformat(saved_at_str)
                    if saved_dt.tzinfo is None:
                        saved_dt = saved_dt.replace(tzinfo=timezone.utc)
                    age_min = (datetime.now(tz=saved_dt.tzinfo) - saved_dt).total_seconds() / 60.0
                    window = str(payload.get("window", "CLOSED") or "CLOSED").upper()
                    if window in ("LIVE", "PRE_MARKET") and age_min > _PREDICTION_CACHE_TTL_MINUTES:
                        return None
                except Exception:
                    pass
            return payload
        except Exception:
            return None

    def save_prediction_cache(self, sector_name: str, payload: dict[str, Any]) -> None:
        cache_day = self._cache_day()
        self._ensure_day_dirs(cache_day)
        out = dict(payload)
        out.setdefault("saved_at", _now_ist().isoformat())
        out.setdefault("market_date", cache_day.isoformat())
        out.setdefault("window", get_current_window())
        atomic_write_json(self._prediction_path(cache_day, sector_name), out, indent=2)

    def load_compare_cache(self, symbols: list[str]) -> dict[str, Any] | None:
        cache_day = self._cache_day()
        normalized_symbols = self._normalize_compare_symbols(symbols)
        paths = [self._compare_path(cache_day, symbols), self._legacy_compare_path(cache_day, symbols)]
        try:
            for path in paths:
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload_symbols = payload.get("symbols", None) if isinstance(payload, dict) else None
                if isinstance(payload_symbols, list) and self._normalize_compare_symbols(payload_symbols) != normalized_symbols:
                    continue
                return payload
        except Exception:
            return None
        return None

    def save_compare_cache(self, symbols: list[str], payload: dict[str, Any]) -> None:
        cache_day = self._cache_day()
        self._ensure_day_dirs(cache_day)
        out = dict(payload)
        out["symbols"] = self._normalize_compare_symbols(symbols)
        out.setdefault("saved_at", _now_ist().isoformat())
        out.setdefault("market_date", cache_day.isoformat())
        out.setdefault("window", get_current_window())
        atomic_write_json(self._compare_path(cache_day, symbols), out, indent=2)

    def invalidate_cache(self, symbol: str = "", sector_name: str = "") -> None:
        cache_day = self._cache_day()
        if symbol:
            path = self._stock_path(cache_day, symbol)
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        if sector_name:
            for path in (self._sector_path(cache_day, sector_name), self._prediction_path(cache_day, sector_name)):
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass


feature_manager = FeatureDataManager()
