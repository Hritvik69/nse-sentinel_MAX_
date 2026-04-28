"""
strategy_engines/_engine_utils.py
──────────────────────────────────
Shared low-level helpers: EMA, RSI (vectorised), yfinance download, safe cast.
Imported by every mode engine — kept minimal and side-effect-free.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

_MAX_CONC = 12                              # aligned with app worker cap
_SEM      = threading.BoundedSemaphore(_MAX_CONC)
_PRELOAD_BATCH_SIZE = 60
_MAX_PRELOAD_BATCH_CONC = 4
_MARKET_OPEN_IST = dtime(9, 15)
_IST_TZ = ZoneInfo("Asia/Kolkata") if ZoneInfo is not None else None

# ── Central data store (zero-API scan) ───────────────────────────────
ALL_DATA: dict[str, pd.DataFrame | None] = {}
_ALL_DATA_LOCK = threading.Lock()
_NO_DATA_TICKERS: set[str] = set()
_NO_DATA_LOCK = threading.Lock()
_LAST_LIVE_CACHE_DATE: date | None = None
_LIVE_CACHE_LOCK = threading.Lock()


# ── optional sklearn ──────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


def safe(v: object, default: float = 0.0) -> float:
    """Return float(v) if finite, else default."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi_vec(close: pd.Series, period: int = 14) -> pd.Series:
    """Fully vectorised RSI series — no per-row Python loop."""
    d = close.diff()
    g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def _normalize_ohlcv_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Standardize yfinance OHLCV output and reject unusable frames."""
    try:
        if df is None or df.empty:
            return None
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        out = out.dropna(subset=["Close", "Volume"])
        return out if len(out) >= 30 else None
    except Exception:
        return None


def _prepare_loaded_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    return _normalize_ohlcv_frame(df)


def _now_ist() -> datetime:
    if _IST_TZ is not None:
        return datetime.now(_IST_TZ)
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _previous_market_day(day: date) -> date:
    cur = day
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def _expected_live_data_date() -> date | None:
    """
    Return the market date a live scan should be using in IST.

    In Time Travel mode we disable staleness enforcement entirely because the
    active cutoff date is intentionally historical.
    """
    try:
        import time_travel_engine as _tt

        if _tt.is_active():
            return None
    except Exception:
        pass

    now_ist = _now_ist()
    today = now_ist.date()

    if today.weekday() >= 5:
        return _previous_market_day(today)

    if now_ist.time() < _MARKET_OPEN_IST:
        return _previous_market_day(today - timedelta(days=1))

    return today


def _frame_last_market_date(df: pd.DataFrame | None) -> date | None:
    try:
        if df is None or df.empty:
            return None
        return pd.to_datetime(df.index[-1]).date()
    except Exception:
        return None


def _is_stale_live_frame(df: pd.DataFrame | None) -> bool:
    expected = _expected_live_data_date()
    if expected is None:
        return False

    last_seen = _frame_last_market_date(df)
    if last_seen is None:
        return True

    return last_seen < expected


def _is_fresh_live_frame(df: pd.DataFrame | None) -> bool:
    return df is not None and not _is_stale_live_frame(df)


def is_fresh_enough(df: pd.DataFrame | None, strict: bool = False) -> bool:
    """
    Public freshness helper for scan callers.

    strict=True enforces the expected live market date from the central engine.
    strict=False keeps the older loose "within the last week" fallback check.
    Time Travel mode bypasses live-date freshness and relies on its own cutoff.
    """
    if df is None or df.empty:
        return False

    try:
        import time_travel_engine as _tt

        if _tt.is_active():
            return True
    except Exception:
        pass

    last_seen = _frame_last_market_date(df)
    if last_seen is None:
        return False

    if strict:
        return not _is_stale_live_frame(df)

    return (_now_ist().date() - last_seen).days <= 7


def _persist_frame_to_csv(ticker_ns: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return

    try:
        from data_downloader import DATA_DIR

        safe = ticker_ns.replace(":", "_").replace("/", "_")
        path = DATA_DIR / f"{safe}.csv"
        df.sort_index().to_csv(path)
    except Exception:
        pass


def _has_recent_no_data(ticker_ns: str) -> bool:
    with _NO_DATA_LOCK:
        return ticker_ns in _coerce_no_data_tickers()


def _coerce_no_data_tickers() -> set[str]:
    """Normalize stale cache state from previous reloads back into a set."""
    global _NO_DATA_TICKERS

    if isinstance(_NO_DATA_TICKERS, set):
        return _NO_DATA_TICKERS
    if isinstance(_NO_DATA_TICKERS, dict):
        _NO_DATA_TICKERS = set(_NO_DATA_TICKERS)
        return _NO_DATA_TICKERS
    if _NO_DATA_TICKERS is None:
        _NO_DATA_TICKERS = set()
        return _NO_DATA_TICKERS

    try:
        _NO_DATA_TICKERS = set(_NO_DATA_TICKERS)
    except TypeError:
        _NO_DATA_TICKERS = set()
    return _NO_DATA_TICKERS


def _reset_live_caches_if_market_day_changed() -> None:
    """
    Reset process-global live caches when the expected market date advances.

    Streamlit can keep the worker alive overnight, so this cannot rely on
    per-session state inside app.py.
    """
    expected = _expected_live_data_date()
    if expected is None:
        return

    global _LAST_LIVE_CACHE_DATE
    with _LIVE_CACHE_LOCK:
        if _LAST_LIVE_CACHE_DATE is None:
            _LAST_LIVE_CACHE_DATE = expected
            return
        if _LAST_LIVE_CACHE_DATE == expected:
            return

        with _ALL_DATA_LOCK:
            ALL_DATA.clear()
        with _NO_DATA_LOCK:
            _coerce_no_data_tickers().clear()
        _LAST_LIVE_CACHE_DATE = expected


def _mark_no_data(ticker_ns: str) -> None:
    with _NO_DATA_LOCK:
        _coerce_no_data_tickers().add(ticker_ns)


def _clear_no_data(ticker_ns: str) -> None:
    with _NO_DATA_LOCK:
        _coerce_no_data_tickers().discard(ticker_ns)


def download_history(ticker_ns: str, period: str = "6mo") -> pd.DataFrame | None:
    """Download daily OHLCV; returns None on failure or if < 30 rows."""
    if _has_recent_no_data(ticker_ns):
        return None

    try:
        with _SEM:
            df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12, threads=False,
            )
        if df is None or df.empty:
            _mark_no_data(ticker_ns)
            return None
        prepared = _prepare_loaded_frame(df)
        if prepared is None:
            _mark_no_data(ticker_ns)
            return None
        if _is_stale_live_frame(prepared):
            return None
        _clear_no_data(ticker_ns)
        return prepared
    except Exception:
        return None


def _fetch_one(ticker_ns: str, period: str) -> tuple[str, pd.DataFrame | None]:
    """Load one ticker for preloading; prefer CSV if available."""
    try:
        from data_downloader import load_csv
        df = _prepare_loaded_frame(load_csv(ticker_ns))
        if df is not None:
            _clear_no_data(ticker_ns)
            return ticker_ns, df
    except Exception:
        pass
    return ticker_ns, download_history(ticker_ns, period=period)


def _download_batch(
    tickers_ns: list[str],
    period: str,
) -> tuple[dict[str, pd.DataFrame | None], bool]:
    """Download a ticker batch in one Yahoo call and split it back by symbol."""
    if not tickers_ns:
        return {}, True

    try:
        with _SEM:
            batch_df = yf.download(
                tickers_ns,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=20,
                threads=True,
                group_by="ticker",
            )
    except Exception:
        return {ticker_ns: None for ticker_ns in tickers_ns}, False

    if batch_df is None or batch_df.empty:
        return {ticker_ns: None for ticker_ns in tickers_ns}, False

    out: dict[str, pd.DataFrame | None] = {}

    if len(tickers_ns) == 1 and not isinstance(batch_df.columns, pd.MultiIndex):
        prepared = _prepare_loaded_frame(batch_df)
        out[tickers_ns[0]] = prepared if _is_fresh_live_frame(prepared) else None
        return out, True

    if not isinstance(batch_df.columns, pd.MultiIndex):
        return {ticker_ns: None for ticker_ns in tickers_ns}, False

    available = set(batch_df.columns.get_level_values(0))
    for ticker_ns in tickers_ns:
        if ticker_ns not in available:
            out[ticker_ns] = None
            continue
        try:
            one_df = batch_df.xs(ticker_ns, axis=1, level=0)
        except Exception:
            out[ticker_ns] = None
            continue
        prepared = _prepare_loaded_frame(one_df)
        out[ticker_ns] = prepared if _is_fresh_live_frame(prepared) else None

    return out, True


def preload_all(
    tickers: list[str],
    period: str = "6mo",
    workers: int = 12,
    progress_callback=None,
) -> None:
    """
    Fill ALL_DATA with OHLCV DataFrames for every ticker in parallel.
    Called once before run_scan() so analyse() can use get_df_for_ticker().
    """
    _reset_live_caches_if_market_day_changed()

    tickers_ns = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    total = len(tickers_ns)
    done = 0
    loaded = 0

    def _emit_progress() -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(done, total, loaded)
        except Exception:
            pass

    with _ALL_DATA_LOCK:
        existing = {ticker_ns: ALL_DATA.get(ticker_ns) for ticker_ns in tickers_ns}
    with _NO_DATA_LOCK:
        known_no_data = set(_coerce_no_data_tickers())

    remaining: list[str] = []
    stale_fallbacks: dict[str, pd.DataFrame] = {}
    for ticker_ns in tickers_ns:
        cached = existing.get(ticker_ns)
        if cached is not None and isinstance(cached, pd.DataFrame) and not cached.empty:
            if _is_stale_live_frame(cached):
                stale_fallbacks[ticker_ns] = cached
                remaining.append(ticker_ns)
            else:
                done += 1
                loaded += 1
        elif ticker_ns in known_no_data:
            done += 1
        else:
            remaining.append(ticker_ns)
    _emit_progress()

    download_queue: list[str] = []
    for ticker_ns in remaining:
        try:
            from data_downloader import load_csv

            csv_df = _prepare_loaded_frame(load_csv(ticker_ns))
        except Exception:
            csv_df = None

        if csv_df is not None:
            if _is_stale_live_frame(csv_df):
                stale_fallbacks.setdefault(ticker_ns, csv_df)
                download_queue.append(ticker_ns)
            else:
                with _ALL_DATA_LOCK:
                    ALL_DATA[ticker_ns] = csv_df
                _clear_no_data(ticker_ns)
                done += 1
                loaded += 1
                _emit_progress()
        else:
            download_queue.append(ticker_ns)

    if not download_queue:
        return

    batches = [
        download_queue[i:i + _PRELOAD_BATCH_SIZE]
        for i in range(0, len(download_queue), _PRELOAD_BATCH_SIZE)
    ]
    batch_workers = min(
        max(1, int(workers)),
        _MAX_PRELOAD_BATCH_CONC,
        len(batches),
    )

    def _fetch_batch(batch: list[str]) -> dict[str, pd.DataFrame | None]:
        frames, batch_ok = _download_batch(batch, period)
        if not batch_ok:
            for ticker_ns in batch:
                frames[ticker_ns] = download_history(ticker_ns, period=period)
        return frames

    with ThreadPoolExecutor(max_workers=batch_workers) as ex:
        futs = {ex.submit(_fetch_batch, batch): batch for batch in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            try:
                batch_frames = fut.result()
            except Exception:
                batch_frames = {ticker_ns: None for ticker_ns in batch}

            with _ALL_DATA_LOCK:
                for ticker_ns in batch:
                    frame = batch_frames.get(ticker_ns)
                    if frame is not None:
                        ALL_DATA[ticker_ns] = frame
                        continue

                    fallback = stale_fallbacks.get(ticker_ns)
                    if fallback is None:
                        continue

                    existing_frame = ALL_DATA.get(ticker_ns)
                    if existing_frame is None or _is_stale_live_frame(existing_frame):
                        ALL_DATA[ticker_ns] = fallback
            for ticker_ns in batch:
                frame = batch_frames.get(ticker_ns)
                if frame is None:
                    _mark_no_data(ticker_ns)
                else:
                    _clear_no_data(ticker_ns)
                    _persist_frame_to_csv(ticker_ns, frame)

            for ticker_ns in batch:
                done += 1
                frame = batch_frames.get(ticker_ns)
                if frame is None:
                    frame = stale_fallbacks.get(ticker_ns)
                if frame is not None:
                    loaded += 1
                _emit_progress()


def preload_history_batch(
    tickers: list[str],
    period: str = "6mo",
    workers: int = 12,
    progress_callback=None,
) -> None:
    """Back-compat alias for preload_all()."""
    preload_all(
        tickers,
        period=period,
        workers=workers,
        progress_callback=progress_callback,
    )


def get_df_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Return preloaded DF for a ticker, with fallback to live download.
    Caches the fallback result in ALL_DATA to prevent repeated API calls.
    """
    ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    with _ALL_DATA_LOCK:
        df = ALL_DATA.get(ticker_ns)
    stale_fallback = None
    if df is not None:
        if not _is_stale_live_frame(df):
            return df
        stale_fallback = df

    try:
        from data_downloader import load_csv
        csv_df = _prepare_loaded_frame(load_csv(ticker_ns))
        if csv_df is not None:
            if _is_stale_live_frame(csv_df):
                if stale_fallback is None:
                    stale_fallback = csv_df
            else:
                _clear_no_data(ticker_ns)
                with _ALL_DATA_LOCK:
                    ALL_DATA[ticker_ns] = csv_df
                return csv_df
    except Exception:
        pass

    fetched = download_history(ticker_ns, period="6mo")
    if fetched is not None:
        with _ALL_DATA_LOCK:
            ALL_DATA[ticker_ns] = fetched
        _persist_frame_to_csv(ticker_ns, fetched)
        return fetched
    return stale_fallback


def add_rank_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Top-Ranked High-Probability scoring columns (trend/momentum/volume/near-high/rsi).
    Fail-safe: on any unexpected error, returns the original df unchanged.
    """
    try:
        if df is None or df.empty:
            return df

        out = df.copy()

        # Requested columns (new-add-on layer)
        out["trend_score"] = 0.0
        out["momentum_score"] = 0.0
        out["volume_score"] = 0.0
        out["near_high_score"] = 0.0
        out["rsi_score"] = 0.0
        out["rank_score"] = 0.0

        # Cache per symbol to avoid duplicate downloads.
        _df_cache: dict[str, pd.DataFrame | None] = {}

        def _get_df_cached(sym: str) -> pd.DataFrame | None:
            if sym in _df_cache:
                return _df_cache[sym]
            _df_cache[sym] = get_df_for_ticker(sym)
            return _df_cache[sym]

        for i, row in out.iterrows():
            sym = row.get("Symbol", None) or row.get("Ticker", None) or ""
            sym = str(sym).strip()
            if not sym:
                continue

            price = safe(row.get("Price (₹)", 0.0), 0.0)
            rsi_v = safe(row.get("RSI", 50.0), 50.0)
            vol_r = safe(row.get("Vol / Avg", 1.0), 1.0)
            d_ema20 = safe(row.get("Δ vs EMA20 (%)", 0.0), 0.0)
            r5d = safe(row.get("5D Return (%)", 0.0), 0.0)
            d20h = safe(row.get("Δ vs 20D High (%)", -5.0), -5.0)

            # Momentum (5D) score: based on 5D return (%)
            momentum_score = float(np.clip(50.0 + r5d * 3.0, 0.0, 100.0))

            # Volume score: normalized volume ratio (cap at 3.5x)
            vol_clip = float(np.clip(vol_r, 0.0, 3.5))
            volume_score = float((vol_clip / 3.5) * 100.0) if 3.5 > 0 else 0.0

            # RSI score: peak near ~60, penalize distance.
            rsi_score = float(np.clip(100.0 - abs(rsi_v - 60.0) * 4.0, 0.0, 100.0))

            trend_score = np.nan
            near_high_score = np.nan

            df_h = None
            try:
                df_h = _get_df_cached(sym)
            except Exception:
                df_h = None

            # Compute 60d trend + rolling-high proximity when OHLCV is available.
            try:
                if df_h is not None and isinstance(df_h, pd.DataFrame) and len(df_h) >= 30:
                    close_s = df_h["Close"].dropna() if "Close" in df_h.columns else None
                    high_s = df_h["High"].dropna() if "High" in df_h.columns else None

                    # trend_score (60d): fraction of last 60 days trading above EMA20 + 60d return component
                    if close_s is not None and len(close_s) >= 60:
                        tail = close_s.tail(60)
                        e20s = ema(close_s, 20)
                        e20_tail = e20s.reindex(tail.index)
                        above_ratio = float((tail > e20_tail).mean()) if len(tail) > 0 else 0.0
                        close60 = safe(close_s.iloc[-60], 0.0)
                        ret60 = (float(close_s.iloc[-1]) / close60 - 1.0) * 100.0 if close60 > 0 else 0.0
                        ret_comp = float(np.clip(50.0 + ret60 * 2.0, 0.0, 100.0))
                        trend_score = float(np.clip(0.6 * (above_ratio * 100.0) + 0.4 * ret_comp, 0.0, 100.0))

                    # near_high_score: proximity to rolling 20d high using High series
                    if high_s is not None and len(high_s) >= 20 and price > 0:
                        roll_high = safe(high_s.tail(20).max(), 0.0)
                        if roll_high > 0:
                            near_ratio = float(price) / float(roll_high)
                            near_high_score = float(np.clip((near_ratio - 0.95) / 0.10 * 100.0, 0.0, 100.0))
            except Exception:
                pass

            # Fail-safe fallbacks (use already-computed row fields)
            try:
                if not np.isfinite(trend_score):
                    trend_score = float(np.clip(50.0 + d_ema20 * 2.5, 0.0, 100.0))
            except Exception:
                trend_score = float(0.0)

            try:
                if not np.isfinite(near_high_score):
                    near_high_score = float(np.clip(50.0 + d20h * 4.0, 0.0, 100.0))
            except Exception:
                near_high_score = float(0.0)

            rank_score = float(
                np.clip(
                    0.25 * float(trend_score)
                    + 0.25 * float(momentum_score)
                    + 0.20 * float(volume_score)
                    + 0.15 * float(near_high_score)
                    + 0.15 * float(rsi_score),
                    0.0,
                    100.0,
                )
            )

            out.at[i, "trend_score"] = round(float(trend_score), 2)
            out.at[i, "momentum_score"] = round(float(momentum_score), 2)
            out.at[i, "volume_score"] = round(float(volume_score), 2)
            out.at[i, "near_high_score"] = round(float(near_high_score), 2)
            out.at[i, "rsi_score"] = round(float(rsi_score), 2)
            out.at[i, "rank_score"] = round(float(rank_score), 2)

        return out
    except Exception:
        pass

    return df


def _series_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    if isinstance(default, pd.Series):
        return pd.to_numeric(default.reindex(df.index), errors="coerce").fillna(0.0)
    return pd.Series(default, index=df.index, dtype="float64")


def _series_text(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype="object")


def _has_flag(text: object) -> bool:
    raw = str(text or "").strip().lower()
    if raw in {"", "-", "nan", "none", "null", "0", "false", "no"}:
        return False
    if "no trap" in raw or "none" == raw:
        return False
    return True


def _pick_reason_main(row: pd.Series) -> str:
    return (
        f"Final {safe(row.get('Final Score', 0.0)):.1f} | "
        f"Pred {safe(row.get('Prediction Score', row.get('ML %', 0.0))):.1f} | "
        f"{str(row.get('Next-Day Signal', row.get('Adjusted Signal', '-'))) or '-'}"
    )


def _pick_reason_csv(row: pd.Series) -> str:
    grade = str(row.get("Grade", "-") or "-")
    return (
        f"Prob {safe(row.get('Next Day Prob', 0.0)):.1f}% | "
        f"Conf {safe(row.get('Confidence', 0.0)):.1f}% | "
        f"Grade {grade} | "
        f"Setup {safe(row.get('Setup Quality', 0.0)):.1f}"
    )


def _pick_reason_breakout(row: pd.Series) -> str:
    return (
        f"Score {safe(row.get('Final Score', 0.0)):.1f} | "
        f"{str(row.get('Signal', '-')) or '-'} | "
        f"Risk {safe(row.get('Risk Score', 0.0)):.0f}"
    )


def get_tomorrow_top_picks(
    df: pd.DataFrame | None,
    source: str = "main",
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Return the top buyable setups for tomorrow from a result DataFrame.

    Supported source values:
      - "main": regular mode scan output from app.py
      - "csv": csv_next_day_engine output
      - "breakout": breakout_radar_engine output
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    src = str(source or "main").strip().lower()
    top_n = max(1, int(top_n))

    if src == "csv":
        prob = _series_num(out, "Next Day Prob")
        conf = _series_num(out, "Confidence")
        setup = _series_num(out, "Setup Quality")
        trigger = _series_num(out, "Trigger Quality")
        hist = _series_num(out, "Historical Win %")
        downside = _series_num(out, "Downside Risk %")
        grade = _series_text(out, "Grade").str.upper().str.strip()
        buy = _series_text(out, "Buy Readiness").str.upper()
        signal = _series_text(out, "Signal").str.lower()
        trap = _series_text(out, "Bull Trap")

        grade_bonus = np.select(
            [grade.eq("A"), grade.eq("B"), grade.eq("C"), grade.eq("D")],
            [8.0, 5.0, 2.0, -4.0],
            default=0.0,
        )
        signal_bonus = np.select(
            [
                signal.str.contains("strong", na=False),
                signal.str.contains("buy", na=False),
                signal.str.contains("watch", na=False),
            ],
            [5.0, 3.0, 1.0],
            default=0.0,
        )
        trap_penalty = trap.apply(lambda x: 12.0 if _has_flag(x) else 0.0).astype(float)
        out["Tomorrow Pick Score"] = (
            0.38 * prob
            + 0.20 * conf
            + 1.80 * setup
            + 1.40 * trigger
            + 0.12 * hist
            + grade_bonus
            + signal_bonus
            - 0.45 * downside
            - trap_penalty
        ).clip(0.0, 100.0).round(1)

        strict_mask = buy.str.contains("BUY READY", na=False) & trap_penalty.eq(0.0)
        soft_mask = prob.ge(50.0) & trap_penalty.le(0.0)
        sort_cols = ["Tomorrow Pick Score", "Next Day Prob", "Confidence", "Setup Quality"]
        reason_fn = _pick_reason_csv

    elif src == "breakout":
        final = _series_num(out, "Final Score")
        risk = _series_num(out, "Risk Score")
        compression = _series_num(out, "Compression Score")
        trend = _series_num(out, "Trend Score")
        volume = _series_num(out, "Volume Score")
        signal = _series_text(out, "Signal").str.upper().str.strip()
        trap_flags = _series_text(out, "Trap Flags")

        signal_bonus = np.select(
            [
                signal.eq("HIGH PROBABILITY BREAKOUT"),
                signal.eq("STRONG SETUP"),
                signal.eq("WATCHLIST"),
                signal.eq("AVOID"),
                signal.eq("TRAP"),
            ],
            [14.0, 9.0, 2.0, -8.0, -18.0],
            default=0.0,
        )
        trap_count = trap_flags.apply(
            lambda x: 0
            if not _has_flag(x)
            else len([p for p in str(x).replace("/", ",").replace("|", ",").split(",") if p.strip()])
        ).astype(float)
        trap_penalty = np.clip(trap_count * 5.0, 0.0, 15.0)
        out["Tomorrow Pick Score"] = (
            0.62 * final
            + 0.25 * compression
            + 0.20 * trend
            + 0.15 * volume
            + signal_bonus
            - 0.40 * risk
            - trap_penalty
        ).clip(0.0, 100.0).round(1)

        strict_mask = signal.isin(["HIGH PROBABILITY BREAKOUT", "STRONG SETUP"]) & risk.le(45.0) & trap_count.le(1.0)
        soft_mask = signal.isin(["HIGH PROBABILITY BREAKOUT", "STRONG SETUP", "WATCHLIST"]) & risk.le(60.0) & trap_count.le(1.0)
        sort_cols = ["Tomorrow Pick Score", "Final Score", "Risk Score"]
        reason_fn = _pick_reason_breakout

    else:
        final = _series_num(out, "Final Score", 0.0)
        pred = _series_num(out, "Prediction Score", _series_num(out, "ML %", 0.0))
        backtest = _series_num(out, "Backtest %")
        ml = _series_num(out, "ML %")
        next_day = _series_text(out, "Next-Day Signal")
        adjusted = _series_text(out, "Adjusted Signal")
        conviction = _series_text(out, "Conviction Tier")
        trap = _series_text(out, "Trap")
        signal_text = (
            adjusted.fillna("") + " " + next_day.fillna("") + " " + conviction.fillna("")
        ).str.lower()

        signal_bonus = np.select(
            [
                signal_text.str.contains("strong green|strong buy|high probability", regex=True, na=False),
                signal_text.str.contains("possible up|buy ready|buyable|green", regex=True, na=False),
                signal_text.str.contains("risky|late entry|watch", regex=True, na=False),
                signal_text.str.contains("weak|avoid|trap|sell", regex=True, na=False),
            ],
            [12.0, 7.0, -3.0, -15.0],
            default=2.0,
        )
        conviction_bonus = np.select(
            [
                conviction.str.contains(r"A\+", case=False, regex=True, na=False),
                conviction.str.contains(r"\bA\b", case=False, regex=True, na=False),
                conviction.str.contains(r"\bB\b", case=False, regex=True, na=False),
                conviction.str.contains(r"\bD\b", case=False, regex=True, na=False),
            ],
            [8.0, 6.0, 3.0, -5.0],
            default=0.0,
        )
        trap_penalty = trap.apply(lambda x: 12.0 if _has_flag(x) else 0.0).astype(float)
        out["Tomorrow Pick Score"] = (
            0.42 * final
            + 0.24 * pred
            + 0.17 * backtest
            + 0.17 * ml
            + signal_bonus
            + conviction_bonus
            - trap_penalty
        ).clip(0.0, 100.0).round(1)

        strict_mask = ~signal_text.str.contains("weak|avoid|trap|sell|late entry|risky", regex=True, na=False) & trap_penalty.eq(0.0)
        soft_mask = ~signal_text.str.contains("weak|avoid|trap|sell", regex=True, na=False)
        sort_cols = ["Tomorrow Pick Score", "Final Score", "Prediction Score", "ML %"]
        reason_fn = _pick_reason_main

    if strict_mask.sum() >= top_n:
        picks = out.loc[strict_mask].copy()
    elif soft_mask.sum() >= top_n:
        picks = out.loc[soft_mask].copy()
    else:
        picks = out.copy()

    sort_present = [col for col in sort_cols if col in picks.columns]
    ascending = [False] * len(sort_present)
    if src == "breakout" and "Risk Score" in sort_present:
        ascending[sort_present.index("Risk Score")] = True
    if sort_present:
        picks = picks.sort_values(sort_present, ascending=ascending, kind="stable")
    picks = picks.head(top_n).reset_index(drop=True)
    picks["Tomorrow Pick Reason"] = picks.apply(reason_fn, axis=1)
    return picks
