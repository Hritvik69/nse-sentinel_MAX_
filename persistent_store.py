"""
persistent_store.py
GitHub-backed persistence layer for Streamlit Cloud.

Managed files in data/ are mirrored to a GitHub repository. The app keeps
reading and writing local files as before; this module restores them on startup
and mirrors local writes in the background.

SETUP (add to .streamlit/secrets.toml or Streamlit Cloud secrets):
    [github_store]
    token  = "ghp_xxxxxxxxxxxxxxxxxxxx"
    owner  = "your-github-username"
    repo   = "your-repo-name"
    branch = "main"

If secrets are missing, the module silently falls back to local-only mode.
"""

from __future__ import annotations

import base64
from io import BytesIO, StringIO
import json
import logging
import queue as _queue
import threading
import time as _time
from pathlib import Path
from typing import Any

import pandas as pd
from atomic_io import atomic_write_bytes

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_LOG = logging.getLogger(__name__)


def _has_real_secret(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    upper = text.upper()
    if "REPLACE" in upper or upper.startswith("YOUR-") or upper.startswith("YOUR_"):
        return False
    return True


def _get_secrets() -> dict | None:
    try:
        import streamlit as st

        cfg = st.secrets.get("github_store", {})
        token = cfg.get("token", "")
        owner = cfg.get("owner", "")
        repo = cfg.get("repo", "")
        branch = cfg.get("branch", "main")
        if not (_has_real_secret(token) and _has_real_secret(owner) and _has_real_secret(repo)):
            return None
        return {"token": token, "owner": owner, "repo": repo, "branch": branch}
    except Exception:
        return None


def is_configured() -> bool:
    """Return True when GitHub-backed persistence has real secrets configured."""
    return _get_secrets() is not None


def _gh_get(secrets: dict, path: str) -> dict | None:
    """GET /repos/{owner}/{repo}/contents/{path}?ref={branch}."""
    try:
        import requests

        url = (
            f"https://api.github.com/repos/{secrets['owner']}/{secrets['repo']}"
            f"/contents/{path}?ref={secrets['branch']}"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {secrets['token']}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _gh_get_ref(secrets: dict) -> dict | None:
    """GET /repos/{owner}/{repo}/git/ref/heads/{branch}."""
    try:
        import requests

        url = (
            f"https://api.github.com/repos/{secrets['owner']}/{secrets['repo']}"
            f"/git/ref/heads/{secrets['branch']}"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {secrets['token']}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"GitHub returned HTTP {resp.status_code}"}
    except Exception as exc:
        return {"error": str(exc)}


def health_check() -> dict:
    """
    Validate whether GitHub-backed persistence can actually read the target repo.

    This is intentionally separate from pull_all(): first deploys may have no
    stored data files yet, but the token/repo/branch should still validate.
    """
    secrets = _get_secrets()
    if secrets is None:
        return {
            "configured": False,
            "connected": False,
            "message": "Missing [github_store] secrets.",
        }
    ref = _gh_get_ref(secrets)
    if isinstance(ref, dict) and not ref.get("error"):
        return {
            "configured": True,
            "connected": True,
            "message": f"Connected to {secrets['owner']}/{secrets['repo']}:{secrets['branch']}.",
        }
    return {
        "configured": True,
        "connected": False,
        "message": str(ref.get("error") if isinstance(ref, dict) else "Unable to validate GitHub storage."),
    }


def _gh_get_raw(secrets: dict, path: str) -> bytes | None:
    """Download raw file bytes through the contents endpoint."""
    try:
        import requests

        url = (
            f"https://api.github.com/repos/{secrets['owner']}/{secrets['repo']}"
            f"/contents/{path}?ref={secrets['branch']}"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"token {secrets['token']}",
                "Accept": "application/vnd.github.raw",
            },
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception:
        return None


def _gh_put_status(secrets: dict, path: str, content_bytes: bytes, sha: str | None = None) -> dict:
    """PUT /repos/{owner}/{repo}/contents/{path}."""
    try:
        import requests

        url = (
            f"https://api.github.com/repos/{secrets['owner']}/{secrets['repo']}"
            f"/contents/{path}"
        )
        payload: dict = {
            "message": f"auto: update {path}",
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": secrets["branch"],
        }
        if sha:
            payload["sha"] = sha
        headers = {
            "Authorization": f"token {secrets['token']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }
        resp = requests.put(url, headers=headers, data=json.dumps(payload), timeout=20)
        if resp.status_code in (200, 201):
            return {"ok": True, "status": resp.status_code}
        return {
            "ok": False,
            "status": resp.status_code,
            "retry_after": resp.headers.get("Retry-After", ""),
        }
    except Exception as exc:
        return {"ok": False, "status": 0, "error": str(exc)[:160]}


def _gh_put(secrets: dict, path: str, content_bytes: bytes, sha: str | None = None) -> bool:
    return bool(_gh_put_status(secrets, path, content_bytes, sha).get("ok"))


_CSV_KEY_COLUMNS = {
    "prediction_feedback_log.csv": (
        "prediction_id",
        ("symbol", "mode", "market_date", "import_source", "prediction_direction"),
    ),
    "tomorrow_master_predictions.csv": (
        "",
        ("ticker", "direction", "computed_at"),
    ),
    "sector_prediction_log.csv": (
        "prediction_id",
        ("sector", "market_date", "ohlc_source", "prediction_direction"),
    ),
    "sector_predictions.csv": (
        "prediction_id",
        ("sector", "market_date", "ohlc_source", "prediction_direction"),
    ),
    "sector_signal_performance.csv": (
        "",
        ("signal_name",),
    ),
}

_VALIDATION_COLUMNS = {
    "actual_next_return_pct",
    "actual_return",
    "correct",
    "outcome_label",
    "validated_at",
    "validated_on",
    "validation_date",
    "validation_status",
    "final_status",
    "exit_price",
    "return_pct",
    "target_policy_version",
}


def _is_blank(value: object) -> bool:
    try:
        if value is None:
            return True
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() in ("", "nan", "None", "NaT")


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)


def _normalise_key_value(value: object) -> str:
    text = "" if _is_blank(value) else str(value).strip()
    return text.upper() if text.endswith(".NS") else text


def _logical_csv_key(row: pd.Series | dict, id_column: str, tuple_columns: tuple[str, ...]) -> tuple[str, ...]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    if id_column:
        pid = getter(id_column, "")
        if not _is_blank(pid):
            return ("id", str(pid).strip())
    values: list[str] = []
    for col in tuple_columns:
        value = getter(col, "")
        if col == "market_date" and _is_blank(value):
            try:
                logged_at = getter("logged_at", getter("predicted_at", ""))
                parsed = pd.to_datetime(logged_at, errors="coerce")
                if not pd.isna(parsed):
                    value = parsed.date().isoformat()
            except Exception:
                value = ""
        values.append(_normalise_key_value(value))
    return tuple(["logical"] + values)


def _row_nonempty_count(row: pd.Series | dict) -> int:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    try:
        keys = list(row.keys())  # type: ignore[union-attr]
    except Exception:
        keys = []
    return sum(0 if _is_blank(getter(key, "")) else 1 for key in keys)


def _merge_row_values(existing: dict[str, object], incoming: dict[str, object]) -> dict[str, object]:
    merged = dict(existing)
    for col in set(existing) | set(incoming):
        old = merged.get(col, "")
        new = incoming.get(col, "")
        if col in _VALIDATION_COLUMNS:
            if _is_blank(old) and not _is_blank(new):
                merged[col] = new
            elif not _is_blank(old):
                merged[col] = old
            else:
                merged[col] = new
            continue
        if not _is_blank(new):
            merged[col] = new
        elif col not in merged:
            merged[col] = ""
    return merged


def _merge_signal_performance(remote: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    columns = list(dict.fromkeys(list(remote.columns) + list(local.columns)))
    rows: dict[str, dict[str, object]] = {}

    def _float(row: dict[str, object], key: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(key, default) or default)
            return value if pd.notna(value) else default
        except Exception:
            return default

    for _, row in pd.concat([remote, local], ignore_index=True, sort=False).iterrows():
        item = {col: row.get(col, "") for col in columns}
        signal = str(item.get("signal_name", "") or "").strip()
        if not signal:
            continue
        current = rows.get(signal)
        if current is None:
            rows[signal] = item
            continue
        old_obs = _float(current, "observations")
        new_obs = _float(item, "observations")
        if new_obs > old_obs or (new_obs == old_obs and _row_nonempty_count(item) >= _row_nonempty_count(current)):
            merged = _merge_row_values(current, item)
        else:
            merged = _merge_row_values(item, current)
        obs = max(_float(current, "observations"), _float(item, "observations"))
        wins = max(_float(current, "wins"), _float(item, "wins"))
        merged["observations"] = str(int(obs)) if obs.is_integer() else str(obs)
        merged["wins"] = str(int(wins)) if wins.is_integer() else str(wins)
        if obs > 0:
            merged["win_rate"] = f"{wins / obs:.4f}"
        rows[signal] = merged

    return pd.DataFrame(list(rows.values()), columns=columns)


def _merge_keyed_csv(remote: pd.DataFrame, local: pd.DataFrame, *, id_column: str, tuple_columns: tuple[str, ...]) -> pd.DataFrame:
    columns = list(dict.fromkeys(list(remote.columns) + list(local.columns)))
    by_key: dict[tuple[str, ...], dict[str, object]] = {}
    order: list[tuple[str, ...]] = []

    for _, row in pd.concat([remote, local], ignore_index=True, sort=False).iterrows():
        item = {col: row.get(col, "") for col in columns}
        key = _logical_csv_key(item, id_column, tuple_columns)
        if key not in by_key:
            by_key[key] = item
            order.append(key)
            continue
        by_key[key] = _merge_row_values(by_key[key], item)

    return pd.DataFrame([by_key[key] for key in order], columns=columns)


def _merge_generic_csv(remote: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([remote, local], ignore_index=True, sort=False)
    return merged.drop_duplicates(keep="last")


def _can_key_csv(remote: pd.DataFrame, local: pd.DataFrame, *, id_column: str, tuple_columns: tuple[str, ...]) -> bool:
    columns = set(remote.columns) | set(local.columns)
    if tuple_columns and all(col in columns for col in tuple_columns):
        return True
    if id_column and id_column in columns:
        combined = pd.concat([remote, local], ignore_index=True, sort=False)
        return not combined[id_column].map(_is_blank).any()
    return False


def _merge_csv_bytes(remote_path: str, remote_bytes: bytes, local_bytes: bytes) -> bytes | None:
    try:
        remote = _read_csv_bytes(remote_bytes)
        local = _read_csv_bytes(local_bytes)
        file_name = Path(remote_path).name
        key_spec = _CSV_KEY_COLUMNS.get(file_name)
        if file_name == "sector_signal_performance.csv":
            merged = _merge_signal_performance(remote, local)
        elif key_spec is not None:
            id_column, tuple_columns = key_spec
            if _can_key_csv(remote, local, id_column=id_column, tuple_columns=tuple_columns):
                merged = _merge_keyed_csv(remote, local, id_column=id_column, tuple_columns=tuple_columns)
            else:
                merged = _merge_generic_csv(remote, local)
        else:
            merged = _merge_generic_csv(remote, local)
        buf = StringIO()
        merged.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
    except Exception:
        return None


def _merge_list_items(remote: list[Any], local: list[Any]) -> list[Any]:
    if all(isinstance(item, dict) and "ticker" in item for item in remote + local):
        merged_by_ticker: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in remote + local:
            key = str(item.get("ticker", "") or "").strip().upper()
            if not key:
                key = json.dumps(item, sort_keys=True, ensure_ascii=True)
            if key not in merged_by_ticker:
                merged_by_ticker[key] = dict(item)
                order.append(key)
            else:
                merged_by_ticker[key] = _merge_json_values(merged_by_ticker[key], item)
        return [merged_by_ticker[key] for key in order]

    seen: set[str] = set()
    merged: list[Any] = []
    for item in remote + local:
        key = json.dumps(item, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _merge_json_values(remote: Any, local: Any) -> Any:
    if isinstance(remote, dict) and isinstance(local, dict):
        out = dict(remote)
        for key, local_value in local.items():
            if key in out:
                out[key] = _merge_json_values(out[key], local_value)
            elif not _is_blank(local_value):
                out[key] = local_value
            else:
                out[key] = local_value
        return out
    if isinstance(remote, list) and isinstance(local, list):
        return _merge_list_items(remote, local)
    if _is_blank(local) and not _is_blank(remote):
        return remote
    return local


def _merge_json_bytes(remote_bytes: bytes, local_bytes: bytes) -> bytes | None:
    try:
        remote = json.loads(remote_bytes.decode("utf-8"))
        local = json.loads(local_bytes.decode("utf-8"))
        merged = _merge_json_values(remote, local)
        return json.dumps(merged, ensure_ascii=True, indent=2).encode("utf-8")
    except Exception:
        return None


def _merge_remote_local(remote_path: str, remote_bytes: bytes | None, local_bytes: bytes) -> bytes | None:
    if not remote_bytes:
        return None
    suffix = Path(remote_path).suffix.lower()
    if suffix == ".csv":
        return _merge_csv_bytes(remote_path, remote_bytes, local_bytes)
    if suffix == ".json":
        return _merge_json_bytes(remote_bytes, local_bytes)
    return None


# Local filename (inside data/) -> remote GitHub path.
#
# This codebase writes data/sector_predictions.csv locally. We mirror it to the
# requested remote name data/sector_prediction_log.csv without changing local
# read/write behavior.
_SYNC_FILES = {
    "prediction_feedback_log.csv": "data/prediction_feedback_log.csv",
    "tomorrow_master_predictions.csv": "data/tomorrow_master_predictions.csv",
    "tomorrow_picks_store.json": "data/tomorrow_picks_store.json",
    "imported_ai_learning_store.json": "data/imported_ai_learning_store.json",
    "learning_status_snapshot.json": "data/learning_status_snapshot.json",
    "sector_predictions.csv": "data/sector_prediction_log.csv",
    "sector_signal_performance.csv": "data/sector_signal_performance.csv",
    "learning_model.pkl": "data/learning_model.pkl",
    "market_snapshot_latest.zip": "data/market_snapshot_latest.zip",
}

_SYNC_ALIASES = {
    "sector_prediction_log.csv": "data/sector_prediction_log.csv",
}

_PUSH_LOCK = threading.Lock()
_PUSH_QUEUE: "_queue.Queue[Path | None]" = _queue.Queue(maxsize=20)
_PENDING_PUSHES: dict[str, Path] = {}
_PENDING_PUSHES_LOCK = threading.Lock()
_PUSH_WORKER_STARTED = False
_PUSH_WORKER_LOCK = threading.Lock()
_TRANSIENT_PUSH_STATUSES = {429, 503}
_PUSH_STATUS_LOCK = threading.Lock()
_LAST_PUSH_ERROR: dict[str, Any] = {}


def _set_last_push_error(path: str, status: object = "", error: object = "") -> None:
    with _PUSH_STATUS_LOCK:
        _LAST_PUSH_ERROR.clear()
        _LAST_PUSH_ERROR.update(
            {
                "path": str(path or ""),
                "status": status,
                "error": str(error or "")[:240],
                "ts": _time.time(),
            }
        )


def _clear_last_push_error(path: str | None = None) -> None:
    with _PUSH_STATUS_LOCK:
        if path is None or _LAST_PUSH_ERROR.get("path") == path:
            _LAST_PUSH_ERROR.clear()


def get_last_push_error() -> dict[str, Any]:
    with _PUSH_STATUS_LOCK:
        return dict(_LAST_PUSH_ERROR)


def _ensure_data_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _push_key(local_path: Path) -> str:
    try:
        return str(local_path.resolve(strict=False)).lower()
    except Exception:
        return str(local_path).lower()


def _take_pending_pushes() -> list[Path]:
    with _PENDING_PUSHES_LOCK:
        pending = list(_PENDING_PUSHES.values())
        _PENDING_PUSHES.clear()
        return pending


def _put_with_transient_retry(secrets: dict, remote_path: str, content: bytes, sha: str | None) -> dict:
    result: dict = {"ok": False, "status": 0}
    retry_after_delay: float | None = None
    for attempt, delay in enumerate((0.0, 0.25, 0.75), start=1):
        if retry_after_delay is not None:
            delay = retry_after_delay
            retry_after_delay = None
        if delay:
            _time.sleep(delay)
        result = _gh_put_status(secrets, remote_path, content, sha)
        status = int(result.get("status") or 0)
        if result.get("ok") or status not in _TRANSIENT_PUSH_STATUSES or attempt >= 3:
            return result
        try:
            retry_after = float(result.get("retry_after") or 0)
            if retry_after > 0:
                retry_after_delay = min(retry_after, 30.0)
        except Exception:
            retry_after_delay = None
    return result


def _rebuild_signal_performance_from_sector_log() -> bool:
    """Regenerate derived signal performance from the merged sector log."""
    sector_log = _DATA_DIR / "sector_predictions.csv"
    perf_path = _DATA_DIR / "sector_signal_performance.csv"
    if not sector_log.exists():
        return False
    try:
        from sector_dynamic_weights import update_signal_performance

        log_df = pd.read_csv(sector_log, dtype=str, keep_default_na=False)
        update_signal_performance(log_df)
        return perf_path.exists()
    except Exception as exc:
        _LOG.error("persistent_store: signal performance rebuild failed: %s", str(exc)[:160])
        return False


def _do_push_blocking(local_path: Path) -> bool:
    """Push one file synchronously."""
    try:
        secrets = _get_secrets()
        if secrets is None:
            return True
        local_name = local_path.name
        remote_path = _SYNC_FILES.get(local_name) or _SYNC_ALIASES.get(local_name)
        if remote_path is None:
            _LOG.debug("persistent_store: %r not in sync manifest -- skipped", local_name)
            return True
        if not local_path.exists():
            _LOG.debug("persistent_store: %s does not exist -- skipped", local_path)
            return True
        if local_name == "sector_signal_performance.csv":
            _rebuild_signal_performance_from_sector_log()
        with _PUSH_LOCK:
            if not local_path.exists():
                _LOG.debug("persistent_store: %s disappeared before push -- skipped", local_path)
                return True
            existing = _gh_get(secrets, remote_path)
            sha = existing["sha"] if isinstance(existing, dict) else None
            if not local_path.exists():
                _LOG.debug("persistent_store: %s disappeared before push -- skipped", local_path)
                return True
            content = local_path.read_bytes()
            result = _put_with_transient_retry(secrets, remote_path, content, sha)
            if result.get("ok"):
                _clear_last_push_error(remote_path)
                return True
            if int(result.get("status") or 0) == 409:
                latest = _gh_get(secrets, remote_path)
                latest_sha = latest["sha"] if isinstance(latest, dict) else None
                remote_raw = _gh_get_raw(secrets, remote_path)
                merged = _merge_remote_local(remote_path, remote_raw, content)
                if merged is None:
                    _LOG.error("persistent_store: GitHub conflict for %s could not be merged", remote_path)
                    return False
                retry = _put_with_transient_retry(secrets, remote_path, merged, latest_sha)
                if retry.get("ok"):
                    _clear_last_push_error(remote_path)
                    return True
            _set_last_push_error(remote_path, result.get("status"), result.get("error", "push failed"))
            _LOG.error("persistent_store: GitHub push failed for %s status=%s", remote_path, result.get("status"))
            return False
    except Exception as exc:
        _set_last_push_error(str(local_path), 0, exc)
        _LOG.error("persistent_store: push failed for %s: %s", local_path.name, str(exc)[:160])
        return False


def _run_push_worker() -> None:
    """Single background worker draining queued GitHub pushes."""
    while True:
        try:
            local_path = _PUSH_QUEUE.get(timeout=5)
        except _queue.Empty:
            continue
        try:
            if local_path is None:
                break
            while True:
                pending = _take_pending_pushes()
                if not pending:
                    break
                for pending_path in pending:
                    _do_push_blocking(pending_path)
        except Exception as exc:
            _set_last_push_error(str(local_path), 0, exc)
            _LOG.exception("persistent_store: push worker failed")
        finally:
            _PUSH_QUEUE.task_done()


def _ensure_push_worker() -> None:
    global _PUSH_WORKER_STARTED
    with _PUSH_WORKER_LOCK:
        if not _PUSH_WORKER_STARTED:
            thread = threading.Thread(target=_run_push_worker, daemon=True)
            thread.name = "persistent_store_push_worker"
            thread.start()
            _PUSH_WORKER_STARTED = True


def pull_all() -> int:
    """
    Download every managed file from GitHub to local data/.

    Call once at app startup before CSV reads. Returns the number of files
    successfully pulled.
    """
    secrets = _get_secrets()
    if secrets is None:
        return 0

    _ensure_data_dir()
    pulled = 0
    sector_log_touched = False
    for local_name, remote_path in _SYNC_FILES.items():
        try:
            data = _gh_get(secrets, remote_path)
            if data is None:
                continue
            content = str(data.get("content", "") or "")
            if content:
                raw = base64.b64decode(content.replace("\n", ""))
            else:
                raw = _gh_get_raw(secrets, remote_path)
                if raw is None:
                    continue
            local_path = _DATA_DIR / local_name
            if local_path.exists() and Path(remote_path).suffix.lower() in {".csv", ".json"}:
                local_bytes = local_path.read_bytes()
                merged = _merge_remote_local(remote_path, raw, local_bytes)
                if merged is None:
                    _LOG.error("persistent_store: pull merge failed for %s; keeping local file", remote_path)
                    continue
                raw = merged
            atomic_write_bytes(local_path, raw)
            if local_name == "sector_predictions.csv":
                sector_log_touched = True
            pulled += 1
        except Exception as exc:
            _LOG.error("persistent_store: pull failed for %s: %s", remote_path, str(exc)[:160])
            continue
    if sector_log_touched:
        _rebuild_signal_performance_from_sector_log()
    return pulled


def push_file(local_path: Path | str, *, block: bool = False) -> bool:
    """
    Push one managed file to GitHub.

    By default this queues work for a single daemon background worker.
    Set block=True for tests or explicit forced-sync scenarios.
    """
    local_path = Path(local_path)

    if block:
        return _do_push_blocking(local_path)

    _ensure_push_worker()
    key = _push_key(local_path)
    with _PENDING_PUSHES_LOCK:
        already_pending = key in _PENDING_PUSHES
        _PENDING_PUSHES[key] = local_path
    if already_pending:
        return True
    try:
        _PUSH_QUEUE.put_nowait(local_path)
    except _queue.Full:
        _LOG.warning("persistent_store: push wake queue full for %s; coalesced latest content", local_path.name)
    return True


def push_all(*, block: bool = False) -> int:
    """Push every existing managed local file."""
    pushed = 0
    for local_name in _SYNC_FILES:
        path = _DATA_DIR / local_name
        if path.exists():
            if push_file(path, block=block):
                pushed += 1
    return pushed
