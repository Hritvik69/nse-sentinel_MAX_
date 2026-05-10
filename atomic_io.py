from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Iterator

import pandas as pd

_LOCKS_GUARD = threading.RLock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _lock_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).lower()


def get_path_lock(path: Path | str) -> threading.RLock:
    key = _lock_key(path)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


@contextmanager
def locked_path(path: Path | str) -> Iterator[None]:
    lock = get_path_lock(path)
    with lock:
        yield


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked_path(target):
        fd = -1
        tmp_name = ""
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=str(target.parent),
            )
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except Exception:
                    pass


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, str(text).encode(encoding))


def atomic_write_json(path: Path | str, payload: object, *, indent: int | None = 2) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=indent)
    atomic_write_text(path, text, encoding="utf-8")


def atomic_write_csv_df(path: Path | str, df: pd.DataFrame, **to_csv_kwargs) -> None:
    buf = StringIO()
    df.to_csv(buf, **to_csv_kwargs)
    atomic_write_text(path, buf.getvalue(), encoding="utf-8")
