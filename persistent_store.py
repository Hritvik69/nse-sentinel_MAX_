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
import json
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"


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


def _gh_put(secrets: dict, path: str, content_bytes: bytes, sha: str | None = None) -> bool:
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
        resp = requests.put(
            url,
            headers={
                "Authorization": f"token {secrets['token']}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=20,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


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
}

_SYNC_ALIASES = {
    "sector_prediction_log.csv": "data/sector_prediction_log.csv",
}

_PUSH_LOCK = threading.Lock()


def _ensure_data_dir() -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


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
            (_DATA_DIR / local_name).write_bytes(raw)
            pulled += 1
        except Exception:
            continue
    return pulled


def push_file(local_path: Path | str, *, block: bool = False) -> bool:
    """
    Push one managed file to GitHub.

    By default this queues a daemon background thread and returns immediately.
    Set block=True for tests or explicit forced-sync scenarios.
    """
    local_path = Path(local_path)

    def _do_push() -> None:
        secrets = _get_secrets()
        if secrets is None:
            return
        local_name = local_path.name
        remote_path = _SYNC_FILES.get(local_name) or _SYNC_ALIASES.get(local_name)
        if remote_path is None:
            import logging
            logging.debug(f"persistent_store: {local_name!r} not in sync manifest -- skipped")
            return
        if not local_path.exists():
            import logging
            logging.debug(f"persistent_store: {local_path} does not exist -- skipped")
            return
        try:
            content = local_path.read_bytes()
            with _PUSH_LOCK:
                existing = _gh_get(secrets, remote_path)
                sha = existing["sha"] if isinstance(existing, dict) else None
                _gh_put(secrets, remote_path, content, sha)
        except Exception:
            pass

    if block:
        _do_push()
        return True

    thread = threading.Thread(target=_do_push, daemon=True)
    thread.start()
    return True


def push_all(*, block: bool = False) -> int:
    """Push every existing managed local file."""
    pushed = 0
    for local_name in _SYNC_FILES:
        path = _DATA_DIR / local_name
        if path.exists():
            push_file(path, block=block)
            pushed += 1
    return pushed
