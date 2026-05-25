"""
model_persistence.py
Save and load trained sklearn models from data/learning_model.pkl.

On save, the pickle is also mirrored through persistent_store so the learning
engine survives Streamlit Cloud restarts.

Loading a pickle requires a trusted SHA-256 from NSE_SENTINEL_MODEL_SHA256 or
Streamlit secrets [model_sha256].learning_model. If that trust check is not
configured, app startup retrains from validated feedback logs instead of
loading an untrusted pickle.
"""

from __future__ import annotations

import hashlib
import logging as _log
import os
import pickle
from pathlib import Path

from atomic_io import atomic_write_bytes, locked_path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_MODEL_PATH = _DATA_DIR / "learning_model.pkl"


def _expected_model_sha256() -> str | None:
    """
    Return a trusted externally configured SHA-256, never one from data/.

    Supported:
      - NSE_SENTINEL_MODEL_SHA256 environment variable
      - Streamlit secrets: [model_sha256] learning_model = "..."
    """
    env_value = os.environ.get("NSE_SENTINEL_MODEL_SHA256", "").strip()
    if env_value:
        return env_value.lower()
    try:
        import streamlit as st

        expected = (st.secrets.get("model_sha256") or {}).get("learning_model")
        if expected:
            return str(expected).strip().lower()
    except Exception:
        pass
    return None


def _check_model_integrity(data: bytes) -> bool:
    expected = _expected_model_sha256()
    if not expected:
        _log.warning(
            "model_persistence: trusted SHA-256 is not configured; skipping persisted pickle load"
        )
        return False
    actual = hashlib.sha256(data).hexdigest().lower()
    if actual != expected:
        _log.error("model_persistence: SHA-256 integrity check failed; model not loaded")
        return False
    return True


def save_model(
    model,
    scaler,
    regime_encoder: dict | None = None,
    sector_encoder: dict | None = None,
    feature_encoders: dict | None = None,
) -> bool:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "scaler": scaler,
            "regime_encoder": regime_encoder or {},
            "sector_encoder": sector_encoder or {},
            "feature_encoders": feature_encoders or {},
        }
        data = pickle.dumps(payload, protocol=4)
        atomic_write_bytes(_MODEL_PATH, data)
        try:
            from persistent_store import push_file as _push_file

            if not _push_file(_MODEL_PATH):
                _log.error("model_persistence: queueing model for persistent sync failed")
        except Exception:
            _log.exception("model_persistence: persistent sync failed")
        return True
    except Exception:
        _log.exception("model_persistence: model save failed")
        return False


def load_model() -> dict | None:
    try:
        if not _MODEL_PATH.exists():
            return None
        with locked_path(_MODEL_PATH):
            data = _MODEL_PATH.read_bytes()
        if not _check_model_integrity(data):
            return None
        payload = pickle.loads(data)
        if not isinstance(payload, dict) or "model" not in payload:
            return None
        return payload
    except Exception:
        return None
