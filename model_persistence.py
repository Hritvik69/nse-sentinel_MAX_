"""
model_persistence.py
Save and load trained sklearn models from data/learning_model.pkl.

On save, the pickle is also mirrored through persistent_store so the learning
engine survives Streamlit Cloud restarts.
"""

from __future__ import annotations

import hashlib
import logging as _log
import pickle
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_MODEL_PATH = _DATA_DIR / "learning_model.pkl"


def _check_model_integrity(data: bytes) -> bool:
    """
    Return False only when Streamlit secrets pin a model hash and it mismatches.

    Configure model_sha256.learning_model in Streamlit secrets after a trusted
    training run to prevent loading a tampered GitHub-synced pickle.
    """
    try:
        import streamlit as st

        expected = (st.secrets.get("model_sha256") or {}).get("learning_model")
        if not expected:
            return True
        actual = hashlib.sha256(data).hexdigest().lower()
        return actual == str(expected).strip().lower()
    except Exception:
        return True


def save_model(
    model,
    scaler,
    regime_encoder: dict | None = None,
    sector_encoder: dict | None = None,
) -> bool:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "scaler": scaler,
            "regime_encoder": regime_encoder or {},
            "sector_encoder": sector_encoder or {},
        }
        _MODEL_PATH.write_bytes(pickle.dumps(payload, protocol=4))
        try:
            from persistent_store import push_file as _push_file

            _push_file(_MODEL_PATH)
        except Exception:
            pass
        return True
    except Exception:
        return False


def load_model() -> dict | None:
    try:
        if not _MODEL_PATH.exists():
            return None
        data = _MODEL_PATH.read_bytes()
        if not _check_model_integrity(data):
            _log.error("model_persistence: SHA-256 integrity check failed; model not loaded")
            return None
        payload = pickle.loads(data)
        if not isinstance(payload, dict) or "model" not in payload:
            return None
        return payload
    except Exception:
        return None
