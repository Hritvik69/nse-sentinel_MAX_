from __future__ import annotations

import hashlib
import re
from pathlib import Path
import unicodedata


_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._^-]+")
_RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(value: object, suffix: str = "", *, max_stem: int = 96) -> str:
    """
    Return a filesystem-safe filename with a collision-resistant suffix when
    sanitization changes the input.

    Normal ticker names such as RELIANCE.NS intentionally remain unchanged.
    """
    raw = str(value or "").strip()
    if not raw:
        raw = "item"

    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    stem = _SAFE_CHARS_RE.sub("_", ascii_text).strip("._ ")
    stem = re.sub(r"_+", "_", stem)
    while ".." in stem:
        stem = stem.replace("..", "_")
    if not stem:
        stem = "item"

    unusual = (
        stem != raw
        or any(ch in raw for ch in ("/", "\\", ":"))
        or len(stem) > max_stem
        or stem.upper() in _RESERVED_WINDOWS_NAMES
        or raw in {".", ".."}
        or raw.startswith(".")
    )

    digest = ""
    if unusual:
        digest = "--" + hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    if stem.upper() in _RESERVED_WINDOWS_NAMES:
        stem = f"_{stem}"

    max_len = max(16, int(max_stem) - len(digest))
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("._ ") or "item"

    suffix_text = str(suffix or "")
    return f"{stem}{digest}{suffix_text}"


def safe_join(root: Path, filename: str) -> Path:
    """Join one filename to root and verify the resolved path stays inside."""
    root_path = Path(root)
    root_resolved = root_path.resolve(strict=False)
    target = (root_path / filename).resolve(strict=False)
    target.relative_to(root_resolved)
    return target


def legacy_colon_slash_filename(value: object, suffix: str = "") -> str:
    """Existing cache naming for backward-compatible reads only."""
    safe = str(value or "").replace(":", "_").replace("/", "_")
    return f"{safe}{suffix}"
