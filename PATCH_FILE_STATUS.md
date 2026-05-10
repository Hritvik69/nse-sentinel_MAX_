# Patch File Status

These legacy `*_patch.py` files are not imported by the production Streamlit app or strategy engine modules as of this hardening pass. They are retained for operator reference in `archive/legacy_patches/` with `.py.txt` suffixes so they cannot be imported accidentally.

Verified with repository search for production imports:

- `archive/legacy_patches/ANIMATIONS_PATCH.py.txt`: no production import references.
- `archive/legacy_patches/app_analyse_patch.py.txt`: self-documenting standalone scan patch; no production import references.
- `archive/legacy_patches/app_sector_intelligence_patch.py.txt`: no production import references.
- `archive/legacy_patches/scan_speed_patch.py.txt`: self-documenting monkey-patch experiment; no production import references.

Production code should prefer the normal modules (`app.py`, `strategy_engines/*`, `nse_animations.py`) so imports remain unambiguous.
