# Patch File Status

These legacy `*_patch.py` files are not imported by the production Streamlit app or strategy engine modules as of this hardening pass. They are retained in place for operator reference instead of being moved or deleted.

Verified with repository search for production imports:

- `ANIMATIONS_PATCH.py`: no production import references.
- `app_analyse_patch.py`: self-documenting standalone scan patch; no production import references.
- `app_sector_intelligence_patch.py`: no production import references.
- `scan_speed_patch.py`: self-documenting monkey-patch experiment; no production import references.

Production code should prefer the normal modules (`app.py`, `strategy_engines/*`, `nse_animations.py`) so imports remain unambiguous.
