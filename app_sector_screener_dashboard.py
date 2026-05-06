try:
    from strategy_engines.app_sector_screener_dashboard import (
        render_sector_screener_dashboard,
    )
except ImportError as exc:
    _IMPORT_ERR = str(exc)

    def render_sector_screener_dashboard(*args, **kwargs):
        import streamlit as st

        st.error(f"app_sector_screener_dashboard could not be loaded: {_IMPORT_ERR}")
