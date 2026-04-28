from __future__ import annotations

import pandas as pd
import streamlit as st

from strategy_engines.nse_autocomplete import (
    configure_nse_stock_search,
    search_nse_stocks,
)

try:
    from sector_master import (
        get_all_sectors,
        get_sector,
        get_sector_description,
        get_sector_peers,
        search_stock,
    )
    _SM_OK = True
    _SM_ERR = ""
except ImportError as exc:
    _SM_OK = False
    _SM_ERR = str(exc).strip() or "sector_master.py import failed"

    def get_all_sectors() -> list[str]:
        return []

    def get_sector(symbol: str) -> str | None:
        return None

    def search_stock(query: str) -> list[tuple[str, str]]:
        return []

    def get_sector_peers(symbol: str) -> list[str]:
        return []

    def get_sector_description(sector_name: str) -> str:
        return sector_name


def stock_search_widget(label: str, key_prefix: str, placeholder: str) -> str:
    """
    Returns bare symbol string e.g. "RELIANCE" or "" if nothing selected.
    """
    query = st.text_input(
        label,
        placeholder=placeholder,
        key=f"{key_prefix}_input",
    ).strip().upper()

    if not query:
        return ""

    matches = search_nse_stocks(query)
    if not matches:
        st.caption("No matches found.")
        return ""

    select_key = f"{key_prefix}_select"
    options = [""] + matches
    if st.session_state.get(select_key, "") not in options:
        st.session_state[select_key] = ""

    chosen = st.selectbox(
        "Select stock",
        options=options,
        key=select_key,
        label_visibility="collapsed",
    )

    if not chosen:
        return ""

    return chosen.split("—", 1)[0].strip()


def render_sector_explorer_section(ticker_universe: list[str] | None = None) -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="margin-bottom:4px;">Sector Explorer</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
        "Fast stock-to-sector lookup from the curated sector database."
        "</div>",
        unsafe_allow_html=True,
    )

    if not _SM_OK:
        st.warning(
            "Sector Explorer is unavailable because `sector_master.py` could not be loaded. "
            f"Import error: {_SM_ERR}"
        )
        return

    _all_sectors = get_all_sectors()
    if not _all_sectors:
        st.warning(
            "Sector Explorer did not find any configured sectors. "
            "Check the `sector_master.py` sector lists."
        )
        return

    _lookup_col1, _lookup_col2 = st.columns([3, 2])
    configure_nse_stock_search(ticker_universe)

    with _lookup_col1:
        _symbol_input = stock_search_widget(
            "Enter stock symbol",
            "sector_exp_search",
            placeholder="e.g. HDFCBANK or company name: HDFC Bank",
        ).strip().upper()

    with _lookup_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        _search_btn = st.button("Find Sector", key="se_search_btn")

    if _search_btn and _symbol_input:
        _exact_sector = get_sector(_symbol_input)
        if _exact_sector:
            _peers = get_sector_peers(_symbol_input)
            st.success(
                f"**{_symbol_input}** -> Primary Sector: **{_exact_sector}**  "
                f"({get_sector_description(_exact_sector)})"
            )
            if _peers:
                st.markdown(
                    f'<div style="font-size:12px;color:#8ab4d8;margin:8px 0 4px;">'
                    f'Sector Peers ({len(_peers)} stocks):</div>',
                    unsafe_allow_html=True,
                )
                st.write(", ".join(_peers))
        else:
            _matches = search_stock(_symbol_input)
            if _matches:
                st.info(
                    f"No exact match for '{_symbol_input}'. "
                    f"Found {len(_matches)} partial match(es)."
                )
                _match_df = pd.DataFrame(_matches, columns=["Symbol", "Sector"])
                _match_df["Description"] = _match_df["Sector"].apply(
                    get_sector_description
                )
                st.dataframe(
                    _match_df,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning(
                    f"'{_symbol_input}' was not found in `sector_master.py`. "
                    "Check the symbol spelling or update the sector map."
                )
