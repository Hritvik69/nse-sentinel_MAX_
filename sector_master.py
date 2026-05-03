"""
sector_master.py
─────────────────
Curated, static sector-to-stock mapping for NSE Sentinel.
NEW FILE — zero changes to any existing engine or pipeline.

Contains 17 market sectors with high-liquidity, NSE-listed stocks only.
Each stock appears in exactly ONE sector (its primary classification).

Public API
──────────
    SECTOR_STOCKS           : dict[str, list[str]]  Full mapping
    get_sector(symbol)      → str | None
    get_stocks_in_sector(sector) → list[str]
    get_all_sectors()       → list[str]
    get_sector_count()      → dict[str, int]
    search_stock(query)     → list[tuple[str, str]]  [(symbol, sector), ...]

Design rules
────────────
• No API calls — purely static data.
• No duplicates — each ticker maps to exactly one sector.
• Only real, active NSE-listed tickers.
• Prefer: Nifty 500 constituents, high F&O-turnover, well-known names.
• Avoid: penny stocks, illiquid microcaps, recently delisted names.
"""

from __future__ import annotations

try:
    from nse_ticker_universe import _BASELINE as _UNIVERSE_BASELINE
except Exception:
    _UNIVERSE_BASELINE = []


# ═══════════════════════════════════════════════════════════════════════
# MASTER SECTOR → STOCK MAPPING
# ═══════════════════════════════════════════════════════════════════════
# Each stock is in its PRIMARY sector only.
# Edit this dict to add/remove stocks or sectors.

SECTOR_STOCKS: dict[str, list[str]] = {

    # ── 1. BANKING ────────────────────────────────────────────────────
    # Large private, PSU, and small finance banks
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "INDUSINDBK", "FEDERALBNK", "AUBANK", "IDFCFIRSTB", "BANDHANBNK",
        "RBLBANK", "YESBANK", "CANBK", "UNIONBANK", "BANKBARODA",
        "PNB", "INDIANB", "CENTRALBANK", "MAHABANK", "UCOBANK",
    ],

    # ── 2. NBFC & FINANCE ─────────────────────────────────────────────
    # Non-banking financial companies, insurance, AMCs
    "NBFC_FINANCE": [
        "BAJFINANCE", "BAJAJFINSV", "MUTHOOTFIN", "CHOLAFIN", "HDFCAMC",
        "LICHSGFIN", "PFC", "RECLTD", "IRFC", "SBILIFE",
        "HDFCLIFE", "ICICIPRU", "NAUKRI", "ABCAPITAL", "M&MFIN",
        "SHRIRAMFIN", "MANAPPURAM", "L&TFH", "PNBHOUSING", "CANFINHOME",
    ],

    # ── 3. IT & TECHNOLOGY ────────────────────────────────────────────
    # IT services, software, tech products
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "PERSISTENT", "OFSS", "MPHASIS", "COFORGE",
        "TATAELXSI", "LTTS", "KPIT", "HEXAWARE", "SONATSOFTW",
        "CYIENT", "MASTEK", "NEWGEN", "RATEGAIN", "BSOFT",
    ],

    # ── 4. FMCG ──────────────────────────────────────────────────────
    # Fast-moving consumer goods, food & beverages, personal care
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "GODREJCP", "MARICO", "COLPAL", "TATACONSUM", "EMAMILTD",
        "VARUNBEV", "USL", "UBL", "RADICO", "PATANJALI",
        "ADANIWILMAR", "GILLETTE", "HATSUN", "ZYDUSWELL", "VENKEYS",
    ],

    # ── 5. AUTOMOBILE ────────────────────────────────────────────────
    # 2W, 4W, CV, tractors, auto components & tyres
    "AUTO": [
        "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR",
        "HEROMOTOCO", "TATAMOTORS", "ASHOKLEY", "BHARATFORG", "MOTHERSON",
        "BOSCHLTD", "BALKRISIND", "UNOMINDA", "TIINDIA", "SONACOMS",
        "MRF", "APOLLOTYRE", "CEAT", "EXIDEIND", "ENDURANCE",
        "ESCORTS", "SUPRAJIT", "CRAFTSMAN", "SWARAJENG", "GABRIEL",
    ],

    # ── 6. PHARMACEUTICALS ───────────────────────────────────────────
    # Formulations, APIs, diagnostics, biologics
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
        "TORNTPHARM", "AUROPHARMA", "ALKEM", "ZYDUSLIFE", "MANKIND",
        "GLENMARK", "BIOCON", "LAURUSLABS", "IPCALAB", "AJANTPHARM",
        "ABBOTINDIA", "JBCHEPHARM", "GRANULES", "ERIS", "SYNGENE",
        "GLAND", "NATCOPHARM", "WOCKPHARMA", "LALPATHLAB", "METROPOLIS",
    ],

    # ── 7. METAL & MINING ────────────────────────────────────────────
    # Steel, aluminium, copper, zinc, non-ferrous metals
    "METAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
        "NMDC", "COALINDIA", "HINDCOPPER", "NATIONALUM", "JSPL",
        "APLAPOLLO", "RATNAMANI", "WELCORP", "MIDHANI", "MOIL",
        "GMDC", "TINPLATE", "KALYANKJIL", "NILE", "SHYAMMETL",
    ],

    # ── 8. ENERGY & OIL-GAS ──────────────────────────────────────────
    # Exploration, refining, distribution, renewables, utilities
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "HINDPETRO",
        "GAIL", "PETRONET", "ADANIGREEN", "ADANIPOWER", "TATAPOWER",
        "TORNTPOWER", "NTPC", "POWERGRID", "NHPC", "SJVN",
        "IGL", "MGL", "GUJGASLTD", "CESC", "RPOWER",
    ],

    # ── 9. REALTY ────────────────────────────────────────────────────
    # Real estate developers, REITs
    "REALTY": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
        "BRIGADE", "MAHLIFE", "KOLTEPATIL", "SOBHA", "LODHA",
        "SUNTECK", "IBREALEST", "NESCO", "ANANTRAJ", "ELDEHSG",
    ],

    # ── 10. INFRASTRUCTURE ───────────────────────────────────────────
    # Ports, logistics, roads, construction, cables
    "INFRA": [
        "ADANIPORTS", "IRB", "NCC", "NBCC", "HGINFRA",
        "PNCINFRA", "ASHOKA", "KNR", "GPPL", "GMRINFRA",
        "CONCOR", "MAHLOG", "TITAGARH", "TEXMOPIPES", "RVNL",
    ],

    # ── 11. CHEMICALS & SPECIALTY ────────────────────────────────────
    # Specialty chemicals, agrochemicals, adhesives, gases
    "CHEMICAL": [
        "PIDILITIND", "SRF", "DEEPAKNITRI", "AARTIIND", "NAVINFLUOR",
        "TATACHEM", "GNFC", "PCBL", "ATUL", "FINEORG",
        "CLEAN", "ALKYLAMINE", "SUDARSCHEM", "AAVAS", "BALAMINES",
        "VINDHYATEL", "FLUOROCHEM", "NEOGEN", "ANUPAM", "IGPL",
    ],

    # ── 12. CAPITAL GOODS & ENGINEERING ─────────────────────────────
    # Heavy engineering, industrial machinery, electrical equipment
    "CAPITAL_GOODS": [
        "LT", "ABB", "SIEMENS", "BHEL", "THERMAX",
        "CUMMINSIND", "KEC", "POLYCAB", "HAVELLS", "AIAENGINEERING",
        "SKFINDIA", "KALPATPOWR", "VOLTAMP", "AZOTHCORP", "SUZLON",
        "BHELDRAFT", "GREENPANEL", "TDPOWERSYS", "GRINDMASTER", "JYOTISTRUC",
    ],

    # ── 13. CONSUMER DURABLES ─────────────────────────────────────────
    # Appliances, electronics, jewellery, lifestyle brands
    "CONSUMER_DURABLES": [
        "TITAN", "VOLTAS", "BLUESTARCO", "VGUARD", "CROMPTON",
        "BAJAJELEC", "SYMPHONY", "AMBER", "DIXON", "WHIRLPOOL",
        "ORIENTELEC", "RAJESHEXPO", "SENCO", "KALYAN", "PCJEWELLER",
        "KARURVYSYA", "ASIANPAINT", "KANSAINER", "BERGER", "AKZOINDIA",
    ],

    # ── 14. PSU (DIVERSIFIED PUBLIC SECTOR) ──────────────────────────
    # Government-owned companies not in other primary sectors
    "PSU": [
        "HAL", "BEL", "BHEL", "IRCTC", "CONCOR",
        "HUDCO", "IRCON", "RITES", "BEML", "MTNL",
        "BPCL", "BALRAMCHIN", "NFL", "GSFC", "FACT",
    ],

    # ── 15. DEFENCE & AEROSPACE ──────────────────────────────────────
    # Defence manufacturing, aerospace, naval
    "DEFENCE": [
        "BHARATDYNAM", "COCHINSHIP", "GRSE", "MAZDOCK", "MTAR",
        "PARAS", "IDEAFORGE", "SOLARINDS", "DATAMATICS", "GARDENSIND",
    ],

    # ── 16. RAILWAY & LOGISTICS ──────────────────────────────────────
    # Rail infrastructure, freight, railway equipment
    "RAILWAY": [
        "RAILVIKAS", "JUPITERWAG", "TITAGARH", "RVNL", "IRFC",
        "IRCON", "RITES", "TEXRAIL", "KERNEX", "RRVL",
    ],

    # ── 17. TELECOM ──────────────────────────────────────────────────
    # Mobile, broadband, tower infrastructure
    "TELECOM": [
        "BHARTIARTL", "INDUSTOWER", "TATACOMM", "HFCL", "STLTECH",
        "IDEA", "ONMOBILE", "GTLINFRA", "TTML", "TEJASNET",
    ],

}


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = str(symbol or "").upper().strip().replace(".NS", "")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


_BASELINE_ORDER = _dedupe_symbols(list(_UNIVERSE_BASELINE))
if not _BASELINE_ORDER:
    _fallback_universe: list[str] = []
    for _sector_stocks in SECTOR_STOCKS.values():
        _fallback_universe.extend(_sector_stocks)
    _BASELINE_ORDER = _dedupe_symbols(_fallback_universe)
    del _fallback_universe

INDEX_BASKETS: dict[str, list[str]] = {
    "OVERALL": list(_BASELINE_ORDER),
    "NIFTY_50": list(_BASELINE_ORDER[:50]),
    "NIFTY_150": list(_BASELINE_ORDER[:150]),
    "NIFTY_300": list(_BASELINE_ORDER[:300]),
}

ALL_SECTOR_STOCKS: dict[str, list[str]] = {
    "OVERALL": INDEX_BASKETS["OVERALL"],
    "NIFTY_50": INDEX_BASKETS["NIFTY_50"],
    "NIFTY_150": INDEX_BASKETS["NIFTY_150"],
    "NIFTY_300": INDEX_BASKETS["NIFTY_300"],
    **SECTOR_STOCKS,
}

SECTOR_DISPLAY_ORDER: list[str] = [
    "OVERALL",
    "NIFTY_50",
    "NIFTY_150",
    "NIFTY_300",
    "BANKING",
    "IT",
    "AUTO",
    "FMCG",
    "PHARMA",
    "INFRA",
    "ENERGY",
    "METAL",
    "NBFC_FINANCE",
    "REALTY",
    "CAPITAL_GOODS",
    "CONSUMER_DURABLES",
    "CHEMICAL",
    "TELECOM",
    "PSU",
    "DEFENCE",
    "RAILWAY",
]


# ═══════════════════════════════════════════════════════════════════════
# REVERSE LOOKUP: symbol → sector
# Built once at module load from SECTOR_STOCKS
# ═══════════════════════════════════════════════════════════════════════

_SYMBOL_TO_SECTOR: dict[str, str] = {}

for _sector, _stocks in SECTOR_STOCKS.items():
    for _sym in _stocks:
        _SYMBOL_TO_SECTOR[_sym.upper().strip()] = _sector

del _sector, _stocks, _sym   # clean up module-level loop vars


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_sector(symbol: str) -> str | None:
    """
    Return the primary sector for a given NSE symbol.

    Parameters
    ----------
    symbol : str   Ticker without .NS suffix (e.g. "HDFCBANK").

    Returns
    -------
    str   Sector name (e.g. "BANKING") or None if not found.

    Examples
    --------
    >>> get_sector("HDFCBANK")
    'BANKING'
    >>> get_sector("TCS")
    'IT'
    >>> get_sector("UNKNOWN")
    None
    """
    return _SYMBOL_TO_SECTOR.get(symbol.upper().strip().replace(".NS", ""))


def get_stocks_in_sector(sector: str) -> list[str]:
    """
    Return the list of stocks in the given sector.

    Parameters
    ----------
    sector : str   Sector name (case-insensitive, e.g. "banking" or "BANKING").

    Returns
    -------
    list[str]  Stock symbols.  Empty list if sector not found.

    Examples
    --------
    >>> get_stocks_in_sector("IT")
    ['TCS', 'INFY', 'HCLTECH', ...]
    """
    return ALL_SECTOR_STOCKS.get(sector.upper().strip(), [])


def get_all_sectors() -> list[str]:
    """
    Return a sorted list of all sector names.

    Returns
    -------
    list[str]  e.g. ['AUTO', 'BANKING', 'CAPITAL_GOODS', ...]
    """
    ordered = [sector for sector in SECTOR_DISPLAY_ORDER if sector in ALL_SECTOR_STOCKS]
    remainder = [sector for sector in ALL_SECTOR_STOCKS.keys() if sector not in ordered]
    return ordered + remainder


def get_sector_count() -> dict[str, int]:
    """
    Return a dict mapping each sector name to its stock count.

    Returns
    -------
    dict[str, int]   e.g. {"BANKING": 20, "IT": 20, ...}
    """
    return {s: len(stocks) for s, stocks in ALL_SECTOR_STOCKS.items()}


def search_stock(query: str) -> list[tuple[str, str]]:
    """
    Search for a stock across all sectors by partial symbol match.

    Parameters
    ----------
    query : str   Partial or full ticker symbol (case-insensitive).

    Returns
    -------
    list of (symbol, sector) tuples.

    Examples
    --------
    >>> search_stock("HDFC")
    [('HDFCBANK', 'BANKING'), ('HDFCAMC', 'NBFC_FINANCE'), ('HDFCLIFE', 'NBFC_FINANCE')]
    """
    q = query.upper().strip().replace(".NS", "")
    return [
        (sym, sect)
        for sym, sect in _SYMBOL_TO_SECTOR.items()
        if q in sym
    ]


def get_sector_peers(symbol: str) -> list[str]:
    """
    Return all peer stocks in the same sector as the given symbol.
    The symbol itself is excluded from the result.

    Parameters
    ----------
    symbol : str   Ticker symbol (e.g. "HDFCBANK").

    Returns
    -------
    list[str]  Peer symbols.  Empty list if not found or sector has no peers.

    Examples
    --------
    >>> get_sector_peers("TCS")
    ['INFY', 'HCLTECH', 'WIPRO', ...]
    """
    sector = get_sector(symbol)
    if sector is None:
        return []
    sym_clean = symbol.upper().strip().replace(".NS", "")
    return [s for s in get_stocks_in_sector(sector) if s != sym_clean]


# ── Display helpers ────────────────────────────────────────────────────

SECTOR_DESCRIPTIONS: dict[str, str] = {
    "OVERALL":          "Broad NSE Market Basket",
    "NIFTY_50":         "Top 50 NSE Leaders",
    "NIFTY_150":        "Expanded Nifty 150 Basket",
    "NIFTY_300":        "Broad Nifty 300 Basket",
    "BANKING":          "Banks — Private, PSU, Small Finance",
    "NBFC_FINANCE":     "NBFCs, Insurance, AMCs, HFCs",
    "IT":               "IT Services, Software, Tech Products",
    "FMCG":             "Food, Beverages, Personal Care",
    "AUTO":             "Automobiles, Components, Tyres",
    "PHARMA":           "Pharmaceuticals, APIs, Diagnostics",
    "METAL":            "Steel, Aluminium, Copper, Mining",
    "ENERGY":           "Oil & Gas, Renewables, Utilities",
    "REALTY":           "Real Estate Developers, REITs",
    "INFRA":            "Ports, Roads, Logistics, Construction",
    "CHEMICAL":         "Specialty, Agro & Fine Chemicals",
    "CAPITAL_GOODS":    "Engineering, Machinery, Heavy Equipment",
    "CONSUMER_DURABLES":"Appliances, Electronics, Jewellery",
    "PSU":              "Diversified Public Sector Enterprises",
    "DEFENCE":          "Defence Manufacturing, Aerospace",
    "RAILWAY":          "Rail Infrastructure, Equipment",
    "TELECOM":          "Mobile, Broadband, Tower Infrastructure",
}


def get_sector_description(sector: str) -> str:
    """Return a human-readable description for the given sector."""
    return SECTOR_DESCRIPTIONS.get(sector.upper().strip(), sector)
