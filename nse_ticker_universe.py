"""
nse_ticker_universe.py
──────────────────────
SINGLE SOURCE OF TRUTH for all NSE tickers used by NSE Sentinel.

JUGAAD FIX (Apr 2025)
─────────────────────
• Separate _GITHUB_HEADERS (no NSE Referer — was causing silent rejects)
• Never cache when count < 2000 — forces retry on next page load
• GitHub timeout raised to 20 s + session warm-up
• More robust URL list
• Larger _BASELINE (all known NSE mainboard stocks)
• nse_tickers.txt now has 1,533 clean unique tickers (replace old file)

Public API
──────────
    get_all_tickers(live=True)  → list[str]     (cached after first call)
    get_bare_symbols()          → list[str]     (without .NS suffix)
    ticker_count()              → int
    invalidate_cache()          → None
"""

from __future__ import annotations

import io
import re
import tempfile
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from atomic_io import atomic_write_text

_LOCK  = threading.Lock()
_cache: dict[bool, list[str]] = {}
_LAST_DIAGNOSTICS: dict[str, str] = {}


def _record_diagnostic(key: str, exc: Exception) -> None:
    _LAST_DIAGNOSTICS[key] = f"{type(exc).__name__}: {exc}"


def get_last_diagnostics() -> dict[str, str]:
    return dict(_LAST_DIAGNOSTICS)

# ══════════════════════════════════════════════════════════════════════
# BASELINE  (all known NSE mainboard — always available, zero network)
# ══════════════════════════════════════════════════════════════════════
_BASELINE: list[str] = [
    # ── LARGE CAP / NIFTY 50 ─────────────────────────────────────────
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
    "BAJFINANCE","HCLTECH","SUNPHARMA","TITAN","ULTRACEMCO","ONGC",
    "NESTLEIND","WIPRO","POWERGRID","NTPC","TECHM","INDUSINDBK","ADANIPORTS",
    "TATAMOTORS","JSWSTEEL","BAJAJFINSV","HINDALCO","GRASIM","DIVISLAB",
    "CIPLA","DRREDDY","BPCL","EICHERMOT","APOLLOHOSP","TATACONSUM","BRITANNIA",
    "COALINDIA","HEROMOTOCO","SHREECEM","SBILIFE","HDFCLIFE","ADANIENT",
    "BAJAJ-AUTO","TATASTEEL","UPL","M&M",
    # ── NIFTY NEXT 50 ────────────────────────────────────────────────
    "ADANIGREEN","ADANITRANS","ATGL","AWL","BAJAJHFL","BANKBARODA",
    "BERGEPAINT","BEL","BHEL","BOSCHLTD","CANBK","CGPOWER","CHOLAFIN",
    "COLPAL","CUMMINSIND","DABUR","DLF","DMART","GODREJCP","GODREJPROP",
    "HAL","HAVELLS","HDFCAMC","ICICIGI","ICICIPRULI","IOC","IRCTC",
    "IRFC","LODHA","LTIM","LTTS","MARICO","MOTHERSON","MUTHOOTFIN",
    "NAUKRI","NHPC","OFSS","PAGEIND","PERSISTENT","PFC","PIDILITIND",
    "POLYCAB","PNB","RECLTD","SHRIRAMFIN","SRF","TORNTPHARM","TRENT",
    "TVSMOTOR","UNIONBANK","VEDL","ZOMATO","ZYDUSLIFE",
    # ── NIFTY MIDCAP 150 ─────────────────────────────────────────────
    "AARTIIND","ABCAPITAL","ABFRL","ACC","AFFLE","AJANTPHARM",
    "ALKEM","ALKYLAMINE","ANGELONE","APLAPOLLO","APOLLOTYRE","APTUS",
    "ASTRAL","AUBANK","AUROPHARMA","BALKRISHIND","BANDHANBNK","BATAINDIA",
    "BIKAJI","BIOCON","BIRLASOFT","BLUESTARCO","BRIGADE","CARBORUNIV",
    "CASTROLIND","CDSL","CESC","CHENNPETRO","CLEAN","COFORGE","CROMPTON",
    "CYIENT","DALBHARAT","DATAMATICS","DCBBANK","DEEPAKNTR","DELHIVERY",
    "DEVYANI","DIXON","DOMS","ECLERX","EIDPARRY","ELGIEQUIP",
    "EMAMILTD","ENDURANCE","EPL","EQUITASBNK","ESCORTS","EXIDEIND",
    "FINEORG","FLUOROCHEM","FORTIS","GABRIEL","GALAXYSURF","GLAND",
    "GLAXO","GLENMARK","GNFC","GRINDWELL","GSFC","GUJGASLTD",
    "HAPPSTMNDS","HATSUN","HFCL","HLEGLAS","HOMEFIRST","HPCL",
    "HUDCO","IEX","IGL","INDIAMART","INDIGO","INOXWIND","IOB",
    "IPCALAB","IREDA","ISEC","JBCHEPHARM","JKCEMENT","JKLAKSHMI",
    "JMFINANCIL","JSL","JUBILANT","JUBLFOOD","KAJARIACER","KALPATPOWR",
    "KALYANKJIL","KANSAINER","KEC","KIMS","KPITTECH","KRSNAA",
    "KSCL","LAURUSLABS","LAXMIMACH","LICHSGFIN","LLOYDSME","LUXIND",
    "MANAPPURAM","MANKIND","MAPMYINDIA","MASTEK","MAXHEALTH",
    "METROPOLIS","MFSL","MIDHANI","MOTILALOFS",
    "MPHASIS","MRPL","NAVINFLUOR","NBCC","NIACL","NILKAMAL",
    "NMDC","NOCIL","NYKAA","OBEROIRLTY","OIL","ORIENTELEC","PATELENG",
    "PAYTM","PCBL","PGHH","PHOENIXLTD","PNBHOUSING","POLICYBZR",
    "POLYMED","PRESTIGE","PRINCEPIPE","RITES","RVNL","SAFARI",
    "SAIL","SCHAEFFLER","SJVN","SKF","SOBHA","SONACOMS",
    "STARHEALTH","SUDARSCHEM","SUNDARMFIN",
    "SUNTECK","SUNTV","SUZLON","SYMPHONY",
    "TANLA","TATACHEM","TATACOMM","TATAPOWER","TEAMLEASE",
    "TIINDIA","TIMKEN","TITAGARH","TORNTPOWER","TRIDENT",
    "TRIVENI","UJJIVANSFB","UTIAMC","VGUARD",
    "VMART","VOLTAS","WELSPUNLIV","WHIRLPOOL",
    "ZEEL","ZENSAR",
    # ── NIFTY SMALLCAP 250 ───────────────────────────────────────────
    "3MINDIA","AAVAS","ACE","ACRYSIL","ADFFOODS","AEGISLOG","AETHER",
    "AIAENG","AKZOINDIA","ALEMBICLTD","ALICON","ALLCARGO","ALOKINDS","AMBIKCO",
    "AMBUJACEM","AMBER","ANANTRAJ","ANUP","APARINDS","APOLLOPIPE",
    "ARCHIDPLY","ARVINDFASHN","ASAHIINDIA","ASHIANA","ASHOKLEY","ASTERDM",
    "ASTRAZEN","ATUL","AVANTIFEED","AXISCADES","AZAD",
    "BAJAJCON","BAJAJHIND","BALKRISHIND","BALMLAWRIE","BALRAMCHIN","BANKBARODA",
    "BASF","BBTC","BECTORFOOD","BHAGERIA","BHARATFORG","BHARATGEAR",
    "BHORUKA","BIRLACABLE","BOROLTD","BPL","BSEINDIA","BURGERKING",
    "BUTTERFLY","CADILAHC","CAMLINFINE","CAMPUS",
    "CANFINHOME","CANTABIL","CAPACITE","CARERATING","CARTRADE",
    "CERA","CHALET","CHAMBLFERT","CHEMPLASTS",
    "CIEINDIA","CMSINFO","COCHINSHIP","COROMANDEL","COSMOFILMS","CRAFTSMAN",
    "CREDITACC","CRISIL","CUMMINSIND","CYIENT","DALBHARAT","DALMIASUG",
    "DBCORP","DCBBANK","DEEPAKFERT","DELTACORP","DHARMAJ","DISHTV",
    "DOLLAR","DREDGECORP","EASEMYTRIP","ECLERX","EIDPARRY","EIL",
    "ELECTCAST","ELGIEQUIP","EMKAY","ESABINDIA","ESAFSFB","ESCORTS",
    "ESTER","ETHOSLTD","FEDERALBNK","FINEORG","FINOLEX",
    "FORCEMOT","FORTIS","GAEL","GALAXYSURF","GANESHBE","GARFIBRES","GARWARE",
    "GATI","GATEWAY","GHCL","GILLETTE","GLOBALVECT",
    "GMBREW","GMRAIRPORT","GNFC","GOACARBON","GOKALDAS","GOLDIAM",
    "GOODLUCK","GPIL","GPPL","GRANULES","GREAVESCOT","GREENPANEL",
    "GREENPLY","GRINDWELL","GRSE","GUFICBIO","GUJALKALI","GUJGASLTD",
    "GULFOILLUB","HCG","HDFCAMC","HECL","HERITGFOOD","HFCL","HGINFRA",
    "HIKAL","HLEGLAS","HMVL","HONAUT","HUBTOWN","HUHTAMAKI",
    "IBREALEST","IDFC","IDFCFIRSTB","INDHOTEL","INDIAMART",
    "INDIANB","INDIGO","INDORAMA","INDOSTAR","INFIBEAM","INTELLECT",
    "IOB","IPCALAB","IRCON","ISEC","ITD","ITDCEM","IVP",
    "JAGRAN","JAIBALAJI","JAICORPLTD","JAMNAAUTO","JASH","JBMA",
    "JCHAC","JIOFIN","JKTYRE","JMFINANCIL","JPPOWER",
    "JTEKTINDIA","JUSTDIAL","JYOTHYLAB","KAJARIACER","KALPATPOWR","KALYANKJIL",
    "KAMDHENU","KANSAINER","KARURVYSYA","KCP","KDDL","KHADIM",
    "KIRIINDUS","KNR","KOLTEPATIL","KOPRAN","KPRMILL","KRBL","KSCL",
    "LATENTVIEW","LAURUSLABS","LEMONTREE","LLOYDSME","LUPIN","LUXIND",
    "MAHINDCIE","MAHINDLOG","MAHSEAMLES","MAITHANALL","MANAPPURAM","MARKSANS",
    "MASTEK","MAWANASUG","MCDOWELL-N","MEDPLUS","MFSL","MIDHANI",
    "MINDAIND","MINDACORP","MINDSPACE","MIRC","MMTC","MOLDTEK",
    "MONTECARLO","MOTILALOFS","MPHASIS","MRPL","MSTCLTD","MUTHOOTCAP",
    "NACLIND","NAHARPOLY","NAHARSPINN","NAVNETEDUL","NBCC",
    "NEULANDLAB","NEWGEN","NILKAMAL","NLCINDIA","NOCIL","NUCLEUS",
    "OBEROIRLTY","OFSS","ORCHPHARMA","ORIENTBELL","ORIENTCEM","ORIENTELEC",
    "PAISALO","PANAMAPET","PARADEEP","PATELENG","PFIZER",
    "PHOENIXLTD","PILANIINVS","POLYMED","POWERMECH","PPAP","PRAJIND",
    "PRICOLLTD","PRISM","PVRINOX","QUESS","QUICKHEAL","RADICO",
    "RAJRATAN","RALLIS","RAMCOCEM","RAYMOND","RBLBANK","RECLTD",
    "REDTAPE","RELAXO","RENUKA","REPCOHOME","RFCL","RITES",
    "ROLEXRINGS","ROSSARI","RPOWER","RUPA","RVNL","SAFARI","SAKSOFT",
    "SANOFI","SAPPHIRE","SEASOFTS","SEQUENT","SESHAPAPER",
    "SHANKARA","SHAREINDIA","SHILPAMED","SHIVALIK","SHOPERSTOP",
    "SIYSIL","SKFINDIA","SKIPPER","SNOWMAN","SOBHA","SOLARA",
    "SOLARINDS","SONACOMS","SOTL","SPARC","STLTECH","SUBROS",
    "SUPRIYA","SUPRAJIT","SURANASOL","SURYAROSNI","SUVENPHAR",
    "SYMPHONY","TAINWALCHM","TANLA","TATACHEM","TATACOMM","TATAPOWER",
    "TDPOWERSYS","THYROCARE","TIINDIA","TIMKEN","TITAGARH",
    "TORNTPOWER","TRIDENT","TRIVENI","UFLEX","UJJIVANSFB","UTIAMC",
    "VGUARD","VIPIND","VMART","VOLTAMP","VOLTAS","VSTIND","WELSPUNLIV",
    "WHIRLPOOL","ZEEL","ZENSAR",
    # ── BANKS & NBFC ─────────────────────────────────────────────────
    "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK",
    "BANKBARODA","PNB","UNIONBANK","CANBK","IDFCFIRSTB","FEDERALBNK",
    "RBLBANK","KARURVYSYA","DCBBANK","AUBANK","EQUITASBNK","UJJIVANSFB",
    "ESAFSFB","CAPITALSFB","JSFB","SURYODAY","UTKARSHBNK","NSDL",
    "BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","MANAPPURAM",
    "SHRIRAMFIN","LICHSGFIN","PNBHOUSING","CANFINHOME","HOMEFIRST",
    "APTUS","AAVAS","REPCO","CREDITACC","SRTRANSFIN","TATAELXSI",
    "HDFCAMC","UTIAMC","ABSLAMC","NAUKRI","ANGELONE","ISEC","IIFL",
    "CDSL","BSEINDIA","MCXINDIA","5PAISA","ICICIGI","ICICIPRULI",
    "SBILIFE","HDFCLIFE","STARHEALTH","MAXFINSERV","MFSL","BAJAJHFL",
    "360ONE","KFINTECH","CAMS","MASFIN","GEOJITFSL","EMKAY","EDELWEISS",
    "RELIGARE","MOTILALOFS","MOFSL",
    # ── IT & TECH ────────────────────────────────────────────────────
    "TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","LTTS","MPHASIS",
    "COFORGE","PERSISTENT","KPITTECH","BIRLASOFT","MASTEK","CYIENT",
    "ECLERX","NIIT","NIITLTD","RATEGAIN","TANLA","NEWGEN",
    "INTELLECT","HAPPSTMNDS","TATAELXSI","ZENSAR","HEXAWARE","NUCLEUS",
    "SAKSOFT","DATAMATICS","QUICKHEAL","INFIBEAM","INDIAMART","AFFLE",
    "NAZARA","LATENTVIEW","MAPMYINDIA","ZAGGLE","NAUKRI",
    "ACCELYA","ONMOBILE","NELCO","MASTECH","MINDTECK","KELLTONTECH",
    "RSSOFTWARE","ISGEC",
    # ── PHARMA / HEALTH ──────────────────────────────────────────────
    "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","BIOCON",
    "LUPIN","ALKEM","AJANTPHARM","IPCALAB","NATCOPHARMA","JBCHEPHARM",
    "GLENMARK","GLAND","GRANULES","LAURUSLABS","FLUOROCHEM","SOLARA",
    "SEQUENT","SUPRIYA","NAVINFLUOR","SUVENPHAR","NEULANDLAB","ORCHPHARMA",
    "FINEORG","ALKYLAMINE","DEEPAKNTR","NOCIL","SUDARSCHEM","VINATIORGA",
    "LXCHEM","AARTI","AARTIIND","VINATI","PCBL","ROSSARI","TATACHEM",
    "GNFC","GSFC","EIDPARRY","COROMANDEL","UPL","RALLIS","DHANUKA",
    "BAYER","PARADEEP","INSECTICID",
    "APOLLOHOSP","MAXHEALTH","FORTIS","KIMS","ASTER","ASTERDM",
    "NARAYANAH","SHALBY","THYROCARE","METROPOLIS","KRSNAA","HCG",
    "CAPLIN","ALEMBICLTD","INDOCO","BLISSGVS","SMITPHARM","STRIDES",
    "SYNGENE","MARKSANS","GUFICBIO","NECCL","SHILPA","PARAS",
    # ── AUTO & ANCILLARIES ───────────────────────────────────────────
    "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
    "TVSMOTOR","ASHOKLEY","ESCORTS","FORCEMOT","SMLISUZU",
    "BALKRISHIND","APOLLOTYRE","JKTYRE","CEATLTD","MINDA",
    "MOTHERSON","BOSCHLTD","ENDURANCE","MINDACORP","SONACOMS","TIINDIA",
    "SUPRAJIT","SUBROS","GABRIEL","WABCOINDIA","JTEKTINDIA","CRAFTSMAN",
    "SCHAEFFLER","SKF","TIMKEN","NRB","MAHINDCIE","PRICOLLTD",
    "SWARAJENG","LUMAX","LUMAXIND","BELRISE","BHAGYANAGAR",
    "TAINWALCHM","TEXRAIL","ROLEXRINGS","SANDHAR","INDSIL","JAYINDAUTO",
    "HARITA","SOMICONV","AUTOAXLES",
    # ── METALS & MINING ──────────────────────────────────────────────
    "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","MOIL",
    "NALCO","HINDCOPPER","TINPLATE","RATNAMANI","JSL",
    "APLAPOLLO","LLOYDSME","JSHL","GPIL","NAVA","WELCORP","SUNFLAG",
    "PRAKASHIND","RAJRATAN",
    # ── ENERGY / OIL & GAS ───────────────────────────────────────────
    "ONGC","BPCL","IOC","HPCL","GAIL","OIL","MRPL","CHENNPETRO",
    "ATGL","IGL","MGL","GUJGASLTD","GSPL","TORNTPOWER","TATAPOWER",
    "ADANIGREEN","ADANIPOWER","ADANITRANS","NHPC","NTPC","POWERGRID",
    "SJVN","IREDA","INOXWIND","SUZLON","RPOWER","JPPOWER","CESC",
    "PTC","NLCINDIA","NTPCGREEN","IRFC",
    # ── FMCG ─────────────────────────────────────────────────────────
    "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","MARICO",
    "GODREJCP","DABUR","EMAMILTD","COLPAL","PGHH","JYOTHYLAB","BAJAJCON",
    "GILLETTE","BIKAJI","AVANTIFEED","KRBL","HATSUN","HERITGFOOD",
    "VSTIND","RADICO","MCDOWELL-N","TILAKNAGAR",
    # ── RETAIL / CONSUMER ────────────────────────────────────────────
    "TRENT","DMART","WESTLIFE","JUBLFOOD","DEVYANI","DIXON","AMBER",
    "SHOPERSTOP","VMART","BATAINDIA","RELAXO","KPRMILL",
    "PAGEIND","VEDANT","GOKALDAS","RAYMOND","CANTABIL","VIPIND",
    "LUXIND","RUPA","DOLLAR","CAMPUS","NYKAA","SAFARI","DOMS","REDTAPE",
    "HAWKINSCOOKE","BARBEQUE","KHAITAN","EASYDAY","BURGERKING",
    # ── REAL ESTATE / INFRA ──────────────────────────────────────────
    "DLF","OBEROIRLTY","GODREJPROP","PHOENIXLTD","BRIGADE","SOBHA",
    "KOLTEPATIL","SUNTECK","LODHA","PRESTIGE","ANANTRAJ","OMAXE",
    "NESCO","IBREALEST","MHRIL","CHALET","LEMONTREE","INDHOTEL",
    "MAHLIFE","ELDEHSG","EMAMIREALTY","SHRIRAMPROP","PURAVANK",
    "PARSVNATH","UNITECHLTD","NIRLON",
    # ── CAPITAL GOODS / ENGINEERING ──────────────────────────────────
    "LT","SIEMENS","ABB","BHEL","THERMAX","CUMMINSIND","ELGIEQUIP",
    "KEC","KALPATPOWR","GRINDWELL","TIMKEN","SKF","SCHAEFFLER","HAL",
    "BEL","MTAR","GRSE","COCHINSHIP","MAZAGON",
    "PATELENG","NBCC","IRCON","HGINFRA","KNR","ASHOKA",
    "ITD","CAPACITE","GPPL","CONCOR","ALLCARGO","AEGISLOG","BLUEDART",
    "GATI","TCI","DREDGECORP","RVNL","RAILTEL","TITAGARH",
    "AIAENG","APARINDS","GREAVESCOT","TDPOWERSYS","VOLTAMP","POWERMECH",
    "KIRLOSKAR","KIRLOSBROS","STERLINGNW","EMCO","APAR","INDSIL",
    "GRAPHITE","POWERINDIA","VBL","TDSL","IDEAFORGE",
    # ── TELECOM / MEDIA ──────────────────────────────────────────────
    "BHARTIARTL","TATACOMM","RAILTEL","HFCL","STLTECH",
    "INDUSTOWER","OPTIEMUS","DISHTV","SUNTV","PVRINOX","DBCORP",
    "JAGRAN","HMVL","NDTV","ZEEMEDIA","NETWORK18","TV18BRDCST",
    "HATHWAY","DEN","NXTDIGITAL","SITI","INOXLEISUR","TVTODAY",
    "BRIGHTCOM","BALAJITELE","TIPS","SHEMAROO","MTNL","TTML",
    # ── AGRI / FERTILISERS ───────────────────────────────────────────
    "UPL","DHANUKA","BAYER","RALLIS","PARADEEP","COROMANDEL","GSFC",
    "GNFC","CHAMBLFERT","KSCL","INSECTICID","DHARMAJ","EIDPARRY",
    "BAJAJHIND","BALRAMCHIN","RENUKA","TRIVENI","MAWANASUG",
    "DHAMPUR","THIRU","SAKTHI","PASUPTAC","KOTASUG","UGAR","BANNARI",
    "PIINDUSTRIES","SUMITOMO",
    # ── LOGISTICS ────────────────────────────────────────────────────
    "DELHIVERY","ALLCARGO","GATI","TCI","BLUEDART","CONCOR","SNOWMAN",
    "INTERGLOBE","SPICEJET","GMRAIRPORT",
    # ── TEXTILES ─────────────────────────────────────────────────────
    "WELSPUNLIV","TRIDENT","RAYMOND","SIYARAM","VIPIND","GOKALDAS",
    "ARVINDFASHN","SUTLEJTEX","AYMSYNTEX","FILATEX","GARFIBRES","GARWARE",
    "MORARJEE","MPDL","AMBIKCO","PATSPIN","KITEX","NITIN",
    "SUMEETIND","VARDHMANTEXT","WINSOME","GINNI","NANDANEXIM",
    "ARVIND","PASUPATI","BOMBTECH","BANSWRAS","HANUNG","MANGLAM",
    # ── PAPER / PACKAGING ────────────────────────────────────────────
    "UFLEX","MOLDTEK","HUHTAMAKI","EPL","COSMOFILMS","PRINCEPIPE","ASTRAL",
    "TNPL","CENTUPAPER","WPIL","JKPAPER","STARPAPERS","ANDHRPAPMILL",
    # ── DEFENCE ──────────────────────────────────────────────────────
    "HAL","BEL","BHEL","MTAR","GRSE","COCHINSHIP","MAZAGON","MIDHANI",
    "IDEAFORGE",
    # ── GEMS & JEWELLERY ─────────────────────────────────────────────
    "TITAN","KALYANKJIL","RAJESHEXPO","GOLDIAM","SENCO","THANGAMAYL",
    "TRIBHOVANDAS","PCJEWELLER","VAIBHAVGBL",
    # ── CONSUMER DURABLES ────────────────────────────────────────────
    "HAVELLS","CROMPTON","ORIENTELEC","BLUESTARCO","VOLTAS","SYMPHONY",
    "VGUARD","CERA","KAJARIACER","SOMANYCER",
    # ── FINTECH / NEW AGE ────────────────────────────────────────────
    "PAYTM","ZOMATO","NYKAA","POLICYBZR","DELHIVERY","CARTRADE",
    "MAPMYINDIA","RATEGAIN","ZAGGLE","NAZARA","LATENTVIEW","EASEMYTRIP",
    "IEX","IRCTC","SWIGGY","BAJAJHOUSING","ACMESOLAR","NTPCGREEN",
    # ── SUGAR / CHEMICALS EXTRA ──────────────────────────────────────
    "AARTISURF","AARTIDRUGS","BALAMINES","CAMPHO","THIRUMALAI","MEGHMANI",
    "IGPL","PAUSHAK","SUDARSCHEM","GUJFLUORO","CAMLIN","LXCHEM",
    "VINATIORGA","NAVINFLUOR","FLUOROCHEM","FINEORG","AARTI","VINATI",
    # ── MISC VERIFIED ────────────────────────────────────────────────
    "GREENPLY","BAJAJELEC","QUESS","TEAMLEASE","JUSTDIAL",
    "AFFLE","RATEGAIN","NAZARA","LATENTVIEW","ZAGGLE","EASEMYTRIP",
    "MMTC","STCINDIA","MSTCLTD","IRCTC","ABBOTINDIA","HONAUT",
    "GLAXO","PFIZER","SANOFI","CARBORUNIV","WENDT","ELGIEQUIP",
    "JNKLINDIA","YATHARTH","FINCARE","ACMESOLAR","MASFIN","NSDL",
    "AADHARHFC","JSFB","UTKARSHBNK","CAPITALSFB","SURYODAY","5PAISA",
    "IIFL","MAXFINSERV","SATIN","FUSION","UGROCAP","MUTHOOTMICFIN",
    "SPANDANA","ARMANFIN","KFINTECH","CAMS","CDSL","MCXINDIA",
    "ISEC","ANGELONE","BSEINDIA","360ONE","GEOJITFSL",
    "ALPHAGEO","ALUFLUORIDE","AMRUTANJAN","ANUP","ANUPAM",
    "ARCHIDPLY","ARIHANT","ASAHIINDIA","ASHIANA","ATLASCYCLE","AVTNPL","AZAD",
    "BEDMUTHA","BIRLACABLE","BIRLACORPN","BLKASHYAP","BODALCHEM","BORLTD","BPL",
    "BURNPUR","BUTTERFLY","CADILAHC","CAMLINFINE","CANFINHOME",
    "CHANDNATEXT","CHIRIPAL","CHORDIA","COMFORTFURN","CONART","CONSOFIN",
    "CONSOFINVT","CORALIND","CRAFTSMAN","CRESSL","CREST","DAAWAT","DALMIABHA",
    "DALMIASUG","DAMODARIND","DATAMATICS","DCMSHRIRAM","DEEPAKFERT",
    "DELTACORP","DHARMAJ","DOLLAR","DREDGECORP","DUCON","DYNAMATECH",
    "ECCL","EDELWEISS","ELECTCAST","ELEKTROBS","EMKAY","EMMFORCE","ENMAS",
    "ESABINDIA","ESTER","ETHOSLTD","EXCELSTEEL","EXICOM","EXIDEIND",
    "FIBERWEB","FRONTIND","GAEL","GANESHBE","GARFIBRES","GARWARE",
    "GATEWAY","GHCL","GLOBALVECT","GMBREW","GOACARBON","GODHA","GOODLUCK",
    "GREAVESCOT","GREENPANEL","GREENPLY","GUFICBIO","GULFOILLUB",
    "HECL","HERITGFOOD","HIKAL","HLEGLAS","HMVL","HUBTOWN","HUHTAMAKI",
    "IBREALEST","IDFC","INDORAMA","INDOSTAR","INFIBEAM","IOB","IVP",
    "JAGRAN","JAIBALAJI","JAICORPLTD","JAMNAAUTO","JASH","JBMA","JCHAC",
    "JIOFIN","JKTYRE","JMFINANCIL","JPPOWER","JTEKTINDIA","JYOTHYLAB",
    "KAMDHENU","KCP","KDDL","KHADIM","KIRIINDUS","KOPRAN","KRBL",
    "LLOYDSME","MAHINDLOG","MAHSEAMLES","MAITHANALL","MARKSANS",
    "MAWANASUG","MEDPLUS","MINDAIND","MINDACORP","MIRC","MMTC",
    "MOLDTEK","MONTECARLO","MUTHOOTCAP","NACLIND","NAHARPOLY","NAHARSPINN",
    "NAVNETEDUL","NEULANDLAB","NEWGEN","NILKAMAL","NUCLEUS",
    "ORCHPHARMA","ORIENTBELL","ORIENTCEM","PAISALO","PANAMAPET",
    "PILANIINVS","POLYMED","PPAP","PRAJIND","PRICOLLTD","PVRINOX",
    "QUICKHEAL","RAJRATAN","RAMCOCEM","REDTAPE","RELAXO",
    "REPCOHOME","RFCL","ROLEXRINGS","ROSSARI","RUPA","SAFARI",
    "SAKSOFT","SAPPHIRE","SESHAPAPER","SHANKARA","SHAREINDIA","SHILPAMED",
    "SHIVALIK","SIYSIL","SKIPPER","SNOWMAN","SOLARA","SOLARINDS",
    "SOTL","SPARC","SUBROS","SUPRIYA","SUPRAJIT","SURANASOL","SURYAROSNI",
    "SUVENPHAR","TAINWALCHM","THYROCARE","TITAGARH","UFLEX","VIPIND",
    "VOLTAMP","VSTIND","WELSPUNLIV","OLECTRA","MAHANAGAR","ANDHRSUGAR",
    "ANDHRPAPMILL","NIRLON","EMAMIREALTY","MAHLIFE","MAHINDHOLIDAY",
    "EIHOTEL","ORIENTHOTEL","ADVANIHOTR","KAMAT","AJMERA","MAHINDCIE",
    "LUMAX","LUMAXIND","SWARAJENG","BELRISE","BHAGYANAGAR",
    "SANDHAR","INDSIL","JAYINDAUTO","SOMICONV","NRB","AUTOAXLES",
    "PIRHEALTH","ACCELYA","ONMOBILE","NELCO","MINDTECK","KELLTONTECH",
    "ISGEC","HEXAWARE","NIIT","NIITLTD","TANLA","SAKSOFT","BIRLASOFT",
    "MOIL","NALCO","HINDCOPPER","TINPLATE","RATNAMANI","JSHL","NAVA",
    "WELCORP","SUNFLAG","PRAKASHIND","KIRLOSKAR","KIRLOSBROS",
    "STERLINGNW","EMCO","APAR","GRAPHITE","POWERINDIA","VBL","TDSL",
    "VAIBHAVGBL","SENCO","THANGAMAYL","GOLDIAM","RAJESHEXPO","TRIBHOVANDAS",
    "PCJEWELLER","EMBDL",
]

_REPO_TICKER_FILE = Path(__file__).with_name("nse_tickers.txt")
_VALID_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{1,20}$")

# ── Separate header sets — NSE Referer is WRONG for GitHub requests ──
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/plain,text/csv,application/octet-stream,*/*",
    "Referer": "https://www.nseindia.com/",
}
_GITHUB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/plain,*/*",
    # NO Referer — GitHub rejects NSE referer silently
}
# Keep old alias for any code that referenced _RAW_HEADERS externally
_RAW_HEADERS = _NSE_HEADERS

_GITHUB_URLS = [
    # PKScreener — most reliable, ~2400 NSE tickers in SYMBOL.NS format
    "https://raw.githubusercontent.com/pkjmesra/PKScreener/main/pkscreener/classes/tickerlist.txt",
    "https://raw.githubusercontent.com/pkjmesra/PKScreener/main/pkscreener/classes/Tickers.txt",
    # Screeni fork — another 2400+ list
    "https://raw.githubusercontent.com/Screeni-python/Screeni/main/classes/tickerlist.txt",
    # NayakwadiS ticker list
    "https://raw.githubusercontent.com/NayakwadiS/nse_Ticker/master/Nse_Ticker25.txt",
    # hi-tech-jazz NSE symbols
    "https://raw.githubusercontent.com/hi-tech-jazz/nse/main/symbols.txt",
    # Alternate PKScreener branch
    "https://raw.githubusercontent.com/pkjmesra/PKScreener/new-features/pkscreener/classes/tickerlist.txt",
]

_NORMALIZE_MAP = {
    "BHARATIARTL": "BHARTIARTL",
    "BHARATIHEXA": "BHARTIHEXA",
    "ANGELBROKING": "ANGELONE",
    "DRREDDYS": "DRREDDY",
    "DMMART": "DMART",
    "PARDEEP": "PARADEEP",
    "PATEL": "PATELENG",
    "MOLD-TEK": "MOLDTEK",
    "DIVI": "DIVISLAB",
    "CAMPUSACTIVEWEAR": "CAMPUS",
    "PARASONFL": "PARAGONFL",
    "RATEAIN": "RATEGAIN",
    "AHEMDABDSTL": "AHMEDABADSTEEL",
}
_DROP_SYMBOLS = {
    "AARTIINDALKEM","AIRPORT","CAREGIVING","INDUIND","RBKL",
    "WABAGGOO","ORDNANCE","MCDONALDS","NATIONAL","PATANJALIFO",
    "TVSTOUCHSCR","TORRENTP",
}

# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

_TMP_TICKER_FILE = str(Path(tempfile.gettempdir()) / "nse_sentinel_live_tickers_v3.txt")


def get_all_tickers(live: bool = True) -> list[str]:
    """
    Return sorted list of 'SYMBOL.NS' strings.

    Cache priority:
      1. Module _cache (memory, this Python process)
      2. /tmp/ file   (written after a successful large fetch;
                       survives Streamlit reruns within same server)
      3. Full _build() (GitHub + NSE + repo txt)

    Never caches a result < 2000 — forces retry on next call.
    """
    with _LOCK:
        if live in _cache:
            cached = _cache[live]
            if len(cached) >= 2000:
                return cached
            _cache.clear()

    # ── Check /tmp/ before any network call ───────────────────────────
    if live:
        try:
            import os
            if os.path.exists(_TMP_TICKER_FILE):
                with open(_TMP_TICKER_FILE, "r", encoding="utf-8") as _fh:
                    _lines = [l.strip() for l in _fh.read().splitlines() if l.strip()]
                if len(_lines) >= 2000:
                    _formatted = sorted(f for f in (_format_symbol(l) for l in _lines) if f)
                    if len(_formatted) >= 2000:
                        with _LOCK:
                            _cache[live] = _formatted
                        return _formatted
        except Exception as exc:
            _record_diagnostic("tmp_cache_read", exc)

    result = _build(live)

    with _LOCK:
        if len(result) >= 2000:
            _cache[live] = result
    return result


def get_bare_symbols() -> list[str]:
    """Return ticker list without .NS suffix."""
    return [t.replace(".NS", "") for t in get_all_tickers()]


def ticker_count() -> int:
    return len(get_all_tickers())


def invalidate_cache(clear_disk: bool = False) -> None:
    """Force re-build on next call."""
    with _LOCK:
        _cache.clear()
    if clear_disk:
        try:
            Path(_TMP_TICKER_FILE).unlink(missing_ok=True)
        except Exception as exc:
            _record_diagnostic("tmp_cache_clear", exc)


# ══════════════════════════════════════════════════════════════════════
# INTERNAL BUILDER
# ══════════════════════════════════════════════════════════════════════

def _build(live: bool) -> list[str]:
    repo_tickers = _load_repo_tickers()
    baseline_tickers = _baseline_tickers()
    tickers = set(repo_tickers)
    tickers.update(baseline_tickers)

    if not live:
        return sorted(tickers)

    # ── Try ALL sources (no early-break) for maximum coverage ─────────
    current_equities = _fetch_nse_equity_list()
    if len(current_equities) >= 1500:
        tickers = set(current_equities)
    else:
        tickers.update(_fetch_github_raw_lists())
        tickers.update(current_equities)
        if len(tickers) < 2500:
            tickers.update(_fetch_bhav_copy())

    if not tickers:
        tickers = set(baseline_tickers)

    result = sorted(tickers)

    # ── Write-back: persist large list to /tmp/ AND repo txt ──────────
    # /tmp/ survives Streamlit reruns within the same server process.
    # repo txt is updated so future cold deploys already have a big list.
    if len(result) >= 2000:
        try:
            bare_content = "\n".join(t.replace(".NS", "") for t in result)
            try:
                atomic_write_text(_TMP_TICKER_FILE, bare_content, encoding="utf-8")
            except Exception as exc:
                _record_diagnostic("tmp_cache_write", exc)
        except Exception as exc:
            _record_diagnostic("ticker_persist", exc)

    return result


def _normalize_symbol(raw: str) -> str | None:
    symbol = str(raw).strip().upper().replace(".NS", "")
    symbol = _NORMALIZE_MAP.get(symbol, symbol)
    if symbol in _DROP_SYMBOLS:
        return None
    if not _VALID_SYMBOL_RE.fullmatch(symbol):
        return None
    return symbol


def _format_symbol(raw: str) -> str | None:
    symbol = _normalize_symbol(raw)
    if symbol is None:
        return None
    return f"{symbol}.NS"


def _baseline_tickers() -> set[str]:
    return {
        formatted
        for formatted in (_format_symbol(symbol) for symbol in _BASELINE)
        if formatted is not None
    }


def _load_repo_tickers() -> set[str]:
    tickers: set[str] = set()
    try:
        if _REPO_TICKER_FILE.exists():
            for line in _REPO_TICKER_FILE.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                formatted = _format_symbol(line)
                if formatted is not None:
                    tickers.add(formatted)
    except Exception as exc:
        _record_diagnostic("repo_ticker_read", exc)
    return tickers


def _fetch_github_raw_lists() -> set[str]:
    """
    Try all GitHub URLs and combine results for maximum ticker count.
    Uses _GITHUB_HEADERS (no NSE Referer) and 20s timeout.
    FIX: removed break-early — all URLs are tried regardless of count.
    """
    tickers: set[str] = set()
    try:
        import requests

        session = requests.Session()
        session.headers.update(_GITHUB_HEADERS)

        for url in _GITHUB_URLS:
            try:
                response = session.get(url, timeout=20)
                if response.status_code != 200 or len(response.content) < 500:
                    continue
                for line in response.text.splitlines():
                    token = line.strip().split(",")[0].replace('"', "").replace("'", "")
                    formatted = _format_symbol(token)
                    if formatted is not None:
                        tickers.add(formatted)
            except Exception:
                continue
    except Exception:
        pass
    return tickers


def _fetch_nse_equity_list() -> set[str]:
    tickers: set[str] = set()
    try:
        import pandas as pd
        import requests

        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        # Establish cookie first
        try:
            session.get("https://www.nseindia.com/", timeout=8)
        except Exception:
            pass
        response = session.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            timeout=15,
        )
        if response.status_code == 200 and len(response.content) > 5000:
            dataframe = pd.read_csv(io.StringIO(response.text))
            column = "SYMBOL" if "SYMBOL" in dataframe.columns else dataframe.columns[0]
            for symbol in dataframe[column].dropna().unique():
                formatted = _format_symbol(symbol)
                if formatted is not None:
                    tickers.add(formatted)
    except Exception:
        pass
    return tickers


def _fetch_bhav_copy() -> set[str]:
    tickers: set[str] = set()
    try:
        import pandas as pd
        import requests

        for days_back in range(0, 7):
            try:
                dt = datetime.now() - timedelta(days=days_back)
                if dt.weekday() >= 5:
                    continue
                date_str = dt.strftime("%d%b%Y").upper()
                url = (
                    f"https://archives.nseindia.com/content/historical/EQUITIES/"
                    f"{dt.year}/{dt.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
                )
                response = requests.get(url, headers=_NSE_HEADERS, timeout=15)
                if response.status_code != 200 or len(response.content) < 1000:
                    continue
                zipped = zipfile.ZipFile(io.BytesIO(response.content))
                dataframe = pd.read_csv(zipped.open(zipped.namelist()[0]))
                for symbol in dataframe["SYMBOL"].dropna().unique():
                    formatted = _format_symbol(symbol)
                    if formatted is not None:
                        tickers.add(formatted)
                if tickers:
                    break
            except Exception:
                continue
    except Exception:
        pass
    return tickers
