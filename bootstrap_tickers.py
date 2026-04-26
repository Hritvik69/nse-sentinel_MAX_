"""
bootstrap_tickers.py
─────────────────────
Run this ONCE on your local PC (not Streamlit Cloud) to generate a
nse_tickers.txt with the full ~2985 NSE ticker list.

After running, commit the updated nse_tickers.txt to your GitHub repo.
The app will then always show ~2985 tickers regardless of hibernation.

Usage:
    cd <your-app-folder>          # same folder that contains app.py
    python bootstrap_tickers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Make sure nse_ticker_universe is importable ───────────────────────
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from nse_ticker_universe import get_all_tickers, invalidate_cache
except ImportError:
    print("ERROR: Could not import nse_ticker_universe.py")
    print("Make sure this script is in the same folder as nse_ticker_universe.py")
    sys.exit(1)

print("Fetching full NSE ticker list from GitHub + NSE...")
print("(This may take 10–30 seconds)")

# Force a fresh fetch (ignore any cached values)
invalidate_cache()
tickers = get_all_tickers(live=True)

print(f"\n✅ Fetched {len(tickers):,} tickers")

# ── Write to nse_tickers.txt ──────────────────────────────────────────
output_path = HERE / "nse_tickers.txt"
bare_symbols = [t.replace(".NS", "") for t in sorted(tickers)]
output_path.write_text("\n".join(bare_symbols), encoding="utf-8")

print(f"✅ Written {len(bare_symbols):,} symbols to {output_path}")
print()
print("Next steps:")
print("  1. git add nse_tickers.txt")
print("  2. git commit -m 'chore: update nse_tickers.txt with full list'")
print("  3. git push")
print()
print("After pushing, the Streamlit Cloud app will show the full count")
print("on every wake-up — permanently, with no network dependency.")
