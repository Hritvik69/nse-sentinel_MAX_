# 📡 NSE Sentinel

**NSE Sentinel** is a Streamlit-based NSE stock scanner built for fast market screening, sector review, single-stock analysis, and historical simulation from one dashboard.

> ⚠️ For educational use only. Not financial advice.

---

## ✨ What It Does

NSE Sentinel helps you:

- 🔎 scan NSE stocks with focused trading modes
- 📊 rank results through a multi-step scoring pipeline
- 🧭 inspect sector-level opportunities
- 🔮 evaluate a single stock with Stock Aura
- ⚔️ compare multiple stocks side by side
- 🕰️ simulate past market dates with Time Travel mode
- 📁 export full scan results to CSV

---

## 🎯 Current Trading Modes

The current app UI uses 3 main modes:

| Mode | Purpose | Backing Engine |
|---|---|---|
| 🟡 `Mode 1 - Relaxed (Wide Scan)` | Broad market scan | Mode 3 engine |
| 🔴 `Mode 2 - Swing` | Swing-trade style setups | Mode 6 engine |
| 🟢 `Mode 3 - Intraday` | Stocks with strong next-day upside potential | Mode 5 engine |

---

## 🧩 Main Sections

### 📈 Multi-Mode Scanner
- scans NSE stocks using the selected mode
- ranks candidates with the app's scoring pipeline
- keeps visible results compact in the UI
- exports the full matched set to CSV

### 🔭 Sector Screener Dashboard
- sector-by-sector and overall basket scanning
- prediction-oriented dashboard
- Time Travel aware caching and scan handling

### 🔮 Stock Aura
- single-stock analysis panel
- quick read on setup quality, direction, and decision support

### ⚔️ Compare Stocks
- side-by-side comparison workflow

### 🧭 Sector Explorer
- fast stock-to-sector lookup

### 🚀 Breakout Radar
- breakout-focused stock discovery section

### 📡 Live Breakout Pulse
- pulse-style breakout monitoring view

### 🕰️ Time Travel Mode
- simulates historical app behavior using a chosen past date
- designed to avoid using future candles in historical views

---

## 🧠 How the App Works

The scanner runs raw market data through a layered enrichment flow:

```text
Scan -> Score -> Enrichment -> Grading -> Signal refinement -> Ranked output
```

This gives the app a cleaner final result than a plain technical filter list.

---

## 🗂️ Project Structure

```text
nse-sentinel_MAX-main/
|-- app.py
|-- README.md
|-- requirements.txt
|-- sector_master.py
|-- nse_tickers.txt
|-- strategy_engines/
|   |-- __init__.py
|   |-- _engine_utils.py
|   |-- _df_extensions.py
|   |-- app_sector_explorer_section.py
|   |-- app_sector_screener_dashboard.py
|   |-- app_sector_screener_section.py
|   |-- app_sector_intelligence_section.py
|   |-- breakout_radar_engine.py
|   |-- live_breakout_pulse_engine.py
|   |-- market_bias_engine.py
|   |-- multi_index_market_bias_engine.py
|   |-- mode1_engine.py
|   |-- mode2_engine.py
|   |-- mode3_engine.py
|   |-- mode4_engine.py
|   |-- mode5_engine.py
|   |-- mode6_engine.py
|   `-- ...
`-- data/
```

---

## ⚙️ Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the app

```powershell
python -m streamlit run app.py
```

Then open:

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 🧪 Notes

- 📦 The app uses cached/local data where possible to reduce repeated downloads.
- 🤖 Some probability features depend on `scikit-learn`.
- 🕰️ Time Travel uses `time_travel_engine.py` plus section-level safeguards.
- 📁 CSV export keeps the **full** matched dataset even when the UI shows fewer visible rows.

---

## 📚 Core Dependencies

Main packages from [requirements.txt](C:/Users/HP/OneDrive/Documents/moye/nse-sentinel_MAX-main/requirements.txt):

- `streamlit`
- `pandas`
- `numpy`
- `yfinance`
- `ta`
- `plotly`
- `scikit-learn`
- `scipy`
- `requests`

---

## ⚠️ Disclaimer

This project is for:

- learning
- analysis
- experimentation

Please verify data independently and use your own judgment before making any trading decision.
