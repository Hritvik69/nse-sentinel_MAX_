# 📡 NSE Sentinel

**NSE Sentinel** is a Streamlit Cloud NSE stock-analysis dashboard built for scanning, Tomorrow's Picks, breakout discovery, sector intelligence, Imported AI Stocks, and self-learning prediction feedback.

> ⚠️ **Educational use only. Not financial advice.**

---

## 🚀 What NSE Sentinel Does

| Area | What It Does |
|---|---|
| 🔎 **Multi-Mode Scanner** | Scans NSE stocks across relaxed, swing, intraday, and breakout workflows. |
| 🌅 **Tomorrow's Picks** | Saves user-facing prediction strips so picks survive app restarts. |
| 🧠 **AI Self-Learning** | Logs predictions, validates outcomes, retrains the learning model, and remembers results. |
| 📦 **Imported AI Stocks** | Keeps a permanent basket of selected stocks for future learning. |
| 🏭 **Sector Intelligence** | Tracks sector-level predictions, calibration, and dynamic signal weights. |
| 💾 **Permanent Cloud Storage** | Mirrors important local `data/` files to GitHub so Streamlit reboot does not wipe them. |

---

## ✅ Permanent Storage Fix

Streamlit Cloud has an **ephemeral filesystem**. If the app reboots, hibernates, or redeploys, files inside the container can disappear.

NSE Sentinel solves that with:

```text
local data/ file
-> persistent_store.py
-> GitHub Contents API
-> permanent file in repo
```

On app startup:

```text
Streamlit starts
-> pulls saved files from GitHub
-> restores local data/
-> restores learning_model.pkl
-> Tomorrow's Picks + AI learning state return
```

On app write:

```text
App writes CSV/JSON/PKL locally
-> background GitHub push
-> data survives reboot
```

### 📁 Files Mirrored To GitHub

| File | Purpose |
|---|---|
| `data/prediction_feedback_log.csv` | Stock prediction logs and validated outcomes |
| `data/tomorrow_master_predictions.csv` | Tomorrow's Picks prediction history |
| `data/tomorrow_picks_store.json` | Saved Tomorrow's Picks strips |
| `data/imported_ai_learning_store.json` | Imported AI Stocks basket |
| `data/learning_status_snapshot.json` | Last learning status snapshot |
| `data/sector_prediction_log.csv` | Sector prediction outcomes |
| `data/sector_signal_performance.csv` | Dynamic sector signal weights |
| `data/learning_model.pkl` | Persisted sklearn learning model |

---

## 🔐 Streamlit Secrets Setup

Add this in:

```text
Streamlit Cloud -> App settings -> Secrets
```

```toml
[github_store]
token  = "PASTE_YOUR_GITHUB_PAT_HERE"
owner  = "Hritvik69"
repo   = "nse-sentinel_MAX_"
branch = "main"
```

The token must be a **classic GitHub Personal Access Token** with the top-level `repo` scope.

Do **not** commit the real token into GitHub.

If secrets are missing or wrong, the app still runs, but persistence becomes local-only and Streamlit reboot can wipe saved data.

---

## 🧠 Imported AI Stocks + Self Improve

Imported AI Stocks is the permanent basket for names you want the learning engine to remember.

### Normal Flow

1. Run a scan or open a Top 3 / breakout panel.
2. Add selected names into **Imported AI Stocks**.
3. Open **Imported AI Stocks**.
4. Click **Self Improve**.
5. The app logs those names into `prediction_feedback_log.csv`.
6. After the next trading day closes, the app fills **Last Outcome** and **Correct**.
7. The learning model retrains and saves itself to GitHub.

> If a row says `Logged = No`, click **Self Improve** first. The app can only validate predictions that were logged.

---

## 🕓 Last Outcome + Correct Logic

`Last Outcome` comes from:

```text
prediction_feedback_log.csv -> actual_next_return_pct
```

`Correct` is calculated after the next trading session close:

| Prediction Type | Correct When |
|---|---|
| 📈 Bullish | Next-session return is positive |
| 📉 Bearish | Next-session return is not positive |
| 🏭 Sector | Sector tracker direction threshold is satisfied |

Example:

```text
Prediction logged: 2026-05-06
Next session close: 2026-05-07 after 4:00 PM IST
App opens/reruns after close
-> pending outcomes backfilled
-> Correct becomes True or False
-> learning model retrains
-> updated files pushed to GitHub
```

⚠️ Streamlit Cloud is not a guaranteed always-on scheduler. If the app is sleeping at 4:00 PM, open or reboot it after market close and the validator will run on startup/rerun.

---

## 🧩 Main Project Files

| File | Role |
|---|---|
| `app.py` | Main Streamlit application |
| `persistent_store.py` | GitHub-backed permanent storage |
| `model_persistence.py` | Save/load `learning_model.pkl` |
| `prediction_feedback_store.py` | Stock prediction log and outcome backfill |
| `learning_engine.py` | ML training, restore, and prediction helpers |
| `nse_learning_brain.py` | Learning cycle, calibration, and Tomorrow's Picks |
| `sector_prediction_tracker.py` | Sector prediction logging and validation |
| `sector_dynamic_weights.py` | Dynamic sector signal weights |
| `strategy_engines/` | Scanner engines and shared utilities |
| `data/` | Local cache plus GitHub-mirrored persistent files |

---

## 🛠️ Local Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

---

## ☁️ Streamlit Cloud Deployment

1. Push code to GitHub.
2. Deploy from `Hritvik69/nse-sentinel_MAX_`.
3. Use the `main` branch unless intentionally changed.
4. Add `[github_store]` secrets.
5. Reboot the app once.
6. Confirm the sidebar shows cloud/GitHub persistence is connected.

---

## 🧪 Validation Checklist

- ✅ Tomorrow's Picks survive Streamlit reboot.
- ✅ Imported AI Stocks survive Streamlit reboot.
- ✅ `prediction_feedback_log.csv` updates in GitHub after **Self Improve**.
- ✅ `Last Outcome` and `Correct` fill after the next trading day close.
- ✅ `learning_model.pkl` appears in GitHub after model training.
- ✅ Sidebar persistence warning disappears after valid secrets are configured.

---

## 📝 Notes

- 📊 Market data comes from Yahoo Finance through `yfinance`; close data can lag briefly after 4:00 PM IST.
- 💾 GitHub persistence is additive: local writes still happen first, then GitHub sync happens in the background.
- 🧠 Learning improves only from logged predictions with validated outcomes.
- 🚫 Do not delete persistent `data/` files in GitHub unless you intentionally want to reset saved learning history.

---

## ⚠️ Disclaimer

NSE Sentinel is for educational and analytical use only. It does not provide investment advice, trading advice, or guaranteed predictions. Always verify data independently and make your own trading decisions.
