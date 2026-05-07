# NSE Sentinel

NSE Sentinel is a Streamlit Cloud stock-analysis dashboard for NSE equities. It combines multi-mode scanning, Tomorrow's Picks, sector intelligence, breakout discovery, Imported AI Stocks, and a self-learning feedback loop that improves from logged prediction outcomes.

This project is for education, research, and analysis only. It is not financial advice.

## What The App Does

- Scans NSE stocks across relaxed, swing, intraday, and breakout-style workflows.
- Builds Tomorrow's Picks from saved prediction strips.
- Stores Imported AI Stocks so selected names can feed the learning engine later.
- Logs prediction rows and validates them after the next trading session close.
- Trains and restores a scikit-learn learning model from persistent storage.
- Tracks sector predictions and dynamic signal performance.
- Supports Streamlit Cloud deployment with GitHub-backed permanent storage.

## Permanent Storage

Streamlit Cloud uses an ephemeral filesystem. That means files written inside the app container can disappear after a reboot, hibernation, or redeploy.

NSE Sentinel fixes this by mirroring important local `data/` files into GitHub using `persistent_store.py`.

Synced files include:

- `data/prediction_feedback_log.csv`
- `data/tomorrow_master_predictions.csv`
- `data/tomorrow_picks_store.json`
- `data/imported_ai_learning_store.json`
- `data/learning_status_snapshot.json`
- `data/sector_prediction_log.csv`
- `data/sector_signal_performance.csv`
- `data/learning_model.pkl`

Startup flow:

```text
Streamlit app starts
-> pulls saved files from GitHub
-> restores local data/
-> restores learning model from learning_model.pkl
-> UI shows saved picks and learning state again
```

Write flow:

```text
App writes local data file
-> persistent_store.push_file()
-> GitHub file is updated in the background
-> data survives Streamlit reboot
```

## Streamlit Secrets Setup

Add this in Streamlit Cloud:

`App settings -> Secrets`

```toml
[github_store]
token  = "PASTE_YOUR_GITHUB_PAT_HERE"
owner  = "Hritvik69"
repo   = "nse-sentinel_MAX_"
branch = "main"
```

The GitHub token must be a classic Personal Access Token with the top-level `repo` scope enabled. Do not commit the real token into this repository.

If secrets are missing or wrong, the app still runs, but saved picks and learning files are local-only and can vanish after a Streamlit reboot.

## Imported AI Stocks

Imported AI Stocks is the permanent basket for names you want the learning engine to remember.

Typical flow:

1. Run a scan or open a Top 3 / breakout panel.
2. Add selected names into Imported AI Stocks.
3. Open Imported AI Stocks.
4. Click `Self Improve` to log those names into the prediction feedback log.
5. After the next trading day close, the app fills `Last Outcome` and `Correct`.

Important: a row must be logged first. If `Logged` says `No`, click `Self Improve` so the app has a prediction row to validate.

## Outcome And Correct Logic

The `Last Outcome` column comes from `actual_next_return_pct` in `prediction_feedback_log.csv`.

The `Correct` column is calculated after the next session close:

- Bullish predictions are correct when the next-session return is positive.
- Bearish predictions are correct when the next-session return is not positive.
- Sector predictions use sector-tracker direction thresholds.

Example:

```text
Prediction logged on 2026-05-06
Next trading session closes on 2026-05-07 after 4:00 PM IST
App opens or reruns after close
-> pending outcomes are backfilled
-> Correct becomes True or False
-> learning model retrains
-> updated files are pushed to GitHub
```

Streamlit Cloud is not a guaranteed always-on scheduler. If the app is sleeping at 4:00 PM, open or reboot the app after market close and it will run the post-close validator on startup/rerun.

## Main Files

```text
app.py                         Main Streamlit app
persistent_store.py            GitHub-backed persistence layer
model_persistence.py           Save/load learning_model.pkl
prediction_feedback_store.py   Stock prediction feedback log
learning_engine.py             ML training and prediction helpers
nse_learning_brain.py          Learning cycle, calibration, Tomorrow's Picks
sector_prediction_tracker.py   Sector prediction logging and outcome tracking
sector_dynamic_weights.py      Dynamic sector signal weights
strategy_engines/              Mode engines and shared scanner utilities
data/                          Local cache and mirrored persistent files
```

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
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

## Streamlit Cloud Deployment

1. Push changes to GitHub.
2. In Streamlit Cloud, deploy from `Hritvik69/nse-sentinel_MAX_`.
3. Confirm the app is using the `main` branch unless intentionally changed.
4. Add the `[github_store]` secrets.
5. Reboot the app once after adding secrets.
6. Confirm the sidebar says cloud/GitHub persistence is connected.

## Validation Checklist

- Tomorrow's Picks stay visible after Streamlit reboot.
- Imported AI Stocks stay visible after Streamlit reboot.
- `prediction_feedback_log.csv` updates in GitHub after `Self Improve`.
- `Last Outcome` and `Correct` fill after the next trading day close.
- `learning_model.pkl` appears in GitHub after a successful learning model train.
- Sidebar persistence warning is gone after valid secrets are configured.

## Notes

- The app uses Yahoo Finance data through `yfinance`; availability can lag briefly after market close.
- Some local cache files are performance helpers, not the source of permanent truth.
- GitHub storage is additive; local writes still happen first, then GitHub is updated.
- Avoid editing or deleting persistent `data/` files in GitHub unless you intentionally want to reset stored learning data.

## Disclaimer

NSE Sentinel is for educational and analytical use only. It does not provide investment advice, trading advice, or guaranteed predictions. Always verify market data and make your own decisions.
