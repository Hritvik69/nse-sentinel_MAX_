# NSE Sentinel

NSE Sentinel is a Streamlit Cloud NSE stock-analysis dashboard for multi-mode stock scanning, Tomorrow's Picks, breakout discovery, sector intelligence, Imported AI Stocks, and self-learning prediction feedback.

> Educational use only. Not financial advice.

## What It Does

| Area | What It Does |
|---|---|
| Multi-mode scanner | Scans NSE stocks across relaxed, swing, intraday, and breakout workflows. |
| Time Travel Mode | Simulates a past market date without mutating live process caches. |
| Tomorrow's Picks | Saves user-facing prediction strips so picks survive app restarts. |
| AI self-learning | Logs predictions, validates outcomes, retrains the learning model, and remembers results. |
| Imported AI Stocks | Keeps a permanent basket of selected stocks for future learning. |
| Sector intelligence | Tracks sector predictions, calibration, and dynamic signal weights. |
| Live Breakout Pulse | Scans momentum/breakout candidates and labels whether data is live, shared, or cached. |
| GitHub-backed persistence | Mirrors important local `data/` files to GitHub for Streamlit Cloud durability. |

## Production Safety Model

The current production hardening focuses on preserving the existing Streamlit UI and strategy logic while making the app safer under concurrent scans, cloud persistence, and app restarts.

### Time Travel

Time Travel Mode is per-session/per-call. It no longer monkey-patches scanner APIs and no longer overwrites live `ALL_DATA`.

Historical and truncated frames are cached separately from live frames, so a simulated scan and a live scan can run concurrently without corrupting each other.

### Model Persistence

`data/learning_model.pkl` is a pickle file, so it is treated as untrusted unless integrity is verified.

The app will only unpickle the persisted model when a trusted external SHA-256 is configured and matches the file:

```powershell
$env:NSE_SENTINEL_MODEL_SHA256 = "expected_sha256_here"
```

or in Streamlit secrets:

```toml
[model_sha256]
learning_model = "expected_sha256_here"
```

If the hash is missing or does not match, the app skips the persisted load and safely retrains or falls back instead of unpickling unknown bytes.

### Atomic Local Writes

CSV, JSON, text, and binary writes use shared atomic helpers in `atomic_io.py`. This reduces the chance of partially written files when Streamlit reruns, GitHub sync, learning refresh, snapshots, or feedback logging overlap.

### GitHub Persistence

Streamlit Cloud has an ephemeral filesystem. NSE Sentinel mirrors managed `data/` files through `persistent_store.py`.

On startup:

```text
Streamlit starts
-> pulls managed files from GitHub
-> writes them locally with atomic file replacement
-> readers use the restored local files
```

On write:

```text
App writes local CSV/JSON/PKL/ZIP atomically
-> queues GitHub push
-> GitHub conflicts are retried with latest SHA
-> CSV/JSON logs are merged where safe
```

Queue failures and persistence errors are logged without printing secrets or tokens.

### HK Dashboard Sync

Tomorrow's Picks can also publish to the HK Dashboard Supabase table after each store sync.
Add these Streamlit secrets:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
```

Keep `SUPABASE_SERVICE_ROLE_KEY` private. Do not put it in Vercel frontend environment variables.

## Files Mirrored To GitHub

| File | Purpose |
|---|---|
| `data/prediction_feedback_log.csv` | Stock prediction logs and validated outcomes |
| `data/tomorrow_master_predictions.csv` | Tomorrow's Picks prediction history |
| `data/tomorrow_picks_store.json` | Saved Tomorrow's Picks strips |
| `data/imported_ai_learning_store.json` | Imported AI Stocks basket |
| `data/learning_status_snapshot.json` | Last learning status snapshot |
| `data/sector_prediction_log.csv` | Sector prediction outcomes |
| `data/sector_signal_performance.csv` | Dynamic sector signal weights |
| `data/learning_model.pkl` | Persisted sklearn learning model, loaded only with trusted SHA-256 |
| `data/market_snapshot_latest.zip` | Latest market snapshot archive |

## Streamlit Secrets Setup

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

The token must be a classic GitHub Personal Access Token with the top-level `repo` scope.

Do not commit the real token into GitHub.

If secrets are missing or wrong, the app still runs, but persistence becomes local-only and Streamlit reboot can wipe saved data.

Optional model integrity secret:

```toml
[model_sha256]
learning_model = "expected_sha256_here"
```

## Prediction Feedback Policy

Stock and sector feedback logs remain backward compatible with old CSV rows, but new rows include additional durable metadata:

| Column | Purpose |
|---|---|
| `market_date` | IST-aware prediction market date |
| `prediction_id` | Stable duplicate-prevention key |
| `prediction_direction` | Explicit bullish/bearish/sideways direction |
| `target_policy_version` | Versioned correctness policy |

Duplicate writes are prevented by symbol/sector, mode where applicable, market date, source, and direction. Historical rows are preserved except exact duplicates during controlled merge paths.

### Last Outcome And Correct

`Last Outcome` comes from `actual_next_return_pct`.

`Correct` is filled after the next trading session close:

| Prediction Type | Correct When |
|---|---|
| Stock bullish | Next-session return is positive |
| Stock bearish | Next-session return is not positive |
| Sector bullish | Next-session return is greater than the sector policy threshold |
| Sector bearish | Next-session return is below the negative sector policy threshold |
| Sector sideways | Absolute next-session return stays within the sector policy threshold |

Example:

```text
Prediction logged: 2026-05-06 IST market date
Next session close: 2026-05-07 after 4:00 PM IST
App opens/reruns after close
-> pending outcomes backfilled
-> Correct becomes True or False
-> learning refresh runs only when feedback materially changed
-> updated files are queued for GitHub sync
```

Streamlit Cloud is not a guaranteed always-on scheduler. If the app is sleeping at 4:00 PM IST, open or reboot it after market close and the validator will run on startup/rerun.

## Snapshots

Market snapshots are written to a temporary directory first, then promoted into place after metadata, counts, and checksums are written. Legacy snapshots without metadata are still supported conservatively.

`snapshot_exists()` now prefers metadata/checksum verification instead of simply counting files.

## Main Project Files

| File | Role |
|---|---|
| `app.py` | Main Streamlit application |
| `atomic_io.py` | Shared atomic write and path-lock helpers |
| `time_travel_engine.py` | Per-session/per-call time travel context and truncated-frame cache |
| `persistent_store.py` | GitHub-backed permanent storage |
| `model_persistence.py` | Secure save/load wrapper for `learning_model.pkl` |
| `prediction_feedback_store.py` | Stock prediction log and outcome backfill |
| `learning_engine.py` | ML training, restore, and prediction helpers |
| `nse_learning_brain.py` | Learning cycle, calibration, and Tomorrow's Picks |
| `sector_prediction_tracker.py` | Sector prediction logging and validation |
| `sector_dynamic_weights.py` | Dynamic sector signal weights |
| `data_session_manager.py` | Market session routing and snapshot management |
| `strategy_engines/` | Scanner engines and shared utilities |
| `tests/test_hardening.py` | Focused production-hardening regression harness |
| `PATCH_FILE_STATUS.md` | Status of legacy patch/archive files |
| `data/` | Local cache plus GitHub-mirrored persistent files |

## Local Setup

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

## Validation

Compile all production Python files:

```powershell
$files = Get-ChildItem -Recurse -Filter *.py |
  Where-Object { $_.FullName -notmatch '\\.git|__pycache__|\\.venv' } |
  ForEach-Object { $_.FullName }
python -m py_compile $files
```

Run focused hardening tests:

```powershell
python -m unittest tests.test_hardening -v
```

Covered cases include:

- concurrent time-travel/live scans
- pickle hash verification
- GitHub 409 and queue-full behavior
- concurrent prediction append/backfill
- durable duplicate logging
- IST date-boundary outcome labeling
- no-data TTL recovery
- Live Breakout Pulse source labeling
- snapshot partial/corrupt/small-valid cases

## Streamlit Cloud Deployment

1. Push code to GitHub.
2. Deploy from `Hritvik69/nse-sentinel_MAX_`.
3. Use the `main` branch unless intentionally changed.
4. Add `[github_store]` secrets.
5. Add `[model_sha256]` if you want persisted pickle model restore enabled.
6. Reboot the app once.
7. Confirm the sidebar shows cloud/GitHub persistence is connected.
8. Confirm Time Travel and normal live/session routing both render without cache leakage.

## Operational Notes

- Market data comes from Yahoo Finance through `yfinance`; close data can lag briefly after 4:00 PM IST.
- GitHub persistence is additive: local writes happen first, then GitHub sync is queued.
- If GitHub sync fails, the app logs the failure and keeps local data.
- Learning improves only from logged predictions with validated outcomes.
- Do not delete persistent `data/` files in GitHub unless you intentionally want to reset saved learning history.
- Do not remove `NSE_SENTINEL_MODEL_SHA256` or `[model_sha256]` and expect `learning_model.pkl` to load; without a trusted hash it is intentionally skipped.

## Disclaimer

NSE Sentinel is for educational and analytical use only. It does not provide investment advice, trading advice, or guaranteed predictions. Always verify data independently and make your own trading decisions.
