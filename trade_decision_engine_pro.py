# learning_engine.py
# ─────────────────────────────────────────────
# Self-learning system using your prediction logs

from __future__ import annotations

import pandas as pd
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


MODEL = None
SCALER = None


# ─────────────────────────────────────────────
# LOAD LOG DATA
# ─────────────────────────────────────────────

def load_log_data(path="data/prediction_feedback_log.csv"):
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


# ─────────────────────────────────────────────
# PREPARE FEATURES
# ─────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    try:
        # Convert outcome to binary
        df["target"] = df["actual_next_return_pct"].apply(
            lambda x: 1 if float(x) > 0 else 0
        )

        features = []

        # Basic features (you can expand later)
        X = pd.DataFrame({
            "prediction_score": pd.to_numeric(df["prediction_score"], errors="coerce"),
            "final_score": pd.to_numeric(df["final_score"], errors="coerce"),
        })

        X = X.fillna(0)
        y = df["target"]

        return X, y

    except Exception:
        return None, None


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_learning_model():

    global MODEL, SCALER

    if not SKLEARN_OK:
        return None

    df = load_log_data()
    if df is None or len(df) < 100:
        return None

    X, y = prepare_features(df)
    if X is None:
        return None

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        MODEL = model
        SCALER = scaler

        return True

    except Exception:
        return None


# ─────────────────────────────────────────────
# PREDICT SUCCESS PROBABILITY
# ─────────────────────────────────────────────

def predict_success(row: dict):

    if MODEL is None or SCALER is None:
        return 50.0  # neutral

    try:
        X = np.array([[
            float(row.get("Prediction Score", 50)),
            float(row.get("Final Score", 50)),
        ]])

        X_scaled = SCALER.transform(X)
        prob = MODEL.predict_proba(X_scaled)[0][1]

        return round(prob * 100, 1)

    except Exception:
        return 50.0
