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
_FEATURE_COLUMNS = [
    "prediction_score",
    "final_score",
    "is_bullish",
    "confidence",
]


class _BackCompatScaler:
    def __init__(self):
        self._scaler = StandardScaler()

    def fit_transform(self, X):
        return self._scaler.fit_transform(_coerce_feature_frame(X))

    def transform(self, X):
        return self._scaler.transform(_coerce_feature_frame(X))


def _coerce_feature_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        frame = pd.DataFrame(arr, columns=_FEATURE_COLUMNS[: arr.shape[1]])

    prediction_score = pd.to_numeric(frame.get("prediction_score", 0), errors="coerce").fillna(0)
    final_score = pd.to_numeric(frame.get("final_score", 0), errors="coerce").fillna(0)

    if "is_bullish" in frame.columns:
        is_bullish = pd.to_numeric(frame["is_bullish"], errors="coerce").fillna(0).astype(int)
    else:
        is_bullish = ((prediction_score >= 50) | (final_score >= 55)).astype(int)

    if "confidence" in frame.columns:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0)
    else:
        combined_score = ((prediction_score + final_score) / 2.0).fillna(0)
        confidence = pd.Series(
            np.select(
                [combined_score >= 60, combined_score >= 45],
                [3, 2],
                default=1,
            ),
            index=frame.index,
            dtype="float64",
        )

    return pd.DataFrame({
        "prediction_score": prediction_score,
        "final_score": final_score,
        "is_bullish": is_bullish,
        "confidence": confidence,
    }).fillna(0)


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

        if "pred_bullish" not in df.columns:
            df["pred_bullish"] = 0
        df["pred_bullish"] = (
            df["pred_bullish"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "1": 1,
                "true": 1,
                "yes": 1,
                "y": 1,
                "bullish": 1,
                "0": 0,
                "false": 0,
                "no": 0,
                "n": 0,
                "bearish": 0,
            })
            .fillna(pd.to_numeric(df["pred_bullish"], errors="coerce"))
            .fillna(0)
            .astype(int)
        )

        if "conviction_tier" in df.columns:
            df["conviction_tier"] = (
                df["conviction_tier"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "low": 1,
                    "medium": 2,
                    "high": 3,
                    "a+": 4,
                    "a": 3,
                    "b": 2,
                    "c": 1,
                    "d": 0,
                })
                .fillna(pd.to_numeric(df["conviction_tier"], errors="coerce"))
                .fillna(0)
            )

        X = pd.DataFrame({
            "prediction_score": pd.to_numeric(df["prediction_score"], errors="coerce"),
            "final_score": pd.to_numeric(df["final_score"], errors="coerce"),

            # ✅ NEW FEATURES
            "is_bullish": df["pred_bullish"].astype(int),

            # optional safety (if column exists)
            "confidence": pd.to_numeric(df.get("conviction_tier", 0), errors="coerce").fillna(0),
        }).fillna(0)

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
        scaler = _BackCompatScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=200)
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
