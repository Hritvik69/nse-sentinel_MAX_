from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from prediction_feedback_store import read_feedback_log
except Exception:
    def read_feedback_log() -> pd.DataFrame:  # type: ignore[misc]
        return pd.DataFrame()

try:
    from sector_prediction_tracker import read_log as read_sector_prediction_log
except Exception:
    def read_sector_prediction_log(sector: str | None = None) -> pd.DataFrame:  # type: ignore[misc]
        return pd.DataFrame()


MODEL = None
SCALER = None
REGIME_ENCODER: dict[str, int] = {}
SECTOR_ENCODER: dict[str, int] = {}

TRAINING_STATUS: dict = {
    "trained": False,
    "samples": 0,
    "stock_samples": 0,
    "sector_samples": 0,
    "accuracy_pct": None,
    "last_trained": "",
    "source": "none",
    "message": "Model not trained yet.",
    "regime_encoder": {},
    "sector_encoder": {},
}

_RAW_FEATURE_COLUMNS = [
    "prediction_score",
    "final_score",
    "is_bullish",
    "conviction_tier",
    "regime",
    "sector",
]

_FEATURE_COLUMNS = [
    "prediction_score",
    "final_score",
    "is_bullish",
    "conviction_score",
    "regime_code",
    "sector_code",
]

_MIN_PRIMARY_SAMPLES = 50
_MIN_TRAIN_SAMPLES = 30


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text.lower() in {"nan", "none"}:
                return default
            for token in ("%", ",", "x", "X", "×"):
                text = text.replace(token, "")
            value = text
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _first_present(row, keys, default=None):
    for key in keys:
        try:
            value = row.get(key, None)
        except Exception:
            value = None
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return default


def _to_binary_flag(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "bullish"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "bearish"}:
        return 0
    return int(_safe_float(value, 0.0) > 0.0)


def _to_target_flag(value) -> int | None:
    text = str(value).strip()
    if text == "True":
        return 1
    if text == "False":
        return 0
    return None


def _normalise_text(value, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip().upper()
    return text if text else default


def _conviction_to_numeric(value) -> float:
    text = str(value).strip().lower()
    mapping = {
        "very_low": 0.5,
        "low": 1.0,
        "medium": 2.0,
        "high": 3.0,
        "very_high": 4.0,
        "a+": 4.0,
        "a": 3.5,
        "b": 2.5,
        "c": 1.5,
        "d": 0.5,
    }
    if text in mapping:
        return mapping[text]
    raw = _safe_float(value, float("nan"))
    if np.isfinite(raw):
        if raw > 10.0:
            return float(np.clip(raw / 25.0, 0.5, 4.0))
        return float(np.clip(raw, 0.5, 4.0))
    return 2.0


def _confidence_bucket(value) -> str:
    score = _safe_float(value, 50.0)
    if score >= 75.0:
        return "very_high"
    if score >= 62.0:
        return "high"
    if score >= 48.0:
        return "medium"
    return "low"


def _build_encoder(values: pd.Series, existing: dict[str, int] | None = None) -> dict[str, int]:
    encoder = dict(existing or {})
    if "UNKNOWN" not in encoder:
        encoder["UNKNOWN"] = 0
    next_code = max(encoder.values(), default=0) + 1
    for value in sorted({_normalise_text(v) for v in values.tolist()}):
        if value not in encoder:
            encoder[value] = next_code
            next_code += 1
    return encoder


def _encode_with_map(values: pd.Series, encoder: dict[str, int]) -> pd.Series:
    return values.map(lambda value: encoder.get(_normalise_text(value), 0)).astype(float)


def _encode_feature_frame(df: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
    global REGIME_ENCODER, SECTOR_ENCODER

    frame = df.copy()
    if "prediction_score" not in frame.columns:
        frame["prediction_score"] = 50.0
    if "final_score" not in frame.columns:
        frame["final_score"] = frame["prediction_score"]
    if "is_bullish" not in frame.columns:
        frame["is_bullish"] = (
            pd.to_numeric(frame["prediction_score"], errors="coerce").fillna(50.0) >= 55.0
        ).astype(int)
    if "conviction_tier" not in frame.columns:
        proxy = pd.to_numeric(frame.get("final_score", 50.0), errors="coerce").fillna(50.0)
        frame["conviction_tier"] = proxy.map(_confidence_bucket)
    if "regime" not in frame.columns:
        frame["regime"] = "UNKNOWN"
    if "sector" not in frame.columns:
        frame["sector"] = "UNKNOWN"

    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce").fillna(50.0)
    frame["final_score"] = pd.to_numeric(frame["final_score"], errors="coerce").fillna(frame["prediction_score"])
    frame["is_bullish"] = frame["is_bullish"].map(_to_binary_flag).astype(float)
    frame["conviction_tier"] = frame["conviction_tier"].fillna("medium").astype(str)
    frame["regime"] = frame["regime"].fillna("UNKNOWN").astype(str)
    frame["sector"] = frame["sector"].fillna("UNKNOWN").astype(str)

    if fit or not REGIME_ENCODER:
        REGIME_ENCODER = _build_encoder(frame["regime"], existing={})
    if fit or not SECTOR_ENCODER:
        SECTOR_ENCODER = _build_encoder(frame["sector"], existing={})

    encoded = pd.DataFrame(
        {
            "prediction_score": frame["prediction_score"].astype(float),
            "final_score": frame["final_score"].astype(float),
            "is_bullish": frame["is_bullish"].astype(float),
            "conviction_score": frame["conviction_tier"].map(_conviction_to_numeric).astype(float),
            "regime_code": _encode_with_map(frame["regime"], REGIME_ENCODER),
            "sector_code": _encode_with_map(frame["sector"], SECTOR_ENCODER),
        }
    )
    return encoded.fillna(0.0)


def load_log_data(path="data/prediction_feedback_log.csv"):
    try:
        if path == "data/prediction_feedback_log.csv":
            df = read_feedback_log()
        else:
            df = pd.read_csv(path)
        return df if isinstance(df, pd.DataFrame) and not df.empty else None
    except Exception:
        return None


def _build_stock_training_rows() -> pd.DataFrame:
    try:
        df = read_feedback_log()
        if df is None or df.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])

        out = df.copy()
        out["_correct"] = out.get("correct", "").apply(_to_target_flag)
        missing_correct = out["_correct"].isna()
        if missing_correct.any() and "actual_next_return_pct" in out.columns:
            actual = pd.to_numeric(out["actual_next_return_pct"], errors="coerce")
            bullish = out.get("pred_bullish", 0).apply(_to_binary_flag)
            derived = np.where(
                ((bullish == 1) & (actual > 0))
                | ((bullish == 0) & (actual <= 0)),
                1,
                0,
            )
            out.loc[missing_correct & actual.notna(), "_correct"] = derived[missing_correct & actual.notna()]

        out = out[out["_correct"].notna()].copy()
        if out.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])

        final_score = pd.to_numeric(out.get("final_score", 0), errors="coerce").fillna(
            pd.to_numeric(out.get("prediction_score", 0), errors="coerce").fillna(50.0)
        )

        train = pd.DataFrame(
            {
                "prediction_score": pd.to_numeric(out.get("prediction_score", 0), errors="coerce").fillna(50.0),
                "final_score": final_score,
                "is_bullish": out.get("pred_bullish", 0).apply(_to_binary_flag).astype(int),
                "conviction_tier": out.get("conviction_tier", "").fillna("").astype(str).replace("", np.nan),
                "regime": out.get("regime", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "sector": out.get("sector", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "target": out["_correct"].astype(int),
                "source": "stock_feedback",
            }
        )
        missing_conviction = train["conviction_tier"].isna() | (train["conviction_tier"].astype(str).str.strip() == "")
        train.loc[missing_conviction, "conviction_tier"] = train.loc[missing_conviction, "final_score"].map(_confidence_bucket)
        return train.dropna(subset=["target"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])


def _build_sector_training_rows() -> pd.DataFrame:
    try:
        df = read_sector_prediction_log()
        if df is None or df.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])

        out = df.copy()
        out["_correct"] = out.get("correct", "").apply(_to_target_flag)
        out = out[out["_correct"].notna()].copy()
        if out.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])

        direction = out.get("direction", "").astype(str).str.strip().str.lower()
        bullish = np.where(direction.eq("bullish"), 1, np.where(direction.eq("bearish"), 0, 0))

        prediction_score = pd.to_numeric(out.get("raw_score", 0), errors="coerce").fillna(50.0)
        confidence = pd.to_numeric(out.get("confidence", 0), errors="coerce").fillna(prediction_score)

        train = pd.DataFrame(
            {
                "prediction_score": prediction_score,
                "final_score": confidence,
                "is_bullish": bullish.astype(int),
                "conviction_tier": confidence.map(_confidence_bucket),
                "regime": out.get("regime", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "sector": out.get("sector", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "target": out["_correct"].astype(int),
                "source": "sector_feedback",
            }
        )
        return train.dropna(subset=["target"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source"])


def prepare_features(df: pd.DataFrame):
    try:
        if df is None or df.empty or "target" not in df.columns:
            return None, None

        y = pd.to_numeric(df["target"], errors="coerce")
        mask = y.notna()
        if not mask.any():
            return None, None

        raw = df.loc[mask, _RAW_FEATURE_COLUMNS].reset_index(drop=True)
        X = _encode_feature_frame(raw, fit=True)
        return X, y.loc[mask].astype(int).reset_index(drop=True)
    except Exception:
        return None, None


def train_learning_model():
    global MODEL, SCALER, TRAINING_STATUS

    status = {
        "trained": False,
        "samples": 0,
        "stock_samples": 0,
        "sector_samples": 0,
        "accuracy_pct": None,
        "last_trained": "",
        "source": "none",
        "message": "Model not trained yet.",
        "regime_encoder": {},
        "sector_encoder": {},
    }

    if not SKLEARN_OK:
        status["message"] = "scikit-learn unavailable."
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}

    stock_rows = _build_stock_training_rows()
    sector_rows = _build_sector_training_rows()
    status["stock_samples"] = int(len(stock_rows))
    status["sector_samples"] = int(len(sector_rows))

    train_rows = stock_rows.copy()
    if len(train_rows) < _MIN_PRIMARY_SAMPLES and not sector_rows.empty:
        train_rows = pd.concat([train_rows, sector_rows], ignore_index=True)
        status["source"] = "stock+sector"
    else:
        status["source"] = "stock_feedback"

    status["samples"] = int(len(train_rows))
    if len(train_rows) < _MIN_TRAIN_SAMPLES:
        status["message"] = f"Need at least {_MIN_TRAIN_SAMPLES} validated samples."
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}

    X, y = prepare_features(train_rows)
    if X is None or y is None or X.empty:
        status["message"] = "Training features unavailable."
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}

    if int(pd.Series(y).nunique()) < 2:
        status["message"] = "Need both wins and losses to train."
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}

    try:
        class_counts = pd.Series(y).value_counts()
        can_split = len(X) >= 36 and int(class_counts.min()) >= 2
        if can_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")
        model.fit(X_train_scaled, y_train)

        accuracy = float(model.score(X_test_scaled, y_test) * 100.0)
        MODEL = model
        SCALER = scaler
        status.update(
            {
                "trained": True,
                "accuracy_pct": round(accuracy, 2),
                "last_trained": datetime.now().isoformat(timespec="minutes"),
                "message": f"Model trained on {len(X)} samples.",
                "regime_encoder": dict(REGIME_ENCODER),
                "sector_encoder": dict(SECTOR_ENCODER),
            }
        )
        TRAINING_STATUS = status
        return {
            "model": model,
            "scaler": scaler,
            "status": status,
            "regime_encoder": dict(REGIME_ENCODER),
            "sector_encoder": dict(SECTOR_ENCODER),
        }
    except Exception as exc:
        status["message"] = f"Training failed: {exc}"
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}


def restore_learning_bundle(bundle: dict | None) -> bool:
    global MODEL, SCALER, TRAINING_STATUS, REGIME_ENCODER, SECTOR_ENCODER
    try:
        if not isinstance(bundle, dict):
            return False
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        status = bundle.get("status")
        if model is None or scaler is None:
            return False
        MODEL = model
        SCALER = scaler
        REGIME_ENCODER = dict(bundle.get("regime_encoder") or {})
        SECTOR_ENCODER = dict(bundle.get("sector_encoder") or {})
        if isinstance(status, dict):
            status = dict(status)
            status["regime_encoder"] = dict(REGIME_ENCODER)
            status["sector_encoder"] = dict(SECTOR_ENCODER)
            TRAINING_STATUS = status
        return True
    except Exception:
        return False


def get_training_status() -> dict:
    try:
        status = dict(TRAINING_STATUS)
        status["regime_encoder"] = dict(REGIME_ENCODER)
        status["sector_encoder"] = dict(SECTOR_ENCODER)
        return status
    except Exception:
        return {
            "trained": False,
            "samples": 0,
            "stock_samples": 0,
            "sector_samples": 0,
            "accuracy_pct": None,
            "last_trained": "",
            "source": "none",
            "message": "Training status unavailable.",
            "regime_encoder": {},
            "sector_encoder": {},
        }


def _row_to_feature_frame(row: dict | pd.Series) -> pd.DataFrame:
    prediction_score = _safe_float(
        _first_present(row, ["Prediction Score", "prediction_score", "raw_score"], 50.0),
        50.0,
    )
    final_score = _safe_float(
        _first_present(row, ["Final Score", "final_score", "confidence", "Confidence", "raw_score"], prediction_score),
        prediction_score,
    )

    pred_bullish = _first_present(row, ["pred_bullish"], None)
    if pred_bullish is None:
        direction = str(_first_present(row, ["direction", "Direction"], "")).strip().lower()
        if direction == "bullish":
            pred_bullish = 1
        elif direction == "bearish":
            pred_bullish = 0
        else:
            pred_bullish = 1 if prediction_score >= 55.0 else 0

    conviction = _first_present(row, ["conviction_tier", "confidence", "Confidence"], None)
    if conviction is None:
        conviction = _confidence_bucket((prediction_score + final_score) / 2.0)
    elif not isinstance(conviction, str):
        conviction = _confidence_bucket(conviction)

    raw = pd.DataFrame(
        [
            {
                "prediction_score": prediction_score,
                "final_score": final_score,
                "is_bullish": _to_binary_flag(pred_bullish),
                "conviction_tier": conviction,
                "regime": _first_present(row, ["regime", "Regime"], "UNKNOWN"),
                "sector": _first_present(row, ["sector", "Sector"], "UNKNOWN"),
            }
        ]
    )
    return _encode_feature_frame(raw, fit=False)


def predict_success(row: dict):
    if MODEL is None or SCALER is None:
        return 50.0

    try:
        X = _row_to_feature_frame(row)
        X_scaled = SCALER.transform(X)
        prob = MODEL.predict_proba(X_scaled)[0][1]
        return round(float(prob) * 100.0, 1)
    except Exception:
        return 50.0
