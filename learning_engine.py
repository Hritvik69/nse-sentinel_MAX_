from __future__ import annotations

from datetime import datetime
import threading

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

try:
    from model_persistence import save_model as _save_model, load_model as _load_model
except Exception:
    def _save_model(*a, **kw):  # type: ignore[misc]
        return False

    def _load_model():  # type: ignore[misc]
        return None


MODEL = None
SCALER = None
REGIME_ENCODER: dict[str, int] = {}
SECTOR_ENCODER: dict[str, int] = {}
FEATURE_ENCODERS: dict[str, dict[str, int]] = {}
_MODEL_LOCK = threading.RLock()
_TRAINING_LOCK = threading.Lock()

TRAINING_STATUS: dict = {
    "trained": False,
    "samples": 0,
    "stock_samples": 0,
    "imported_ai_samples": 0,
    "sector_samples": 0,
    "active_feature_count": 0,
    "recency_weighting_active": False,
    "accuracy_pct": None,
    "validation_accuracy_pct": None,
    "training_accuracy_pct": None,
    "last_trained": "",
    "source": "none",
    "message": "Model not trained yet.",
    "regime_encoder": {},
    "sector_encoder": {},
    "feature_encoders": {},
}

_NUMERIC_FEATURE_COLUMNS = [
    "prediction_score",
    "final_score",
    "is_bullish",
    "rsi",
    "vol_avg_ratio",
    "delta_ema20_pct",
]

_CATEGORICAL_FEATURE_COLUMNS = [
    "conviction_tier",
    "mode",
    "import_source",
    "import_category",
    "strategy_strip",
    "trap_risk",
    "market_bias",
    "regime",
    "sector",
]

_RAW_FEATURE_COLUMNS = [
    *_NUMERIC_FEATURE_COLUMNS,
    *_CATEGORICAL_FEATURE_COLUMNS,
]

_FEATURE_COLUMNS = [
    *_NUMERIC_FEATURE_COLUMNS,
    *[f"{column}_code" for column in _CATEGORICAL_FEATURE_COLUMNS],
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


def _normalise_mode(value) -> str:
    try:
        if value is None:
            return "UNKNOWN"
        text = str(value or "").strip().upper()
        if not text or text in {"NAN", "NONE"}:
            return "UNKNOWN"
        raw = text.replace("MODE", "").replace("M", "").strip()
        mode_f = float(raw)
        if np.isfinite(mode_f):
            return f"M{int(mode_f)}"
    except Exception:
        pass
    return _normalise_text(value)


def _normalise_trap(value) -> str:
    text = _normalise_text(value)
    if "HIGH" in text or "TRAP" in text or "FAKE" in text:
        return "HIGH"
    if "MED" in text or "CAUTION" in text or "WEAK" in text:
        return "MEDIUM"
    if "LOW" in text or "CLEAN" in text:
        return "LOW"
    return text


def _normalise_strip(value) -> str:
    text = _normalise_text(value)
    for label in ("RELAX", "SWING", "INTRADAY", "MOMENTUM", "BREAKOUT"):
        if label in text:
            return label
    if "PULSE" in text or "RADAR" in text:
        return "BREAKOUT"
    return text


def _normalise_categorical_column(column: str, value) -> str:
    if column == "mode":
        return _normalise_mode(value)
    if column == "trap_risk":
        return _normalise_trap(value)
    if column == "strategy_strip":
        return _normalise_strip(value)
    return _normalise_text(value)


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


def _build_feature_encoders(frame: pd.DataFrame, existing: dict[str, dict[str, int]] | None = None) -> dict[str, dict[str, int]]:
    encoders = {key: dict(value) for key, value in dict(existing or {}).items()}
    for column in _CATEGORICAL_FEATURE_COLUMNS:
        values = frame[column] if column in frame.columns else pd.Series(["UNKNOWN"])
        seed = encoders.get(column, {})
        encoder = dict(seed)
        if "UNKNOWN" not in encoder:
            encoder["UNKNOWN"] = 0
        next_code = max(encoder.values(), default=0) + 1
        normalized_values = {_normalise_categorical_column(column, value) for value in values.tolist()}
        for value in sorted(normalized_values):
            if value not in encoder:
                encoder[value] = next_code
                next_code += 1
        encoders[column] = encoder
    return encoders


def _encode_with_map(values: pd.Series, encoder: dict[str, int], *, column: str = "") -> pd.Series:
    return values.map(lambda value: encoder.get(_normalise_categorical_column(column, value), 0)).astype(float)


def _encode_feature_frame(
    df: pd.DataFrame,
    *,
    fit: bool = False,
    regime_encoder: dict[str, int] | None = None,
    sector_encoder: dict[str, int] | None = None,
    feature_encoders: dict[str, dict[str, int]] | None = None,
    update_global: bool = False,
) -> pd.DataFrame:
    global REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS

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
    if "mode" not in frame.columns:
        frame["mode"] = "UNKNOWN"
    if "import_source" not in frame.columns:
        frame["import_source"] = "UNKNOWN"
    if "import_category" not in frame.columns:
        frame["import_category"] = "UNKNOWN"
    if "strategy_strip" not in frame.columns:
        frame["strategy_strip"] = "UNKNOWN"
    if "trap_risk" not in frame.columns:
        frame["trap_risk"] = "UNKNOWN"
    if "market_bias" not in frame.columns:
        frame["market_bias"] = "UNKNOWN"
    if "rsi" not in frame.columns:
        frame["rsi"] = 50.0
    if "vol_avg_ratio" not in frame.columns:
        frame["vol_avg_ratio"] = 1.0
    if "delta_ema20_pct" not in frame.columns:
        frame["delta_ema20_pct"] = 0.0

    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce").fillna(50.0)
    frame["final_score"] = pd.to_numeric(frame["final_score"], errors="coerce").fillna(frame["prediction_score"])
    frame["is_bullish"] = frame["is_bullish"].map(_to_binary_flag).astype(float)
    frame["conviction_tier"] = frame["conviction_tier"].fillna("medium").astype(str)
    frame["rsi"] = pd.to_numeric(frame["rsi"], errors="coerce").fillna(50.0).clip(0.0, 100.0)
    frame["vol_avg_ratio"] = pd.to_numeric(frame["vol_avg_ratio"], errors="coerce").fillna(1.0).clip(0.0, 10.0)
    frame["delta_ema20_pct"] = pd.to_numeric(frame["delta_ema20_pct"], errors="coerce").fillna(0.0).clip(-40.0, 40.0)
    for column in _CATEGORICAL_FEATURE_COLUMNS:
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)

    if fit:
        feature_encoders = _build_feature_encoders(frame)
        regime_encoder = dict(feature_encoders.get("regime", {}))
        sector_encoder = dict(feature_encoders.get("sector", {}))
        if update_global:
            with _MODEL_LOCK:
                REGIME_ENCODER = dict(regime_encoder)
                SECTOR_ENCODER = dict(sector_encoder)
                FEATURE_ENCODERS = {key: dict(value) for key, value in feature_encoders.items()}
    elif feature_encoders is None:
        with _MODEL_LOCK:
            feature_encoders = {key: dict(value) for key, value in FEATURE_ENCODERS.items()}
        if regime_encoder is not None:
            feature_encoders["regime"] = dict(regime_encoder)
        if sector_encoder is not None:
            feature_encoders["sector"] = dict(sector_encoder)
        if "regime" not in feature_encoders:
            feature_encoders["regime"] = dict(REGIME_ENCODER)
        if "sector" not in feature_encoders:
            feature_encoders["sector"] = dict(SECTOR_ENCODER)
    else:
        feature_encoders = {key: dict(value) for key, value in feature_encoders.items()}
        if regime_encoder is not None:
            feature_encoders["regime"] = dict(regime_encoder)
        if sector_encoder is not None:
            feature_encoders["sector"] = dict(sector_encoder)

    for column in _CATEGORICAL_FEATURE_COLUMNS:
        if column not in feature_encoders:
            feature_encoders[column] = {"UNKNOWN": 0}

    encoded_values: dict[str, pd.Series] = {
        "prediction_score": frame["prediction_score"].astype(float),
        "final_score": frame["final_score"].astype(float),
        "is_bullish": frame["is_bullish"].astype(float),
        "rsi": frame["rsi"].astype(float),
        "vol_avg_ratio": frame["vol_avg_ratio"].astype(float),
        "delta_ema20_pct": frame["delta_ema20_pct"].astype(float),
    }
    for column in _CATEGORICAL_FEATURE_COLUMNS:
        if column == "conviction_tier":
            encoded_values["conviction_tier_code"] = frame[column].map(_conviction_to_numeric).astype(float)
        else:
            encoded_values[f"{column}_code"] = _encode_with_map(
                frame[column],
                feature_encoders.get(column, {"UNKNOWN": 0}),
                column=column,
            )
    encoded = pd.DataFrame(encoded_values)
    for column in _FEATURE_COLUMNS:
        if column not in encoded.columns:
            encoded[column] = 0.0
    encoded = encoded[_FEATURE_COLUMNS]
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


def _blank_series(index, default: object = "") -> pd.Series:
    return pd.Series([default] * len(index), index=index)


def _derive_strategy_strip(row) -> str:
    text = " ".join(
        [
            str(_first_present(row, ["strategy_strip", "Strategy Strip", "Tomorrow Strip", "Strip"], "") or ""),
            str(_first_present(row, ["import_category", "Import Category"], "") or ""),
            str(_first_present(row, ["import_source", "Import Source"], "") or ""),
        ]
    ).lower()
    for label in ("relax", "swing", "intraday", "momentum", "breakout"):
        if label in text:
            return label.title()
    if "pulse" in text or "radar" in text:
        return "Breakout"
    return "UNKNOWN"


def _is_imported_training_row(row) -> bool:
    source = str(_first_present(row, ["import_source", "Import Source"], "") or "").strip()
    category = str(_first_present(row, ["import_category", "Import Category"], "") or "").strip()
    strip = str(_first_present(row, ["strategy_strip", "Strategy Strip", "Tomorrow Strip"], "") or "").strip()
    return bool(source or category or strip)


def _coerce_training_frame_columns(out: pd.DataFrame) -> pd.DataFrame:
    for column in _RAW_FEATURE_COLUMNS:
        if column not in out.columns:
            if column == "prediction_score":
                out[column] = 50.0
            elif column == "final_score":
                out[column] = out.get("prediction_score", 50.0)
            elif column == "is_bullish":
                out[column] = 0
            elif column == "rsi":
                out[column] = 50.0
            elif column == "vol_avg_ratio":
                out[column] = 1.0
            elif column == "delta_ema20_pct":
                out[column] = 0.0
            else:
                out[column] = "UNKNOWN"
    return out


def _build_stock_training_rows() -> pd.DataFrame:
    try:
        df = read_feedback_log()
        if df is None or df.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])

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
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])

        final_score = pd.to_numeric(out.get("final_score", 0), errors="coerce").fillna(
            pd.to_numeric(out.get("prediction_score", 0), errors="coerce").fillna(50.0)
        )
        index = out.index
        actual_returns = pd.to_numeric(out.get("actual_next_return_pct", _blank_series(index)), errors="coerce")
        market_bias = out.get("market_bias", _blank_series(index, "UNKNOWN")).fillna("UNKNOWN").astype(str)
        import_source = out.get("import_source", _blank_series(index, "UNKNOWN")).fillna("UNKNOWN").astype(str)
        import_category = out.get("import_category", _blank_series(index, "UNKNOWN")).fillna("UNKNOWN").astype(str)
        strategy_strip = out.apply(_derive_strategy_strip, axis=1)
        mode_series = out.get("mode", _blank_series(index, "UNKNOWN")).fillna("UNKNOWN").astype(str)
        trap_risk = out.get("trap_risk", _blank_series(index, "UNKNOWN")).fillna("UNKNOWN").astype(str)
        rsi = pd.to_numeric(out.get("rsi", _blank_series(index, 50.0)), errors="coerce").fillna(50.0)
        vol_avg_ratio = pd.to_numeric(out.get("vol_avg_ratio", _blank_series(index, 1.0)), errors="coerce").fillna(1.0)
        delta_ema20 = pd.to_numeric(out.get("delta_ema20_pct", _blank_series(index, 0.0)), errors="coerce").fillna(0.0)
        sample_date = out.get("market_date", out.get("logged_at", _blank_series(index, "")))

        train = pd.DataFrame(
            {
                "prediction_score": pd.to_numeric(out.get("prediction_score", 0), errors="coerce").fillna(50.0),
                "final_score": final_score,
                "is_bullish": out.get("pred_bullish", 0).apply(_to_binary_flag).astype(int),
                "conviction_tier": out.get("conviction_tier", "").fillna("").astype(str).replace("", np.nan),
                "mode": mode_series,
                "import_source": import_source,
                "import_category": import_category,
                "strategy_strip": strategy_strip,
                "rsi": rsi,
                "vol_avg_ratio": vol_avg_ratio,
                "delta_ema20_pct": delta_ema20,
                "trap_risk": trap_risk,
                "market_bias": market_bias,
                "regime": out.get("regime", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "sector": out.get("sector", "UNKNOWN").fillna("UNKNOWN").astype(str),
                "target": out["_correct"].astype(int),
                "source": out.apply(lambda row: "imported_ai" if _is_imported_training_row(row) else "stock_feedback", axis=1),
                "sample_date": sample_date,
                "actual_next_return_pct": actual_returns,
                "is_imported_ai": out.apply(_is_imported_training_row, axis=1).astype(int),
            }
        )
        missing_conviction = train["conviction_tier"].isna() | (train["conviction_tier"].astype(str).str.strip() == "")
        train.loc[missing_conviction, "conviction_tier"] = train.loc[missing_conviction, "final_score"].map(_confidence_bucket)
        train = _coerce_training_frame_columns(train)
        return train.dropna(subset=["target"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])


def _build_sector_training_rows() -> pd.DataFrame:
    try:
        df = read_sector_prediction_log()
        if df is None or df.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])

        out = df.copy()
        out["_correct"] = out.get("correct", "").apply(_to_target_flag)
        out = out[out["_correct"].notna()].copy()
        if out.empty:
            return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])

        direction = out.get("direction", "").astype(str).str.strip().str.lower()
        bullish = np.where(direction.eq("bullish"), 1, np.where(direction.eq("bearish"), 0, 0))

        prediction_score = pd.to_numeric(out.get("raw_score", 0), errors="coerce").fillna(50.0)
        confidence = pd.to_numeric(out.get("confidence", 0), errors="coerce").fillna(prediction_score)
        index = out.index
        market_bias = out["market_bias"].fillna("UNKNOWN").astype(str) if "market_bias" in out.columns else _blank_series(index, "UNKNOWN")
        regime = out["regime"].fillna("UNKNOWN").astype(str) if "regime" in out.columns else _blank_series(index, "UNKNOWN")
        sector = out["sector"].fillna("UNKNOWN").astype(str) if "sector" in out.columns else _blank_series(index, "UNKNOWN")
        sample_date = out["market_date"] if "market_date" in out.columns else out["logged_at"] if "logged_at" in out.columns else _blank_series(index, "")
        actual_return = pd.to_numeric(out["return_pct"], errors="coerce") if "return_pct" in out.columns else _blank_series(index, np.nan)

        train = pd.DataFrame(
            {
                "prediction_score": prediction_score,
                "final_score": confidence,
                "is_bullish": bullish.astype(int),
                "conviction_tier": confidence.map(_confidence_bucket),
                "mode": "SECTOR",
                "import_source": "sector_prediction",
                "import_category": "Sector",
                "strategy_strip": "Sector",
                "rsi": 50.0,
                "vol_avg_ratio": 1.0,
                "delta_ema20_pct": 0.0,
                "trap_risk": "UNKNOWN",
                "market_bias": market_bias,
                "regime": regime,
                "sector": sector,
                "target": out["_correct"].astype(int),
                "source": "sector_feedback",
                "sample_date": sample_date,
                "actual_next_return_pct": actual_return,
                "is_imported_ai": 0,
            }
        )
        train = _coerce_training_frame_columns(train)
        return train.dropna(subset=["target"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=_RAW_FEATURE_COLUMNS + ["target", "source", "sample_date", "actual_next_return_pct", "is_imported_ai"])


def _build_sample_weights(df: pd.DataFrame) -> tuple[np.ndarray | None, bool]:
    try:
        if df is None or df.empty:
            return None, False
        weights = pd.Series(1.0, index=df.index, dtype=float)
        recency_active = False
        date_series = None
        for column in ("sample_date", "market_date", "logged_at"):
            if column in df.columns:
                parsed = pd.to_datetime(df[column], errors="coerce")
                if parsed.notna().any():
                    date_series = parsed
                    break
        if date_series is not None:
            latest = date_series.max()
            age_days = (latest - date_series).dt.days.clip(lower=0)
            recency = np.exp(-age_days.fillna(90.0) / 120.0)
            weights *= 0.55 + (0.95 * recency)
            recency_active = True
        if "actual_next_return_pct" in df.columns:
            returns = pd.to_numeric(df["actual_next_return_pct"], errors="coerce").abs().fillna(0.0)
            weights *= 1.0 + returns.clip(upper=5.0) / 12.0
        weights = weights.clip(lower=0.25, upper=2.5)
        return weights.to_numpy(dtype=float), bool(recency_active)
    except Exception:
        return None, False


def _prepare_training_features(df: pd.DataFrame):
    try:
        if df is None or df.empty or "target" not in df.columns:
            return None, None, {}, None, False

        y = pd.to_numeric(df["target"], errors="coerce")
        mask = y.notna()
        if not mask.any():
            return None, None, {}, None, False

        raw = df.loc[mask, _RAW_FEATURE_COLUMNS].reset_index(drop=True)
        feature_encoders = _build_feature_encoders(raw)
        X = _encode_feature_frame(
            raw,
            fit=False,
            feature_encoders=feature_encoders,
        )
        sample_weights, recency_active = _build_sample_weights(df.loc[mask].reset_index(drop=True))
        return X, y.loc[mask].astype(int).reset_index(drop=True), feature_encoders, sample_weights, recency_active
    except Exception:
        return None, None, {}, None, False


def prepare_features(df: pd.DataFrame):
    X, y, feature_encoders, _sample_weights, _recency_active = _prepare_training_features(df)
    if X is not None:
        with _MODEL_LOCK:
            global REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS
            FEATURE_ENCODERS = {key: dict(value) for key, value in feature_encoders.items()}
            REGIME_ENCODER = dict(FEATURE_ENCODERS.get("regime", {}))
            SECTOR_ENCODER = dict(FEATURE_ENCODERS.get("sector", {}))
    return X, y


def train_learning_model():
    global MODEL, SCALER, TRAINING_STATUS
    if not _TRAINING_LOCK.acquire(blocking=False):
        status = dict(TRAINING_STATUS)
        status["message"] = "Training already in progress."
        return {"model": MODEL, "scaler": SCALER, "status": status}

    try:
        return _train_learning_model_locked()
    finally:
        _TRAINING_LOCK.release()


def _train_learning_model_locked():
    global MODEL, SCALER, TRAINING_STATUS, REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS

    status = {
        "trained": False,
        "samples": 0,
        "stock_samples": 0,
        "imported_ai_samples": 0,
        "sector_samples": 0,
        "active_feature_count": len(_FEATURE_COLUMNS),
        "recency_weighting_active": False,
        "accuracy_pct": None,
        "validation_accuracy_pct": None,
        "training_accuracy_pct": None,
        "last_trained": "",
        "source": "none",
        "message": "Model not trained yet.",
        "regime_encoder": {},
        "sector_encoder": {},
        "feature_encoders": {},
    }

    if not SKLEARN_OK:
        status["message"] = "scikit-learn unavailable."
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}

    stock_rows = _build_stock_training_rows()
    sector_rows = _build_sector_training_rows()
    status["stock_samples"] = int(len(stock_rows))
    if isinstance(stock_rows, pd.DataFrame) and not stock_rows.empty and "is_imported_ai" in stock_rows.columns:
        status["imported_ai_samples"] = int(pd.to_numeric(stock_rows["is_imported_ai"], errors="coerce").fillna(0).astype(int).sum())
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

    X, y, feature_encoders, sample_weights, recency_active = _prepare_training_features(train_rows)
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
            if sample_weights is None:
                sample_weights = np.ones(len(X), dtype=float)
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X,
                y,
                sample_weights,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
            w_train = sample_weights if sample_weights is not None else None
            w_test = sample_weights if sample_weights is not None else None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")
        model.fit(X_train_scaled, y_train, sample_weight=w_train)

        score_pct = float(model.score(X_test_scaled, y_test, sample_weight=w_test) * 100.0)
        training_pct = float(model.score(X_train_scaled, y_train, sample_weight=w_train) * 100.0)
        with _MODEL_LOCK:
            MODEL = model
            SCALER = scaler
            FEATURE_ENCODERS = {key: dict(value) for key, value in feature_encoders.items()}
            REGIME_ENCODER = dict(FEATURE_ENCODERS.get("regime", {}))
            SECTOR_ENCODER = dict(FEATURE_ENCODERS.get("sector", {}))
        status.update(
            {
                "trained": True,
                "active_feature_count": len(_FEATURE_COLUMNS),
                "recency_weighting_active": bool(recency_active),
                "accuracy_pct": round(score_pct, 2) if can_split else None,
                "validation_accuracy_pct": round(score_pct, 2) if can_split else None,
                "training_accuracy_pct": round(training_pct, 2),
                "last_trained": datetime.now().isoformat(timespec="minutes"),
                "message": (
                    f"Model trained on {len(X)} samples."
                    if can_split
                    else f"Model trained on {len(X)} samples; validation held until enough holdout data is available."
                ),
                "regime_encoder": dict(FEATURE_ENCODERS.get("regime", {})),
                "sector_encoder": dict(FEATURE_ENCODERS.get("sector", {})),
                "feature_encoders": FEATURE_ENCODERS,
            }
        )
        TRAINING_STATUS = status
        try:
            _save_model(model, scaler, REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS)
        except TypeError:
            _save_model(model, scaler, REGIME_ENCODER, SECTOR_ENCODER)
        return {
            "model": model,
            "scaler": scaler,
            "status": status,
            "regime_encoder": dict(FEATURE_ENCODERS.get("regime", {})),
            "sector_encoder": dict(FEATURE_ENCODERS.get("sector", {})),
            "feature_encoders": FEATURE_ENCODERS,
        }
    except Exception as exc:
        status["message"] = f"Training failed: {exc}"
        TRAINING_STATUS = status
        return {"model": None, "scaler": None, "status": status}


def load_persisted_model() -> bool:
    """
    Restore a previously trained model from disk.

    Call this at app startup after persistent_store.pull_all().
    """
    global MODEL, SCALER, TRAINING_STATUS, REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS
    try:
        payload = _load_model()
        if payload is None:
            return False
        model = payload["model"]
        scaler = payload["scaler"]
        regime_encoder = dict(payload.get("regime_encoder", {}) or {})
        sector_encoder = dict(payload.get("sector_encoder", {}) or {})
        feature_encoders = {
            key: dict(value)
            for key, value in dict(payload.get("feature_encoders", {}) or {}).items()
            if isinstance(value, dict)
        }
        if not feature_encoders:
            feature_encoders = {"regime": regime_encoder, "sector": sector_encoder}
        with _MODEL_LOCK:
            MODEL = model
            SCALER = scaler
            REGIME_ENCODER = regime_encoder
            SECTOR_ENCODER = sector_encoder
            FEATURE_ENCODERS = _build_feature_encoders(pd.DataFrame([{column: "UNKNOWN" for column in _CATEGORICAL_FEATURE_COLUMNS}]), existing=feature_encoders)
        if model is not None and scaler is not None:
            TRAINING_STATUS["trained"] = True
            TRAINING_STATUS["message"] = "Model restored from persistent storage."
            TRAINING_STATUS["regime_encoder"] = regime_encoder
            TRAINING_STATUS["sector_encoder"] = sector_encoder
            TRAINING_STATUS["feature_encoders"] = FEATURE_ENCODERS
            TRAINING_STATUS["active_feature_count"] = len(_FEATURE_COLUMNS)
            return True
        return False
    except Exception:
        return False


def restore_learning_bundle(bundle: dict | None) -> bool:
    global MODEL, SCALER, TRAINING_STATUS, REGIME_ENCODER, SECTOR_ENCODER, FEATURE_ENCODERS
    try:
        if not isinstance(bundle, dict):
            return False
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        status = bundle.get("status")
        if model is None or scaler is None:
            return False
        regime_encoder = dict(bundle.get("regime_encoder") or {})
        sector_encoder = dict(bundle.get("sector_encoder") or {})
        feature_encoders = {
            key: dict(value)
            for key, value in dict(bundle.get("feature_encoders", {}) or {}).items()
            if isinstance(value, dict)
        }
        if not feature_encoders:
            feature_encoders = {"regime": regime_encoder, "sector": sector_encoder}
        feature_encoders = _build_feature_encoders(
            pd.DataFrame([{column: "UNKNOWN" for column in _CATEGORICAL_FEATURE_COLUMNS}]),
            existing=feature_encoders,
        )
        with _MODEL_LOCK:
            MODEL = model
            SCALER = scaler
            REGIME_ENCODER = regime_encoder
            SECTOR_ENCODER = sector_encoder
            FEATURE_ENCODERS = feature_encoders
        if isinstance(status, dict):
            status = dict(status)
            status["regime_encoder"] = dict(FEATURE_ENCODERS.get("regime", regime_encoder))
            status["sector_encoder"] = dict(FEATURE_ENCODERS.get("sector", sector_encoder))
            status["feature_encoders"] = FEATURE_ENCODERS
            status["active_feature_count"] = len(_FEATURE_COLUMNS)
            TRAINING_STATUS = status
        return True
    except Exception:
        return False


def get_training_status() -> dict:
    try:
        status = dict(TRAINING_STATUS)
        status["regime_encoder"] = dict(REGIME_ENCODER)
        status["sector_encoder"] = dict(SECTOR_ENCODER)
        status["feature_encoders"] = {key: dict(value) for key, value in FEATURE_ENCODERS.items()}
        status["active_feature_count"] = int(status.get("active_feature_count", 0) or len(_FEATURE_COLUMNS))
        status["recency_weighting_active"] = bool(status.get("recency_weighting_active", False))
        status["imported_ai_samples"] = int(status.get("imported_ai_samples", 0) or 0)
        return status
    except Exception:
        return {
            "trained": False,
            "samples": 0,
            "stock_samples": 0,
            "imported_ai_samples": 0,
            "sector_samples": 0,
            "active_feature_count": len(_FEATURE_COLUMNS),
            "recency_weighting_active": False,
            "accuracy_pct": None,
            "validation_accuracy_pct": None,
            "training_accuracy_pct": None,
            "last_trained": "",
            "source": "none",
            "message": "Training status unavailable.",
            "regime_encoder": {},
            "sector_encoder": {},
            "feature_encoders": {},
        }


def _extract_feature_dict(row: dict | pd.Series) -> dict:
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

    return {
        "prediction_score": prediction_score,
        "final_score": final_score,
        "is_bullish": _to_binary_flag(pred_bullish),
        "conviction_tier": conviction,
        "mode": _first_present(row, ["mode", "Mode", "Import Mode", "Mode ID"], "UNKNOWN"),
        "import_source": _first_present(row, ["import_source", "Import Source"], "UNKNOWN"),
        "import_category": _first_present(row, ["import_category", "Import Category"], "UNKNOWN"),
        "strategy_strip": _first_present(row, ["strategy_strip", "Strategy Strip", "Tomorrow Strip", "Strip"], "UNKNOWN"),
        "rsi": _safe_float(_first_present(row, ["rsi", "RSI"], 50.0), 50.0),
        "vol_avg_ratio": _safe_float(_first_present(row, ["vol_avg_ratio", "Vol / Avg", "Volume Ratio"], 1.0), 1.0),
        "delta_ema20_pct": _safe_float(
            _first_present(
                row,
                ["delta_ema20_pct", "Delta vs EMA20 (%)", "EMA Distance (%)", "Î” vs EMA20 (%)", "Δ vs EMA20 (%)"],
                0.0,
            ),
            0.0,
        ),
        "trap_risk": _first_present(row, ["trap_risk", "Trap Risk", "Trap Check", "Trap"], "UNKNOWN"),
        "market_bias": _first_present(row, ["market_bias", "Market Bias", "bias"], "UNKNOWN"),
        "regime": _first_present(row, ["regime", "Regime"], "UNKNOWN"),
        "sector": _first_present(row, ["sector", "Sector"], "UNKNOWN"),
    }


def _row_to_feature_frame(
    row: dict | pd.Series,
    *,
    regime_encoder: dict[str, int] | None = None,
    sector_encoder: dict[str, int] | None = None,
    feature_encoders: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    raw = pd.DataFrame([_extract_feature_dict(row)])
    return _encode_feature_frame(
        raw,
        fit=False,
        regime_encoder=regime_encoder,
        sector_encoder=sector_encoder,
        feature_encoders=feature_encoders,
    )


def predict_success(row: dict):
    with _MODEL_LOCK:
        mdl = MODEL
        scl = SCALER
        regime_encoder = dict(REGIME_ENCODER)
        sector_encoder = dict(SECTOR_ENCODER)
        feature_encoders = {key: dict(value) for key, value in FEATURE_ENCODERS.items()}
    if mdl is None or scl is None:
        return 50.0

    try:
        X = _row_to_feature_frame(
            row,
            regime_encoder=regime_encoder,
            sector_encoder=sector_encoder,
            feature_encoders=feature_encoders,
        )
        X_scaled = scl.transform(X)
        prob = mdl.predict_proba(X_scaled)[0][1]
        return round(float(prob) * 100.0, 1)
    except Exception:
        return 50.0


def batch_predict_success(df: pd.DataFrame) -> pd.Series:
    """Batch version of predict_success: encode once, transform once."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    default = pd.Series(50.0, index=df.index, dtype=float)

    with _MODEL_LOCK:
        mdl = MODEL
        scl = SCALER
        regime_encoder = dict(REGIME_ENCODER)
        sector_encoder = dict(SECTOR_ENCODER)
        feature_encoders = {key: dict(value) for key, value in FEATURE_ENCODERS.items()}
    if mdl is None or scl is None:
        return default

    try:
        rows = [_extract_feature_dict(row) for _, row in df.iterrows()]
        feat_df = pd.DataFrame(rows)
        feat_enc = _encode_feature_frame(
            feat_df,
            fit=False,
            regime_encoder=regime_encoder,
            sector_encoder=sector_encoder,
            feature_encoders=feature_encoders,
        )
        probs = mdl.predict_proba(scl.transform(feat_enc))[:, 1]
        return pd.Series([round(float(p) * 100.0, 1) for p in probs], index=df.index, dtype=float)
    except Exception:
        return default
