from __future__ import annotations

import ast
import importlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    from persistent_store import push_file as _push_file
except Exception:
    def _push_file(*a, **kw):  # type: ignore[misc]
        pass

try:
    import learning_engine as _learning_engine
except Exception:
    _learning_engine = None  # type: ignore[assignment]

if _learning_engine is not None:
    predict_success = getattr(_learning_engine, "predict_success", lambda row: 50.0)
    train_learning_model = getattr(
        _learning_engine,
        "train_learning_model",
        lambda: {"model": None, "scaler": None, "status": {"trained": False, "message": "Learning engine unavailable."}},
    )
else:
    def predict_success(row: dict) -> float:  # type: ignore[misc]
        return 50.0

    def train_learning_model():  # type: ignore[misc]
        return {"model": None, "scaler": None, "status": {"trained": False, "message": "Learning engine unavailable."}}

from prediction_feedback_store import backfill_actual_returns, feedback_summary, read_feedback_log
from sector_dynamic_weights import (
    get_dynamic_weights,
    get_signal_performance_report,
    update_signal_performance,
)
from sector_evaluation_engine import _calibration, compute_full_evaluation
from sector_prediction_tracker import backfill_outcomes, read_log as read_sector_log
from sector_regime_engine import detect_regime, regime_weight_adjustments
from sector_master import get_sector
from strategy_engines import get_engine_functions
from trade_decision_simple import apply_trade_decision_simple_any
from tomorrow_prediction_engine import get_tomorrow_prediction

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_MASTER_PREDICTIONS_PATH = _DATA_DIR / "tomorrow_master_predictions.csv"

_SESSION_STATUS_KEY = "_learning_brain_status"
_SESSION_SIGNATURE_KEY = "_learning_brain_signature"
_SESSION_PREDICTIONS_KEY = "tomorrow_predictions"
_SESSION_PREDICTIONS_MAP_KEY = "_tomorrow_predictions_map"

_MODE_FEATURES = {
    1: ["rsi", "vol_ratio", "near_high", "ret_1d", "ret_3d", "breakout", "ema_trend"],
    2: ["rsi", "vol_ratio", "ema_dist", "ret_5d", "ret_20d", "ema_trend"],
    3: ["rsi", "vol_ratio", "ema_dist", "dist_high", "ret_5d", "early_sig", "ema_trend"],
    4: ["rsi", "vol_ratio", "ema_dist", "ret_20d", "near_20h", "ema_trend"],
    5: ["rsi", "vol_ratio", "near_5h", "ret_1d", "ret_5d", "vol_spk", "ema_trend"],
    6: ["rsi", "vol_ratio", "ema_dist", "ema_slope", "ret_5d", "rsi_ctrl", "vol_ctrl"],
}


def _toast(message: str) -> None:
    try:
        st.toast(message)
    except Exception:
        return


def _now_iso() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return str(datetime.now())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if text in {"", "nan", "None", "-", "—"}:
                return default
            for token in ("%", ",", "x", "X", "×"):
                text = text.replace(token, "")
            value = text
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _plain_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(".NS", "")


def _normalise_symbol(symbol: str) -> str:
    plain = _plain_symbol(symbol)
    return f"{plain}.NS" if plain else ""


def _is_tradeable_key(key: str, value: object) -> bool:
    try:
        if not isinstance(value, pd.DataFrame) or value.empty:
            return False
        text = str(key or "").strip().upper()
        if not text or text.startswith("^") or "%" in text:
            return False
        cols = {str(col) for col in value.columns}
        return {"Open", "High", "Low", "Close", "Volume"}.issubset(cols)
    except Exception:
        return False


def _master_signature(all_data: dict[str, Any]) -> str:
    try:
        tradeable = []
        latest_bar = ""
        for key, value in dict(all_data or {}).items():
            if not _is_tradeable_key(key, value):
                continue
            tradeable.append(_plain_symbol(key))
            try:
                hist_index = pd.to_datetime(value.index, errors="coerce")
                hist_index = hist_index[~pd.isna(hist_index)]
                if len(hist_index) > 0:
                    bar_text = pd.Timestamp(hist_index.max()).date().isoformat()
                    if bar_text > latest_bar:
                        latest_bar = bar_text
            except Exception:
                continue
        tradeable = sorted(set(tradeable))
        return "|".join(
            [
                str(len(tradeable)),
                tradeable[-1] if tradeable else "none",
                latest_bar or "unknown",
                str(int((_DATA_DIR / "prediction_feedback_log.csv").stat().st_mtime)) if (_DATA_DIR / "prediction_feedback_log.csv").exists() else "0",
                str(int((_DATA_DIR / "sector_predictions.csv").stat().st_mtime)) if (_DATA_DIR / "sector_predictions.csv").exists() else "0",
            ]
        )
    except Exception:
        return f"fallback|{len(all_data or {})}"


def _save_master_predictions(df: pd.DataFrame) -> None:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_MASTER_PREDICTIONS_PATH, index=False)
        _push_file(_MASTER_PREDICTIONS_PATH)
    except Exception:
        return


def _deserialize_prediction_row(row: dict) -> dict:
    try:
        out = dict(row or {})
        for key in ("signals", "weights", "regime_snapshot", "indicators"):
            value = out.get(key)
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                try:
                    out[key] = ast.literal_eval(value)
                except Exception:
                    continue
        return out
    except Exception:
        return dict(row or {})


def load_master_predictions() -> pd.DataFrame:
    try:
        if _MASTER_PREDICTIONS_PATH.exists():
            return pd.read_csv(_MASTER_PREDICTIONS_PATH)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_master_predictions_frame() -> pd.DataFrame:
    try:
        cached = st.session_state.get(_SESSION_PREDICTIONS_KEY)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached.copy()
        df = load_master_predictions()
        if not df.empty:
            st.session_state[_SESSION_PREDICTIONS_KEY] = df.copy()
            st.session_state[_SESSION_PREDICTIONS_MAP_KEY] = {
                _plain_symbol(str(row.get("ticker", ""))): _deserialize_prediction_row(dict(row))
                for _, row in df.iterrows()
            }
        return df
    except Exception:
        return pd.DataFrame()


def get_cached_prediction(ticker: str) -> dict:
    try:
        plain = _plain_symbol(ticker)
        pred_map = st.session_state.get(_SESSION_PREDICTIONS_MAP_KEY)
        if isinstance(pred_map, dict) and plain in pred_map:
            return _deserialize_prediction_row(dict(pred_map.get(plain) or {}))

        df = get_master_predictions_frame()
        if df.empty or "ticker" not in df.columns:
            return {}
        match = df[df["ticker"].astype(str).str.upper().str.replace(".NS", "", regex=False) == plain]
        if match.empty:
            return {}
        row = _deserialize_prediction_row(dict(match.iloc[0].to_dict()))
        pred_map = st.session_state.get(_SESSION_PREDICTIONS_MAP_KEY, {})
        if isinstance(pred_map, dict):
            pred_map[plain] = row
            st.session_state[_SESSION_PREDICTIONS_MAP_KEY] = pred_map
        return row
    except Exception:
        return {}


def summarize_cached_predictions(tickers: list[str]) -> dict:
    try:
        predictions = []
        for ticker in list(tickers or []):
            pred = get_cached_prediction(ticker)
            if pred:
                predictions.append(pred)
        if not predictions:
            return {
                "tickers": [],
                "predictions": [],
                "direction": "Sideways",
                "confidence": 0.0,
                "score": 50.0,
                "risk": "HIGH",
                "action": "Wait",
                "key_signal": "data_unavailable",
            }

        scores = np.array([_safe_float(pred.get("raw_score", pred.get("score", 50.0)), 50.0) for pred in predictions], dtype=float)
        confs = np.array([_safe_float(pred.get("confidence"), 0.0) for pred in predictions], dtype=float)
        avg_score = float(np.nanmean(scores)) if len(scores) else 50.0
        avg_conf = float(np.nanmean(confs)) if len(confs) else 0.0

        if avg_score >= 55.0:
            direction = "Bullish"
        elif avg_score <= 45.0:
            direction = "Bearish"
        else:
            direction = "Sideways"

        risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        risk = max(
            (str(pred.get("risk", "MEDIUM")).upper() for pred in predictions),
            key=lambda value: risk_rank.get(value, 1),
            default="MEDIUM",
        )
        key_signal = max(
            (str(pred.get("key_signal", "momentum")) for pred in predictions),
            key=lambda signal: sum(1 for pred in predictions if str(pred.get("key_signal", "")) == signal),
            default="momentum",
        )
        action_votes = [str(pred.get("action", "Wait")) for pred in predictions]
        if sum("Buy Tomorrow" in vote for vote in action_votes) >= max(1, len(predictions) // 2 + 1) and risk != "HIGH":
            action = "Buy Tomorrow"
        elif sum("Avoid" in vote for vote in action_votes) >= max(1, len(predictions) // 2 + 1):
            action = "Avoid"
        elif sum("Watch" in vote for vote in action_votes) > 0:
            action = "Watch"
        else:
            action = "Wait"

        return {
            "tickers": [_plain_symbol(str(pred.get("ticker", ""))) for pred in predictions],
            "predictions": predictions,
            "direction": direction,
            "confidence": round(_clamp(avg_conf), 1),
            "score": round(_clamp(avg_score), 1),
            "risk": risk,
            "action": action,
            "key_signal": key_signal,
        }
    except Exception:
        return {
            "tickers": [],
            "predictions": [],
            "direction": "Sideways",
            "confidence": 0.0,
            "score": 50.0,
            "risk": "HIGH",
            "action": "Wait",
            "key_signal": "engine_error",
        }


def _close_feedback_loop(all_data: dict[str, Any]) -> dict:
    result = {
        "filled_stock": 0,
        "filled_sector": 0,
        "pending_stock": 0,
        "pending_sector": 0,
        "message": "✅ Closed feedback loop: 0 outcomes filled",
    }
    try:
        stock_before = read_feedback_log()
        if not stock_before.empty:
            stock_pending = stock_before[
                stock_before["actual_next_return_pct"].astype(str).str.strip().isin(["", "nan", "None"])
                | (~stock_before["correct"].astype(str).str.strip().isin(["True", "False"]))
            ]
            result["pending_stock"] = int(len(stock_pending))
        sector_before = read_sector_log()
        if not sector_before.empty:
            sector_pending = sector_before[
                sector_before["return_pct"].astype(str).str.strip().isin(["", "nan", "None"])
                | (~sector_before["correct"].astype(str).str.strip().isin(["True", "False"]))
            ]
            result["pending_sector"] = int(len(sector_pending))

        result["filled_stock"] = int(backfill_actual_returns(all_data))
        result["filled_sector"] = int(backfill_outcomes(all_data))
        total_filled = result["filled_stock"] + result["filled_sector"]
        result["message"] = f"✅ Closed feedback loop: {total_filled} outcomes filled"
        return result
    except Exception as exc:
        result["message"] = f"✅ Closed feedback loop: 0 outcomes filled ({exc})"
        return result


def _apply_time_travel_cutoff(df: pd.DataFrame | None) -> pd.DataFrame | None:
    try:
        from time_travel_engine import apply_time_travel_cutoff

        return apply_time_travel_cutoff(df)
    except Exception:
        return df


def _weighted_recent_rows(data: pd.DataFrame, recent_days: int = 60, boost_days: int = 30) -> pd.DataFrame:
    try:
        if data is None or data.empty:
            return pd.DataFrame()
        out = data.copy()
        if "_dt" not in out.columns:
            out["_dt"] = pd.to_datetime(out.index, errors="coerce")
        out["_dt"] = pd.to_datetime(out["_dt"], errors="coerce")
        out = out.dropna(subset=["_dt"]).copy()
        if out.empty:
            return data
        max_dt = out["_dt"].max()
        out = out[out["_dt"] >= (max_dt - pd.Timedelta(days=recent_days))].copy()
        recent = out[out["_dt"] >= (max_dt - pd.Timedelta(days=boost_days))].copy()
        if recent.empty:
            return out
        return pd.concat([out, recent], ignore_index=True)
    except Exception:
        return data


def _resolve_training_history(all_data: dict[str, Any], ticker: str) -> pd.DataFrame | None:
    try:
        store = dict(all_data or {})
        for key in (_normalise_symbol(ticker), _plain_symbol(ticker)):
            hist = store.get(key)
            if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
                continue
            frame = hist.copy()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            required = ["Open", "High", "Low", "Close", "Volume"]
            if not set(required).issubset({str(col) for col in frame.columns}):
                continue
            frame = frame[required].copy()
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            frame = frame[~frame.index.isna()].sort_index()
            if getattr(frame.index, "tz", None) is not None:
                frame.index = frame.index.tz_localize(None)
            for col in required:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna(subset=["Close", "Volume"])
            frame = _apply_time_travel_cutoff(frame)
            if frame is not None and not frame.empty:
                return frame
        return None
    except Exception:
        return None


def _fit_mode_model(mode: int, all_data: dict[str, Any]) -> dict:
    status = {
        "mode": mode,
        "trained": False,
        "samples": 0,
        "accuracy_pct": None,
        "suppressed": False,
        "suppression_factor": 1.0,
        "message": "Not trained",
    }
    try:
        module = importlib.import_module(f"strategy_engines.mode{mode}_engine")
        if not getattr(module, "SKLEARN_OK", False):
            status["message"] = "sklearn unavailable"
            return status

        build_fn = getattr(module, f"_build_features_mode{mode}", None)
        tickers = list(getattr(module, "_TRAIN_TICKERS", []) or [])
        feat_cols = list(_MODE_FEATURES.get(mode, []))
        if build_fn is None or not tickers or not feat_cols:
            status["message"] = "mode interface incomplete"
            return status

        rows = []
        for ticker in tickers:
            try:
                hist = _resolve_training_history(all_data, ticker)
                if hist is None or hist.empty:
                    continue
                built = build_fn(hist["Close"], hist["Volume"])
                if built is None or built.empty:
                    continue
                built = built.copy()
                built["_dt"] = pd.to_datetime(built.index, errors="coerce")
                rows.append(built)
            except Exception:
                continue

        if not rows:
            status["message"] = "no training rows"
            return status

        data = pd.concat(rows, ignore_index=True)
        data = _weighted_recent_rows(data)
        status["samples"] = int(len(data))
        if len(data) < 30 or data["target"].nunique(dropna=True) < 2:
            status["message"] = "insufficient weighted rows"
            return status

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = data[feat_cols].astype(float).fillna(0.0).values
        y = data["target"].astype(int).values

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=mode,
                stratify=y,
            )
        except Exception:
            split = max(1, int(len(X) * 0.8))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        model = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", random_state=mode)
        model.fit(X_tr_scaled, y_tr)
        accuracy = float(model.score(X_te_scaled, y_te) * 100.0)

        lock = getattr(module, "_LOCK", None)
        if lock is not None:
            with lock:
                module._MODEL = model
                module._SCALER = scaler
        else:
            module._MODEL = model
            module._SCALER = scaler

        status["trained"] = True
        status["accuracy_pct"] = round(accuracy, 2)
        status["message"] = "trained"
        return status
    except Exception as exc:
        status["message"] = str(exc)
        return status


def _mode_feedback_adjustments(feedback_df: pd.DataFrame, mode: int) -> dict:
    out = {
        "mode": mode,
        "validated": 0,
        "accuracy_pct": None,
        "suppressed": False,
        "suppression_factor": 1.0,
        "recent_accuracy_pct": None,
    }
    try:
        if feedback_df is None or feedback_df.empty:
            return out
        rows = feedback_df[feedback_df["mode"].astype(str).str.strip() == str(mode)].copy()
        rows = rows[rows["correct"].astype(str).isin(["True", "False"])].copy()
        if rows.empty:
            return out
        out["validated"] = int(len(rows))
        out["accuracy_pct"] = round(float((rows["correct"] == "True").mean() * 100.0), 2)
        recent = rows.tail(20)
        if not recent.empty:
            recent_acc = float((recent["correct"] == "True").mean() * 100.0)
            out["recent_accuracy_pct"] = round(recent_acc, 2)
            if len(recent) >= 20 and recent_acc < 50.0:
                out["suppressed"] = True
                out["suppression_factor"] = 0.5
        return out
    except Exception:
        return out


def _retrain_mode_models(feedback_df: pd.DataFrame, all_data: dict[str, Any]) -> dict:
    summary = {"by_mode": {}, "best_mode": None, "worst_mode": None, "message": ""}
    try:
        parts = []
        for mode in range(1, 7):
            trained = _fit_mode_model(mode, all_data)
            feedback_adj = _mode_feedback_adjustments(feedback_df, mode)
            merged = {**trained, **feedback_adj}
            accuracy = merged.get("accuracy_pct")
            if accuracy is None:
                accuracy = trained.get("accuracy_pct")
                merged["accuracy_pct"] = accuracy
            suppression = float(merged.get("suppression_factor", 1.0) or 1.0)
            merged["ensemble_weight"] = max(0.15, (_safe_float(accuracy, 50.0) / 100.0) * suppression)
            summary["by_mode"][mode] = merged
            if accuracy is not None:
                parts.append((mode, float(accuracy)))

        if parts:
            best = max(parts, key=lambda item: item[1])
            worst = min(parts, key=lambda item: item[1])
            summary["best_mode"] = {"mode": best[0], "accuracy_pct": round(best[1], 2)}
            summary["worst_mode"] = {"mode": worst[0], "accuracy_pct": round(worst[1], 2)}
            summary["message"] = " | ".join([f"Mode{mode}={acc:.1f}%" for mode, acc in parts])
        return summary
    except Exception as exc:
        summary["message"] = str(exc)
        return summary


def _sector_accuracy_snapshot(sector_log: pd.DataFrame) -> dict:
    out = {"by_sector": {}, "best_sector": "", "worst_sector": ""}
    try:
        if sector_log is None or sector_log.empty:
            return out
        val = sector_log[sector_log["correct"].astype(str).isin(["True", "False"])].copy()
        if val.empty:
            return out
        by_sector = {}
        for sector, grp in val.groupby("sector"):
            by_sector[str(sector)] = round(float((grp["correct"] == "True").mean() * 100.0), 2)
        out["by_sector"] = by_sector
        if by_sector:
            out["best_sector"] = max(by_sector, key=by_sector.get)
            out["worst_sector"] = min(by_sector, key=by_sector.get)
        return out
    except Exception:
        return out


def _update_dynamic_signal_weights(sector_log: pd.DataFrame) -> dict:
    result = {"processed": 0, "report": pd.DataFrame(), "message": "", "top_signal": {}, "weakest_signal": {}}
    try:
        before = get_signal_performance_report()
        val = sector_log[sector_log["correct"].astype(str).isin(["True", "False"])].copy() if sector_log is not None else pd.DataFrame()
        if not val.empty:
            val["predicted_at"] = pd.to_datetime(val["predicted_at"], errors="coerce")
            latest = val["predicted_at"].max()
            window = val[val["predicted_at"] >= (latest - pd.Timedelta(days=60))].copy() if pd.notna(latest) else val.copy()
            if not window.empty and "predicted_at" in window.columns:
                recent = window[window["predicted_at"] >= (latest - pd.Timedelta(days=30))].copy() if pd.notna(latest) else pd.DataFrame()
                if not recent.empty:
                    window = pd.concat([window, recent], ignore_index=True)
            result["processed"] = int(update_signal_performance(window))

        report = get_signal_performance_report()
        result["report"] = report
        if not report.empty:
            top_row = report.sort_values("Dynamic Weight", ascending=False).iloc[0]
            weak_row = report.sort_values("Dynamic Weight", ascending=True).iloc[0]
            result["top_signal"] = {
                "signal": str(top_row["Signal"]),
                "weight_pct": _safe_float(top_row["Dynamic Weight"]),
                "delta_pct": _safe_float(top_row["Δ Weight"]),
            }
            result["weakest_signal"] = {
                "signal": str(weak_row["Signal"]),
                "weight_pct": _safe_float(weak_row["Dynamic Weight"]),
                "delta_pct": _safe_float(weak_row["Δ Weight"]),
            }
            result["message"] = (
                f"📊 Weights updated: {result['top_signal']['signal']} "
                f"{result['top_signal']['weight_pct']:.1f}% | "
                f"{result['weakest_signal']['signal']} {result['weakest_signal']['weight_pct']:.1f}%"
            )
        elif not before.empty:
            result["message"] = "📊 Weights refreshed from existing performance history"
        else:
            result["message"] = "📊 Weights updated: no validated signal history yet"
        return result
    except Exception as exc:
        result["message"] = f"📊 Weights update skipped: {exc}"
        return result


def _calibration_status(sector_log: pd.DataFrame) -> dict:
    out = {
        "error_pct": 0.0,
        "factor": 1.0,
        "bucket_count": 0,
        "avg_confidence": 0.0,
        "actual_accuracy": 0.0,
        "message": "🎯 Calibration error: 0.0% | Factor applied: 1.00x",
    }
    try:
        if sector_log is None or sector_log.empty:
            return out
        val = sector_log[sector_log["correct"].astype(str).isin(["True", "False"])].copy()
        if val.empty:
            return out
        buckets, error = _calibration(val)
        total = sum(bucket.count for bucket in buckets)
        if total <= 0:
            return out
        avg_conf = sum(bucket.avg_confidence * bucket.count for bucket in buckets) / total
        actual_acc = sum(bucket.actual_accuracy * bucket.count for bucket in buckets) / total
        factor = actual_acc / max(avg_conf, 1e-9) if avg_conf > 0 else 1.0
        out["error_pct"] = round(float(error), 2)
        out["factor"] = round(float(np.clip(factor, 0.65, 1.20)), 4)
        out["bucket_count"] = len(buckets)
        out["avg_confidence"] = round(float(avg_conf), 2)
        out["actual_accuracy"] = round(float(actual_acc), 2)
        out["message"] = f"🎯 Calibration error: {out['error_pct']:.1f}% | Factor applied: {out['factor']:.2f}x"
        return out
    except Exception as exc:
        out["message"] = f"🎯 Calibration error: 0.0% | Factor applied: 1.00x ({exc})"
        return out


def _mode_ensemble_components(row: dict, mode_status: dict) -> dict:
    scores = []
    probs = []
    for mode in range(1, 7):
        try:
            score_fn, _, ml_fn, _ = get_engine_functions(mode)
            score, _ = score_fn(row)
            mode_meta = mode_status.get("by_mode", {}).get(mode, {}) if isinstance(mode_status, dict) else {}
            weight = _safe_float(mode_meta.get("ensemble_weight"), 0.5)
            if bool(mode_meta.get("trained")):
                prob = _safe_float(ml_fn(row), 50.0)
            else:
                prob = 50.0
            scores.append((float(score), weight))
            probs.append((prob, weight))
        except Exception:
            continue

    def _weighted_average(items: list[tuple[float, float]], default: float = 50.0) -> float:
        total_w = sum(weight for _, weight in items)
        if total_w <= 0:
            return default
        return sum(value * weight for value, weight in items) / total_w

    return {
        "mode_score": round(_weighted_average(scores), 2),
        "ml_probability": round(_weighted_average(probs), 2),
    }


def _risk_bump(risk: str) -> str:
    text = str(risk or "MEDIUM").upper()
    if text == "LOW":
        return "MEDIUM"
    return "HIGH"


def _regime_snapshot_from_base(base: dict, current_regime: str) -> dict:
    try:
        indicators = dict(base.get("indicators") or {})
        ema20 = _safe_float(indicators.get("EMA20"), 0.0)
        ema50 = _safe_float(indicators.get("EMA50"), 0.0)
        regime_key = str(current_regime or base.get("regime", "UNKNOWN") or "UNKNOWN").strip().upper()
        mapping = {
            "TRENDING_UP": {
                "regime": "Trending Up",
                "description": "Trend-following backdrop with higher bullish follow-through.",
                "color": "#00ff88",
                "emoji": "▲",
            },
            "TRENDING_DOWN": {
                "regime": "Trending Down",
                "description": "Risk-off backdrop. Bearish setups deserve more respect.",
                "color": "#ff3b5c",
                "emoji": "▼",
            },
            "RANGE_BOUND": {
                "regime": "Range Bound",
                "description": "Mean reversion regime. Breakouts need extra confirmation.",
                "color": "#8ab4d8",
                "emoji": "◆",
            },
            "HIGH_VOLATILITY": {
                "regime": "High Volatility",
                "description": "Wide ranges reduce forecast stability and widen risk.",
                "color": "#f0b429",
                "emoji": "⚡",
            },
        }
        snap = dict(
            mapping.get(
                regime_key,
                {
                    "regime": str(current_regime or "Unknown").replace("_", " ").title(),
                    "description": "",
                    "color": "#8ab4d8",
                    "emoji": "•",
                },
            )
        )
        snap["ema_aligned"] = ema20 >= ema50
        return snap
    except Exception:
        return {
            "regime": str(current_regime or "Unknown").replace("_", " ").title(),
            "description": "",
            "color": "#8ab4d8",
            "emoji": "•",
            "ema_aligned": False,
        }


def _build_master_prediction(ticker: str, all_data: dict[str, Any], mode_status: dict, calibration: dict, sector_accuracy: dict, regime_state) -> dict | None:
    try:
        base = get_tomorrow_prediction(ticker, all_data, 2)
        if not isinstance(base, dict) or not base or _safe_float(base.get("confidence"), 0.0) <= 0:
            return None

        indicators = dict(base.get("indicators") or {})
        plain = _plain_symbol(str(base.get("ticker", ticker)))
        sector = get_sector(plain) or "UNKNOWN"
        row = {
            "Ticker": plain,
            "Price (₹)": _safe_float(indicators.get("Close"), 0.0),
            "RSI": _safe_float(indicators.get("RSI"), 50.0),
            "Vol / Avg": _safe_float(indicators.get("Vol / Avg"), 1.0),
            "Δ vs EMA20 (%)": _safe_float(indicators.get("Δ vs EMA20 (%)"), 0.0),
            "Δ vs 20D High (%)": _safe_float(indicators.get("Δ vs 20D High (%)"), 0.0),
            "5D Return (%)": _safe_float(indicators.get("5D Return (%)"), 0.0),
            "20D Return (%)": _safe_float(indicators.get("20D Return (%)"), 0.0),
            "EMA 20": _safe_float(indicators.get("EMA20"), 0.0),
            "EMA 50": _safe_float(indicators.get("EMA50"), 0.0),
            "Prediction Score": _safe_float(base.get("score"), 50.0),
            "Final Score": _safe_float(base.get("confidence"), 50.0),
            "Trap Risk": str(base.get("risk", "MEDIUM") or "MEDIUM"),
            "Sector": sector,
            "regime": str(base.get("regime", getattr(regime_state, "regime", "UNKNOWN"))),
            "conviction_tier": "High" if _safe_float(base.get("confidence"), 0.0) >= 70 else "Medium" if _safe_float(base.get("confidence"), 0.0) >= 55 else "Low",
        }

        ensemble = _mode_ensemble_components(row, mode_status)
        pred_bullish = 1 if ensemble["mode_score"] >= 50.0 else 0
        meta_prob = _safe_float(
            predict_success(
                {
                    "Prediction Score": ensemble["mode_score"],
                    "Final Score": ensemble["ml_probability"],
                    "pred_bullish": pred_bullish,
                    "conviction_tier": row["conviction_tier"],
                    "regime": row["regime"],
                    "sector": sector,
                }
            ),
            50.0,
        )

        current_regime = str(getattr(regime_state, "regime", "RANGE_BOUND") or "RANGE_BOUND")
        if current_regime == "TRENDING_UP":
            regime_fit = 100.0 if ensemble["mode_score"] >= 55.0 else 45.0
        elif current_regime == "TRENDING_DOWN":
            regime_fit = 100.0 if ensemble["mode_score"] <= 45.0 else 45.0
        elif current_regime == "RANGE_BOUND":
            regime_fit = 100.0 if 45.0 <= ensemble["mode_score"] <= 55.0 else 55.0
        else:
            regime_fit = 60.0

        sector_acc = _safe_float(sector_accuracy.get("by_sector", {}).get(sector), 55.0)
        raw_score = _clamp(
            (
                0.26 * ensemble["mode_score"]
                + 0.20 * ensemble["ml_probability"]
                + 0.18 * meta_prob
                + 0.14 * _safe_float(base.get("score"), 50.0)
                + 0.10 * _safe_float(base.get("mtf_score"), 50.0)
                + 0.06 * regime_fit
                + 0.06 * sector_acc
            )
        )

        if raw_score >= 56.0:
            direction = "Bullish"
        elif raw_score <= 44.0:
            direction = "Bearish"
        else:
            direction = "Sideways"

        base_conf = (
            0.30 * _safe_float(base.get("confidence"), 0.0)
            + 0.20 * abs(ensemble["mode_score"] - 50.0) * 2.0
            + 0.15 * abs(ensemble["ml_probability"] - 50.0) * 2.0
            + 0.15 * meta_prob
            + 0.10 * _safe_float(base.get("mtf_score"), 50.0)
            + 0.10 * sector_acc
        )
        sector_factor = 1.08 if sector_acc > 65.0 else 0.88 if sector_acc < 48.0 else 1.0
        confidence = _clamp(base_conf * _safe_float(calibration.get("factor"), 1.0) * sector_factor)

        if _safe_float(indicators.get("RSI"), 50.0) >= 68.0 and _safe_float(indicators.get("Vol / Avg"), 1.0) < 1.1 and _safe_float(indicators.get("Δ vs EMA20 (%)"), 0.0) > 3.5:
            confidence = _clamp(confidence - 5.0)
            base["risk"] = _risk_bump(str(base.get("risk", "MEDIUM")))

        action_frame = apply_trade_decision_simple_any(
            pd.DataFrame(
                [
                    {
                        "RSI": row["RSI"],
                        "Vol / Avg": row["Vol / Avg"],
                        "Prediction Score": raw_score,
                        "Final Score": confidence,
                        "Δ vs EMA20 (%)": row["Δ vs EMA20 (%)"],
                        "Trap Risk": str(base.get("risk", "MEDIUM")),
                        "Meta Prob": meta_prob,
                        "Calibrated Confidence": confidence,
                        "Regime Fit": regime_fit,
                        "Accuracy History": sector_acc,
                    }
                ]
            )
        )
        action = str(action_frame.iloc[0].get("Action", "Wait") if not action_frame.empty else "Wait")
        hold_days = str(action_frame.iloc[0].get("Hold Days", "-") if not action_frame.empty else "-").replace("—", "-")
        if direction == "Bearish":
            action = "🔴 Avoid"
            hold_days = "-"
        elif direction == "Sideways" and "Buy Tomorrow" in action:
            action = "🟡 Watch"

        return {
            "ticker": plain,
            "direction": direction,
            "confidence": round(confidence, 2),
            "raw_score": round(raw_score, 2),
            "score": round(raw_score, 2),
            "action": action,
            "hold_days": hold_days,
            "risk": str(base.get("risk", "MEDIUM")),
            "regime": current_regime,
            "regime_snapshot": _regime_snapshot_from_base(base, current_regime),
            "key_signal": str(base.get("key_signal", "momentum")),
            "sector": sector,
            "mode_score": ensemble["mode_score"],
            "ml_probability": ensemble["ml_probability"],
            "meta_model_output": round(meta_prob, 2),
            "mtf_alignment": _safe_float(base.get("mtf_score"), 50.0),
            "regime_fit": round(regime_fit, 2),
            "sector_accuracy": round(sector_acc, 2),
            "signals": dict(base.get("signals") or {}),
            "weights": dict(base.get("weights") or {}),
            "atr": round(_safe_float(base.get("atr"), 0.0), 4),
            "label_tag": str(base.get("label_tag", "") or ""),
            "indicators": dict(indicators),
            "computed_at": _now_iso(),
        }
    except Exception:
        return None


def _build_master_predictions(all_data: dict[str, Any], mode_status: dict, calibration: dict, sector_accuracy: dict, regime_state) -> dict:
    result = {"count": 0, "df": pd.DataFrame(), "message": "🔮 Tomorrow predictions: 0 tickers computed"}
    try:
        rows = []
        seen = set()
        for key, value in dict(all_data or {}).items():
            if not _is_tradeable_key(key, value):
                continue
            plain = _plain_symbol(key)
            if not plain or plain in seen:
                continue
            seen.add(plain)
            pred = _build_master_prediction(plain, all_data, mode_status, calibration, sector_accuracy, regime_state)
            if pred:
                rows.append(pred)

        if not rows:
            return result

        df = pd.DataFrame(rows).sort_values(["confidence", "raw_score"], ascending=[False, False]).reset_index(drop=True)
        _save_master_predictions(df)
        st.session_state[_SESSION_PREDICTIONS_KEY] = df.copy()
        st.session_state[_SESSION_PREDICTIONS_MAP_KEY] = {
            _plain_symbol(str(row.get("ticker", ""))): dict(row)
            for _, row in df.iterrows()
        }
        st.session_state["tomorrow_predictions"] = df.copy()
        result["count"] = int(len(df))
        result["df"] = df
        result["message"] = f"🔮 Tomorrow predictions: {len(df)} tickers computed"
        return result
    except Exception as exc:
        result["message"] = f"🔮 Tomorrow predictions: 0 tickers computed ({exc})"
        return result


def run_learning_cycle(all_data: dict[str, Any], *, force: bool = False) -> dict:
    try:
        signature = _master_signature(all_data)
        cached = st.session_state.get(_SESSION_STATUS_KEY)
        if not force and st.session_state.get(_SESSION_SIGNATURE_KEY) == signature and isinstance(cached, dict):
            return cached

        status: dict[str, Any] = {
            "started_at": _now_iso(),
            "feedback": {},
            "feedback_summary": {},
            "meta_model": {},
            "mode_models": {},
            "sector_model": {},
            "weights": {},
            "regime": {},
            "calibration": {},
            "predictions": {},
            "evaluation": {},
            "completed_at": "",
        }

        status["feedback"] = _close_feedback_loop(all_data)
        _toast(status["feedback"]["message"])

        status["feedback_summary"] = feedback_summary()
        feedback_df = read_feedback_log()
        sector_log = read_sector_log()

        meta_model = train_learning_model()
        status["meta_model"] = meta_model.get("status", {}) if isinstance(meta_model, dict) else {}
        if isinstance(meta_model, dict) and meta_model.get("model") is not None and meta_model.get("scaler") is not None:
            st.session_state["_learning_model_bundle"] = meta_model
        meta_acc = _safe_float(status["meta_model"].get("accuracy_pct"), 0.0)
        _toast(f"🧠 Meta model retrained: {meta_acc:.1f}%")

        status["mode_models"] = _retrain_mode_models(feedback_df, all_data)
        if status["mode_models"].get("message"):
            _toast(f"🧠 Mode models retrained: {status['mode_models']['message']}")

        status["sector_model"] = _sector_accuracy_snapshot(sector_log)

        status["weights"] = _update_dynamic_signal_weights(sector_log)
        _toast(status["weights"].get("message", "📊 Weights updated"))

        regime_state = detect_regime(all_data)
        current_regime = getattr(regime_state, "regime", "RANGE_BOUND")
        status["regime"] = {
            "regime": current_regime,
            "confidence": round(_safe_float(getattr(regime_state, "confidence", 50.0)), 2),
            "realized_vol": round(_safe_float(getattr(regime_state, "realized_vol", 0.0)), 2),
            "adx_proxy": round(_safe_float(getattr(regime_state, "adx_proxy", 0.0)), 2),
            "adjustments": regime_weight_adjustments(current_regime),
            "message": (
                f"📈 Regime: {current_regime} "
                f"(conf: {_safe_float(getattr(regime_state, 'confidence', 50.0), 50.0):.0f}%) — weights adjusted"
            ),
        }
        st.session_state["current_regime"] = current_regime
        _toast(status["regime"]["message"])

        status["calibration"] = _calibration_status(sector_log)
        st.session_state["confidence_correction_factor"] = status["calibration"].get("factor", 1.0)
        _toast(status["calibration"]["message"])

        status["evaluation"] = {
            "report": compute_full_evaluation(),
        }

        status["predictions"] = _build_master_predictions(
            all_data,
            status["mode_models"],
            status["calibration"],
            status["sector_model"],
            regime_state,
        )
        _toast(status["predictions"].get("message", "🔮 Tomorrow predictions ready"))

        status["completed_at"] = _now_iso()
        final_signature = _master_signature(all_data)
        st.session_state[_SESSION_STATUS_KEY] = status
        st.session_state[_SESSION_SIGNATURE_KEY] = final_signature
        st.session_state["learning_cycle_status"] = status
        return status
    except Exception as exc:
        fallback = {
            "started_at": _now_iso(),
            "completed_at": _now_iso(),
            "error": str(exc),
            "feedback": {},
            "feedback_summary": {},
            "meta_model": {},
            "mode_models": {},
            "sector_model": {},
            "weights": {},
            "regime": {},
            "calibration": {},
            "predictions": {},
            "evaluation": {},
        }
        st.session_state[_SESSION_STATUS_KEY] = fallback
        st.session_state["learning_cycle_status"] = fallback
        return fallback
