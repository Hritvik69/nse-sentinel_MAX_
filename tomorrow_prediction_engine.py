from __future__ import annotations

import importlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    from enhanced_logic_engine import apply_enhanced_logic
except Exception:
    def apply_enhanced_logic(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return df

try:
    from learning_engine import predict_success
except Exception:
    def predict_success(row: dict) -> float:  # type: ignore[misc]
        return 50.0

try:
    from sector_dynamic_weights import get_dynamic_weights
except Exception:
    def get_dynamic_weights(sector=None, regime=None, regime_adjustments=None):  # type: ignore[misc]
        return {
            "ema_slope": 0.12,
            "price_vs_ema": 0.12,
            "candle_direction": 0.08,
            "body_strength": 0.08,
            "volume_confirm": 0.10,
            "volatility": 0.06,
            "momentum": 0.12,
            "sector_strength": 0.12,
            "bullish_pct": 0.10,
            "money_flow": 0.05,
            "participation": 0.05,
        }

try:
    from sector_mtf_engine import compute_mtf_alignment
except Exception:
    compute_mtf_alignment = None  # type: ignore[assignment]

try:
    from sector_regime_engine import detect_regime, regime_weight_adjustments
except Exception:
    detect_regime = None  # type: ignore[assignment]

    def regime_weight_adjustments(regime: str) -> dict[str, float]:  # type: ignore[misc]
        return {}

try:
    from trade_decision_simple import apply_trade_decision_simple_any
except Exception:
    def apply_trade_decision_simple_any(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return df

try:
    from strategy_engines import get_engine_functions
except Exception:
    get_engine_functions = None  # type: ignore[assignment]

try:
    from strategy_engines._engine_utils import ALL_DATA as _GLOBAL_ALL_DATA
except Exception:
    _GLOBAL_ALL_DATA = {}

_IMPORTED_PERFORMANCE_CACHE: dict[str, object] = {"summary": None}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if text == "" or text.lower() in {"nan", "none"}:
                return default
            value = text
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _norm_score(value: float, low: float, high: float) -> float:
    try:
        if high == low:
            return 50.0
        return _clamp((float(value) - low) / (high - low) * 100.0)
    except Exception:
        return 50.0


def _normalise_ticker(ticker: str) -> str:
    text = str(ticker or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".NS") else f"{text}.NS"


def _plain_ticker(ticker: str) -> str:
    return _normalise_ticker(ticker).replace(".NS", "")


def _get_time_travel_cutoff():
    try:
        from feature_data_manager import get_time_travel_date
        cutoff = get_time_travel_date()
        if cutoff is not None:
            return pd.to_datetime(cutoff).date()
    except Exception:
        pass
    try:
        from time_travel_engine import get_reference_date
        cutoff = get_reference_date()
        if cutoff is not None:
            return pd.to_datetime(cutoff).date()
    except Exception:
        pass
    return None


def _apply_history_cutoff(hist: pd.DataFrame | None, cutoff_date=None, min_rows: int = 30) -> pd.DataFrame | None:
    if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty or cutoff_date is None:
        return hist
    try:
        cutoff = pd.to_datetime(cutoff_date).date()
        idx_dates = pd.to_datetime(hist.index, errors="coerce").date
        trimmed = hist.loc[idx_dates <= cutoff].copy()
        if len(trimmed) < min_rows:
            return None
        return trimmed
    except Exception:
        return hist


def _resolve_all_data(all_data: dict | None) -> dict:
    if isinstance(all_data, dict) and all_data:
        return all_data
    return _GLOBAL_ALL_DATA if isinstance(_GLOBAL_ALL_DATA, dict) else {}


def _resolve_regime_store(all_data: dict | None, cutoff_date=None) -> dict:
    store = _resolve_all_data(all_data)
    if cutoff_date is None or not isinstance(store, dict):
        return store
    trimmed_store: dict = {}
    for key, df in store.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            trimmed_store[key] = df
            continue
        trimmed = _apply_history_cutoff(df, cutoff_date, min_rows=5)
        if isinstance(trimmed, pd.DataFrame) and not trimmed.empty:
            trimmed_store[key] = trimmed
    return trimmed_store


def _resolve_history(ticker: str, all_data: dict | None, cutoff_date=None) -> pd.DataFrame | None:
    try:
        store = _resolve_all_data(all_data)
        ticker_ns = _normalise_ticker(ticker)
        if not ticker_ns:
            return None
        for key in (ticker_ns, ticker_ns.replace(".NS", "")):
            df = store.get(key)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            hist = df.copy()
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(hist.columns):
                continue
            hist = hist[list(required)].copy()
            hist.index = pd.to_datetime(hist.index, errors="coerce")
            hist = hist[~hist.index.isna()].sort_index()
            if getattr(hist.index, "tz", None) is not None:
                hist.index = hist.index.tz_localize(None)
            for col in required:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
            hist = hist.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            hist = _apply_history_cutoff(hist, cutoff_date, min_rows=30)
            if hist is not None and len(hist) >= 30:
                return hist
        return None
    except Exception:
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _resolve_mode(mode: int | str | None) -> int:
    if isinstance(mode, int) and mode in {1, 2, 3, 4, 5, 6, 7}:
        return mode
    text = str(mode or "").strip().lower()
    mapping = {
        "momentum": 1,
        "balanced": 2,
        "relaxed": 3,
        "institutional": 4,
        "intraday": 5,
        "swing": 6,
        "chart": 2,
        "prediction_chart": 2,
        "sector": 2,
        "sector_dashboard": 2,
        "sector_prediction": 2,
        "aura": 6,
        "stock_aura": 6,
        "tomorrow_picks": 5,
        "momentum_sr": 7,
        "mode7": 7,
    }
    return mapping.get(text, 2)


def _strategy_strip_for_mode(mode) -> str:
    mode_int = _resolve_mode(mode)
    return {
        1: "Momentum",
        3: "Relax",
        5: "Intraday",
        6: "Swing",
        7: "Momentum",
    }.get(mode_int, "UNKNOWN")


def _normalise_bucket_text(value: object) -> str:
    return str(value or "").strip().upper()


def _find_performance_bucket(rows: object, *candidates: object) -> dict:
    try:
        wanted = {_normalise_bucket_text(candidate) for candidate in candidates if str(candidate or "").strip()}
        for row in list(rows or []):
            if _normalise_bucket_text(row.get("bucket")) in wanted:
                return dict(row)
    except Exception:
        return {}
    return {}


def _imported_history_adjustment(mode, trap_risk: object) -> dict[str, float]:
    out = {"score": 0.0, "confidence": 0.0}
    try:
        summary = _IMPORTED_PERFORMANCE_CACHE.get("summary")
        if not isinstance(summary, dict):
            from prediction_feedback_store import summarize_imported_ai_performance

            summary = summarize_imported_ai_performance(min_bucket_rows=3)
            _IMPORTED_PERFORMANCE_CACHE["summary"] = summary

        mode_int = _resolve_mode(mode)
        mode_bucket = _find_performance_bucket(summary.get("by_mode", []), str(mode_int), f"M{mode_int}")
        strip_bucket = _find_performance_bucket(summary.get("by_strategy_strip", []), _strategy_strip_for_mode(mode))
        trap_bucket = _find_performance_bucket(summary.get("by_trap_risk", []), trap_risk)
        for bucket in (mode_bucket, strip_bucket, trap_bucket):
            rows = int(bucket.get("rows", 0) or 0)
            if rows < 3:
                continue
            accuracy = _safe_float(bucket.get("accuracy_pct"), 50.0)
            avg_return = _safe_float(bucket.get("avg_return_pct"), 0.0)
            if accuracy < 45.0 or avg_return < -0.35:
                out["confidence"] -= 4.0
                out["score"] -= 1.5
            elif accuracy >= 62.0 and avg_return > 0.35:
                out["confidence"] += 2.0
                out["score"] += 0.8
        return out
    except Exception:
        return out


def _predict_mode_ml(row: dict, mode: int | str | None) -> float:
    resolved_mode = _resolve_mode(mode)
    try:
        if get_engine_functions is None:
            return 50.0
        try:
            module = importlib.import_module(f"strategy_engines.mode{resolved_mode}_engine")
            if getattr(module, "_MODEL", None) is None or getattr(module, "_SCALER", None) is None:
                return 50.0
        except Exception:
            return 50.0
        _, _, predict_ml_fn, _ = get_engine_functions(resolved_mode)
        return _safe_float(predict_ml_fn(row), 50.0)
    except Exception:
        return 50.0


def _label_from_direction(direction: str, score: float, confidence: float) -> str:
    if direction == "Bullish":
        if confidence >= 70 or score >= 65:
            return "Strong Momentum"
        if confidence >= 55 or score >= 58:
            return "Moderate Upside"
        return "Early Strength"
    if direction == "Bearish":
        if confidence >= 70 or score <= 35:
            return "Strong Breakdown"
        if confidence >= 55 or score <= 42:
            return "Moderate Downside"
        return "Weakening Trend"
    return "Range-Bound Setup"


def _build_trade_row(indicators: dict, score: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "RSI": indicators.get("RSI", 50.0),
                "Vol / Avg": indicators.get("Vol / Avg", 1.0),
                "Prediction Score": score,
                "Final Score": score,
                "Δ vs EMA20 (%)": indicators.get("Δ vs EMA20 (%)", 0.0),
                "5D Return (%)": indicators.get("5D Return (%)", 0.0),
                "Trap Risk": indicators.get("Trap Risk", "MEDIUM"),
            }
        ]
    )
    try:
        return apply_enhanced_logic(frame)
    except Exception:
        return frame


def get_tomorrow_prediction(ticker, all_data, mode):
    try:
        cutoff_date = _get_time_travel_cutoff()
        hist = _resolve_history(str(ticker or ""), all_data, cutoff_date=cutoff_date)
        regime_store = _resolve_regime_store(all_data, cutoff_date=cutoff_date)
        if hist is None or hist.empty:
            return {
                "ticker": _plain_ticker(str(ticker or "")),
                "direction": "Sideways",
                "confidence": 0.0,
                "score": 50.0,
                "regime": "UNKNOWN",
                "key_signal": "data_unavailable",
                "risk": "HIGH",
                "action": "🔵 Wait",
                "hold_days": "—",
                "mode": _resolve_mode(mode),
                "ml_probability": 50.0,
                "learned_probability": 50.0,
                "signals": {},
                "weights": {},
                "label_tag": "Insufficient Data",
                "atr": 0.0,
            }

        hist = hist.tail(120).copy()
        close = hist["Close"].astype(float)
        open_ = hist["Open"].astype(float)
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        volume = hist["Volume"].astype(float)

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        rsi_series = _rsi(close, 14)
        atr_series = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1).ewm(span=14, adjust=False).mean()

        last_close = _safe_float(close.iloc[-1], 0.0)
        last_open = _safe_float(open_.iloc[-1], last_close)
        last_high = _safe_float(high.iloc[-1], last_close)
        last_low = _safe_float(low.iloc[-1], last_close)
        last_vol = _safe_float(volume.iloc[-1], 0.0)
        avg_vol = _safe_float(volume.tail(20).mean(), last_vol if last_vol > 0 else 1.0)

        rsi_value = _safe_float(rsi_series.iloc[-1], 50.0)
        ema20_value = _safe_float(ema20.iloc[-1], last_close)
        ema50_value = _safe_float(ema50.iloc[-1], ema20_value)
        vol_ratio = last_vol / max(avg_vol, 1e-9)
        ret_5d = ((last_close / max(_safe_float(close.iloc[-6], last_close), 1e-9)) - 1.0) * 100.0 if len(close) >= 6 else 0.0
        ret_20d = ((last_close / max(_safe_float(close.iloc[-21], last_close), 1e-9)) - 1.0) * 100.0 if len(close) >= 21 else ret_5d
        delta_ema20 = ((last_close / max(ema20_value, 1e-9)) - 1.0) * 100.0 if ema20_value else 0.0
        high_20d = _safe_float(close.tail(20).max(), last_close)
        delta_20d_high = ((last_close / max(high_20d, 1e-9)) - 1.0) * 100.0 if high_20d else 0.0
        ema_slope_pct = ((ema20_value / max(_safe_float(ema20.iloc[-4], ema20_value), 1e-9)) - 1.0) * 100.0 if len(ema20) >= 4 else 0.0
        atr_value = _safe_float(atr_series.iloc[-1], 0.0)
        atr_pct = (atr_value / max(last_close, 1e-9)) * 100.0 if last_close else 0.0

        mtf = compute_mtf_alignment(hist) if compute_mtf_alignment is not None else None
        mtf_score = _safe_float(getattr(mtf, "alignment_score", 50.0), 50.0)
        mtf_agreement = bool(getattr(mtf, "agreement", False))

        regime_state = detect_regime(regime_store) if detect_regime is not None else None
        regime_name = str(getattr(regime_state, "regime", "RANGE_BOUND") or "RANGE_BOUND")
        regime_adjustments = regime_weight_adjustments(regime_name)
        weights = get_dynamic_weights(regime=regime_name, regime_adjustments=regime_adjustments)

        candle_direction_score = 100.0 if last_close >= last_open else 0.0
        candle_body = abs(last_close - last_open)
        candle_range = max(last_high - last_low, 1e-9)
        body_strength_score = _clamp((candle_body / candle_range) * 100.0)
        ema_slope_score = _norm_score(ema_slope_pct, -1.5, 1.5)
        price_vs_ema_score = _clamp(50.0 + delta_ema20 * 6.0 + (6.0 if ema20_value >= ema50_value else -6.0))
        volume_confirm_score = _norm_score(vol_ratio, 0.7, 2.2)
        volatility_score = _clamp(65.0 - (atr_pct * 7.5))
        momentum_score = _clamp(0.55 * rsi_value + 0.45 * _norm_score(ret_5d, -10.0, 10.0))
        bullish_pct_score = _clamp(
            (
                int(last_close > ema20_value)
                + int(ema20_value > ema50_value)
                + int(rsi_value > 50.0)
                + int(ret_5d > 0.0)
                + int(vol_ratio > 1.0)
            )
            / 5.0
            * 100.0
        )
        money_flow_score = _clamp(50.0 + np.sign(last_close - last_open) * min(abs(vol_ratio - 1.0) * 30.0, 25.0))
        participation_score = _clamp((0.6 * mtf_score) + (0.4 * volume_confirm_score))
        sector_strength_score = mtf_score

        signals = {
            "ema_slope": round(ema_slope_score, 2),
            "price_vs_ema": round(price_vs_ema_score, 2),
            "candle_direction": round(candle_direction_score, 2),
            "body_strength": round(body_strength_score, 2),
            "volume_confirm": round(volume_confirm_score, 2),
            "volatility": round(volatility_score, 2),
            "momentum": round(momentum_score, 2),
            "sector_strength": round(sector_strength_score, 2),
            "bullish_pct": round(bullish_pct_score, 2),
            "money_flow": round(money_flow_score, 2),
            "participation": round(participation_score, 2),
        }

        contributions = {}
        composite_bias = 0.0
        for signal_name, signal_value in signals.items():
            weight = _safe_float(weights.get(signal_name, 0.0), 0.0)
            contribution = ((signal_value - 50.0) / 50.0) * weight
            contributions[signal_name] = contribution
            composite_bias += contribution

        weighted_signal_score = _clamp(50.0 + (composite_bias * 50.0))

        indicator_row = {
            "Ticker": _plain_ticker(str(ticker or "")),
            "RSI": round(rsi_value, 2),
            "Vol / Avg": round(vol_ratio, 2),
            "Δ vs EMA20 (%)": round(delta_ema20, 2),
            "Δ vs 20D High (%)": round(delta_20d_high, 2),
            "5D Return (%)": round(ret_5d, 2),
            "20D Return (%)": round(ret_20d, 2),
            "EMA20": round(ema20_value, 2),
            "EMA50": round(ema50_value, 2),
            "Prediction Score": round(weighted_signal_score, 2),
            "Final Score": round(weighted_signal_score, 2),
            "pred_bullish": 1 if weighted_signal_score >= 55.0 else 0,
        }

        trade_df = _build_trade_row(indicator_row, weighted_signal_score)
        trap_risk = str(trade_df.iloc[0].get("Trap Risk", "MEDIUM") if not trade_df.empty else "MEDIUM").strip().upper() or "MEDIUM"

        ml_probability = _predict_mode_ml(indicator_row, mode)
        base_direction_score = (0.58 * weighted_signal_score) + (0.42 * ml_probability)

        learn_row = dict(indicator_row)
        learn_row["conviction_tier"] = (
            "High" if base_direction_score >= 65.0 else "Medium" if base_direction_score >= 50.0 else "Low"
        )
        learn_row["mode"] = _resolve_mode(mode)
        learn_row["strategy_strip"] = _strategy_strip_for_mode(mode)
        learn_row["trap_risk"] = trap_risk
        learn_row["regime"] = regime_name
        learn_row["market_bias"] = regime_name
        learn_row["sector"] = "UNKNOWN"
        learn_row["import_category"] = learn_row["strategy_strip"]
        learn_row["import_source"] = "Tomorrow Prediction"
        learn_row["delta_ema20_pct"] = round(delta_ema20, 2)
        learn_row["vol_avg_ratio"] = round(vol_ratio, 2)
        learn_row["rsi"] = round(rsi_value, 2)
        learned_probability = _safe_float(predict_success(learn_row), 50.0)
        history_adjustment = _imported_history_adjustment(mode, trap_risk)

        direction_sign = 1.0 if base_direction_score >= 50.0 else -1.0
        learned_edge = learned_probability - 50.0
        learned_adjustment = direction_sign * learned_edge * 0.24
        if learned_probability >= 60.0:
            learned_adjustment += direction_sign * min((learned_probability - 60.0) * 0.15, 3.0)
        elif learned_probability <= 45.0:
            learned_adjustment -= direction_sign * min((45.0 - learned_probability) * 0.10, 2.0)
        corrected_score = base_direction_score + learned_adjustment + _safe_float(history_adjustment.get("score"), 0.0)
        if trap_risk == "HIGH":
            corrected_score -= 4.0 if corrected_score >= 50.0 else -4.0
        corrected_score = _clamp(corrected_score)

        direction_strength = min(abs(corrected_score - 50.0) * 2.0, 100.0)
        confidence = (
            0.45 * direction_strength
            + 0.25 * min(abs(ml_probability - 50.0) * 2.0, 100.0)
            + 0.20 * learned_probability
            + 0.10 * mtf_score
        )
        confidence += _safe_float(history_adjustment.get("confidence"), 0.0)
        if learned_probability >= 60.0:
            confidence += min((learned_probability - 60.0) * 0.18, 4.0)
        elif learned_probability <= 45.0:
            confidence -= min((45.0 - learned_probability) * 0.28, 6.0)
        if mtf_agreement:
            confidence += 4.0
        if trap_risk == "HIGH":
            confidence -= 7.0
            confidence = min(confidence, 58.0)
        elif trap_risk == "MEDIUM":
            confidence -= 3.0
        confidence = _clamp(confidence)

        if 45.0 <= corrected_score <= 55.0 or confidence < 22.0:
            direction = "Sideways"
        elif corrected_score > 55.0:
            direction = "Bullish"
        else:
            direction = "Bearish"

        if direction == "Bearish":
            action = "🔴 Avoid"
            hold_days = "—"
        else:
            trade_df = apply_trade_decision_simple_any(trade_df)
            action = str(trade_df.iloc[0].get("Action", "🟡 Watch") if not trade_df.empty else "🟡 Watch")
            hold_days = str(trade_df.iloc[0].get("Hold Days", "—") if not trade_df.empty else "—")
            hold_days = hold_days.replace("â€“", "-").replace("—", "-")
            if direction == "Sideways":
                action = "🔵 Wait" if confidence < 40.0 else "🟡 Watch"
            elif trap_risk == "HIGH" and action == "🟢 Buy Tomorrow":
                action = "🟡 Watch"

        if trap_risk not in {"LOW", "MEDIUM", "HIGH"}:
            trap_risk = "MEDIUM"

        key_signal = "momentum"
        if contributions:
            key_signal = max(contributions, key=lambda key: abs(contributions[key]))

        return {
            "ticker": _plain_ticker(str(ticker or "")),
            "direction": direction,
            "confidence": round(confidence, 1),
            "score": round(corrected_score, 1),
            "regime": regime_name,
            "key_signal": key_signal,
            "risk": trap_risk,
            "action": action,
            "hold_days": hold_days or "—",
            "mode": _resolve_mode(mode),
            "ml_probability": round(ml_probability, 1),
            "learned_probability": round(learned_probability, 1),
            "weights": weights,
            "signals": signals,
            "mtf_score": round(mtf_score, 1),
            "label_tag": _label_from_direction(direction, corrected_score, confidence),
            "atr": round(atr_value, 2),
            "indicators": {
                "RSI": round(rsi_value, 2),
                "Vol / Avg": round(vol_ratio, 2),
                "5D Return (%)": round(ret_5d, 2),
                "20D Return (%)": round(ret_20d, 2),
                "Δ vs EMA20 (%)": round(delta_ema20, 2),
                "Δ vs 20D High (%)": round(delta_20d_high, 2),
                "EMA20": round(ema20_value, 2),
                "EMA50": round(ema50_value, 2),
                "Close": round(last_close, 2),
            },
        }
    except Exception:
        return {
            "ticker": _plain_ticker(str(ticker or "")),
            "direction": "Sideways",
            "confidence": 0.0,
            "score": 50.0,
            "regime": "UNKNOWN",
            "key_signal": "engine_error",
            "risk": "HIGH",
            "action": "🔵 Wait",
            "hold_days": "—",
            "mode": _resolve_mode(mode),
            "ml_probability": 50.0,
            "learned_probability": 50.0,
            "signals": {},
            "weights": {},
            "label_tag": "Engine Error",
            "atr": 0.0,
        }


def summarize_tomorrow_predictions(tickers, all_data, mode):
    try:
        ticker_list = []
        for ticker in list(tickers or []):
            text = _plain_ticker(str(ticker or ""))
            if text and text not in ticker_list:
                ticker_list.append(text)

        predictions = []
        if ticker_list:
            with ThreadPoolExecutor(max_workers=min(8, len(ticker_list))) as ex:
                futures = {
                    ex.submit(get_tomorrow_prediction, ticker, all_data, mode): ticker
                    for ticker in ticker_list
                }
                for fut in as_completed(futures):
                    try:
                        pred = fut.result()
                        if isinstance(pred, dict) and pred.get("ticker"):
                            predictions.append(pred)
                    except Exception:
                        continue

        if not predictions:
            return {
                "tickers": [],
                "predictions": [],
                "direction": "Sideways",
                "confidence": 0.0,
                "score": 50.0,
                "regime": "UNKNOWN",
                "key_signal": "data_unavailable",
                "risk": "HIGH",
                "action": "Wait",
                "hold_days": "-",
            }

        scores = np.array([_safe_float(pred.get("score"), 50.0) for pred in predictions], dtype=float)
        confidences = np.array([_safe_float(pred.get("confidence"), 0.0) for pred in predictions], dtype=float)

        avg_score = float(np.nanmean(scores)) if len(scores) else 50.0
        avg_conf = float(np.nanmean(confidences)) if len(confidences) else 0.0

        if avg_score > 55.0:
            direction = "Bullish"
        elif avg_score < 45.0:
            direction = "Bearish"
        else:
            direction = "Sideways"

        signal_totals = {}
        for pred in predictions:
            for signal_name, signal_value in dict(pred.get("signals") or {}).items():
                try:
                    signal_totals.setdefault(signal_name, []).append(_safe_float(signal_value, 50.0) - 50.0)
                except Exception:
                    continue

        key_signal = "momentum"
        if signal_totals:
            key_signal = max(
                signal_totals,
                key=lambda name: abs(float(np.nanmean(signal_totals.get(name, [0.0])))),
            )

        risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        risk = max(
            (str(pred.get("risk", "MEDIUM")).strip().upper() for pred in predictions),
            key=lambda value: risk_rank.get(value, 1),
            default="MEDIUM",
        )

        action_votes = [str(pred.get("action", "") or "").strip() for pred in predictions]
        buy_votes = sum("Buy Tomorrow" in action for action in action_votes)
        avoid_votes = sum("Avoid" in action for action in action_votes)
        wait_votes = sum(("Wait" in action) or ("Watch" in action) for action in action_votes)

        if direction == "Bearish" or avoid_votes >= max(1, len(predictions) // 2 + 1):
            action = "Avoid"
        elif buy_votes >= max(1, len(predictions) // 2 + 1) and risk != "HIGH":
            action = "Buy Tomorrow"
        elif wait_votes > 0 or direction == "Sideways":
            action = "Watch"
        else:
            action = "Wait"

        hold_days = "-"
        for pred in predictions:
            candidate = str(pred.get("hold_days", "") or "").strip()
            if candidate and candidate not in {"-", "—"}:
                hold_days = candidate
                break

        regimes = [str(pred.get("regime", "") or "").strip() for pred in predictions if str(pred.get("regime", "") or "").strip()]
        regime = max(regimes, key=regimes.count) if regimes else "UNKNOWN"

        return {
            "tickers": ticker_list,
            "predictions": predictions,
            "direction": direction,
            "confidence": round(_clamp(avg_conf), 1),
            "score": round(_clamp(avg_score), 1),
            "regime": regime,
            "key_signal": key_signal,
            "risk": risk if risk in risk_rank else "MEDIUM",
            "action": action,
            "hold_days": hold_days,
        }
    except Exception:
        return {
            "tickers": [],
            "predictions": [],
            "direction": "Sideways",
            "confidence": 0.0,
            "score": 50.0,
            "regime": "UNKNOWN",
            "key_signal": "engine_error",
            "risk": "HIGH",
            "action": "Wait",
            "hold_days": "-",
        }
