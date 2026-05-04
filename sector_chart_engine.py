"""
sector_chart_engine.py

Polished sector chart renderer for the sector prediction panel.

Design goals:
- show the recent sector OHLC cleanly with strong candle contrast
- add a future prediction candle in blue instead of reusing bull/bear colors
- keep the chart surface visually quiet by avoiding text annotations on the plot
- mirror the richer feel of the Tomorrow Stock chart without duplicating its UI
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


_MIN_CHART_ROWS = 30
_TARGET_CHART_ROWS = 60

_PAPER_BG = "#08111f"
_PLOT_BG = "#0b1628"
_GRID = "rgba(255,255,255,0.065)"
_GRID_MINOR = "rgba(255,255,255,0.03)"
_TEXT = "#dce7f8"
_MUTED = "#7f97b7"

_BULL_HEX = "#00ff88"
_BEAR_HEX = "#ff3b5c"
_SIDE_HEX = "#8fb3ff"
_EMA20_HEX = "#f5a623"
_EMA50_HEX = "#3b82f6"
_PRED_HEX = "#4da3ff"

_PRED_GLOW_ALPHA = 0.14
_PRED_EDGE_HEX = "#dbeafe"
_PRED_WICK_HEX = "#8ec5ff"
_PRED_PATH_ALPHA = 0.78


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    r, g, b = rgb
    return f"rgba({r},{g},{b},{alpha:.2f})"


_BULL_RGB = _hex_to_rgb(_BULL_HEX)
_BEAR_RGB = _hex_to_rgb(_BEAR_HEX)
_SIDE_RGB = _hex_to_rgb(_SIDE_HEX)
_EMA20_RGB = _hex_to_rgb(_EMA20_HEX)
_EMA50_RGB = _hex_to_rgb(_EMA50_HEX)
_PRED_RGB = _hex_to_rgb(_PRED_HEX)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _direction_colour(direction: str) -> str:
    return {"Bullish": _BULL_HEX, "Bearish": _BEAR_HEX, "Sideways": _SIDE_HEX}.get(direction, _SIDE_HEX)


def _confidence_alpha(confidence: float) -> float:
    return float(np.clip(0.82 + ((confidence - 50.0) / 45.0) * 0.10, 0.80, 0.96))


def _candle_line_width(confidence: float) -> float:
    return float(np.clip(1.6 + max(confidence - 50.0, 0.0) / 45.0, 1.6, 2.4))


def _range_padding(ohlc: pd.DataFrame) -> float:
    high = float(ohlc["High"].max())
    low = float(ohlc["Low"].min())
    spread = max(high - low, abs(float(ohlc["Close"].iloc[-1])) * 0.012, 1e-6)
    return spread * 0.05


def _has_meaningful_volume(ohlc: pd.DataFrame) -> bool:
    try:
        return bool(pd.to_numeric(ohlc["Volume"], errors="coerce").fillna(0).gt(0).any())
    except Exception:
        return False


def _next_session_date(last_index_value: object) -> pd.Timestamp:
    next_date = pd.Timestamp(last_index_value) + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    return next_date


def _atr_like(ohlc: pd.DataFrame) -> float:
    try:
        span = pd.to_numeric(ohlc["High"] - ohlc["Low"], errors="coerce").dropna()
        atr = float(span.tail(14).mean()) if not span.empty else 0.0
    except Exception:
        atr = 0.0
    price_floor = abs(float(ohlc["Close"].iloc[-1])) * 0.008
    return max(atr, price_floor, 1e-6)


def _build_predicted_candle(ohlc: pd.DataFrame, direction: str) -> dict[str, float | pd.Timestamp]:
    last_close = float(ohlc["Close"].iloc[-1])
    atr = _atr_like(ohlc)
    next_date = _next_session_date(ohlc.index[-1])

    if direction == "Bullish":
        open_value = last_close
        close_value = last_close + atr * 0.78
        high_value = max(open_value, close_value) + atr * 0.22
        low_value = min(open_value, close_value) - atr * 0.18
    elif direction == "Bearish":
        open_value = last_close
        close_value = last_close - atr * 0.78
        high_value = max(open_value, close_value) + atr * 0.18
        low_value = min(open_value, close_value) - atr * 0.22
    else:
        body = atr * 0.14
        open_value = last_close - body * 0.40
        close_value = last_close + body * 0.40
        high_value = max(open_value, close_value) + atr * 0.24
        low_value = min(open_value, close_value) - atr * 0.24

    return {
        "date": next_date,
        "open": float(open_value),
        "high": float(high_value),
        "low": float(low_value),
        "close": float(close_value),
        "atr": float(atr),
    }


def _projection_geometry(dates: list[pd.Timestamp]) -> tuple[pd.Timedelta, pd.Timedelta]:
    try:
        gap = (dates[-1] - dates[-2]).total_seconds() / 86400 if len(dates) >= 2 else 1.0
        half_day = pd.Timedelta(days=gap * 0.36)
        glow_day = pd.Timedelta(days=gap * 0.54)
        return half_day, glow_day
    except Exception:
        return pd.Timedelta(hours=9), pd.Timedelta(hours=13)


def _projection_shapes(
    ohlc: pd.DataFrame,
    pred_candle: dict[str, float | pd.Timestamp],
) -> list[dict]:
    dates = list(pd.to_datetime(ohlc.index))
    half_day, glow_day = _projection_geometry(dates)
    pred_date = pd.Timestamp(pred_candle["date"])
    pred_open = float(pred_candle["open"])
    pred_close = float(pred_candle["close"])
    pred_high = float(pred_candle["high"])
    pred_low = float(pred_candle["low"])
    sep_x = pred_date - half_day * 1.55

    body_y0 = min(pred_open, pred_close)
    body_y1 = max(pred_open, pred_close)
    if body_y1 <= body_y0:
        body_y1 = body_y0 + max(abs(pred_high - pred_low) * 0.05, 1e-6)

    last_close = float(ohlc["Close"].iloc[-1])

    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": sep_x,
            "x1": pred_date + glow_day * 0.8,
            "y0": 0.0,
            "y1": 1.0,
            "fillcolor": _rgba(_PRED_RGB, 0.04),
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": sep_x,
            "x1": sep_x,
            "y0": 0.0,
            "y1": 1.0,
            "line": {"color": "rgba(142,197,255,0.28)", "width": 1.15, "dash": "dot"},
            "layer": "below",
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": dates[-1],
            "x1": pred_date,
            "y0": last_close,
            "y1": pred_close,
            "line": {"color": _rgba(_PRED_RGB, _PRED_PATH_ALPHA), "width": 2.0, "dash": "dot"},
            "layer": "above",
        },
        {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": pred_date - glow_day,
            "x1": pred_date + glow_day,
            "y0": body_y0,
            "y1": body_y1,
            "fillcolor": _rgba(_PRED_RGB, _PRED_GLOW_ALPHA),
            "line": {"color": _rgba(_PRED_RGB, 0.0), "width": 0},
            "layer": "above",
        },
    ]


def _volume_colors(ohlc: pd.DataFrame, confidence: float) -> list[str]:
    alpha = float(np.clip(_confidence_alpha(confidence) - 0.10, 0.70, 0.88))
    return [
        _rgba(_BULL_RGB if close_value >= open_value else _BEAR_RGB, alpha)
        for open_value, close_value in zip(ohlc["Open"], ohlc["Close"])
    ]


def _signal_bar(name: str, value: float) -> str:
    filled = int(np.clip(round(value / 10.0), 0, 10))
    bar = "█" * filled + "░" * (10 - filled)
    return f"{name:<18} {bar} {value:5.1f}"


def _signal_hover_text(prediction) -> str:
    sig = getattr(prediction, "signals", None)
    if sig is None:
        return ""
    lines = [
        _signal_bar("EMA Slope", getattr(sig, "ema_slope", 50.0)),
        _signal_bar("Price vs EMA", getattr(sig, "price_vs_ema", 50.0)),
        _signal_bar("Momentum", getattr(sig, "momentum", 50.0)),
        _signal_bar("Volume Confirm", getattr(sig, "volume_confirm", 50.0)),
        _signal_bar("Sector Strength", getattr(sig, "sector_strength", 50.0)),
        _signal_bar("Bullish %", getattr(sig, "bullish_pct", 50.0)),
        _signal_bar("Money Flow", getattr(sig, "money_flow", 50.0)),
        _signal_bar("Participation", getattr(sig, "participation", 50.0)),
    ]
    return "<br>".join(lines)


def build_sector_chart(prediction) -> "go.Figure | None":
    if not _PLOTLY_OK:
        return None

    ohlc = getattr(prediction, "ohlc_df", None)
    if ohlc is None:
        return None
    ohlc = ohlc.copy()
    if len(ohlc) < _MIN_CHART_ROWS:
        return None

    ohlc = ohlc.tail(_TARGET_CHART_ROWS).copy()
    ohlc.index = pd.to_datetime(ohlc.index)
    ohlc = ohlc.sort_index()

    direction = str(getattr(prediction, "direction", "Sideways") or "Sideways")
    confidence = float(getattr(prediction, "confidence", 50.0) or 50.0)
    sector = str(getattr(prediction, "sector", "Sector") or "Sector")
    pred_candle = _build_predicted_candle(ohlc, direction)

    alpha = _confidence_alpha(confidence)
    line_width = _candle_line_width(confidence)
    show_volume = _has_meaningful_volume(ohlc)

    ema20 = _ema(ohlc["Close"], 20)
    ema50 = _ema(ohlc["Close"], 50)

    rows = 2 if show_volume else 1
    row_heights = [0.78, 0.22] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.025 if show_volume else 0.0,
    )

    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["Open"],
            high=ohlc["High"],
            low=ohlc["Low"],
            close=ohlc["Close"],
            increasing={
                "fillcolor": _rgba(_BULL_RGB, alpha),
                "line": {"color": _BULL_HEX, "width": line_width},
            },
            decreasing={
                "fillcolor": _rgba(_BEAR_RGB, alpha),
                "line": {"color": _BEAR_HEX, "width": line_width},
            },
            name="Price",
            showlegend=True,
            opacity=1.0,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=ema20,
            mode="lines",
            line={"color": _rgba(_EMA20_RGB, 0.16), "width": 5.0},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=ema20,
            mode="lines",
            line={"color": _EMA20_HEX, "width": 2.05},
            name="EMA 20",
            hovertemplate="EMA20: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=ema50,
            mode="lines",
            line={"color": _rgba(_EMA50_RGB, 0.16), "width": 4.5},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=ema50,
            mode="lines",
            line={"color": _EMA50_HEX, "width": 1.7, "dash": "dot"},
            name="EMA 50",
            hovertemplate="EMA50: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    pred_date = pd.Timestamp(pred_candle["date"])
    pred_open = float(pred_candle["open"])
    pred_high = float(pred_candle["high"])
    pred_low = float(pred_candle["low"])
    pred_close = float(pred_candle["close"])

    fig.add_trace(
        go.Candlestick(
            x=[pred_date],
            open=[pred_open],
            high=[pred_high],
            low=[pred_low],
            close=[pred_close],
            increasing={
                "fillcolor": _PRED_HEX,
                "line": {"color": _PRED_EDGE_HEX, "width": 2.4},
            },
            decreasing={
                "fillcolor": _PRED_HEX,
                "line": {"color": _PRED_EDGE_HEX, "width": 2.4},
            },
            whiskerwidth=0.9,
            name="Projection",
            showlegend=True,
            opacity=1.0,
        ),
        row=1,
        col=1,
    )

    fig.add_shape(
        type="line",
        x0=pred_date,
        x1=pred_date,
        y0=pred_low,
        y1=pred_high,
        line={"color": _PRED_WICK_HEX, "width": 3.0},
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[pred_date],
            y=[pred_close],
            mode="markers",
            marker={
                "size": 11,
                "color": _PRED_HEX,
                "line": {"color": _PRED_EDGE_HEX, "width": 1.6},
                "symbol": "diamond",
            },
            name="Projected Close",
            showlegend=False,
            hovertemplate=(
                "Projected Close: %{y:.2f}<br>"
                f"Projected Open: {pred_open:.2f}<br>"
                f"Projected High: {pred_high:.2f}<br>"
                f"Projected Low: {pred_low:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    hover_text = _signal_hover_text(prediction)
    if hover_text:
        fig.add_trace(
            go.Scatter(
                x=[ohlc.index[-1]],
                y=[float(ohlc["High"].iloc[-1]) + _range_padding(ohlc) * 0.22],
                mode="markers",
                marker={"color": "rgba(0,0,0,0)", "size": 1},
                hovertext=hover_text,
                hoverinfo="text",
                name="Signals",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if show_volume:
        fig.add_trace(
            go.Bar(
                x=ohlc.index,
                y=ohlc["Volume"],
                marker_color=_volume_colors(ohlc, confidence),
                marker_line_width=0,
                name="Volume",
                hovertemplate="Volume: %{y:.3s}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    title_text = (
        f"<b>{sector.replace('_', ' ').title()}</b>"
        f"<span style='color:{_MUTED};font-size:11px;'>"
        f"  ·  Daily  ·  {len(ohlc)} sessions</span>"
    )

    fig.update_layout(
        title={"text": title_text, "x": 0.01, "xanchor": "left", "font": {"size": 17, "color": _TEXT}},
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _TEXT, "family": "Segoe UI, Arial, sans-serif"},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(11, 22, 40, 0.98)",
            "bordercolor": "rgba(255,255,255,0.12)",
            "font": {"color": _TEXT, "family": "Consolas, Menlo, monospace"},
        },
        margin={"l": 4, "r": 56, "t": 54, "b": 8},
        height=600 if show_volume else 500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "x": 0.0,
            "y": 1.05,
            "font": {"size": 10, "color": _TEXT},
            "bgcolor": "rgba(0,0,0,0)",
        },
        shapes=_projection_shapes(ohlc, pred_candle),
    )

    fig.update_xaxes(
        type="date",
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        showgrid=True,
        gridcolor=_GRID_MINOR,
        zeroline=False,
        showline=False,
        tickfont={"size": 10, "color": _MUTED},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 2, "label": "2M", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            "bgcolor": "#10203a",
            "activecolor": "#18345f",
            "bordercolor": "#18345f",
            "font": {"color": _TEXT, "size": 10},
            "x": 0.0,
            "y": 1.0,
            "xanchor": "left",
        },
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        zeroline=False,
        showline=False,
        side="right",
        tickfont={"size": 10, "color": _MUTED},
        row=1,
        col=1,
    )

    if show_volume:
        fig.update_xaxes(
            type="date",
            rangebreaks=[{"bounds": ["sat", "mon"]}],
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont={"size": 10, "color": _MUTED},
            row=2,
            col=1,
        )
        fig.update_yaxes(
            showgrid=False,
            tickformat=".2s",
            zeroline=False,
            showline=False,
            side="right",
            tickfont={"size": 10, "color": _MUTED},
            row=2,
            col=1,
        )

    fig.update_traces(selector={"type": "candlestick"}, whiskerwidth=0.82)
    return fig
