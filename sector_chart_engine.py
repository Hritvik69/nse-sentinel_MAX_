"""
sector_chart_engine.py

TradingView-style sector chart renderer for the sector prediction panel.

Rules enforced here:
- never draw fewer than 30 daily candles
- prefer the most recent 60 candles
- keep candles high-contrast on a dark chart
- show prediction confidence without fading the chart into invisibility
- make the final candle and direction label obvious at a glance
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

_PAPER_BG = "#0a1020"
_PLOT_BG = "#0d1526"
_GRID = "rgba(148, 163, 184, 0.10)"
_TEXT = "#e7eefc"
_MUTED = "#8ea0bd"

_BULL_HEX = "#00ff88"
_BEAR_HEX = "#ff3b5c"
_SIDE_HEX = "#8fb3ff"
_EMA_HEX = "#f7c948"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    r, g, b = rgb
    return f"rgba({r},{g},{b},{alpha:.2f})"


_BULL_RGB = _hex_to_rgb(_BULL_HEX)
_BEAR_RGB = _hex_to_rgb(_BEAR_HEX)
_SIDE_RGB = _hex_to_rgb(_SIDE_HEX)
_EMA_RGB = _hex_to_rgb(_EMA_HEX)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _direction_icon(direction: str) -> str:
    return {"Bullish": "▲", "Bearish": "▼", "Sideways": "◆"}.get(direction, "●")


def _direction_colour(direction: str) -> str:
    return {"Bullish": _BULL_HEX, "Bearish": _BEAR_HEX, "Sideways": _SIDE_HEX}.get(direction, _SIDE_HEX)


def _direction_rgb(direction: str) -> tuple[int, int, int]:
    return {"Bullish": _BULL_RGB, "Bearish": _BEAR_RGB, "Sideways": _SIDE_RGB}.get(direction, _SIDE_RGB)


def _confidence_alpha(confidence: float) -> float:
    return float(np.clip(0.80 + ((confidence - 50.0) / 45.0) * 0.12, 0.78, 0.95))


def _candle_line_width(confidence: float) -> float:
    return float(np.clip(1.6 + max(confidence - 50.0, 0.0) / 45.0, 1.6, 2.5))


def _range_padding(ohlc: pd.DataFrame) -> float:
    high = float(ohlc["High"].max())
    low = float(ohlc["Low"].min())
    spread = max(high - low, abs(float(ohlc["Close"].iloc[-1])) * 0.01, 1e-6)
    return spread * 0.05


def _has_meaningful_volume(ohlc: pd.DataFrame) -> bool:
    try:
        return bool(pd.to_numeric(ohlc["Volume"], errors="coerce").fillna(0).gt(0).any())
    except Exception:
        return False


def _forecast_zone_shape(ohlc: pd.DataFrame, direction: str, confidence: float) -> dict:
    shade_start = pd.Timestamp(ohlc.index[max(int(len(ohlc) * 0.85), 0)])
    shade_end = pd.Timestamp(ohlc.index[-1]) + pd.Timedelta(days=1)
    alpha = float(np.clip(0.06 + ((confidence - 50.0) / 50.0) * 0.05, 0.05, 0.14))
    return {
        "type": "rect",
        "xref": "x",
        "yref": "paper",
        "x0": shade_start,
        "x1": shade_end,
        "y0": 0.0,
        "y1": 1.0,
        "fillcolor": _rgba(_direction_rgb(direction), alpha),
        "line": {"width": 0},
        "layer": "below",
    }


def _last_candle_focus_shapes(ohlc: pd.DataFrame, direction: str, confidence: float) -> list[dict]:
    last_x = pd.Timestamp(ohlc.index[-1])
    last_high = float(ohlc["High"].iloc[-1])
    last_low = float(ohlc["Low"].iloc[-1])
    last_close = float(ohlc["Close"].iloc[-1])
    pad = _range_padding(ohlc)
    border_rgb = _direction_rgb(direction)
    border_alpha = float(np.clip(_confidence_alpha(confidence) + 0.05, 0.84, 1.0))
    focus_fill_alpha = float(np.clip(0.07 + ((confidence - 50.0) / 50.0) * 0.04, 0.05, 0.12))
    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": last_x - pd.Timedelta(hours=12),
            "x1": last_x + pd.Timedelta(hours=12),
            "y0": last_low - pad * 0.20,
            "y1": last_high + pad * 0.20,
            "fillcolor": _rgba(border_rgb, focus_fill_alpha),
            "line": {"color": _rgba(border_rgb, border_alpha), "width": 2.1},
            "layer": "above",
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": pd.Timestamp(ohlc.index[0]),
            "x1": last_x,
            "y0": last_close,
            "y1": last_close,
            "line": {
                "color": _rgba(border_rgb, 0.48),
                "width": 1.2,
                "dash": "dot",
            },
            "layer": "above",
        },
    ]


def _last_candle_annotation(ohlc: pd.DataFrame, direction: str, confidence: float) -> dict:
    last_x = pd.Timestamp(ohlc.index[-1])
    last_high = float(ohlc["High"].iloc[-1])
    pad = _range_padding(ohlc)
    direction_colour = _direction_colour(direction)
    label = f"{_direction_icon(direction)} {direction} {confidence:.0f}%"
    return {
        "x": last_x,
        "y": last_high + pad * 0.65,
        "xref": "x",
        "yref": "y",
        "text": f"<b>{label}</b>",
        "showarrow": True,
        "arrowhead": 2,
        "arrowsize": 1.2,
        "arrowwidth": 2.2,
        "arrowcolor": direction_colour,
        "ax": 0,
        "ay": -58,
        "bgcolor": "rgba(10,16,32,0.96)",
        "bordercolor": direction_colour,
        "borderwidth": 2,
        "borderpad": 6,
        "font": {
            "color": direction_colour,
            "size": 13,
            "family": "Segoe UI, Arial, sans-serif",
        },
    }


def _volume_colors(ohlc: pd.DataFrame, confidence: float) -> list[str]:
    alpha = float(np.clip(_confidence_alpha(confidence) - 0.10, 0.70, 0.88))
    colors: list[str] = []
    for open_value, close_value in zip(ohlc["Open"], ohlc["Close"]):
        colors.append(_rgba(_BULL_RGB if close_value >= open_value else _BEAR_RGB, alpha))
    return colors


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

    alpha = _confidence_alpha(confidence)
    line_width = _candle_line_width(confidence)
    show_volume = _has_meaningful_volume(ohlc)
    ema20 = _ema(ohlc["Close"], 20)

    rows = 2 if show_volume else 1
    row_heights = [0.77, 0.23] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03 if show_volume else 0.0,
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
                "line": {"color": _rgba(_BULL_RGB, 1.0), "width": line_width},
            },
            decreasing={
                "fillcolor": _rgba(_BEAR_RGB, alpha),
                "line": {"color": _rgba(_BEAR_RGB, 1.0), "width": line_width},
            },
            whiskerwidth=0.75,
            name="Price",
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
            line={"color": _rgba(_EMA_RGB, 0.18), "width": 5.0},
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
            line={"color": _EMA_HEX, "width": 2.15, "dash": "dot"},
            name="EMA20",
            hovertemplate="EMA20: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    last_close = float(ohlc["Close"].iloc[-1])
    fig.add_trace(
        go.Scatter(
            x=[ohlc.index[-1]],
            y=[last_close],
            mode="markers",
            marker={
                "size": 10 + max(confidence - 50.0, 0.0) * 0.05,
                "color": _direction_colour(direction),
                "line": {"color": "#f8fafc", "width": 1.4},
                "symbol": "diamond",
            },
            name="Signal",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    hover_text = _signal_hover_text(prediction)
    if hover_text:
        fig.add_trace(
            go.Scatter(
                x=[ohlc.index[-1]],
                y=[float(ohlc["High"].iloc[-1]) + _range_padding(ohlc) * 0.25],
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
                name="Volume",
                hovertemplate="Volume: %{y:.3s}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    title_colour = _direction_colour(direction)
    title_text = (
        f"<b>{sector}</b>  "
        f"{_direction_icon(direction)} "
        f"<span style='color:{title_colour}'>{direction} {confidence:.0f}%</span>"
    )

    source_text = str(getattr(prediction, "ohlc_source", "") or "").replace("_", " ").title()
    if source_text:
        source_text += f" · {int(getattr(prediction, 'ohlc_bars', len(ohlc)) or len(ohlc))} candles"

    annotations = [_last_candle_annotation(ohlc, direction, confidence)]
    if source_text:
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 1.08,
                "text": source_text,
                "showarrow": False,
                "font": {"size": 10, "color": _MUTED, "family": "Segoe UI, Arial, sans-serif"},
                "align": "left",
            }
        )

    shapes = [_forecast_zone_shape(ohlc, direction, confidence), *_last_candle_focus_shapes(ohlc, direction, confidence)]

    fig.update_layout(
        title={"text": title_text, "x": 0.01, "font": {"size": 18, "color": _TEXT}},
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _TEXT, "family": "Segoe UI, Arial, sans-serif"},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(12, 18, 32, 0.98)",
            "bordercolor": "rgba(255,255,255,0.12)",
            "font": {"color": _TEXT, "family": "Consolas, Menlo, monospace"},
        },
        margin={"l": 22, "r": 28, "t": 78, "b": 22},
        height=580 if show_volume else 470,
        xaxis_rangeslider_visible=False,
        annotations=annotations,
        shapes=shapes,
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 1.01,
            "font": {"size": 10, "color": _MUTED},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )

    fig.update_xaxes(
        type="date",
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        showgrid=True,
        gridcolor=_GRID,
        tickfont={"size": 10, "color": _MUTED},
        showspikes=True,
        spikemode="across",
        spikecolor="rgba(255,255,255,0.22)",
        spikethickness=1,
        spikedash="dot",
        zeroline=False,
        showline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID,
        tickfont={"size": 10, "color": _MUTED},
        side="right",
        zeroline=False,
        showline=False,
        row=1,
        col=1,
    )

    if show_volume:
        fig.update_xaxes(
            type="date",
            rangebreaks=[{"bounds": ["sat", "mon"]}],
            showgrid=False,
            tickfont={"size": 10, "color": _MUTED},
            zeroline=False,
            showline=False,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            showgrid=False,
            tickformat=".2s",
            tickfont={"size": 10, "color": _MUTED},
            side="right",
            zeroline=False,
            showline=False,
            row=2,
            col=1,
        )

    return fig
