"""
sector_chart_engine.py
═══════════════════════
Chart-driven probability system.

Builds a Plotly candlestick chart where candle color intensity is
proportional to prediction confidence.  The last candle always
carries a visible annotation: "Bullish 72%" (or Bearish / Sideways).

Visual probability mapping (MANDATORY)
───────────────────────────────────────
• Bullish  prediction → candles shaded green; intensity ∝ confidence
• Bearish  prediction → candles shaded red;   intensity ∝ confidence
• Sideways prediction → candles shaded grey

Only the last 60 candles are charted (legibility).

Public API
──────────
    build_sector_chart(prediction) → plotly Figure
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


# ══════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(r: int, g: int, b: int, a: float) -> str:
    return f"rgba({r},{g},{b},{a:.2f})"


# Base palette
_BULL_COLOR  = (0, 200, 150)    # teal-green
_BEAR_COLOR  = (255, 60, 80)    # vivid red
_SIDE_COLOR  = (140, 160, 190)  # steel blue-grey
_VOLUME_BULL = (0, 180, 130)
_VOLUME_BEAR = (220, 50, 70)
_EMA_COLOR   = "#f0b429"        # amber


def _candle_colors(
    ohlc: pd.DataFrame,
    direction: str,
    confidence: float,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Return (increasing_fill, increasing_line, decreasing_fill, decreasing_line)
    for the chart's candlestick trace.

    Intensity is derived from confidence (0–100) so that:
      50% confidence → faded (alpha ~0.4)
      95% confidence → vivid (alpha ~0.9)
    """
    alpha = float(np.clip(0.35 + (confidence / 100) * 0.55, 0.35, 0.90))

    if direction == "Bullish":
        inc_fill = _rgba(*_BULL_COLOR, alpha)
        inc_line = _rgba(*_BULL_COLOR, min(alpha + 0.1, 1.0))
        dec_fill = _rgba(200, 60, 80, alpha * 0.55)     # faded bear
        dec_line = _rgba(200, 60, 80, alpha * 0.65)
    elif direction == "Bearish":
        inc_fill = _rgba(0, 160, 120, alpha * 0.55)     # faded bull
        inc_line = _rgba(0, 160, 120, alpha * 0.65)
        dec_fill = _rgba(*_BEAR_COLOR, alpha)
        dec_line = _rgba(*_BEAR_COLOR, min(alpha + 0.1, 1.0))
    else:  # Sideways
        inc_fill = _rgba(*_SIDE_COLOR, 0.55)
        inc_line = _rgba(*_SIDE_COLOR, 0.75)
        dec_fill = _rgba(*_SIDE_COLOR, 0.55)
        dec_line = _rgba(*_SIDE_COLOR, 0.75)

    return inc_fill, inc_line, dec_fill, dec_line


def _direction_icon(direction: str) -> str:
    return {"Bullish": "▲", "Bearish": "▼", "Sideways": "◆"}.get(direction, "●")


def _direction_colour(direction: str) -> str:
    return {"Bullish": "#00d4a8", "Bearish": "#ff4d6d", "Sideways": "#8ab4d8"}.get(direction, "#8ab4d8")


# ══════════════════════════════════════════════════════════════════════
# EMA HELPER
# ══════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


# ══════════════════════════════════════════════════════════════════════
# SIGNAL BADGE (last candle annotation)
# ══════════════════════════════════════════════════════════════════════

def _last_candle_annotation(ohlc: pd.DataFrame, direction: str, confidence: float) -> dict:
    last_idx   = ohlc.index[-1]
    last_high  = float(ohlc["High"].iloc[-1])
    arrow      = _direction_icon(direction)
    colour     = _direction_colour(direction)
    label_text = f"{arrow} {direction} {confidence:.0f}%"

    return dict(
        x=str(last_idx),
        y=last_high * 1.015,
        xref="x",
        yref="y",
        text=f"<b>{label_text}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=colour,
        font=dict(color=colour, size=13, family="monospace"),
        bgcolor="rgba(10,14,26,0.85)",
        bordercolor=colour,
        borderwidth=1.5,
        borderpad=5,
        ax=0,
        ay=-45,
    )


# ══════════════════════════════════════════════════════════════════════
# BACKGROUND SHADING
# ══════════════════════════════════════════════════════════════════════

def _background_shape(ohlc: pd.DataFrame, direction: str, confidence: float) -> dict | None:
    """Shade the right 15 % of the chart in the prediction colour."""
    if len(ohlc) < 10:
        return None
    shade_start = ohlc.index[int(len(ohlc) * 0.85)]
    shade_end   = ohlc.index[-1]
    alpha       = float(np.clip(0.04 + (confidence / 100) * 0.08, 0.04, 0.12))
    if direction == "Bullish":
        fill = _rgba(*_BULL_COLOR, alpha)
    elif direction == "Bearish":
        fill = _rgba(*_BEAR_COLOR, alpha)
    else:
        return None

    return dict(
        type="rect",
        xref="x", yref="paper",
        x0=str(shade_start), x1=str(shade_end),
        y0=0, y1=1,
        fillcolor=fill,
        line_width=0,
        layer="below",
    )


# ══════════════════════════════════════════════════════════════════════
# VOLUME BARS
# ══════════════════════════════════════════════════════════════════════

def _volume_colors(ohlc: pd.DataFrame, direction: str, confidence: float) -> list[str]:
    alpha = float(np.clip(0.30 + (confidence / 100) * 0.45, 0.30, 0.75))
    colors = []
    for i in range(len(ohlc)):
        bull = ohlc["Close"].iloc[i] >= ohlc["Open"].iloc[i]
        if bull:
            colors.append(_rgba(*_VOLUME_BULL, alpha if direction == "Bullish" else alpha * 0.5))
        else:
            colors.append(_rgba(*_VOLUME_BEAR, alpha if direction == "Bearish" else alpha * 0.5))
    return colors


# ══════════════════════════════════════════════════════════════════════
# SIGNAL PANEL TEXT
# ══════════════════════════════════════════════════════════════════════

def _signal_bar(name: str, value: float) -> str:
    """Return a short text bar representation of a 0–100 signal."""
    filled = int(value / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    return f"{name:<18} {bar} {value:5.1f}"


# ══════════════════════════════════════════════════════════════════════
# MAIN CHART BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_sector_chart(prediction) -> "go.Figure | None":  # SectorPrediction
    """
    Build and return a Plotly Figure with:
      • Candlestick (color-intensity ∝ confidence)
      • EMA20 overlay
      • Volume bars (color-matched)
      • Prediction annotation on last candle
      • Background shading in the trend zone
      • Signal breakdown panel (hover text)

    Returns None if Plotly is not installed or data is unavailable.
    """
    if not _PLOTLY_OK:
        return None

    ohlc = prediction.ohlc_df
    if ohlc is None or len(ohlc) < 5:
        return None

    # Use last 60 candles for chart legibility
    ohlc = ohlc.tail(60).copy()

    direction  = prediction.direction
    confidence = prediction.confidence
    sector     = prediction.sector

    inc_fill, inc_line, dec_fill, dec_line = _candle_colors(ohlc, direction, confidence)

    # ── Date labels ───────────────────────────────────────────────────
    x_labels = [str(d)[:10] for d in ohlc.index]

    # ── EMA20 ─────────────────────────────────────────────────────────
    ema20 = _ema(ohlc["Close"], 20)

    # ── Layout ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Candlestick ───────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=x_labels,
            open=ohlc["Open"],
            high=ohlc["High"],
            low=ohlc["Low"],
            close=ohlc["Close"],
            increasing=dict(fillcolor=inc_fill, line=dict(color=inc_line, width=1)),
            decreasing=dict(fillcolor=dec_fill, line=dict(color=dec_line, width=1)),
            name="OHLC",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── EMA20 ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=ema20,
            line=dict(color=_EMA_COLOR, width=1.5, dash="dot"),
            name="EMA20",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # ── Volume ────────────────────────────────────────────────────────
    vol_colors = _volume_colors(ohlc, direction, confidence)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=ohlc["Volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Annotations ───────────────────────────────────────────────────
    annotations = [_last_candle_annotation(ohlc, direction, confidence)]

    # ── Background shape ──────────────────────────────────────────────
    shapes = []
    bg = _background_shape(ohlc, direction, confidence)
    if bg:
        shapes.append(bg)

    # ── Layout styling ────────────────────────────────────────────────
    dir_colour = _direction_colour(direction)
    icon       = _direction_icon(direction)
    title_text = (
        f"<b>{sector}</b>  {icon} <span style='color:{dir_colour}'>"
        f"{direction} {confidence:.0f}%</span>"
    )

    # Signal hover annotation on last candle
    sig = prediction.signals
    hover_lines = [
        _signal_bar("EMA Slope",      sig.ema_slope),
        _signal_bar("Momentum",       sig.momentum),
        _signal_bar("Volume Confirm", sig.volume_confirm),
        _signal_bar("Candle Dir",     sig.candle_direction),
        _signal_bar("Body Strength",  sig.body_strength),
        _signal_bar("Consecutive",    sig.consecutive),
        _signal_bar("Sector Strength",sig.sector_strength),
        _signal_bar("Bullish %",      sig.bullish_pct),
        _signal_bar("Money Flow",     sig.money_flow),
    ]
    # Invisible scatter on last candle to carry hover signal data
    fig.add_trace(
        go.Scatter(
            x=[x_labels[-1]],
            y=[float(ohlc["High"].iloc[-1]) * 1.02],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=1),
            hovertext="<br>".join(hover_lines),
            hoverinfo="text",
            name="Signals",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="#e8eaf0"), x=0.01),
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        font=dict(color="#8ab4d8", family="monospace"),
        xaxis_rangeslider_visible=False,
        annotations=annotations,
        shapes=shapes,
        legend=dict(
            orientation="h", x=0.01, y=1.02,
            font=dict(size=10, color="#8ab4d8"),
        ),
        margin=dict(l=50, r=30, t=70, b=40),
        height=520,
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(100,120,160,0.12)",
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(100,120,160,0.12)",
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, tickformat=".2s",
        row=2, col=1,
    )

    return fig