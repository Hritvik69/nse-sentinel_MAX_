from __future__ import annotations

import numpy as np
import pandas as pd


def build_weighted_synthetic_ohlc(
    *,
    open_panel: pd.DataFrame,
    high_panel: pd.DataFrame,
    low_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    weights: pd.Series,
    base_close: pd.Series,
) -> pd.DataFrame:
    """Build a normalized 100-base weighted OHLC basket and enforce invariants."""
    valid_cols = list(close_panel.columns)
    weights = pd.to_numeric(weights.reindex(valid_cols), errors="coerce").fillna(0.0)
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=valid_cols, dtype=float)
    weights = weights / max(float(weights.sum()), 1e-9)

    scale = 100.0 / pd.to_numeric(base_close.reindex(valid_cols), errors="coerce").replace(0, np.nan)
    norm_open = open_panel[valid_cols].mul(scale, axis=1)
    norm_high = high_panel[valid_cols].mul(scale, axis=1)
    norm_low = low_panel[valid_cols].mul(scale, axis=1)
    norm_close = close_panel[valid_cols].mul(scale, axis=1)

    out = pd.DataFrame(
        {
            "Open": norm_open.mul(weights, axis=1).sum(axis=1),
            "High": norm_high.mul(weights, axis=1).sum(axis=1),
            "Low": norm_low.mul(weights, axis=1).sum(axis=1),
            "Close": norm_close.mul(weights, axis=1).sum(axis=1),
            "Volume": volume_panel[valid_cols].sum(axis=1),
        },
        index=close_panel.index,
    )
    out["High"] = out[["Open", "High", "Close"]].max(axis=1)
    out["Low"] = out[["Open", "Low", "Close"]].min(axis=1)
    return out
