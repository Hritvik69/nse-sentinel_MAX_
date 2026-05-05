"""
strategy_engines/__init__.py
─────────────────────────────
Dispatcher for the 6 independent strategy engines.
`get_engine_functions(mode)` returns a 4-tuple of callables:
    (compute_score_fn, backtest_fn, predict_ml_fn, check_bull_trap_fn)

Each function is mode-specific — no shared logic.

UPGRADE v2: Also exports zero-API helpers for the central data engine.
  backtest_with_preloaded(mode, row, ticker)  — uses ALL_DATA, zero API calls
  preload_all(tickers)                        — fills ALL_DATA in parallel
  get_df_for_ticker(ticker)                   — ALL_DATA lookup with fallback
"""

from __future__ import annotations

from typing import Callable

# ── zero-API central engine exports ───────────────────────────────────
from strategy_engines._engine_utils import (
    ALL_DATA,
    preload_all,
    get_df_for_ticker,
    preload_history_batch,
)


# ── lazy imports — only load the engine for the requested mode ────────
def get_engine_functions(mode: int) -> tuple[
    Callable,   # compute_score(row)  → (float, dict)
    Callable,   # backtest(row, ticker) → float
    Callable,   # predict_ml(row) → float
    Callable,   # check_bull_trap(row) → str
]:
    """
    Returns (score_fn, backtest_fn, ml_fn, trap_fn) for the given mode.
    Raises ImportError if the engine module cannot be loaded.
    """
    if mode == 1:
        from strategy_engines.mode1_engine import (
            compute_score_mode1   as score_fn,
            backtest_mode1        as bt_fn,
            predict_ml_mode1      as ml_fn,
            check_bull_trap_mode1 as trap_fn,
        )
    elif mode == 2:
        from strategy_engines.mode2_engine import (
            compute_score_mode2   as score_fn,
            backtest_mode2        as bt_fn,
            predict_ml_mode2      as ml_fn,
            check_bull_trap_mode2 as trap_fn,
        )
    elif mode == 3:
        from strategy_engines.mode3_engine import (
            compute_score_mode3   as score_fn,
            backtest_mode3        as bt_fn,
            predict_ml_mode3      as ml_fn,
            check_bull_trap_mode3 as trap_fn,
        )
    elif mode == 4:
        from strategy_engines.mode4_engine import (
            compute_score_mode4   as score_fn,
            backtest_mode4        as bt_fn,
            predict_ml_mode4      as ml_fn,
            check_bull_trap_mode4 as trap_fn,
        )
    elif mode == 5:
        from strategy_engines.mode5_engine import (
            compute_score_mode5   as score_fn,
            backtest_mode5        as bt_fn,
            predict_ml_mode5      as ml_fn,
            check_bull_trap_mode5 as trap_fn,
        )
    elif mode == 6:
        from strategy_engines.mode6_engine import (
            compute_score_mode6   as score_fn,
            backtest_mode6        as bt_fn,
            predict_ml_mode6      as ml_fn,
            check_bull_trap_mode6 as trap_fn,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return score_fn, bt_fn, ml_fn, trap_fn


# ── convenience: pre-load all train functions for background warm-up ──
def get_train_function(mode: int) -> Callable:
    """Returns the train_model_modeX() function for the given mode."""
    _MAP = {
        1: "strategy_engines.mode1_engine.train_model_mode1",
        2: "strategy_engines.mode2_engine.train_model_mode2",
        3: "strategy_engines.mode3_engine.train_model_mode3",
        4: "strategy_engines.mode4_engine.train_model_mode4",
        5: "strategy_engines.mode5_engine.train_model_mode5",
        6: "strategy_engines.mode6_engine.train_model_mode6",
    }
    module_path, fn_name = _MAP[mode].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def backtest_with_preloaded(mode: int, row: dict, ticker: str) -> float:
    """
    Lazy wrapper so the dataframe backtest helpers do not load during the
    initial app shell render. Import them only when a scan actually needs them.
    """
    try:
        from strategy_engines._df_extensions import backtest_with_preloaded as _bt_with_preloaded

        return _bt_with_preloaded(mode, row, ticker)
    except Exception:
        return 50.0
