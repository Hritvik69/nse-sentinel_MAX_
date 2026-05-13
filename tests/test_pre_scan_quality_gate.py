from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _ohlcv(closes: np.ndarray | list[float], volume: float = 1000.0) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2025-11-01", periods=len(close))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), float(volume)),
        },
        index=idx,
    )


class PreScanQualityGateTests(unittest.TestCase):
    def test_mode5_blocks_3pland_style_downtrend(self) -> None:
        from pre_scan_quality_gate import apply_gate_to_scan_df

        scan = pd.DataFrame(
            [
                {
                    "Symbol": "3PLAND",
                    "Mode ID": 5,
                    "Final Score": 100.0,
                    "Tomorrow Pick Score": 100.0,
                    "AI Confidence": 29.0,
                    "RSI": 55.0,
                    "Vol / Avg": 1.05,
                }
            ]
        )
        hist = {"3PLAND.NS": _ohlcv(np.linspace(44.0, 28.0, 120))}

        kept = apply_gate_to_scan_df(scan, hist, mode=5, drop_blocked=True)
        self.assertTrue(kept.empty)

        audited = apply_gate_to_scan_df(scan, hist, mode=5, drop_blocked=False)
        self.assertTrue(bool(audited.loc[0, "Gate Blocked"]))
        self.assertEqual(float(audited.loc[0, "Final Score"]), 0.0)
        self.assertEqual(float(audited.loc[0, "Tomorrow Pick Score"]), 0.0)
        self.assertIn("G1", str(audited.loc[0, "Gate Reasons"]))

    def test_mode2_bearish_ema_stack_is_capped_not_blocked(self) -> None:
        from pre_scan_quality_gate import apply_gate_to_scan_df

        scan = pd.DataFrame(
            [
                {
                    "Symbol": "SOFT",
                    "Mode ID": 2,
                    "Final Score": 95.0,
                    "Tomorrow Pick Score": 96.0,
                    "RSI": 56.0,
                    "Vol / Avg": 1.4,
                }
            ]
        )
        hist = {"SOFT.NS": _ohlcv(np.linspace(100.0, 92.0, 120))}

        out = apply_gate_to_scan_df(scan, hist, mode=2, drop_blocked=False)
        self.assertFalse(bool(out.loc[0, "Gate Blocked"]))
        self.assertLessEqual(float(out.loc[0, "Final Score"]), 72.0)
        self.assertIn("EMA stack bearish", str(out.loc[0, "Gate Reasons"]))

    def test_ai_confidence_caps_tomorrow_score_and_validation_drops_below_60(self) -> None:
        from pre_scan_quality_gate import patch_tomorrow_score, validate_tomorrow_picks

        picks = pd.DataFrame(
            [
                {
                    "Symbol": "LOWAI",
                    "Mode ID": 5,
                    "Tomorrow Pick Score": 100.0,
                    "Final Score": 90.0,
                    "AI Confidence": 29.0,
                    "AI Action": "Buy Tomorrow",
                    "Trap Check": "Clean",
                    "Gate EMA20": 110.0,
                    "Gate EMA50": 100.0,
                    "Gate EMA20 Slope 5D %": 0.4,
                    "Gate 20D Return %": 3.0,
                    "Gate 60D Return %": 5.0,
                    "Gate Vol Ratio": 1.5,
                    "RSI": 58.0,
                }
            ]
        )

        patched = patch_tomorrow_score(picks, mode=5)
        self.assertEqual(float(patched.loc[0, "Tomorrow Pick Score"]), 58.0)
        self.assertTrue(bool(patched.loc[0, "Gate Buy Valid"]))
        self.assertTrue(validate_tomorrow_picks(patched).empty)

    def test_validation_keeps_clean_high_confidence_pick(self) -> None:
        from pre_scan_quality_gate import patch_tomorrow_score, validate_tomorrow_picks

        picks = pd.DataFrame(
            [
                {
                    "Symbol": "GOOD",
                    "Mode ID": 6,
                    "Tomorrow Pick Score": 88.0,
                    "Final Score": 84.0,
                    "AI Confidence": 61.0,
                    "AI Action": "Watch",
                    "Trap Check": "LOW",
                    "Gate EMA20": 110.0,
                    "Gate EMA50": 100.0,
                    "Gate EMA20 Slope 5D %": 0.6,
                    "Gate 20D Return %": 6.0,
                    "Gate 60D Return %": 8.0,
                    "Gate Vol Ratio": 1.6,
                    "RSI": 60.0,
                }
            ]
        )

        patched = patch_tomorrow_score(picks, mode=6)
        valid = validate_tomorrow_picks(patched)
        self.assertEqual(valid["Symbol"].tolist(), ["GOOD"])
        self.assertLessEqual(float(valid.loc[0, "Tomorrow Pick Score"]), 90.0)

    def test_validation_does_not_drop_clean_pick_when_ai_columns_are_missing(self) -> None:
        from pre_scan_quality_gate import validate_tomorrow_picks

        picks = pd.DataFrame(
            [
                {
                    "Symbol": "NOAI",
                    "Tomorrow Pick Score": 72.0,
                    "Trap Check": "Clean",
                    "Gate Buy Valid": True,
                }
            ]
        )

        valid = validate_tomorrow_picks(picks)
        self.assertEqual(valid["Symbol"].tolist(), ["NOAI"])


if __name__ == "__main__":
    unittest.main()
