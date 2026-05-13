from __future__ import annotations

import unittest
from datetime import date

import pandas as pd


class TomorrowAccuracyTop3Tests(unittest.TestCase):
    def test_rank_top3_applies_tomorrow_score_and_hard_eliminations(self) -> None:
        from nse_sentinel_top3 import rank_top3_from_rows

        rows = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Final Score": 80.0,
                    "RSI": 58.0,
                    "Vol / Avg": 2.6,
                    "5D Return (%)": 2.0,
                    "Trap Risk": "LOW",
                    "Action": "Buy Tomorrow",
                    "Δ EMA20 (%)": 2.0,
                    "Δ vs 20D High (%)": -0.8,
                    "Sector Strength": 70.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                    "Hold Days": "2 days",
                },
                {
                    "Ticker": "BBB",
                    "Final Score": 75.0,
                    "RSI": 62.0,
                    "Vol / Avg": 1.9,
                    "5D Return (%)": 4.2,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Δ EMA20 (%)": 2.5,
                    "Δ vs 20D High (%)": -2.5,
                    "Sector Strength": 60.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
                {
                    "Ticker": "CCC",
                    "Final Score": 72.0,
                    "RSI": 66.5,
                    "Vol / Avg": 1.4,
                    "5D Return (%)": 1.5,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Closing Strength": "NEUTRAL",
                },
                {
                    "Ticker": "DDD",
                    "Final Score": 88.0,
                    "RSI": 61.0,
                    "Vol / Avg": 2.1,
                    "5D Return (%)": 2.2,
                    "Trap Risk": "HIGH",
                    "Action": "Watch",
                },
                {
                    "Ticker": "EEE",
                    "Final Score": 60.0,
                    "RSI": 49.0,
                    "Vol / Avg": 1.1,
                    "5D Return (%)": 7.5,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Δ vs 20D High (%)": -5.5,
                    "Regime": "RANGE_BOUND",
                },
            ]
        )

        result = rank_top3_from_rows(rows, as_of=date(2026, 5, 13))

        self.assertEqual([candidate["ticker"] for candidate in result["top"]], ["AAA", "BBB", "CCC"])
        self.assertEqual(result["evaluated"], 5)
        self.assertEqual(result["eliminated"], 1)
        self.assertEqual(result["scored"], 4)
        self.assertEqual(result["skipped"], 1)
        self.assertAlmostEqual(result["top"][0]["tomorrow_score"], 98.0, places=2)
        self.assertAlmostEqual(result["top"][1]["tomorrow_score"], 71.25, places=2)
        self.assertAlmostEqual(result["top"][2]["tomorrow_score"], 58.2, places=2)
        self.assertIn("DDD (Trap Risk HIGH)", result["text"])

    def test_missing_fields_use_prompt_defaults(self) -> None:
        from nse_sentinel_top3 import rank_top3_from_rows

        rows = pd.DataFrame(
            [
                {
                    "Ticker": "MISS",
                    "RSI": 56.0,
                    "Vol / Avg": 1.4,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                }
            ]
        )

        result = rank_top3_from_rows(rows)
        candidate = result["top"][0]

        self.assertAlmostEqual(candidate["base_score"], 55.0, places=2)
        self.assertAlmostEqual(candidate["bridged_base"], 19.25, places=2)
        self.assertEqual(candidate["checks"]["freshness"]["status"], "PARTIAL")
        self.assertEqual(candidate["checks"]["freshness"]["points"], 6)
        self.assertEqual(candidate["checks"]["proximity"]["status"], "PARTIAL")
        self.assertEqual(candidate["checks"]["proximity"]["points"], 6)
        self.assertEqual(candidate["checks"]["sector"]["points"], 3)
        self.assertEqual(candidate["hold_days"], "1-2 days")
        self.assertAlmostEqual(candidate["tomorrow_score"], 55.25, places=2)

    def test_tie_breaker_prefers_fresher_move_within_three_points(self) -> None:
        from nse_sentinel_top3 import rank_top3_from_rows

        rows = pd.DataFrame(
            [
                {
                    "Ticker": "FRESH",
                    "Final Score": 68.0,
                    "RSI": 62.0,
                    "Vol / Avg": 1.9,
                    "5D Return (%)": 1.5,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Δ vs 20D High (%)": -2.5,
                    "Sector Strength": 55.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
                {
                    "Ticker": "LATEISH",
                    "Final Score": 71.0,
                    "RSI": 62.0,
                    "Vol / Avg": 1.9,
                    "5D Return (%)": 5.0,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Δ vs 20D High (%)": -1.0,
                    "Sector Strength": 55.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
            ]
        )

        result = rank_top3_from_rows(rows)

        self.assertEqual([candidate["ticker"] for candidate in result["top"]], ["FRESH", "LATEISH"])
        self.assertLess(
            result["top"][0]["tomorrow_score"],
            result["top"][1]["tomorrow_score"],
        )


if __name__ == "__main__":
    unittest.main()
