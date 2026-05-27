from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


def _sample_ohlcv(end: str = "2026-05-11", periods: int = 45) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=periods)
    close = [100.0 + float(i) for i in range(periods)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [value + 1.0 for value in close],
            "Low": [value - 1.0 for value in close],
            "Close": close,
            "Volume": [1000 + i for i in range(periods)],
        },
        index=idx,
    )


class PredictionChartDataRegressionTests(unittest.TestCase):
    def test_source_context_keeps_explicit_ai_picks(self) -> None:
        import app_prediction_chart_section as chart

        self.assertEqual(
            chart._pick_source_mode_from_context("AI", "Tomorrow's Picks - Swing"),
            "AI",
        )
        self.assertEqual(
            chart._pick_source_mode_from_context("Manual", "A-I-L IN ONE - Final Aura Verdict"),
            "AI",
        )

    def test_feedback_logger_infers_ai_source_from_ail_context(self) -> None:
        import prediction_feedback_store as pfs

        tmp = Path(tempfile.mkdtemp())
        old_data_dir = pfs.DATA_DIR
        old_log_path = pfs.LOG_PATH
        old_push_file = pfs._push_file
        try:
            pfs.DATA_DIR = tmp  # type: ignore[assignment]
            pfs.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            pfs._push_file = lambda *_args, **_kwargs: True  # type: ignore[assignment]
            pfs._invalidate_cache()

            frame = pd.DataFrame(
                [
                    {
                        "Symbol": "AILCTX",
                        "Import Source": "A-I-L IN ONE - Final Aura Verdict",
                        "Import Category": "Momentum",
                        "Prediction Score": 72.0,
                        "Final Score": 74.0,
                        "Signal": "BUY",
                    }
                ]
            )

            pfs.log_scan_predictions(frame, 7, {"bias": "Bullish", "regime": "TRENDING_UP"})
            logged = pfs.read_feedback_log()

            self.assertEqual(logged.loc[0, "source_mode"], "AI")
        finally:
            pfs.DATA_DIR = old_data_dir  # type: ignore[assignment]
            pfs.LOG_PATH = old_log_path  # type: ignore[assignment]
            pfs._push_file = old_push_file  # type: ignore[assignment]
            pfs._invalidate_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_status_badge_helper_does_not_abort_chart_render(self) -> None:
        import feature_data_manager as fdm

        fdm.render_data_status_badge(
            {
                "source_kind": "SNAPSHOT",
                "source": "snapshot",
                "window": "CLOSED",
                "market_date": "2026-05-11",
                "as_of": "11 May 2026 20:15 IST",
                "note": "",
            },
            label="TEST",
        )

    def test_closed_window_uses_legacy_csv_cache_for_chart_data(self) -> None:
        import data_downloader
        import feature_data_manager as fdm

        tmp = Path(tempfile.mkdtemp())
        old_data_dir = data_downloader.DATA_DIR
        old_snapshot_root = fdm._SCANNER_SNAPSHOT_ROOT
        old_yf = fdm.yf
        old_window = fdm._get_real_current_window
        old_expected = fdm._get_real_expected_data_date
        old_all_data = dict(fdm.ALL_DATA)
        try:
            csv_dir = tmp / "data"
            csv_dir.mkdir()
            data_downloader.DATA_DIR = csv_dir
            fdm._SCANNER_SNAPSHOT_ROOT = tmp / "snapshots"
            fdm._SCANNER_SNAPSHOT_ROOT.mkdir()
            fdm.yf = None
            fdm._get_real_current_window = lambda: "CLOSED"  # type: ignore[assignment]
            fdm._get_real_expected_data_date = lambda: date(2026, 5, 12)  # type: ignore[assignment]
            fdm.ALL_DATA.clear()

            _sample_ohlcv().to_csv(csv_dir / "TEST.NS.csv")
            manager = fdm.FeatureDataManager(cache_root=tmp / "feature_cache")

            df = manager.get_stock_data("TEST", period="3mo", interval="1d")

            self.assertIsNotNone(df)
            self.assertGreaterEqual(len(df), 30)
            self.assertEqual(manager.get_last_status("TEST.NS")["source"], "csv_cache")
        finally:
            data_downloader.DATA_DIR = old_data_dir
            fdm._SCANNER_SNAPSHOT_ROOT = old_snapshot_root
            fdm.yf = old_yf
            fdm._get_real_current_window = old_window  # type: ignore[assignment]
            fdm._get_real_expected_data_date = old_expected  # type: ignore[assignment]
            fdm.ALL_DATA.clear()
            fdm.ALL_DATA.update(old_all_data)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_closed_window_missing_snapshot_still_reaches_final_fallback(self) -> None:
        import feature_data_manager as fdm

        tmp = Path(tempfile.mkdtemp())
        old_snapshot_root = fdm._SCANNER_SNAPSHOT_ROOT
        old_window = fdm._get_real_current_window
        old_expected = fdm._get_real_expected_data_date
        old_all_data = dict(fdm.ALL_DATA)
        old_fetch = fdm.FeatureDataManager._fetch_yfinance
        try:
            cache_day = date(2026, 5, 12)
            fdm._SCANNER_SNAPSHOT_ROOT = tmp / "snapshots"
            (fdm._SCANNER_SNAPSHOT_ROOT / cache_day.isoformat()).mkdir(parents=True)
            fdm._get_real_current_window = lambda: "CLOSED"  # type: ignore[assignment]
            fdm._get_real_expected_data_date = lambda: cache_day  # type: ignore[assignment]
            fdm.ALL_DATA.clear()

            fallback = _sample_ohlcv()

            def fake_fetch(self, symbol, period, interval, min_rows=5, *, cutoff=None):
                return fallback.copy()

            fdm.FeatureDataManager._fetch_yfinance = fake_fetch  # type: ignore[assignment]
            manager = fdm.FeatureDataManager(cache_root=tmp / "feature_cache")

            df = manager.get_stock_data("TEST", period="3mo", interval="1d")

            self.assertIsNotNone(df)
            self.assertGreaterEqual(len(df), 30)
            status = manager.get_last_status("TEST.NS")
            self.assertEqual(status["source"], "live_feature")
            self.assertIn("final chart-data fallback", status["note"])
        finally:
            fdm._SCANNER_SNAPSHOT_ROOT = old_snapshot_root
            fdm._get_real_current_window = old_window  # type: ignore[assignment]
            fdm._get_real_expected_data_date = old_expected  # type: ignore[assignment]
            fdm.FeatureDataManager._fetch_yfinance = old_fetch  # type: ignore[assignment]
            fdm.ALL_DATA.clear()
            fdm.ALL_DATA.update(old_all_data)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
