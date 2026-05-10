from __future__ import annotations

import hashlib
import os
import pickle
import queue
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _sample_ohlcv(start: str = "2026-05-01", periods: int = 40, close_start: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=periods)
    close = [close_start + float(i) for i in range(periods)]
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


class HardeningRegressionTests(unittest.TestCase):
    def test_time_travel_context_does_not_corrupt_live_cache(self) -> None:
        import time_travel_engine as tt
        from strategy_engines import _engine_utils as eu

        original_all_data = dict(eu.ALL_DATA)
        original_is_fresh = getattr(eu, "_is_data_fresh")
        try:
            eu.ALL_DATA.clear()
            eu._is_data_fresh = lambda df: True  # type: ignore[assignment]
            live_df = _sample_ohlcv(periods=45)
            eu.ALL_DATA["TEST.NS"] = live_df
            tt.clear_cache()

            results: dict[str, int] = {}

            def run_live() -> None:
                frame = eu.get_df_for_ticker("TEST")
                results["live"] = len(frame) if frame is not None else -1

            def run_tt() -> None:
                tt.activate("2026-05-15")
                try:
                    frame = eu.get_df_for_ticker("TEST")
                    results["tt"] = len(frame) if frame is not None else -1
                finally:
                    tt.restore()

            threads = [threading.Thread(target=run_tt), threading.Thread(target=run_live)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertLess(results["tt"], results["live"])
            self.assertEqual(results["live"], len(live_df))
            self.assertEqual(len(eu.ALL_DATA["TEST.NS"]), len(live_df))
        finally:
            tt.restore()
            tt.clear_cache()
            eu.ALL_DATA.clear()
            eu.ALL_DATA.update(original_all_data)
            eu._is_data_fresh = original_is_fresh  # type: ignore[assignment]

    def test_model_pickle_requires_trusted_sha(self) -> None:
        import model_persistence as mp

        tmp = Path(tempfile.mkdtemp())
        old_dir, old_path = mp._DATA_DIR, mp._MODEL_PATH
        old_env = os.environ.pop("NSE_SENTINEL_MODEL_SHA256", None)
        try:
            mp._DATA_DIR = tmp  # type: ignore[assignment]
            mp._MODEL_PATH = tmp / "learning_model.pkl"  # type: ignore[assignment]
            payload = {"model": "m", "scaler": "s"}
            data = pickle.dumps(payload, protocol=4)
            mp._MODEL_PATH.write_bytes(data)

            with self.assertLogs(level="WARNING"):
                self.assertIsNone(mp.load_model())
            os.environ["NSE_SENTINEL_MODEL_SHA256"] = "0" * 64
            with self.assertLogs(level="ERROR"):
                self.assertIsNone(mp.load_model())
            os.environ["NSE_SENTINEL_MODEL_SHA256"] = hashlib.sha256(data).hexdigest()
            self.assertEqual(mp.load_model(), payload)
        finally:
            if old_env is not None:
                os.environ["NSE_SENTINEL_MODEL_SHA256"] = old_env
            else:
                os.environ.pop("NSE_SENTINEL_MODEL_SHA256", None)
            mp._DATA_DIR, mp._MODEL_PATH = old_dir, old_path  # type: ignore[assignment]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persistent_store_queue_full_and_409_merge(self) -> None:
        import persistent_store as ps

        tmp = Path(tempfile.mkdtemp())
        old_queue = ps._PUSH_QUEUE
        old_ensure = ps._ensure_push_worker
        old_secrets = ps._get_secrets
        old_get = ps._gh_get
        old_raw = ps._gh_get_raw
        old_put = ps._gh_put_status
        try:
            ps._PUSH_QUEUE = queue.Queue(maxsize=1)  # type: ignore[assignment]
            ps._PUSH_QUEUE.put_nowait(Path("already-there"))
            ps._ensure_push_worker = lambda: None  # type: ignore[assignment]
            with self.assertLogs("persistent_store", level="ERROR"):
                self.assertFalse(ps.push_file(tmp / "prediction_feedback_log.csv"))

            local = tmp / "prediction_feedback_log.csv"
            local.write_text("id,value\n2,local\n", encoding="utf-8")
            calls: list[bytes] = []

            ps._get_secrets = lambda: {"token": "token", "owner": "owner", "repo": "repo", "branch": "main"}  # type: ignore[assignment]
            ps._gh_get = lambda secrets, path: {"sha": f"sha-{len(calls)}"}  # type: ignore[assignment]
            ps._gh_get_raw = lambda secrets, path: b"id,value\n1,remote\n"  # type: ignore[assignment]

            def fake_put(secrets, path, content, sha=None):
                calls.append(content)
                if len(calls) == 1:
                    return {"ok": False, "status": 409}
                return {"ok": True, "status": 200}

            ps._gh_put_status = fake_put  # type: ignore[assignment]
            self.assertTrue(ps.push_file(local, block=True))
            merged = calls[-1].decode("utf-8")
            self.assertIn("remote", merged)
            self.assertIn("local", merged)
        finally:
            ps._PUSH_QUEUE = old_queue  # type: ignore[assignment]
            ps._ensure_push_worker = old_ensure  # type: ignore[assignment]
            ps._get_secrets = old_secrets  # type: ignore[assignment]
            ps._gh_get = old_get  # type: ignore[assignment]
            ps._gh_get_raw = old_raw  # type: ignore[assignment]
            ps._gh_put_status = old_put  # type: ignore[assignment]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_prediction_feedback_dedupe_backfill_and_ist_market_date(self) -> None:
        import prediction_feedback_store as store

        tmp = Path(tempfile.mkdtemp())
        old_log_path = store.LOG_PATH
        old_push = store._push_file
        try:
            store.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            store._push_file = lambda *a, **kw: True  # type: ignore[assignment]
            store._invalidate_cache()

            row = pd.DataFrame(
                [
                    {
                        "Symbol": "TEST",
                        "Prediction Score": 62.0,
                        "Signal": "Buy",
                        "Logged At": "2026-05-10T18:45:00+00:00",
                        "Import Source": "unit",
                    }
                ]
            )

            threads = [threading.Thread(target=store.log_scan_predictions, args=(row, 2, {})) for _ in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            logged = pd.read_csv(store.LOG_PATH, dtype=str)
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged.loc[0, "market_date"], "2026-05-11")
            self.assertEqual(logged.loc[0, "prediction_direction"], "Bullish")
            self.assertTrue(str(logged.loc[0, "prediction_id"]))

            hist = _sample_ohlcv("2026-05-11", periods=2, close_start=100.0)
            filled = store.backfill_actual_returns({"TEST": hist})
            self.assertEqual(filled, 1)
            backfilled = pd.read_csv(store.LOG_PATH, dtype=str)
            self.assertEqual(backfilled.loc[0, "correct"], "True")
            self.assertEqual(backfilled.loc[0, "target_policy_version"], "stock_next_session_v2")
        finally:
            store.LOG_PATH = old_log_path  # type: ignore[assignment]
            store._push_file = old_push  # type: ignore[assignment]
            store._invalidate_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sector_prediction_duplicate_logging(self) -> None:
        import sector_prediction_tracker as tracker

        tmp = Path(tempfile.mkdtemp())
        old_path = tracker._LOG_PATH
        old_push = tracker._push_file
        try:
            tracker._LOG_PATH = tmp / "sector_predictions.csv"  # type: ignore[assignment]
            tracker._push_file = lambda *a, **kw: True  # type: ignore[assignment]
            tracker._invalidate_log_cache()
            signals = SimpleNamespace(
                ema_slope=50.0,
                price_vs_ema=50.0,
                candle_direction=50.0,
                body_strength=50.0,
                consecutive=50.0,
                volume_confirm=50.0,
                volatility=50.0,
                momentum=50.0,
                sector_strength=50.0,
                bullish_pct=50.0,
                money_flow=50.0,
                participation=50.0,
            )
            prediction = SimpleNamespace(
                sector="IT",
                direction="Bullish",
                confidence=70.0,
                raw_score=65.0,
                signals=signals,
                predicted_at="2026-05-10T18:45:00+00:00",
                entry_price=100.0,
                leader_ticker="TEST",
                stocks_used=[],
                ohlc_source="unit",
                ohlc_symbol="TEST",
            )

            self.assertTrue(tracker.log_prediction(prediction))
            self.assertTrue(tracker.log_prediction(prediction))
            logged = pd.read_csv(tracker._LOG_PATH, dtype=str)
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged.loc[0, "market_date"], "2026-05-11")
            self.assertEqual(logged.loc[0, "prediction_direction"], "Bullish")
        finally:
            tracker._LOG_PATH = old_path  # type: ignore[assignment]
            tracker._push_file = old_push  # type: ignore[assignment]
            tracker._invalidate_log_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_data_ttl_recovery(self) -> None:
        from strategy_engines import _engine_utils as eu

        old_entries = eu._NO_DATA_TICKERS
        try:
            eu._NO_DATA_TICKERS = {"TEST.NS": {"marked_at": 1.0, "reason": "old"}}  # type: ignore[assignment]
            self.assertFalse(eu._has_recent_no_data("TEST.NS"))
            eu._mark_no_data("TEST.NS", "empty")
            self.assertTrue(eu._has_recent_no_data("TEST.NS"))
            eu.clear_no_data_cache("TEST.NS")
            self.assertFalse(eu._has_recent_no_data("TEST.NS"))
        finally:
            eu._NO_DATA_TICKERS = old_entries  # type: ignore[assignment]

    def test_live_breakout_pulse_labels_direct_live_source(self) -> None:
        import data_session_manager as dsm
        import live_breakout_pulse_engine as pulse

        old_window = dsm.get_current_window
        old_universe = pulse._build_live_universe
        old_download = pulse._download_live_batch
        old_score = pulse._score_ticker
        old_shared_ok = pulse._SHARED_DATA_OK
        try:
            dsm.get_current_window = lambda: "LIVE"  # type: ignore[assignment]
            pulse._SHARED_DATA_OK = True  # type: ignore[assignment]
            pulse._build_live_universe = lambda: ["TEST.NS"]  # type: ignore[assignment]
            pulse._download_live_batch = lambda batch, cutoff: {batch[0]: _sample_ohlcv()}  # type: ignore[assignment]
            pulse._score_ticker = lambda ticker, cutoff, df_override=None: {  # type: ignore[assignment]
                "Symbol": ticker.replace(".NS", ""),
                "Final Score": 75.0,
                "Signal": "LIVE BREAKOUT",
            }

            df = pulse.run_live_breakout_pulse()
            self.assertEqual(df.attrs.get("data_source"), "direct_yfinance_live")
            self.assertEqual(df.attrs.get("universe_scanned"), 1)
        finally:
            dsm.get_current_window = old_window  # type: ignore[assignment]
            pulse._build_live_universe = old_universe  # type: ignore[assignment]
            pulse._download_live_batch = old_download  # type: ignore[assignment]
            pulse._score_ticker = old_score  # type: ignore[assignment]
            pulse._SHARED_DATA_OK = old_shared_ok  # type: ignore[assignment]

    def test_snapshot_metadata_partial_corrupt_and_small_valid(self) -> None:
        import data_session_manager as dsm

        tmp = Path(tempfile.mkdtemp())
        old_root, old_archive = dsm._SNAPSHOT_ROOT, dsm._SNAPSHOT_ARCHIVE
        try:
            dsm._SNAPSHOT_ROOT = tmp / "snapshots"  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = tmp / "market_snapshot_latest.zip"  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            snap = dsm.get_snapshot_path("2026-05-08")
            snap.mkdir(parents=True)
            csv_path = snap / "TEST.NS.csv"
            _sample_ohlcv(periods=1).to_csv(csv_path)

            dsm.atomic_write_json(snap / "_meta.json", {"complete": False, "saved": 1, "checksums": {}})
            self.assertFalse(dsm.snapshot_exists("2026-05-08"))

            dsm.atomic_write_json(
                snap / "_meta.json",
                {"complete": True, "saved": 1, "checksums": {csv_path.name: "bad"}},
            )
            dsm._invalidate_snapshot_caches()
            self.assertFalse(dsm.snapshot_exists("2026-05-08"))

            digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            dsm.atomic_write_json(
                snap / "_meta.json",
                {"complete": True, "saved": 1, "checksums": {csv_path.name: digest}},
            )
            dsm._invalidate_snapshot_caches()
            self.assertTrue(dsm.snapshot_exists("2026-05-08"))
        finally:
            dsm._SNAPSHOT_ROOT = old_root  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = old_archive  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
