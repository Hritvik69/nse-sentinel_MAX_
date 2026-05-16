from __future__ import annotations

import hashlib
import base64
import io
import json
import os
import pickle
import queue
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
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

    def test_persistent_store_keyed_conflict_merges_preserve_validations(self) -> None:
        import persistent_store as ps

        remote = (
            "prediction_id,symbol,mode,market_date,import_source,prediction_direction,actual_next_return_pct,correct,outcome_label\n"
            "p1,TEST,2,2026-05-08,scan,Bullish,1.2300,True,correct\n"
            "p2,OLD,2,2026-05-08,scan,Bearish,-0.5000,True,correct\n"
        ).encode("utf-8")
        local = (
            "prediction_id,symbol,mode,market_date,import_source,prediction_direction,actual_next_return_pct,correct,outcome_label,final_score\n"
            "p1,TEST,2,2026-05-08,scan,Bullish,,,,72.5\n"
            "p3,LOCAL,2,2026-05-09,scan,Bullish,,,,80.0\n"
        ).encode("utf-8")

        merged_bytes = ps._merge_remote_local("data/prediction_feedback_log.csv", remote, local)
        self.assertIsNotNone(merged_bytes)
        merged = pd.read_csv(io.BytesIO(merged_bytes), dtype=str, keep_default_na=False)
        self.assertEqual(len(merged), 3)
        p1 = merged[merged["prediction_id"] == "p1"].iloc[0]
        self.assertEqual(p1["actual_next_return_pct"], "1.2300")
        self.assertEqual(p1["correct"], "True")
        self.assertEqual(p1["outcome_label"], "correct")
        self.assertEqual(p1["final_score"], "72.5")

        sector_remote = (
            "prediction_id,sector,market_date,ohlc_source,prediction_direction,exit_price,return_pct,correct\n"
            "s1,IT,2026-05-08,weighted_sector_basket,Bullish,101.0,1.0,True\n"
        ).encode("utf-8")
        sector_local = (
            "prediction_id,sector,market_date,ohlc_source,prediction_direction,exit_price,return_pct,correct,confidence\n"
            "s1,IT,2026-05-08,weighted_sector_basket,Bullish,,,,75\n"
            "s2,AUTO,2026-05-08,leader,Bearish,,,,60\n"
        ).encode("utf-8")
        sector_bytes = ps._merge_remote_local("data/sector_prediction_log.csv", sector_remote, sector_local)
        sector = pd.read_csv(io.BytesIO(sector_bytes), dtype=str, keep_default_na=False)
        self.assertEqual(len(sector), 2)
        s1 = sector[sector["prediction_id"] == "s1"].iloc[0]
        self.assertEqual(s1["exit_price"], "101.0")
        self.assertEqual(s1["correct"], "True")

        perf_remote = b"signal_name,observations,wins,win_rate,last_updated,dynamic_weight\nema_slope,10,6,0.6000,old,0.1\n"
        perf_local = b"signal_name,observations,wins,win_rate,last_updated,dynamic_weight\nema_slope,12,7,0.5833,new,0.11\nmomentum,3,2,0.6667,new,0.05\n"
        perf_bytes = ps._merge_remote_local("data/sector_signal_performance.csv", perf_remote, perf_local)
        perf = pd.read_csv(io.BytesIO(perf_bytes), dtype=str, keep_default_na=False)
        self.assertEqual(set(perf["signal_name"]), {"ema_slope", "momentum"})
        ema = perf[perf["signal_name"] == "ema_slope"].iloc[0]
        self.assertEqual(ema["observations"], "12")
        self.assertEqual(ema["wins"], "7")

    def test_persistent_store_pull_merges_remote_with_newer_local(self) -> None:
        import persistent_store as ps

        tmp = Path(tempfile.mkdtemp())
        old_data_dir = ps._DATA_DIR
        old_sync = ps._SYNC_FILES
        old_secrets = ps._get_secrets
        old_get = ps._gh_get
        old_raw = ps._gh_get_raw
        try:
            ps._DATA_DIR = tmp  # type: ignore[assignment]
            ps._SYNC_FILES = {"prediction_feedback_log.csv": "data/prediction_feedback_log.csv"}  # type: ignore[assignment]
            local_path = tmp / "prediction_feedback_log.csv"
            local_path.write_text(
                "prediction_id,symbol,mode,market_date,import_source,prediction_direction,actual_next_return_pct,correct\n"
                "p1,TEST,2,2026-05-08,scan,Bullish,2.0000,True\n"
                "p_local,LOCAL,2,2026-05-09,scan,Bullish,,\n",
                encoding="utf-8",
            )
            remote = (
                "prediction_id,symbol,mode,market_date,import_source,prediction_direction,actual_next_return_pct,correct\n"
                "p1,TEST,2,2026-05-08,scan,Bullish,,\n"
                "p_remote,REMOTE,2,2026-05-07,scan,Bearish,-1.0000,True\n"
            ).encode("utf-8")
            ps._get_secrets = lambda: {"token": "t", "owner": "o", "repo": "r", "branch": "main"}  # type: ignore[assignment]
            ps._gh_get = lambda secrets, path: {"content": base64.b64encode(remote).decode("ascii")}  # type: ignore[assignment]
            ps._gh_get_raw = lambda secrets, path: None  # type: ignore[assignment]

            self.assertEqual(ps.pull_all(), 1)
            pulled = pd.read_csv(local_path, dtype=str, keep_default_na=False)
            self.assertEqual(set(pulled["prediction_id"]), {"p1", "p_local", "p_remote"})
            p1 = pulled[pulled["prediction_id"] == "p1"].iloc[0]
            self.assertEqual(p1["actual_next_return_pct"], "2.0000")
            self.assertEqual(p1["correct"], "True")
        finally:
            ps._DATA_DIR = old_data_dir  # type: ignore[assignment]
            ps._SYNC_FILES = old_sync  # type: ignore[assignment]
            ps._get_secrets = old_secrets  # type: ignore[assignment]
            ps._gh_get = old_get  # type: ignore[assignment]
            ps._gh_get_raw = old_raw  # type: ignore[assignment]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persistent_store_nested_json_merge(self) -> None:
        import persistent_store as ps

        remote = {
            "picks": ["AAA", "BBB"],
            "notes": "remote note",
            "sections": {"breakout": ["AAA"], "swing": ["BBB"]},
            "records": [{"ticker": "AAA", "categories": ["Remote"], "snapshot": {"score": 1}}],
        }
        local = {
            "picks": ["BBB", "CCC"],
            "notes": "",
            "sections": {"breakout": ["CCC"]},
            "records": [{"ticker": "AAA", "categories": ["Local"], "snapshot": {"signal": "BUY"}}],
        }
        merged_bytes = ps._merge_remote_local(
            "data/tomorrow_picks_store.json",
            json.dumps(remote).encode("utf-8"),
            json.dumps(local).encode("utf-8"),
        )
        merged = json.loads(merged_bytes.decode("utf-8"))
        self.assertEqual(merged["notes"], "remote note")
        self.assertEqual(merged["picks"], ["AAA", "BBB", "CCC"])
        self.assertEqual(merged["sections"]["breakout"], ["AAA", "CCC"])
        self.assertEqual(merged["records"][0]["categories"], ["Remote", "Local"])
        self.assertEqual(merged["records"][0]["snapshot"]["score"], 1)
        self.assertEqual(merged["records"][0]["snapshot"]["signal"], "BUY")

    def test_snapshot_save_is_thread_safe_and_archives_atomically(self) -> None:
        import data_session_manager as dsm

        tmp = Path(tempfile.mkdtemp())
        old_root = dsm._SNAPSHOT_ROOT
        old_archive = dsm._SNAPSHOT_ARCHIVE
        old_window = dsm.get_current_window
        try:
            dsm._SNAPSHOT_ROOT = tmp / "snapshots"  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = tmp / "market_snapshot_latest.zip"  # type: ignore[assignment]
            dsm.get_current_window = lambda: "CLOSED"  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            snap_day = date(2026, 5, 8)
            data = {"TEST.NS": _sample_ohlcv("2026-05-01", periods=6)}

            results: list[int] = []

            def save_once() -> int:
                return dsm.save_closing_snapshot(data, snap_day)

            with ThreadPoolExecutor(max_workers=5) as executor:
                for future in as_completed([executor.submit(save_once) for _ in range(5)]):
                    results.append(future.result())

            self.assertEqual(sum(results), 1)
            self.assertTrue(dsm.snapshot_exists(snap_day))
            self.assertTrue(dsm._SNAPSHOT_ARCHIVE.exists())
            leftovers = list((tmp / "snapshots").glob(".2026-05-08.*.tmp"))
            self.assertEqual(leftovers, [])
        finally:
            dsm._SNAPSHOT_ROOT = old_root  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = old_archive  # type: ignore[assignment]
            dsm.get_current_window = old_window  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_time_travel_cutoff_fails_closed_and_handles_timezone_index(self) -> None:
        import time_travel_engine as tt

        malformed = _sample_ohlcv(periods=20)
        malformed.index = ["bad-index"] * len(malformed)
        self.assertIsNone(tt.truncate_df(malformed, date(2026, 5, 8), min_rows=5))

        tz_df = _sample_ohlcv("2026-05-01", periods=10)
        tz_df.index = pd.date_range("2026-05-01", periods=10, freq="B", tz="Asia/Kolkata")
        truncated = tt.truncate_df(tz_df, date(2026, 5, 8), min_rows=5)
        self.assertIsNotNone(truncated)
        self.assertLessEqual(pd.to_datetime(truncated.index[-1]).date(), date(2026, 5, 8))

    def test_breakout_and_feature_time_travel_cutoffs_fail_closed(self) -> None:
        import breakout_radar_engine as radar
        import feature_data_manager as fdm

        malformed = _sample_ohlcv(periods=60)
        malformed.index = ["not-a-date"] * len(malformed)
        self.assertIsNone(radar._strict_time_travel_slice(malformed, date(2026, 5, 8), min_rows=45))

        manager = fdm.FeatureDataManager(cache_root=Path(tempfile.mkdtemp()))
        try:
            self.assertIsNone(
                manager._apply_time_travel_cutoff(malformed, cutoff=date(2026, 5, 8), min_rows=5)
            )
        finally:
            shutil.rmtree(manager.cache_root, ignore_errors=True)

    def test_ticker_universe_does_not_cache_degraded_then_recovers(self) -> None:
        import nse_ticker_universe as uni

        old_build = uni._build
        old_tmp = uni._TMP_TICKER_FILE
        tmp = Path(tempfile.mkdtemp())
        try:
            uni._TMP_TICKER_FILE = str(tmp / "tickers.txt")  # type: ignore[assignment]
            uni.invalidate_cache(clear_disk=True)
            calls = {"n": 0}

            def fake_build(live: bool) -> list[str]:
                calls["n"] += 1
                if calls["n"] == 1:
                    return [f"LOW{i}.NS" for i in range(10)]
                return [f"FULL{i}.NS" for i in range(2000)]

            uni._build = fake_build  # type: ignore[assignment]
            self.assertEqual(len(uni.get_all_tickers(live=True)), 10)
            self.assertNotIn(True, uni._cache)
            self.assertEqual(len(uni.get_all_tickers(live=True)), 2000)
            self.assertIn(True, uni._cache)
        finally:
            uni._build = old_build  # type: ignore[assignment]
            uni._TMP_TICKER_FILE = old_tmp  # type: ignore[assignment]
            uni.invalidate_cache(clear_disk=True)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_feature_cache_rejects_short_period_and_corrupt_snapshot(self) -> None:
        import data_session_manager as dsm
        import feature_data_manager as fdm

        tmp = Path(tempfile.mkdtemp())
        old_snapshot_root = fdm._SCANNER_SNAPSHOT_ROOT
        old_dsm_root = dsm._SNAPSHOT_ROOT
        old_dsm_archive = dsm._SNAPSHOT_ARCHIVE
        try:
            manager = fdm.FeatureDataManager(cache_root=tmp / "feature_cache")
            cache_day = date(2026, 5, 8)
            short_df = _sample_ohlcv("2026-04-01", periods=25)
            manager._save_stock_cache(
                "TEST.NS",
                short_df,
                period="2mo",
                interval="1d",
                source="unit",
                cache_day=cache_day,
            )
            loaded, _ = manager._load_stock_cache(
                "TEST.NS",
                period="1y",
                interval="1d",
                cache_day=cache_day,
                min_rows=5,
            )
            self.assertIsNone(loaded)

            fdm._SCANNER_SNAPSHOT_ROOT = tmp / "snapshots"  # type: ignore[assignment]
            dsm._SNAPSHOT_ROOT = fdm._SCANNER_SNAPSHOT_ROOT  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = tmp / "market_snapshot_latest.zip"  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            snap_dir = fdm._SCANNER_SNAPSHOT_ROOT / cache_day.isoformat()
            snap_dir.mkdir(parents=True)
            csv_path = snap_dir / "TEST.NS.csv"
            _sample_ohlcv("2026-05-01", periods=6).to_csv(csv_path)
            dsm.atomic_write_json(
                snap_dir / "_meta.json",
                {"complete": True, "saved": 1, "checksums": {csv_path.name: "bad"}},
            )
            loaded_snapshot, _ = manager._load_scanner_snapshot("TEST.NS", cache_day, min_rows=5)
            self.assertIsNone(loaded_snapshot)
        finally:
            fdm._SCANNER_SNAPSHOT_ROOT = old_snapshot_root  # type: ignore[assignment]
            dsm._SNAPSHOT_ROOT = old_dsm_root  # type: ignore[assignment]
            dsm._SNAPSHOT_ARCHIVE = old_dsm_archive  # type: ignore[assignment]
            dsm._invalidate_snapshot_caches()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_cache_uses_hash_path_and_stores_symbols(self) -> None:
        import feature_data_manager as fdm

        tmp = Path(tempfile.mkdtemp())
        old_expected_date = fdm.get_expected_data_date
        try:
            fdm.get_expected_data_date = lambda: date(2026, 5, 8)  # type: ignore[assignment]
            manager = fdm.FeatureDataManager(cache_root=tmp)
            symbols = [f"VERYLONGSYMBOLNAME{i:03d}.NS" for i in range(80)]
            payload = {"battle_df": "{}"}
            manager.save_compare_cache(symbols, payload)
            compare_dir = manager._compare_dir(manager._cache_day())
            files = list(compare_dir.glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertLess(len(files[0].name), 80)
            saved = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(saved["symbols"], manager._normalize_compare_symbols(symbols))
            self.assertIsNotNone(manager.load_compare_cache(list(reversed(symbols))))
        finally:
            fdm.get_expected_data_date = old_expected_date  # type: ignore[assignment]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_import_prefers_saved_tomorrow_store_over_stale_visible_symbols(self) -> None:
        from app_compare_stocks_section import select_compare_tomorrow_import_symbols

        saved_store = {
            "picks": [
                "ADANIPORTS",
                "GMRP&UI",
                "BUILDPRO",
                "DRREDDY",
                "ASKAUTOLTD",
                "INSECTICID",
                "ALPA",
                "KIRLOSENG",
            ],
            "sections": {
                "relax": ["ADANIPORTS"],
                "swing": [],
                "intraday": ["GMRP&UI", "BUILDPRO", "DRREDDY"],
                "momentum": ["ASKAUTOLTD", "INSECTICID", "ALPA"],
                "breakout": ["KIRLOSENG"],
            },
        }
        stale_visible = [f"RANDOM{i}" for i in range(19)]

        self.assertEqual(
            select_compare_tomorrow_import_symbols(saved_store, stale_visible, limit=19),
            [
                "ADANIPORTS",
                "GMRP&UI",
                "BUILDPRO",
                "DRREDDY",
                "ASKAUTOLTD",
                "INSECTICID",
                "ALPA",
                "KIRLOSENG",
            ],
        )

    def test_ail_dataframe_omits_none_height_for_streamlit_cloud(self) -> None:
        import app_ail_in_one_section as ail_ui

        captured: dict[str, object] = {}
        old_dataframe = ail_ui.st.dataframe

        def fake_dataframe(*_args, **kwargs):
            captured.update(kwargs)

        try:
            ail_ui.st.dataframe = fake_dataframe  # type: ignore[assignment]
            ail_ui._render_dataframe(pd.DataFrame([{"Mode": 1, "Raw Hits": 3}]), ["Mode", "Raw Hits"])
        finally:
            ail_ui.st.dataframe = old_dataframe  # type: ignore[assignment]

        self.assertEqual(captured.get("width"), "stretch")
        self.assertTrue(captured.get("hide_index"))
        self.assertNotIn("height", captured)

    def test_ail_category_mapping_uses_mode_registry_truth(self) -> None:
        from ail_in_one_engine import classify_scan_results

        categories = classify_scan_results(
            pd.DataFrame(
                [
                    {"Symbol": "M1MOM", "Mode ID": 1},
                    {"Symbol": "M2BAL", "Mode ID": 2},
                    {"Symbol": "M3RELAX", "Mode ID": 3},
                    {"Symbol": "M4INST", "Mode ID": 4},
                    {"Symbol": "M5INTRA", "Mode ID": 5},
                    {"Symbol": "M6SWING", "Mode ID": 6},
                    {"Symbol": "M7MOM", "Mode ID": 7},
                ]
            )
        )

        def symbols(name: str) -> set[str]:
            frame = categories.get(name, pd.DataFrame())
            return set(frame["Symbol"].tolist()) if isinstance(frame, pd.DataFrame) and "Symbol" in frame else set()

        selfEqual = self.assertEqual
        selfEqual(symbols("Relaxed"), {"M3RELAX"})
        selfEqual(symbols("Intraday"), {"M5INTRA"})
        selfEqual(symbols("Momentum"), {"M1MOM", "M7MOM"})
        selfEqual(symbols("Swing"), {"M6SWING"})
        selfEqual(symbols("Institutional"), {"M4INST"})
        self.assertNotIn("M2BAL", symbols("Swing"))

    def test_ail_confidence_uses_real_components_without_fallback_label(self) -> None:
        from ail_confidence_engine import compute_smart_confidence

        row = {
            "Symbol": "AAA",
            "Prediction Score": 72,
            "Confidence": 66,
            "Trap Risk Score": 34,
            "Setup Cleanliness": 70,
            "Volume Quality": 74,
            "Regime Alignment": 68,
            "Sector Support": 65,
            "Learned Prob %": 61,
        }
        confidence = compute_smart_confidence(row, {"market_bias": {"bias": "Bullish", "regime": "Trending Up"}})
        self.assertGreater(confidence["score"], 55)
        self.assertNotIn("Fallback", confidence["label"])
        self.assertIn("prediction", confidence["components"])
        self.assertGreater(confidence["coverage"], 50)

        thin = compute_smart_confidence({"Symbol": "EMPTY"}, {})
        self.assertEqual(thin["score"], 0.0)
        self.assertEqual(thin["label"], "Insufficient evidence")

    def test_ail_diversity_role_selection_avoids_single_stock_sweep(self) -> None:
        from ail_ranking_engine import build_master_rankings, select_category_leaders

        df = pd.DataFrame(
            [
                {
                    "Symbol": "AAA",
                    "AIL Categories": "Momentum, Breakout",
                    "Mode ID": 1,
                    "AIL Master Score": 82.0,
                    "AIL Confidence": 75.0,
                    "AIL Risk Adjusted Score": 70.0,
                    "Momentum Quality": 88.0,
                    "Volume Quality": 78.0,
                    "Trap Risk Score": 42.0,
                    "Setup Cleanliness": 70.0,
                    "Regime Alignment": 72.0,
                    "Sector Support": 68.0,
                    "Risk Reward Score": 72.0,
                    "RSI": 64.0,
                },
                {
                    "Symbol": "BBB",
                    "AIL Categories": "Swing, Relaxed",
                    "Mode ID": 6,
                    "AIL Master Score": 79.0,
                    "AIL Confidence": 73.0,
                    "AIL Risk Adjusted Score": 78.0,
                    "Momentum Quality": 68.0,
                    "Volume Quality": 70.0,
                    "Trap Risk Score": 28.0,
                    "Setup Cleanliness": 84.0,
                    "Regime Alignment": 68.0,
                    "Sector Support": 66.0,
                    "Risk Reward Score": 80.0,
                    "RSI": 54.0,
                    "Entry Timing": "Early pullback",
                },
                {
                    "Symbol": "CCC",
                    "AIL Categories": "Institutional",
                    "Mode ID": 4,
                    "AIL Master Score": 77.0,
                    "AIL Confidence": 71.0,
                    "AIL Risk Adjusted Score": 74.0,
                    "Momentum Quality": 65.0,
                    "Volume Quality": 72.0,
                    "Trap Risk Score": 35.0,
                    "Setup Cleanliness": 76.0,
                    "Regime Alignment": 86.0,
                    "Sector Support": 82.0,
                    "Risk Reward Score": 73.0,
                    "RSI": 58.0,
                },
            ]
        )
        ranked = build_master_rankings(df, market_bias={"bias": "Bullish", "regime": "Trending Up"})
        summary = select_category_leaders(ranked)
        winners = [item["symbol"] for item in summary.values() if item.get("symbol")]
        self.assertGreaterEqual(len(set(winners)), 2)
        self.assertIn("AAA", winners)
        self.assertIn("BBB", winners)

    def test_ail_pipeline_orchestrates_modes_battle_aura_and_logging(self) -> None:
        from ail_in_one_engine import AIL_CATEGORY_ORDER, AIL_MODES, run_ail_pipeline

        tickers = ["AAA", "BBB", "NOHIST"]
        all_data = {
            "AAA.NS": _sample_ohlcv(periods=45),
            "BBB.NS": _sample_ohlcv(periods=45, close_start=120.0),
        }
        scan_calls: list[int] = []
        battle_calls: list[int] = []
        aura_calls: list[str] = []
        logged_frames: list[pd.DataFrame] = []

        def fake_scan(symbols, mode, workers=12):
            scan_calls.append(int(mode))
            rows = []
            for idx, symbol in enumerate(symbols):
                rows.append(
                    {
                        "Symbol": symbol,
                        "Final Score": 72 + idx + int(mode),
                        "Prediction Score": 66 + idx + int(mode),
                        "Confidence": 60 + idx,
                        "RSI": 54 + idx,
                        "Vol / Avg": 1.4,
                        "Delta vs 20D High (%)": -1.0,
                        "Signal": "BUY",
                    }
                )
            return rows, 0.01

        def fake_enhance(rows, mode):
            return pd.DataFrame(rows)

        def fake_battle(df, **_kwargs):
            battle_calls.append(len(df))
            out = df.copy()
            base = pd.to_numeric(out.get("Prediction Score", 0), errors="coerce").fillna(0)
            out["Smart Potential Score"] = base + 5
            out["Bullish Probability"] = base
            out["Smart Confidence"] = pd.to_numeric(out.get("Confidence", 0), errors="coerce").fillna(0)
            out["Setup Cleanliness"] = 72
            out["Momentum Quality"] = 74
            out["Volume Quality"] = 76
            out["Trap Risk Score"] = 38
            out["Regime Alignment"] = 65
            out["Risk Reward Score"] = 68
            out["Smart Notes"] = "Injected battle score"
            return out

        class FakeAura:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol
                self.verdict = "BUY TOMORROW"
                self.timing = "Tomorrow"
                self.aura_score = 79.0
                self.timing_reason = "Injected aura"
                self.entry_low = 100.0
                self.entry_high = 102.0
                self.sl_price = 96.0
                self.sl_pct = 4.0
                self.target1 = 106.0
                self.target2 = 110.0
                self.rr_ratio = 2.0
                self.market_note = "OK"
                self.reasons_positive = ["trend"]
                self.reasons_warning = []
                self.reasons_reject = []

        def fake_aura(_hist, symbol, _market_bias):
            aura_calls.append(symbol)
            return FakeAura(symbol)

        def fake_log(df, _mode, _market_bias):
            logged_frames.append(df.copy())

        result = run_ail_pipeline(
            tickers,
            run_scan_fn=fake_scan,
            enhance_results_fn=fake_enhance,
            compute_market_bias_fn=lambda: {"bias": "Bullish", "regime": "TRENDING_UP"},
            compute_battle_scores_fn=fake_battle,
            run_aura_engine_fn=fake_aura,
            log_scan_predictions_fn=fake_log,
            all_data=all_data,
        )

        self.assertEqual(scan_calls, list(AIL_MODES))
        self.assertTrue(all(len(result.category_top3[name].get("top_df", pd.DataFrame())) > 0 for name in AIL_CATEGORY_ORDER))
        self.assertEqual(len(battle_calls), 1)
        self.assertIn("AIL Master Score", result.final_ranked_df.columns)
        self.assertIn("AIL Confidence", result.final_ranked_df.columns)
        self.assertTrue((pd.to_numeric(result.final_ranked_df["AIL Confidence"], errors="coerce").fillna(0) > 0).any())
        self.assertTrue(result.final_ranked_df["Smart Notes"].astype(str).str.contains("Injected battle score").any())
        for payload in result.category_top3.values():
            frame = payload.get("top_df", pd.DataFrame())
            if isinstance(frame, pd.DataFrame) and not frame.empty and "AIL Top3 Confidence" in frame.columns:
                self.assertFalse(frame["AIL Top3 Confidence"].astype(str).str.contains("Fallback", case=False).any())
        self.assertTrue(set(aura_calls).issubset({"AAA", "BBB"}))
        self.assertNotIn("NOHIST", aura_calls)
        self.assertEqual(len(logged_frames), 1)
        self.assertTrue((logged_frames[0]["Import Source"] == "A-I-L IN ONE").all())
        self.assertEqual(result.health.get("modes_scanned"), 7)
        self.assertEqual(result.health.get("raw_hits"), 21)
        self.assertEqual(result.health.get("enhanced_candidates"), 21)
        self.assertEqual(result.health.get("aura_verdicts"), len(aura_calls))
        self.assertEqual(result.health.get("logged_predictions"), len(result.final_ranked_df))

    def test_ail_feedback_logging_dedupes_and_tags_import_source(self) -> None:
        import ail_in_one_engine as ail
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
            final_df = pd.DataFrame(
                [
                    {
                        "Symbol": "AAA",
                        "Mode ID": 3,
                        "Prediction Score": 68.0,
                        "Final Score": 74.0,
                        "Signal": "BUY",
                        "Sector": "TEST",
                    }
                ]
            )

            first_count, first_error = ail.log_ail_predictions(final_df, market_bias={"bias": "Bullish"})
            second_count, second_error = ail.log_ail_predictions(final_df, market_bias={"bias": "Bullish"})
            logged = pfs.read_feedback_log()

            self.assertEqual(first_error, "")
            self.assertEqual(second_error, "")
            self.assertEqual(first_count, 1)
            self.assertEqual(second_count, 0)
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged.loc[0, "import_source"], "A-I-L IN ONE")
            self.assertEqual(logged.loc[0, "import_category"], "Master Ranking")
        finally:
            pfs.DATA_DIR = old_data_dir  # type: ignore[assignment]
            pfs.LOG_PATH = old_log_path  # type: ignore[assignment]
            pfs._push_file = old_push_file  # type: ignore[assignment]
            pfs._invalidate_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_live_breakout_direct_download_uses_bounded_limiter(self) -> None:
        import live_breakout_pulse_engine as pulse

        old_sem = pulse._SHARED_YF_SEM
        old_download = pulse.yf.download
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_download(*args, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                return _sample_ohlcv(periods=40)
            finally:
                with lock:
                    active -= 1

        try:
            pulse._SHARED_YF_SEM = threading.BoundedSemaphore(2)
            pulse.yf.download = fake_download  # type: ignore[assignment]
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(lambda i: pulse._download_live(f"TEST{i}.NS", None), range(6)))
            self.assertLessEqual(max_active, 2)
        finally:
            pulse._SHARED_YF_SEM = old_sem
            pulse.yf.download = old_download  # type: ignore[assignment]

    def test_live_breakout_price_alias_contract(self) -> None:
        import live_breakout_pulse_engine as pulse

        df = _sample_ohlcv("2026-04-01", periods=60)
        df["Close"] = [100 + i * 0.8 for i in range(len(df))]
        df["Open"] = df["Close"] - 0.5
        df["High"] = df["Close"] + 1.0
        df["Low"] = df["Close"] - 1.0
        df["Volume"] = 1000
        df.iloc[-1, df.columns.get_loc("Volume")] = 5000
        row = pulse._score_ticker("TEST.NS", None, df_override=df)
        self.assertIsNotNone(row)
        self.assertIn("Price (\u20b9)", row)
        legacy_alias = "Price (\u00e2\u201a\u00b9)"
        double_encoded_alias = "Price (\u00c3\u00a2\u00e2\u20ac\u0161\u00c2\u00b9)"
        self.assertIn(legacy_alias, row)
        self.assertIn(double_encoded_alias, row)
        self.assertEqual(row["Price (\u20b9)"], row[legacy_alias])


if __name__ == "__main__":
    unittest.main()
