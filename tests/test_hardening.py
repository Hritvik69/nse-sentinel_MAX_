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
from datetime import date, datetime
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
        original_download = getattr(eu, "download_history")
        original_scan_plan = getattr(eu, "_get_scan_data_plan")
        try:
            eu.ALL_DATA.clear()
            eu._is_data_fresh = lambda df: True  # type: ignore[assignment]
            eu.download_history = lambda *_args, **_kwargs: None  # type: ignore[assignment]
            eu._get_scan_data_plan = lambda: {"force_live_refresh": False, "use_snapshot": False}  # type: ignore[assignment]
            live_df = _sample_ohlcv(periods=45)
            eu.ALL_DATA["TEST"] = live_df
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
            eu.download_history = original_download  # type: ignore[assignment]
            eu._get_scan_data_plan = original_scan_plan  # type: ignore[assignment]

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
            ps._PENDING_PUSHES.clear()
            ps._PUSH_QUEUE = queue.Queue(maxsize=1)  # type: ignore[assignment]
            ps._PUSH_QUEUE.put_nowait(Path("already-there"))
            ps._ensure_push_worker = lambda: None  # type: ignore[assignment]
            self.assertTrue(ps.push_file(tmp / "prediction_feedback_log.csv"))
            self.assertEqual(len(ps._PENDING_PUSHES), 1)
            ps._PENDING_PUSHES.clear()

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
            ps._PENDING_PUSHES.clear()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persistent_store_coalesces_latest_bytes_and_retries_transient(self) -> None:
        import persistent_store as ps

        tmp = Path(tempfile.mkdtemp())
        old_queue = ps._PUSH_QUEUE
        old_ensure = ps._ensure_push_worker
        old_secrets = ps._get_secrets
        old_get = ps._gh_get
        old_put = ps._gh_put_status
        old_sleep = ps._time.sleep
        try:
            local = tmp / "prediction_feedback_log.csv"
            local.write_text("id,value\n1,old\n", encoding="utf-8")
            ps._PENDING_PUSHES.clear()
            ps._PUSH_QUEUE = queue.Queue(maxsize=20)  # type: ignore[assignment]
            ps._ensure_push_worker = lambda: None  # type: ignore[assignment]
            self.assertTrue(ps.push_file(local))
            local.write_text("id,value\n1,new\n", encoding="utf-8")
            self.assertTrue(ps.push_file(local))
            pending = ps._take_pending_pushes()
            self.assertEqual(pending, [local])

            ps._get_secrets = lambda: {"token": "token", "owner": "owner", "repo": "repo", "branch": "main"}  # type: ignore[assignment]
            ps._gh_get = lambda secrets, path: {"sha": "sha"}  # type: ignore[assignment]
            ps._time.sleep = lambda *_args, **_kwargs: None  # type: ignore[assignment]
            calls: list[bytes] = []

            def fake_put(secrets, path, content, sha=None):
                calls.append(content)
                if len(calls) == 1:
                    return {"ok": False, "status": 503}
                return {"ok": True, "status": 200}

            ps._gh_put_status = fake_put  # type: ignore[assignment]
            self.assertTrue(ps._do_push_blocking(pending[0]))
            self.assertEqual(calls[-1].decode("utf-8").replace("\r\n", "\n"), "id,value\n1,new\n")
            self.assertEqual(len(calls), 2)
        finally:
            ps._PUSH_QUEUE = old_queue  # type: ignore[assignment]
            ps._ensure_push_worker = old_ensure  # type: ignore[assignment]
            ps._get_secrets = old_secrets  # type: ignore[assignment]
            ps._gh_get = old_get  # type: ignore[assignment]
            ps._gh_put_status = old_put  # type: ignore[assignment]
            ps._time.sleep = old_sleep  # type: ignore[assignment]
            ps._PENDING_PUSHES.clear()
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

            dsm.atomic_write_json(
                snap / "_meta.json",
                {"complete": True, "saved": 1, "checksums": {}},
            )
            dsm._invalidate_snapshot_caches()
            self.assertFalse(dsm.snapshot_exists("2026-05-08"))

            dsm.atomic_write_json(
                snap / "_meta.json",
                {"complete": True, "saved": 1, "checksums": {"MISSING.NS.csv": "0" * 64}},
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

    def test_safe_filename_containment_and_collision_hashes(self) -> None:
        from safe_paths import safe_filename, safe_join

        tmp = Path(tempfile.mkdtemp())
        try:
            unsafe_values = [
                r"..\evil",
                "../evil",
                "A:B",
                "A/B",
                "A B",
                "निफ्टी",
                "X" * 180,
            ]
            names = [safe_filename(value, ".csv") for value in unsafe_values]
            self.assertEqual(len(names), len(set(names)))
            for name in names:
                path = safe_join(tmp, name)
                path.relative_to(tmp.resolve(strict=False))
                self.assertNotIn("..", name)
                self.assertNotIn("\\", name)
                self.assertNotIn("/", name)
            self.assertEqual(safe_filename("RELIANCE.NS", ".csv"), "RELIANCE.NS.csv")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_update_data_if_old_only_refreshes_stale_or_missing(self) -> None:
        import data_downloader as dd

        tmp = Path(tempfile.mkdtemp())
        old_dir = dd.DATA_DIR
        old_update = dd.update_all_data
        try:
            dd.DATA_DIR = tmp  # type: ignore[assignment]
            fresh = tmp / "FRESH.NS.csv"
            stale = tmp / "STALE.NS.csv"
            fresh.write_text("Date,Close,Volume\n2026-05-01,1,1\n", encoding="utf-8")
            stale.write_text("Date,Close,Volume\n2026-05-01,1,1\n", encoding="utf-8")
            now = time.time()
            os.utime(fresh, (now, now))
            os.utime(stale, (now - 72 * 3600, now - 72 * 3600))
            seen: list[str] = []

            def fake_update(tickers, period="6mo"):
                seen.extend(tickers)
                return {"updated": len(tickers), "skipped": 0, "failed": 0, "failures": {}}

            dd.update_all_data = fake_update  # type: ignore[assignment]
            updated = dd.update_data_if_old(["FRESH", "STALE", "MISS"], max_age_hours=24)
            self.assertEqual(updated, 2)
            self.assertEqual(seen, ["STALE.NS", "MISS.NS"])
        finally:
            dd.DATA_DIR = old_dir  # type: ignore[assignment]
            dd.update_all_data = old_update  # type: ignore[assignment]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_all_data_accessors_return_copies(self) -> None:
        from strategy_engines import _engine_utils as eu

        old_data = dict(eu.ALL_DATA)
        try:
            eu.ALL_DATA.clear()
            eu.ALL_DATA["AAA.NS"] = _sample_ohlcv(periods=8)
            frame = eu.get_all_data_frame("AAA.NS")
            self.assertIsNotNone(frame)
            frame.iloc[0, frame.columns.get_loc("Close")] = 9999
            self.assertNotEqual(float(eu.ALL_DATA["AAA.NS"]["Close"].iloc[0]), 9999.0)
            snap = eu.get_all_data_snapshot(["AAA.NS"])
            snap["AAA.NS"].iloc[1, snap["AAA.NS"].columns.get_loc("Close")] = 8888
            self.assertNotEqual(float(eu.ALL_DATA["AAA.NS"]["Close"].iloc[1]), 8888.0)
        finally:
            eu.ALL_DATA.clear()
            eu.ALL_DATA.update(old_data)

    def test_schema_migrations_are_persisted_to_disk(self) -> None:
        import prediction_feedback_store as pfs
        import sector_prediction_tracker as tracker

        tmp = Path(tempfile.mkdtemp())
        old_pfs_log = pfs.LOG_PATH
        old_tracker_path = tracker._LOG_PATH
        try:
            pfs.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            pfs._invalidate_cache()
            pfs.LOG_PATH.write_text("symbol,mode,market_date,prediction_score,correct\nAAA,2,2026-05-08,70,True\n", encoding="utf-8")
            pfs.read_feedback_log()
            persisted = pd.read_csv(pfs.LOG_PATH, dtype=str)
            self.assertEqual(list(persisted.columns), pfs._FIELDNAMES)

            tracker._LOG_PATH = tmp / "sector_predictions.csv"  # type: ignore[assignment]
            tracker._invalidate_log_cache()
            tracker._LOG_PATH.write_text("sector,direction,confidence\nIT,Bullish,70\n", encoding="utf-8")
            tracker.read_log()
            sector_persisted = pd.read_csv(tracker._LOG_PATH, dtype=str)
            self.assertEqual(list(sector_persisted.columns), tracker._FIELDNAMES)
        finally:
            pfs.LOG_PATH = old_pfs_log  # type: ignore[assignment]
            pfs._invalidate_cache()
            tracker._LOG_PATH = old_tracker_path  # type: ignore[assignment]
            tracker._invalidate_log_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_weighted_sector_ohlc_does_not_let_tiny_outlier_dominate(self) -> None:
        import sector_prediction_engine as spe
        import sector_prediction_tracker as tracker

        idx = pd.bdate_range("2026-04-01", periods=40)

        def frame(close: float, volume: float, high_mult: float = 1.02, low_mult: float = 0.98) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "Open": [close] * len(idx),
                    "High": [close * high_mult] * len(idx),
                    "Low": [close * low_mult] * len(idx),
                    "Close": [close] * len(idx),
                    "Volume": [volume] * len(idx),
                },
                index=idx,
            )

        all_data = {
            "BIG1": frame(100, 1_000_000),
            "BIG1.NS": frame(100, 1_000_000),
            "BIG2": frame(100, 1_000_000),
            "BIG2.NS": frame(100, 1_000_000),
            "TINY": frame(100, 1, high_mult=10.0, low_mult=0.1),
            "TINY.NS": frame(100, 1, high_mult=10.0, low_mult=0.1),
        }
        agg, used = spe._aggregate_weighted_sector_ohlc(["BIG1", "BIG2", "TINY"], all_data)
        self.assertIsNotNone(agg)
        self.assertEqual(set(used), {"BIG1", "BIG2", "TINY"})
        self.assertLess(float(agg["High"].iloc[-1]), 103.0)
        self.assertGreater(float(agg["Low"].iloc[-1]), 97.0)
        self.assertTrue((agg["High"] >= agg[["Open", "Close"]].max(axis=1)).all())
        self.assertTrue((agg["Low"] <= agg[["Open", "Close"]].min(axis=1)).all())

        rebuilt = tracker._rebuild_logged_weighted_basket(["BIG1", "BIG2", "TINY"], all_data)
        self.assertIsNotNone(rebuilt)
        self.assertLess(float(rebuilt["High"].iloc[-1]), 103.0)
        self.assertGreater(float(rebuilt["Low"].iloc[-1]), 97.0)

    def test_sector_prediction_cache_metadata_detects_changed_inputs(self) -> None:
        import sector_prediction_engine as spe

        scan_a = pd.DataFrame({"Symbol": ["AAA"], "Final Score": [70.0]})
        scan_b = pd.DataFrame({"Symbol": ["AAA"], "Final Score": [80.0]})
        meta_a = spe._sector_cache_metadata("IT", scan_a, {}, ["AAA"], "RANGE_BOUND", {"ema_slope": 0.1})
        payload = {"cache_metadata": dict(meta_a)}
        self.assertTrue(spe._cache_metadata_matches(payload, meta_a))
        meta_b = spe._sector_cache_metadata("IT", scan_b, {}, ["AAA"], "RANGE_BOUND", {"ema_slope": 0.1})
        self.assertFalse(spe._cache_metadata_matches(payload, meta_b))

    def test_learning_prediction_does_not_mutate_encoders(self) -> None:
        import learning_engine as le

        class DummyScaler:
            def transform(self, values):
                return values

        class DummyModel:
            def predict_proba(self, values):
                return [[0.4, 0.6] for _ in range(len(values))]

        old_model, old_scaler = le.MODEL, le.SCALER
        old_regime, old_sector = dict(le.REGIME_ENCODER), dict(le.SECTOR_ENCODER)
        try:
            le.MODEL = DummyModel()
            le.SCALER = DummyScaler()
            le.REGIME_ENCODER = {"UNKNOWN": 0, "RANGE": 1}
            le.SECTOR_ENCODER = {"UNKNOWN": 0, "IT": 1}
            before_regime = dict(le.REGIME_ENCODER)
            before_sector = dict(le.SECTOR_ENCODER)
            out = le.predict_success({"Regime": "BRAND_NEW", "Sector": "NEW_SECTOR", "Prediction Score": 70})
            self.assertEqual(out, 60.0)
            self.assertEqual(le.REGIME_ENCODER, before_regime)
            self.assertEqual(le.SECTOR_ENCODER, before_sector)
        finally:
            le.MODEL, le.SCALER = old_model, old_scaler
            le.REGIME_ENCODER = old_regime
            le.SECTOR_ENCODER = old_sector

    def test_expanded_imported_learning_trains_with_recency_weights(self) -> None:
        import learning_engine as le
        import prediction_feedback_store as pfs

        if not le.SKLEARN_OK:
            self.skipTest("scikit-learn unavailable")

        tmp = Path(tempfile.mkdtemp())
        old_log_path = pfs.LOG_PATH
        old_model, old_scaler = le.MODEL, le.SCALER
        old_regime, old_sector = dict(le.REGIME_ENCODER), dict(le.SECTOR_ENCODER)
        old_features = {key: dict(value) for key, value in le.FEATURE_ENCODERS.items()}
        old_status = dict(le.TRAINING_STATUS)
        old_save = le._save_model
        try:
            pfs.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            pfs._invalidate_cache()
            rows = []
            for idx in range(42):
                correct = idx % 3 != 0
                bullish = idx % 2 == 0
                rows.append(
                    {
                        "logged_at": f"2026-05-{(idx % 20) + 1:02d}T15:30:00+05:30",
                        "market_date": f"2026-05-{(idx % 20) + 1:02d}",
                        "prediction_id": f"imp-{idx}",
                        "symbol": f"AAA{idx}",
                        "sector": "IT" if idx % 2 == 0 else "BANK",
                        "mode": 7 if idx % 2 == 0 else 3,
                        "import_source": "Tomorrow's Picks - Momentum",
                        "import_category": "Momentum" if idx % 2 == 0 else "Relax",
                        "strategy_strip": "Momentum" if idx % 2 == 0 else "Relax",
                        "prediction_direction": "Bullish" if bullish else "Bearish",
                        "target_policy_version": "stock_next_session_v2",
                        "prediction_score": 68 + (idx % 8),
                        "final_score": 62 + (idx % 10),
                        "signal": "BUY" if bullish else "AVOID",
                        "conviction_tier": "High" if idx % 2 == 0 else "Medium",
                        "market_bias": "Bullish",
                        "regime": "TRENDING_UP",
                        "rsi": 52 + (idx % 20),
                        "vol_avg_ratio": 1.1 + ((idx % 5) * 0.1),
                        "delta_ema20_pct": -2 + (idx % 5),
                        "trap_risk": "LOW" if idx % 4 else "MEDIUM",
                        "pred_bullish": "1" if bullish else "0",
                        "actual_next_return_pct": "1.4" if correct else "-1.1",
                        "correct": "True" if correct else "False",
                        "outcome_label": "correct" if correct else "incorrect",
                        "outcome_quality": "WIN" if correct else "LOSS",
                    }
                )
            pd.DataFrame(rows).to_csv(pfs.LOG_PATH, index=False)
            le.MODEL = None
            le.SCALER = None
            le.REGIME_ENCODER = {}
            le.SECTOR_ENCODER = {}
            le.FEATURE_ENCODERS = {}
            le.TRAINING_STATUS = dict(old_status)
            le._save_model = lambda *_args, **_kwargs: True  # type: ignore[assignment]

            result = le.train_learning_model()
            status = result.get("status", {})

            self.assertTrue(status.get("trained"))
            self.assertGreaterEqual(int(status.get("imported_ai_samples", 0)), 40)
            self.assertGreater(int(status.get("active_feature_count", 0)), 6)
            self.assertTrue(bool(status.get("recency_weighting_active")))
            prob = le.predict_success({"Prediction Score": 70, "Sector": "NEW", "Regime": "UNKNOWN"})
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 100.0)
        finally:
            pfs.LOG_PATH = old_log_path  # type: ignore[assignment]
            pfs._invalidate_cache()
            le.MODEL, le.SCALER = old_model, old_scaler
            le.REGIME_ENCODER = old_regime
            le.SECTOR_ENCODER = old_sector
            le.FEATURE_ENCODERS = old_features
            le.TRAINING_STATUS = old_status
            le._save_model = old_save
            shutil.rmtree(tmp, ignore_errors=True)

    def test_imported_ai_performance_summary_buckets(self) -> None:
        import prediction_feedback_store as pfs

        tmp = Path(tempfile.mkdtemp())
        old_log_path = pfs.LOG_PATH
        try:
            pfs.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            pfs._invalidate_cache()
            rows = [
                {
                    "logged_at": "2026-05-10T15:30:00+05:30",
                    "market_date": "2026-05-10",
                    "prediction_id": "p1",
                    "symbol": "AAA",
                    "sector": "IT",
                    "mode": 7,
                    "import_source": "Tomorrow's Picks - Momentum",
                    "import_category": "Momentum",
                    "strategy_strip": "Momentum",
                    "prediction_direction": "Bullish",
                    "target_policy_version": "stock_next_session_v2",
                    "prediction_score": 72,
                    "final_score": 70,
                    "signal": "BUY",
                    "conviction_tier": "High",
                    "market_bias": "Bullish",
                    "regime": "TRENDING_UP",
                    "rsi": 61,
                    "vol_avg_ratio": 1.4,
                    "delta_ema20_pct": 2.0,
                    "trap_risk": "LOW",
                    "pred_bullish": "1",
                    "actual_next_return_pct": 2.5,
                    "correct": "True",
                    "outcome_label": "correct",
                    "outcome_quality": "BIG_WIN",
                },
                {
                    "logged_at": "2026-05-11T15:30:00+05:30",
                    "market_date": "2026-05-11",
                    "prediction_id": "p2",
                    "symbol": "BBB",
                    "sector": "AUTO",
                    "mode": 3,
                    "import_source": "Tomorrow's Picks - Relax",
                    "import_category": "Relax",
                    "strategy_strip": "Relax",
                    "prediction_direction": "Bullish",
                    "target_policy_version": "stock_next_session_v2",
                    "prediction_score": 66,
                    "final_score": 64,
                    "signal": "BUY",
                    "conviction_tier": "Medium",
                    "market_bias": "Bullish",
                    "regime": "RANGE_BOUND",
                    "rsi": 54,
                    "vol_avg_ratio": 1.1,
                    "delta_ema20_pct": 0.5,
                    "trap_risk": "HIGH",
                    "pred_bullish": "1",
                    "actual_next_return_pct": -2.2,
                    "correct": "False",
                    "outcome_label": "incorrect",
                    "outcome_quality": "BIG_LOSS",
                },
            ]
            pd.DataFrame(rows).to_csv(pfs.LOG_PATH, index=False)

            summary = pfs.summarize_imported_ai_performance()

            self.assertEqual(summary["total_logged"], 2)
            self.assertEqual(summary["validated"], 2)
            self.assertEqual(summary["accuracy_pct"], 50.0)
            self.assertEqual(summary["best_category"]["bucket"], "Momentum")
            self.assertEqual(summary["worst_category"]["bucket"], "Relax")
            self.assertFalse(summary["recent"].empty)
        finally:
            pfs.LOG_PATH = old_log_path  # type: ignore[assignment]
            pfs._invalidate_cache()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_tomorrow_prediction_accepts_expanded_learning_features(self) -> None:
        import prediction_feedback_store as pfs
        import tomorrow_prediction_engine as tpe

        tmp = Path(tempfile.mkdtemp())
        old_log_path = pfs.LOG_PATH
        old_cache = dict(tpe._IMPORTED_PERFORMANCE_CACHE)
        try:
            pfs.LOG_PATH = tmp / "prediction_feedback_log.csv"  # type: ignore[assignment]
            pfs._invalidate_cache()
            tpe._IMPORTED_PERFORMANCE_CACHE.clear()
            tpe._IMPORTED_PERFORMANCE_CACHE.update({"summary": None})

            hist = _sample_ohlcv(periods=60)
            result = tpe.get_tomorrow_prediction("TEST", {"TEST": hist, "TEST.NS": hist}, 7)

            self.assertEqual(result["ticker"], "TEST")
            self.assertIn(result["direction"], {"Bullish", "Bearish", "Sideways"})
            self.assertGreaterEqual(float(result["learned_probability"]), 0.0)
            self.assertLessEqual(float(result["learned_probability"]), 100.0)
        finally:
            pfs.LOG_PATH = old_log_path  # type: ignore[assignment]
            pfs._invalidate_cache()
            tpe._IMPORTED_PERFORMANCE_CACHE.clear()
            tpe._IMPORTED_PERFORMANCE_CACHE.update(old_cache)
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

    def test_ail_aura_verdict_adds_encoded_chart_links(self) -> None:
        import app_ail_in_one_section as ail_ui

        source = pd.DataFrame([{"Symbol": "GMRP&UI", "Aura Score": 89.0}])
        linked = ail_ui._with_chart_links(source)

        self.assertIn("Chart", linked.columns)
        self.assertEqual(
            linked.loc[0, "Chart"],
            "https://www.tradingview.com/chart/?symbol=NSE:GMRP%26UI",
        )

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
        selfEqual(symbols("Momentum"), {"M1MOM", "M2BAL", "M7MOM"})
        selfEqual(symbols("Swing"), {"M6SWING"})
        selfEqual(symbols("Institutional"), {"M4INST"})
        self.assertNotIn("M2BAL", symbols("Swing"))

    def test_ail_internal_pool_widens_without_changing_displayed_top3(self) -> None:
        from ail_in_one_engine import _candidate_pool_from_top3, extract_top_candidates

        rows = []
        for idx in range(9):
            rows.append(
                {
                    "Symbol": f"RANK{idx}",
                    "Mode ID": 1,
                    "Mode Name": "Momentum",
                    "Final Score": 92.0 - idx * 3.0,
                    "Prediction Score": 90.0 - idx * 3.0,
                    "Backtest %": 78.0 - idx,
                    "ML %": 76.0 - idx,
                    "Confidence": 74.0 - idx,
                    "RSI": 58.0,
                    "Vol / Avg": 1.8,
                    "Trap Risk": "LOW",
                    "Signal": "BUY",
                }
            )
        rows.append(
            {
                "Symbol": "RESCUE",
                "Mode ID": 1,
                "Mode Name": "Momentum",
                "Final Score": 45.0,
                "Prediction Score": 44.0,
                "Backtest %": 45.0,
                "ML %": 45.0,
                "Confidence": 44.0,
                "AIL Opportunity Score": 48.0,
                "RSI": 54.0,
                "Vol / Avg": 1.4,
                "Trap Risk": "LOW",
                "Signal": "WATCH",
            }
        )

        result = extract_top_candidates({"Momentum": pd.DataFrame(rows)}, market_bias={"bias": "Bullish"}, top_n=3)
        top_symbols = result["Momentum"]["top_df"]["Symbol"].tolist()
        pool = _candidate_pool_from_top3(result)

        self.assertEqual(len(top_symbols), 3)
        self.assertNotIn("RESCUE", top_symbols)
        self.assertGreater(len(pool), len(top_symbols))
        self.assertIn("RESCUE", set(pool["Symbol"].tolist()))
        self.assertIn("AIL Internal Pool Reason", pool.columns)

    def test_ail_top3_consensus_compares_normal_screener_with_tomorrow_accuracy(self) -> None:
        from ail_in_one_engine import extract_top_candidates

        frame = pd.DataFrame(
            [
                {
                    "Symbol": "FAST",
                    "Mode ID": 1,
                    "Mode Name": "Momentum",
                    "Final Score": 94.0,
                    "Prediction Score": 90.0,
                    "Backtest %": 80.0,
                    "ML %": 82.0,
                    "Confidence": 78.0,
                    "RSI": 74.0,
                    "Vol / Avg": 2.4,
                    "5D Return (%)": 2.2,
                    "Trap Risk": "LOW",
                    "Action": "Buy Tomorrow",
                    "Signal": "Strong Buy",
                    "Grade": "A",
                    "Conviction Tier": "HIGH",
                    "Setup Quality": "HIGH",
                    "Entry Timing": "GOOD",
                    "Delta vs EMA20 (%)": 2.5,
                    "Delta vs 20D High (%)": -0.5,
                    "Sector Strength": 72.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
                {
                    "Symbol": "CLEAN",
                    "Mode ID": 1,
                    "Mode Name": "Momentum",
                    "Final Score": 82.0,
                    "Prediction Score": 78.0,
                    "Backtest %": 70.0,
                    "ML %": 72.0,
                    "Confidence": 70.0,
                    "RSI": 59.0,
                    "Vol / Avg": 1.9,
                    "5D Return (%)": 1.8,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Signal": "Possible Up",
                    "Grade": "A",
                    "Conviction Tier": "HIGH",
                    "Setup Quality": "HIGH",
                    "Entry Timing": "GOOD",
                    "Delta vs EMA20 (%)": 2.0,
                    "Delta vs 20D High (%)": -1.0,
                    "Sector Strength": 68.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
                {
                    "Symbol": "STEADY",
                    "Mode ID": 1,
                    "Mode Name": "Momentum",
                    "Final Score": 74.0,
                    "Prediction Score": 70.0,
                    "Backtest %": 64.0,
                    "ML %": 66.0,
                    "Confidence": 65.0,
                    "RSI": 62.0,
                    "Vol / Avg": 1.7,
                    "5D Return (%)": 2.5,
                    "Trap Risk": "LOW",
                    "Action": "Watch",
                    "Signal": "Green",
                    "Grade": "B",
                    "Conviction Tier": "MEDIUM",
                    "Setup Quality": "MEDIUM",
                    "Entry Timing": "NEUTRAL",
                    "Delta vs EMA20 (%)": 2.4,
                    "Delta vs 20D High (%)": -2.0,
                    "Sector Strength": 60.0,
                    "Regime": "TRENDING_UP",
                    "Closing Strength": "STRONG",
                },
                {
                    "Symbol": "LATE",
                    "Mode ID": 1,
                    "Mode Name": "Momentum",
                    "Final Score": 70.0,
                    "Prediction Score": 65.0,
                    "Backtest %": 55.0,
                    "ML %": 57.0,
                    "Confidence": 58.0,
                    "RSI": 69.0,
                    "Vol / Avg": 1.1,
                    "5D Return (%)": 8.0,
                    "Trap Risk": "MEDIUM",
                    "Action": "Watch",
                    "Signal": "Watch",
                    "Grade": "C",
                    "Conviction Tier": "LOW",
                    "Setup Quality": "LOW",
                    "Entry Timing": "LATE",
                    "Delta vs EMA20 (%)": 5.0,
                    "Delta vs 20D High (%)": 1.2,
                    "Sector Strength": 52.0,
                    "Regime": "RANGE_BOUND",
                    "Closing Strength": "NEUTRAL",
                },
            ]
        )

        result = extract_top_candidates({"Momentum": frame}, market_bias={"bias": "Bullish", "regime": "TRENDING_UP"}, top_n=3)
        top_df = result["Momentum"]["top_df"]

        self.assertIn("AIL Top3 Normal Score", top_df.columns)
        self.assertIn("AIL Top3 Tomorrow Score", top_df.columns)
        self.assertIn("AIL Top3 Source", top_df.columns)
        self.assertIn("FAST", set(top_df["Symbol"].tolist()))

        fast = top_df.loc[top_df["Symbol"].eq("FAST")].iloc[0]
        self.assertEqual(fast["AIL Top3 Source"], "Normal mode screener")
        self.assertTrue(bool(fast["AIL Top3 Prompt Eliminated"]))
        self.assertGreater(float(fast["AIL Top3 Normal Score"]), 90.0)
        self.assertGreater(float(fast["AIL Top3 Score"]), 70.0)
        self.assertFalse(top_df["AIL Top3 Confidence"].astype(str).str.contains("Fallback", case=False).any())

    def test_ail_confidence_uses_real_components_without_fallback_label(self) -> None:
        from ail_confidence_engine import apply_evidence_coverage_damping, compute_smart_confidence

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

        damped = apply_evidence_coverage_damping(
            pd.DataFrame([{"Symbol": "THIN", "AIL Confidence": 80.0, "AIL Confidence Coverage": 20.0}])
        )
        self.assertLess(float(damped.loc[0, "AIL Confidence"]), 80.0)
        self.assertIn("Thin evidence", str(damped.loc[0, "AIL Evidence Label"]))

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

    def test_ail_market_state_temporal_adjustments_respect_closed_sessions(self) -> None:
        from ail_market_state_engine import apply_market_state_adjustments, detect_market_state

        closing = detect_market_state({"window": "LIVE"}, now=datetime(2026, 5, 15, 15, 30))
        post_close = detect_market_state({"window": "CLOSED", "use_snapshot": True}, now=datetime(2026, 5, 15, 17, 5))
        pre_market = detect_market_state({"window": "PRE_MARKET", "use_snapshot": True}, now=datetime(2026, 5, 15, 8, 45))
        weekend = detect_market_state({"window": "WEEKEND", "use_snapshot": True}, now=datetime(2026, 5, 16, 12, 0))

        self.assertEqual(closing["state"], "CLOSING")
        self.assertEqual(post_close["state"], "POST_CLOSE")
        self.assertEqual(pre_market["state"], "PRE_MARKET")
        self.assertEqual(weekend["state"], "WEEKEND")

        row = pd.DataFrame(
            [
                {
                    "Symbol": "MOMO",
                    "AIL Categories": "Momentum, Intraday",
                    "Momentum Quality": 82.0,
                    "Volume Quality": 72.0,
                    "Setup Cleanliness": 48.0,
                    "Structure Quality": 44.0,
                    "Trap Risk Score": 58.0,
                    "RSI": 74.0,
                }
            ]
        )
        live_fit = apply_market_state_adjustments(row, {"state": "LIVE"}).loc[0, "AIL Temporal Fit"]
        closed_fit = apply_market_state_adjustments(row, {"state": "POST_CLOSE", "use_snapshot": True}).loc[0, "AIL Temporal Fit"]
        self.assertLess(closed_fit, live_fit)
        self.assertIn("closed-market momentum reduced", apply_market_state_adjustments(row, {"state": "POST_CLOSE"}).loc[0, "AIL Temporal Notes"])

    def test_ail_conflict_calibration_regime_and_health_layers_are_bounded(self) -> None:
        from ail_calibration_engine import apply_confidence_calibration, build_confidence_buckets, calibrate_confidence_score
        from ail_conflict_engine import apply_conflict_penalties, detect_signal_conflicts
        from ail_health_engine import compute_orchestration_health
        from ail_regime_orchestrator import apply_regime_preference, compute_regime_strategy_bias

        row = {
            "Symbol": "RISKY",
            "AIL Categories": "Momentum, Breakout",
            "Momentum Quality": 86.0,
            "Volume Quality": 42.0,
            "Setup Cleanliness": 44.0,
            "Structure Quality": 40.0,
            "Trap Risk Score": 72.0,
            "Bullish Probability": 75.0,
            "RSI": 76.0,
        }
        conflicts = detect_signal_conflicts(row)
        self.assertGreater(conflicts["conflict_score"], 25.0)
        conflict_df = apply_conflict_penalties(pd.DataFrame([row]))
        self.assertEqual(len(conflict_df), 1)
        self.assertIn("AIL Conflict Score", conflict_df.columns)
        self.assertGreater(conflict_df.loc[0, "AIL Conflict Penalty"], 0)

        clean_conflict_df = apply_conflict_penalties(pd.DataFrame([{"Symbol": "CLEAN", "AIL Categories": "Swing", "Trap Risk Score": 30.0}]))
        self.assertEqual(float(clean_conflict_df.loc[0, "AIL Conflict Penalty"]), 0.0)
        self.assertEqual(float(clean_conflict_df.loc[0, "AIL Conflict Multiplier"]), 1.0)

        feedback = pd.DataFrame(
            [
                {
                    "import_source": "A-I-L IN ONE",
                    "prediction_score": "82",
                    "actual_next_return_pct": "-1.2",
                    "prediction_direction": "Bullish",
                    "pred_bullish": "True",
                    "correct": "False",
                }
                for _ in range(6)
            ]
        )
        buckets = build_confidence_buckets(feedback, min_rows=5)
        calibrated, adjustment, note = calibrate_confidence_score(82.0, buckets)
        self.assertLess(calibrated, 82.0)
        self.assertLess(adjustment, 0)
        self.assertIn("gap", note.lower())

        calibration = {"buckets": buckets}
        calibrated_df = apply_confidence_calibration(pd.DataFrame([{"Symbol": "CAL", "AIL Confidence": 82.0}]), calibration)
        recalibrated_df = apply_confidence_calibration(calibrated_df.assign(**{"AIL Confidence": calibrated_df["AIL Calibrated Confidence"]}), calibration)
        self.assertEqual(float(calibrated_df.loc[0, "AIL Calibrated Confidence"]), float(recalibrated_df.loc[0, "AIL Calibrated Confidence"]))
        self.assertEqual(float(recalibrated_df.loc[0, "AIL Calibration Base Confidence"]), 82.0)

        bias = compute_regime_strategy_bias({"bias": "Weak", "regime": "HIGH_VOLATILITY"}, {"state": "POST_CLOSE"})
        regime_df = apply_regime_preference(pd.DataFrame([row]), bias)
        self.assertIn("AIL Regime Strategy Fit", regime_df.columns)
        self.assertGreaterEqual(regime_df.loc[0, "AIL Regime Multiplier"], 0.94)
        self.assertLessEqual(regime_df.loc[0, "AIL Regime Multiplier"], 1.06)

        result = SimpleNamespace(final_ranked_df=conflict_df, health={}, elapsed_sec=12.5)
        health = compute_orchestration_health(result, {"state": "LIVE", "use_snapshot": True}, {"buckets": buckets, "drift": {"status": "drift", "max_gap": 20}})
        self.assertIn("market state", health["AIL Health Flags"])
        self.assertEqual(health["market_state_health"]["status"], "stale")

    def test_ail_penalty_guards_preserve_asymmetric_opportunity(self) -> None:
        from ail_confidence_health import analyze_confidence_distribution, preserve_high_conviction, preserve_speculative_conviction
        from ail_opportunity_engine import preserve_high_upside_candidates
        from ail_penalty_guard import cap_total_penalty, detect_over_suppression, prevent_confidence_collapse
        from ail_philosophy_guard import detect_philosophy_flattening, preserve_mode_identity

        df = pd.DataFrame(
            [
                {
                    "Symbol": "IGNITE",
                    "AIL Categories": "Momentum, Breakout",
                    "Mode ID": 7,
                    "Smart Potential Score": 84.0,
                    "AIL Master Score": 61.0,
                    "AIL Confidence": 49.0,
                    "AIL Calibrated Confidence": 48.0,
                    "Momentum Quality": 88.0,
                    "Volume Quality": 82.0,
                    "Setup Cleanliness": 64.0,
                    "Structure Quality": 72.0,
                    "Trap Risk Score": 58.0,
                    "Risk Reward Score": 78.0,
                    "RSI": 68.0,
                },
                {
                    "Symbol": "EARLY",
                    "AIL Categories": "Relaxed",
                    "Mode ID": 3,
                    "Smart Potential Score": 76.0,
                    "AIL Master Score": 60.0,
                    "AIL Confidence": 51.0,
                    "AIL Calibrated Confidence": 50.0,
                    "Momentum Quality": 55.0,
                    "Volume Quality": 60.0,
                    "Setup Cleanliness": 70.0,
                    "Structure Quality": 66.0,
                    "Trap Risk Score": 38.0,
                    "Risk Reward Score": 72.0,
                    "RSI": 54.0,
                    "Entry Timing": "Early accumulation",
                },
                {
                    "Symbol": "SAFE",
                    "AIL Categories": "Swing",
                    "Mode ID": 6,
                    "Smart Potential Score": 66.0,
                    "AIL Master Score": 64.0,
                    "AIL Confidence": 55.0,
                    "AIL Calibrated Confidence": 55.0,
                    "Momentum Quality": 58.0,
                    "Volume Quality": 58.0,
                    "Setup Cleanliness": 68.0,
                    "Structure Quality": 62.0,
                    "Trap Risk Score": 35.0,
                    "Risk Reward Score": 66.0,
                    "RSI": 57.0,
                },
                {
                    "Symbol": "INST",
                    "AIL Categories": "Institutional",
                    "Mode ID": 4,
                    "Smart Potential Score": 70.0,
                    "AIL Master Score": 63.0,
                    "AIL Confidence": 56.0,
                    "AIL Calibrated Confidence": 56.0,
                    "Momentum Quality": 62.0,
                    "Volume Quality": 61.0,
                    "Setup Cleanliness": 69.0,
                    "Structure Quality": 68.0,
                    "Trap Risk Score": 42.0,
                    "Risk Reward Score": 67.0,
                    "Regime Alignment": 74.0,
                    "RSI": 59.0,
                },
            ]
        )

        guarded = preserve_high_upside_candidates(df)
        guarded = preserve_mode_identity(guarded)
        guarded = preserve_high_conviction(guarded)
        guarded = preserve_speculative_conviction(guarded)
        guarded = prevent_confidence_collapse(guarded)
        guarded = cap_total_penalty(guarded)

        ignite = guarded.loc[guarded["Symbol"].eq("IGNITE")].iloc[0]
        early = guarded.loc[guarded["Symbol"].eq("EARLY")].iloc[0]
        self.assertGreaterEqual(ignite["AIL Master Score"], 76.0)
        self.assertGreater(ignite["AIL Opportunity Score"], 40.0)
        self.assertGreater(early["AIL Philosophy Score"], 60.0)
        self.assertGreaterEqual(ignite["AIL Calibrated Confidence"], 58.0)

        confidence_health = analyze_confidence_distribution(guarded)
        philosophy_health = detect_philosophy_flattening(guarded)
        suppression_health = detect_over_suppression(guarded)
        self.assertIn(confidence_health["status"], {"healthy", "compressed"})
        self.assertEqual(philosophy_health["status"], "healthy")
        self.assertEqual(suppression_health["status"], "healthy")

    def test_ail_penalty_guard_does_not_double_count_master_boosts(self) -> None:
        from ail_penalty_guard import cap_total_penalty

        base = {
            "Symbol": "BOOST",
            "Smart Potential Score": 70.0,
            "AIL Master Score": 70.0,
            "AIL Opportunity Boost": 4.0,
            "AIL Philosophy Boost": 3.0,
            "AIL Confidence Health Boost": 2.0,
        }
        unmarked = cap_total_penalty(pd.DataFrame([base])).loc[0]
        marked = cap_total_penalty(
            pd.DataFrame([{**base, "AIL Boosts Applied In Master": True, "AIL Boost Ledger": "master opportunity 4.0"}])
        ).loc[0]

        self.assertEqual(float(unmarked["AIL Master Score"]), 79.0)
        self.assertEqual(float(marked["AIL Master Score"]), 70.0)
        self.assertIn("already applied", str(marked["AIL Penalty Guard Notes"]))

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
        self.assertIn("AIL Market State", result.final_ranked_df.columns)
        self.assertIn("AIL Temporal Fit", result.final_ranked_df.columns)
        self.assertIn("AIL Agreement Score", result.final_ranked_df.columns)
        self.assertIn("AIL Conflict Score", result.final_ranked_df.columns)
        self.assertIn("AIL Calibrated Confidence", result.final_ranked_df.columns)
        self.assertIn("AIL Regime Strategy Fit", result.final_ranked_df.columns)
        self.assertIn("AIL Orchestration Reasoning", result.final_ranked_df.columns)
        self.assertIn("AIL Opportunity Score", result.final_ranked_df.columns)
        self.assertIn("AIL Philosophy Score", result.final_ranked_df.columns)
        self.assertIn("AIL Suppression Index", result.final_ranked_df.columns)
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
        self.assertIn("AIL Health Flags", result.health)

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
