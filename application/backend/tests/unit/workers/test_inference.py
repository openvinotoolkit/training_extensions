# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import queue
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
from loguru import logger

from app.models.inference import InferenceWorkerStatus, InferenceWorkerStatusCode
from app.stream.stream_data import StreamData
from app.workers.inference import InferenceWorker, PredictionReorderBuffer
from app.workers.shm_status import STATUS_SHM_SIZE, read_status


@pytest.fixture(autouse=True)
def fxt_loguru_caplog(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield
    logger.remove(handler_id)


@pytest.fixture
def fxt_create_stream_data():
    def create_sample(ts):
        return StreamData(
            frame_data=np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8),
            timestamp=ts,
            source_metadata={},
        )

    return create_sample


class TestPredictionReorderBuffer:
    def test_register_timestamp_and_add_prediction(self, fxt_create_stream_data):
        buffer = PredictionReorderBuffer(max_size=5)
        ts1, ts2, ts3, ts4, ts5 = 1.0, 2.0, 3.0, 4.0, 5.0
        sd1 = fxt_create_stream_data(ts1)
        sd2 = fxt_create_stream_data(ts2)
        sd3 = fxt_create_stream_data(ts3)
        sd5 = fxt_create_stream_data(ts5)

        # Register timestamps in order 1-5, add stream data out of order and with gaps
        for ts in [ts1, ts2, ts3, ts4, ts5]:
            buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts2, sd2)
        buffer.add_prediction_for_timestamp(ts1, sd1)
        buffer.add_prediction_for_timestamp(ts3, sd3)
        buffer.add_prediction_for_timestamp(ts5, sd5)

        # Buffer should output predictions in registration order and without gaps
        ready = buffer.get_ready_predictions()
        assert ready == [sd1, sd2, sd3]

    def test_full_register(self, fxt_create_stream_data):
        # Buffer should only hold 3 expected timestamps
        buffer = PredictionReorderBuffer(max_size=3)

        # Prepare timstamps and stream data
        ts1, ts2, ts3, ts4, ts5 = 1.0, 2.0, 3.0, 4.0, 5.0
        sd1 = fxt_create_stream_data(ts1)
        sd2 = fxt_create_stream_data(ts2)
        sd3 = fxt_create_stream_data(ts3)
        sd4 = fxt_create_stream_data(ts4)
        sd5 = fxt_create_stream_data(ts5)

        # Register first three and add stream data for ts1 and ts3
        for ts in [ts1, ts2, ts3]:
            buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts1, sd1)
        buffer.add_prediction_for_timestamp(ts3, sd3)

        # Register last two, first two should be dropped, including stream data
        buffer.register_expected_timestamp(ts4)
        buffer.register_expected_timestamp(ts5)

        # Add stream data for ts2, ts4, ts5. Stream data for ts2 should be dropped
        buffer.add_prediction_for_timestamp(ts2, sd2)
        buffer.add_prediction_for_timestamp(ts5, sd5)
        buffer.add_prediction_for_timestamp(ts4, sd4)

        assert buffer.get_ready_predictions() == [sd3, sd4, sd5]

    def test_add_unexpected_prediction_warns(self, fxt_create_stream_data, fxt_loguru_caplog, caplog):
        buffer = PredictionReorderBuffer(max_size=3)
        ts = 42.0
        sd = fxt_create_stream_data(ts)
        with caplog.at_level("WARNING"):
            buffer.add_prediction_for_timestamp(ts, sd)
        assert "unexpected timestamp" in caplog.text
        assert buffer.get_ready_predictions() == []

    def test_clear(self, fxt_create_stream_data):
        buffer = PredictionReorderBuffer(max_size=2)
        ts = 1.0
        buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts, fxt_create_stream_data(ts))
        buffer.clear()
        assert buffer.get_ready_predictions() == []


class TestEnqueuePrediction:
    """The inference completion callback must never block, otherwise a model unload
    (which waits for all in-flight requests) would deadlock the inference process."""

    def _make_worker(self, pred_queue):
        # Bypass the heavy __init__: _enqueue_prediction only uses self._pred_queue.
        worker = object.__new__(InferenceWorker)
        worker._pred_queue = pred_queue
        return worker

    def test_enqueue_when_not_full(self, fxt_create_stream_data):
        import queue as queue_mod

        q = queue_mod.Queue(maxsize=2)
        worker = self._make_worker(q)
        sd = fxt_create_stream_data(1.0)

        worker._enqueue_prediction(sd)

        assert q.qsize() == 1
        assert q.get_nowait() is sd

    def test_enqueue_drops_oldest_when_full(self, fxt_create_stream_data):
        import queue as queue_mod

        q = queue_mod.Queue(maxsize=2)
        worker = self._make_worker(q)
        sd_old = fxt_create_stream_data(1.0)
        sd_mid = fxt_create_stream_data(2.0)
        sd_new = fxt_create_stream_data(3.0)

        worker._enqueue_prediction(sd_old)
        worker._enqueue_prediction(sd_mid)
        # Queue is now full; this must not block and must evict the oldest item.
        worker._enqueue_prediction(sd_new)

        assert q.qsize() == 2
        first = q.get_nowait()
        second = q.get_nowait()
        # Oldest (sd_old) was evicted; the two newest remain in order.
        assert first is sd_mid
        assert second is sd_new

    def test_enqueue_does_not_block_when_full(self, fxt_create_stream_data):
        """A full queue whose consumer never drains must not stall the callback."""
        import queue as queue_mod
        import threading

        q = queue_mod.Queue(maxsize=1)
        worker = self._make_worker(q)
        q.put_nowait(fxt_create_stream_data(0.0))

        done = threading.Event()

        def _run():
            worker._enqueue_prediction(fxt_create_stream_data(1.0))
            done.set()

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=2)
        assert done.is_set(), "_enqueue_prediction blocked on a full queue"


@pytest.fixture
def fxt_status_shm():
    """Real backing shared-memory block standing in for the scheduler-owned inference status shm."""
    shm = SharedMemory(create=True, size=STATUS_SHM_SIZE)
    try:
        yield shm
    finally:
        shm.close()
        shm.unlink()


@pytest.fixture
def fxt_status_shm_lock():
    return mp.get_context("spawn").Lock()


class TestInferenceStatusReporting:
    """Tests that the worker reports OK and ERROR inference statuses to shared memory."""

    def test_report_status_ok(self, fxt_status_shm, fxt_status_shm_lock):
        """_report_status writes an OK status that can be read back from shared memory."""
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = fxt_status_shm
        worker._inference_status_shm_lock = fxt_status_shm_lock
        model_id = uuid4()

        worker._report_status(code=InferenceWorkerStatusCode.OK, model_id=model_id)

        status = read_status(InferenceWorkerStatus, fxt_status_shm, fxt_status_shm_lock)
        assert status is not None
        assert status.code == InferenceWorkerStatusCode.OK
        assert status.model_id == model_id

    def test_on_inference_completed_uses_project_label_colors(
        self, fxt_status_shm, fxt_status_shm_lock, fxt_create_stream_data
    ):
        """Predictions are rendered with the colors of the labels of the active project."""
        label_colors = {"cat": "#00FF00", "dog": "#FF0000"}
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = fxt_status_shm
        worker._inference_status_shm_lock = fxt_status_shm_lock
        worker._metrics_service = Mock()
        worker._model_service = Mock(label_colors=label_colors)
        worker._pred_queue = queue.Queue(maxsize=4)  # pyrefly: ignore[bad-assignment]
        setattr(worker, "_InferenceWorker__prediction_buffer", PredictionReorderBuffer())

        ts = 1.0
        worker._prediction_buffer.register_expected_timestamp(ts)
        stream_data = fxt_create_stream_data(ts)
        userdata = {"inference_start_time": ts, "model_id": uuid4(), "stream_data": stream_data}

        with patch(
            "app.workers.inference.Visualizer.overlay_predictions", return_value=stream_data.frame_data
        ) as mock_overlay:
            worker._on_inference_completed(Mock(), userdata)

        assert mock_overlay.call_args.kwargs["label_colors"] == label_colors

    def test_report_status_error(self, fxt_status_shm, fxt_status_shm_lock):
        """_report_status writes an ERROR status (with message) that can be read back."""
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = fxt_status_shm
        worker._inference_status_shm_lock = fxt_status_shm_lock
        model_id = uuid4()

        worker._report_status(
            code=InferenceWorkerStatusCode.ERROR, model_id=model_id, message="Unhandled error in inference loop"
        )

        status = read_status(InferenceWorkerStatus, fxt_status_shm, fxt_status_shm_lock)
        assert status is not None
        assert status.code == InferenceWorkerStatusCode.ERROR
        assert status.model_id == model_id
        assert status.message == "Unhandled error in inference loop"

    def test_report_status_noop_without_shm(self, fxt_status_shm, fxt_status_shm_lock):
        """_report_status is a no-op (does not raise, writes nothing) when no shm is attached."""
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = None
        worker._inference_status_shm_lock = fxt_status_shm_lock

        worker._report_status(code=InferenceWorkerStatusCode.OK, model_id=uuid4())

        # Nothing was written to the (separate, still-empty) block.
        assert read_status(InferenceWorkerStatus, fxt_status_shm, fxt_status_shm_lock) is None

    def test_on_inference_completed_reports_ok(self, fxt_status_shm, fxt_status_shm_lock, fxt_create_stream_data):
        """A completed inference reports an OK status for the model that produced it."""
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = fxt_status_shm
        worker._inference_status_shm_lock = fxt_status_shm_lock
        worker._metrics_service = Mock()
        # A plain queue.Queue stands in for the multiprocessing queue (only *_nowait APIs are used here).
        worker._pred_queue = queue.Queue(maxsize=4)  # pyrefly: ignore[bad-assignment]
        # Set the name-mangled private buffer via setattr to avoid static "missing-attribute" analysis.
        setattr(worker, "_InferenceWorker__prediction_buffer", PredictionReorderBuffer())

        ts = 1.0
        model_id = uuid4()
        worker._prediction_buffer.register_expected_timestamp(ts)
        stream_data = fxt_create_stream_data(ts)
        userdata = {"inference_start_time": ts, "model_id": model_id, "stream_data": stream_data}

        with patch("app.workers.inference.Visualizer.overlay_predictions", return_value=stream_data.frame_data):
            worker._on_inference_completed(Mock(), userdata)

        status = read_status(InferenceWorkerStatus, fxt_status_shm, fxt_status_shm_lock)
        assert status is not None
        assert status.code == InferenceWorkerStatusCode.OK
        assert status.model_id == model_id

    def test_run_loop_reports_error_on_failure(self, fxt_status_shm, fxt_status_shm_lock):
        """An unhandled error in the inference loop reports an ERROR status for the loaded model."""
        worker = object.__new__(InferenceWorker)
        worker._inference_status_shm = fxt_status_shm
        worker._inference_status_shm_lock = fxt_status_shm_lock
        model_id = uuid4()
        worker._loaded_model = Mock(model_id=model_id)

        def _raise() -> None:
            raise RuntimeError("boom")

        worker._refresh_loaded_model = _raise  # type: ignore[method-assign]
        worker.stop_aware_sleep = lambda *_: None  # type: ignore[method-assign]

        iterations = {"n": 0}

        def _should_stop() -> bool:
            iterations["n"] += 1
            return iterations["n"] > 1  # run exactly one iteration, then stop

        worker.should_stop = _should_stop  # type: ignore[method-assign]

        worker.run_loop()

        status = read_status(InferenceWorkerStatus, fxt_status_shm, fxt_status_shm_lock)
        assert status is not None
        assert status.code == InferenceWorkerStatusCode.ERROR
        assert status.model_id == model_id
        assert status.message == "Unhandled error in inference loop"
