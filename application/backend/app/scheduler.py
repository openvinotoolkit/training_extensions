# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import threading
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import psutil
from loguru import logger

from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.inference import InferenceServer
from app.services.metrics_service import SIZE
from app.webrtc.broadcaster import FrameBroadcaster
from app.workers import (
    DispatchingWorker,
    InferenceServerMonitorThread,
    InferenceWorker,
    InferenceWorkerConfig,
    StreamLoader,
)
from app.workers.shm_status import STATUS_SHM_SIZE
from app.workers.sink_status_holder import SinkStatusHolder


class Scheduler:
    """Manages application processes and threads"""

    FRAME_QUEUE_SIZE = 5
    PREDICTION_QUEUE_SIZE = 5

    def __init__(
        self,
        event_bus: EventBus,
        data_collector: DataCollector,
        inference_server: InferenceServer,
        mp_ctx: BaseContext,
    ) -> None:
        logger.info("Initializing Scheduler...")
        self._event_bus = event_bus
        self._data_collector = data_collector
        self._inference_server = inference_server
        self._mp_ctx = mp_ctx

        # Queue for the frames acquired from the stream source and decoded
        self.frame_queue = self._mp_ctx.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        # Queue for the inference results (predictions)
        self.pred_queue = self._mp_ctx.Queue(maxsize=self.PREDICTION_QUEUE_SIZE)
        # Shared memory for source status (IPC: StreamLoader -> Scheduler/API)
        self.source_status_shm = SharedMemory(create=True, size=STATUS_SHM_SIZE)
        self.source_status_lock = mp_ctx.Lock()
        # Shared memory for inference worker status (IPC: InferenceWorker -> Scheduler/API)
        self.inference_status_shm = SharedMemory(create=True, size=STATUS_SHM_SIZE)
        self.inference_status_lock = mp_ctx.Lock()
        # Broadcaster for pushing predictions to the visualization stream (WebRTC)
        self.rtc_stream_broadcaster: FrameBroadcaster[np.ndarray] = FrameBroadcaster[np.ndarray]()
        # Event to sync all processes on application shutdown
        self.mp_stop_event = self._mp_ctx.Event()

        # Shared memory for metrics collector
        self.shm_metrics = SharedMemory(create=True, size=SIZE)
        self.shm_metrics_lock = self._mp_ctx.Lock()

        # Thread-safe holder for sink status
        self.sink_status_holder = SinkStatusHolder()

        self.processes: list[BaseProcess] = []
        self.threads: list[threading.Thread] = []
        logger.info("Scheduler initialized")

    def start_workers(self) -> None:
        """Start all worker processes and threads"""
        logger.info("Starting worker processes...")

        # Create and start processes
        stream_loader_proc = StreamLoader(
            frame_queue=self.frame_queue,
            status_shm_name=self.source_status_shm.name,
            status_shm_lock=self.source_status_lock,
            stop_event=self.mp_stop_event,
            source_changed_condition=self._event_bus.source_changed_condition,
            logger_=logger,  # type: ignore
        )

        inference_worker_config = InferenceWorkerConfig(
            frame_queue=self.frame_queue,
            pred_queue=self.pred_queue,
            stop_event=self.mp_stop_event,
            model_reload_event=self._event_bus.model_reload_event,
            inference_params_event=self._event_bus.inference_params_event,
            shm_name=self.shm_metrics.name,
            shm_lock=self.shm_metrics_lock,
            inference_status_shm_name=self.inference_status_shm.name,
            inference_status_shm_lock=self.inference_status_lock,
            logger_=logger,  # type: ignore
        )
        inference_server_proc = InferenceWorker(inference_worker_config)

        inference_server_monitor_thread = InferenceServerMonitorThread(
            server=self._inference_server, stop_event=self.mp_stop_event
        )

        dispatching_thread = DispatchingWorker(
            event_bus=self._event_bus,
            pred_queue=self.pred_queue,
            rtc_stream_broadcaster=self.rtc_stream_broadcaster,
            stop_event=self.mp_stop_event,
            data_collector=self._data_collector,
            sink_status_holder=self.sink_status_holder,
        )

        # Start all workers
        stream_loader_proc.start()
        inference_server_proc.start()
        inference_server_monitor_thread.start()
        dispatching_thread.start()

        # Track processes and threads
        self.processes.extend([stream_loader_proc, inference_server_proc])
        self.threads.extend([dispatching_thread, inference_server_monitor_thread])

        logger.info("All worker processes started successfully")

    def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")

        # Signal all processes to stop
        self.mp_stop_event.set()

        # Get current process info for debugging
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
        logger.debug("Alive children of process '{}': {}", pid, alive_children)

        # Join threads first
        for thread in self.threads:
            if thread.is_alive():
                logger.debug("Joining thread: {}", thread.name)
                thread.join(timeout=10)
                if thread.is_alive():
                    logger.warning("Thread {} did not terminate within timeout", thread.name)

        # Join processes in reverse order so that consumers are terminated before producers.
        for process in self.processes[::-1]:
            if process.is_alive():
                logger.debug("Joining process: {}", process.name)
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning("Force terminating process: {}", process.name)
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.error("Force killing process {}", process.name)
                        process.kill()

        logger.info("All workers shut down gracefully")

        # Clear references
        self.processes.clear()
        self.threads.clear()
        self.shm_metrics.close()
        self.shm_metrics.unlink()
        self.source_status_shm.close()
        self.source_status_shm.unlink()
        self.inference_status_shm.close()
        self.inference_status_shm.unlink()

        self._cleanup_queues()

    def _cleanup_queues(self) -> None:
        """Final queue cleanup"""
        for q, name in [(self.frame_queue, "frame_queue"), (self.pred_queue, "pred_queue")]:
            if q is not None:
                try:
                    # https://runebook.dev/en/articles/python/library/multiprocessing/multiprocessing.Queue.close
                    q.close()
                    q.join_thread()
                    logger.debug("Successfully cleaned up {}", name)
                except Exception as e:
                    logger.warning("Error cleaning up {}: {}", name, e)
