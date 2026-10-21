# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass

import numpy as np
from loguru import logger

from app.db import get_db_session
from app.models import DisconnectedSinkConfig, Sink, SinkType
from app.models.sink import SinkStatus, SinkStatusCode
from app.services import DispatchService, SinkService
from app.services.data_collect import DataCollector
from app.services.dispatchers import Dispatcher, DispatchError
from app.services.event.event_bus import EventBus, EventType
from app.stream.stream_data import StreamData
from app.webrtc import FrameBroadcaster
from app.workers.base import BaseThreadWorker
from app.workers.sink_status_holder import SinkStatusHolder


class DispatchingWorker(BaseThreadWorker):
    """
    A thread that pulls predictions from the queue and dispatches them to the configured outputs
    and WebRTC visualization stream.
    """

    ROLE = "Dispatching"

    def __init__(
        self,
        event_bus: EventBus,
        pred_queue: mp.Queue,
        rtc_stream_broadcaster: FrameBroadcaster[np.ndarray],
        stop_event: EventClass,
        data_collector: DataCollector,
        sink_status_holder: SinkStatusHolder,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._event_bus = event_bus
        self._pred_queue = pred_queue
        self._rtc_stream_broadcaster = rtc_stream_broadcaster

        self._data_collector = data_collector
        self._sink_status_holder = sink_status_holder

        self._sink: Sink
        self._destinations: list[Dispatcher] = []

        self._sink, self._destinations = self._load_sink()
        logger.info("Active sink set to {}", self._sink)
        event_bus.subscribe(
            [EventType.SINK_CHANGED, EventType.PIPELINE_STATUS_CHANGED],
            self._reload_sink,
        )
        event_bus.subscribe([EventType.SOURCE_CHANGED], self._on_source_changed)
        event_bus.subscribe([EventType.PIPELINE_STATUS_CHANGED], self._on_pipeline_status_changed)

    def setup(self) -> None:
        pass

    def _load_sink(self) -> tuple[Sink, list[Dispatcher]]:
        with get_db_session() as db:
            active_sink = SinkService(event_bus=self._event_bus, db_session=db).get_active_sink()
        sink = active_sink if active_sink is not None else DisconnectedSinkConfig()

        # Building the dispatchers (e.g. instantiating an MQTT client) may fail for a
        # misconfigured or unsupported sink; never let that crash the dispatching worker.
        try:
            destinations = DispatchService.get_destinations(output_configs=[sink])
        except Exception as e:
            logger.exception("Failed to build dispatchers for sink id={} name={!r}", sink.id, sink.name)
            self._report_sink_error(sink, f"Failed to initialize sink: {e}")
            return sink, []

        # Connect each dispatcher independently. A sink that is not reachable when the pipeline is
        # enabled (see the separate sink test endpoint) must not crash the worker: log it, report
        # the sink health as ERROR and drop the unreachable destination so the pipeline keeps
        # running (predictions are still routed to the WebRTC stream).
        connected: list[Dispatcher] = []
        for destination in destinations:
            try:
                destination.connect()
                connected.append(destination)
            except Exception as e:
                logger.exception(
                    "Failed to connect sink dispatcher: sink id={} name={!r} dispatcher={}",
                    sink.id,
                    sink.name,
                    type(destination).__name__,
                )
                self._report_sink_error(sink, f"Failed to connect to sink: {e}")
        return sink, connected

    def _reload_sink(self) -> None:
        self._sink, self._destinations = self._load_sink()
        logger.info("Active sink set to {}", self._sink)

    def _report_sink_error(self, sink: Sink, message: str) -> None:
        """Publish an ERROR sink status (skipped when no sink is configured)."""
        if sink.sink_type == SinkType.DISCONNECTED:
            return
        self._sink_status_holder.status = SinkStatus(code=SinkStatusCode.ERROR, sink_id=sink.id, message=message)

    def _on_source_changed(self) -> None:
        self._rtc_stream_broadcaster.clear()
        logger.info("Cleared WebRTC broadcaster after source change")

    def _on_pipeline_status_changed(self) -> None:
        self._rtc_stream_broadcaster.clear()
        logger.info("Cleared WebRTC broadcaster after pipeline status change")

    def run_loop(self) -> None:
        while not self.should_stop():
            try:
                self._process_next_prediction()
            except Exception:
                # Defense in depth: a single malformed prediction or unexpected error must never
                # kill the dispatching thread, otherwise the pipeline would silently stop routing
                # results (to sinks and to the WebRTC preview). Log it and keep going.
                logger.exception("Unexpected error in dispatching loop; continuing")
                self.stop_aware_sleep(0.5)

    def _process_next_prediction(self) -> None:
        # Read from the queue
        try:
            stream_data: StreamData = self._pred_queue.get(timeout=1)
        except queue.Empty:
            logger.debug("Nothing to dispatch yet")
            return

        inference_data = stream_data.inference_data
        if inference_data is None:
            logger.error("Missing inference data in stream_data; skipping dispatch")
            return

        image_with_visualization = inference_data.visualized_prediction
        prediction = inference_data.prediction

        # Broadcast to the WebRTC stream first so the live preview stays responsive even if an
        # external sink or data collection step below is momentarily slow (e.g. a blocking disk or
        # DB write) or fails. broadcast() is non-blocking (drop-oldest per consumer) and is guarded
        # so a visualization error never blocks sinks or data collection.
        try:
            self._rtc_stream_broadcaster.broadcast(image_with_visualization)
        except Exception:
            logger.exception("Failed to broadcast frame to WebRTC stream")

        # Postprocess and dispatch results to external sinks (folder, MQTT, ROS, webhook, ...).
        # Skipped when no sink is configured; WebRTC and data collection still run regardless.
        if self._sink.sink_type != SinkType.DISCONNECTED:
            self._dispatch_to_sinks(stream_data, image_with_visualization, prediction)

        # Collect the image to project dataset if needed. Guarded so a collection failure
        # (e.g. a DB or disk error) does not stop dispatching or crash the thread.
        try:
            self._data_collector.collect(
                timestamp=stream_data.timestamp,
                frame_data=stream_data.frame_data,
                inference_data=inference_data,
            )
        except Exception:
            logger.exception("Failed to collect frame to the project dataset; continuing")

    def _dispatch_to_sinks(self, stream_data: StreamData, image_with_visualization, prediction) -> None:  # noqa: ANN001
        """Dispatch results to every configured destination, isolating per-destination failures."""
        dispatch_error_message: str | None = None
        if not self._destinations:
            # Keep the last reported sink status (e.g. connect/build failure from _load_sink)
            # instead of flipping it to OK when nothing is actually connected.
            return
        for destination in self._destinations:
            try:
                destination.dispatch(
                    original_image=stream_data.frame_data,
                    image_with_visualization=image_with_visualization,
                    predictions=prediction,
                )
            except DispatchError:
                dispatch_error_message = "Failed to dispatch to one or more destinations"
                logger.exception(
                    "Failed to dispatch results to sink: id={}, name={!r} dispatcher={}",
                    self._sink.id,
                    self._sink.name,
                    type(destination).__name__,
                )
            except Exception as e:
                # A dispatcher may raise something other than DispatchError; treat it the same way
                # (report sink error, keep the pipeline running) instead of letting it bubble up.
                dispatch_error_message = f"Unexpected error dispatching to sink: {e}"
                logger.exception(
                    "Unexpected error dispatching results to sink: id={}, name={!r} dispatcher={}",
                    self._sink.id,
                    self._sink.name,
                    type(destination).__name__,
                )

        if dispatch_error_message is not None:
            self._sink_status_holder.status = SinkStatus(
                code=SinkStatusCode.ERROR,
                sink_id=self._sink.id,
                message=dispatch_error_message,
            )
        else:
            self._sink_status_holder.status = SinkStatus(code=SinkStatusCode.OK, sink_id=self._sink.id)
