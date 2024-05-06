# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Async executor based on ModelAPI."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from model_api.pipelines import AsyncPipeline

if TYPE_CHECKING:
    import numpy as np
    from demo_package.model_wrapper import ModelWrapper


from demo_package.streamer import get_streamer
from demo_package.visualizers import BaseVisualizer, dump_frames


class AsyncExecutor:
    """Async inferencer.

    Args:
        model: model for inference
        visualizer: visualizer of inference results
    """

    def __init__(self, model: ModelWrapper, visualizer: BaseVisualizer) -> None:
        self.model = model
        self.visualizer = visualizer
        self.async_pipeline = AsyncPipeline(self.model.core_model)

    def run(self, input_stream: int | str, loop: bool = False) -> None:
        """Async inference for input stream (image, video stream, camera)."""
        streamer = get_streamer(input_stream, loop)
        next_frame_id = 0
        next_frame_id_to_show = 0
        stop_visualization = False
        saved_frames = []

        for frame in streamer:
            results = self.async_pipeline.get_result(next_frame_id_to_show)
            while results:
                start_time = time.perf_counter()
                output = self.render_result(results)
                next_frame_id_to_show += 1
                self.visualizer.show(output)
                if self.visualizer.output:
                    saved_frames.append(output)
                stop_visualization = self.visualizer.is_quit()
                # visualize video not faster than the original FPS
                self.visualizer.video_delay(time.perf_counter() - start_time, streamer)
                results = self.async_pipeline.get_result(next_frame_id_to_show)
            if stop_visualization:
                break
            self.async_pipeline.submit_data(frame, next_frame_id, {"frame": frame})
            next_frame_id += 1
        self.async_pipeline.await_all()
        for next_id in range(next_frame_id_to_show, next_frame_id):
            start_time = time.perf_counter()
            results = self.async_pipeline.get_result(next_id)
            if not results:
                msg = "Async pipeline returned None results"
                raise RuntimeError(msg)
            output = self.render_result(results)
            self.visualizer.show(output)
            if self.visualizer.output:
                saved_frames.append(output)
            # visualize video not faster than the original FPS
            self.visualizer.video_delay(time.perf_counter() - start_time, streamer)
        dump_frames(saved_frames, self.visualizer.output, input_stream, streamer)

    def render_result(self, results: tuple[Any, dict]) -> np.ndarray:
        """Render for results of inference."""
        predictions, frame_meta = results
        current_frame = frame_meta["frame"]
        return self.visualizer.draw(current_frame, predictions)
