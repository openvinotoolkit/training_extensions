# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Synchronous Executor based on ModelAPI."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from demo_package.model_wrapper import ModelWrapper
    from demo_package.visualizers import BaseVisualizer

from demo_package.streamer.streamer import get_streamer
from demo_package.visualizers import dump_frames


class SyncExecutor:
    """Synchronous executor for model inference.

    Args:
        model (ModelContainer): model for inference
        visualizer (Visualizer): visualizer of inference results. Defaults to None.
    """

    def __init__(self, model: ModelWrapper, visualizer: BaseVisualizer) -> None:
        self.model = model
        self.visualizer = visualizer

    def run(self, input_stream: int | str, loop: bool = False) -> None:
        """Run demo using input stream (image, video stream, camera)."""
        streamer = get_streamer(input_stream, loop)
        saved_frames = []

        for frame in streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            start_time = time.perf_counter()
            predictions, _ = self.model(frame)
            output = self.visualizer.draw(frame, predictions)
            self.visualizer.show(output)
            if output is not None:
                saved_frames.append(output)
            if self.visualizer.is_quit():
                break
            # visualize video not faster than the original FPS
            self.visualizer.video_delay(time.perf_counter() - start_time, streamer)

        dump_frames(saved_frames, self.visualizer.output, input_stream, streamer)
