"""Synchronous Executor based on ModelAPI."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time
from typing import Union

from ..model_wrapper import ModelWrapper
from ..streamer import get_streamer
from ..visualizers import BaseVisualizer, dump_frames


class SyncExecutor:
    """Synchronous executor for model inference.

    Args:
        model (ModelContainer): model for inference
        visualizer (Visualizer): visualizer of inference results. Defaults to None.
    """

    def __init__(self, model: ModelWrapper, visualizer: BaseVisualizer) -> None:
        self.model = model
        self.visualizer = visualizer

    def run(self, input_stream: Union[int, str], loop: bool = False) -> None:
        """Run demo using input stream (image, video stream, camera)."""
        streamer = get_streamer(input_stream, loop)
        saved_frames = []

        for frame in streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            start_time = time.perf_counter()
            predictions, frame_meta = self.model(frame)
            output = self.visualizer.draw(frame, predictions, frame_meta)
            self.visualizer.show(output)
            if output is not None:
                saved_frames.append(output)
            if self.visualizer.is_quit():
                break
            # visualize video not faster than the original FPS
            self.visualizer.video_delay(time.perf_counter() - start_time, streamer)

        dump_frames(saved_frames, self.visualizer.output, input_stream, streamer)
