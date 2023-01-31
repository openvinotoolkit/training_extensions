"""Async executor based on ModelAPI."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Tuple, Union

import numpy as np
from openvino.model_zoo.model_api.pipelines import AsyncPipeline

from otx.api.usecases.exportable_code.demo.demo_package.model_container import (
    ModelContainer,
)
from otx.api.usecases.exportable_code.demo.demo_package.utils import (
    create_output_converter,
)
from otx.api.usecases.exportable_code.streamer import get_streamer
from otx.api.usecases.exportable_code.visualizers import Visualizer


class AsyncExecutor:
    """Async inferencer.

    Args:
        model: model for inference
        visualizer: visualizer of inference results
    """

    def __init__(self, model: ModelContainer, visualizer: Visualizer) -> None:
        self.model = model.core_model
        self.visualizer = visualizer
        self.converter = create_output_converter(model.task_type, model.labels)
        self.async_pipeline = AsyncPipeline(self.model)

    def run(self, input_stream: Union[int, str], loop: bool = False) -> None:
        """Async inference for input stream (image, video stream, camera)."""
        streamer = get_streamer(input_stream, loop)
        next_frame_id = 0
        next_frame_id_to_show = 0
        stop_visualization = False

        for frame in streamer:
            results = self.async_pipeline.get_result(next_frame_id_to_show)
            while results:
                output = self.render_result(results)
                next_frame_id_to_show += 1
                self.visualizer.show(output)
                if self.visualizer.is_quit():
                    stop_visualization = True
                results = self.async_pipeline.get_result(next_frame_id_to_show)
            if stop_visualization:
                break
            self.async_pipeline.submit_data(frame, next_frame_id, {"frame": frame})
            next_frame_id += 1
        self.async_pipeline.await_all()
        for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
            results = self.async_pipeline.get_result(next_frame_id_to_show)
            output = self.render_result(results)
            self.visualizer.show(output)

    def render_result(self, results: Tuple[Any, dict]) -> np.ndarray:
        """Render for results of inference."""
        predictions, frame_meta = results
        annotation_scene = self.converter.convert_to_annotation(predictions, frame_meta)
        current_frame = frame_meta["frame"]
        output = self.visualizer.draw(current_frame, annotation_scene, frame_meta)
        return output
