"""
Sync Executor based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union

from ote_sdk.usecases.exportable_code.demo.demo_package.model_container import (
    ModelContainer,
)
from ote_sdk.usecases.exportable_code.demo.demo_package.utils import (
    create_output_converter,
)
from ote_sdk.usecases.exportable_code.streamer import get_streamer
from ote_sdk.usecases.exportable_code.visualizers import HandlerVisualizer, Visualizer


class SyncExecutor:
    """
    Synd executor for model inference

    Args:
        model: model for inference
        visualizer: for visualize inference results
    """

    def __init__(self, model: ModelContainer, visualizer: Visualizer) -> None:
        self.model = model.core_model
        self.visualizer = visualizer
        self.converter = create_output_converter(model.task_type, model.labels)

    def run(self, input_stream: Union[int, str], loop=False):
        """
        Run demo using input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream, loop)

        with HandlerVisualizer(self.visualizer) as visualizer:
            for frame in streamer:
                # getting result include preprocessing, infer, postprocessing for sync infer
                predictions, frame_meta = self.model(frame)
                annotation_scene = self.converter.convert_to_annotation(
                    predictions, frame_meta
                )
                # any user's visualizer
                output = visualizer.draw(frame, annotation_scene, frame_meta)
                visualizer.show(output)
                if visualizer.is_quit():
                    break
