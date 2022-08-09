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
from ote_sdk.usecases.exportable_code.visualizers import IVisualizer


class SyncExecutor:
    """
    Synd executor for model inference

    Args:
        model: model for inference
        visualizer: visualizer of inference results
    """

    def __init__(self, model: ModelContainer, visualizer: IVisualizer) -> None:
        self.model = model.core_model
        self.visualizer = visualizer
        self.converter = create_output_converter(model.task_type, model.labels)

    def run(self, input_stream: Union[int, str], loop: bool = False) -> None:
        """
        Run demo using input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream, loop)

        for frame in streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            predictions, frame_meta = self.model(frame)
            annotation_scene = self.converter.convert_to_annotation(
                predictions, frame_meta
            )
            output = self.visualizer.draw(frame, annotation_scene, frame_meta)
            self.visualizer.show(output)
            if self.visualizer.is_quit():
                break
