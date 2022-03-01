"""
Sync Demo based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from ote_sdk.usecases.exportable_code.streamer import get_streamer


class SyncInferencer:
    """
    Synd demo for model inference

    Args:
        model: model for inference
        visualizer: for visualize inference results
        converter: convert model ourtput to annotation scene
    """

    def __init__(self, models, converters, visualizer) -> None:
        self.model = models[0]
        self.visualizer = visualizer
        self.converter = converters[0]

    def run(self, input_stream, loop):
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

            # any user's visualizer
            output = self.visualizer.draw(frame, annotation_scene, frame_meta)
            self.visualizer.show(output)

            if self.visualizer.is_quit():
                break
