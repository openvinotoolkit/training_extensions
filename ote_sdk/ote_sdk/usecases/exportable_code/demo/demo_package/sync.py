"""
Sync Demo based on ModelAPI
"""
# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

from ote_sdk.usecases.exportable_code.streamer import get_streamer


class SyncDemo:
    """
    Synd demo for model inference

    Args:
        model: model for inference
        visualizer: for visualize inference results
        converter: convert model ourtput to annotation scene
    """

    def __init__(self, model, visualizer, converter) -> None:
        self.model = model
        self.visualizer = visualizer
        self.converter = converter

    def run(self, input_stream):
        """
        Run demo using input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream)
        for frame in streamer:
            # getting result include preprocessing, infer, postprocessing for sync infer
            dict_data, input_meta = self.model.preprocess(frame)
            raw_result = self.model.infer_sync(dict_data)
            predictions = self.model.postprocess(raw_result, input_meta)
            annotation_scene = self.converter.convert_to_annotation(
                predictions, input_meta
            )

            # any user's visualizer
            output = self.visualizer.draw(frame, annotation_scene)
            self.visualizer.show(output)

            if self.visualizer.is_quit():
                break
