"""
Async inferencer based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino.model_zoo.model_api.pipelines import AsyncPipeline

from ote_sdk.usecases.exportable_code.streamer import get_streamer


class AsyncInferencer:
    """
    Async inferencer

    Args:
        model: model for inference
        converter: convert model ourtput to annotation scene
        visualizer: for visualize inference results
    """

    def __init__(self, models, converters, visualizer) -> None:
        self.model = models[0]
        self.visualizer = visualizer
        self.converter = converters[0]
        self.async_pipeline = AsyncPipeline(self.model)

    def run(self, input_stream):
        """
        Async inference for input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream)
        next_frame_id = 0
        next_frame_id_to_show = 0
        stop = False
        for frame in streamer:
            results = self.async_pipeline.get_result(next_frame_id_to_show)
            while results:
                output = self.render_result(results)
                next_frame_id_to_show += 1
                self.visualizer.show(output)
                if self.visualizer.is_quit():
                    stop = True
                results = self.async_pipeline.get_result(next_frame_id_to_show)
            if stop:
                break
            self.async_pipeline.submit_data(frame, next_frame_id, {"frame": frame})
            next_frame_id += 1

        self.async_pipeline.await_all()
        for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
            results = self.async_pipeline.get_result(next_frame_id_to_show)
            output = self.render_result(results)
            self.visualizer.show(output)

    def render_result(self, results):
        """
        Render for results of inference
        """
        predictions, frame_meta = results
        annotation_scene = self.converter.convert_to_annotation(predictions, frame_meta)
        current_frame = frame_meta["frame"]
        # any user's visualizer
        output = self.visualizer.draw(current_frame, annotation_scene)
        return output
