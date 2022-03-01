"""
Sync Pipeline based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List

import numpy as np
from openvino.model_zoo.model_api.models import Model

from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    IPredictionToAnnotationConverter,
)
from ote_sdk.usecases.exportable_code.streamer import get_streamer
from ote_sdk.usecases.exportable_code.visualizers import Visualizer


class ChainInferencer:
    """
    Sync pipeline for task-chain inference

    Args:
        models: List of models for inference in correct order
        visualizer: for visualize inference results
        converters: convert model ourtput to annotation scene
    """

    def __init__(
        self,
        models: List[Model],
        converters: List[IPredictionToAnnotationConverter],
        visualizer: Visualizer,
    ) -> None:
        self.models = models
        self.visualizer = visualizer
        self.converters = converters

    # pylint: disable=too-many-locals
    def single_run(self, input_image) -> AnnotationSceneEntity:
        """
        Inference for single image
        """
        current_objects = [(input_image, Annotation(Rectangle(0, 0, 1, 1), labels=[]))]
        result_scene = AnnotationSceneEntity([], AnnotationSceneKind.PREDICTION)
        for index, model in enumerate(self.models):
            new_objects = []
            for item, parent_annotation in current_objects:
                predictions, frame_meta = model(item)
                annotation_scene = self.converters[index].convert_to_annotation(
                    predictions, frame_meta
                )
                for annotation in annotation_scene.annotations:
                    new_item, item_annotation = self.crop(
                        item, parent_annotation, annotation
                    )
                    new_objects.append((new_item, item_annotation))
                    if parent_annotation.shape == item_annotation.shape:
                        for label in item_annotation.get_labels():
                            parent_annotation.append_label(label)
                    else:
                        result_scene.append_annotation(item_annotation)
            current_objects = new_objects
        return result_scene

    @staticmethod
    def crop(item: np.ndarray, parent_annotation, item_annotation):
        """
        Glue for models
        """
        new_item = item_annotation.shape.to_rectangle().crop_numpy_array(item)
        item_annotation.shape = item_annotation.shape.normalize_wrt_roi_shape(
            parent_annotation.shape
        )
        return new_item, item_annotation

    def run(self, input_stream, loop):
        """
        Run demo using input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream, loop)
        for frame in streamer:
            # getting result for single image
            annotation_scene = self.single_run(frame)

            # any user's visualizer
            output = self.visualizer.draw(frame, annotation_scene)
            self.visualizer.show(output)
            if self.visualizer.is_quit():
                break
