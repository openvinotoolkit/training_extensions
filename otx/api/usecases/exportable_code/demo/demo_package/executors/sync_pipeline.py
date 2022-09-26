"""Sync pipeline executor based on ModelAPI."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Tuple, Union

import numpy as np

from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.usecases.exportable_code.demo.demo_package.model_container import (
    ModelContainer,
)
from otx.api.usecases.exportable_code.demo.demo_package.utils import (
    create_output_converter,
)
from otx.api.usecases.exportable_code.streamer import get_streamer
from otx.api.usecases.exportable_code.visualizers import Visualizer
from otx.api.utils.shape_factory import ShapeFactory


class ChainExecutor:
    """Sync executor for task-chain inference.

    Args:
        models: list of models for inference
        visualizer: visualizer of inference results
    """

    def __init__(
        self,
        models: List[ModelContainer],
        visualizer: Visualizer,
    ) -> None:
        self.models = models
        self.visualizer = visualizer
        self.converters = []
        for model in self.models:
            self.converters.append(create_output_converter(model.task_type, model.labels))

    # pylint: disable=too-many-locals
    def single_run(self, input_image: np.ndarray) -> AnnotationSceneEntity:
        """Inference for single image."""
        current_objects = [(input_image, Annotation(Rectangle(0, 0, 1, 1), labels=[]))]
        result_scene = AnnotationSceneEntity([], AnnotationSceneKind.PREDICTION)
        for index, model in enumerate(self.models):
            new_objects = []
            for item, parent_annotation in current_objects:
                predictions, frame_meta = model.core_model(item)
                annotation_scene = self.converters[index].convert_to_annotation(predictions, frame_meta)
                for annotation in annotation_scene.annotations:
                    new_item, item_annotation = self.crop(item, parent_annotation, annotation)
                    new_objects.append((new_item, item_annotation))
                    if model.task_type.is_global:
                        for label in item_annotation.get_labels():
                            parent_annotation.append_label(label)
                    else:
                        result_scene.append_annotation(item_annotation)
            current_objects = new_objects
        return result_scene

    @staticmethod
    def crop(
        item: np.ndarray, parent_annotation: Annotation, item_annotation: Annotation
    ) -> Tuple[np.ndarray, Annotation]:
        """Crop operation between chain stages."""
        new_item = ShapeFactory.shape_as_rectangle(item_annotation.shape).crop_numpy_array(item)
        item_annotation.shape = item_annotation.shape.normalize_wrt_roi_shape(
            ShapeFactory.shape_as_rectangle(parent_annotation.shape)
        )
        return new_item, item_annotation

    def run(self, input_stream: Union[int, str], loop: bool = False) -> None:
        """Run demo using input stream (image, video stream, camera)."""
        streamer = get_streamer(input_stream, loop)

        for frame in streamer:
            # getting result for single image
            annotation_scene = self.single_run(frame)
            output = self.visualizer.draw(frame, annotation_scene, {})
            self.visualizer.show(output)
            if self.visualizer.is_quit():
                break
