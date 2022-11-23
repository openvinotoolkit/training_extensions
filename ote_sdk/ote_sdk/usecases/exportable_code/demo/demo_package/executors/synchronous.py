"""
Sync Executor based on ModelAPI
"""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union

from ote_sdk.entities.model_template import TaskType
from ote_sdk.usecases.exportable_code.demo.demo_package.model_container import (
    ModelContainer,
)
from ote_sdk.usecases.exportable_code.demo.demo_package.utils import (
    create_output_converter,
)
from ote_sdk.usecases.exportable_code.streamer import get_streamer
from ote_sdk.usecases.exportable_code.visualizers import IVisualizer
from ote_sdk.utils import Tiler
from ote_sdk.utils.detection_utils import detection2array


class SyncExecutor:
    """
    Synd executor for model inference

    Args:
        model: model for inference
        visualizer: visualizer of inference results
    """

    def __init__(self, model: ModelContainer, visualizer: IVisualizer) -> None:
        self.model = model.core_model
        self.params = model.parameters
        self.visualizer = visualizer
        self.converter = create_output_converter(model.task_type, model.labels)
        self.task_type = model.task_type
        self.tiler = self.setup_tiler()

    def setup_tiler(self):
        """Setup tiler

        Returns:
            Tiler: tiler module
        """
        if (
            not self.params.get("tiling_parameters")
            or not self.params["tiling_parameters"]["enable_tiling"]["value"]
        ):
            return None

        segm = False
        if (
            self.task_type is TaskType.ROTATED_DETECTION
            or self.task_type is TaskType.INSTANCE_SEGMENTATION
        ):
            segm = True
        tile_size = self.params["tiling_parameters"]["tile_size"]["value"]
        tile_overlap = self.params["tiling_parameters"]["tile_overlap"]["value"]
        max_number = self.params["tiling_parameters"]["tile_max_number"]["value"]
        tiler = Tiler(tile_size, tile_overlap, max_number, self.model, segm)
        return tiler

    def infer(self, frame):
        """Infer with original image

        Args:
            frame (np.ndarray): image

        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """
        # getting result include preprocessing, infer, postprocessing for sync infer
        predictions, frame_meta = self.model(frame)
        predictions = detection2array(predictions)
        annotation_scene = self.converter.convert_to_annotation(predictions, frame_meta)
        return annotation_scene, frame_meta

    def infer_tile(self, frame):
        """Infer by patching full image to tiles

        Args:
            frame (np.ndarray): image

        Returns:
            annotation_scene (AnnotationScene): prediction
            frame_meta (Dict): dict with original shape
        """

        detections, _ = self.tiler.predict(frame)
        annotation_scene = self.converter.convert_to_annotation(
            detections, metadata={"original_shape": frame.shape}
        )
        return annotation_scene, None

    def run(self, input_stream: Union[int, str], loop: bool = False) -> None:
        """
        Run demo using input stream (image, video stream, camera)
        """
        streamer = get_streamer(input_stream, loop)

        for frame in streamer:
            annotation_scene, frame_meta = (
                self.infer_tile(frame) if self.tiler else self.infer(frame)
            )
            output = self.visualizer.draw(frame, annotation_scene, frame_meta)
            self.visualizer.show(output)
            if self.visualizer.is_quit():
                break
