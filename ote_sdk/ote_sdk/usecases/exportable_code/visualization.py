"""
Visualizer for results of prediction
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import cv2
import numpy as np

from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.usecases.exportable_code.streamer.streamer import MediaType
from ote_sdk.utils.shape_drawer import ShapeDrawer


class Visualizer:
    """
    Visualize the predicted output by drawing the annotations on the input image.

    :example:

        >>> predictions = inference_model.predict(frame)
        >>> annotation = prediction_converter.convert_to_annotation(predictions)
        >>> output = visualizer.draw(frame, annotation.shape, annotation.get_labels())
        >>> visualizer.show(output)
    """

    def __init__(
        self,
        media_type: Optional[MediaType] = None,
        window_name: Optional[str] = None,
        show_count: bool = False,
        is_one_label: bool = False,
        delay: Optional[int] = None,
    ):
        self.window_name = "Window" if window_name is None else window_name
        self.shape_drawer = ShapeDrawer(show_count, is_one_label)

        self.delay = delay
        if delay is None:
            self.delay = (
                0 if (media_type is None or media_type == MediaType.IMAGE) else 1
            )

    def draw(self, image: np.ndarray, annotation: AnnotationSceneEntity) -> np.ndarray:
        """
        Draw annotations on the image
        :param image: Input image
        :param annotation: Annotations to be drawn on the input image
        :return: Output image with annotations.
        """
        # TODO: Conversion is to be made in `show` not here.
        #   This requires ShapeDrawer.draw to be updated
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.shape_drawer.draw(image, annotation, labels=[])

    def show(self, image: np.ndarray) -> None:
        """
        Show result image
        """
        # TODO: RGB2BGR Conversion is to be made here.
        #   This requires ShapeDrawer.draw to be updated
        cv2.imshow(self.window_name, image)

    def is_quit(self) -> bool:
        """
        Check user wish to quit
        """
        return ord("q") == cv2.waitKey(self.delay)
