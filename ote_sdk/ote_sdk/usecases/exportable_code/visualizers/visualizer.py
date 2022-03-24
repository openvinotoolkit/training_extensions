"""
Visualizer for results of prediction
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Optional

import cv2
import numpy as np

from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.utils.shape_drawer import ShapeDrawer


class IVisualizer(metaclass=abc.ABCMeta):
    """
    Interface for converter
    """

    @abc.abstractmethod
    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationSceneEntity,
        meta: dict,
    ) -> np.ndarray:
        """
        Draw annotations on the image
        :param image: Input image
        :param annotation: Annotations to be drawn on the input image
        :param metadata: Metadata is needed to render
        :return: Output image with annotations.
        """
        raise NotImplementedError

    def show(self, image: np.ndarray) -> None:
        """
        Show result image
        """

        raise NotImplementedError


class HandlerVisualizer:
    """
    Handler for visualizers

    Args:
        visualizer: visualize inference results
    """

    def __init__(self, visualizer: IVisualizer) -> None:
        self.visualizer = visualizer

    def __enter__(self):
        cv2.namedWindow(
            self.visualizer.window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )
        if hasattr(self.visualizer, "trackbar_name"):
            cv2.createTrackbar(
                self.visualizer.trackbar_name,
                self.visualizer.window_name,
                0,
                100,
                lambda x: x,
            )

        return self.visualizer

    def __exit__(self, *exc) -> None:
        cv2.destroyAllWindows()


class Visualizer(IVisualizer):
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
        window_name: Optional[str] = None,
        show_count: bool = False,
        is_one_label: bool = False,
        delay: Optional[int] = None,
    ) -> None:
        self.window_name = "Window" if window_name is None else window_name
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )
        self.shape_drawer = ShapeDrawer(show_count, is_one_label)

        self.delay = delay
        if delay is None:
            self.delay = 1

    def draw(
        self,
        image: np.ndarray,
        annotation: AnnotationSceneEntity,
        meta: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Draw annotations on the image
        :param image: Input image
        :param annotation: Annotations to be drawn on the input image
        :return: Output image with annotations.
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.shape_drawer.draw(image, annotation, labels=[])

    def show(self, image: np.ndarray) -> None:
        """
        Show result image
        """

        cv2.imshow(self.window_name, image)

    def is_quit(self) -> bool:
        """
        Check user wish to quit
        """
        return ord("q") == cv2.waitKey(self.delay)
