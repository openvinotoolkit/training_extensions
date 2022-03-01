"""
Visualizer for results of anomaly task prediction
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import cv2
import numpy as np

from ote_sdk.entities.annotation import AnnotationSceneEntity

from .visualizer import Visualizer


class AnomalyVisualizer(Visualizer):
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
    ):
        super().__init__(window_name, show_count, is_one_label, delay)
        self.trackbar_name = "Opacity"
        cv2.createTrackbar(self.trackbar_name, self.window_name, 0, 100, lambda x: x)

    @staticmethod
    def to_heat_mask(mask: np.ndarray) -> np.ndarray:
        """
        Create heat mask from saliency map
        :param mask: saliency map
        """
        heat_mask = cv2.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.applyColorMap(heat_mask.astype(np.uint8), cv2.COLORMAP_JET)

    # pylint:disable=signature-differs
    def draw(  # type: ignore[override]
        self, image: np.ndarray, annotation: AnnotationSceneEntity, meta: dict  # type: ignore[override]
    ) -> np.ndarray:  # type: ignore[override]
        """
        Draw annotations on the image
        :param image: Input image
        :param annotation: Annotations to be drawn on the input image
        :param metadata: Metadata with saliency map
        :return: Output image with annotations.
        """

        heat_mask = self.to_heat_mask(1 - meta["anomaly_map"])
        alpha = cv2.getTrackbarPos(self.trackbar_name, self.window_name) / 100.0
        image = (1 - alpha) * image + alpha * heat_mask
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        return self.shape_drawer.draw(image, annotation, labels=[])
