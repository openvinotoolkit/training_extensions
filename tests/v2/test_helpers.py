# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from otx.v2.api.entities.annotation import (
    Annotation,
)
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.shapes.ellipse import Ellipse
from otx.v2.api.entities.shapes.polygon import Point, Polygon
from otx.v2.api.entities.shapes.rectangle import Rectangle

if TYPE_CHECKING:
    from otx.v2.api.entities.label import LabelEntity


def generate_random_annotated_image(
    image_width: int,
    image_height: int,
    labels: Sequence[LabelEntity],
    min_size=50,
    max_size=250,
    shape: str | None = None,
    max_shapes: int = 10,
    intensity_range: list[tuple[int, int]] | None = None,
    random_seed: int | None = None,
    use_mask_as_annotation: bool = False,
) -> tuple[np.ndarray, list[Annotation]]:
    """
    Generate a random image with the corresponding annotation entities.

    :param intensity_range: Intensity range for RGB channels ((r_min, r_max), (g_min, g_max), (b_min, b_max))
    :param max_shapes: Maximum amount of shapes in the image
    :param shape: {"rectangle", "ellipse", "triangle"}
    :param image_height: Height of the image
    :param image_width: Width of the image
    :param labels: Task Labels that should be applied to the respective shape
    :param min_size: Minimum size of the shape(s)
    :param max_size: Maximum size of the shape(s)
    :param random_seed: Seed to initialize the random number generator
    :param use_mask_as_annotation: If True, masks will be added in annotation
    :return: uint8 array, list of shapes
    """
    from skimage.draw import random_shapes, rectangle

    if intensity_range is None:
        intensity_range = [(100, 200)]

    image1: np.ndarray | None = None
    sc_labels = []
    # Sporadically, it might happen there is no shape in the image, especially on low-res images.
    # It'll retry max 5 times until we see a shape, and otherwise raise a runtime error
    if shape == "ellipse":  # ellipse shape is not available in random_shapes function. use circle instead
        shape = "circle"
    for _ in range(5):
        rand_image, sc_labels = random_shapes(
            (image_height, image_width),
            min_shapes=1,
            max_shapes=max_shapes,
            intensity_range=intensity_range,
            min_size=min_size,
            max_size=max_size,
            shape=shape,
            random_seed=random_seed,
        )
        num_shapes = len(sc_labels)
        if num_shapes > 0:
            image1 = rand_image
            break

    if image1 is None:
        msg = "Was not able to generate a random image that contains any shapes"
        raise RuntimeError(msg)

    annotations: list[Annotation] = []
    for sc_label in sc_labels:
        sc_label_name = sc_label[0]
        sc_label_shape_r = sc_label[1][0]
        sc_label_shape_c = sc_label[1][1]
        y_min, y_max = max(0.0, float(sc_label_shape_r[0] / image_height)), min(
            1.0, float(sc_label_shape_r[1] / image_height)
        )
        x_min, x_max = max(0.0, float(sc_label_shape_c[0] / image_width)), min(
            1.0, float(sc_label_shape_c[1] / image_width)
        )

        import secrets

        if sc_label_name == "ellipse":
            # Fix issue with newer scikit-image libraries that generate ellipses.
            # For now we render a rectangle on top of it
            sc_label_name = "rectangle"
            rr, cc = rectangle(
                start=(sc_label_shape_r[0], sc_label_shape_c[0]),
                end=(sc_label_shape_r[1] - 1, sc_label_shape_c[1] - 1),
                shape=image1.shape,
            )
            image1[rr, cc] = (
                secrets.randbelow(201),
                secrets.randbelow(201),
                secrets.randbelow(201),
            )
        if sc_label_name == "circle":
            sc_label_name = "ellipse"

        label_matches = [label for label in labels if sc_label_name == label.name]

        if len(label_matches) > 0:
            label = label_matches[0]
            box_annotation = Annotation(
                Rectangle(x1=x_min, y1=y_min, x2=x_max, y2=y_max),
                labels=[ScoredLabel(label, probability=1.0)],
            )

            annotation: Annotation

            if label.name == "ellipse":
                annotation = Annotation(
                    Ellipse(
                        x1=box_annotation.shape.x1,
                        y1=box_annotation.shape.y1,
                        x2=box_annotation.shape.x2,
                        y2=box_annotation.shape.y2,
                    ),
                    labels=box_annotation.get_labels(include_empty=True),
                )
            elif label.name == "triangle":
                points = [
                    Point(
                        x=(box_annotation.shape.x1 + box_annotation.shape.x2) / 2,
                        y=box_annotation.shape.y1,
                    ),
                    Point(x=box_annotation.shape.x1, y=box_annotation.shape.y2),
                    Point(x=box_annotation.shape.x2, y=box_annotation.shape.y2),
                ]

                annotation = Annotation(
                    Polygon(points=points),
                    labels=box_annotation.get_labels(include_empty=True),
                )
            else:
                annotation = box_annotation

            annotations.append(annotation)

            if use_mask_as_annotation:
                mask = np.zeros_like(image1, dtype=np.uint8)
                y_min, y_max = int(y_min * image_height), int(y_max * image_height)
                x_min, x_max = int(x_min * image_width), int(x_max * image_width)

                coords_object = np.where(image1[y_min:y_max, x_min:x_max] < 255)
                mask[y_min:y_max, x_min:x_max][coords_object] = 1
                mask = mask.sum(axis=-1)
                mask[mask > 0] = 1
                mask_annotation = Annotation(
                    Image(data=mask, size=mask.shape),
                    labels=box_annotation.get_labels(include_empty=True),
                )
                annotations.append(mask_annotation)

    return image1, annotations
