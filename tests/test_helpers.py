# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import contextlib
import logging
import os
import random
import tempfile
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from bson import ObjectId

from otx.api.configuration import ConfigurableParameters
from otx.api.configuration.elements import (
    ParameterGroup,
    add_parameter_group,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    string_attribute,
)
from otx.api.configuration.model_lifecycle import ModelLifecycle
from otx.api.entities.annotation import Annotation
from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelGroup, LabelGroupType, LabelSchemaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.polygon import Point, Polygon
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.image import Image

logger = logging.getLogger(__name__)


def generate_unique_id() -> ID:
    """
    Generates unique ID for testing
    :return:
    """
    return ID(ObjectId())


class LabelSchemaExample:
    def __init__(self) -> None:
        self.label_domain = Domain.CLASSIFICATION

        self.flowering = self.new_label_by_name("flowering")
        self.no_plant = self.new_label_by_name("no_plant")
        self.vegetative = self.new_label_by_name("vegetative")

    def new_label_by_name(self, name: str, is_empty: bool = False) -> LabelEntity:
        label = LabelEntity(name=name, color=Color.random(), domain=self.label_domain, is_empty=is_empty)
        label.id_ = generate_unique_id()
        return label

    def add_hierarchy(self, label_schema: LabelSchemaEntity) -> Tuple[LabelEntity, LabelEntity, LabelEntity]:
        """Adds children to flowering, no_plant and vegetative"""
        label_schema.add_group(
            LabelGroup(
                "plant_state",
                [self.flowering, self.no_plant, self.vegetative],
                LabelGroupType.EXCLUSIVE,
            )
        )
        flower_partial_visible = self.new_label_by_name("flower_partial_visible")
        flower_fully_visible = self.new_label_by_name("flower_fully_visible")
        label_schema.add_group(
            LabelGroup(
                "flowering_state",
                [flower_fully_visible, flower_partial_visible],
                LabelGroupType.EXCLUSIVE,
            )
        )
        label_schema.add_child(self.flowering, flower_partial_visible)
        label_schema.add_child(self.flowering, flower_fully_visible)

        assert self.flowering == label_schema.get_parent(flower_partial_visible)
        assert label_schema.get_parent(self.no_plant) is None

        few_leaves = self.new_label_by_name("few_leaves")
        label_schema.add_group(LabelGroup("leaf_state", [few_leaves], LabelGroupType.EXCLUSIVE))
        label_schema.add_child(self.vegetative, few_leaves)
        return few_leaves, flower_fully_visible, flower_partial_visible


def generate_random_annotated_image(
    image_width: int,
    image_height: int,
    labels: Sequence[LabelEntity],
    min_size=50,
    max_size=250,
    shape: Optional[str] = None,
    max_shapes: int = 10,
    intensity_range: List[Tuple[int, int]] = None,
    use_mask_as_annotation: bool = False,
) -> Tuple[np.ndarray, List[Annotation]]:
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
    :param use_mask_as_annotation: If True, masks will be added in annotation
    :return: uint8 array, list of shapes
    """
    from skimage.draw import random_shapes, rectangle

    if intensity_range is None:
        intensity_range = [(100, 200)]

    image1: Optional[np.ndarray] = None
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
        )
        num_shapes = len(sc_labels)
        if num_shapes > 0:
            image1 = rand_image
            break

    if image1 is None:
        raise RuntimeError("Was not able to generate a random image that contains any shapes")

    annotations: List[Annotation] = []
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
                # disable B311 random - used for the random sampling not for security/crypto
                random.randint(0, 200),  # nosec B311
                random.randint(0, 200),  # nosec B311
                random.randint(0, 200),  # nosec B311
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
        else:
            logger.warning(
                "Generated a random image, but was not able to associate a label with a shape. "
                f"The name of the shape was `{sc_label_name}`. "
            )

    return image1, annotations


@contextlib.contextmanager
def generate_random_image_folder(width: int = 480, height: int = 360, number_of_images: int = 10) -> Iterator[str]:
    """
    Generates a folder with random images, cleans up automatically if used in a `with` statement

    :param width: height of the images. Defaults to 480.
    :param height: width of the images. Defaults to 360.
    :param number_of_images: number of generated images. Defaults to 10.

    :return: The temporary directory
    """
    temp_dir = tempfile.TemporaryDirectory()

    for n in range(number_of_images):
        temp_file = os.path.join(temp_dir.name, f"{n}.jpg")
        _write_random_image(width, height, temp_file)

    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_video_folder(
    width: int = 480,
    height: int = 360,
    number_of_videos: int = 10,
    number_of_frames: int = 150,
) -> Iterator[str]:
    """
    Generates a folder with random videos, cleans up automatically if used in a `with` statement
    :param width: Width of the video. Defaults to 480.
    :param height: Height of the video. Defaults to 360.
    :param number_of_videos: Number of videos to generate. Defaults to 10.
    :param number_of_frames: Number of frames in each video. Defaults to 150.

    :return: A temporary directory with videos
    """
    temp_dir = tempfile.TemporaryDirectory()

    for n in range(number_of_videos):
        temp_file = os.path.join(temp_dir.name, f"{n}.mp4")
        _write_random_video(width, height, number_of_frames, temp_file)

    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_single_image(width: int = 480, height: int = 360) -> Iterator[str]:
    """
    Generates a random image, cleans up automatically if used in a `with` statement
    :param width: Width of the image. Defaults to 480.
    :param height: Height of the image. Defaults to 360.

    :return: Path to an image file
    """

    temp_dir = tempfile.TemporaryDirectory()
    temp_file = os.path.join(temp_dir.name, "temp_image.jpg")
    _write_random_image(width, height, temp_file)

    try:
        yield temp_file
    finally:
        temp_dir.cleanup()


@contextlib.contextmanager
def generate_random_single_video(width: int = 480, height: int = 360, number_of_frames: int = 150) -> Iterator[str]:
    """
    Generates a random video, cleans up automatically if used in a `with` statement
    :param width: Width of the video. Defaults to 480.
    :param height: Height of the video. Defaults to 360.
    :param number_of_frames: Number of frames in the video. Defaults to 150.

    :return: Path to a video file
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = os.path.join(temp_dir.name, "temp_video.mp4")
    _write_random_video(width, height, number_of_frames, temp_file)

    try:
        yield temp_file
    finally:
        temp_dir.cleanup()


def _write_random_image(width: int, height: int, filename: str):
    img = np.uint8(np.random.random((height, width, 3)) * 255)
    cv2.imwrite(filename, img)


def _write_random_video(width: int, height: int, number_of_frames: int, filename: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    f = filename
    videowriter = cv2.VideoWriter(f, fourcc, 30, (width, height))

    for _ in range(number_of_frames):
        img = np.uint8(np.random.random((height, width, 3)) * 255)
        videowriter.write(img)

    videowriter.release()


class ConfigExample(ConfigurableParameters):
    header = string_attribute("Test configuration for an object detection task")
    description = header

    class __LearningParameters(ParameterGroup):
        header = string_attribute("Test Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=512,
            header="Test batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        num_iters = configurable_integer(
            default_value=1,
            min_value=1,
            max_value=100000,
            header="Number of training iterations",
            description="Increasing this value causes the results to be more robust but training time will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        learning_rate = configurable_float(
            default_value=0.01,
            min_value=1e-07,
            max_value=1e-01,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        learning_rate_warmup_iters = configurable_integer(
            default_value=100,
            min_value=1,
            max_value=10000,
            header="Number of iterations for learning rate warmup",
            description="Test learning rate warmup",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        num_workers = configurable_integer(
            default_value=4,
            min_value=2,
            max_value=10,
            header="num_workers test header",
            description="num_workers test description",
            affects_outcome_of=ModelLifecycle.NONE,
        )

    class __Postprocessing(ParameterGroup):
        header = string_attribute("Test Postprocessing")
        description = header

        result_based_confidence_threshold = configurable_boolean(
            default_value=True,
            header="Test Result based confidence threshold",
            description="Test confidence threshold is derived from the results",
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

        confidence_threshold = configurable_float(
            default_value=0.25,
            min_value=0,
            max_value=1,
            header="Test Confidence threshold",
            description="This threshold only takes effect if the threshold is not set based on the result.--Only test",
            affects_outcome_of=ModelLifecycle.INFERENCE,
        )

    learning_parameters = add_parameter_group(__LearningParameters)
    postprocessing = add_parameter_group(__Postprocessing)
